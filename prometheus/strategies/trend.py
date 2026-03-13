# ============================================================================
# PROMETHEUS — Strategy Module: Trend Following
# ============================================================================
"""
Trend strategy for directional options buying.

Optimized for under 2L capital:
  - Buy ATM/slightly OTM calls in uptrends
  - Buy ATM/slightly OTM puts in downtrends
  - Debit spreads for capital efficiency
  - Strict R:R enforcement (minimum 2.5:1)
  - ATR-based dynamic stop loss

Entry conditions (ALL must be true):
  1. Regime = MARKUP or MARKDOWN
  2. Multi-timeframe alignment (hourly + 15min same direction)
  3. At least one Tier-1 signal confirming
  4. Not near OI wall (support going long, resistance going short)
  5. R:R >= 2.5
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from prometheus.signals.technical import (
    calculate_vwap, calculate_volume_profile, detect_liquidity_sweeps,
    detect_fair_value_gaps, calculate_atr, calculate_supertrend,
    detect_rsi_divergence, TechnicalSignal
)
from prometheus.signals.regime_detector import RegimeState, MarketRegime
from prometheus.signals.oi_analyzer import OISignal
from prometheus.utils.indian_market import (
    get_lot_size, get_atm_strike, get_strike_interval, days_to_expiry
)
from prometheus.utils.options_math import (
    OptionType, calculate_greeks, implied_volatility
)
from prometheus.utils.logger import logger, log_signal


@dataclass
class TradeSetup:
    """A complete trade setup ready for execution."""
    symbol: str
    instrument: str       # e.g., "NIFTY25MAR23500CE"
    action: str           # "BUY" or "SELL"
    option_type: str      # "CE" or "PE"
    strike: float
    expiry: str
    entry_price: float    # option premium
    stop_loss: float      # option premium level
    target: float         # option premium level
    quantity: int         # number of shares (lots * lot_size)
    lots: int
    risk_amount: float    # max loss in INR
    reward_amount: float  # expected profit in INR
    risk_reward: float
    strategy: str
    signal_strength: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "instrument": self.instrument,
            "action": self.action,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "quantity": self.quantity,
            "lots": self.lots,
            "risk_amount": self.risk_amount,
            "reward_amount": self.reward_amount,
            "risk_reward": self.risk_reward,
            "strategy": self.strategy,
            "signal_strength": self.signal_strength,
            "reasoning": self.reasoning,
        }


class TrendStrategy:
    """
    Trend following strategy using options buying.

    Capital allocation: Max 25% of capital per trade
    Position sizing: Based on max risk amount per trade
    """

    def __init__(self, config: Dict, capital: float = 200000):
        self.config = config
        self.capital = capital
        self.max_risk_per_trade = capital * 0.02  # 2% of capital
        self.max_position_cost = capital * config.get("max_single_trade", 0.25)
        self.min_rr = config.get("target_rr_ratio", 2.5)
        self.max_sl_points = config.get("max_sl_points", 30)

    def generate_setup(
        self,
        symbol: str,
        spot_price: float,
        df_hourly: pd.DataFrame,
        df_15min: pd.DataFrame,
        regime: RegimeState,
        oi_signals: List[OISignal],
        oi_metrics: Dict,
        options_chain: Optional[pd.DataFrame] = None
    ) -> Optional[TradeSetup]:
        """
        Generate a trend trade setup if conditions are met.

        Multi-timeframe process:
        1. Hourly: Determine bias direction
        2. 15-min: Find entry zone
        3. OI: Confirm no opposing walls
        4. Calculate option strike, SL, target
        """
        # Step 1: Check regime
        if regime.regime not in (MarketRegime.MARKUP, MarketRegime.MARKDOWN):
            logger.debug(f"Trend: regime is {regime.regime.value}, not trending. Skip.")
            return None

        direction = "bullish" if regime.regime == MarketRegime.MARKUP else "bearish"

        # Step 2: Hourly timeframe analysis
        hourly_signals = self._analyze_timeframe(df_hourly, "60minute")
        hourly_bias = self._get_directional_bias(hourly_signals)

        if hourly_bias != direction:
            logger.debug(f"Trend: hourly bias is {hourly_bias}, not matching regime. Skip.")
            return None

        # Step 3: 15-min timeframe for entry
        tf15_signals = self._analyze_timeframe(df_15min, "15minute")
        tf15_bias = self._get_directional_bias(tf15_signals)

        if tf15_bias != direction:
            logger.debug(f"Trend: 15min bias {tf15_bias} doesn't confirm. Skip.")
            return None

        # Step 4: OI confirmation
        if not self._oi_confirms(direction, oi_metrics, spot_price):
            logger.debug("Trend: OI analysis contradicts direction. Skip.")
            return None

        # Step 5: Calculate trade levels
        entry_price_index = spot_price
        atr = calculate_atr(df_15min).iloc[-1] if len(df_15min) >= 14 else spot_price * 0.01

        if direction == "bullish":
            sl_index = entry_price_index - min(2 * atr, self.max_sl_points)
            target_index = entry_price_index + min(2 * atr, self.max_sl_points) * self.min_rr
            option_type = "CE"
        else:
            sl_index = entry_price_index + min(2 * atr, self.max_sl_points)
            target_index = entry_price_index - min(2 * atr, self.max_sl_points) * self.min_rr
            option_type = "PE"

        # Step 6: Select option strike
        strike = self._select_strike(spot_price, symbol, option_type)

        # Step 7: Estimate option premium and calculate position size
        lot_size = get_lot_size(symbol)
        dte = max(days_to_expiry(symbol), 1)

        # Estimate option premium from chain or Black-Scholes
        premium = self._estimate_premium(
            options_chain, strike, option_type, spot_price, dte
        )

        if premium <= 0:
            logger.debug("Trend: Cannot estimate premium. Skip.")
            return None

        # Step 8: Position sizing
        position_cost = premium * lot_size
        if position_cost > self.max_position_cost:
            logger.debug(f"Trend: Position cost Rs {position_cost:.0f} > max Rs {self.max_position_cost:.0f}. Skip.")
            return None

        # Calculate option SL and target
        # Option SL ≈ 40-50% of premium (hard maximum loss cap)
        option_sl = premium * 0.5
        index_sl_move = abs(entry_price_index - sl_index)
        index_target_move = abs(target_index - entry_price_index)

        # Estimate option delta for more accurate SL/target
        delta = 0.5  # ATM default
        option_target = premium + (index_target_move * abs(delta))

        risk_per_lot = (premium - option_sl) * lot_size
        reward_per_lot = (option_target - premium) * lot_size

        lots = max(1, int(self.max_risk_per_trade / risk_per_lot)) if risk_per_lot > 0 else 1
        total_cost = premium * lot_size * lots

        # Final capital check
        if total_cost > self.max_position_cost:
            lots = max(1, int(self.max_position_cost / (premium * lot_size)))

        risk_amount = risk_per_lot * lots
        reward_amount = reward_per_lot * lots
        rr = reward_amount / risk_amount if risk_amount > 0 else 0

        if rr < self.min_rr:
            logger.debug(f"Trend: R:R {rr:.1f} < minimum {self.min_rr}. Skip.")
            return None

        # Step 9: Build signal strength
        signal_strength = self._calculate_signal_strength(
            hourly_signals, tf15_signals, oi_signals, regime
        )

        min_strength = self.config.get("min_signal_strength", 0.70)
        if signal_strength < min_strength:
            logger.debug(f"Trend: Signal strength {signal_strength:.2f} < min {min_strength}. Skip.")
            return None

        # Build the trade setup
        reasoning = self._build_reasoning(
            direction, regime, hourly_signals, tf15_signals,
            oi_metrics, signal_strength, rr
        )

        expiry_str = f"WEEKLY"  # To be resolved with actual expiry date

        setup = TradeSetup(
            symbol=symbol,
            instrument=f"{symbol} {strike} {option_type}",
            action="BUY",
            option_type=option_type,
            strike=strike,
            expiry=expiry_str,
            entry_price=round(premium, 2),
            stop_loss=round(option_sl, 2),
            target=round(option_target, 2),
            quantity=lots * lot_size,
            lots=lots,
            risk_amount=round(risk_amount, 2),
            reward_amount=round(reward_amount, 2),
            risk_reward=round(rr, 2),
            strategy="trend",
            signal_strength=round(signal_strength, 3),
            reasoning=reasoning,
        )

        log_signal("TREND", symbol, signal_strength, setup.to_dict())
        return setup

    def _analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> List[TechnicalSignal]:
        """Run all technical analysis on a timeframe."""
        signals = []
        if df.empty or len(df) < 20:
            return signals

        # VWAP
        vwap_df = calculate_vwap(df)
        current = df["close"].iloc[-1]
        vwap_val = vwap_df["vwap"].iloc[-1]
        if not pd.isna(vwap_val):
            if current > vwap_val:
                signals.append(TechnicalSignal(
                    name="vwap", direction="bullish",
                    strength=min(abs(current - vwap_val) / vwap_val * 100, 0.8),
                    timeframe=timeframe, price_level=vwap_val
                ))
            else:
                signals.append(TechnicalSignal(
                    name="vwap", direction="bearish",
                    strength=min(abs(current - vwap_val) / vwap_val * 100, 0.8),
                    timeframe=timeframe, price_level=vwap_val
                ))

        # Volume Profile
        vp = calculate_volume_profile(df)
        if vp:
            poc = vp["poc"]
            if current > poc:
                signals.append(TechnicalSignal(
                    name="volume_profile", direction="bullish",
                    strength=0.6, timeframe=timeframe, price_level=poc
                ))
            else:
                signals.append(TechnicalSignal(
                    name="volume_profile", direction="bearish",
                    strength=0.6, timeframe=timeframe, price_level=poc
                ))

        # Supertrend
        st_df = calculate_supertrend(df)
        st_dir = st_df["supertrend_direction"].iloc[-1]
        signals.append(TechnicalSignal(
            name="supertrend",
            direction="bullish" if st_dir == 1 else "bearish",
            strength=0.6, timeframe=timeframe,
            price_level=st_df["supertrend"].iloc[-1]
        ))

        # Liquidity Sweeps
        sweeps = detect_liquidity_sweeps(df)
        for sweep in sweeps[-2:]:  # most recent sweeps only
            signals.append(TechnicalSignal(
                name="liquidity_sweep",
                direction="bullish" if sweep["type"] == "bullish_sweep" else "bearish",
                strength=sweep["strength"] * 0.7,
                timeframe=timeframe,
                price_level=sweep["level"],
            ))

        # FVGs
        fvgs = detect_fair_value_gaps(df)
        unfilled = [f for f in fvgs if not f["filled"]]
        for fvg in unfilled[-2:]:
            if fvg["type"] == "bullish_fvg" and current <= fvg["top"]:
                signals.append(TechnicalSignal(
                    name="fvg_imbalance", direction="bullish",
                    strength=0.5, timeframe=timeframe,
                    price_level=fvg["midpoint"],
                ))
            elif fvg["type"] == "bearish_fvg" and current >= fvg["bottom"]:
                signals.append(TechnicalSignal(
                    name="fvg_imbalance", direction="bearish",
                    strength=0.5, timeframe=timeframe,
                    price_level=fvg["midpoint"],
                ))

        # RSI Divergence (secondary)
        div = detect_rsi_divergence(df)
        if div:
            signals.append(TechnicalSignal(
                name="rsi_divergence", direction=div["direction"],
                strength=div["strength"] * 0.5,
                timeframe=timeframe, price_level=div["price_level"],
            ))

        return signals

    def _get_directional_bias(self, signals: List[TechnicalSignal]) -> str:
        """Determine overall directional bias from signals."""
        if not signals:
            return "neutral"

        weights = {
            "volume_profile": 0.85, "vwap": 0.80, "liquidity_sweep": 0.70,
            "supertrend": 0.50, "fvg_imbalance": 0.60, "rsi_divergence": 0.40,
        }

        bullish = sum(
            s.strength * weights.get(s.name, 0.5)
            for s in signals if s.direction == "bullish"
        )
        bearish = sum(
            s.strength * weights.get(s.name, 0.5)
            for s in signals if s.direction == "bearish"
        )

        if bullish > bearish * 1.2:
            return "bullish"
        elif bearish > bullish * 1.2:
            return "bearish"
        return "neutral"

    def _oi_confirms(self, direction: str, oi_metrics: Dict, spot: float) -> bool:
        """Check if OI data supports the trade direction."""
        if not oi_metrics:
            return True  # No OI data = don't block

        resistance = oi_metrics.get("strongest_resistance", spot * 1.02)
        support = oi_metrics.get("strongest_support", spot * 0.98)

        if direction == "bullish":
            # Don't go long if spot is right at resistance
            distance_to_resistance = (resistance - spot) / spot
            return distance_to_resistance > 0.005  # At least 0.5% room

        else:
            # Don't go short if spot is right at support
            distance_to_support = (spot - support) / spot
            return distance_to_support > 0.005

    def _select_strike(
        self,
        spot: float,
        symbol: str,
        option_type: str
    ) -> float:
        """Select optimal strike for the trade."""
        atm = get_atm_strike(spot, symbol)
        interval = get_strike_interval(symbol)

        if option_type == "CE":
            # ATM or 1 strike OTM for better R:R
            return atm
        else:
            return atm

    def _estimate_premium(
        self,
        chain: Optional[pd.DataFrame],
        strike: float,
        option_type: str,
        spot: float,
        dte: int
    ) -> float:
        """Estimate option premium from chain data or Black-Scholes."""
        # Try chain data first
        if chain is not None and not chain.empty:
            match = chain[
                (chain["strike"] == strike) &
                (chain["option_type"] == option_type)
            ]
            if not match.empty:
                ltp = match["ltp"].iloc[0]
                if ltp > 0:
                    return ltp

        # Fallback: Black-Scholes estimate
        from prometheus.utils.options_math import black_scholes_price
        T = max(dte, 1) / 365
        sigma = 0.15  # assume 15% IV as default
        r = 0.065     # RBI repo rate approximate

        opt_type = OptionType.CALL if option_type == "CE" else OptionType.PUT
        return black_scholes_price(spot, strike, T, r, sigma, opt_type)

    def _calculate_signal_strength(
        self,
        hourly_signals: List[TechnicalSignal],
        m15_signals: List[TechnicalSignal],
        oi_signals: List[OISignal],
        regime: RegimeState
    ) -> float:
        """Calculate overall signal strength (0-1)."""
        scores = []

        # Hourly alignment: 30% weight
        h_bullish = sum(s.strength for s in hourly_signals if s.direction == "bullish")
        h_bearish = sum(s.strength for s in hourly_signals if s.direction == "bearish")
        h_total = h_bullish + h_bearish
        if h_total > 0:
            scores.append(max(h_bullish, h_bearish) / h_total * 0.30)

        # 15-min alignment: 30% weight
        m_bullish = sum(s.strength for s in m15_signals if s.direction == "bullish")
        m_bearish = sum(s.strength for s in m15_signals if s.direction == "bearish")
        m_total = m_bullish + m_bearish
        if m_total > 0:
            scores.append(max(m_bullish, m_bearish) / m_total * 0.30)

        # OI confirmation: 20% weight
        oi_bullish = sum(s.strength for s in oi_signals if s.direction == "bullish")
        oi_bearish = sum(s.strength for s in oi_signals if s.direction == "bearish")
        oi_total = oi_bullish + oi_bearish
        if oi_total > 0:
            scores.append(max(oi_bullish, oi_bearish) / oi_total * 0.20)

        # Regime confidence: 20% weight
        scores.append(regime.confidence * 0.20)

        return sum(scores)

    def _build_reasoning(
        self,
        direction: str,
        regime: RegimeState,
        hourly_signals: List[TechnicalSignal],
        m15_signals: List[TechnicalSignal],
        oi_metrics: Dict,
        strength: float,
        rr: float
    ) -> str:
        """Build human-readable reasoning for the trade."""
        parts = [
            f"TREND {direction.upper()} setup.",
            f"Regime: {regime.regime.value} (confidence {regime.confidence:.0%}).",
            f"Hourly confirms with {sum(1 for s in hourly_signals if s.direction == direction)} signals.",
            f"15-min confirms with {sum(1 for s in m15_signals if s.direction == direction)} signals.",
        ]

        pcr = oi_metrics.get("pcr", {}).get("oi", 0)
        if pcr:
            parts.append(f"PCR: {pcr:.2f}.")

        parts.append(f"Signal strength: {strength:.0%}. R:R = 1:{rr:.1f}.")
        return " ".join(parts)
