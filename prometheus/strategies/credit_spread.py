# ============================================================================
# PROMETHEUS — Strategy: Hedged Credit Spread (Theta Decay Engine)
# ============================================================================
"""
Hedged Defined-Risk Credit Spread Strategy for Sideways / Range-Bound Regimes.

Generates:
1. Bull Put Spreads (Neutral to Mildly Bullish): Sell 1-OTM PE + Buy 3-OTM PE Hedge
2. Bear Call Spreads (Neutral to Mildly Bearish): Sell 1-OTM CE + Buy 3-OTM CE Hedge

Designed for small-to-medium capital accounts with defined risk and SEBI margin reduction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple

from prometheus.utils.indian_market import (
    get_lot_size, get_atm_strike, get_strike_interval,
    days_to_expiry, get_expiry_date, is_weekly_expiry_day
)
from prometheus.execution.kite_executor import generate_tradingsymbol
from prometheus.signals.technical import calculate_atr, calculate_vwap, calculate_supertrend
from prometheus.utils.logger import logger


class CreditSpreadStrategy:
    """Generates defined-risk credit spreads for sideways market regimes."""

    def __init__(
        self,
        strike_otm_steps: int = 1,
        hedge_otm_steps: int = 3,
        target_decay_pct: float = 0.70,     # Take profit when 70% of credit decays
        breakeven_decay_pct: float = 0.50,  # Lock BE when 50% of credit decays
        max_loss_multiplier: float = 1.5,   # Hard SL at 1.5x initial credit
        min_credit_pct: float = 0.15,       # Min credit must be >= 15% of strike width
        max_days_to_expiry: Optional[int] = None,  # When set (e.g. 1 in intraday), strictly enforce 0-DTE / 1-DTE
    ):
        self.strike_otm_steps = strike_otm_steps
        self.hedge_otm_steps = hedge_otm_steps
        self.target_decay_pct = target_decay_pct
        self.breakeven_decay_pct = breakeven_decay_pct
        self.max_loss_multiplier = max_loss_multiplier
        self.min_credit_pct = min_credit_pct
        self.max_days_to_expiry = max_days_to_expiry

    def evaluate_spread(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY 50",
        capital: float = 50000.0,
        option_chain = None,
    ) -> Optional[Dict]:
        """Evaluate if market is in a range-bound state and generate a credit spread.

        Args:
            df: OHLCV DataFrame with timestamp.
            symbol: Index symbol name.
            capital: Available capital.
            option_chain: Live AngelOneOptionChain instance for real market LTPs.

        Returns:
            Dict containing 2-leg spread details and exit thresholds, or None.
        """
        if df is None or len(df) < 15:
            return None

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        current_row = df.iloc[-1]
        current_ts = current_row["timestamp"]
        current_time = current_ts.time() if hasattr(current_ts, "time") else dtime(10, 0)
        current_date = current_ts.date() if hasattr(current_ts, "date") else None

        # Trading window: 09:50 to 14:15
        if current_time < dtime(9, 50) or current_time > dtime(14, 15):
            return None

        # Extract today's bars
        if current_date:
            today_bars = df[df["timestamp"].dt.date == current_date].copy()
        else:
            today_bars = df.iloc[-25:].copy()

        if len(today_bars) < 3:
            return None

        close = float(current_row["close"])
        atr_s = calculate_atr(df, period=14)
        atr = float(atr_s.iloc[-1]) if len(atr_s) > 0 and not np.isnan(atr_s.iloc[-1]) else close * 0.005

        # ── 1. Range & Volatility Check (Ensure Sideways Regime) ──
        today_high = float(today_bars["high"].max())
        today_low = float(today_bars["low"].min())
        day_range = today_high - today_low

        if day_range > 2.5 * atr:
            # Market is already trending strongly — skip credit spread
            return None

        # Check VWAP and EMAs for rigorous alignment
        vwap_df = calculate_vwap(df)
        if isinstance(vwap_df, pd.DataFrame) and "vwap" in vwap_df.columns:
            vwap_val = vwap_df["vwap"].iloc[-1]
        elif isinstance(vwap_df, pd.Series):
            vwap_val = vwap_df.iloc[-1]
        else:
            vwap_val = close
        vwap = float(vwap_val) if pd.notna(vwap_val) else close
        
        ema9_s = df["close"].ewm(span=9, adjust=False).mean()
        ema21_s = df["close"].ewm(span=21, adjust=False).mean()
        ema9 = float(ema9_s.iloc[-1]) if len(ema9_s) > 0 else close
        ema21 = float(ema21_s.iloc[-1]) if len(ema21_s) > 0 else close

        # Check SuperTrend
        st_df = calculate_supertrend(df, period=10, multiplier=3.0)
        st_dir = int(st_df["supertrend_direction"].iloc[-1]) if len(st_df) > 0 else 0

        # Strike parameters
        atm_strike = get_atm_strike(close, symbol)
        interval = get_strike_interval(symbol)
        lot_size = get_lot_size(symbol)
        expiry_date = get_expiry_date(symbol, from_date=current_date)
        expiry_str = expiry_date.strftime("%Y-%m-%d") if expiry_date else ""

        # Enforce DTE restriction for intraday credit spreads (0-DTE or 1-DTE only)
        if self.max_days_to_expiry is not None and current_date and expiry_date:
            days_to_exp = (expiry_date - current_date).days
            if days_to_exp > self.max_days_to_expiry:
                logger.info(
                    f"CreditSpread skipped for {symbol}: Expiry {expiry_str} is {days_to_exp} days away. "
                    f"Intraday credit spreads strictly require <= {self.max_days_to_expiry} DTE for rapid theta decay."
                )
                return None

        # ── 2. Select Spread Type with Structural Safety & Trend Alignment ──
        # Bear Call Spread requires: Below VWAP and not in a rapid bullish rebound
        # Bull Put Spread requires: Above VWAP and not in a rapid bearish collapse
        is_bearish = (close < vwap) and (ema9 <= ema21)
        is_bullish = (close > vwap) and (ema9 >= ema21)
        
        # If trend is neutral/choppy within range:
        if not is_bearish and not is_bullish:
            midpoint = (today_high + today_low) / 2.0
            if close >= midpoint or st_dir == -1:
                is_bearish = True  # Near or above range midpoint -> sell Bear Call Spread above resistance
            else:
                is_bullish = True  # Below range midpoint -> sell Bull Put Spread below support

        # ── Dynamic 2.0σ Strike Buffer (Pillar 2) ──
        sigma_buffer = round((2.0 * atr) / interval) * interval
        sigma_buffer = max(interval, sigma_buffer)

        # ── Institutional Open Interest (OI) Wall Scan (Pillar 3) ──
        oi_wall_strike = None
        oi_wall_shares = 0
        if option_chain is not None and hasattr(option_chain, "get_option_chain"):
            try:
                target_opt = "CE" if is_bearish else "PE"
                chain_df = option_chain.get_option_chain(symbol, spot_price=close, expiry_date=expiry_str)
                if isinstance(chain_df, pd.DataFrame) and not chain_df.empty and "oi" in chain_df.columns:
                    side_df = chain_df[chain_df["option_type"] == target_opt]
                    if not side_df.empty and side_df["oi"].max() > 0:
                        max_oi_row = side_df.loc[side_df["oi"].idxmax()]
                        oi_wall_strike = float(max_oi_row["strike"])
                        oi_wall_shares = int(max_oi_row["oi"])
            except Exception as e:
                logger.debug(f"OI wall discovery error for {symbol}: {e}")

        otm_steps = self.strike_otm_steps
        
        if is_bearish:
            # Bear Call Spread: Place short strike above today's high / resistance + 2.0σ buffer
            spread_type = "BEAR_CALL_SPREAD"
            high_strike = get_atm_strike(today_high + sigma_buffer, symbol)
            calculated_strike = atm_strike + max(otm_steps * interval, sigma_buffer)
            short_strike = max(high_strike, calculated_strike)
            # If an institutional Call OI wall is identified above spot, ensure strike is at or beyond the wall
            if oi_wall_strike and oi_wall_strike >= atm_strike:
                short_strike = max(short_strike, oi_wall_strike)
            long_strike = short_strike + (self.hedge_otm_steps * interval)
            opt_str = "CE"
            action = "SELL_CALL_SPREAD"
        elif is_bullish:
            # Bull Put Spread: Place short strike below today's low / support - 2.0σ buffer
            spread_type = "BULL_PUT_SPREAD"
            low_strike = get_atm_strike(today_low - sigma_buffer, symbol)
            calculated_strike = atm_strike - max(otm_steps * interval, sigma_buffer)
            short_strike = min(low_strike, calculated_strike)
            # If an institutional Put OI wall is identified below spot, ensure strike is at or below the wall
            if oi_wall_strike and oi_wall_strike <= atm_strike:
                short_strike = min(short_strike, oi_wall_strike)
            long_strike = short_strike - (self.hedge_otm_steps * interval)
            opt_str = "PE"
            action = "SELL_PUT_SPREAD"
        else:
            return None

        # Generate Kite Tradingsymbols
        sym_map = {
            "NIFTY 50": "NIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "SENSEX": "SENSEX",
            "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
        }
        underlying = sym_map.get(symbol, symbol.upper())
        short_tradingsymbol = generate_tradingsymbol(underlying, expiry_str, short_strike, opt_str)
        long_tradingsymbol = generate_tradingsymbol(underlying, expiry_str, long_strike, opt_str)

        # ── 3. Real Market Option Pricing (No Black-Scholes) ──
        short_premium = 0.0
        long_premium = 0.0

        if option_chain is not None:
            try:
                # 1. Direct real premium lookup via AngelOneOptionChain
                if hasattr(option_chain, "get_real_premium"):
                    sq = option_chain.get_real_premium(symbol, short_strike, opt_str, expiry=expiry_str, spot_price=close)
                    lq = option_chain.get_real_premium(symbol, long_strike, opt_str, expiry=expiry_str, spot_price=close)
                    if sq and sq.get("ltp", 0) > 0:
                        short_premium = float(sq["ltp"])
                        if sq.get("tradingsymbol"):
                            short_tradingsymbol = sq["tradingsymbol"]
                    if lq and lq.get("ltp", 0) > 0:
                        long_premium = float(lq["ltp"])
                        if lq.get("tradingsymbol"):
                            long_tradingsymbol = lq["tradingsymbol"]

                # 2. Direct lookup via get_option_ltp
                if short_premium <= 0 and hasattr(option_chain, "get_option_ltp"):
                    s_ltp = option_chain.get_option_ltp(short_tradingsymbol)
                    l_ltp = option_chain.get_option_ltp(long_tradingsymbol)
                    if s_ltp and float(s_ltp) > 0:
                        short_premium = float(s_ltp)
                    if l_ltp and float(l_ltp) > 0:
                        long_premium = float(l_ltp)
            except Exception as e:
                logger.warning(f"CreditSpread live premium fetch error for {symbol}: {e}")

        strike_width = abs(short_strike - long_strike)

        # Strict Rule: No fallback mathematical formulas allowed. Must have live option prices.
        if short_premium <= 0 or long_premium <= 0:
            logger.warning(
                f"CreditSpread skipped for {symbol}: Live option prices unavailable "
                f"({short_tradingsymbol}=Rs {short_premium:.2f}, {long_tradingsymbol}=Rs {long_premium:.2f}) — "
                f"synthetic formulas strictly banned."
            )
            return None

        net_credit = round(short_premium - long_premium, 2)
        min_required_credit = round(strike_width * self.min_credit_pct, 2)
        if net_credit < min_required_credit or net_credit <= 0:
            logger.info(
                f"CreditSpread skipped for {symbol}: Net credit Rs {net_credit:.2f} is below "
                f"minimum threshold Rs {min_required_credit:.2f} ({self.min_credit_pct*100:.0f}% of strike width) "
                f"or non-positive — refusing synthetic fill."
            )
            return None

        max_profit = net_credit * lot_size
        max_loss = (strike_width - net_credit) * lot_size

        # Inverted Trailing & Exit Thresholds
        target_decay_val = round(net_credit * (1.0 - self.target_decay_pct), 2)     # e.g. 40 * 0.30 = 12.00 (70% profit)
        breakeven_decay_val = round(net_credit * (1.0 - self.breakeven_decay_pct), 2) # e.g. 40 * 0.50 = 20.00 (50% profit)
        hard_sl_val = round(net_credit * self.max_loss_multiplier, 2)              # e.g. 40 * 1.5 = 60.00 (Hard SL)

        # Generate Kite Tradingsymbols
        sym_map = {
            "NIFTY 50": "NIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "SENSEX": "SENSEX",
            "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
        }
        underlying = sym_map.get(symbol, symbol.upper())
        short_tradingsymbol = generate_tradingsymbol(underlying, expiry_str, short_strike, opt_str)
        long_tradingsymbol = generate_tradingsymbol(underlying, expiry_str, long_strike, opt_str)

        # 2-Leg structure (Hedge leg executed first to ensure SEBI margin reduction)
        legs = [
            {
                "leg_index": 1,
                "action": "BUY",
                "instrument": long_tradingsymbol,
                "tradingsymbol": long_tradingsymbol,
                "strike": long_strike,
                "option_type": opt_str,
                "premium": round(long_premium, 2),
                "is_hedge": True,
                "lot_size": lot_size,
            },
            {
                "leg_index": 2,
                "action": "SELL",
                "instrument": short_tradingsymbol,
                "tradingsymbol": short_tradingsymbol,
                "strike": short_strike,
                "option_type": opt_str,
                "premium": round(short_premium, 2),
                "is_hedge": False,
                "lot_size": lot_size,
            }
        ]

        # Realistic NSE/BSE SPAN + Exposure Margin for hedged spreads.
        # Account for SEBI derivatives framework (minimum Rs 15 Lakhs contract notional):
        # 1. Base SPAN margin for hedged spreads: ~Rs 32,000 - 35,000 depending on index
        # 2. Mandatory Expiry Day 2% ELM (Extreme Loss Margin) on short leg:
        #    SEBI circular mandates +2% ELM on contract notional on expiry day, which
        #    CANNOT be discounted by hedges: 2% of (Strike * Lot Size) = ~Rs 30,000!
        is_expiry = is_weekly_expiry_day(symbol, current_date)
        expiry_elm = (0.02 * float(short_strike) * lot_size) if is_expiry else 0.0

        if "BANK" in symbol:
            base_margin = 35000.0 + (strike_width * lot_size * 0.8)
        elif "SENSEX" in symbol:
            base_margin = 34000.0 + (strike_width * lot_size * 0.6)
        else:  # NIFTY 50 / FINNIFTY / MIDCAP
            base_margin = 32000.0 + (strike_width * lot_size * 0.7)

        margin_required = base_margin + expiry_elm

        # ── Quantitative Conviction & Probability of Profit Classifier ──
        is_0dte = bool(current_date and expiry_date and (expiry_date - current_date).days == 0)
        strike_dist = abs(short_strike - close)
        otm_sigma = round(strike_dist / max(atr, 1.0), 2)
        
        # Rigorous Pillar Checks:
        # Pillar 1: Statistical buffer (strike must be >= 1.5σ away from spot)
        is_far_otm = bool(otm_sigma >= 1.5)
        # Pillar 2: Trend alignment (spot actually favorable relative to VWAP)
        trend_aligned = bool((is_bearish and close <= vwap) or (is_bullish and close >= vwap))
        # Pillar 3: Institutional Open Interest wall protection
        is_wall_shielded = bool(
            (oi_wall_strike and is_bearish and short_strike >= oi_wall_strike) or
            (oi_wall_strike and is_bullish and short_strike <= oi_wall_strike)
        )
        
        # Theoretical Probability of Profit (POP) based on standard normal distribution & moneyness
        # POP approx = Phi(otm_sigma) with haircut for fat-tail risk in index options
        # e.g., 2.0σ -> ~93%, 1.75σ -> ~89%, 1.5σ -> ~85%, 1.0σ -> ~78%
        import math
        def _norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        
        raw_pop = _norm_cdf(otm_sigma) - 0.04  # 4% tail-risk haircut
        if not is_0dte:
            raw_pop -= 0.06  # discount on non-expiry days due to multi-day vega/gamma risk
        pop_pct = round(max(0.65, min(0.96, raw_pop)) * 100.0, 1)

        # Sure-Shot / Tier 1 requires 0-DTE + Far OTM (>=1.5σ) + VWAP trend alignment
        is_sure_shot = bool(is_0dte and is_far_otm and trend_aligned)
        
        # Dynamically scaled score (7.0 to 9.5) based on true confluences
        score_calc = 7.0
        if is_0dte:
            score_calc += 1.0
        if is_far_otm:
            score_calc += 0.8
        if trend_aligned:
            score_calc += 0.4
        if is_wall_shielded:
            score_calc += 0.3
        signal_score = min(9.5, round(score_calc, 1))
        
        # Harmonize confidence and strength
        confidence = round(pop_pct / 100.0, 2)
        signal_strength = signal_score

        return {
            "strategy": "Hedged_Credit_Spread",
            "strategy_type": "credit_spread",
            "spread_type": spread_type,
            "action": action,
            "direction": "neutral_range",
            "symbol": symbol,
            "underlying_price": round(close, 2),
            "spot_price": round(close, 2),
            "entry_price": round(net_credit, 2),
            "entry_premium": round(net_credit, 2),
            "strike": float(short_strike),
            "option_type": opt_str,
            "stop_loss": hard_sl_val,
            "target": target_decay_val,
            "net_credit": round(net_credit, 2),
            "strike_width": strike_width,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "expiry": expiry_str,
            "lot_size": lot_size,
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "target_decay_price": target_decay_val,
            "breakeven_decay_price": breakeven_decay_val,
            "hard_sl_price": hard_sl_val,
            "margin_required": round(margin_required, 2),
            "legs": legs,
            "tradingsymbol": f"{short_tradingsymbol}/{long_tradingsymbol}",
            "instrument": f"{short_tradingsymbol}/{long_tradingsymbol}",
            "trade_mode": "intraday",
            "is_sure_shot": is_sure_shot,
            "signal_score": signal_score,
            "confidence": confidence,
            "signal_strength": signal_strength,
            "pop_pct": pop_pct,
            "theoretical_pop": pop_pct,
            "otm_sigma": otm_sigma,
            "oi_shielded": is_wall_shielded,
            "oi_wall_strike": oi_wall_strike,
            "oi_wall_shares": oi_wall_shares,
            "bar_timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
        }
