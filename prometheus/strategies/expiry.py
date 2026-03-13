# ============================================================================
# PROMETHEUS — Strategy: Expiry Day Module
# ============================================================================
"""
Expiry-day strategies optimized for under 2L capital.

Focus: Debit spreads and directional option buying near expiry.

Why expiry day?
  - Theta decay is maximum (non-linear acceleration)
  - ATM options lose 40-60% of remaining value on expiry day
  - But if there's a directional move, OTM options can 2x-5x
  - Weekly expiry every Thursday (NIFTY/BANKNIFTY/FINNIFTY)

Strategies:
  1. Directional Debit Spread — lower cost, defined risk, capital efficient
  2. Momentum Breakout Buying — buy ATM option when strong breakout detected
  3. Last-hour Scalp — very short holding period, quick in/out

Capital constraint adaptation:
  - No naked selling (margins too high)
  - Prefer debit spreads over single options (cost reduction)
  - Maximum 1-2 trades per expiry (preserve capital)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, date, timedelta

from prometheus.utils.indian_market import (
    get_lot_size, get_atm_strike, get_strike_interval,
    days_to_expiry, get_expiry_date, minutes_to_close, is_market_open
)
from prometheus.utils.options_math import black_scholes_price, OptionType
from prometheus.utils.logger import logger


@dataclass
class ExpirySetup:
    """An expiry day trade setup."""
    symbol: str
    strategy_type: str      # "debit_spread", "momentum_buy", "scalp"
    legs: List[Dict]
    total_cost: float       # max risk
    max_profit: float
    breakeven: float
    entry_time: str
    expected_exit: str      # "3:25 PM" or "target/SL"
    signal_strength: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy_type": self.strategy_type,
            "legs": self.legs,
            "total_cost": self.total_cost,
            "max_profit": self.max_profit,
            "breakeven": self.breakeven,
            "signal_strength": self.signal_strength,
            "reasoning": self.reasoning,
        }


class ExpiryStrategy:
    """
    Expiry day trading strategies.
    """

    def __init__(self, config: Dict, capital: float = 200000):
        self.config = config
        self.capital = capital
        self.max_trade_cost = capital * 0.15  # max 15% per expiry trade
        self.dte_max = config.get("days_to_expiry_max", 2)

    def check_expiry_opportunity(
        self,
        symbol: str,
        spot_price: float,
        data: pd.DataFrame,
        options_chain: Optional[pd.DataFrame] = None,
        iv_percentile: float = 50.0,
    ) -> Optional[ExpirySetup]:
        """
        Check if there's an expiry day trading opportunity.

        Conditions:
        1. Within 0-2 days of expiry
        2. Clear directional signal from intraday data
        3. Reasonable cost within capital limits
        """
        dte = days_to_expiry(symbol)

        if dte > self.dte_max:
            return None

        # Check if market is open
        if not is_market_open():
            return None

        mins_left = minutes_to_close()

        # On expiry day itself
        if dte == 0:
            # Don't trade in last 5 minutes (too risky, wide spreads)
            if mins_left < 5:
                return None

            # Momentum breakout in first 2 hours
            if mins_left > 200:
                return self._momentum_breakout(symbol, spot_price, data, options_chain)

            # Directional debit spread between 11 AM and 2 PM
            if 90 < mins_left < 270:
                return self._debit_spread(symbol, spot_price, data, options_chain, dte)

        # 1-2 days before expiry
        elif 0 < dte <= 2:
            return self._debit_spread(symbol, spot_price, data, options_chain, dte)

        return None

    def _momentum_breakout(
        self,
        symbol: str,
        spot: float,
        data: pd.DataFrame,
        chain: Optional[pd.DataFrame]
    ) -> Optional[ExpirySetup]:
        """
        Expiry day momentum breakout.

        Logic:
        - First 30 minutes establishes range
        - Break above range high → buy CE
        - Break below range low → buy PE
        - ATM option with tight SL (50% premium loss)
        """
        if len(data) < 6:  # need at least 30 min of 5-min data
            return None

        # Opening range (first 30 minutes)
        opening_data = data.head(6)
        range_high = opening_data["high"].max()
        range_low = opening_data["low"].min()
        range_size = range_high - range_low

        # Current price
        current = data["close"].iloc[-1]

        # Check for breakout with volume confirmation
        avg_volume = data["volume"].mean()
        recent_volume = data["volume"].tail(3).mean()
        volume_surge = recent_volume > avg_volume * 1.5

        direction = None
        if current > range_high and volume_surge:
            direction = "bullish"
            option_type = "CE"
        elif current < range_low and volume_surge:
            direction = "bearish"
            option_type = "PE"
        else:
            return None

        # Build trade
        strike = get_atm_strike(spot, symbol)
        lot_size = get_lot_size(symbol)

        premium = self._estimate_expiry_premium(spot, strike, option_type, 0)
        if premium <= 0:
            return None

        total_cost = premium * lot_size
        if total_cost > self.max_trade_cost:
            return None

        # Target: range size move in premium
        delta = 0.5  # ATM delta
        target_premium = premium + (range_size * delta)
        max_profit = (target_premium - premium) * lot_size

        sl_premium = premium * 0.50  # 50% SL on premium

        return ExpirySetup(
            symbol=symbol,
            strategy_type="momentum_buy",
            legs=[{
                "action": "BUY",
                "option_type": option_type,
                "strike": strike,
                "premium": premium,
                "lots": 1,
                "lot_size": lot_size,
                "sl_premium": sl_premium,
                "target_premium": target_premium,
            }],
            total_cost=round(total_cost, 2),
            max_profit=round(max_profit, 2),
            breakeven=round(strike + premium if option_type == "CE" else strike - premium, 2),
            entry_time=datetime.now().strftime("%H:%M"),
            expected_exit="Target/SL or 3:15 PM",
            signal_strength=round(min(0.75, 0.50 + (recent_volume / avg_volume - 1) * 0.25), 3),
            reasoning=(
                f"Expiry day momentum breakout. "
                f"Opening range: {range_low:.0f}-{range_high:.0f} "
                f"(size: {range_size:.0f} pts). "
                f"Breakout {direction} with {recent_volume/avg_volume:.1f}x volume surge. "
                f"Buy {option_type} {strike} @ {premium:.1f}. "
                f"SL: {sl_premium:.1f} | Target: {target_premium:.1f}. "
                f"Max risk: Rs {total_cost:.0f}."
            ),
        )

    def _debit_spread(
        self,
        symbol: str,
        spot: float,
        data: pd.DataFrame,
        chain: Optional[pd.DataFrame],
        dte: int
    ) -> Optional[ExpirySetup]:
        """
        Debit spread near expiry — capital efficient.

        Bull Call Spread: Buy ATM CE + Sell OTM CE  → net debit, defined risk
        Bear Put Spread: Buy ATM PE + Sell OTM PE   → net debit, defined risk

        Cost is much lower than single option buying.
        Max profit when underlying moves past short strike.
        """
        if len(data) < 20:
            return None

        # Determine direction from recent price action
        ema_fast = data["close"].ewm(span=8).mean()
        ema_slow = data["close"].ewm(span=21).mean()

        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]

        # Trend determination
        if current_fast > current_slow * 1.001:
            direction = "bullish"
            option_type = "CE"
        elif current_fast < current_slow * 0.999:
            direction = "bearish"
            option_type = "PE"
        else:
            return None  # No clear direction

        interval = get_strike_interval(symbol)
        atm_strike = get_atm_strike(spot, symbol)
        lot_size = get_lot_size(symbol)

        if option_type == "CE":
            buy_strike = atm_strike
            sell_strike = atm_strike + interval
        else:
            buy_strike = atm_strike
            sell_strike = atm_strike - interval

        # Estimate premiums
        buy_premium = self._estimate_expiry_premium(spot, buy_strike, option_type, dte)
        sell_premium = self._estimate_expiry_premium(spot, sell_strike, option_type, dte)

        if buy_premium <= 0 or sell_premium <= 0:
            return None

        net_debit = buy_premium - sell_premium
        if net_debit <= 0:
            return None

        total_cost = net_debit * lot_size
        max_profit = (interval - net_debit) * lot_size

        if total_cost > self.max_trade_cost:
            return None

        # Signal strength based on trend clarity and cost efficiency
        trend_strength = abs(current_fast - current_slow) / current_slow * 100
        cost_efficiency = max_profit / total_cost if total_cost > 0 else 0
        signal_strength = min(0.80, 0.40 + trend_strength * 0.1 + cost_efficiency * 0.1)

        spread_type = "Bull Call Spread" if option_type == "CE" else "Bear Put Spread"

        return ExpirySetup(
            symbol=symbol,
            strategy_type="debit_spread",
            legs=[
                {
                    "action": "BUY",
                    "option_type": option_type,
                    "strike": buy_strike,
                    "premium": buy_premium,
                    "lots": 1,
                    "lot_size": lot_size,
                },
                {
                    "action": "SELL",
                    "option_type": option_type,
                    "strike": sell_strike,
                    "premium": sell_premium,
                    "lots": 1,
                    "lot_size": lot_size,
                },
            ],
            total_cost=round(total_cost, 2),
            max_profit=round(max_profit, 2),
            breakeven=round(buy_strike + net_debit if option_type == "CE" else buy_strike - net_debit, 2),
            entry_time=datetime.now().strftime("%H:%M"),
            expected_exit=f"Expiry day close ({dte}d to expiry)",
            signal_strength=round(signal_strength, 3),
            reasoning=(
                f"{spread_type} for {symbol} ({dte} DTE). "
                f"Buy {option_type} {buy_strike} @ {buy_premium:.1f} + "
                f"Sell {option_type} {sell_strike} @ {sell_premium:.1f}. "
                f"Net debit: {net_debit:.1f} ({total_cost:.0f} INR). "
                f"Max profit: Rs {max_profit:.0f} ({max_profit/total_cost*100:.0f}% return). "
                f"Max loss = net debit = Rs {total_cost:.0f}."
            ),
        )

    def _estimate_expiry_premium(
        self,
        spot: float,
        strike: float,
        option_type: str,
        dte: int
    ) -> float:
        """Estimate option premium near expiry using Black-Scholes."""
        T = max(dte + 0.1, 0.01) / 365  # at least ~1 hour worth of time
        sigma = 0.15  # conservative IV estimate
        opt = OptionType.CALL if option_type == "CE" else OptionType.PUT
        price = black_scholes_price(spot, strike, T, 0.065, sigma, opt)
        return max(round(price, 2), 0.5)  # minimum 0.5 premium
