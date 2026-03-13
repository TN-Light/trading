# ============================================================================
# PROMETHEUS — Strategy Module: Volatility Trading
# ============================================================================
"""
Volatility-based strategies for event-driven and VIX-regime trades.

Optimized for under 2L capital:
  - Buy straddles/strangles BEFORE volatility events (RBI, Budget, US Fed)
  - Buy options when IV percentile is LOW (cheap vol)
  - Hold through the event, exit on IV expansion
  - Use debit spreads to reduce cost

Key insight: In Indian markets, options are systematically mispriced
around scheduled events. IV typically underprices the actual move on
RBI policy days and Budget day.

Events calendar:
  - RBI Monetary Policy (bi-monthly) → BANKNIFTY
  - Union Budget (Feb) → NIFTY broad
  - US Fed FOMC (6-weekly) → FII flow impact
  - Quarterly results season → Stock options
  - India VIX mean-reversion after spikes
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, date, timedelta

from prometheus.utils.indian_market import get_lot_size, get_atm_strike, get_strike_interval
from prometheus.utils.options_math import (
    OptionType, black_scholes_price, calculate_greeks,
    iv_percentile, iv_rank
)
from prometheus.utils.logger import logger, log_signal


# Known event types and their typical impact
EVENT_IMPACT = {
    "rbi_policy": {
        "instruments": ["NIFTY BANK", "NIFTY 50"],
        "typical_move_pct": 1.5,      # typical 1.5% move on BANKNIFTY
        "iv_expansion_typical": 30,    # IV typically expands 30%
        "entry_days_before": 2,
        "exit_after": "event",
    },
    "union_budget": {
        "instruments": ["NIFTY 50", "NIFTY BANK"],
        "typical_move_pct": 2.5,
        "iv_expansion_typical": 50,
        "entry_days_before": 3,
        "exit_after": "event",
    },
    "us_fed_fomc": {
        "instruments": ["NIFTY 50"],
        "typical_move_pct": 1.0,
        "iv_expansion_typical": 20,
        "entry_days_before": 1,
        "exit_after": "next_day",
    },
    "quarterly_results": {
        "instruments": [],  # specific stock
        "typical_move_pct": 3.0,
        "iv_expansion_typical": 40,
        "entry_days_before": 2,
        "exit_after": "event",
    },
    "vix_mean_reversion": {
        "instruments": ["NIFTY 50"],
        "typical_move_pct": 0.0,
        "iv_expansion_typical": 0,
        "entry_days_before": 0,
        "exit_after": "vix_normalize",
    },
}


@dataclass
class VolatilitySetup:
    """A volatility trade setup."""
    symbol: str
    strategy_type: str          # "long_straddle", "long_strangle", "debit_spread"
    legs: List[Dict]            # each leg: {action, option_type, strike, premium, lots}
    total_cost: float           # max loss = total cost for buying strategies
    max_profit: float           # estimated max profit
    breakeven_up: float
    breakeven_down: float
    event: str
    entry_date: str
    expected_exit: str
    signal_strength: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy_type": self.strategy_type,
            "legs": self.legs,
            "total_cost": self.total_cost,
            "max_profit": self.max_profit,
            "breakeven_up": self.breakeven_up,
            "breakeven_down": self.breakeven_down,
            "event": self.event,
            "signal_strength": self.signal_strength,
            "reasoning": self.reasoning,
        }


class VolatilityStrategy:
    """
    Volatility trading strategies — buy cheap vol before events.
    """

    def __init__(self, config: Dict, capital: float = 200000):
        self.config = config
        self.capital = capital
        self.max_position_cost = capital * config.get("max_position_cost", 0.20)
        self.iv_buy_threshold = config.get("iv_percentile_trigger", 30)
        self.entry_days_before = config.get("event_proximity_days", 2)

    def check_event_opportunity(
        self,
        symbol: str,
        spot_price: float,
        current_iv: float,
        iv_history: np.ndarray,
        upcoming_events: List[Dict],
        options_chain: Optional[pd.DataFrame] = None,
    ) -> Optional[VolatilitySetup]:
        """
        Check if there's a volatility buying opportunity.

        Conditions:
        1. Known event within entry_days_before trading days
        2. Current IV percentile is LOW (vol is cheap)
        3. Position cost within capital limits
        """
        # Check for upcoming events
        today = date.today()
        matching_event = None

        for event in upcoming_events:
            event_date = event.get("date")
            if isinstance(event_date, str):
                event_date = datetime.strptime(event_date, "%Y-%m-%d").date()

            days_until = (event_date - today).days
            event_type = event.get("type", "")
            event_config = EVENT_IMPACT.get(event_type, {})
            entry_window = event_config.get("entry_days_before", self.entry_days_before)

            if 0 < days_until <= entry_window:
                if symbol in event_config.get("instruments", [symbol]):
                    matching_event = event
                    matching_event["days_until"] = days_until
                    matching_event["config"] = event_config
                    break

        if not matching_event:
            return None

        # Check IV percentile
        if len(iv_history) > 0:
            current_iv_pct = iv_percentile(iv_history, current_iv)
            current_iv_rnk = iv_rank(iv_history, current_iv)
        else:
            current_iv_pct = 50  # assume mid if no history
            current_iv_rnk = 50

        # Buy vol when IV is LOW (it will expand as event approaches)
        if current_iv_pct > self.iv_buy_threshold:
            logger.debug(
                f"Volatility: IV percentile {current_iv_pct:.0f} > threshold {self.iv_buy_threshold}. "
                f"Vol not cheap enough."
            )
            return None

        # Build the trade
        setup = self._build_long_straddle(
            symbol, spot_price, current_iv, matching_event, options_chain
        )

        return setup

    def check_vix_mean_reversion(
        self,
        vix: float,
        vix_history: pd.Series,
        symbol: str = "NIFTY 50",
        spot_price: float = 0
    ) -> Optional[Dict]:
        """
        VIX mean-reversion signal.

        When India VIX spikes above 20 and starts declining, buy NIFTY calls
        (market tends to stabilize/recover as fear subsides).

        When VIX drops below 12, buy cheap OTM strangles
        (volatility is compressed and will eventually expand).
        """
        if vix_history.empty or len(vix_history) < 20:
            return None

        vix_20d_mean = vix_history.tail(20).mean()
        vix_percentile = (vix_history < vix).sum() / len(vix_history) * 100

        signal = None

        # VIX spike recovery — bullish signal for indices
        if vix > 20 and vix < vix_history.tail(3).max():
            # VIX peaked and is declining
            signal = {
                "type": "vix_spike_recovery",
                "direction": "bullish",
                "strength": min((vix - 20) / 15, 0.8),
                "vix": vix,
                "reasoning": (
                    f"VIX at {vix:.1f} (above 20) but declining from peak. "
                    f"Fear subsiding — bullish for NIFTY."
                ),
            }

        # VIX compression — buy vol (strangles)
        elif vix < 12:
            signal = {
                "type": "vix_compression",
                "direction": "neutral",  # direction unknown, just that vol will expand
                "strength": min((12 - vix) / 5, 0.7),
                "vix": vix,
                "reasoning": (
                    f"VIX at {vix:.1f} — below 12, extremely compressed. "
                    f"Cheap to buy strangles. Vol expansion likely."
                ),
            }

        return signal

    def _build_long_straddle(
        self,
        symbol: str,
        spot: float,
        iv: float,
        event: Dict,
        chain: Optional[pd.DataFrame]
    ) -> Optional[VolatilitySetup]:
        """Build a long straddle (buy ATM call + ATM put)."""
        strike = get_atm_strike(spot, symbol)
        lot_size = get_lot_size(symbol)

        # Get premiums
        call_premium = self._get_premium(chain, strike, "CE", spot, iv, event["days_until"])
        put_premium = self._get_premium(chain, strike, "PE", spot, iv, event["days_until"])

        if call_premium <= 0 or put_premium <= 0:
            return None

        total_premium = call_premium + put_premium
        total_cost_per_lot = total_premium * lot_size

        # Check capital limits
        if total_cost_per_lot > self.max_position_cost:
            # Try a strangle instead (cheaper)
            return self._build_long_strangle(symbol, spot, iv, event, chain)

        lots = max(1, int(self.max_position_cost / total_cost_per_lot))
        total_cost = total_cost_per_lot * lots

        # Breakeven points
        breakeven_up = strike + total_premium
        breakeven_down = strike - total_premium

        typical_move = event["config"].get("typical_move_pct", 1.5) / 100
        expected_spot_move = spot * typical_move
        estimated_profit = (expected_spot_move - total_premium) * lot_size * lots

        signal_strength = self._calculate_vol_signal_strength(event, iv)

        return VolatilitySetup(
            symbol=symbol,
            strategy_type="long_straddle",
            legs=[
                {
                    "action": "BUY", "option_type": "CE", "strike": strike,
                    "premium": call_premium, "lots": lots, "lot_size": lot_size
                },
                {
                    "action": "BUY", "option_type": "PE", "strike": strike,
                    "premium": put_premium, "lots": lots, "lot_size": lot_size
                },
            ],
            total_cost=round(total_cost, 2),
            max_profit=round(max(estimated_profit, 0), 2),
            breakeven_up=round(breakeven_up, 2),
            breakeven_down=round(breakeven_down, 2),
            event=event.get("type", "unknown"),
            entry_date=str(date.today()),
            expected_exit=event.get("date", "unknown"),
            signal_strength=round(signal_strength, 3),
            reasoning=(
                f"Long Straddle on {symbol} before {event.get('type', 'event')} "
                f"({event.get('days_until', '?')} days away). "
                f"IV is low — cheap entry. "
                f"Expected {event['config'].get('typical_move_pct', 1.5)}% move. "
                f"Total cost: Rs {total_cost:.0f}. "
                f"Breakevens: {breakeven_down:.0f} / {breakeven_up:.0f}."
            ),
        )

    def _build_long_strangle(
        self,
        symbol: str,
        spot: float,
        iv: float,
        event: Dict,
        chain: Optional[pd.DataFrame]
    ) -> Optional[VolatilitySetup]:
        """Build a long strangle (buy OTM call + OTM put) — cheaper than straddle."""
        interval = get_strike_interval(symbol)
        atm = get_atm_strike(spot, symbol)
        call_strike = atm + interval
        put_strike = atm - interval
        lot_size = get_lot_size(symbol)

        call_premium = self._get_premium(chain, call_strike, "CE", spot, iv, event["days_until"])
        put_premium = self._get_premium(chain, put_strike, "PE", spot, iv, event["days_until"])

        if call_premium <= 0 or put_premium <= 0:
            return None

        total_premium = call_premium + put_premium
        total_cost_per_lot = total_premium * lot_size

        if total_cost_per_lot > self.max_position_cost:
            return None

        lots = max(1, int(self.max_position_cost / total_cost_per_lot))
        total_cost = total_cost_per_lot * lots

        breakeven_up = call_strike + total_premium
        breakeven_down = put_strike - total_premium

        signal_strength = self._calculate_vol_signal_strength(event, iv)

        return VolatilitySetup(
            symbol=symbol,
            strategy_type="long_strangle",
            legs=[
                {
                    "action": "BUY", "option_type": "CE", "strike": call_strike,
                    "premium": call_premium, "lots": lots, "lot_size": lot_size
                },
                {
                    "action": "BUY", "option_type": "PE", "strike": put_strike,
                    "premium": put_premium, "lots": lots, "lot_size": lot_size
                },
            ],
            total_cost=round(total_cost, 2),
            max_profit=0,  # unlimited theoretically
            breakeven_up=round(breakeven_up, 2),
            breakeven_down=round(breakeven_down, 2),
            event=event.get("type", "unknown"),
            entry_date=str(date.today()),
            expected_exit=event.get("date", "unknown"),
            signal_strength=round(signal_strength, 3),
            reasoning=(
                f"Long Strangle on {symbol} before {event.get('type', 'event')}. "
                f"Cheaper than straddle. "
                f"Total cost: Rs {total_cost:.0f}. Max loss = total cost."
            ),
        )

    def _get_premium(
        self,
        chain: Optional[pd.DataFrame],
        strike: float,
        option_type: str,
        spot: float,
        iv: float,
        dte: int
    ) -> float:
        """Get option premium from chain or estimate via Black-Scholes."""
        if chain is not None and not chain.empty:
            match = chain[
                (chain["strike"] == strike) &
                (chain["option_type"] == option_type)
            ]
            if not match.empty:
                ltp = match["ltp"].iloc[0]
                if ltp > 0:
                    return ltp

        T = max(dte, 1) / 365
        sigma = iv / 100 if iv > 1 else iv  # handle both 0.15 and 15 formats
        if sigma <= 0:
            sigma = 0.15
        opt = OptionType.CALL if option_type == "CE" else OptionType.PUT
        return black_scholes_price(spot, strike, T, 0.065, sigma, opt)

    def _calculate_vol_signal_strength(self, event: Dict, iv: float) -> float:
        """Calculate signal strength for volatility trade."""
        # Base strength from event importance
        event_type = event.get("type", "")
        base_strength = {
            "rbi_policy": 0.80,
            "union_budget": 0.85,
            "us_fed_fomc": 0.65,
            "quarterly_results": 0.70,
            "vix_mean_reversion": 0.60,
        }.get(event_type, 0.50)

        # Boost if IV is particularly cheap
        if iv < 12:
            base_strength += 0.10
        elif iv < 15:
            base_strength += 0.05

        return min(base_strength, 1.0)
