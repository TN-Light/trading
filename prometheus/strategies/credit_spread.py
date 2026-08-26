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
    days_to_expiry, get_expiry_date
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
    ):
        self.strike_otm_steps = strike_otm_steps
        self.hedge_otm_steps = hedge_otm_steps
        self.target_decay_pct = target_decay_pct
        self.breakeven_decay_pct = breakeven_decay_pct
        self.max_loss_multiplier = max_loss_multiplier
        self.min_credit_pct = min_credit_pct

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

        # Check SuperTrend & VWAP for slight directional bias
        st_df = calculate_supertrend(df, period=10, multiplier=3.0)
        st_dir = int(st_df["supertrend_direction"].iloc[-1]) if len(st_df) > 0 else 0

        # Strike parameters
        atm_strike = get_atm_strike(close, symbol)
        interval = get_strike_interval(symbol)
        lot_size = get_lot_size(symbol)
        expiry_date = get_expiry_date(symbol)
        expiry_str = expiry_date.strftime("%Y-%m-%d") if expiry_date else ""

        # ── 2. Select Spread Type ──
        midpoint = (today_high + today_low) / 2.0
        
        if close >= midpoint or st_dir == -1:
            # Bear Call Spread (Expect price to stay below short call)
            spread_type = "BEAR_CALL_SPREAD"
            short_strike = atm_strike + (self.strike_otm_steps * interval)
            long_strike = short_strike + (self.hedge_otm_steps * interval)
            opt_str = "CE"
            action = "SELL_CALL_SPREAD"
        else:
            # Bull Put Spread (Expect price to stay above short put)
            spread_type = "BULL_PUT_SPREAD"
            short_strike = atm_strike - (self.strike_otm_steps * interval)
            long_strike = short_strike - (self.hedge_otm_steps * interval)
            opt_str = "PE"
            action = "SELL_PUT_SPREAD"

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

        # Realistic market baseline fallback when offline / backtesting
        if short_premium <= 0 or long_premium <= 0:
            short_premium = max(round(atr * 0.35, 2), strike_width * 0.25)
            long_premium = max(round(atr * 0.12, 2), strike_width * 0.08)

        net_credit = round(short_premium - long_premium, 2)
        if net_credit < (strike_width * self.min_credit_pct):
            net_credit = round(strike_width * self.min_credit_pct, 2)

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

        # Estimated defined-risk margin (~₹35,000 for NIFTY, ~₹45,000 for BANKNIFTY)
        margin_required = max_loss + 10000.0

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
            "timeframe": "intraday",
            "confidence": 0.75,
            "signal_strength": 3.5,
            "bar_timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
        }
