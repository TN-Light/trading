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
        max_days_to_expiry: Optional[int] = 1,  # Strictly <= 1 DTE (0-DTE or 1-DTE only)
        max_days_to_expiry_monthly: Optional[int] = None,  # None: follow max_days_to_expiry (banning DTE >= 2 across all indices)
        max_range_pct: float = 0.018,              # 1.80% default day-range gate
        start_time: Optional[dtime] = None,        # Expiry afternoon window: 12:30 PM (when specified)
        end_time: Optional[dtime] = None,          # 15:00 PM
        require_long_gamma: bool = False,          # Dealer Long Gamma Confirmation (net_gex > +1.5)
        min_net_gex: float = 1.5,                  # Minimum Net GEX threshold
        short_gamma_ban_threshold: float = -0.5,   # Ban option selling when net_gex < -0.5
        require_macro_trend: bool = False,         # 1H Macro Trend Co-Alignment (Close < EMA20 < EMA50 & Spot < VWAP for Calls)
    ):
        self.strike_otm_steps = strike_otm_steps
        self.hedge_otm_steps = hedge_otm_steps
        self.target_decay_pct = target_decay_pct
        self.breakeven_decay_pct = breakeven_decay_pct
        self.max_loss_multiplier = max_loss_multiplier
        self.min_credit_pct = min_credit_pct
        self.max_days_to_expiry = max_days_to_expiry
        self.max_days_to_expiry_monthly = max_days_to_expiry_monthly
        self.max_range_pct = max_range_pct
        self.start_time = start_time
        self.end_time = end_time
        self.require_long_gamma = require_long_gamma
        self.min_net_gex = min_net_gex
        self.short_gamma_ban_threshold = short_gamma_ban_threshold
        self.require_macro_trend = require_macro_trend

    @staticmethod
    def get_atm_strike(spot_price: float, symbol: str) -> float:
        """Round spot price to the nearest strike interval for the symbol."""
        return get_atm_strike(spot_price, symbol)

    def evaluate_spread(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY 50",
        capital: float = 50000.0,
        option_chain = None,
        trend: Optional[str] = None,
        df_1h: Optional[pd.DataFrame] = None,
        net_gex: Optional[float] = None,
        stopped_out_directions: Optional[set] = None,
    ) -> Optional[Dict]:
        """Evaluate if market is in a range-bound state and generate a credit spread.

        Args:
            df: OHLCV DataFrame with timestamp.
            symbol: Index symbol name.
            capital: Available capital.
            option_chain: Live AngelOneOptionChain instance for real market LTPs.
            trend: Optional trend hint string.
            df_1h: Optional 1-Hour candles for macro trend alignment.
            net_gex: Optional dealer Net Gamma Exposure.
            stopped_out_directions: Optional set of (symbol, direction) tuples locked out today.

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
        current_time = current_ts.time() if hasattr(current_ts, "time") else dtime(13, 0)
        current_date = current_ts.date() if hasattr(current_ts, "date") else None

        # Trading window check: Strictly 12:30 to 15:00 when start_time/end_time configured, else 09:50 to 15:00
        min_start = self.start_time if self.start_time is not None else dtime(9, 50)
        max_end = self.end_time if self.end_time is not None else dtime(15, 0)
        if current_time < min_start or current_time > max_end:
            logger.info(
                f"CreditSpread skipped for {symbol}: Time {current_time.strftime('%H:%M')} outside "
                f"{min_start.strftime('%H:%M')}–{max_end.strftime('%H:%M')} afternoon window "
                f"(option selling strictly requires afternoon terminal theta decay)."
            )
            return None

        # ── Dealer Gamma Regime Gate (Pillar GEX) ──
        if net_gex is not None and net_gex < self.short_gamma_ban_threshold:
            logger.info(
                f"CreditSpread skipped for {symbol}: Dealer Short Gamma regime (net_gex={net_gex:.2f} < {self.short_gamma_ban_threshold}). "
                f"Option selling is strictly banned in Short Gamma due to explosive trend runner risk."
            )
            return None

        if self.require_long_gamma:
            if net_gex is None:
                logger.info(
                    f"CreditSpread skipped for {symbol}: Dealer GEX telemetry unavailable. "
                    f"Option selling strictly requires net_gex > +{self.min_net_gex:.2f} Long Gamma confirmation."
                )
                return None
            if net_gex < self.min_net_gex:
                logger.info(
                    f"CreditSpread skipped for {symbol}: net_gex={net_gex:.2f} is below Long Gamma confirmation threshold "
                    f"(+{self.min_net_gex:.2f}). Option selling requires confirmed dealer Long Gamma pin."
                )
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

        # Dynamic, index-aware range gate:
        # Runaway trend days typically exceed max_range_pct of spot or 4.0x ATR.
        max_allowed_range = max(4.0 * atr, close * self.max_range_pct)
        if day_range > max_allowed_range:
            if len(today_bars) >= 8:
                recent_bars = today_bars.iloc[-8:]
                recent_range = float(recent_bars["high"].max() - recent_bars["low"].min())
                max_recent_range = max(2.0 * atr, close * 0.008)
                if recent_range <= max_recent_range:
                    logger.info(
                        "CreditSpread rolling consolidation override: cumulative range wide "
                        "due to morning gap, but 2-hour range <= 0.8% spot"
                    )
                else:
                    logger.info(
                        f"CreditSpread skipped for {symbol}: Day range {day_range:.1f} pts exceeds "
                        f"max sideways threshold {max_allowed_range:.1f} pts (strong trend day detected)."
                    )
                    return None
            else:
                logger.info(
                    f"CreditSpread skipped for {symbol}: Day range {day_range:.1f} pts exceeds "
                    f"max sideways threshold {max_allowed_range:.1f} pts (strong trend day detected, "
                    f"insufficient bars for 2-hour consolidation override)."
                )
                return None

        # Check VWAP and EMAs for rigorous alignment
        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
            if "volume" in df.columns and df["volume"].sum() > 0:
                vwap_df = calculate_vwap(df)
                if isinstance(vwap_df, pd.DataFrame) and "vwap" in vwap_df.columns:
                    vwap_val = vwap_df["vwap"].iloc[-1]
                elif isinstance(vwap_df, pd.Series):
                    vwap_val = vwap_df.iloc[-1]
                else:
                    vwap_val = float(typical_price.iloc[-1])
                vwap = float(vwap_val) if (pd.notna(vwap_val) and vwap_val > 0) else float(typical_price.rolling(20, min_periods=1).mean().iloc[-1])
            else:
                vwap = float(typical_price.rolling(20, min_periods=1).mean().iloc[-1])
        except Exception:
            vwap = close
        
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
        dte = (expiry_date - current_date).days if current_date and expiry_date else None

        # Enforce DTE restriction for intraday credit spreads (0-DTE or near-term, decoupled for monthly)
        allowed_dte = (
            self.max_days_to_expiry_monthly
            if symbol in ("NIFTY BANK", "NIFTY FIN SERVICE", "BANKNIFTY", "FINNIFTY")
            else self.max_days_to_expiry
        )
        if allowed_dte is not None and current_date and expiry_date:
            days_to_exp = dte
            if days_to_exp > allowed_dte:
                logger.info(
                    f"CreditSpread skipped for {symbol}: Expiry {expiry_str} is {days_to_exp} days away. "
                    f"Intraday credit spreads strictly require <= {allowed_dte} DTE for rapid theta decay."
                )
                return None

        # ── 2. Select Spread Type with Macro Trend & VWAP Co-Alignment ──
        # Determine 1H Macro Trend
        is_1h_bull = False
        is_1h_bear = False

        if df_1h is not None and len(df_1h) >= 3:
            c_1h = df_1h["close"]
            ema20_1h = float(c_1h.ewm(span=min(20, len(c_1h)), adjust=False).mean().iloc[-1])
            ema50_1h = float(c_1h.ewm(span=min(50, len(c_1h)), adjust=False).mean().iloc[-1])
            close_1h = float(c_1h.iloc[-1])
            is_1h_bull = (close_1h > ema20_1h) and (ema20_1h >= ema50_1h)
            is_1h_bear = (close_1h < ema20_1h) and (ema20_1h <= ema50_1h)
        elif trend is not None:
            t_str = str(trend).lower()
            if t_str in ("bullish", "up", "buy", "1h_bull"):
                is_1h_bull = True
            elif t_str in ("bearish", "down", "sell", "1h_bear"):
                is_1h_bear = True
        elif self.require_macro_trend and len(df) >= 20:
            try:
                df_res = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(df_res["timestamp"]):
                    df_res["timestamp"] = pd.to_datetime(df_res["timestamp"])
                synth_1h = (
                    df_res.set_index("timestamp")
                    .resample("1h")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                    .dropna()
                    .reset_index()
                )
                if len(synth_1h) >= 5:
                    c_syn = synth_1h["close"]
                    ema20_syn = float(c_syn.ewm(span=min(20, len(c_syn)), adjust=False).mean().iloc[-1])
                    ema50_syn = float(c_syn.ewm(span=min(50, len(c_syn)), adjust=False).mean().iloc[-1])
                    close_syn = float(c_syn.iloc[-1])
                    is_1h_bull = (close_syn > ema20_syn) and (ema20_syn >= ema50_syn)
                    is_1h_bear = (close_syn < ema20_syn) and (ema20_syn <= ema50_syn)
            except Exception as syn_err:
                logger.debug(f"Could not synthesize 1H trend from 15M: {syn_err}")

        # Rigorous Macro Trend Co-Alignment:
        # Never sell a Bear Call Spread unless 1H Trend is Bearish (Close < EMA20 < EMA50) and spot is below VWAP.
        # Never sell a Bull Put Spread unless 1H Trend is Bullish (Close > EMA20 > EMA50) and spot is above VWAP.
        is_above_vwap = (close > vwap)
        is_below_vwap = (close < vwap)

        if self.require_macro_trend:
            if is_1h_bear and is_below_vwap:
                is_bearish = True
                is_bullish = False
            elif is_1h_bull and is_above_vwap:
                is_bullish = True
                is_bearish = False
            else:
                logger.info(
                    f"CreditSpread skipped for {symbol}: Macro trend co-alignment failed "
                    f"(1H Bullish={is_1h_bull}, 1H Bearish={is_1h_bear}, Spot {close:.1f} vs VWAP {vwap:.1f}). "
                    f"Never sell Call Spreads in an uptrend or Put Spreads in a downtrend."
                )
                return None
        else:
            # Fallback when require_macro_trend=False (e.g. isolated test environments)
            is_bearish = (close < vwap) and (ema9 <= ema21)
            is_bullish = (close > vwap) and (ema9 >= ema21)
            if not is_bearish and not is_bullish:
                midpoint = (today_high + today_low) / 2.0
                if trend and str(trend).lower() in ("bullish", "up", "buy"):
                    is_bullish = True
                elif trend and str(trend).lower() in ("bearish", "down", "sell"):
                    is_bearish = True
                elif close >= midpoint or st_dir == -1:
                    is_bearish = True
                else:
                    is_bullish = True

        # Directional Symbol Lockout: If a spread suffered a stop-loss today, lock out that (symbol, direction) pair
        candidate_spread_type = "BEAR_CALL_SPREAD" if is_bearish else "BULL_PUT_SPREAD"
        if stopped_out_directions:
            if (symbol, candidate_spread_type) in stopped_out_directions:
                logger.info(
                    f"CreditSpread skipped for {symbol}: ({symbol}, {candidate_spread_type}) "
                    f"already suffered an intraday stop-out today (directional martingale lockout active)."
                )
                return None

        # ── Dynamic 2.0σ Strike Buffer (Pillar 2) ──
        _base_index_buffer = {
            "NIFTY 50": 150.0,
            "NIFTY BANK": 400.0,
            "SENSEX": 600.0,
            "NIFTY MIDCAP SELECT": 100.0,
            "NIFTY FIN SERVICE": 150.0,
        }.get(symbol, 3.0 * interval)

        # DTE-aware buffer: multi-day options have more extrinsic value at closer strikes
        dte_for_buffer = dte if dte is not None else 0

        if dte_for_buffer >= 2:
            # 2-3 DTE: tighter buffer (70% of base) — options still have meaningful premium at closer strikes
            min_index_buffer = round(_base_index_buffer * 0.70 / interval) * interval
        elif dte_for_buffer == 1:
            # 1-DTE: moderate buffer (85% of base)
            min_index_buffer = round(_base_index_buffer * 0.85 / interval) * interval
        else:
            # 0-DTE: full buffer — need maximum distance for terminal gamma safety
            min_index_buffer = _base_index_buffer

        daily_proxy_buffer = round((3.5 * atr) / interval) * interval
        sigma_buffer = max(min_index_buffer, daily_proxy_buffer)

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

        otm_steps = max(self.strike_otm_steps, int(min_index_buffer / interval))
        
        if is_bearish:
            # Bear Call Spread: Place short strike above today's high / resistance
            spread_type = "BEAR_CALL_SPREAD"
            if dte == 0:
                short_strike = max(self.get_atm_strike(today_high + interval, symbol), atm_strike + (2 * interval))
            else:
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
            # Bull Put Spread: Place short strike below today's low / support
            spread_type = "BULL_PUT_SPREAD"
            if dte == 0:
                short_strike = min(self.get_atm_strike(today_low - interval, symbol), atm_strike - (2 * interval))
            else:
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
            "NIFTY FIN SERVICE": "FINNIFTY",
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

                # 3. Hedge Leg Fallback Probing:
                # If the primary calculated hedge strike is unlisted or missing in the broker API
                # (e.g. 23800 CE omitted from Angel One on 2026-09-22), probe adjacent liquid strikes
                # (+1, +2, or -1 step) so profitable, defined-risk spreads are not discarded.
                if short_premium > 0 and long_premium <= 0 and hasattr(option_chain, "get_real_premium"):
                    candidate_steps = [
                        self.hedge_otm_steps + 1,
                        self.hedge_otm_steps + 2,
                        self.hedge_otm_steps - 1,
                    ]
                    for step in candidate_steps:
                        if step < 2:
                            continue
                        probe_strike = (
                            short_strike + (step * interval)
                            if is_bearish
                            else short_strike - (step * interval)
                        )
                        cand_q = option_chain.get_real_premium(
                            symbol, probe_strike, opt_str, expiry=expiry_str, spot_price=close
                        )
                        if cand_q:
                            ltp_val = float(cand_q.get("ltp", 0) or 0)
                            ask_val = float(cand_q.get("ask", 0) or cand_q.get("best_ask", 0) or cand_q.get("sell_price", 0) or 0)
                            fill_val = ask_val if ask_val > 0 else ltp_val
                            oi_val = int(cand_q.get("oi", 0) or cand_q.get("open_interest", 0) or 0)
                            vol_val = int(cand_q.get("volume", 0) or 0)
                            if fill_val > 0 and (oi_val > 0 or vol_val > 0 or fill_val >= 0.5):
                                long_strike = probe_strike
                                long_premium = fill_val
                                long_tradingsymbol = cand_q.get(
                                    "tradingsymbol",
                                    generate_tradingsymbol(underlying, expiry_str, long_strike, opt_str)
                                )
                                logger.info(
                                    f"CreditSpread hedge strike fallback for {symbol}: Primary "
                                    f"{short_strike + (self.hedge_otm_steps * interval if is_bearish else -self.hedge_otm_steps * interval)} "
                                    f"unavailable; adjusted to liquid hedge strike {long_strike}{opt_str} @ Rs {long_premium:.2f} ({long_tradingsymbol})"
                                )
                                break
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
        # DTE-scaled minimum credit: closer expiries have lower premium floors (theta has already decayed)
        effective_dte = dte if dte is not None else 99
        if effective_dte == 0:
            effective_min_pct = 0.05   # 0-DTE: terminal theta, calibrate hurdle to 5% of strike width (min ₹7.50 on Nifty)
        elif effective_dte == 1:
            effective_min_pct = 0.10   # 1-DTE: near-expiry, substantial theta still available
        elif effective_dte <= 3:
            effective_min_pct = 0.12   # 2-3 DTE: meaningful premium, justify overnight risk
        else:
            effective_min_pct = self.min_credit_pct  # 4+: full requirement
        min_required_credit = round(strike_width * effective_min_pct, 2)
        if net_credit < min_required_credit or net_credit <= 0:
            logger.info(
                f"CreditSpread skipped for {symbol}: Net credit Rs {net_credit:.2f} is below "
                f"minimum threshold Rs {min_required_credit:.2f} ({effective_min_pct*100:.0f}% of strike width) "
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
            "NIFTY FIN SERVICE": "FINNIFTY",
        }
        underlying = sym_map.get(symbol, symbol.upper())
        if not short_tradingsymbol:
            short_tradingsymbol = generate_tradingsymbol(underlying, expiry_str, short_strike, opt_str)
        if not long_tradingsymbol:
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
        daily_move = max(min_index_buffer, 3.5 * max(atr, 1.0))
        otm_sigma = round((strike_dist / daily_move) * 2.0, 2)
        
        # Rigorous Pillar Checks:
        # Pillar 1: Statistical buffer (requires >= 2.0σ away from spot for true Tier 1 conviction)
        is_far_otm = bool(otm_sigma >= 2.0)
        is_mod_otm = bool(otm_sigma >= 1.5)
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

        # Sure-Shot / Tier 1 requires 0-DTE + Far OTM (>=2.0σ) + VWAP trend alignment
        is_sure_shot = bool(is_0dte and is_far_otm and trend_aligned)
        
        # Dynamically scaled score (7.0 to 9.5) based on true confluences
        score_calc = 7.0
        if is_0dte:
            score_calc += 1.0
        if is_far_otm:
            score_calc += 0.8
        elif is_mod_otm:
            score_calc += 0.4
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
            "is_0dte": is_0dte,
            "days_to_expiry": (expiry_date - current_date).days if current_date and expiry_date else None,
            "bar_timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
        }
