# ============================================================================
# PROMETHEUS — Price Action & Momentum Scalping Engine
# ============================================================================
"""
Specialized Intraday Momentum & Price Action Signal Generator for Indian F&O.

Focuses on high-velocity momentum expansions:
1. Opening Range Breakout (ORB: 09:15 - 09:45 box breakout)
2. VWAP Alignment & Slope (institutional flow direction)
3. SuperTrend (10, 3) Trend Confirmation
4. Consolidation Box Squeeze & Breakout
5. Volume Expansion
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from typing import Dict, Optional, Tuple, List

from prometheus.signals.technical import (
    calculate_vwap, calculate_session_vwap, calculate_supertrend, calculate_atr, calculate_ema
)
from prometheus.utils.logger import logger


class PriceActionMomentumScanner:
    """Detects high-probability intraday price action and momentum breakouts."""

    def __init__(
        self,
        orb_start: dtime = dtime(9, 15),
        orb_end: dtime = dtime(9, 45),
        min_atr_buffer: float = 0.05,
        volume_surge_mult: float = 1.15,
    ):
        self.orb_start = orb_start
        self.orb_end = orb_end
        self.min_atr_buffer = min_atr_buffer
        self.volume_surge_mult = volume_surge_mult

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY 50",
        is_expiry_day: bool = False,
    ) -> Optional[Dict]:
        """Evaluate latest candle for price action / momentum breakout.

        Args:
            df: DataFrame with OHLCV data and 'timestamp' column.
            symbol: Index symbol name.
            is_expiry_day: Whether today is weekly expiry day for this symbol.

        Returns:
            Dict with signal metadata or None if no actionable breakout.
        """
        if df is None or len(df) < 15:
            return None

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        current_row = df.iloc[-1]
        current_ts = current_row["timestamp"]
        current_time = current_ts.time() if hasattr(current_ts, "time") else dtime(10, 0)
        current_date = current_ts.date() if hasattr(current_ts, "date") else None

        # Session Time Gate: Skip first 35 mins (09:15 - 09:50) to avoid open chop
        # On expiry sessions, allow signals up to 15:05 (Power Hour)
        cutoff_time = dtime(15, 5) if is_expiry_day else dtime(14, 30)
        if current_time < dtime(9, 50) or current_time > cutoff_time:
            return None

        # Dead zone check: 11:45 - 13:00 (lunchtime chop)
        if dtime(11, 45) <= current_time <= dtime(13, 0):
            return None

        # Extract today's bars
        if current_date:
            today_bars = df[df["timestamp"].dt.date == current_date].copy()
        else:
            today_bars = df.iloc[-25:].copy()

        if len(today_bars) < 3:
            return None

        # ── 1. Calculate Technical Overlays ──
        close = float(current_row["close"])
        high = float(current_row["high"])
        low = float(current_row["low"])
        open_p = float(current_row["open"])

        # ATR for volatility sizing
        atr_s = calculate_atr(df, period=14)
        atr = float(atr_s.iloc[-1]) if len(atr_s) > 0 and not np.isnan(atr_s.iloc[-1]) else close * 0.005

        # VWAP (session anchored with typical price fallback on 0 volume)
        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            if "volume" in df.columns and df["volume"].sum() > 0:
                vwap_df = calculate_session_vwap(df)
                vwap_val = float(vwap_df["vwap"].iloc[-1])
                vwap = vwap_val if not np.isnan(vwap_val) and vwap_val > 0 else float(typical_price.iloc[-1])
            else:
                vwap = float(typical_price.rolling(20, min_periods=1).mean().iloc[-1])
        except Exception:
            vwap = close

        # SuperTrend (10, 3)
        st_df = calculate_supertrend(df, period=10, multiplier=3.0)
        st_dir = int(st_df["supertrend_direction"].iloc[-1]) if len(st_df) > 0 else 0
        st_val = float(st_df["supertrend"].iloc[-1]) if len(st_df) > 0 else close

        # EMA 9 & 21
        ema9_s = calculate_ema(df["close"], period=9)
        ema21_s = calculate_ema(df["close"], period=21)
        ema9 = float(ema9_s.iloc[-1]) if len(ema9_s) > 0 else close
        ema21 = float(ema21_s.iloc[-1]) if len(ema21_s) > 0 else close

        # ── 2. Opening Range Calculation (09:15 - 09:45) ──
        orb_bars = today_bars[
            (today_bars["timestamp"].dt.time >= self.orb_start)
            & (today_bars["timestamp"].dt.time <= self.orb_end)
        ]

        orb_high = None
        orb_low = None
        if len(orb_bars) >= 2:
            orb_high = float(orb_bars["high"].max())
            orb_low = float(orb_bars["low"].min())

        # ── 3. Momentum Setup Detection ──
        buffer = atr * self.min_atr_buffer
        bull_reasons: List[str] = []
        bear_reasons: List[str] = []
        bull_score = 0.0
        bear_score = 0.0

        # Check A: ORB Breakout
        if orb_high is not None and orb_low is not None:
            if close > (orb_high + buffer):
                bull_score += 2.0
                bull_reasons.append(f"ORB_Breakout_High({orb_high:.1f})")
            elif close < (orb_low - buffer):
                bear_score += 2.0
                bear_reasons.append(f"ORB_Breakdown_Low({orb_low:.1f})")

        # Check B: VWAP Positioning & Slope
        if close > vwap:
            vwap_dist = (close - vwap) / close
            if vwap_dist > 0.0005:
                bull_score += 1.5
                bull_reasons.append("Above_VWAP")
        elif close < vwap:
            vwap_dist = (vwap - close) / close
            if vwap_dist > 0.0005:
                bear_score += 1.5
                bear_reasons.append("Below_VWAP")

        # Check C: SuperTrend Alignment
        if st_dir == 1:
            bull_score += 1.5
            bull_reasons.append(f"SuperTrend_Bull({st_val:.1f})")
        elif st_dir == -1:
            bear_score += 1.5
            bear_reasons.append(f"SuperTrend_Bear({st_val:.1f})")

        # Check D: EMA Alignment
        if ema9 > ema21:
            bull_score += 1.0
            bull_reasons.append("EMA_9x21_Bullish")
        elif ema9 < ema21:
            bear_score += 1.0
            bear_reasons.append("EMA_9x21_Bearish")

        # Check E: Consolidation Box Breakout (Last 4 bars squeeze)
        if len(today_bars) >= 5:
            recent_4 = today_bars.iloc[-5:-1]
            box_high = float(recent_4["high"].max())
            box_low = float(recent_4["low"].min())
            box_range = box_high - box_low
            
            # If previous 4 bars were in tight consolidation (< 1.2 * ATR)
            if box_range <= 1.5 * atr:
                if close > box_high:
                    bull_score += 1.5
                    bull_reasons.append("Consolidation_Breakout_Up")
                elif close < box_low:
                    bear_score += 1.5
                    bear_reasons.append("Consolidation_Breakdown_Down")

        # ── 4. Decision & Trade Structuring ──
        # Minimum score threshold of 3.5 requires at least ORB + VWAP or SuperTrend + Box
        min_threshold = 3.5
        net_edge = bull_score - bear_score

        if bull_score >= min_threshold and net_edge >= 1.5:
            action = "BUY_CE"
            direction = "bullish"
            reasons = bull_reasons
            confidence = min(0.60 + (bull_score / 10.0), 0.90)
            
            # SL: below recent swing low or SuperTrend line
            recent_low = float(today_bars.iloc[-3:]["low"].min())
            sl = max(recent_low - buffer, close - (1.5 * atr))
            target = close + (2.0 * atr)
            rr = (target - close) / max(close - sl, 1.0)

            return {
                "symbol": symbol,
                "action": action,
                "direction": direction,
                "confidence": round(confidence, 2),
                "edge_score": round(bull_score, 2),
                "entry_price": round(close, 2),
                "stop_loss": round(sl, 2),
                "target": round(target, 2),
                "risk_reward": round(rr, 2),
                "strategy": f"PriceAction_Momentum ({'+'.join(reasons)})",
                "reasons": reasons,
                "bar_timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
            }

        elif bear_score >= min_threshold and net_edge <= -1.5:
            action = "BUY_PE"
            direction = "bearish"
            reasons = bear_reasons
            confidence = min(0.60 + (bear_score / 10.0), 0.90)

            # SL: above recent swing high or SuperTrend line
            recent_high = float(today_bars.iloc[-3:]["high"].max())
            sl = min(recent_high + buffer, close + (1.5 * atr))
            target = close - (2.0 * atr)
            rr = (close - target) / max(sl - close, 1.0)

            return {
                "symbol": symbol,
                "action": action,
                "direction": direction,
                "confidence": round(confidence, 2),
                "edge_score": round(bear_score, 2),
                "entry_price": round(close, 2),
                "stop_loss": round(sl, 2),
                "target": round(target, 2),
                "risk_reward": round(rr, 2),
                "strategy": f"PriceAction_Momentum ({'+'.join(reasons)})",
                "reasons": reasons,
                "bar_timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
            }

        return None

    def evaluate_5m_expiry_surge(
        self,
        df_5m: pd.DataFrame,
        df_15m: Optional[pd.DataFrame] = None,
        symbol: str = "NIFTY 50",
    ) -> Optional[Dict]:
        """Detect fast 5-minute VWAP reclaim / gamma surges on weekly expiry sessions."""
        if df_5m is None or len(df_5m) < 15:
            return None

        if not pd.api.types.is_datetime64_any_dtype(df_5m["timestamp"]):
            df_5m = df_5m.copy()
            df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])

        current_row = df_5m.iloc[-1]
        prev_row = df_5m.iloc[-2]
        current_ts = current_row["timestamp"]
        current_time = current_ts.time() if hasattr(current_ts, "time") else dtime(10, 0)
        current_date = current_ts.date() if hasattr(current_ts, "date") else None

        # Expiry fast trigger operates from 09:35 to 15:05 (Expiry Power Hour)
        if current_time < dtime(9, 35) or current_time > dtime(15, 5):
            return None

        if current_date:
            today_bars = df_5m[df_5m["timestamp"].dt.date == current_date].copy()
        else:
            today_bars = df_5m.iloc[-30:].copy()

        if len(today_bars) < 4:
            return None

        close = float(current_row["close"])
        high = float(current_row["high"])
        low = float(current_row["low"])
        prev_close = float(prev_row["close"])

        # 5-min VWAP
        try:
            typical_price = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3
            if "volume" in df_5m.columns and df_5m["volume"].sum() > 0:
                vwap_df = calculate_session_vwap(df_5m)
                vwap_val = float(vwap_df["vwap"].iloc[-1])
                vwap = vwap_val if not np.isnan(vwap_val) and vwap_val > 0 else float(typical_price.iloc[-1])
            else:
                vwap = float(typical_price.rolling(20, min_periods=1).mean().iloc[-1])
        except Exception:
            vwap = close

        # 5-min ATR
        atr_s = calculate_atr(df_5m, period=14)
        atr_5m = float(atr_s.iloc[-1]) if len(atr_s) > 0 and not np.isnan(atr_s.iloc[-1]) else close * 0.003

        # Volume expansion check (if available)
        vol_surge = True
        if "volume" in df_5m.columns and df_5m["volume"].iloc[-10:-1].mean() > 0:
            avg_vol = float(df_5m["volume"].iloc[-10:-1].mean())
            curr_vol = float(current_row["volume"])
            vol_surge = (curr_vol >= avg_vol * 1.20)

        # 1. Bullish 5-min VWAP Reclaim or Consolidation Breakout
        recent_3 = today_bars.iloc[-4:-1]
        box_high_3 = float(recent_3["high"].max())
        box_low_3 = float(recent_3["low"].min())

        is_bull_reclaim = (prev_close <= vwap * 1.001 and close > vwap * 1.001)
        is_bull_breakout = (close > box_high_3 and (close - prev_close) > 0.5 * atr_5m)

        if (is_bull_reclaim or is_bull_breakout) and close > vwap and vol_surge:
            reasons = ["5M_Expiry_Surge"]
            if is_bull_reclaim:
                reasons.append("VWAP_Reclaim_Up")
            if is_bull_breakout:
                reasons.append("Box_Breakout_Up")

            sl = max(low - (0.5 * atr_5m), close - (1.5 * atr_5m))
            target = close + (2.0 * atr_5m)
            rr = (target - close) / max(close - sl, 1.0)
            return {
                "symbol": symbol,
                "action": "BUY_CE",
                "direction": "bullish",
                "confidence": 0.80,
                "edge_score": 4.5,
                "entry_price": round(close, 2),
                "stop_loss": round(sl, 2),
                "target": round(target, 2),
                "risk_reward": round(rr, 2),
                "strategy": f"Expiry_FastTrigger ({'+'.join(reasons)})",
                "reasons": reasons,
                "timeframe": "5minute",
                "fast_expiry_surge": True,
                "bar_timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
            }

        # 2. Bearish 5-min VWAP Breakdown
        is_bear_breakdown = (prev_close >= vwap * 0.999 and close < vwap * 0.999)
        is_bear_box_break = (close < box_low_3 and (prev_close - close) > 0.5 * atr_5m)

        if (is_bear_breakdown or is_bear_box_break) and close < vwap and vol_surge:
            reasons = ["5M_Expiry_Surge"]
            if is_bear_breakdown:
                reasons.append("VWAP_Breakdown_Down")
            if is_bear_box_break:
                reasons.append("Box_Breakdown_Down")

            sl = min(high + (0.5 * atr_5m), close + (1.5 * atr_5m))
            target = close - (2.0 * atr_5m)
            rr = (close - target) / max(sl - close, 1.0)
            return {
                "symbol": symbol,
                "action": "BUY_PE",
                "direction": "bearish",
                "confidence": 0.80,
                "edge_score": 4.5,
                "entry_price": round(close, 2),
                "stop_loss": round(sl, 2),
                "target": round(target, 2),
                "risk_reward": round(rr, 2),
                "strategy": f"Expiry_FastTrigger ({'+'.join(reasons)})",
                "reasons": reasons,
                "timeframe": "5minute",
                "fast_expiry_surge": True,
                "bar_timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
            }

        return None
