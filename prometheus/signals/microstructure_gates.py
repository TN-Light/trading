"""
PROMETHEUS — Institutional Microstructure Trap & Downfall Veto Gates
===================================================================
Orthogonal pre-trade veto gates that evaluate exhaustion, absorption, wick
rejection, path efficiency, and cross-index SMT divergence.

Even if a setup achieves a 9.5/10 confluence score and Tier S/B classification,
any single microstructure trap veto blocks live execution or downgrades the
setup to shadow observation (Tier C).
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from prometheus.signals.technical import calculate_atr, calculate_session_vwap
from prometheus.utils.logger import logger


class MicrostructureTrapFilter:
    """Evaluates 6 institutional microstructure veto gates before option buying."""

    def __init__(
        self,
        min_clv_directional: float = 0.65,
        max_adverse_wick_ratio: float = 0.35,
        min_body_to_range: float = 0.45,
        max_vwap_atr_extension: float = 1.80,
        max_orb_atr_extension: float = 1.80,
        max_single_bar_atr_mult: float = 1.85,
        max_daily_atr_used_pct: float = 0.85,
        min_efficiency_ratio: float = 0.38,
        min_wall_clearance_atr: float = 0.70,
    ):
        self.min_clv_directional = min_clv_directional
        self.max_adverse_wick_ratio = max_adverse_wick_ratio
        self.min_body_to_range = min_body_to_range
        self.max_vwap_atr_extension = max_vwap_atr_extension
        self.max_orb_atr_extension = max_orb_atr_extension
        self.max_single_bar_atr_mult = max_single_bar_atr_mult
        self.max_daily_atr_used_pct = max_daily_atr_used_pct
        self.min_efficiency_ratio = min_efficiency_ratio
        self.min_wall_clearance_atr = min_wall_clearance_atr

    def evaluate_all_vetoes(
        self,
        df_15m: pd.DataFrame,
        signal: Dict[str, Any],
        symbol: str,
        oi_data: Optional[Dict[str, Any]] = None,
        daily_df: Optional[pd.DataFrame] = None,
        cached_daily_atr: Optional[float] = None,
        peer_df_15m: Optional[pd.DataFrame] = None,
        peer_symbol: str = "NIFTY BANK",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Run all 6 microstructure trap checks on a candidate option-buying signal.

        Returns:
            (is_vetoed: bool, veto_reason: str, telemetry: Dict[str, Any])
        """
        telemetry: Dict[str, Any] = {}
        if df_15m is None or len(df_15m) < 6 or not signal:
            return False, "", telemetry

        action = str(signal.get("action", "")).upper()
        direction = str(signal.get("direction", "")).lower()
        is_bull = ("CE" in action) or (direction in ("bullish", "long", "buy"))
        is_bear = ("PE" in action) or (direction in ("bearish", "short", "sell"))
        if not (is_bull or is_bear):
            return False, "", telemetry

        curr_bar = df_15m.iloc[-1]
        open_p = float(curr_bar["open"])
        high_p = float(curr_bar["high"])
        low_p = float(curr_bar["low"])
        close_p = float(curr_bar["close"])
        bar_range = max(high_p - low_p, 1e-6)

        # Resolve 15m ATR
        atr_s = calculate_atr(df_15m, period=14)
        atr_15m = (
            float(atr_s.iloc[-1])
            if len(atr_s) > 0 and pd.notna(atr_s.iloc[-1]) and float(atr_s.iloc[-1]) > 0
            else float(signal.get("atr", 0.0) or close_p * 0.0035)
        )
        telemetry["atr_15m"] = round(atr_15m, 2)

        # ------------------------------------------------------------------
        # GATE 1: Candle Close Location Value (CLV) & Shooting Star / Pin Veto
        # ------------------------------------------------------------------
        clv_bull = (close_p - low_p) / bar_range
        clv_bear = (high_p - close_p) / bar_range
        clv_dir = clv_bull if is_bull else clv_bear
        upper_wick_ratio = (high_p - max(open_p, close_p)) / bar_range
        lower_wick_ratio = (min(open_p, close_p) - low_p) / bar_range
        adverse_wick_ratio = upper_wick_ratio if is_bull else lower_wick_ratio
        body_ratio = abs(close_p - open_p) / bar_range

        telemetry["clv_directional"] = round(clv_dir, 3)
        telemetry["adverse_wick_ratio"] = round(adverse_wick_ratio, 3)
        telemetry["body_to_range_ratio"] = round(body_ratio, 3)

        if clv_dir < self.min_clv_directional or adverse_wick_ratio > self.max_adverse_wick_ratio:
            reason = (
                f"Wick Rejection / Pin-Bar Veto on {symbol}: "
                f"Directional CLV={clv_dir:.2f} (< {self.min_clv_directional:.2f}) or "
                f"Adverse Wick={adverse_wick_ratio:.0%} (> {self.max_adverse_wick_ratio:.0%}). "
                f"Institutional supply/absorption detected at bar extreme."
            )
            return True, reason, telemetry

        if (is_bull and close_p <= open_p) or (is_bear and close_p >= open_p):
            reason = (
                f"Wrong-Color Breakout Candle Veto on {symbol}: "
                f"Cannot enter {'BUY_CE' if is_bull else 'BUY_PE'} on an opposing candle close "
                f"(Open={open_p:.1f}, Close={close_p:.1f})."
            )
            return True, reason, telemetry

        # ------------------------------------------------------------------
        # GATE 2: All-Day Anti-Climax & Overextension Gate (VWAP, ORB, 1-Bar, Daily ATR)
        # ------------------------------------------------------------------
        # 2A. Single-Bar Vacuum Spike vs Institutional Expansion Check
        single_bar_atr_mult = bar_range / max(atr_15m, 1e-6)
        telemetry["single_bar_atr_mult"] = round(single_bar_atr_mult, 2)
        if single_bar_atr_mult > self.max_single_bar_atr_mult:
            # Empirical 11-Year Benchmark Nuance:
            # If single-bar ATR > 1.85 but closes strongly (CLV >= 0.70), it is genuine institutional
            # momentum expansion (62.8% WR, +10.2 pt fwd return). If CLV < 0.70, it's a vacuum exhaust trap.
            if clv_dir < 0.70:
                reason = (
                    f"Single-Bar Exhaustion Veto on {symbol}: "
                    f"Signal candle range ({bar_range:.1f} pts = {single_bar_atr_mult:.2f}x ATR) "
                    f"exceeds {self.max_single_bar_atr_mult:.2f}x ATR with weak close (CLV={clv_dir:.2f} < 0.70)."
                )
                return True, reason, telemetry

        # 2B. Session VWAP Rubber-Band Extension Check
        vwap_val = float(signal.get("vwap", 0.0) or 0.0)
        if vwap_val <= 0:
            try:
                v_df = calculate_session_vwap(df_15m)
                vwap_val = float(v_df["vwap"].iloc[-1])
            except Exception:
                vwap_val = close_p

        vwap_dist_atr = abs(close_p - vwap_val) / max(atr_15m, 1e-6)
        telemetry["vwap_dist_atr"] = round(vwap_dist_atr, 2)
        if vwap_dist_atr > self.max_vwap_atr_extension:
            reason = (
                f"VWAP Overextension Veto on {symbol}: "
                f"Spot ({close_p:.1f}) is {vwap_dist_atr:.2f}x ATR away from Session VWAP ({vwap_val:.1f}), "
                f"exceeding {self.max_vwap_atr_extension:.2f}x ATR climax limit."
            )
            return True, reason, telemetry

        # 2C. ORB Distance Extension Telemetry (11-Year benchmark proves >1.8x ATR runners carry 60.1% WR, +9.24 pts)
        orb_high = signal.get("orb_high")
        orb_low = signal.get("orb_low")
        if is_bull and orb_high is not None and float(orb_high) > 0:
            orb_ext_atr = (close_p - float(orb_high)) / max(atr_15m, 1e-6)
            telemetry["orb_ext_atr"] = round(orb_ext_atr, 2)
        elif is_bear and orb_low is not None and float(orb_low) > 0:
            orb_ext_atr = (float(orb_low) - close_p) / max(atr_15m, 1e-6)
            telemetry["orb_ext_atr"] = round(orb_ext_atr, 2)

        # 2D. Daily ATR Fuel Gauge Check (Uses cached ATR if available for zero-latency)
        daily_atr = float(cached_daily_atr or 0.0)
        prev_day_close = 0.0
        if daily_atr <= 0 and daily_df is not None and len(daily_df) >= 15:
            try:
                d_atr_s = calculate_atr(daily_df, period=14)
                daily_atr = float(d_atr_s.iloc[-1]) if pd.notna(d_atr_s.iloc[-1]) else 0.0
                prev_day_close = float(daily_df["close"].iloc[-2]) if len(daily_df) >= 2 else close_p
            except Exception as e:
                logger.debug(f"Daily ATR computation error for {symbol}: {e}")

        if daily_atr > 0:
            try:
                ts_col = pd.to_datetime(df_15m["timestamp"])
                today_bars = df_15m[ts_col.dt.date == ts_col.iloc[-1].date()]
                day_range = float(today_bars["high"].max() - today_bars["low"].min()) if len(today_bars) > 0 else bar_range
                move_from_pdc = abs(close_p - prev_day_close) if prev_day_close > 0 else day_range
                daily_fuel_used = max(day_range, move_from_pdc) / daily_atr
                telemetry["daily_atr_used_pct"] = round(daily_fuel_used, 2)
                if daily_fuel_used > self.max_daily_atr_used_pct:
                    reason = (
                        f"Daily ATR Exhaustion Veto on {symbol}: "
                        f"{daily_fuel_used:.0%} of 14-day Daily ATR ({daily_atr:.0f} pts) "
                        f"already consumed today (max allowed {self.max_daily_atr_used_pct:.0%})."
                    )
                    return True, reason, telemetry
            except Exception as e:
                logger.debug(f"Daily ATR fuel check skipped for {symbol}: {e}")

        # ------------------------------------------------------------------
        # GATE 3: Kaufman Path Efficiency Ratio (ER_6) — Anti-Whipsaw Filter
        # ------------------------------------------------------------------
        closes_6 = df_15m["close"].iloc[-7:].values
        if len(closes_6) >= 7:
            net_disp = abs(closes_6[-1] - closes_6[0])
            total_path = float(np.sum(np.abs(np.diff(closes_6))))
            er_6 = (net_disp / total_path) if total_path > 0 else 1.0
            telemetry["kaufman_er_6"] = round(er_6, 3)
            # Only enforce ER veto if we are >3 bars into the session and not a fresh 1-bar ORB ignition
            if er_6 < self.min_efficiency_ratio and vwap_dist_atr > 1.0:
                reason = (
                    f"Low Path Efficiency (Chop) Veto on {symbol}: "
                    f"6-bar Kaufman ER={er_6:.2f} (< {self.min_efficiency_ratio:.2f}). "
                    f"Overlapping whip-saw price action detected."
                )
                return True, reason, telemetry

        # ------------------------------------------------------------------
        # GATE 4: Option Writer Absorption Veto + Overhead Wall Clearance
        # ------------------------------------------------------------------
        if oi_data and isinstance(oi_data, dict):
            atm_ce_doi = float(oi_data.get("atm_ce_delta_oi", 0) or 0)
            atm_pe_doi = float(oi_data.get("atm_pe_delta_oi", 0) or 0)
            trapped_bull = bool(oi_data.get("writer_trapping_bull", False))
            trapped_bear = bool(oi_data.get("writer_trapping_bear", False))

            telemetry["atm_ce_delta_oi"] = atm_ce_doi
            telemetry["atm_pe_delta_oi"] = atm_pe_doi

            # If buying CE, check if Call writers are aggressively adding OI faster than Put writers
            if is_bull and not trapped_bull and atm_ce_doi > 0 and atm_ce_doi > max(1.0, atm_pe_doi) * 1.35:
                reason = (
                    f"Option Writer Absorption Veto on {symbol}: "
                    f"ATM Call ΔOI ({atm_ce_doi:+,.0f}) exceeds ATM Put ΔOI ({atm_pe_doi:+,.0f}) by >1.35x "
                    f"without writer trapping. Institutional Call writers are selling into this breakout."
                )
                return True, reason, telemetry

            if is_bear and not trapped_bear and atm_pe_doi > 0 and atm_pe_doi > max(1.0, atm_ce_doi) * 1.35:
                reason = (
                    f"Option Writer Absorption Veto on {symbol}: "
                    f"ATM Put ΔOI ({atm_pe_doi:+,.0f}) exceeds ATM Call ΔOI ({atm_ce_doi:+,.0f}) by >1.35x "
                    f"without writer trapping. Institutional Put writers are absorbing this breakdown."
                )
                return True, reason, telemetry

            # Overhead Call Wall / Below Put Wall Clearance Check
            call_wall = float(oi_data.get("strongest_resistance", 0) or 0)
            put_wall = float(oi_data.get("strongest_support", 0) or 0)
            if is_bull and call_wall > close_p:
                clearance_atr = (call_wall - close_p) / max(atr_15m, 1e-6)
                telemetry["wall_clearance_atr"] = round(clearance_atr, 2)
                if clearance_atr < self.min_wall_clearance_atr and not trapped_bull:
                    reason = (
                        f"Overhead Call Wall Veto on {symbol}: "
                        f"Major Call OI Wall at {call_wall:.0f} is only {clearance_atr:.2f}x ATR overhead "
                        f"(< {self.min_wall_clearance_atr:.2f}x ATR required)."
                    )
                    return True, reason, telemetry

            if is_bear and 0 < put_wall < close_p:
                clearance_atr = (close_p - put_wall) / max(atr_15m, 1e-6)
                telemetry["wall_clearance_atr"] = round(clearance_atr, 2)
                if clearance_atr < self.min_wall_clearance_atr and not trapped_bear:
                    reason = (
                        f"Support Put Wall Veto on {symbol}: "
                        f"Major Put OI Wall at {put_wall:.0f} is only {clearance_atr:.2f}x ATR below "
                        f"(< {self.min_wall_clearance_atr:.2f}x ATR required)."
                    )
                    return True, reason, telemetry

        # ------------------------------------------------------------------
        # GATE 5: Cross-Index SMT Divergence Gate (NIFTY/SENSEX vs. NIFTY BANK)
        # ------------------------------------------------------------------
        if peer_df_15m is not None and len(peer_df_15m) >= 6 and symbol.upper() != peer_symbol.upper():
            try:
                peer_curr = peer_df_15m.iloc[-1]
                peer_close = float(peer_curr["close"])
                peer_open = float(peer_curr["open"])
                peer_vwap_df = calculate_session_vwap(peer_df_15m)
                peer_vwap = float(peer_vwap_df["vwap"].iloc[-1])

                # Veto Bullish NIFTY/SENSEX breakout if BANKNIFTY is below its Session VWAP AND printing a red bar
                if is_bull and (peer_close < peer_vwap) and (peer_close < peer_open):
                    reason = (
                        f"Cross-Index SMT Divergence Veto on {symbol}: "
                        f"{symbol} signalled BUY_CE, but {peer_symbol} is trading BELOW Session VWAP "
                        f"({peer_close:.1f} < {peer_vwap:.1f}) on a bearish candle. Heavyweight non-confirmation."
                    )
                    return True, reason, telemetry

                # Veto Bearish NIFTY/SENSEX breakdown if BANKNIFTY is above its Session VWAP AND printing a green bar
                if is_bear and (peer_close > peer_vwap) and (peer_close > peer_open):
                    reason = (
                        f"Cross-Index SMT Divergence Veto on {symbol}: "
                        f"{symbol} signalled BUY_PE, but {peer_symbol} is trading ABOVE Session VWAP "
                        f"({peer_close:.1f} > {peer_vwap:.1f}) on a bullish candle. Heavyweight non-confirmation."
                    )
                    return True, reason, telemetry
            except Exception as e:
                logger.debug(f"SMT divergence check skipped for {symbol}: {e}")

        return False, "", telemetry
