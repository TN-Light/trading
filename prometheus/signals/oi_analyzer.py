# ============================================================================
# PROMETHEUS — Signal Engine: Open Interest Analysis
# ============================================================================
"""
OI-based analysis for Indian F&O markets.

What this validates from research:
  - OI buildup at strikes DOES indicate institutional positioning
  - OI change velocity is MORE useful than absolute OI (our upgrade)
  - PCR extremes are contrarian indicators (validated)
  - Max Pain has statistical significance near expiry (2-3 days)

What we upgrade:
  - Distinguish OI buildup (new positions) vs unwinding (closing positions)
  - Cross-strike OI distribution analysis (not just green/red bars)
  - OI change velocity (rate of change matters more than level)
  - Combine OI with IV for positioning intent detection
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from prometheus.utils.options_math import max_pain, pcr_ratio
from prometheus.utils.logger import logger


@dataclass
class OISignal:
    """An OI-based market signal."""
    signal_type: str       # "support", "resistance", "buildup", "unwinding", "pcr_extreme"
    direction: str         # "bullish", "bearish", "neutral"
    strength: float        # 0 to 1
    strike: float = 0.0
    details: str = ""


class OIAnalyzer:
    """
    Comprehensive Open Interest analysis engine.
    Processes options chain data to extract institutional positioning signals.
    """

    def __init__(self):
        self.oi_history: List[pd.DataFrame] = []  # Store snapshots for velocity calc
        self.max_history = 100  # Keep last 100 snapshots

    def analyze(self, chain_df: pd.DataFrame, spot_price: float, timestamp: Optional[Any] = None) -> Dict:
        """
        Run full OI analysis on options chain data.

        Returns:
            Dict with all OI metrics and signals
        """
        if chain_df.empty:
            return {"signals": [], "metrics": {}}

        # Validate required columns to prevent history poisoning and runtime crashes
        required_cols = {"option_type", "oi", "strike"}
        if not required_cols.issubset(set(chain_df.columns)):
            logger.warning(f"Malformed option chain DF missing required columns: {required_cols - set(chain_df.columns)}")
            return {"signals": [], "metrics": {}}

        calls = chain_df[chain_df["option_type"] == "CE"].copy()
        puts = chain_df[chain_df["option_type"] == "PE"].copy()

        # Store snapshot for velocity calculation (only after schema validation)
        self.oi_history.append(chain_df.copy())
        if len(self.oi_history) > self.max_history:
            self.oi_history.pop(0)

        signals = []
        metrics = {}

        # 1. PCR Analysis
        pcr = self._calculate_pcr(calls, puts)
        metrics["pcr"] = pcr
        pcr_signal = self._interpret_pcr(pcr)
        if pcr_signal:
            signals.append(pcr_signal)

        # 2. Max Pain
        # Bug (2026-08-17 audit): NIFTY 50 / NIFTY BANK / NIFTY MIDCAP SELECT
        # option chains often list DIFFERENT strike sets for CE vs PE — the
        # exchange's strike-listing algorithm is moneyness-driven, not
        # symmetric. Passing raw calls["oi"] (say 20 rows) and puts["oi"]
        # (say 17 rows) into max_pain triggered:
        #   ValueError: operands could not be broadcast together with
        #   shapes (20,) (17,)
        # …which silently crashed the entire analyze(NIFTY) call inside
        # the legacy /scan path, leaving the operator with a /scan table
        # that contained only SENSEX (the only chain with balanced CE/PE
        # counts after strikes_around_atm filtering).
        #
        # Fix: pivot CE and PE OI onto the UNION of their strikes, zero-
        # filling whichever side is missing a strike. The three arrays
        # (strikes, call_oi, put_oi) then have equal length and max_pain's
        # per-strike inner products broadcast correctly. This is also
        # mathematically equivalent to the proper max-pain definition
        # (zero OI contributes zero pain) — no approximation introduced.
        if not calls.empty and not puts.empty:
            ce_pivot = calls.groupby("strike")["oi"].sum()
            pe_pivot = puts.groupby("strike")["oi"].sum()
            oi_pivot = pd.DataFrame(
                {"ce_oi": ce_pivot, "pe_oi": pe_pivot}
            ).fillna(0.0).sort_index()
            if len(oi_pivot) >= 2:
                mp = max_pain(
                    oi_pivot.index.values.astype(float),
                    oi_pivot["ce_oi"].values.astype(float),
                    oi_pivot["pe_oi"].values.astype(float),
                    spot_price,
                )
                metrics["max_pain"] = mp
                metrics["max_pain_distance_pct"] = round(
                    (spot_price - mp) / spot_price * 100, 2
                )

        # 3. OI-based Support/Resistance
        support_resistance = self._find_oi_support_resistance(calls, puts, spot_price)
        metrics.update(support_resistance)
        sr_signals = self._interpret_support_resistance(support_resistance, spot_price)
        signals.extend(sr_signals)

        # 4. OI Buildup vs Unwinding
        buildup_signals = self._analyze_oi_buildup(calls, puts, spot_price)
        signals.extend(buildup_signals)

        # 5. OI Change Velocity
        if len(self.oi_history) >= 2:
            velocity_signals = self._analyze_oi_velocity(spot_price)
            signals.extend(velocity_signals)

        # 6. IV Skew Analysis
        iv_signals = self._analyze_iv_skew(calls, puts, spot_price)
        signals.extend(iv_signals)

        # 7. Overall OI Sentiment
        metrics["oi_sentiment"] = self._calculate_oi_sentiment(signals)

        # 8. Institutional Commitment Ratio (Residual Flow Telemetry)
        # Ratio of Net Absolute OI Change to Volume near ATM
        atm_mask = abs(chain_df["strike"] - spot_price) < spot_price * 0.02 if (not chain_df.empty and "strike" in chain_df.columns) else None
        atm_df = chain_df[atm_mask] if (atm_mask is not None and not chain_df.empty) else pd.DataFrame()

        tot_oi_change = float(abs(atm_df["oi_change"]).sum()) if not atm_df.empty and "oi_change" in atm_df.columns else 0.0
        tot_volume = float(atm_df["volume"].sum()) if not atm_df.empty and "volume" in atm_df.columns else 0.0

        # Check for granular volume (bar-level or rolling volume) to avoid session denominator dilution
        has_bar_volume = "bar_volume" in atm_df.columns and float(atm_df["bar_volume"].sum()) > 0
        has_rolling_volume = "rolling_volume" in atm_df.columns and float(atm_df["rolling_volume"].sum()) > 0
        has_interval_volume = "interval_volume" in atm_df.columns and float(atm_df["interval_volume"].sum()) > 0

        if has_bar_volume:
            eff_volume = float(atm_df["bar_volume"].sum())
            eff_oi_change = float(abs(atm_df["delta_oi"]).sum()) if ("delta_oi" in atm_df.columns and abs(atm_df["delta_oi"]).sum() > 0) else tot_oi_change
            raw_cr = (eff_oi_change / max(eff_volume, 1.0))
            time_norm_factor = 1.0
        elif has_rolling_volume:
            eff_volume = float(atm_df["rolling_volume"].sum())
            eff_oi_change = float(abs(atm_df["delta_oi"]).sum()) if ("delta_oi" in atm_df.columns and abs(atm_df["delta_oi"]).sum() > 0) else tot_oi_change
            raw_cr = (eff_oi_change / max(eff_volume, 1.0))
            time_norm_factor = 1.0
        elif has_interval_volume:
            eff_volume = float(atm_df["interval_volume"].sum())
            eff_oi_change = float(abs(atm_df["delta_oi"]).sum()) if ("delta_oi" in atm_df.columns and abs(atm_df["delta_oi"]).sum() > 0) else tot_oi_change
            raw_cr = (eff_oi_change / max(eff_volume, 1.0))
            time_norm_factor = 1.0
        else:
            raw_cr = (tot_oi_change / max(tot_volume, 1.0)) if tot_volume > 0 else 0.0
            time_norm_factor = self._calculate_time_normalized_baseline(chain_df, timestamp)

        commitment_ratio = round(min(1.0, max(0.0, raw_cr * time_norm_factor)), 3)
        metrics["commitment_ratio"] = commitment_ratio
        metrics["raw_commitment_ratio"] = round(min(1.0, max(0.0, raw_cr)), 3)
        metrics["time_normalized_baseline"] = round(time_norm_factor, 2)

        return {"signals": signals, "metrics": metrics}

    def _calculate_time_normalized_baseline(
        self, chain_df: pd.DataFrame, timestamp: Optional[Any] = None
    ) -> float:
        """
        Calculate time-of-day normalization factor for cumulative session volume.

        In Indian markets (09:15 to 15:30 IST, 375 mins), cumulative session volume
        monotonically increases, causing afternoon institutional block flows to be
        diluted when dividing by full session volume. This normalizes afternoon
        flow against an opening reference window (45 mins).

        Returns 1.0 if timestamp is not available, outside market hours, or within opening window.
        """
        try:
            ts_val = timestamp
            if ts_val is None:
                for col in ("timestamp", "time", "datetime", "bar_time"):
                    if col in chain_df.columns and not chain_df[col].dropna().empty:
                        ts_val = chain_df[col].dropna().iloc[-1]
                        break

            if ts_val is None:
                return 1.0

            from prometheus.utils.indian_market import IST
            from datetime import datetime, time as dt_time
            hour, minute = None, None

            if isinstance(ts_val, (int, float)):
                import datetime as _dt
                dt = _dt.datetime.fromtimestamp(ts_val, tz=IST)
                hour, minute = dt.hour, dt.minute
            elif isinstance(ts_val, (pd.Timestamp, datetime)):
                if getattr(ts_val, "tzinfo", None) is not None:
                    ts_ist = ts_val.astimezone(IST)
                    hour, minute = ts_ist.hour, ts_ist.minute
                else:
                    hour, minute = ts_val.hour, ts_val.minute
            elif isinstance(ts_val, dt_time):
                hour, minute = ts_val.hour, ts_val.minute
            elif isinstance(ts_val, str):
                ts_str = ts_val.strip()
                if "T" in ts_str or " " in ts_str or "-" in ts_str:
                    try:
                        p_ts = pd.to_datetime(ts_str)
                        hour, minute = p_ts.hour, p_ts.minute
                    except Exception:
                        pass
                if hour is None and ":" in ts_str:
                    parts = ts_str.split(":")
                    hour, minute = int(parts[0]), int(parts[1])

            if hour is None or minute is None:
                return 1.0

            # Indian market session: 09:15 to 15:30 (total 375 minutes)
            market_open_mins = 9 * 60 + 15   # 555
            market_close_mins = 15 * 60 + 30 # 930
            curr_mins = hour * 60 + minute

            if curr_mins < market_open_mins or curr_mins > market_close_mins:
                return 1.0

            elapsed_mins = curr_mins - market_open_mins
            baseline_window_mins = 45.0  # Opening reference window (09:15 to 10:00)

            if elapsed_mins <= baseline_window_mins:
                return 1.0

            # Square-root dampening prevents over-amplifying low-volume late-day noise
            # while scaling afternoon institutional blocks back to comparable morning baseline
            norm_factor = math.sqrt(elapsed_mins / baseline_window_mins)
            return min(3.5, max(1.0, norm_factor))
        except Exception:
            return 1.0

    def _calculate_pcr(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
        """Calculate various PCR metrics."""
        def _to_scalar(value) -> float:
            if isinstance(value, pd.Series):
                return float(pd.to_numeric(value, errors="coerce").fillna(0).sum())
            if isinstance(value, np.ndarray):
                return float(np.nansum(value))
            try:
                return float(value)
            except Exception:
                return 0.0

        call_oi_total = _to_scalar(calls["oi"].sum())
        put_oi_total = _to_scalar(puts["oi"].sum())
        call_vol_total = _to_scalar(calls["volume"].sum())
        put_vol_total = _to_scalar(puts["volume"].sum())

        return {
            "oi": pcr_ratio(put_oi_total, call_oi_total),
            "volume": pcr_ratio(put_vol_total, call_vol_total) if call_vol_total > 0 else 0,
            "call_oi_total": int(round(call_oi_total)),
            "put_oi_total": int(round(put_oi_total)),
        }

    def _interpret_pcr(self, pcr: Dict) -> Optional[OISignal]:
        """
        Interpret PCR as contrarian indicator.

        PCR > 1.3 → Extreme put buying → Contrarian BULLISH
        PCR < 0.7 → Extreme call buying → Contrarian BEARISH
        PCR 0.8–1.2 → Neutral zone
        """
        oi_pcr = pcr["oi"]
        if oi_pcr <= 0.0:
            return None

        if oi_pcr > 1.3:
            return OISignal(
                signal_type="pcr_extreme",
                direction="bullish",
                strength=min((oi_pcr - 1.3) / 0.5, 1.0),
                details=f"PCR={oi_pcr:.2f} — extreme put OI suggests support below, contrarian bullish"
            )
        elif oi_pcr < 0.7:
            return OISignal(
                signal_type="pcr_extreme",
                direction="bearish",
                strength=min((0.7 - oi_pcr) / 0.3, 1.0),
                details=f"PCR={oi_pcr:.2f} — extreme call OI suggests overconfidence, contrarian bearish"
            )
        return None

    def _find_oi_support_resistance(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        spot: float
    ) -> Dict:
        """
        Find support/resistance from OI concentration.

        Logic: High PUT OI at a strike = support (put writers defend that level)
               High CALL OI at a strike = resistance (call writers defend that level)
        """
        # Immediate support: Highest PUT OI below spot
        puts_below = puts[puts["strike"] < spot].nlargest(3, "oi")
        support_levels = puts_below["strike"].tolist() if not puts_below.empty else []

        # Immediate resistance: Highest CALL OI above spot
        calls_above = calls[calls["strike"] > spot].nlargest(3, "oi")
        resistance_levels = calls_above["strike"].tolist() if not calls_above.empty else []

        # Strongest single levels
        strongest_support = support_levels[0] if support_levels else spot * 0.98
        strongest_resistance = resistance_levels[0] if resistance_levels else spot * 1.02

        return {
            "oi_support_levels": support_levels,
            "oi_resistance_levels": resistance_levels,
            "strongest_support": strongest_support,
            "strongest_resistance": strongest_resistance,
            "range_width": strongest_resistance - strongest_support,
        }

    def _interpret_support_resistance(
        self,
        sr: Dict,
        spot: float
    ) -> List[OISignal]:
        """Generate signals based on proximity to OI-based S/R."""
        signals = []
        support = sr["strongest_support"]
        resistance = sr["strongest_resistance"]
        range_width = sr["range_width"]

        if range_width == 0:
            return signals

        # Position within range (0 = at support, 1 = at resistance)
        position = (spot - support) / range_width if range_width > 0 else 0.5

        if position < 0.2:
            signals.append(OISignal(
                signal_type="support_proximity",
                direction="bullish",
                strength=0.7 * (1 - position / 0.2),
                strike=support,
                details=f"Spot near OI support {support}. Put OI walls below suggest bounce."
            ))
        elif position > 0.8:
            signals.append(OISignal(
                signal_type="resistance_proximity",
                direction="bearish",
                strength=0.7 * ((position - 0.8) / 0.2),
                strike=resistance,
                details=f"Spot near OI resistance {resistance}. Call OI walls above suggest rejection."
            ))

        return signals

    def _analyze_oi_buildup(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        spot: float
    ) -> List[OISignal]:
        """
        Analyze OI change to distinguish buildup vs unwinding.

        Buildup + price rise = Long buildup (bullish continuation)
        Buildup + price fall = Short buildup (bearish continuation)
        Unwinding + price rise = Short covering (weak bullish)
        Unwinding + price fall = Long unwinding (weak bearish)
        """
        signals = []

        # Look at OI change in ATM and near-ATM strikes
        atm_calls = calls[abs(calls["strike"] - spot) < spot * 0.02]
        atm_puts = puts[abs(puts["strike"] - spot) < spot * 0.02]

        total_call_oi_change = atm_calls["oi_change"].sum() if not atm_calls.empty else 0
        total_put_oi_change = atm_puts["oi_change"].sum() if not atm_puts.empty else 0
        total_call_oi = atm_calls["oi"].sum() if not atm_calls.empty else 0
        total_put_oi = atm_puts["oi"].sum() if not atm_puts.empty else 0

        # ISSUE-13 FIX: Adaptive threshold based on near-ATM total OI (3% of total or min 5,000 shares)
        # Prevents breaking on non-NIFTY symbols (SENSEX, MIDCPNIFTY, single stocks)
        call_thresh = max(5000, total_call_oi * 0.03) if total_call_oi > 0 else 50000
        put_thresh = max(5000, total_put_oi * 0.03) if total_put_oi > 0 else 50000

        # Significant call OI buildup near ATM = resistance strengthening
        if total_call_oi_change > call_thresh:
            signals.append(OISignal(
                signal_type="call_oi_buildup",
                direction="bearish",
                strength=min(total_call_oi_change / max(call_thresh * 5.0, 1.0), 0.8),
                details=f"Call OI +{total_call_oi_change:,} near ATM = sellers adding positions, bearish"
            ))

        # Significant put OI buildup near ATM = support strengthening
        if total_put_oi_change > put_thresh:
            signals.append(OISignal(
                signal_type="put_oi_buildup",
                direction="bullish",
                strength=min(total_put_oi_change / max(put_thresh * 5.0, 1.0), 0.8),
                details=f"Put OI +{total_put_oi_change:,} near ATM = put sellers defending support, bullish"
            ))

        # Call OI unwinding = resistance weakening = bullish
        if total_call_oi_change < -call_thresh:
            signals.append(OISignal(
                signal_type="call_oi_unwinding",
                direction="bullish",
                strength=min(abs(total_call_oi_change) / max(call_thresh * 5.0, 1.0), 0.6),
                details=f"Call OI {total_call_oi_change:,} near ATM = call writers covering, resistance weakening"
            ))

        # Put OI unwinding = support weakening = bearish
        if total_put_oi_change < -put_thresh:
            signals.append(OISignal(
                signal_type="put_oi_unwinding",
                direction="bearish",
                strength=min(abs(total_put_oi_change) / max(put_thresh * 5.0, 1.0), 0.6),
                details=f"Put OI {total_put_oi_change:,} near ATM = put writers covering, support weakening"
            ))

        return signals

    def _analyze_oi_velocity(self, spot: float) -> List[OISignal]:
        """
        Analyze the RATE OF CHANGE of OI — velocity matters more than level.

        Rapid OI increase = aggressive new positioning
        Rapid OI decrease = stop-loss cascade or profit-taking
        """
        signals = []
        if len(self.oi_history) < 2:
            return signals

        current = self.oi_history[-1]
        previous = self.oi_history[-2]

        # Compare total OI
        current_total_oi = current["oi"].sum()
        previous_total_oi = previous["oi"].sum()

        if previous_total_oi > 0:
            oi_change_pct = (current_total_oi - previous_total_oi) / previous_total_oi * 100

            if oi_change_pct > 5:
                signals.append(OISignal(
                    signal_type="oi_velocity_spike",
                    direction="neutral",
                    strength=min(oi_change_pct / 20, 0.8),
                    details=f"OI velocity +{oi_change_pct:.1f}% — aggressive new positioning detected"
                ))
            elif oi_change_pct < -5:
                signals.append(OISignal(
                    signal_type="oi_velocity_drop",
                    direction="neutral",
                    strength=min(abs(oi_change_pct) / 20, 0.8),
                    details=f"OI velocity {oi_change_pct:.1f}% — mass unwinding/exits detected"
                ))

        return signals

    def _analyze_iv_skew(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        spot: float
    ) -> List[OISignal]:
        """
        Analyze IV skew — difference in IV between OTM puts and OTM calls.

        High put IV vs call IV = fear/hedging demand = potential bottom
        High call IV vs put IV = speculative call buying = potential top
        """
        signals = []

        otm_puts = puts[puts["strike"] < spot * 0.97]
        otm_calls = calls[calls["strike"] > spot * 1.03]

        if otm_puts.empty or otm_calls.empty:
            return signals

        put_iv_mean = otm_puts["iv"].mean()
        call_iv_mean = otm_calls["iv"].mean()

        if put_iv_mean > 0 and call_iv_mean > 0:
            skew = put_iv_mean - call_iv_mean

            if skew > 5:
                signals.append(OISignal(
                    signal_type="iv_skew",
                    direction="bullish",
                    strength=min(skew / 15, 0.7),
                    details=f"Put IV ({put_iv_mean:.1f}) >> Call IV ({call_iv_mean:.1f}). "
                            f"Fear premium elevated — contrarian bullish."
                ))
            elif skew < -5:
                signals.append(OISignal(
                    signal_type="iv_skew",
                    direction="bearish",
                    strength=min(abs(skew) / 15, 0.7),
                    details=f"Call IV ({call_iv_mean:.1f}) >> Put IV ({put_iv_mean:.1f}). "
                            f"Speculative call buying — contrarian bearish."
                ))

        return signals

    def _calculate_oi_sentiment(self, signals: List[OISignal]) -> Dict:
        """Aggregate all OI signals into overall sentiment score."""
        if not signals:
            return {"score": 0.0, "direction": "neutral", "confidence": 0.0}

        bullish_score = sum(s.strength for s in signals if s.direction == "bullish")
        bearish_score = sum(s.strength for s in signals if s.direction == "bearish")
        total = bullish_score + bearish_score

        if total == 0:
            return {"score": 0.0, "direction": "neutral", "confidence": 0.0}

        # Net score: positive = bullish, negative = bearish
        net_score = (bullish_score - bearish_score) / max(total, 1)
        direction = "bullish" if net_score > 0.1 else "bearish" if net_score < -0.1 else "neutral"
        confidence = min(abs(net_score), 1.0)

        return {
            "score": round(net_score, 3),
            "direction": direction,
            "confidence": round(confidence, 3),
            "bullish_signals": sum(1 for s in signals if s.direction == "bullish"),
            "bearish_signals": sum(1 for s in signals if s.direction == "bearish"),
        }
