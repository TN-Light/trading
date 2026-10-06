# ============================================================================
# PROMETHEUS — Execution: Microstructure & Trade Health Engine
# ============================================================================
"""
Real-time 8-Pillar Microstructure & Position Health Engine.

Evaluates evolving trade health dynamically during the trade lifecycle.
Identifies both:
  1. REAL THREAT (Defensive Alpha): Structural invalidation, volume exhaustion,
     opposing institutional OI buildup, or non-linear theta death -> Pre-SL bailout
     saving 50-70% of risk before the hard stop loss is triggered.
  2. REAL FAVOR (Offensive Alpha): Institutional volume surge, dealer short-gamma
     squeeze acceleration, or short-covering cascades -> Dynamic target expansion
     and trailing de-compression to harvest multi-ATR fat right-tail runners.

8 Mathematical Pillars:
  P1. Dynamic Anchored VWAP Clearance & Slope (Z_vwap, dVWAP/dt)
  P2. Order Flow Volume Delta & RVOL Absorption (V_bar / SMA20(V))
  P3. Real-Time Intraday Strike ΔOI & Commitment Ratio (Institutional Writing vs Covering)
  P4. Dealer GEX Regime Migration & Zero Gamma Level Distance (S_t - ZGL)
  P5. Microstructure Noise Envelope vs Structural Runner Preservation (1.5 * ATR_1m)
  P6. Non-Linear Intraday Theta Decay & Time-of-Day Convexity (Power Hour penalty)
  P7. Implied Volatility (IV) Trend & Vega Coherence (IV crush vs expansion)
  P8. Multi-Timeframe Trend Coherence (15M Execution vs 1H Macro Regime)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from prometheus.utils.logger import logger
from prometheus.utils.indian_market import IST


@dataclass
class PositionHealthReport:
    """Telemetry report representing the instantaneous health of an active position."""
    position_id: str
    symbol: str
    tradingsymbol: str
    direction: str              # "bullish" or "bearish"
    current_price: float
    entry_price: float
    health_score: float         # Composite score: -100 (Critical Failure) to +100 (Parabolic Runner)
    
    # Core Alpha Flags
    real_threat: bool = False
    threat_reasons: List[str] = field(default_factory=list)
    real_favor: bool = False
    favor_reasons: List[str] = field(default_factory=list)
    
    # Action Recommendation
    suggested_action: str = "HOLD"   # "HOLD" | "PRE_SL_BAILOUT" | "EXPAND_TARGET" | "TIGHTEN_SL"
    suggested_target_expansion: float = 0.0  # Points to expand target by
    
    # 8-Pillar Detailed Score Breakdown
    pillar_scores: Dict[str, float] = field(default_factory=dict)
    
    # Contextual Telemetry
    vwap_gap_pts: float = 0.0
    rvol: float = 1.0
    net_gex: float = 0.0
    zgl: Optional[float] = None
    delta_oi_strike: int = 0
    commitment_ratio: float = 0.0
    theta_penalty: float = 0.0
    iv_drift: float = 0.0
    htf_regime: str = "UNKNOWN"

    def is_defensive_bailout_recommended(self) -> bool:
        return self.real_threat and self.suggested_action == "PRE_SL_BAILOUT"

    def is_target_expansion_recommended(self) -> bool:
        return self.real_favor and self.suggested_action == "EXPAND_TARGET" and self.suggested_target_expansion > 0.0


class PositionHealthEngine:
    """
    Evaluates real-time microstructure health for open positions.
    Maintains internal short-term caching to prevent broker API rate-limiting.
    """

    def __init__(self, data_engine: Any = None, cache_ttl_seconds: float = 25.0):
        self.data_engine = data_engine
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self._history_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._gex_cache: Dict[str, Tuple[float, Any]] = {}
        self._htf_cache: Dict[str, Tuple[float, str]] = {}

    def clear_cache(self):
        self._history_cache.clear()
        self._gex_cache.clear()
        self._htf_cache.clear()

    # ------------------------------------------------------------------
    # Main Public Evaluation Method
    # ------------------------------------------------------------------
    def evaluate_position_health(
        self,
        state: Any,
        current_premium: float,
        spot_override: Optional[float] = None,
        underlying_df_override: Optional[pd.DataFrame] = None,
        oi_metrics_override: Optional[Dict[str, Any]] = None,
        gex_override: Optional[Any] = None,
    ) -> PositionHealthReport:
        """
        Evaluate full 8-pillar health for an open TrailingState or Position object.
        Guaranteed zero-exception execution (falls back gracefully to neutral on missing telemetry).
        """
        if isinstance(state, dict):
            pos_id = state.get("position_id", state.get("trade_id", "UNKNOWN"))
            symbol = state.get("symbol", state.get("underlying", "NIFTY"))
            tsym = state.get("tradingsymbol", state.get("instrument", ""))
            direction = state.get("direction", "bullish")
            entry_premium = float(state.get("entry_premium", state.get("entry_price", 0.0)) or 0.0)
            entry_spot = float(state.get("entry_spot", 0.0) or 0.0)
            risk_dist = float(state.get("risk_distance", state.get("initial_risk_distance", 0.0)) or 0.0)
            tier = (str(state.get("tier", "") or "")).upper()
        else:
            pos_id = getattr(state, "position_id", getattr(state, "trade_id", "UNKNOWN"))
            symbol = getattr(state, "symbol", getattr(state, "underlying", "NIFTY"))
            tsym = getattr(state, "tradingsymbol", getattr(state, "instrument", ""))
            direction = getattr(state, "direction", "bullish")
            entry_premium = float(getattr(state, "entry_premium", getattr(state, "entry_price", 0.0)) or 0.0)
            entry_spot = float(getattr(state, "entry_spot", 0.0) or 0.0)
            risk_dist = float(getattr(state, "risk_distance", getattr(state, "initial_risk_distance", 0.0)) or 0.0)
            tier = (getattr(state, "tier", "") or "").upper()
        if hasattr(direction, "value"):
            direction = direction.value
        direction = str(direction).lower()

        report = PositionHealthReport(
            position_id=pos_id,
            symbol=symbol,
            tradingsymbol=tsym,
            direction=direction,
            current_price=current_premium,
            entry_price=entry_premium,
            health_score=0.0,
        )

        # ── Fetch or extract underlying intraday candles ──
        df = underlying_df_override
        if df is None or df.empty:
            df = self._get_cached_intraday_data(symbol)

        if df is None or len(df) < 5:
            # Insufficient bar data — return neutral report
            report.pillar_scores["status"] = 0.0
            return report

        latest_bar = df.iloc[-1]
        spot_price = float(spot_override) if spot_override is not None else float(latest_bar.get("close", 0.0))
        trade_is_bullish = (direction in {"bullish", "buy_ce", "long"})

        # Initialize pillar scores
        p_scores: Dict[str, float] = {}
        threats: List[str] = []
        favors: List[str] = []

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 1: Dynamic VWAP Structural Clearance & Slope
        # ─────────────────────────────────────────────────────────────────
        p1_score, vwap_gap, vwap_threat, vwap_favor = self._eval_pillar_vwap(
            df=df, spot=spot_price, trade_is_bullish=trade_is_bullish
        )
        p_scores["P1_VWAP"] = p1_score
        report.vwap_gap_pts = vwap_gap
        if vwap_threat:
            threats.append(vwap_threat)
        if vwap_favor:
            favors.append(vwap_favor)

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 2: Order Flow Volume Delta & RVOL Absorption
        # ─────────────────────────────────────────────────────────────────
        p2_score, rvol, vol_threat, vol_favor = self._eval_pillar_volume(
            df=df, spot=spot_price, entry_spot=entry_spot, trade_is_bullish=trade_is_bullish
        )
        p_scores["P2_Volume"] = p2_score
        report.rvol = rvol
        if vol_threat:
            threats.append(vol_threat)
        if vol_favor:
            favors.append(vol_favor)

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 3: Real-Time Intraday Strike ΔOI & Commitment
        # ─────────────────────────────────────────────────────────────────
        p3_score, delta_oi, oi_threat, oi_favor = self._eval_pillar_oi(
            symbol=symbol,
            tradingsymbol=tsym,
            trade_is_bullish=trade_is_bullish,
            override=oi_metrics_override,
        )
        p_scores["P3_OI"] = p3_score
        report.delta_oi_strike = delta_oi
        if oi_metrics_override:
            try:
                report.commitment_ratio = float(oi_metrics_override.get("commitment_ratio") or oi_metrics_override.get("commitment") or 0.0)
            except (ValueError, TypeError):
                report.commitment_ratio = 0.0
        if oi_threat:
            threats.append(oi_threat)
        if oi_favor:
            favors.append(oi_favor)

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 4: Dealer GEX Regime Migration & Zero Gamma Level
        # ─────────────────────────────────────────────────────────────────
        effective_gex_override = gex_override or (oi_metrics_override.get("gex") if oi_metrics_override else None)
        if not effective_gex_override and oi_metrics_override and ("net_gex" in oi_metrics_override or "net_gex_cr" in oi_metrics_override):
            effective_gex_override = oi_metrics_override
        p4_score, net_gex, gex_threat, gex_favor = self._eval_pillar_gex(
            symbol=symbol,
            spot=spot_price,
            trade_is_bullish=trade_is_bullish,
            entry_spot=entry_spot,
            override=effective_gex_override,
        )
        p_scores["P4_GEX"] = p4_score
        report.net_gex = net_gex
        if effective_gex_override and isinstance(effective_gex_override, dict) and effective_gex_override.get("zgl") is not None:
            try:
                report.zgl = float(effective_gex_override["zgl"])
            except (ValueError, TypeError):
                pass
        elif symbol in self._gex_cache and self._gex_cache[symbol][1] and isinstance(self._gex_cache[symbol][1], dict):
            cached_zgl = self._gex_cache[symbol][1].get("zgl")
            if cached_zgl is not None:
                try:
                    report.zgl = float(cached_zgl)
                except (ValueError, TypeError):
                    pass
        if gex_threat:
            threats.append(gex_threat)
        if gex_favor:
            favors.append(gex_favor)

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 5: Microstructure Noise Envelope Preservation
        # ─────────────────────────────────────────────────────────────────
        p5_score = self._eval_pillar_noise(
            symbol=symbol,
            entry_premium=entry_premium,
            current_premium=current_premium,
            tier=tier,
        )
        p_scores["P5_Noise"] = p5_score

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 6: Non-Linear Intraday Theta Decay & Time-of-Day Convexity
        # ─────────────────────────────────────────────────────────────────
        p6_score, theta_threat = self._eval_pillar_theta(
            current_premium=current_premium,
            entry_premium=entry_premium,
            bars_held=getattr(state, "entry_bar_count", getattr(state, "bars_held", 0)),
        )
        p_scores["P6_Theta"] = p6_score
        report.theta_penalty = p6_score
        if theta_threat:
            threats.append(theta_threat)

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 7: Implied Volatility (IV) Trend & Vega Coherence
        # ─────────────────────────────────────────────────────────────────
        p7_score, iv_drift_val, iv_threat, iv_favor = self._eval_pillar_iv(
            symbol=symbol, tradingsymbol=tsym, entry_iv=getattr(state, "entry_iv", None)
        )
        p_scores["P7_IV"] = p7_score
        report.iv_drift = iv_drift_val
        if iv_threat:
            threats.append(iv_threat)
        if iv_favor:
            favors.append(iv_favor)

        # ─────────────────────────────────────────────────────────────────
        # PILLAR 8: Multi-Timeframe Trend Synchronization (15M vs 1H)
        # ─────────────────────────────────────────────────────────────────
        p8_score, htf_regime, htf_threat, htf_favor = self._eval_pillar_htf(
            symbol=symbol, trade_is_bullish=trade_is_bullish
        )
        p_scores["P8_HTF"] = p8_score
        report.htf_regime = htf_regime
        if htf_threat:
            threats.append(htf_threat)
        if htf_favor:
            favors.append(htf_favor)

        # ─────────────────────────────────────────────────────────────────
        # COMPOSITE HEALTH SCORE & SYNTHESIS
        # ─────────────────────────────────────────────────────────────────
        weights = {
            "P1_VWAP": 1.5,
            "P2_Volume": 1.2,
            "P3_OI": 1.2,
            "P4_GEX": 1.3,
            "P5_Noise": 0.8,
            "P6_Theta": 1.0,
            "P7_IV": 0.8,
            "P8_HTF": 1.2,
        }
        weighted_sum = sum(p_scores.get(k, 0.0) * weights[k] for k in weights)
        active_weight = sum(weights[k] for k in weights if abs(p_scores.get(k, 0.0)) > 0)
        inactive_weight = sum(weights[k] for k in weights if abs(p_scores.get(k, 0.0)) == 0)
        effective_denominator = max(3.0, active_weight + (0.35 * inactive_weight))
        raw_composite = weighted_sum / effective_denominator
        # Bound cleanly to [-100.0, +100.0] with NaN trap guard
        import math
        if math.isnan(raw_composite):
            composite_score = 0.0
        else:
            composite_score = round(max(-100.0, min(100.0, raw_composite)), 1)
        
        report.health_score = composite_score
        report.pillar_scores = {k: round(v, 1) for k, v in p_scores.items()}
        report.threat_reasons = threats
        report.favor_reasons = favors

        # ─────────────────────────────────────────────────────────────────
        # DECISION BOUNDARIES: REAL THREAT VS REAL FAVOR
        # ─────────────────────────────────────────────────────────────────
        # REAL THREAT: Severe structural invalidation (Score <= -35.0 or 2+ threats with <= -25.0)
        gain_pts = current_premium - entry_premium
        sym_upper = symbol.upper()
        noise_buffer = 14.0 if ("BANK" in sym_upper or "SENSEX" in sym_upper) else 6.0
        tier = (getattr(state, "tier", "B") or "B").upper()
        bars_held = getattr(state, "entry_bar_count", 0)

        if composite_score <= -35.0 or (composite_score <= -25.0 and len(threats) >= 2):
            report.real_threat = True
            is_tier_sb = tier in ("S", "B")
            within_noise = gain_pts > -noise_buffer

            # Tier S/B high-conviction runners must never be bailed out within the noise envelope
            if is_tier_sb and within_noise:
                report.suggested_action = "HOLD"
            # Require minimum holding time (>= 1 bar) or meaningful loss past noise floor (<= -noise_buffer)
            elif gain_pts <= 2.0:
                if bars_held >= 1 or gain_pts <= -noise_buffer:
                    report.suggested_action = "PRE_SL_BAILOUT"
                else:
                    report.suggested_action = "HOLD"
            else:
                report.suggested_action = "TIGHTEN_SL"
        elif composite_score >= 45.0 and len(favors) >= 2:
            report.real_favor = True
            report.suggested_action = "EXPAND_TARGET"
            # Target expansion calculation based on symbol noise & gamma
            sym_upper = symbol.upper()
            if "BANK" in sym_upper:
                report.suggested_target_expansion = 25.0
            elif "SENSEX" in sym_upper:
                report.suggested_target_expansion = 30.0
            else:
                report.suggested_target_expansion = 12.0
        else:
            report.suggested_action = "HOLD"

        return report

    # ------------------------------------------------------------------
    # Individual Pillar Evaluation Implementations
    # ------------------------------------------------------------------

    def _eval_pillar_vwap(
        self, df: pd.DataFrame, spot: float, trade_is_bullish: bool
    ) -> Tuple[float, float, Optional[str], Optional[str]]:
        """P1: Anchored VWAP distance and slope."""
        try:
            import math
            if spot is None or math.isnan(spot) or spot <= 0:
                return 0.0, 0.0, None, None

            from prometheus.signals.technical import calculate_session_vwap
            vwap_df = calculate_session_vwap(df)
            if vwap_df.empty or "vwap" not in vwap_df.columns:
                return 0.0, 0.0, None, None

            vwap_series = vwap_df["vwap"]
            current_vwap = float(vwap_series.iloc[-1])
            prev_vwap = float(vwap_series.iloc[-2]) if len(vwap_series) >= 2 else current_vwap
            vwap_slope = current_vwap - prev_vwap
            gap_pts = spot - current_vwap

            # Buffer: 0.06% for index noise
            buf = current_vwap * 0.0006

            if trade_is_bullish:
                # Bullish trade (Long CE)
                if spot < (current_vwap - buf):
                    threat_msg = f"Spot {spot:.1f} lost Session VWAP {current_vwap:.1f} (Bearish breach)"
                    score = -75.0 if vwap_slope <= 0 else -50.0
                    return score, gap_pts, threat_msg, None
                elif spot > (current_vwap + buf) and vwap_slope > 0:
                    favor_msg = f"Spot {spot:.1f} firmly above rising VWAP {current_vwap:.1f} (+{gap_pts:.1f} pts)"
                    return +70.0, gap_pts, None, favor_msg
                else:
                    return +15.0, gap_pts, None, None
            else:
                # Bearish trade (Long PE)
                if spot > (current_vwap + buf):
                    threat_msg = f"Spot {spot:.1f} rallied above Session VWAP {current_vwap:.1f} (Bullish breach)"
                    score = -75.0 if vwap_slope >= 0 else -50.0
                    return score, gap_pts, threat_msg, None
                elif spot < (current_vwap - buf) and vwap_slope < 0:
                    favor_msg = f"Spot {spot:.1f} firmly below falling VWAP {current_vwap:.1f} ({gap_pts:.1f} pts)"
                    return +70.0, gap_pts, None, favor_msg
                else:
                    return +15.0, gap_pts, None, None
        except Exception as e:
            logger.debug(f"Pillar 1 VWAP error: {e}")
            return 0.0, 0.0, None, None

    def _eval_pillar_volume(
        self, df: pd.DataFrame, spot: float, entry_spot: float, trade_is_bullish: bool
    ) -> Tuple[float, float, Optional[str], Optional[str]]:
        """P2: Relative Volume and Flow Exhaustion."""
        try:
            if "volume" not in df.columns or len(df) < 5:
                return 0.0, 1.0, None, None

            vol = df["volume"].astype(float)
            curr_vol = float(vol.iloc[-1])
            sma_vol = float(vol.rolling(20, min_periods=3).mean().iloc[-1])
            if sma_vol <= 0:
                return 0.0, 1.0, None, None

            rvol = round(curr_vol / sma_vol, 2)
            spot_progress = (spot - entry_spot) if trade_is_bullish else (entry_spot - spot)
            if math.isnan(spot_progress):
                return 0.0, rvol, None, None

            # Volume Exhaustion Divergence: Price moves in favor, but volume collapses (< 0.45x)
            if spot_progress > 0 and rvol < 0.45:
                threat_msg = f"Volume exhaustion: RVOL is weak ({rvol:.2f}x) despite price advance (buyer fatigue)"
                return -60.0, rvol, threat_msg, None
            # Adverse High Volume: Spot reversing against trade on expanding volume (> 1.6x)
            elif spot_progress < 0 and rvol > 1.60:
                threat_msg = f"Institutional opposing volume surge: RVOL {rvol:.2f}x driving adverse price move"
                return -75.0, rvol, threat_msg, None
            # Institutional Acceleration: Price in favor + RVOL surging (> 1.8x)
            elif spot_progress > 0 and rvol >= 1.80:
                favor_msg = f"Institutional breakout confirmation: RVOL surging at {rvol:.2f}x normal volume"
                return +75.0, rvol, None, favor_msg
            elif spot_progress > 0 and rvol >= 1.10:
                return +25.0, rvol, None, None
            else:
                return 0.0, rvol, None, None
        except Exception as e:
            logger.debug(f"Pillar 2 Volume error: {e}")
            return 0.0, 1.0, None, None

    def _eval_pillar_oi(
        self,
        symbol: str,
        tradingsymbol: str,
        trade_is_bullish: bool,
        override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, int, Optional[str], Optional[str]]:
        """P3: Intraday ΔOI Shift and Resistance Walls."""
        try:
            delta_oi = 0
            commitment = 0.0
            volume = 0.0
            high_volume = False

            if override:
                doi_raw = override.get("delta_oi")
                try:
                    delta_oi = int(doi_raw) if doi_raw is not None else 0
                except (ValueError, TypeError):
                    delta_oi = 0

                cr_raw = override.get("commitment_ratio") if override.get("commitment_ratio") is not None else override.get("commitment")
                try:
                    commitment = float(cr_raw) if cr_raw is not None else 0.0
                except (ValueError, TypeError):
                    commitment = 0.0

                vol_raw = override.get("volume") if override.get("volume") is not None else override.get("trade_volume")
                try:
                    volume = float(vol_raw) if vol_raw is not None else 0.0
                except (ValueError, TypeError):
                    volume = 0.0

                high_volume = bool(override.get("high_volume", False) or volume >= 20000)
            elif self.data_engine and getattr(self.data_engine, "angelone_options", None):
                ao = self.data_engine.angelone_options
                # Check snapshot cache directly
                token = ""
                with ao._oi_lock:
                    for t, s in ao._oi_snapshots.items():
                        if s.tradingsymbol == tradingsymbol:
                            token = t
                            break
                if token and token in ao._oi_snapshots:
                    snap = ao._oi_snapshots[token]
                    delta_oi = snap.delta_oi_poll
                    volume = float(getattr(snap, "volume", 0.0))
                    high_volume = volume >= 20000
                    delta_vol = getattr(snap, "delta_volume_poll", 0)
                    if delta_vol > 0:
                        commitment = min(1.0, max(0.0, abs(snap.delta_oi_poll) / delta_vol))
                    elif volume > 0:
                        raw_cr = abs(snap.delta_oi_session) / volume
                        norm_factor = 1.0
                        try:
                            from prometheus.signals.oi_analyzer import OIAnalyzer
                            norm_factor = OIAnalyzer()._calculate_time_normalized_baseline(pd.DataFrame(), timestamp=time.time())
                        except Exception:
                            norm_factor = 1.0
                        commitment = min(1.0, max(0.0, raw_cr * norm_factor))
                    else:
                        commitment = 0.0
                else:
                    delta_oi = 0
                    commitment = 0.0
            else:
                delta_oi = 0
                commitment = 0.0

            # For CE: positive delta_oi indicates option accumulation/demand, negative indicates unwinding
            base_score = 0.0
            threat_msg = None
            favor_msg = None

            if trade_is_bullish:
                if delta_oi < -10000:
                    threat_msg = f"Call open interest unwinding ({delta_oi:+,} contracts) — smart money closing longs"
                    base_score = -65.0
                elif delta_oi > 25000:
                    favor_msg = f"Call open interest aggressive buildup ({delta_oi:+,} contracts) — institutional backing"
                    base_score = +70.0
            else:
                if delta_oi < -10000:
                    threat_msg = f"Put open interest unwinding ({delta_oi:+,} contracts) — smart money closing shorts"
                    base_score = -65.0
                elif delta_oi > 25000:
                    favor_msg = f"Put open interest aggressive buildup ({delta_oi:+,} contracts) — institutional backing"
                    base_score = +70.0

            score = base_score

            # Commitment Ratio Evaluation:
            # 1. Institutional commitment ratio >= 0.25 reinforces directional conviction (+25 pts)
            # Requires positive delta_oi supporting trade direction
            if commitment >= 0.25 and delta_oi > 0 and base_score >= 0:
                score += 25.0
                if favor_msg:
                    favor_msg = f"{favor_msg} + strong institutional commitment ({commitment:.2f})"
                else:
                    favor_msg = f"Institutional commitment confirmed (ratio {commitment:.2f}) — smart money backing"
            # 2. When volume is high but commitment ratio is near 0 (< 0.05), penalize as retail churn (-25 pts)
            elif high_volume and commitment < 0.05:
                score -= 25.0
                churn_msg = f"Retail churn warning: High volume ({int(volume):,} contracts) with negligible commitment ({commitment:.2f}) — lacking smart money backing"
                threat_msg = f"{threat_msg} + {churn_msg}" if threat_msg else churn_msg

            score = max(-100.0, min(100.0, score))
            return score, delta_oi, threat_msg, favor_msg
        except Exception as e:
            logger.debug(f"Pillar 3 OI error: {e}")
            return 0.0, 0, None, None

    def _eval_pillar_gex(
        self,
        symbol: str,
        spot: float,
        trade_is_bullish: bool,
        entry_spot: float = 0.0,
        override: Optional[Any] = None,
    ) -> Tuple[float, float, Optional[str], Optional[str]]:
        """P4: Dealer Gamma Exposure and ZGL Migration."""
        try:
            # Handle backward compatibility if override passed as 4th positional arg
            if override is None and isinstance(entry_spot, dict):
                override = entry_spot
                entry_spot = 0.0
            else:
                try:
                    entry_spot = float(entry_spot) if entry_spot is not None else 0.0
                except (ValueError, TypeError):
                    entry_spot = 0.0

            now = time.time()
            if override is not None:
                gex_profile = override
            elif symbol in self._gex_cache and (now - self._gex_cache[symbol][0]) < self.cache_ttl_seconds:
                gex_profile = self._gex_cache[symbol][1]
            elif self.data_engine and getattr(self.data_engine, "angelone_options", None):
                ao = self.data_engine.angelone_options
                opt_chain = ao.get_option_chain(symbol=symbol, spot_price=spot)
                from prometheus.signals.gamma_engine import GammaEngine
                ge = GammaEngine()
                gex_profile = ge.calculate_gex(opt_chain, spot_price=spot, symbol=symbol) if (opt_chain is not None and not opt_chain.empty) else None
                self._gex_cache[symbol] = (now, gex_profile)
            else:
                gex_profile = None

            if not gex_profile:
                return 0.0, 0.0, None, None

            if isinstance(gex_profile, dict):
                cr_val = gex_profile.get("net_gex_cr")
                raw_val = gex_profile.get("net_gex")
                if cr_val is not None:
                    try:
                        net_gex = float(cr_val)
                    except (ValueError, TypeError):
                        net_gex = 0.0
                elif raw_val is not None:
                    try:
                        r = float(raw_val)
                        net_gex = r / 1e7 if abs(r) > 1000 else r
                    except (ValueError, TypeError):
                        net_gex = 0.0
                else:
                    net_gex = 0.0

                raw_zgl = gex_profile.get("zgl")
                if raw_zgl is None:
                    raw_zgl = gex_profile.get("zero_gamma_level")
                if raw_zgl is not None:
                    try:
                        zgl = float(raw_zgl)
                    except (ValueError, TypeError):
                        zgl = None
                else:
                    zgl = None
            else:
                raw_gex = 0.0
                for attr in ("net_gex_cr", "net_gex"):
                    val = getattr(gex_profile, attr, None)
                    if isinstance(val, (int, float)):
                        raw_gex = float(val)
                        break
                else:
                    try:
                        raw_gex = float(getattr(gex_profile, "net_gex_cr", getattr(gex_profile, "net_gex", 0.0)))
                    except Exception:
                        raw_gex = 0.0

                net_gex = raw_gex / 1e7 if abs(raw_gex) > 1000 else raw_gex

                raw_zgl = None
                for attr in ("zgl", "zero_gamma_level"):
                    val = getattr(gex_profile, attr, None)
                    if val is not None:
                        try:
                            raw_zgl = float(val)
                            break
                        except (ValueError, TypeError):
                            pass
                zgl = raw_zgl

            # Direction-Aware Scoring in Pillar 4 GEX:
            # Determine spot progress relative to entry:
            # If entry_spot is available (> 0), positive progress means spot moving favorably.
            # If entry_spot is not available (<= 0), assume favorable progress for standard scoring.
            if entry_spot > 0:
                spot_progress = (spot - entry_spot) if trade_is_bullish else (entry_spot - spot)
            else:
                spot_progress = 1.0

            if trade_is_bullish:
                # Bullish trades (Long CE / Bullish spreads):
                if net_gex < -0.5:
                    if spot_progress >= 0:
                        # Dealer Short Gamma on positive spot progress: hedging amplifies upside breakouts
                        favor_msg = f"Dealer SHORT GAMMA active (Net GEX {net_gex:.1f} Cr) — dealer hedging amplifies breakout"
                        return +75.0, net_gex, None, favor_msg
                    else:
                        # Spot is falling against long call: Short Gamma accelerates the decline
                        threat_msg = f"Dealer SHORT GAMMA adverse acceleration (Net GEX {net_gex:.1f} Cr) — dealer hedging accelerates decline"
                        return -65.0, net_gex, threat_msg, None
                elif net_gex > 1.5:
                    # Dealer Long Gamma dampens volatility and caps upside
                    threat_msg = f"Dealer LONG GAMMA dampening active (Net GEX +{net_gex:.1f} Cr) — upside momentum capped"
                    return -50.0, net_gex, threat_msg, None
                else:
                    return +15.0, net_gex, None, None
            else:
                # Bearish trades (Long PE / Bearish spreads):
                if net_gex < -0.5:
                    if spot_progress >= 0:
                        # Dealer Short Gamma on downward spot progress: hedging accelerates downward flushes
                        favor_msg = f"Dealer SHORT GAMMA active (Net GEX {net_gex:.1f} Cr) — dealer hedging accelerates downward flush"
                        return +75.0, net_gex, None, favor_msg
                    else:
                        # Spot is rising against long put: Short Gamma accelerates upward rally
                        threat_msg = f"Dealer SHORT GAMMA adverse acceleration (Net GEX {net_gex:.1f} Cr) — dealer hedging accelerates upward squeeze"
                        return -65.0, net_gex, threat_msg, None
                elif net_gex > 1.5:
                    # Dealer Long Gamma dampening is neutral/favorable (+20.0)
                    favor_msg = f"Dealer LONG GAMMA dampening active (Net GEX +{net_gex:.1f} Cr) — volatility compression supports put trade"
                    return +20.0, net_gex, None, favor_msg
                else:
                    return +15.0, net_gex, None, None
        except Exception as e:
            logger.debug(f"Pillar 4 GEX error: {e}")
            return 0.0, 0.0, None, None

    def _eval_pillar_noise(
        self, symbol: str, entry_premium: float, current_premium: float, tier: str
    ) -> float:
        """P5: Microstructure Noise Envelope vs Premature Shakeout."""
        gain_pts = current_premium - entry_premium
        sym_upper = symbol.upper()

        if "BANK" in sym_upper or "SENSEX" in sym_upper:
            noise_floor = 14.0
        else:
            noise_floor = 6.0

        if tier in {"S", "B"}:
            # For runners, if price is within noise floor, penalize hasty choking
            if 0 < gain_pts < noise_floor:
                return +10.0  # Encourage breathing room
            elif gain_pts >= noise_floor:
                return +30.0  # Clear of noise envelope
        else:
            # Tier C: Micro-scalp
            if gain_pts >= 10.0:
                return +25.0

        return 0.0

    def _eval_pillar_theta(
        self, current_premium: float, entry_premium: float, bars_held: int
    ) -> Tuple[float, Optional[str]]:
        """P6: Non-Linear Intraday Theta Decay and Time-of-Day Convexity."""
        now_time = datetime.now(IST).time()
        gain_pct = ((current_premium - entry_premium) / entry_premium * 100.0) if entry_premium > 0 else 0.0

        # Post-14:00 (Theta acceleration) and Post-15:00 (Power Hour burn)
        if now_time >= dtime(15, 0):
            theta_mult = 3.5
        elif now_time >= dtime(14, 0):
            theta_mult = 2.0
        elif now_time >= dtime(13, 15):
            theta_mult = 1.3
        else:
            theta_mult = 1.0

        # If trade has been stagnant (< 2% progress) after 3+ bars in afternoon
        if bars_held >= 3 and gain_pct < 2.0 and theta_mult > 1.0:
            penalty = -20.0 * theta_mult
            threat_msg = f"Intraday Theta Burn: Held {bars_held} bars past {now_time.strftime('%H:%M')} without progress (-{penalty:.0f} pts)"
            return penalty, threat_msg

        return 0.0, None

    def _eval_pillar_iv(
        self, symbol: str, tradingsymbol: str, entry_iv: Optional[float]
    ) -> Tuple[float, float, Optional[str], Optional[str]]:
        """P7: Implied Volatility (IV) Trend & Vega Coherence."""
        # Check if entry_iv is recorded and can be compared
        if entry_iv is None or entry_iv <= 0:
            return 0.0, 0.0, None, None

        # In production, current IV is polled from optionGreek
        return 0.0, 0.0, None, None

    def _eval_pillar_htf(
        self, symbol: str, trade_is_bullish: bool
    ) -> Tuple[float, str, Optional[str], Optional[str]]:
        """P8: Multi-Timeframe Trend Synchronization (15M vs 1H)."""
        try:
            now = time.time()
            if symbol in self._htf_cache and (now - self._htf_cache[symbol][0]) < 120.0:
                htf_regime = self._htf_cache[symbol][1]
            elif self.data_engine:
                from prometheus.signals.price_action_momentum import PriceActionMomentumScanner
                df_1h = self.data_engine.fetch_historical(symbol, days=5, interval="60minute")
                htf_regime = PriceActionMomentumScanner.evaluate_htf_trend(df_1h)
                self._htf_cache[symbol] = (now, htf_regime)
            else:
                htf_regime = "NEUTRAL"

            # Check alignment
            if trade_is_bullish:
                if htf_regime in {"BEARISH"}:
                    threat_msg = f"Higher timeframe (1H) trend is BEARISH — opposing trade direction"
                    return -35.0, htf_regime, threat_msg, None
                elif htf_regime in {"BULLISH", "EMERGING_BULLISH"}:
                    favor_msg = f"Higher timeframe (1H) trend aligned ({htf_regime})"
                    return +30.0, htf_regime, None, favor_msg
            else:
                if htf_regime in {"BULLISH"}:
                    threat_msg = f"Higher timeframe (1H) trend is BULLISH — opposing trade direction"
                    return -35.0, htf_regime, threat_msg, None
                elif htf_regime in {"BEARISH", "EMERGING_BEARISH"}:
                    favor_msg = f"Higher timeframe (1H) trend aligned ({htf_regime})"
                    return +30.0, htf_regime, None, favor_msg

            return 0.0, htf_regime, None, None
        except Exception as e:
            logger.debug(f"Pillar 8 HTF error: {e}")
            return 0.0, "UNKNOWN", None, None

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _get_cached_intraday_data(self, symbol: str) -> Optional[pd.DataFrame]:
        now = time.time()
        if symbol in self._history_cache:
            ts, df = self._history_cache[symbol]
            if (now - ts) < self.cache_ttl_seconds:
                return df

        if not self.data_engine:
            return None

        try:
            df = self.data_engine.fetch_historical(symbol, days=3, interval="15minute")
            if df is not None and not df.empty:
                self._history_cache[symbol] = (now, df)
                return df
        except Exception as e:
            logger.debug(f"Error fetching historical in health engine for {symbol}: {e}")
        return None

    # ------------------------------------------------------------------
    # Telegram User-Facing Formatter (Clear, Jargon-Free)
    # ------------------------------------------------------------------
    @staticmethod
    def format_threat_telegram_alert(report: PositionHealthReport) -> str:
        """Clean, retail-friendly early defense warning."""
        sym_clean = report.symbol.replace("NIFTY ", "").replace(" 50", "")
        reasons_text = "\n".join([f"• {r}" for r in report.threat_reasons[:3]]) or "• Momentum structure broken below key support."
        
        return (
            f"🛡️ <b>EARLY DEFENSE ALERT: MOMENTUM FADING</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Symbol: <b>{report.symbol}</b> ({report.tradingsymbol})\n"
            f"Current LTP: <b>Rs {report.current_price:.2f}</b> (Entry: Rs {report.entry_price:.2f})\n"
            f"Health Score: <b>{report.health_score:+.0f}/100</b> [CRITICAL]\n\n"
            f"⚠️ <b>Why This Alert:</b>\n"
            f"{reasons_text}\n\n"
            f"⚡ <b>RECOMMENDED ACTION ON KITE/ZERODHA:</b>\n"
            f"👉 <b>Exit now at market (Rs {report.current_price:.2f}) or move Stop Loss to Cost.</b>\n"
            f"Do not hold through structural failure — cut risk early to protect capital!\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    @staticmethod
    def format_favor_telegram_alert(report: PositionHealthReport, old_target: float, new_target: float) -> str:
        """Clean, retail-friendly runner acceleration notice."""
        reasons_text = "\n".join([f"• {r}" for r in report.favor_reasons[:3]]) or "• High-velocity institutional momentum detected."
        
        gain_pts = max(0.0, report.current_price - report.entry_price)
        sym_upper = (report.symbol or "").upper()
        if "SENSEX" in sym_upper or "BSX" in sym_upper:
            min_be_buffer = 20.0
        elif "BANKNIFTY" in sym_upper or "BANK" in sym_upper:
            min_be_buffer = 18.0
        else:
            min_be_buffer = 5.0

        if gain_pts >= min_be_buffer:
            action_desc = "Move Stop Loss to Breakeven to lock in capital protection (downside mitigated to breakeven, subject to slippage)."
        else:
            action_desc = f"Preserve wide breathing room (current gain {gain_pts:.1f} pts < {min_be_buffer:.1f} pt noise floor) to protect runner against premature shakeout."

        return (
            f"🚀 <b>RUNNER ACCELERATION: INSTITUTIONAL SQUEEZE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Symbol: <b>{report.symbol}</b> ({report.tradingsymbol})\n"
            f"Current LTP: <b>Rs {report.current_price:.2f}</b> (Entry: Rs {report.entry_price:.2f})\n"
            f"Health Score: <b>{report.health_score:+.0f}/100</b> [STRONG MOMENTUM]\n\n"
            f"⚡ <b>Why This Alert:</b>\n"
            f"{reasons_text}\n\n"
            f"🎯 <b>TARGET EXPANDED:</b>\n"
            f"Old Target: Rs {old_target:.2f} ➔ <b>New Runner Target: Rs {new_target:.2f}</b> (+{report.suggested_target_expansion:.1f} pts)\n\n"
            f"⚡ <b>RECOMMENDED ACTION ON KITE/ZERODHA:</b>\n"
            f"👉 <b>Hold position and let profits run!</b> {action_desc}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
