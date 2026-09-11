"""
Prometheus 5-Tier Signal Classification Module.

Classifies incoming signals into institutional tiers:
  - 🏆 TIER S: Perfect Storm (5-factor confluence, Option Buying, Live Execution)
  - ⭐ TIER A: Sure-Shot Spread (0-DTE, >=2.0σ, OI Wall Shield, Live Execution)
  - 🌟 TIER B: Golden Setup (1H Trend + ORB + VWAP, Option Buying, Live Execution)
  - 📊 TIER C: Standard / Non-Expiry (Paper Trading Engine Only)
  - 📋 TIER D: Low Confluence / Observe (Suppressed / Log Only)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, time as dtime
import pandas as pd


def classify_signal_tier(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a refined signal into one of five distinct tiers (S, A, B, C, D).

    Args:
        signal: Dictionary containing signal metadata, scores, reasons, and strategy.

    Returns:
        Dict with:
            - tier: 'S' | 'A' | 'B' | 'C' | 'D'
            - tier_name: Human readable tier name
            - tier_badge: HTML badge formatted for Telegram
            - action_instruction: Plain English execution instruction
            - is_live_eligible: True if approved for live broker capital, False if paper-only/observe
            - classification_reasons: List of reasons justifying the tier assignment
    """
    if not signal or not isinstance(signal, dict):
        return _build_tier_result("D", "OBSERVE", False, ["Empty or invalid signal"])

    action = str(signal.get("action", "HOLD")).upper()
    if action == "HOLD":
        return _build_tier_result("D", "OBSERVE", False, ["Action is HOLD"])

    strat_type = str(signal.get("strategy_type", "")).lower()
    strat_name = str(signal.get("strategy", "")).lower()
    is_spread = strat_type == "credit_spread" or "SPREAD" in action or "credit_spread" in strat_name
    
    score = float(signal.get("signal_score", 0.0) or signal.get("edge_score", 0.0) or 0.0)
    reasons: List[str] = signal.get("reasons", []) or []
    
    # Check timestamp if available
    bar_ts = signal.get("bar_timestamp")
    bar_time: Optional[dtime] = None
    if bar_ts:
        try:
            ts_dt = pd.to_datetime(bar_ts)
            bar_time = ts_dt.time()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # TRACK 1: CREDIT SPREADS (Option Selling)
    # ─────────────────────────────────────────────────────────────
    if is_spread:
        is_0dte = bool(signal.get("is_0dte", False))
        # Derive 0-DTE from expiry if present
        if not is_0dte and signal.get("expiry") and signal.get("bar_timestamp"):
            try:
                exp_d = pd.to_datetime(signal["expiry"]).date()
                curr_d = pd.to_datetime(signal["bar_timestamp"]).date()
                is_0dte = (exp_d == curr_d)
            except Exception:
                pass

        otm_sigma = float(signal.get("otm_sigma", 0.0) or 0.0)
        oi_shielded = bool(signal.get("oi_shielded", False) or signal.get("is_wall_shielded", False))
        trend_aligned = bool(signal.get("trend_aligned", True))
        is_sure_shot = bool(signal.get("is_sure_shot", False))

        # TIER A ("Sure-Shot" Credit Spread):
        # Requirements: 0-DTE (or is_sure_shot) + Strike >= 1.95σ OTM + Trend Aligned + Score >= 9.0
        if ((is_0dte or is_sure_shot) and otm_sigma >= 1.95 and (score >= 9.0 or is_sure_shot)):
            cls_reasons = [
                "0-DTE Expiry Session" if is_0dte else "High Conviction Sure-Shot Profile",
                f"{otm_sigma:.1f}σ OTM Statistical Clearance (>=2.0σ)",
                "Session VWAP Trend Aligned",
            ]
            if oi_shielded:
                cls_reasons.append("Institutional OI Wall Protected")
            score_label = f" ({score:.1f}/10)" if score > 0 else ""
            return _build_tier_result(
                tier="A",
                tier_name="SURE_SHOT_SPREAD",
                is_live_eligible=True,
                reasons=cls_reasons,
                badge="⭐ <b>[TIER A: HIGH CONVICTION SPREAD — 0-DTE]</b>",
                instruction=f"🛡️ <b>ACTION:</b> Live Trade — High Conviction{score_label} Defined Risk Spread (~Rs 35k–45k Margin)"
            )

        # TIER C (Standard Credit Spread):
        # Multi-day spread (1-DTE / 2-DTE / Monthly) or standard conviction (< 9.0)
        cls_reasons = []
        if not is_0dte:
            cls_reasons.append("Non-0DTE Spread (Multi-day gamma/vega exposure)")
        if otm_sigma < 1.95:
            cls_reasons.append(f"OTM distance is {otm_sigma:.1f}σ (< 2.0σ requirement)")
        if score < 9.0:
            cls_reasons.append(f"Conviction score is {score:.1f}/10 (< 9.0)")

        return _build_tier_result(
            tier="C",
            tier_name="STANDARD_SPREAD",
            is_live_eligible=False,
            reasons=cls_reasons or ["Standard credit spread configuration"],
            badge="📊 <b>[TIER C: STANDARD SPREAD — PAPER ONLY]</b>",
            instruction="📝 <b>ACTION:</b> Paper Trading Engine Only (Shadow Tracking)"
        )

    # ─────────────────────────────────────────────────────────────
    # TRACK 2: DIRECTIONAL OPTION BUYING (CE / PE)
    # ─────────────────────────────────────────────────────────────
    is_buying = "BUY" in action or action in ("BUY_CE", "BUY_PE")
    if is_buying:
        has_orb = any("ORB" in r for r in reasons)
        has_vwap = any("VWAP" in r for r in reasons)
        has_vol_surge = any("Volume_Surge" in r for r in reasons) or bool(signal.get("has_volume_surge", False))
        
        is_htf_bull = any("1H_Trend_Bullish" in r for r in reasons)
        is_htf_bear = any("1H_Trend_Bearish" in r for r in reasons)
        is_htf_aligned = (("CE" in action and is_htf_bull) or ("PE" in action and is_htf_bear))
        
        is_golden = bool(signal.get("is_golden_setup", False)) or "golden_setup" in strat_name
        is_morning_power_hour = True
        if bar_time:
            # 09:35 to 10:35 AM IST is prime institutional impulse window
            is_morning_power_hour = (bar_time >= dtime(9, 35) and bar_time <= dtime(10, 35))

        # TIER S ("Perfect Storm" — Elite 5-Factor Confluence):
        # Requirements:
        #   1. 15M ORB Breakout
        #   2. Session VWAP Alignment
        #   3. Volume Surge Confirmed (>=1.15x)
        #   4. Strict 1H HTF Trend Aligned (BULLISH for CE, BEARISH for PE; NEUTRAL not allowed for S)
        #   5. High Edge Score (>= 7.0)
        #   6. Morning Power Hour (09:35 - 10:35)
        if has_orb and has_vwap and has_vol_surge and is_htf_aligned and score >= 7.0 and is_morning_power_hour:
            return _build_tier_result(
                tier="S",
                tier_name="PERFECT_STORM",
                is_live_eligible=True,
                reasons=[
                    "15M ORB High/Low Decisive Breakout",
                    "Session VWAP Confluence",
                    "Volume Expansion Surge Confirmed (>=1.15x SMA10)",
                    "1-Hour HTF EMA20/50 Strict Trend Alignment",
                    "Morning Power Hour Execution Window (09:35-10:35 AM)",
                    f"Confluence Edge Score: {score:.1f}/10"
                ],
                badge="🏆 <b>[TIER S: PERFECT STORM — FULL CONVICTION]</b>",
                instruction="🔥 <b>ACTION:</b> Live Trade — Scaled Size (Max 2 Lots / 10% Capital Risk Cap)"
            )

        # TIER B ("Golden Setup" — Primary High-Probability Option Buying):
        # Requirements:
        #   1. Golden setup flag OR (ORB + VWAP alignment)
        #   2. Edge score >= 4.0
        #   3. 1H Trend non-conflicting (BULLISH, BEARISH, or NEUTRAL)
        #   4. Time window: before 11:45 AM (or before 13:00)
        is_morning_window = True
        if bar_time:
            is_morning_window = (bar_time <= dtime(11, 45))

        if (is_golden or (has_orb and has_vwap)) and score >= 4.0 and is_morning_window:
            return _build_tier_result(
                tier="B",
                tier_name="GOLDEN_SETUP",
                is_live_eligible=True,
                reasons=[
                    "15M Opening Range Breakout (with ATR Buffer)",
                    "Session VWAP Clearance >= 0.10%",
                    "1-Hour HTF Trend Non-Conflicting (BULLISH / BEARISH / NEUTRAL)",
                    f"Confluence Edge Score: {score:.1f}/10"
                ],
                badge="🌟 <b>[TIER B: GOLDEN SETUP — 1 LOT CONSERVATIVE]</b>",
                instruction="🎯 <b>ACTION:</b> Live Trade — Conservative 1 Lot Execution (Breakeven Trail at +10 pts)"
            )

        # TIER C (Standard Momentum / Partial Confluences):
        # Triggered when 2-3 factors are present but missing decisive ORB or 1H confirmation
        if score >= 3.5:
            missing = []
            if not has_orb:
                missing.append("Missing 15M ORB Breakout")
            if not has_vwap:
                missing.append("Missing VWAP Clearance")
            if not is_morning_window:
                missing.append("Outside Morning Window (After 11:45 AM)")
            return _build_tier_result(
                tier="C",
                tier_name="STANDARD_MOMENTUM",
                is_live_eligible=False,
                reasons=[f"Partial Confluence (Score {score:.1f}/10)"] + missing,
                badge="📊 <b>[TIER C: STANDARD SIGNAL — PAPER ONLY]</b>",
                instruction="📝 <b>ACTION:</b> Paper Trading Engine Only (Tracking Statistical Expectancy)"
            )

    # ─────────────────────────────────────────────────────────────
    # TRACK 3: WEAK / FILTERED SETUPS
    # ─────────────────────────────────────────────────────────────
    return _build_tier_result(
        tier="D",
        tier_name="OBSERVE_ONLY",
        is_live_eligible=False,
        reasons=[f"Insufficient Confluence (Score {score:.1f}/10 < 3.5 threshold)"],
        badge="📋 <b>[TIER D: LOW CONFLUENCE — OBSERVE]</b>",
        instruction="🚫 <b>ACTION:</b> Filtered / Data Collection Only (Do Not Execute)"
    )


def _build_tier_result(
    tier: str,
    tier_name: str,
    is_live_eligible: bool,
    reasons: List[str],
    badge: str = "",
    instruction: str = ""
) -> Dict[str, Any]:
    """Helper to structure consistent classification output."""
    if not badge:
        badge = f"[{tier}] {tier_name}"
    if not instruction:
        instruction = "LIVE TRADE" if is_live_eligible else "PAPER ONLY"

    return {
        "tier": tier,
        "tier_name": tier_name,
        "tier_badge": badge,
        "action_instruction": instruction,
        "is_live_eligible": is_live_eligible,
        "classification_reasons": reasons,
    }
