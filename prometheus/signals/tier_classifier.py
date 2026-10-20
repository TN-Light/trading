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


def classify_signal_tier(
    signal: Dict[str, Any],
    enforce_instrument_allocation: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Classify a refined signal into one of five distinct tiers (S, A, B, C, D).

    Args:
        signal: Dictionary containing signal metadata, scores, reasons, and strategy.
        enforce_instrument_allocation: If True, enforces Team A capital policy
            (de-allocates Bank Nifty & FinNifty directional buying to Paper Only).

    Returns:
        Dict with:
            - tier: 'S' | 'A' | 'B' | 'C' | 'D'
            - tier_name: Human readable tier name
            - tier_badge: HTML badge formatted for Telegram
            - action_instruction: Plain English execution instruction
            - is_live_eligible: True if approved for live broker capital, False if paper-only/observe
            - classification_reasons: List of reasons justifying the tier assignment
            - commitment_ratio: Optional shadow telemetry metric (|ΔOI|/Volume)
    """
    if enforce_instrument_allocation is not None and isinstance(signal, dict):
        signal = dict(signal)
        signal["enforce_instrument_allocation"] = enforce_instrument_allocation
    res = _classify_signal_tier_core(signal)
    if signal and isinstance(signal, dict):
        comm_ratio = signal.get("commitment_ratio")
        if comm_ratio is not None:
            try:
                cr_val = float(comm_ratio)
                res["commitment_ratio"] = cr_val
                if cr_val > 0:
                    res["classification_reasons"].append(f"Institutional Commitment: {cr_val:.2f} (|ΔOI|/Vol)")
            except Exception:
                pass
    return res


def _classify_signal_tier_core(signal: Dict[str, Any]) -> Dict[str, Any]:
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
                tier_name="DEFINED_RISK_CREDIT_SPREAD",
                is_live_eligible=True,
                reasons=cls_reasons,
                badge="⭐ <b>[TIER A: HIGH CONVICTION SPREAD — 0-DTE]</b>",
                instruction=f"🛡️ <b>ACTION:</b> Live Trade — High Conviction{score_label} Defined Risk Spread (~Rs 35k–45k Margin)"
            )

        # TIER B (High-Conviction Multi-Day Spread):
        # Requirements: 1-3 DTE + Score >= 8.0 + OTM >= 1.5σ + Trend Aligned
        # 1-2 DTE spreads with strong conviction deserve live execution
        days_to_exp = signal.get("days_to_expiry")
        if days_to_exp is not None and 1 <= days_to_exp <= 3 and otm_sigma >= 1.5 and score >= 8.0 and trend_aligned:
            cls_reasons = [
                f"{days_to_exp}-DTE Spread (Multi-day theta decay)",
                f"{otm_sigma:.1f}σ OTM Statistical Clearance (>= 1.5σ)",
                "Session VWAP Trend Aligned",
            ]
            if oi_shielded:
                cls_reasons.append("Institutional OI Wall Protected")
            score_label = f" ({score:.1f}/10)" if score > 0 else ""
            return _build_tier_result(
                tier="B",
                tier_name="HIGH_CONVICTION_SPREAD",
                is_live_eligible=True,
                reasons=cls_reasons,
                badge=f"🌟 <b>[TIER B: HIGH CONVICTION {days_to_exp}-DTE SPREAD]</b>",
                instruction=f"🛡️ <b>ACTION:</b> Live Trade — High Conviction{score_label} Defined Risk Spread (~Rs 35k–45k Margin)"
            )

        # TIER C (Standard Credit Spread):
        # Multi-day spread (1-DTE / 2-DTE / Monthly) or standard conviction (< 9.0)
        cls_reasons = []
        if not is_0dte:
            cls_reasons.append("Non-0DTE Spread (Multi-day gamma/vega exposure)")
        if otm_sigma < 1.95:
            cls_reasons.append(f"OTM distance is {otm_sigma:.1f}σ (below 2.0σ requirement)")
        if score < 9.0:
            cls_reasons.append(f"Conviction score is {score:.1f}/10 (below 9.0)")

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
        has_htf_explicit = is_htf_bull or is_htf_bear or ("is_htf_aligned" in signal)
        is_htf_aligned = (("CE" in action and is_htf_bull) or ("PE" in action and is_htf_bear)) or bool(signal.get("is_htf_aligned", False))
        
        is_golden = bool(signal.get("is_golden_setup", False)) or "golden_setup" in strat_name

        # Team A Capital Allocation Mandate:
        # Bank Nifty and FinNifty directional option buying carry heavy spread friction and statutory ticket drag
        # (-Rs 5,760.78 net BN, -Rs 3,143.86 net FN on single lots).
        # Directional buying capital is de-allocated from BN/FN and focused exclusively on SENSEX, NIFTY 50,
        # and defined-risk credit spreads (+Rs 1,524.62 decay capture, 69% win rate).
        sym_upper = str(signal.get("symbol", "") or signal.get("underlying", "")).upper()
        enforce_alloc = bool(signal.get("enforce_instrument_allocation", False))
        if enforce_alloc and ("BANK" in sym_upper or "FIN" in sym_upper):
            return _build_tier_result(
                tier="C",
                tier_name="DEALLOCATED_ASSET_PAPER_ONLY",
                is_live_eligible=False,
                reasons=[
                    f"Team A Capital Mandate: Directional option buying de-allocated for {signal.get('symbol', 'BN/FN')} due to statutory fee drag & spread friction (-Rs 5,760 BN / -Rs 3,143 FN empirical loss). Live capital focused on SENSEX, NIFTY 50, and credit spreads.",
                    f"Confluence Edge Score: {score:.1f}/10 (Statistical Paper Tracking Only)"
                ],
                badge="📊 <b>[TIER C: DE-ALLOCATED ASSET — PAPER ONLY]</b>",
                instruction="📝 <b>ACTION:</b> Paper Trading Engine Only (BN/FN Directional Buying De-Allocated to Paper)"
            )

        # Dealer Gamma Regime Invariant (Option Buying vs Selling):
        # In Long Gamma (net_gex > +1.5 Cr), dealers supply counter-trend liquidity and dampen range.
        # Directional breakout expansion is suppressed. Option buying is capped to Tier C (Paper Only).
        net_gex_val = float(signal.get("net_gex", 0.0) or signal.get("gex", 0.0) or 0.0)
        net_gex_cr = float(
            signal.get("net_gex_cr", 0.0) or
            signal.get("net_gex_crores", 0.0) or
            (net_gex_val / 1e7 if abs(net_gex_val) >= 1e6 else net_gex_val)
        )
        gamma_reg = str(signal.get("gamma_regime", "")).upper()
        is_dealer_long_gamma = (net_gex_cr > 1.5) or (gamma_reg == "LONG_GAMMA" and net_gex_cr > 0.5)

        # Time Window Validation (60-day Angel One empirical tick distribution):
        # Window A (Morning): 09:30 - 11:30 IST (Primary ORB & momentum impulses)
        # Lunch Dead Zone: 11:30 - 13:15 IST (Directional movement collapses by ~60%; Option Buying BLOCKED)
        # Window B (Afternoon): 13:15 - 14:15 IST (High-conviction breakout & expiry continuation)
        is_morning_power_hour = True
        is_afternoon_squeeze = True
        is_valid_entry_window = True
        is_lunch_dead_zone = False

        if bar_time:
            # Morning power hour: 09:35 - 10:35 AM IST
            is_morning_power_hour = (dtime(9, 35) <= bar_time <= dtime(10, 35))
            # Afternoon squeeze window: 13:15 - 14:15 PM IST
            is_afternoon_squeeze = (dtime(13, 15) <= bar_time <= dtime(14, 15))
            # Lunch dead zone: 11:30 - 13:15 IST
            is_lunch_dead_zone = (dtime(11, 30) < bar_time < dtime(13, 15))
            # Valid entry windows (09:30-11:30 or 13:15-14:15)
            is_valid_entry_window = (
                (dtime(9, 30) <= bar_time <= dtime(11, 30)) or
                (dtime(13, 15) <= bar_time <= dtime(14, 15))
            )

        # LUNCH DEAD ZONE HARD GATE:
        # Strictly prevent option buying between 11:30 and 13:15 to eliminate theta decay cremation
        # EXCEPTION: Institutional Trend Day (ADX >= 25, Volume >= 1.5x 20-SMA, ORB Breakout,
        # HTF trend aligned, and NOT in dealer long gamma)
        is_trend_day = bool(signal.get("is_institutional_trend_day"))
        if is_lunch_dead_zone:
            if is_trend_day and (is_htf_aligned if has_htf_explicit else True) and not is_dealer_long_gamma:
                adx_disp = signal.get("adx", 25.0)
                return _build_tier_result(
                    tier="B",
                    tier_name="INSTITUTIONAL_TREND_DAY",
                    is_live_eligible=True,
                    reasons=[
                        "Trend-Day Lunch Bypass Gate Active",
                        f"Institutional Trend Confirmed (ADX={adx_disp} >= 25, Volume >= 1.5x 20-SMA)",
                        "1-Hour HTF Trend Strictly Aligned (BULLISH for CE / BEARISH for PE)",
                        f"Short/Neutral Gamma Regime ({net_gex_cr:+.1f}Cr)",
                        "ORB Breakout expansion overriding midday theta dead zone",
                        f"Confluence Edge Score: {score:.1f}/10"
                    ],
                    badge="🚀 <b>[TIER B: INSTITUTIONAL TREND DAY CONTINUATION]</b>",
                    instruction="⚡ <b>ACTION:</b> Execute Trend Continuation (ORB Trend Day Active Through Lunch)"
                )
            lunch_reasons = [
                "Lunch Dead Zone Gate (11:30-13:15 IST)",
                "15M directional range collapses by ~60%; Option Buying strictly blocked to prevent theta decay chop",
                f"Confluence Edge Score: {score:.1f}/10"
            ]
            if is_dealer_long_gamma:
                lunch_reasons.append(f"Dealer Long Gamma ({net_gex_cr:+.1f}Cr) vetoes midday expansion")
            if is_trend_day and not is_htf_aligned:
                lunch_reasons.append("1H HTF Trend not strictly aligned during lunch")
            return _build_tier_result(
                tier="C",
                tier_name="STANDARD_MOMENTUM",
                is_live_eligible=False,
                reasons=lunch_reasons,
                badge="📊 <b>[TIER C: LUNCH DEAD ZONE — PAPER ONLY]</b>",
                instruction="📝 <b>ACTION:</b> Paper Trading Engine Only (Lunch Dead Zone 11:30-13:15: Option Buying Gated)"
            )

        # TIER S ("Perfect Storm" — Elite 5-Factor Confluence):
        # Requirements:
        #   1. 15M ORB Breakout
        #   2. Session VWAP Alignment
        #   3. Volume Surge Confirmed (>=1.15x)
        #   4. Strict 1H HTF Trend Aligned (BULLISH for CE, BEARISH for PE; NEUTRAL not allowed for S)
        #   5. High Edge Score (>= 7.0)
        #   6. Morning Power Hour (09:35 - 10:35) OR Afternoon Squeeze Window (13:15 - 14:15)
        #   7. Dealer Regime NOT Long Gamma (net_gex <= 1.5 Cr)
        if has_orb and has_vwap and has_vol_surge and is_htf_aligned and score >= 7.0 and (is_morning_power_hour or is_afternoon_squeeze) and not is_dealer_long_gamma:
            window_label = "Morning Power Hour (09:35-10:35 AM)" if (bar_time and bar_time <= dtime(11, 30)) else "Afternoon Squeeze Window (13:15-14:15 PM)"
            return _build_tier_result(
                tier="S",
                tier_name="PERFECT_STORM",
                is_live_eligible=True,
                reasons=[
                    "15M ORB High/Low Decisive Breakout",
                    "Session VWAP Confluence",
                    "Volume Expansion Surge Confirmed (>=1.15x SMA10)",
                    "1-Hour HTF EMA20/50 Strict Trend Alignment",
                    f"Short/Neutral Dealer Gamma Regime ({net_gex_cr:+.1f}Cr)",
                    f"{window_label} Execution Window",
                    f"Confluence Edge Score: {score:.1f}/10"
                ],
                badge="🏆 <b>[TIER S: PERFECT STORM — FULL CONVICTION]</b>",
                instruction="🔥 <b>ACTION:</b> Live Trade — Conservative 1 Lot Execution"
            )

        # Check if 0-DTE expiry session for this option buying signal
        is_0dte_buying = False
        if signal.get("expiry") and signal.get("bar_timestamp"):
            try:
                exp_d = pd.to_datetime(signal["expiry"]).date()
                curr_d = pd.to_datetime(signal["bar_timestamp"]).date()
                is_0dte_buying = (exp_d == curr_d)
            except Exception:
                pass
        if not is_0dte_buying and bool(signal.get("is_0dte", False)):
            is_0dte_buying = True

        # TIER B ("Golden Setup" / "Regime Scalp in Long Gamma"):
        # Requirements:
        #   1. 15M ORB + Session VWAP alignment
        #   2. STRICT 1H HTF Trend Alignment (BULLISH for CE, BEARISH for PE; NEUTRAL is banned)
        #   3. Edge score >= 6.5 (Standard non-0DTE) OR >= 8.0 (0-DTE Golden Setup or Long Gamma Regime Scalp)
        #   4. Valid Trading Window: 09:30-11:30 AM OR 13:15-14:15 PM
        min_tier_b_score = 8.0 if (is_0dte_buying or is_dealer_long_gamma) else 6.5
        if (is_golden or (has_orb and has_vwap)) and is_htf_aligned and score >= min_tier_b_score and is_valid_entry_window:
            window_desc = "Morning Window (09:30-11:30)" if (bar_time and bar_time <= dtime(11, 30)) else "Afternoon Window (13:15-14:15)"
            reasons_b = [
                "15M Opening Range Breakout (with ATR Buffer)",
                "Session VWAP Clearance >= 0.10%",
                "1-Hour HTF Trend Strictly Aligned (BULLISH for CE / BEARISH for PE)",
                f"Valid Trading Window ({window_desc})",
            ]
            if is_dealer_long_gamma:
                reasons_b.append(f"Dealer Long Gamma Active ({net_gex_cr:+.1f}Cr): Compressed Scalp Boundaries Active (GEMINI.md Rule 3)")
                reasons_b.append(f"High Conviction Setup ({score:.1f}/10 >= 8.0)")
                badge_text = "🎯 <b>[TIER B: REGIME SCALP — LONG GAMMA COMPRESSED TARGET]</b>"
                instruction_text = "🎯 <b>ACTION:</b> Live Trade — Compressed Scalp Target (Dealer Long Gamma Active: Take Profit at Scalp Boundary)"
            elif is_0dte_buying:
                reasons_b.append(f"Short/Neutral Gamma Regime ({net_gex_cr:+.1f}Cr)")
                reasons_b.append(f"0-DTE Golden Setup High Conviction ({score:.1f}/10 >= 8.0)")
                badge_text = "🌟 <b>[TIER B: 0-DTE HIGH CONVICTION GOLDEN SETUP]</b>"
                instruction_text = "🎯 <b>ACTION:</b> Live Trade — Conservative 1 Lot Execution (Half-Risk Cut & Breakeven Trail at +18/+20 pts outside noise floor)"
            else:
                reasons_b.append(f"Short/Neutral Gamma Regime ({net_gex_cr:+.1f}Cr)")
                reasons_b.append(f"Confluence Edge Score: {score:.1f}/10")
                badge_text = "🌟 <b>[TIER B: GOLDEN SETUP — 1 LOT CONSERVATIVE]</b>"
                instruction_text = "🎯 <b>ACTION:</b> Live Trade — Conservative 1 Lot Execution (Half-Risk Cut & Breakeven Trail at +18/+20 pts outside noise floor)"

            return _build_tier_result(
                tier="B",
                tier_name="REGIME_SCALP_LONG_GAMMA" if is_dealer_long_gamma else "GOLDEN_SETUP",
                is_live_eligible=True,
                reasons=reasons_b,
                badge=badge_text,
                instruction=instruction_text
            )

        # TIER C (Standard Momentum / Partial Confluences):
        # Triggered when score >= 3.5 but missing decisive 1H trend, score < 6.5, or 0-DTE buying < 8.0
        if score >= 3.5:
            missing = []
            if not is_htf_aligned:
                missing.append("1H HTF Trend is NEUTRAL or Conflicting (Chop Risk)")
            if not has_orb:
                missing.append("Missing 15M ORB Breakout")
            if not has_vwap:
                missing.append("Missing VWAP Clearance")
            if is_0dte_buying and score < 8.0:
                missing.append(f"0-DTE Expiry Option Buying gated (Requires score >= 8.0 for Tier B, got {score:.1f}/10)")
            elif score < 6.5:
                missing.append(f"Score {score:.1f}/10 is below Tier B requirement (6.5+)")
            if is_dealer_long_gamma and score < 8.0:
                missing.append(f"Dealer Long Gamma Regime (Net GEX {net_gex_cr:+.1f}Cr > +1.5Cr: Requires Score >= 8.0 for Tier B Scalp, got {score:.1f}/10; routed to Paper)")
            if not is_valid_entry_window:
                if bar_time and bar_time > dtime(14, 15):
                    missing.append("Post-Cutoff Window (After 14:15 IST)")
                else:
                    missing.append("Outside Valid Entry Windows (09:30-11:30, 13:15-14:15)")
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
        reasons=[f"Insufficient Confluence (Score {score:.1f}/10 below 3.5 threshold)"],
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
