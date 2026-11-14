"""
Unit tests for Institutional Microstructure Trap & Downfall Veto Gates.
Tests all empirically calibrated gates (11-Year NIFTY 50 / BANKNIFTY benchmark):
1. Candle Close Location Value (CLV < 0.30), Adverse Wick (> 55%), and Opposing Close Veto (Hard Invalidation).
2. Gate 2A: Giant Bar (> 1.85x ATR) Climax vs. Ignition Separation (CLV >= 0.70 & VWAP <= 1.5x tagged as is_ignition_bar).
3. Gate 2B: Marginal ORB Poke Trap Veto (0.00x - 0.12x ATR without expansion).
4. Trend runners (> 1.80x ATR from VWAP / ORB) are PERMITTED (not vetoed).
5. Gate 2C: Daily ATR Fuel Gauge with Live Date Alignment.
6. Gate 4: Option Writer Absorption & OI Wall Clearance.
7. Gate 5: Cross-Index SMT Divergence (NIFTY/SENSEX vs NIFTY BANK).
8. Tier Classifier integration (Automatic demotion to Tier C Paper Only).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prometheus.signals.microstructure_gates import MicrostructureTrapFilter
from prometheus.signals.tier_classifier import classify_signal_tier


def _create_mock_df(candles):
    """Helper to create 15m DataFrame from candle tuples (O, H, L, C, V)."""
    base_ts = datetime(2026, 11, 13, 9, 15)
    records = []
    for i, c in enumerate(candles):
        records.append({
            "timestamp": base_ts + timedelta(minutes=15 * i),
            "open": float(c[0]),
            "high": float(c[1]),
            "low": float(c[2]),
            "close": float(c[3]),
            "volume": float(c[4]) if len(c) > 4 else 1000.0,
        })
    return pd.DataFrame(records)


def test_nov13_sensex_shooting_star_pin_bar_is_strictly_vetoed():
    """Verify that the Nov 13 SENSEX 10:45 candle (CLV=0.0, 38%+ upper wick, red close)
    is strictly HARD-INVALIDATED by Gate 1."""
    filter_gate = MicrostructureTrapFilter()

    # SENSEX Candle 55 (10:30-10:45): Open 81828.98, High 81879.51, Low 81735.76, Close 81735.76
    candles = [
        (81700, 81750, 81680, 81730, 2000),
        (81730, 81780, 81710, 81760, 2500),
        (81760, 81800, 81740, 81780, 3000),
        (81780, 81820, 81760, 81810, 2800),
        (81810, 81840, 81790, 81828, 3500),
        (81828.98, 81879.51, 81735.76, 81735.76, 5000),  # Candle 55: Shooting Star
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 81856.0,
        "orb_high": 81835.82,
        "atr": 98.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="SENSEX",
    )

    assert is_vetoed is True, "Nov 13 SENSEX Shooting Star was NOT vetoed!"
    assert "Wick Rejection" in reason or "Wrong-Color" in reason
    assert telemetry["clv_directional"] == 0.0
    assert telemetry["is_hard_invalidation"] is True


def test_strong_green_breakout_candle_passes_gate1():
    """Verify that a normal green breakout candle (e.g. CLV = 0.60) passes Gate 1 cleanly."""
    filter_gate = MicrostructureTrapFilter()

    candles = [
        (25100, 25130, 25090, 25120, 1000),
        (25120, 25150, 25110, 25140, 1200),
        (25140, 25170, 25130, 25160, 1500),
        (25160, 25190, 25150, 25180, 1800),
        (25180, 25210, 25170, 25200, 2000),
        (25200, 25260, 25195, 25245, 3000),  # Green breakout: CLV = (25245-25195)/65 = 0.769 > 0.30
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25245.0,
        "orb_high": 25200.0,
        "atr": 25.0,
        "vwap": 25220.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is False, f"Genuine strong breakout was wrongly vetoed: {reason}"
    assert telemetry["clv_directional"] >= 0.70
    assert telemetry["is_hard_invalidation"] is False


def test_cross_index_smt_divergence_vetoes_nifty_call():
    """Verify that NIFTY BUY_CE is vetoed when NIFTY BANK is trading below its Session VWAP
    on a bearish red candle (heavyweight non-confirmation)."""
    filter_gate = MicrostructureTrapFilter()

    candles_nifty = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25200, 25255, 25195, 25250),
    ]
    df_nifty = _create_mock_df(candles_nifty)

    candles_bn = [
        (61200, 61250, 61180, 61220),
        (61220, 61260, 61200, 61240),
        (61240, 61270, 61220, 61250),
        (61250, 61260, 61190, 61210),
        (61210, 61220, 61150, 61170),
        (61170, 61180, 61100, 61110),  # Red bar, dumping below VWAP
    ]
    df_bn = _create_mock_df(candles_bn)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25250.0,
        "orb_high": 25200.0,
        "atr": 25.0,
        "vwap": 25220.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df_nifty,
        signal=signal,
        symbol="NIFTY 50",
        peer_df_15m=df_bn,
        peer_symbol="NIFTY BANK",
    )

    assert is_vetoed is True, "SMT Divergence did not veto NIFTY Call!"
    assert "Cross-Index SMT Divergence" in reason
    assert telemetry.get("is_hard_invalidation") is False


def test_option_writer_absorption_vetoes_call_buy():
    """Verify that when ATM Call ΔOI is surging (> 1.35x Put ΔOI) without writer trapping,
    the Call breakout is vetoed as an option-seller absorption trap."""
    filter_gate = MicrostructureTrapFilter()

    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25200, 25250, 25195, 25245),
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25245.0,
        "atr": 25.0,
        "vwap": 25210.0,
    }

    oi_data = {
        "atm_ce_delta_oi": 35000.0,
        "atm_pe_delta_oi": 5000.0,
        "writer_trapping_bull": False,
        "writer_trapping_bear": True,
        "strongest_resistance": 25300.0,
        "strongest_support": 25000.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
        oi_data=oi_data,
    )

    assert is_vetoed is True, "Option writer absorption was not vetoed!"
    assert "Option Writer Absorption Veto" in reason


def test_vwap_overextension_permitted_when_strong_close():
    """Verify the 11-Year Empirical Law: A strong breakout stretched > 1.80x ATR from Session VWAP
    with strong close (CLV >= 0.70) is an institutional trend runner (53.25% NIFTY / 48.75%-50.4% BNF WR)
    and is PERMITTED (not vetoed)."""
    filter_gate = MicrostructureTrapFilter()

    candles = [
        (25000, 25030, 24990, 25020),
        (25020, 25060, 25010, 25050),
        (25050, 25100, 25040, 25090),
        (25090, 25150, 25080, 25140),
        (25140, 25200, 25130, 25190),
        (25190, 25260, 25180, 25255),  # 25,255 vs VWAP 25,200 is 55 pts away on ATR 25 (2.2x ATR), CLV = 75/80 = 0.938
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25255.0,
        "atr": 25.0,
        "vwap": 25200.0,  # 55 pts away -> 2.2x ATR > 1.80x ATR
        "orb_high": 25200.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is False, f"Trend runner > 1.8x ATR from VWAP with strong close was wrongly vetoed: {reason}"
    assert telemetry["vwap_dist_atr"] >= 1.80
    assert telemetry["clv_directional"] >= 0.70


def test_vwap_overextension_vetoed_when_weak_close():
    """Verify that VWAP extension > 1.80x ATR is VETOED when accompanied by a weak close (CLV < 0.70)
    as a Climax Exhaustion trap."""
    filter_gate = MicrostructureTrapFilter()

    # Candle 6: Range 30 pts on ATR 25 (1.20x ATR <= 1.85x, so passes Gate 2A)
    # High 25255, Low 25225, Open 25230, Close 25245 -> CLV = 20/30 = 0.667 (< 0.70)
    # Adverse upper wick = 10/30 = 33.3% (<= 55% -> passes Gate 1)
    candles = [
        (25000, 25030, 24990, 25020),
        (25020, 25060, 25010, 25050),
        (25050, 25100, 25040, 25090),
        (25090, 25150, 25080, 25140),
        (25140, 25200, 25130, 25190),
        (25230, 25255, 25225, 25245),
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25245.0,
        "atr": 25.0,
        "vwap": 25180.0,  # 65 pts away -> 2.60x ATR > 1.80x ATR
        "orb_high": 25200.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is True, "VWAP overextension with weak close was NOT vetoed!"
    assert "VWAP Climax Exhaustion Veto" in reason
    assert telemetry.get("is_hard_invalidation") is False


def test_gate2b_marginal_orb_poke_vetoed():
    """Verify Gate 2B: Marginal poke (0.00x - 0.15x ATR above ORB) on normal bar is VETOED,
    preventing entry into +0.01x ATR dead noise (35.0% BNF win rate)."""
    filter_gate = MicrostructureTrapFilter()

    # Normal candle range = 20 pts on ATR 25 (0.80x ATR < 1.25x)
    # ORB High = 25200. Close = 25202 -> ORB extension = 2 / 25 = 0.08x ATR (< 0.15x ATR)
    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25205, 25175, 25200),
        (25190, 25205, 25185, 25202),  # Poke by +2 pts above ORB High
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25202.0,
        "atr": 25.0,
        "orb_high": 25200.0,
        "vwap": 25195.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is True, "Marginal ORB poke was NOT vetoed!"
    assert "Marginal ORB Poke Veto" in reason
    assert telemetry.get("is_hard_invalidation") is False


def test_orb_high_extension_permitted_when_strong_close():
    """Verify that ORB extension > 2.50x ATR is PERMITTED when accompanied by strong close (CLV >= 0.70)
    as it represents the highest-WR trend runner bucket (47.6% NIFTY / 45.4% BNF)."""
    filter_gate = MicrostructureTrapFilter()

    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25200, 25275, 25198, 25270),  # Strong green bar, Close 25270 on ORB High 25200 (70/25 = 2.8x ATR), CLV = 72/77 = 0.935
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25270.0,
        "atr": 25.0,
        "orb_high": 25200.0,
        "vwap": 25230.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is False, f"ORB extension > 2.5x with strong close was wrongly vetoed: {reason}"
    assert telemetry["orb_ext_atr"] >= 2.50
    assert telemetry["clv_directional"] >= 0.70


def test_orb_high_extension_vetoed_when_weak_close():
    """Verify that ORB extension > 2.50x ATR is VETOED when accompanied by weak close (CLV < 0.70)."""
    filter_gate = MicrostructureTrapFilter()

    # Candle 6: Range 35 pts on ATR 25 (1.40x ATR <= 1.85x, passes Gate 2A)
    # Open 25240, High 25270, Low 25235, Close 25255 -> CLV = 20/35 = 0.571 (< 0.70)
    # Adverse upper wick = 15/35 = 42.9% (<= 55% -> passes Gate 1)
    # ORB High = 25180. Close = 25255 -> ORB extension = 75 / 25 = 3.00x ATR (> 2.50x)
    # VWAP = 25235 -> VWAP distance = 20 / 25 = 0.80x ATR (passes VWAP check)
    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25240, 25270, 25235, 25255),
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25255.0,
        "atr": 25.0,
        "orb_high": 25180.0,
        "vwap": 25235.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is True, "ORB extension > 2.5x with weak close was NOT vetoed!"
    assert "ORB Climax Exhaustion Veto" in reason


def test_kaufman_er_disabled_by_default_telemetry_logged():
    """Verify Gate 3: Kaufman ER_6 < 0.38 passes by default (min_efficiency_ratio = 0.0)
    because pre-breakout low efficiency is a coiling base (46.1% WR). Telemetry is still logged."""
    filter_gate = MicrostructureTrapFilter()

    # 8 zig-zag choppy candles to produce low Kaufman ER
    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25140, 25100, 25110),
        (25110, 25140, 25100, 25135),
        (25135, 25150, 25110, 25115),
        (25115, 25145, 25105, 25140),
        (25140, 25150, 25110, 25115),
        (25115, 25150, 25110, 25140),
        (25140, 25160, 25130, 25150),  # CLV = 20/30 = 0.667
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25150.0,
        "atr": 25.0,
        "vwap": 25130.0,
        "orb_high": 25140.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is False, f"Coiling base with low ER was wrongly vetoed: {reason}"
    assert "kaufman_er_6" in telemetry
    assert telemetry["kaufman_er_6"] < 0.50


def test_institutional_range_expansion_tagged_as_ignition_bar():
    """Verify that a Giant Bar (> 1.85x ATR) launching near VWAP (<= 1.5x) with strong close
    (CLV >= 0.70) is recognized as institutional ignition and tagged as is_ignition_bar = True."""
    filter_gate = MicrostructureTrapFilter()

    # Candle 6 range = 62 pts on ATR 25 -> 2.48x ATR (> 1.85x), Low 25198, High 25260, Close 25258 -> CLV = 0.967
    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25205, 25175, 25200),
        (25200, 25260, 25198, 25258),
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25258.0,
        "atr": 25.0,
        "vwap": 25230.0,  # 28 pts away -> 1.12x ATR <= 1.50x ATR
        "orb_high": 25210.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is False
    assert telemetry["single_bar_atr_mult"] > 1.85
    assert telemetry["clv_directional"] >= 0.70
    assert telemetry.get("is_ignition_bar") is True


def test_giant_bar_weak_close_is_vetoed():
    """Verify that a Giant Bar (> 1.85x ATR) with weak close (CLV < 0.70) is vetoed as an exhaustion trap."""
    filter_gate = MicrostructureTrapFilter()

    # Candle 6 range = 60 pts on ATR 25 -> 2.4x ATR, High 25260, Low 25200, Open 25210, Close 25240
    # Passes Gate 1 (Upper wick = 20/60 = 33% <= 55%, CLV = 40/60 = 0.667 >= 0.30, Green candle)
    # But fails Gate 2A: Range 2.4x ATR > 1.85x with CLV = 0.667 < 0.70
    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25205, 25175, 25200),
        (25210, 25260, 25200, 25240),  # Range 60 pts, CLV = 0.667
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25240.0,
        "atr": 25.0,
        "vwap": 25210.0,
        "orb_high": 25200.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is True
    assert "Single-Bar Exhaustion Veto" in reason
    assert telemetry.get("is_hard_invalidation") is False


def test_tier_classifier_demotes_microstructure_vetoed_signal_to_tier_c():
    """Verify that when a signal has is_microstructure_vetoed = True,
    classify_signal_tier demotes it to Tier C Paper Only."""
    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "symbol": "SENSEX",
        "instrument": "SENSEX26N1981900CE",
        "signal_score": 9.5,
        "is_golden_setup": True,
        "is_microstructure_vetoed": True,
        "reasons": ["VETOED: Marginal ORB Poke Veto on SENSEX"],
    }

    res = classify_signal_tier(signal)

    assert res["tier"] == "C"
    assert res["is_live_eligible"] is False
    assert "MICROSTRUCTURE_TRAP_VETOED" in res["tier_name"]
    assert "TIER C: MICROSTRUCTURE VETO" in res["tier_badge"]


def test_daily_atr_prev_day_close_date_resolution_off_by_one_fix():
    """Verify Bug 1 fix: When daily_df.iloc[-1] is yesterday vs today, prev_day_close resolves correctly."""
    filter_gate = MicrostructureTrapFilter()

    candles_intra = [
        (25200, 25220, 25190, 25210),
        (25210, 25230, 25200, 25220),
        (25220, 25240, 25210, 25230),
        (25230, 25250, 25220, 25240),
        (25240, 25260, 25230, 25250),
        (25250, 25270, 25240, 25265),
    ]
    df_intra = _create_mock_df(candles_intra)

    # Case A: Live market hours — daily_df contains only completed daily bars up to yesterday (2026-11-12)
    daily_records_a = [
        {"timestamp": datetime(2026, 11, 10), "open": 25000, "high": 25100, "low": 24950, "close": 25050},
        {"timestamp": datetime(2026, 11, 11), "open": 25050, "high": 25150, "low": 25000, "close": 25100},
        {"timestamp": datetime(2026, 11, 12), "open": 25100, "high": 25220, "low": 25080, "close": 25200},  # Yesterday PDC
    ]
    df_daily_a = pd.DataFrame(daily_records_a)

    sig = {"action": "BUY_CE", "direction": "LONG", "entry_price": 25265.0, "atr": 25.0, "vwap": 25240.0}
    _, _, telem_a = filter_gate.evaluate_all_vetoes(df_intra, sig, "NIFTY 50", daily_df=df_daily_a, cached_daily_atr=150.0)
    assert telem_a["prev_day_close"] == 25200.0, f"Expected 25200.0, got {telem_a.get('prev_day_close')}"

    # Case B: EOD / replay — daily_df contains today's incomplete/completed bar (2026-11-13) as iloc[-1]
    daily_records_b = daily_records_a + [
        {"timestamp": datetime(2026, 11, 13), "open": 25200, "high": 25270, "low": 25190, "close": 25265},  # Today
    ]
    df_daily_b = pd.DataFrame(daily_records_b)

    _, _, telem_b = filter_gate.evaluate_all_vetoes(df_intra, sig, "NIFTY 50", daily_df=df_daily_b, cached_daily_atr=150.0)
    assert telem_b["prev_day_close"] == 25200.0, f"Expected 25200.0 from iloc[-2], got {telem_b.get('prev_day_close')}"
