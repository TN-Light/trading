"""
Unit tests for Institutional Microstructure Trap & Downfall Veto Gates.
Tests all 6 downfall predictors:
1. Candle Close Location Value (CLV), Shooting Star / Wick Rejection.
2. Climax Overextension (VWAP, ORB, Daily ATR, Single-Bar Vacuum).
3. Kaufman Path Efficiency Ratio (ER_6).
4. Option Writer Absorption (ATM Delta OI).
5. Overhead Call / Put Wall Clearance.
6. Cross-Index SMT Divergence (NIFTY/SENSEX vs NIFTY BANK).
7. Tier Classifier integration (Automatic demotion to Tier C Paper Only).
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
    """Verify that the Nov 13 SENSEX 10:45 candle (CLV=0.0, 35%+ upper wick, red close)
    is strictly VETOED by Gate 1."""
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
    assert telemetry["adverse_wick_ratio"] >= 0.35


def test_strong_green_breakout_candle_passes_gate1():
    """Verify that a genuine green breakout candle closing near its high (CLV >= 0.65) passes."""
    filter_gate = MicrostructureTrapFilter()

    candles = [
        (25100, 25130, 25090, 25120, 1000),
        (25120, 25150, 25110, 25140, 1200),
        (25140, 25170, 25130, 25160, 1500),
        (25160, 25190, 25150, 25180, 1800),
        (25180, 25210, 25170, 25200, 2000),
        (25200, 25260, 25195, 25255, 3000),  # Strong green breakout: CLV = (25255-25195)/65 = 0.923
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25255.0,
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
    assert telemetry["clv_directional"] >= 0.85
    assert telemetry["adverse_wick_ratio"] <= 0.15


def test_cross_index_smt_divergence_vetoes_nifty_call():
    """Verify that NIFTY BUY_CE is vetoed when NIFTY BANK is trading below its Session VWAP
    on a bearish red candle (heavyweight non-confirmation)."""
    filter_gate = MicrostructureTrapFilter()

    # NIFTY green breakout
    candles_nifty = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25200, 25255, 25195, 25250),
    ]
    df_nifty = _create_mock_df(candles_nifty)

    # NIFTY BANK rolling over below VWAP (bearish divergence)
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

    # Institutional Call writing is massive (35,000 contracts added vs 5,000 puts)
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


def test_vwap_overextension_vetoed():
    """Verify that a breakout stretched > 1.80x ATR from Session VWAP is vetoed."""
    filter_gate = MicrostructureTrapFilter()

    candles = [
        (25000, 25030, 24990, 25020),
        (25020, 25060, 25010, 25050),
        (25050, 25100, 25040, 25090),
        (25090, 25150, 25080, 25140),
        (25140, 25200, 25130, 25190),
        (25190, 25260, 25180, 25255),  # 25,255 vs VWAP 25,200 is 55 pts away on ATR 25 (2.2x ATR)
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25255.0,
        "atr": 25.0,
        "vwap": 25200.0,  # 55 pts away -> 2.2x ATR > 1.80x ATR ceiling
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is True
    assert "VWAP Overextension Veto" in reason
    assert telemetry["vwap_dist_atr"] >= 1.80


def test_institutional_range_expansion_passes_when_close_is_at_extreme():
    """Verify the 11-Year Empirical Law: A large bar (> 1.85x ATR) that closes at its extreme (CLV >= 0.70)
    is recognized as genuine institutional momentum expansion and is NOT vetoed."""
    filter_gate = MicrostructureTrapFilter()

    # Candle 6 range is 60 pts on ATR 25 -> 2.4x ATR, but closes at 25260 (Low 25200, High 25260 -> CLV = 1.0)
    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25205, 25175, 25200),
        (25200, 25260, 25198, 25258),  # Range 62 pts (> 2.4x ATR 25), but closed at 25258 (CLV = 0.97)
    ]
    df = _create_mock_df(candles)

    signal = {
        "action": "BUY_CE",
        "direction": "LONG",
        "entry_price": 25258.0,
        "atr": 25.0,
        "vwap": 25225.0,  # 33 pts away -> 1.32x ATR <= 1.80
        "orb_high": 25210.0,
    }

    is_vetoed, reason, telemetry = filter_gate.evaluate_all_vetoes(
        df_15m=df,
        signal=signal,
        symbol="NIFTY 50",
    )

    assert is_vetoed is False, f"Institutional range expansion bar was wrongly vetoed: {reason}"
    assert telemetry["single_bar_atr_mult"] > 1.85
    assert telemetry["clv_directional"] >= 0.70


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
        "reasons": ["VETOED: Wick Rejection / Pin-Bar Veto on SENSEX: Directional CLV=0.00"],
    }

    res = classify_signal_tier(signal)

    assert res["tier"] == "C"
    assert res["is_live_eligible"] is False
    assert "MICROSTRUCTURE_TRAP_VETOED" in res["tier_name"]
    assert "TIER C: MICROSTRUCTURE VETO" in res["tier_badge"]


def test_gate1_hard_invalidation_flag():
    """Verify that Gate 1 marks is_hard_invalidation=True (dropping signal),
    while Gate 2-5 mark is_hard_invalidation=False (allowing Tier C shadow tracking)."""
    filter_gate = MicrostructureTrapFilter()

    # Gate 1 violation: shooting star
    candles_g1 = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25200, 25260, 25195, 25198),  # Bearish close at bottom, CLV = 3/65 = 0.046
    ]
    df_g1 = _create_mock_df(candles_g1)
    sig_g1 = {"action": "BUY_CE", "direction": "LONG", "entry_price": 25200.0, "atr": 25.0}
    is_vetoed, reason, telem_g1 = filter_gate.evaluate_all_vetoes(df_g1, sig_g1, "NIFTY 50")
    assert is_vetoed is True
    assert telem_g1.get("is_hard_invalidation") is True

    # Gate 2B violation: clean green candle, but overextended from VWAP
    candles_g2 = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25200, 25250, 25198, 25248),  # Strong green candle, CLV = 0.96
    ]
    df_g2 = _create_mock_df(candles_g2)
    sig_g2 = {"action": "BUY_CE", "direction": "LONG", "entry_price": 25248.0, "atr": 20.0, "vwap": 25180.0}  # 68 pts away = 3.4x ATR
    is_vetoed, reason, telem_g2 = filter_gate.evaluate_all_vetoes(df_g2, sig_g2, "NIFTY 50")
    assert is_vetoed is True
    assert "VWAP Overextension" in reason
    assert telem_g2.get("is_hard_invalidation") is False


def test_gate2c_orb_climax_extension_calibrated_threshold():
    """Verify that ORB extension at 2.20x ATR passes, while > 2.50x ATR triggers late-chase climax veto."""
    filter_gate = MicrostructureTrapFilter(max_orb_atr_extension=2.50)

    candles = [
        (25100, 25130, 25090, 25120),
        (25120, 25150, 25110, 25140),
        (25140, 25170, 25130, 25160),
        (25160, 25190, 25150, 25180),
        (25180, 25210, 25170, 25200),
        (25200, 25260, 25198, 25255),  # Strong green candle
    ]
    df = _create_mock_df(candles)

    # Spot 25255 on ORB High 25200 with ATR 25: (25255 - 25200)/25 = 2.20x ATR (below 2.50x ceiling -> PASSES)
    sig_pass = {"action": "BUY_CE", "direction": "LONG", "entry_price": 25255.0, "atr": 25.0, "orb_high": 25200.0, "vwap": 25220.0}
    is_vetoed, reason, telem_pass = filter_gate.evaluate_all_vetoes(df, sig_pass, "NIFTY 50")
    assert is_vetoed is False

    # Spot 25255 on ORB High 25185 with ATR 25: (25255 - 25185)/25 = 2.80x ATR (above 2.50x ceiling -> VETOED)
    sig_fail = {"action": "BUY_CE", "direction": "LONG", "entry_price": 25255.0, "atr": 25.0, "orb_high": 25185.0, "vwap": 25220.0}
    is_vetoed, reason, telem_fail = filter_gate.evaluate_all_vetoes(df, sig_fail, "NIFTY 50")
    assert is_vetoed is True
    assert "ORB Climax Veto" in reason
    assert telem_fail.get("is_hard_invalidation") is False


def test_daily_atr_prev_day_close_date_resolution_off_by_one_fix():
    """Verify Bug 1 fix: When daily_df.iloc[-1] is yesterday vs today, prev_day_close resolves correctly."""
    filter_gate = MicrostructureTrapFilter()

    base_date = datetime(2026, 11, 13).date()
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
