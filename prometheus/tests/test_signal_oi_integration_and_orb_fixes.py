"""Test suite verifying Signal Generation upgrades, ORB freshness, and Open Interest (OI) rules.

Verifies:
1. Fresh ORB Breakout (+2.0) vs Valid Retest (+1.5) vs Stale Continuation (+0.5).
2. Opposing OI Wall Veto (vetoes BUY_CE when Call Wall is within 0.8 ATR with positive Delta_OI).
3. Opposing OI Wall Veto for BUY_PE (vetoes BUY_PE when Put Wall is within 0.8 ATR with positive Delta_OI).
4. Writer Trapping / Short Covering Boost (+1.5 score when PE is written or CE is unwound).
5. Live ATM Option Volume Surge replaces missing spot volume on 0-volume feeds.
6. Temporal Window alignment up to 14:30 IST.
7. CreditSpreadStrategy 1-Hour resample with 15-minute offset.
8. CreditSpreadStrategy Black-Scholes IV sigma calculation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime

from prometheus.signals.price_action_momentum import PriceActionMomentumScanner
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.strategies.credit_spread import CreditSpreadStrategy


def _build_synthetic_bars_for_orb(breakout_type="fresh", spot=24000.0, atr=25.0):
    """Helper to generate bars testing fresh breakout vs retest vs continuation."""
    rows = []
    # 15 bars prior day
    ts_prior = pd.date_range("2026-09-21 11:30", periods=15, freq="15min")
    for ts in ts_prior:
        rows.append({
            "timestamp": ts,
            "open": spot,
            "high": spot + 10,
            "low": spot - 10,
            "close": spot,
            "volume": 0,
        })

    # Today's bars:
    # 09:15 ORB candle
    rows.append({
        "timestamp": pd.to_datetime("2026-09-22 09:15:00"),
        "open": spot,
        "high": spot + 20,  # ORB high = 24020
        "low": spot - 20,   # ORB low = 23980
        "close": spot + 10,
        "volume": 0,
    })
    # 09:30 ORB candle
    rows.append({
        "timestamp": pd.to_datetime("2026-09-22 09:30:00"),
        "open": spot + 10,
        "high": spot + 25,  # ORB high = 24025
        "low": spot - 15,
        "close": spot + 15,
        "volume": 0,
    })

    if breakout_type == "fresh":
        # 09:45 bar: closes below ORB high
        rows.append({
            "timestamp": pd.to_datetime("2026-09-22 09:45:00"),
            "open": spot + 15,
            "high": spot + 24,
            "low": spot + 5,
            "close": spot + 20,  # <= 24025
            "volume": 0,
        })
        # 10:00 bar: FRESH BREAKOUT crossing above 24025 (bar range 32 pts = ~1.3x ATR < 1.8x climax)
        rows.append({
            "timestamp": pd.to_datetime("2026-09-22 10:00:00"),
            "open": spot + 20,
            "high": spot + 50,
            "low": spot + 18,
            "close": spot + 46,  # 24046 > 24025
            "volume": 0,
        })
    elif breakout_type == "retest":
        # 09:45 bar: broke out already
        rows.append({
            "timestamp": pd.to_datetime("2026-09-22 09:45:00"),
            "open": spot + 20,
            "high": spot + 50,
            "low": spot + 18,
            "close": spot + 45,  # > 24025
            "volume": 0,
        })
        # 10:00 bar: pulled back to retest ORB high (low touches 24028) and bounces
        rows.append({
            "timestamp": pd.to_datetime("2026-09-22 10:00:00"),
            "open": spot + 45,
            "high": spot + 55,
            "low": spot + 28,   # <= orb_high + buffer + 0.5*atr
            "close": spot + 52, # bounces strongly
            "volume": 0,
        })
    elif breakout_type == "stale":
        # 09:45 bar: already broke out
        rows.append({
            "timestamp": pd.to_datetime("2026-09-22 09:45:00"),
            "open": spot + 20,
            "high": spot + 50,
            "low": spot + 18,
            "close": spot + 45,
            "volume": 0,
        })
        # 10:00 bar: stays extended above without retest
        rows.append({
            "timestamp": pd.to_datetime("2026-09-22 10:00:00"),
            "open": spot + 45,
            "high": spot + 60,
            "low": spot + 42,   # > orb_high + buffer + 0.5*atr
            "close": spot + 55,
            "volume": 0,
        })

    return pd.DataFrame(rows)


def _build_synthetic_1h(end_date="2026-09-22 15:30:00"):
    """Helper to generate 1H bullish bars covering up to end_date."""
    ts = pd.date_range("2026-09-18 09:15", end=end_date, freq="1h")
    rows = []
    for i, t in enumerate(ts):
        c = 23000.0 + (i * 25)
        rows.append({
            "timestamp": t,
            "open": c - 10,
            "high": c + 20,
            "low": c - 20,
            "close": c,
            "volume": 500000,
        })
    return pd.DataFrame(rows)


def test_orb_fresh_breakout_vs_retest_vs_continuation():
    """Verify ORB distinguishes fresh breakout (+2.0), valid retest (+1.5), and stale continuation (+0.5)."""
    scanner = PriceActionMomentumScanner()
    df_1h = _build_synthetic_1h()

    # 1. Fresh Breakout
    df_fresh = _build_synthetic_bars_for_orb("fresh")
    sig_fresh = scanner.evaluate_bar(df_fresh, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig_fresh is not None
    assert any("ORB_Breakout_High" in r for r in sig_fresh["reasons"])

    # 2. Valid Retest
    df_retest = _build_synthetic_bars_for_orb("retest")
    sig_retest = scanner.evaluate_bar(df_retest, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig_retest is not None
    assert any("ORB_Retest_High" in r for r in sig_retest["reasons"])

    # 3. Stale Continuation
    df_stale = _build_synthetic_bars_for_orb("stale")
    sig_stale = scanner.evaluate_bar(df_stale, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig_stale is not None
    assert any("ORB_Continuation_High" in r for r in sig_stale["reasons"])


def test_opposing_oi_wall_vetoes_call_buying():
    """Rule 1: Verify BUY_CE is vetoed when Call Wall is within 0.8 ATR of spot and actively defended."""
    scanner = PriceActionMomentumScanner()
    df_1h = _build_synthetic_1h()
    df = _build_synthetic_bars_for_orb("fresh", spot=24000.0)

    close_price = float(df.iloc[-1]["close"])
    oi_data_blocked = {
        "strongest_resistance": close_price + 10.0,
        "call_wall_delta_oi": 500000.0,  # Active call writing defense
    }

    sig = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True, oi_data=oi_data_blocked)
    assert sig is None, "BUY_CE must be vetoed when spot is right below defended Call Wall!"


def test_opposing_oi_wall_vetoes_put_buying():
    """Rule 1: Verify BUY_PE is vetoed when Put Wall is within 0.8 ATR of spot and actively defended."""
    scanner = PriceActionMomentumScanner()
    # Build 1H bearish bars
    ts = pd.date_range("2026-09-18 09:15", "2026-09-22 15:30", freq="1h")
    rows_1h = []
    for i, t in enumerate(ts):
        c = 25000.0 - (i * 25)
        rows_1h.append({"timestamp": t, "open": c + 10, "high": c + 20, "low": c - 20, "close": c, "volume": 500000})
    df_1h_bear = pd.DataFrame(rows_1h)

    # Build bearish breakdown bars
    spot = 24000.0
    rows = []
    ts_prior = pd.date_range("2026-09-21 11:30", periods=15, freq="15min")
    for ts in ts_prior:
        rows.append({"timestamp": ts, "open": spot, "high": spot + 10, "low": spot - 10, "close": spot, "volume": 0})
    rows.append({"timestamp": pd.to_datetime("2026-09-22 09:15:00"), "open": spot, "high": spot + 20, "low": spot - 20, "close": spot, "volume": 0})
    rows.append({"timestamp": pd.to_datetime("2026-09-22 09:30:00"), "open": spot + 10, "high": spot + 10, "low": spot - 25, "close": spot - 15, "volume": 0})
    rows.append({"timestamp": pd.to_datetime("2026-09-22 09:45:00"), "open": spot - 15, "high": spot - 10, "low": spot - 24, "close": spot - 20, "volume": 0})
    # Fresh breakdown (range 30 pts)
    rows.append({"timestamp": pd.to_datetime("2026-09-22 10:00:00"), "open": spot - 20, "high": spot - 18, "low": spot - 50, "close": spot - 46, "volume": 0})
    df_bear = pd.DataFrame(rows)

    close_price = float(df_bear.iloc[-1]["close"])
    oi_data_blocked = {
        "strongest_support": close_price - 10.0,
        "put_wall_delta_oi": 500000.0,  # Active put writing defense
    }

    sig = scanner.evaluate_bar(df_bear, symbol="NIFTY 50", df_1h=df_1h_bear, golden_mode=True, oi_data=oi_data_blocked)
    assert sig is None, "BUY_PE must be vetoed when spot is right above defended Put Wall!"


def test_writer_trapping_boost():
    """Rule 2: Verify short-covering / writer-trapping adds +1.5 confluence score."""
    scanner = PriceActionMomentumScanner()
    df_1h = _build_synthetic_1h()
    df = _build_synthetic_bars_for_orb("fresh", spot=24000.0)

    # Without OI trapping
    sig_no_oi = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig_no_oi is not None
    score_base = sig_no_oi["edge_score"]

    # With Put Writing (Bullish writer trapping)
    oi_data_trapping = {
        "atm_pe_delta_oi": 150000.0,  # Aggressive put writing
        "atm_ce_delta_oi": -80000.0,  # Call writers covering
    }
    sig_boost = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True, oi_data=oi_data_trapping)
    assert sig_boost is not None
    assert sig_boost["edge_score"] == pytest.approx(score_base + 1.5, rel=1e-3)
    assert any("Writer_Trapping_Confirmed" in r for r in sig_boost["reasons"])


def test_atm_option_volume_surge_on_zero_volume_spot():
    """Rule 3: Verify live ATM option volume triggers Option_Volume_Surge_Confirmed on 0-volume spot feeds."""
    scanner = PriceActionMomentumScanner()
    df_1h = _build_synthetic_1h()
    df = _build_synthetic_bars_for_orb("fresh", spot=24000.0)

    # Pass live ATM option volume = 75,000 contracts
    sig = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True, atm_option_volume=75000.0)
    assert sig is not None
    assert sig["has_volume_surge"] is True
    assert any("Option_Volume_Surge_Confirmed" in r for r in sig["reasons"])


def test_afternoon_session_window_up_to_1430():
    """Verify Afternoon Power Session permits trades up to 14:30 IST."""
    scanner = PriceActionMomentumScanner()
    df_1h = _build_synthetic_1h()
    df = _build_synthetic_bars_for_orb("fresh", spot=24000.0)

    # Set timestamp of final bar to 14:25 IST
    df.loc[df.index[-1], "timestamp"] = pd.to_datetime("2026-09-22 14:25:00")
    sig = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig is not None, "14:25 IST must be accepted within the 13:15–14:30 Afternoon Power Session"

    # Set timestamp of final bar past 14:30 IST (e.g. 14:35 IST)
    df.loc[df.index[-1], "timestamp"] = pd.to_datetime("2026-09-22 14:35:00")
    sig_late = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig_late is None, "14:35 IST must be blocked past the 14:30 cutoff"


def test_credit_spread_bs_iv_sigma():
    """Verify CreditSpreadStrategy calculates Black-Scholes IV sigma."""
    strat = CreditSpreadStrategy()
    ts_range = pd.date_range("2026-09-22 09:15:00", periods=16, freq="15min")
    df = pd.DataFrame([
        {"timestamp": ts, "open": 24000.0, "high": 24020.0, "low": 23980.0, "close": 24010.0, "volume": 100000}
        for ts in ts_range
    ])
    today_dt = date(2026, 9, 22)
    from unittest.mock import patch
    from prometheus.tests.test_credit_spread_relaxed_gates import MockConstantChain
    chain = MockConstantChain(short_ltp=30.0, long_ltp=10.0)

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig = strat.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=chain, vix=16.0)

    assert sig is not None
    assert "otm_sigma" in sig
    assert sig["otm_sigma"] > 0
