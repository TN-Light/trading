"""Comprehensive Unit Tests and Mathematical/Empirical Proofs for
Net GEX, ZGL, and Commitment Ratio Upgrades.

Covers:
1. Fix Live Pillar 4 GEX Execution Call (passing symbol=symbol to avoid TypeError)
2. Direction-Aware Scoring in Pillar 4 GEX (Bullish vs Bearish in Short/Long Gamma)
3. Wire Commitment Ratio into Pillar 3 (conviction reinforcement vs retail churn penalty)
4. Correct Dimensional Rupee Scaling & Lot Multiplier in GEX (Q * Gamma * S^2 * 0.01)
5. Fix ZGL Solver Fallback & Solver Accuracy (unipolar fallback None, has_zgl flag, ATM precision)
6. Address Time-of-Day Denominator Dilution in Commitment Ratio (bar volume & time baseline)
"""

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from prometheus.signals.gamma_engine import GammaEngine, calculate_black_scholes_gamma
from prometheus.signals.oi_analyzer import OIAnalyzer
from prometheus.execution.position_health import PositionHealthEngine, PositionHealthReport


# ============================================================================
# Upgrade 1: Fix Live Pillar 4 GEX Execution Call
# ============================================================================

def test_pillar4_live_execution_call_passes_symbol():
    """Verify _eval_pillar_gex passes symbol to ge.calculate_gex without raising TypeError."""
    engine = PositionHealthEngine()
    
    mock_data = MagicMock()
    mock_ao = MagicMock()
    
    # Real option chain DataFrame
    chain_df = pd.DataFrame([
        {"strike_price": 25000, "option_type": "CE", "open_interest": 100000, "iv": 0.15},
        {"strike_price": 25000, "option_type": "PE", "open_interest": 100000, "iv": 0.15},
    ])
    mock_ao.get_option_chain.return_value = chain_df
    mock_data.angelone_options = mock_ao
    engine.data_engine = mock_data

    # Call _eval_pillar_gex with symbol="NIFTY 50"
    score, net_gex, threat, favor = engine._eval_pillar_gex(
        symbol="NIFTY 50",
        spot=25000.0,
        trade_is_bullish=True,
        entry_spot=25000.0,
    )
    
    # Must NOT fail with TypeError, and option chain must have been queried
    mock_ao.get_option_chain.assert_called_once_with(symbol="NIFTY 50", spot_price=25000.0)
    # The GEX cache must now contain the computed profile
    assert "NIFTY 50" in engine._gex_cache
    assert engine._gex_cache["NIFTY 50"][1] is not None
    assert "net_gex" in engine._gex_cache["NIFTY 50"][1]


# ============================================================================
# Upgrade 2: Direction-Aware Scoring in Pillar 4 GEX
# ============================================================================

def test_pillar4_direction_aware_bullish_trade():
    """Verify direction-aware scoring for Bullish (Long CE) trades."""
    engine = PositionHealthEngine()

    # Case A: Bullish trade in Short Gamma with favorable spot progress (spot > entry_spot)
    # Dealer hedging amplifies upside breakouts -> Favorable (+75)
    gex_short = {"net_gex": -2.0, "net_gex_cr": -2.0, "zgl": 24900.0, "gamma_regime": "SHORT_GAMMA"}
    score, gex_val, threat, favor = engine._eval_pillar_gex(
        symbol="NIFTY 50", spot=25100.0, trade_is_bullish=True, entry_spot=25000.0, override=gex_short
    )
    assert score == +75.0
    assert threat is None
    assert favor is not None
    assert "amplifies breakout" in favor

    # Case B: Bullish trade in Short Gamma with adverse spot progress (spot < entry_spot)
    # Spot falling against long call -> Short Gamma accelerates decline -> Penalty (-65)
    score_adv, gex_val, threat_adv, favor_adv = engine._eval_pillar_gex(
        symbol="NIFTY 50", spot=24900.0, trade_is_bullish=True, entry_spot=25000.0, override=gex_short
    )
    assert score_adv == -65.0
    assert threat_adv is not None
    assert "accelerates decline" in threat_adv
    assert favor_adv is None

    # Case C: Bullish trade in Long Gamma (net_gex > 1.5)
    # Volatility dampening caps upside -> Penalty (-50)
    gex_long = {"net_gex": 3.0, "net_gex_cr": 3.0, "zgl": 24800.0, "gamma_regime": "LONG_GAMMA"}
    score_lg, gex_val, threat_lg, favor_lg = engine._eval_pillar_gex(
        symbol="NIFTY 50", spot=25100.0, trade_is_bullish=True, entry_spot=25000.0, override=gex_long
    )
    assert score_lg == -50.0
    assert threat_lg is not None
    assert "capped" in threat_lg


def test_pillar4_direction_aware_bearish_trade():
    """Verify direction-aware scoring for Bearish (Long PE) trades."""
    engine = PositionHealthEngine()

    # Case A: Bearish trade in Short Gamma with downward spot progress (spot < entry_spot)
    # Spot flushing down -> Short Gamma accelerates downward flush -> Favorable (+75)
    gex_short = {"net_gex": -2.5, "net_gex_cr": -2.5, "zgl": 25100.0, "gamma_regime": "SHORT_GAMMA"}
    score, gex_val, threat, favor = engine._eval_pillar_gex(
        symbol="NIFTY 50", spot=24900.0, trade_is_bullish=False, entry_spot=25000.0, override=gex_short
    )
    assert score == +75.0
    assert threat is None
    assert favor is not None
    assert "downward flush" in favor

    # Case B: Bearish trade in Short Gamma with upward spot progress (spot > entry_spot)
    # Spot rising against put -> Short Gamma accelerates squeeze against put -> Penalty (-65)
    score_adv, gex_val, threat_adv, favor_adv = engine._eval_pillar_gex(
        symbol="NIFTY 50", spot=25100.0, trade_is_bullish=False, entry_spot=25000.0, override=gex_short
    )
    assert score_adv == -65.0
    assert threat_adv is not None
    assert "upward squeeze" in threat_adv
    assert favor_adv is None

    # Case C: Bearish trade in Long Gamma (net_gex > 1.5)
    # Volatility compression is neutral/favorable for put holding (+20)
    gex_long = {"net_gex": 2.5, "net_gex_cr": 2.5, "zgl": 24800.0, "gamma_regime": "LONG_GAMMA"}
    score_lg, gex_val, threat_lg, favor_lg = engine._eval_pillar_gex(
        symbol="NIFTY 50", spot=24900.0, trade_is_bullish=False, entry_spot=25000.0, override=gex_long
    )
    assert score_lg == +20.0
    assert threat_lg is None
    assert favor_lg is not None
    assert "volatility compression supports put trade" in favor_lg


# ============================================================================
# Upgrade 3: Wire Commitment Ratio into Pillar 3
# ============================================================================

def test_pillar3_commitment_ratio_reinforcement_and_churn_penalty():
    """Verify Pillar 3 rewards institutional commitment (>= 0.25) and penalizes retail churn (< 0.05)."""
    engine = PositionHealthEngine()

    # Case A: Institutional commitment ratio >= 0.25 reinforces conviction (+25 pts)
    # Base delta_oi is moderate (no extreme buildup threshold), but commitment = 0.35
    score, doi, threat, favor = engine._eval_pillar_oi(
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY55000CE",
        trade_is_bullish=True,
        override={"delta_oi": 5000, "commitment_ratio": 0.35},
    )
    assert score == +25.0
    assert favor is not None
    assert "Institutional commitment confirmed" in favor
    assert "0.35" in favor

    # Case B: High volume with commitment ratio < 0.05 -> Retail Churn penalty (-25 pts)
    score_churn, doi_c, threat_c, favor_c = engine._eval_pillar_oi(
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY55000CE",
        trade_is_bullish=True,
        override={"delta_oi": 500, "commitment_ratio": 0.02, "volume": 50000},
    )
    assert score_churn == -25.0
    assert threat_c is not None
    assert "Retail churn warning" in threat_c
    assert "50,000 contracts" in threat_c

    # Case C: Aggressive buildup (+70) PLUS institutional commitment (+25) -> Capped at +95
    score_combo, doi_cb, threat_cb, favor_cb = engine._eval_pillar_oi(
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY55000CE",
        trade_is_bullish=True,
        override={"delta_oi": 30000, "commitment_ratio": 0.40},
    )
    assert score_combo == +95.0
    assert favor_cb is not None
    assert "aggressive buildup" in favor_cb
    assert "strong institutional commitment (0.40)" in favor_cb


# ============================================================================
# Upgrade 4: Correct Dimensional Rupee Scaling & Lot Multiplier in GEX
# ============================================================================

def test_gamma_engine_rupee_notional_formula():
    """Empirical mathematical proof of GEX_INR = Q * Gamma * S^2 * 0.01 without lot_size multiplier."""
    engine = GammaEngine(risk_free_rate=0.07)
    spot = 25000.0
    strike = 25000.0
    dte = 2.0
    iv = 0.15
    q_shares = 100000  # 100k open interest in underlying shares

    # Theoretical Black-Scholes Gamma
    gamma = calculate_black_scholes_gamma(spot=spot, strike=strike, dte=dte, sigma=iv, r=0.07)
    assert gamma > 0

    # Theoretical Rupee GEX per 1% move:
    # 1% spot move = 0.01 * S = 250 INR
    # Delta change in shares = Q * Gamma * 250
    # Rupee notional to hedge = (Q * Gamma * 250) * S = Q * Gamma * S^2 * 0.01
    expected_rupee_gex = q_shares * gamma * (spot ** 2) * 0.01
    expected_gex_cr = round(expected_rupee_gex / 1e7, 2)

    chain = pd.DataFrame([
        {"strike_price": strike, "option_type": "CE", "open_interest": q_shares, "iv": iv}
    ])
    res = engine.calculate_gex(chain, spot_price=spot, symbol="NIFTY 50", dte=dte)

    # Net GEX in Crores must match exact analytical expectation
    assert math.isclose(res["net_gex_cr"], expected_gex_cr, rel_tol=1e-3)
    assert math.isclose(res["net_gex"], expected_rupee_gex, rel_tol=1e-3)


def test_gamma_engine_scale_consistency_across_indices():
    """Verify dimensional scale consistency across NIFTY 50, BANK NIFTY, and SENSEX."""
    engine = GammaEngine()

    indices = [
        ("NIFTY 50", 25000.0, 25000.0),
        ("NIFTY BANK", 50000.0, 50000.0),
        ("SENSEX", 80000.0, 80000.0),
    ]

    for sym, spot, strike in indices:
        chain = pd.DataFrame([
            {"strike_price": strike, "option_type": "CE", "open_interest": 50000, "iv": 0.15},
            {"strike_price": strike, "option_type": "PE", "open_interest": 20000, "iv": 0.15},
        ])
        res = engine.calculate_gex(chain, spot_price=spot, symbol=sym, dte=1.0)
        assert res["call_gex_cr"] > 0
        assert res["put_gex_cr"] < 0
        assert res["net_gex_cr"] > 0
        # GEX must be a reasonable number of Crores (not inflated by extra lot size)
        assert 1.0 < res["net_gex_cr"] < 5000.0


# ============================================================================
# Upgrade 5: Fix ZGL Solver Fallback & Solver Accuracy
# ============================================================================

def test_zgl_solver_unipolar_returns_none_and_has_zgl_false():
    """Verify that unipolar chains do NOT falsely report spot_price as ZGL."""
    engine = GammaEngine()
    spot = 25000.0

    # Chain with ONLY Calls: Net GEX is positive everywhere, no zero crossing exists
    call_only_chain = pd.DataFrame([
        {"strike_price": 24800, "option_type": "CE", "open_interest": 100000, "iv": 0.15},
        {"strike_price": 25000, "option_type": "CE", "open_interest": 200000, "iv": 0.15},
        {"strike_price": 25200, "option_type": "CE", "open_interest": 150000, "iv": 0.15},
    ])
    res = engine.calculate_gex(call_only_chain, spot_price=spot, symbol="NIFTY 50", dte=2.0)

    assert res["has_zgl"] is False
    assert res["zgl"] is None
    assert res["gamma_regime"] == "LONG_GAMMA"


def test_zgl_solver_bipolar_determines_accurate_crossing():
    """Verify high-resolution bisection finds the exact strike where Net GEX flips zero."""
    engine = GammaEngine()
    spot = 25000.0

    # Puts heavy below (negative gamma), Calls heavy above (positive gamma)
    chain = pd.DataFrame([
        {"strike_price": 24500, "option_type": "PE", "open_interest": 300000, "iv": 0.15},
        {"strike_price": 24800, "option_type": "PE", "open_interest": 200000, "iv": 0.15},
        {"strike_price": 25000, "option_type": "CE", "open_interest": 50000, "iv": 0.15},
        {"strike_price": 25200, "option_type": "CE", "open_interest": 250000, "iv": 0.15},
        {"strike_price": 25500, "option_type": "CE", "open_interest": 350000, "iv": 0.15},
    ])
    res = engine.calculate_gex(chain, spot_price=spot, symbol="NIFTY 50", dte=1.0)

    assert res["has_zgl"] is True
    assert res["zgl"] is not None
    # ZGL must be a plausible strike within the scan range
    assert 24500 < res["zgl"] < 25500


def test_empty_chain_returns_has_zgl_false_and_none():
    """Verify empty DataFrame returns zgl=None and has_zgl=False."""
    engine = GammaEngine()
    res = engine.calculate_gex(pd.DataFrame(), spot_price=25000.0, symbol="NIFTY 50")
    assert res["zgl"] is None
    assert res["has_zgl"] is False
    assert res["net_gex"] == 0.0


# ============================================================================
# Upgrade 6: Address Time-of-Day Denominator Dilution in Commitment Ratio
# ============================================================================

def test_commitment_ratio_uses_bar_volume_when_available():
    """Verify that bar_volume or rolling_volume bypasses cumulative volume dilution."""
    analyzer = OIAnalyzer()

    # Suppose cumulative volume has reached 1,000,000, but current bar volume is only 20,000
    # Institutional ΔOI in this bar is 10,000
    chain_df = pd.DataFrame([
        {
            "option_type": "CE",
            "strike": 25000.0,
            "oi": 150000,
            "oi_change": 10000,
            "delta_oi": 10000,
            "volume": 1000000,         # Cumulative session volume (would dilute to 0.01)
            "bar_volume": 20000,        # Recent bar volume
        }
    ])
    res = analyzer.analyze(chain_df, spot_price=25000.0)
    metrics = res["metrics"]

    # Ratio should be 10000 / 20000 = 0.500 (NOT 10000 / 1000000 = 0.010)
    assert metrics["commitment_ratio"] == 0.500
    assert metrics["time_normalized_baseline"] == 1.0


def test_commitment_ratio_time_of_day_normalization():
    """Verify afternoon time-of-day normalization prevents institutional block dilution."""
    analyzer = OIAnalyzer()

    # Session at 14:00 (2:00 PM IST)
    # Cumulative volume is 500,000. ΔOI is 40,000.
    # Raw ratio = 40,000 / 500,000 = 0.080 (which would otherwise look like retail churn / no conviction)
    chain_df = pd.DataFrame([
        {
            "option_type": "CE",
            "strike": 25000.0,
            "oi": 150000,
            "oi_change": 40000,
            "volume": 500000,
            "timestamp": "2026-10-06 14:00:00",
        }
    ])
    res = analyzer.analyze(chain_df, spot_price=25000.0)
    metrics = res["metrics"]

    # Baseline factor must be > 1.0 at 14:00 (approx sqrt(285/45) ~= 2.52)
    assert metrics["time_normalized_baseline"] > 2.0
    assert metrics["raw_commitment_ratio"] == 0.080
    # Normalized commitment ratio elevates flow to institutional conviction territory (>= 0.20)
    assert metrics["commitment_ratio"] >= 0.20
    assert metrics["commitment_ratio"] > metrics["raw_commitment_ratio"]


def test_commitment_ratio_retail_churn_still_penalized_in_afternoon():
    """Verify that pure retail churn (< 0.05) remains recognized as churn even with time normalization."""
    analyzer = OIAnalyzer()

    # At 14:00 IST, cumulative volume is 1,000,000, but ΔOI is only 5,000 (0.005 raw ratio)
    chain_df = pd.DataFrame([
        {
            "option_type": "CE",
            "strike": 25000.0,
            "oi": 150000,
            "oi_change": 5000,
            "volume": 1000000,
            "timestamp": "2026-10-06 14:00:00",
        }
    ])
    res = analyzer.analyze(chain_df, spot_price=25000.0)
    metrics = res["metrics"]

    # Even normalized: 0.005 * 2.52 = 0.013 < 0.05
    assert metrics["commitment_ratio"] < 0.05
