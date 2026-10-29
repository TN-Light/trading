"""
prometheus/tests/test_oct29_target_inflation_and_blacklist_fixes.py
==================================================================
Regression and invariant tests for the October 29 quantitative upgrades:
  1. Elimination of causal inversion in target calibration (no target inflation to sl_pts * 1.2).
  2. Structural Risk Ceiling & Trade Rejection Gate on excessive SL risk (Nifty > 25, Sensex > 55, BNF > 65).
  3. Permanent blacklisting of NIFTY FIN SERVICE across main.py, engine.py, and settings.yaml.
  4. Preservation of 5-tuple interface contract and uncompressed runner calibration for pristine setups.
"""

import pytest
import yaml
from prometheus.signals.target_calibrator import calibrate_target_and_sl, TargetCalibrationResult
from prometheus.papertrade.engine import PaperTradeEngine
from prometheus.papertrade.signal_source import from_signal_dict


class DummyFeed:
    def get_ltp(self, instrument: str) -> float:
        return 171.45


def test_finnifty_removed_from_settings_yaml():
    """Verify NIFTY FIN SERVICE is absent from allowed_instruments and instruments in settings.yaml."""
    with open("prometheus/config/settings.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    intraday = conf.get("intraday", {})
    allowed = intraday.get("allowed_instruments", [])
    instruments = intraday.get("instruments", [])
    assert "NIFTY FIN SERVICE" not in allowed
    assert "FINNIFTY" not in allowed
    assert "NIFTY FIN SERVICE" not in instruments
    assert "FINNIFTY" not in instruments


def test_target_calibrator_rejects_today_trade_6_and_7():
    """
    Verify target calibrator strictly REJECTS setups matching today's Trade #6 and Trade #7
    where bloated structural stops caused massive drawdowns and inactivity kill switches.
    """
    # Trade #6: NIFTY 50 Golden Setup with 50.4 pt SL
    res6 = calibrate_target_and_sl(
        symbol="NIFTY 50",
        spot_price=22858.3,
        spot_atr=25.0,
        opt_ltp=171.45,
        edge_score=6.5,
        tier="B",
        is_htf_aligned=True,
        structural_sl_pts=50.4,
    )
    assert res6.is_rejected is True
    assert "Excessive SL risk" in res6.rejection_reason
    assert res6.target_gain_pts < 25.0  # Must NOT be 60.5 pts!

    # Trade #7: SENSEX PriceAction Momentum with 75.7 pt SL
    res7 = calibrate_target_and_sl(
        symbol="SENSEX",
        spot_price=73500.0,
        spot_atr=98.0,
        opt_ltp=212.10,
        edge_score=6.5,
        tier="C",
        structural_sl_pts=75.7,
    )
    assert res7.is_rejected is True
    assert "Excessive SL risk" in res7.rejection_reason


def test_target_calibrator_accepts_today_trade_1_2_3():
    """
    Verify target calibrator ACCEPTS setups matching today's pristine morning wave (Trades #1, #2, #3).
    """
    # Trade #1: NIFTY 50 Tier C Long Gamma (14.0 pt compressed scalp target)
    res1 = calibrate_target_and_sl(
        symbol="NIFTY 50",
        spot_price=22796.9,
        spot_atr=25.0,
        opt_ltp=162.15,
        edge_score=6.5,
        tier="C",
        is_htf_aligned=True,
        gamma_regime="LONG_GAMMA",
        net_gex=42231703864.44,
        structural_sl_pts=12.0,
    )
    assert res1.is_rejected is False
    assert res1.is_compressed is True
    assert 8.0 <= res1.target_gain_pts <= 14.0

    # Trade #2: BANK NIFTY (28.6 pt target)
    res2 = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=54453.5,
        spot_atr=88.0,
        opt_ltp=1030.10,
        edge_score=6.5,
        tier="C",
        is_htf_aligned=True,
        structural_sl_pts=35.0,
    )
    assert res2.is_rejected is False

    # Trade #3: SENSEX (32.0 pt target)
    res3 = calibrate_target_and_sl(
        symbol="SENSEX",
        spot_price=73506.9,
        spot_atr=98.0,
        opt_ltp=202.90,
        edge_score=6.5,
        tier="C",
        is_htf_aligned=True,
        structural_sl_pts=30.0,
    )
    assert res3.is_rejected is False


def test_engine_rejects_blacklisted_finnifty_signal():
    """Verify PaperTradeEngine strictly rejects FINNIFTY signals."""
    engine = PaperTradeEngine(feed=DummyFeed(), recorder=None, enforce_risk_ceiling=True)
    sig = from_signal_dict({
        "symbol": "NIFTY FIN SERVICE",
        "instrument": "FINNIFTY26OCT24500CE",
        "direction": "LONG",
        "strategy": "PriceAction_Momentum",
        "entry_price": 475.0,
        "stop_loss": 465.0,
        "target": 495.0,
    })
    tid = engine.process_new_signal(sig)
    assert tid is None
    assert engine.signals_skipped_other >= 1


def test_engine_rejects_excessive_sl_signal():
    """Verify PaperTradeEngine strictly rejects signals with excessive SL risk."""
    engine = PaperTradeEngine(feed=DummyFeed(), recorder=None, enforce_risk_ceiling=True)
    # NIFTY 50 with 40 pt SL (exceeds 25.0 pt threshold)
    sig_nifty = from_signal_dict({
        "symbol": "NIFTY 50",
        "instrument": "NIFTY03NOV2622850CE",
        "direction": "LONG",
        "strategy": "Golden_Setup",
        "entry_price": 171.45,
        "stop_loss": 131.45,  # 40 pt risk
        "target": 231.95,
    })
    tid = engine.process_new_signal(sig_nifty)
    assert tid is None
    assert engine.signals_skipped_other >= 1


def test_target_calibrator_result_namedtuple_contract():
    """Verify TargetCalibrationResult preserves 5-tuple unpacking and dictionary access."""
    res = TargetCalibrationResult(
        target_gain_pts=14.0,
        sl_pts=10.0,
        tgt_price=174.0,
        sl_price=150.0,
        is_compressed=True,
        is_rejected=False,
        rejection_reason="",
    )
    tgt, sl, tp, sp, comp = res
    assert (tgt, sl, tp, sp, comp) == (14.0, 10.0, 174.0, 150.0, True)
    assert len(res) == 5
    assert res.is_rejected is False
    assert res["is_rejected"] is False
    d = res.to_dict()
    assert d["is_rejected"] is False
    assert d["rejection_reason"] == ""
