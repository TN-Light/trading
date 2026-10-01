import pytest
from datetime import datetime, timezone, timedelta
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.types import Position, Direction, TradeSnapshot
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.signals.target_calibrator import (
    calibrate_target_and_sl,
    calculate_structural_sl,
    TargetCalibrationResult,
)

IST = timezone(timedelta(hours=5, minutes=30))


class DummyFeed:
    def __init__(self, ltp_map=None):
        self.ltp_map = ltp_map or {}

    def get_ltp(self, instrument: str) -> float:
        return float(self.ltp_map.get(instrument, 0.0))


class MockRecorder:
    def __init__(self):
        self.recorded_open = []
        self.deleted_open = []

    def record_open_position(self, pos_dict):
        self.recorded_open.append(dict(pos_dict))

    def delete_open_position(self, trade_id):
        self.deleted_open.append(trade_id)


def compute_target_and_sl(symbol: str, spot_price: float, opt_ltp: float, edge_score: float, spot_atr: float = 0.0, is_low_vix: bool = False):
    """Mirror the main.py dynamic target and SL calculation."""
    sym_u = symbol.upper()
    if spot_atr <= 0:
        if "BANK" in sym_u:
            spot_atr = 88.0
        elif "SENSEX" in sym_u or "BSX" in sym_u:
            spot_atr = 98.0
        elif "NIFTY" in sym_u:
            spot_atr = 25.0
        else:
            spot_atr = max(10.0, spot_price * 0.0035)

    atm_delta = 0.50
    eom = atm_delta * spot_atr

    if edge_score >= 8.0:
        quality_mult = 1.15
    elif edge_score >= 6.5:
        quality_mult = 0.85
    else:
        quality_mult = 0.65

    if is_low_vix:
        quality_mult *= 0.80

    if "BANK" in sym_u:
        noise_floor = 20.0
        min_target = 18.0
    elif "SENSEX" in sym_u or "BSX" in sym_u:
        noise_floor = 22.0
        min_target = 20.0
    else:
        noise_floor = 8.0
        min_target = 6.0

    target_gain_pts = round(max(min_target, eom * quality_mult), 1)
    if opt_ltp > 0:
        target_gain_pts = min(target_gain_pts, round(opt_ltp * 0.45, 1))

    sl_pts = max(noise_floor, round(0.55 * eom, 1))
    max_sl_cap = max(noise_floor, round(target_gain_pts * 1.15, 1))
    sl_pts = min(sl_pts, max_sl_cap)
    if opt_ltp > 0:
        sl_pts = min(sl_pts, round(opt_ltp * 0.30, 1))

    tgt_price = round(opt_ltp + target_gain_pts, 2)
    sl_price = round(max(1.0, opt_ltp - sl_pts), 2)
    breakeven_trigger_pts = round(target_gain_pts * 0.50, 1)

    return {
        "target_gain_pts": target_gain_pts,
        "sl_pts": sl_pts,
        "tgt_price": tgt_price,
        "sl_price": sl_price,
        "breakeven_trigger_pts": breakeven_trigger_pts,
        "eom": eom,
    }


def test_banknifty_achievable_target_vs_fantasy_target():
    """Verify that Bank Nifty receives an achievable 25-45 pt target instead of the buggy 90-120 pt fantasy target."""
    opt_ltp = 415.16
    spot_price = 56100.0

    # Old buggy calculation produced:
    # min(415.16 * 0.28, max(14.0, 415.16 * 0.22)) = 91.34 pts (Target = 506.5)
    old_buggy_target_gain = round(min(opt_ltp * 0.28, max(14.0, opt_ltp * 0.22)), 2)
    assert old_buggy_target_gain > 90.0

    # New dynamic calculation for moderate conviction (edge_score=6.0)
    res_mod = compute_target_and_sl("NIFTY BANK", spot_price, opt_ltp, edge_score=6.0, spot_atr=88.0)
    assert 25.0 <= res_mod["target_gain_pts"] <= 30.0
    assert res_mod["tgt_price"] == round(opt_ltp + res_mod["target_gain_pts"], 2)

    # New dynamic calculation for strong/golden conviction (edge_score=7.0)
    res_strong = compute_target_and_sl("NIFTY BANK", spot_price, opt_ltp, edge_score=7.0, spot_atr=88.0)
    assert 35.0 <= res_strong["target_gain_pts"] <= 40.0

    # Bank Nifty Stop Loss must strictly respect the noise floor (>= 20 pts)
    assert res_mod["sl_pts"] >= 20.0
    assert res_strong["sl_pts"] >= 20.0


def test_nifty_target_and_noise_floor():
    """Verify NIFTY 50 targets and stop loss bounds."""
    opt_ltp = 120.0
    spot_price = 25000.0

    res = compute_target_and_sl("NIFTY 50", spot_price, opt_ltp, edge_score=6.0, spot_atr=25.0)
    # Target should be reasonable (~7-10 pts), not 26+ pts
    assert 7.0 <= res["target_gain_pts"] <= 12.0
    # Stop loss must respect noise floor of 8.0 pts
    assert res["sl_pts"] >= 8.0


def test_sensex_target_and_noise_floor():
    """Verify SENSEX targets and stop loss bounds."""
    opt_ltp = 350.0
    spot_price = 82000.0

    res = compute_target_and_sl("SENSEX", spot_price, opt_ltp, edge_score=7.0, spot_atr=98.0)
    assert 35.0 <= res["target_gain_pts"] <= 45.0
    # SENSEX Stop loss must respect noise floor of 22.0 pts
    assert res["sl_pts"] >= 22.0


def test_position_tracker_breakeven_at_50pct_target_and_disk_persistence():
    """Verify PositionTracker advances SL to breakeven at 50% target and flushes to disk immediately."""
    mock_rec = MockRecorder()
    feed = DummyFeed()
    fill_sim = FillSimulator(feed=feed)
    tracker = PositionTracker(
        fill_sim=fill_sim,
        cost_model=CostModel(),
        enable_trailing=True,
        recorder=mock_rec,
    )

    now = datetime(2026, 9, 18, 10, 0, tzinfo=IST)
    pos = Position(
        trade_id="TR-TEST-001",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY29SEP2656100PE",
        underlying="NIFTY BANK",
        direction=Direction.SHORT,
        quantity=30,
        entry_price=415.0,
        entry_time=now,
        stop_loss=390.0,
        target=450.0,
        max_bars=16,
        target_gain_pts=35.0,
    )
    tracker.open_position(pos)
    assert len(mock_rec.recorded_open) == 1
    assert mock_rec.recorded_open[-1]["stop_loss"] == 390.0

    # Bank Nifty cost buffer is 1.9 pts
    # 50% of target (35.0 pts) = 17.5 pts
    # be_gain_threshold = min(10.0, 17.5) = 10.0 pts
    # be_trigger = 10.0 + 1.9 = 11.9 pts

    # Price moves to 420.0 (+5.0 pts): Breakeven should NOT trigger yet
    tracker._maybe_advance_trailing_stop(pos, 420.0)
    assert not pos.breakeven_set
    assert pos.stop_loss == 390.0

    # Price moves to 428.0 (+13.0 pts, 0.52R): Progressive Stage 1 (Half-Risk Cut) triggers!
    # Cuts risk by 50%: 415.0 - 0.5 * 25.0 = 402.50 (cushion is 25.5 pts, outside noise floor)
    tracker._maybe_advance_trailing_stop(pos, 428.0)
    assert pos.half_risk_set is True
    assert not pos.breakeven_set
    assert abs(pos.stop_loss - 402.50) < 1e-4

    # Price moves to 437.0 (+22.0 pts, 0.88R >= 0.85R): Progressive Stage 2 (Breakeven) triggers!
    tracker._maybe_advance_trailing_stop(pos, 437.0)
    assert pos.breakeven_set is True
    expected_be_sl = 415.0 + 1.9  # Entry + brokerage
    assert abs(pos.stop_loss - expected_be_sl) < 1e-4

    # Critical: Check that recorder persisted the updated SL to disk!
    assert len(mock_rec.recorded_open) >= 3
    assert abs(mock_rec.recorded_open[-1]["stop_loss"] - expected_be_sl) < 1e-4
    assert mock_rec.recorded_open[-1]["breakeven_set"] == 1


# =====================================================================
# R1 Tests: Tier C Adaptive Target Compression & Retest Expansion Bypass
# =====================================================================

def test_tier_c_banknifty_positive_gamma_compression():
    """Verify that Tier C Bank Nifty in positive gamma (LONG_GAMMA) compresses to [20.0, 30.0] pts."""
    res = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=56100.0,
        spot_atr=88.0,
        opt_ltp=415.0,
        edge_score=5.5,
        tier="C",
        is_htf_aligned=False,
        gamma_regime="LONG_GAMMA",
        net_gex=12211530.95,
    )
    assert res.is_compressed is True
    assert 20.0 <= res.target_gain_pts <= 30.0
    # Expected: round(44.0 * 0.55, 1) = 24.2 pts
    assert res.target_gain_pts == pytest.approx(24.2, rel=1e-2)
    assert res.tgt_price == round(415.0 + 24.2, 2)
    assert res.breakeven_trigger_pts == round(24.2 * 0.50, 1)


def test_tier_c_banknifty_counter_trend_compression():
    """Verify that Tier C Bank Nifty counter-trend (not is_htf_aligned) compresses to [20.0, 30.0] pts even in neutral gamma."""
    res = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=56100.0,
        spot_atr=88.0,
        opt_ltp=415.0,
        edge_score=5.5,
        tier="C",
        is_htf_aligned=False,
        gamma_regime="NEUTRAL",
        net_gex=0.0,
    )
    assert res.is_compressed is True
    assert 20.0 <= res.target_gain_pts <= 30.0
    assert res.target_gain_pts == pytest.approx(24.2, rel=1e-2)


def test_tier_c_wide_structural_sl_bypasses_retest_expansion():
    """
    Verify that Tier C signals with wide structural SL strictly BYPASS dynamic retest expansion.
    Under standard logic, sl_pts * 1.2 = 89.8 * 1.2 = 107.8 pts would inflate the target.
    Under Tier C compression, retest expansion is bypassed and target remains at scalp boundary (24.2 pts).
    """
    res = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=54778.90,
        spot_atr=88.0,
        opt_ltp=1055.05,
        edge_score=5.5,
        tier="C",
        is_htf_aligned=False,
        gamma_regime="LONG_GAMMA",
        net_gex=12211530.95,
        structural_sl_pts=89.8,
    )
    # Target MUST remain compressed and strictly bypass expansion
    assert res.is_compressed is True
    assert 20.0 <= res.target_gain_pts <= 30.0
    assert res.target_gain_pts == pytest.approx(24.2, rel=1e-2)
    assert res.target_gain_pts < 35.0  # Must NOT inflate to 107.8 pts!
    # SL preserves structural protection
    assert res.sl_pts == pytest.approx(89.8, rel=1e-2)


def test_tier_b_retains_uncompressed_target_and_retest_expansion():
    """
    Verify that Tier B (e.g. Institutional Trend Day / Golden Setup) retains full uncompressed
    multi-ATR target calculation and dynamic retest expansion up to 107.8 pts.
    """
    res = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=54778.90,
        spot_atr=88.0,
        opt_ltp=1055.05,
        edge_score=6.5,
        tier="B",
        is_htf_aligned=True,
        gamma_regime="LONG_GAMMA",
        net_gex=12211530.95,
        structural_sl_pts=89.8,
    )
    assert res.is_compressed is False
    # Retest expansion MUST trigger: max(target, round(89.8 * 1.2, 1)) = 107.8 pts
    assert res.target_gain_pts == pytest.approx(107.8, rel=1e-2)
    assert res.sl_pts == pytest.approx(89.8, rel=1e-2)


def test_tier_s_perfect_storm_uncompressed_target():
    """Verify that Tier S Perfect Storm retains uncompressed multi-ATR runner targets."""
    res = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=56100.0,
        spot_atr=88.0,
        opt_ltp=415.0,
        edge_score=8.5,
        tier="S",
        is_htf_aligned=True,
        gamma_regime="SHORT_GAMMA",
        net_gex=-15000000.0,
        structural_sl_pts=35.0,
    )
    assert res.is_compressed is False
    # eom = 44.0, quality_mult = 1.15 => 50.6 pts
    assert res.target_gain_pts == pytest.approx(50.6, rel=1e-2)
    assert res.target_gain_pts > 45.0


def test_tier_c_sensex_compression():
    """Verify that Tier C SENSEX signals in positive gamma compress to [22.0, 32.0] pts."""
    res = calibrate_target_and_sl(
        symbol="SENSEX",
        spot_price=82000.0,
        spot_atr=98.0,
        opt_ltp=350.0,
        edge_score=5.5,
        tier="C",
        is_htf_aligned=False,
        gamma_regime="LONG_GAMMA",
        net_gex=53660506.23,
    )
    assert res.is_compressed is True
    assert 22.0 <= res.target_gain_pts <= 32.0
    # Expected: round(49.0 * 0.55, 1) = 27.0 pts
    assert res.target_gain_pts == pytest.approx(27.0, rel=1e-2)


def test_tier_c_nifty_compression():
    """Verify that Tier C NIFTY 50 signals in positive gamma compress to [8.0, 14.0] pts."""
    res = calibrate_target_and_sl(
        symbol="NIFTY 50",
        spot_price=25000.0,
        spot_atr=25.0,
        opt_ltp=120.0,
        edge_score=5.5,
        tier="C",
        is_htf_aligned=False,
        gamma_regime="LONG_GAMMA",
        net_gex=94883049.03,
    )
    assert res.is_compressed is True
    assert 8.0 <= res.target_gain_pts <= 14.0
    # Expected: round(12.5 * 0.75, 1) = 9.4 pts
    assert res.target_gain_pts == pytest.approx(9.4, rel=1e-2)


def test_interface_contract_5tuple_unpacking():
    """Verify that calibrate_target_and_sl return value unpacks into a 5-tuple matching the PROJECT.md interface contract."""
    res = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=56100.0,
        spot_atr=88.0,
        opt_ltp=415.0,
        edge_score=5.5,
        tier="C",
        is_htf_aligned=False,
        gamma_regime="LONG_GAMMA",
        net_gex=12211530.95,
        structural_sl_pts=35.0,
    )
    target_gain_pts, sl_pts, tgt_price, sl_price, is_compressed = res
    assert target_gain_pts == pytest.approx(24.2, rel=1e-2)
    assert sl_pts == pytest.approx(35.0, rel=1e-2)
    assert tgt_price == pytest.approx(439.2, rel=1e-2)
    assert sl_price == pytest.approx(380.0, rel=1e-2)
    assert is_compressed is True
    assert res["target_gain_pts"] == target_gain_pts
    assert res.to_dict()["is_compressed"] is True

