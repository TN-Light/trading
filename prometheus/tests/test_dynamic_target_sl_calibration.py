import pytest
from datetime import datetime, timezone, timedelta
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.types import Position, Direction, TradeSnapshot
from prometheus.papertrade.fill_simulator import FillSimulator

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

    # Price moves to 428.0 (+13.0 pts): Breakeven triggers!
    tracker._maybe_advance_trailing_stop(pos, 428.0)
    assert pos.breakeven_set
    expected_be_sl = 415.0 + 1.9  # Entry + brokerage
    assert abs(pos.stop_loss - expected_be_sl) < 1e-4

    # Critical: Check that recorder persisted the updated SL to disk!
    assert len(mock_rec.recorded_open) >= 2
    assert abs(mock_rec.recorded_open[-1]["stop_loss"] - expected_be_sl) < 1e-4
    assert mock_rec.recorded_open[-1]["breakeven_set"] == 1
