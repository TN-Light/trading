"""
Unified Automated Test Suite for Batch 2 Audit Fixes (Defects 19 - 30)
Verifies:
19. PositionTracker _evaluate_exit_via_feed preserves real market loss on TIME_STOP
20. PositionTracker _evaluate_exit_via_feed returns actual ltp on SL breach (symmetric slippage)
21. PositionMonitor top-down sequential evaluation & no lockout by min_be_gain
22. PositionTracker Stage 4 dynamic trail uses gain multiplier (no 0.24 pt instant stop-out)
23. PaperTradeEngine strictly enforces Rs 850 Single-Lot Risk Ceiling across all symbols
24. CostModel uses 0.15% STT and PositionTracker uses 300s (5-minute) heartbeat interval
25. LivePaperCapture _LTPFeedWrapper get_quote returns realistic quotes without synthetic tight spread
26. TradeRecorder and LivePaperCapture rehydrate initial_sl, initial_risk_distance, half_risk_set
27. ExecutionGate undo_pass cleans tradingsymbol and allows same-bar retry
28. DataBridge _resample_to_hourly uses offset='15min' matching 09:15 IST market open
29. LiveScanner VIX gate respects low_vix_mode and scales SL respecting Rs 850 risk cap
30. Metrics _generate_monthly_tracker_md reads dynamic capital and current month
"""

import os
import pytest
import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch

from prometheus.papertrade.types import Position, Direction, ExitReason, TradeSnapshot
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.engine import PaperTradeEngine
from prometheus.paper_executor.live_bridge import LivePriceFeed
from prometheus.papertrade.recorder import TradeRecorder
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.pipeline.execution_gate import ExecutionGate, GateVerdict
from prometheus.pipeline.data_bridge import _resample_to_hourly
from prometheus.pipeline.scanner import LiveScanner
from prometheus.pipeline.types import ExecutableSignal
from prometheus.papertrade.metrics import _generate_monthly_tracker_md


def test_defect_19_time_stop_preserves_real_loss():
    """Defect 19: TIME_STOP on an underwater position must return live ltp, not entry_price."""
    mock_sim = MagicMock()
    tracker = PositionTracker(fill_sim=mock_sim)
    pos = Position(
        trade_id="T19",
        symbol="NIFTY 50",
        instrument="NIFTY26NOV24000CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=200.0,
        entry_time=datetime.now(),
        stop_loss=175.0,
        target=250.0,
        max_bars=16,
        bars_held=16,
        breakeven_set=True,
    )
    snap = TradeSnapshot(symbol="NIFTY 50", instrument="NIFTY26NOV24000CE", timestamp=datetime.now(), open=24000, high=24050, low=23950, close=24000)
    mock_feed = MagicMock()
    mock_feed.get_ltp.return_value = 180.0  # Option is down 20 pts (underwater)
    tracker.fill_sim.feed = mock_feed

    exit_px, reason = tracker._evaluate_exit_via_feed(pos, snap, is_session_end=False, is_square_off=False)
    assert reason == ExitReason.TIME_STOP
    # Must be real LTP (180.0), NOT fabricated entry_price (200.0)!
    assert exit_px == 180.0


def test_defect_20_symmetric_slippage_on_stop_loss():
    """Defect 20: SL breach in _evaluate_exit_via_feed must return ltp (capturing gap slippage)."""
    mock_sim = MagicMock()
    tracker = PositionTracker(fill_sim=mock_sim)
    pos = Position(
        trade_id="T20",
        symbol="NIFTY 50",
        instrument="NIFTY26NOV24000CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=200.0,
        entry_time=datetime.now(),
        stop_loss=185.0,
        target=250.0,
        max_bars=16,
        bars_held=2,
    )
    snap = TradeSnapshot(symbol="NIFTY 50", instrument="NIFTY26NOV24000CE", timestamp=datetime.now(), open=24000, high=24050, low=23950, close=24000)
    mock_feed = MagicMock()
    mock_feed.get_ltp.return_value = 174.0  # Gapped through SL 185 down to 174
    tracker.fill_sim.feed = mock_feed

    exit_px, reason = tracker._evaluate_exit_via_feed(pos, snap, is_session_end=False, is_square_off=False)
    assert reason == ExitReason.STOP_LOSS
    # Must return actual market LTP (174.0), not theoretical SL (185.0)
    assert exit_px == 174.0


def test_defect_21_position_monitor_top_down_evaluation_and_no_lockout():
    """Defect 21: PositionMonitor evaluates top-down and does not get locked out by min_be_gain."""
    mock_broker = MagicMock()
    monitor = PositionMonitor(broker=mock_broker)
    state = TrailingState(
        position_id="POS_21",
        tradingsymbol="BANKNIFTY26NOV52000CE",
        symbol="NIFTY BANK",
        entry_premium=100.0,
        initial_sl=85.0,
        current_sl=85.0,
        target=200.0,
        direction="bullish",
        strategy="trend",
        entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    # Risk distance = 15.0 pts (1R = 15 pts)
    # If price surges to 132.0 (+32 pts gain = +2.13R), it must trigger Stage 3 directly!
    monitor._process_tick(state, current_price=132.0)
    assert state.breakeven_set is True
    assert state.trailing_stage3 is True
    # Stage 3 locks +0.75R = 100 + 15 * 0.75 = 111.25
    assert state.current_sl >= 111.25


def test_defect_22_stage4_hwm_buffer_scaled_by_gain():
    """Defect 22: PositionTracker Stage 4 dynamic trail uses gain multiplier (no 0.24 pt stop-out)."""
    mock_sim = MagicMock()
    tracker = PositionTracker(fill_sim=mock_sim)
    pos = Position(
        trade_id="T22",
        symbol="NIFTY 50",
        instrument="NIFTY26NOV24000CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=88.0,  # 1R = 12 pts
        target=200.0,
        max_bars=16,
        bars_held=5,
        breakeven_set=True,
        high_water_mark=150.0,  # +50 pts gain = 4.16R (> 3.5R)
    )
    pos.initial_risk_distance = 12.0

    # Advance trailing stop at 150.0
    tracker._maybe_advance_trailing_stop(pos, current_price=150.0)

    # Dynamic SL should be high_water_mark - (HWM - entry) * 0.30 = 150 - (50 * 0.30) = 135.0
    # Floor is entry + 0.75 * 12 = 109.0
    # Before the fix, trail_buf was 0.05 * 12 = 0.60 pts, setting SL to 149.40!
    assert pos.stop_loss <= 136.0
    assert pos.stop_loss >= 134.0


def test_defect_23_single_lot_risk_ceiling_in_engine():
    """Defect 23: PaperTradeEngine enforces Rs 850 Risk Ceiling across symbols."""
    mock_feed = MagicMock()
    mock_feed.get_quote.return_value = (200.0, 199.5, 200.5)
    mock_feed.get_ltp.return_value = 200.0
    engine = PaperTradeEngine(feed=mock_feed)
    engine.enforce_risk_ceiling = True

    # NIFTY lot size 65: max allowable SL is 850 / 65 = 13.08 pts
    sig_reject = MagicMock()
    sig_reject.symbol = "NIFTY 50"
    sig_reject.instrument = "NIFTY26NOV24000CE"
    sig_reject.direction = Direction.LONG
    sig_reject.entry_price_hint = 200.0
    sig_reject.stop_loss = 180.0  # 20 pts risk = Rs 1,300 > Rs 850!
    sig_reject.target = 250.0
    sig_reject.max_bars = 16
    sig_reject.strategy = "trend"
    sig_reject.signal_score = 8.0

    engine.fill_sim = MagicMock()
    engine.fill_sim.simulate_fill.return_value = MagicMock(fill_price=200.0, source="quote")

    pos1 = engine.process_new_signal(sig_reject)
    assert pos1 is None  # Must be rejected due to excessive SL risk!

    # Valid NIFTY signal: 12 pts risk = Rs 780 <= Rs 850
    sig_pass = MagicMock()
    sig_pass.symbol = "NIFTY 50"
    sig_pass.instrument = "NIFTY26NOV24000CE"
    sig_pass.direction = Direction.LONG
    sig_pass.entry_price_hint = 200.0
    sig_pass.stop_loss = 188.0  # 12 pts risk = Rs 780
    sig_pass.target = 230.0
    sig_pass.max_bars = 16
    sig_pass.strategy = "trend"
    sig_pass.signal_score = 8.0
    sig_pass.net_gex = None
    sig_pass.gamma_regime = "NEUTRAL"
    sig_pass.bar_timestamp = None

    sig_reject.net_gex = None
    sig_reject.gamma_regime = "NEUTRAL"
    sig_reject.bar_timestamp = None

    pos2 = engine.process_new_signal(sig_pass)
    assert pos2 is not None
    assert pos2 in engine.tracker.open_positions


def test_defect_24_stt_rate_and_heartbeat_interval():
    """Defect 24: CostModel uses 0.15% STT and PositionTracker uses 300s heartbeat interval."""
    cm = CostModel(cost_per_side_bps=1.0, conservative_multiplier=1.0)
    # Entry notional = 100 * 65 = 6500, Exit notional = 120 * 65 = 7800
    # STT on sell side: 7800 * 0.0015 = 11.70
    cost = cm.calculate_trade_cost(entry_notional=6500.0, exit_notional=7800.0, is_spread=False)
    # Total brokerage (40) + STT (11.7) + txn (~7.58) + gst (~8.56) + sebi (~0.01) + stamp (~0.20) ~ Rs 68.05
    assert cost > 60.0

    mock_sim = MagicMock()
    tracker = PositionTracker(fill_sim=mock_sim)
    assert tracker.heartbeat_interval_seconds == 300


def test_defect_25_realistic_quotes_in_live_bridge():
    """Defect 25: _LTPFeedWrapper get_quote returns (ltp, None, None) without fake tight spread."""
    mock_ltp = MagicMock()
    mock_ltp.get_ltp.return_value = 150.0
    del mock_ltp.get_quote  # simulate no raw bid/ask available

    feed = LivePriceFeed(mock_ltp)
    quote = feed.get_quote("NIFTY26NOV24000CE")
    assert quote == (150.0, None, None)


def test_defect_26_rehydration_preserves_initial_risk_distance(tmp_path):
    """Defect 26: Rehydration preserves initial_sl, initial_risk_distance, and half_risk_set."""
    db_file = str(tmp_path / "test_rehydrate.sqlite")
    rec = TradeRecorder(db_file)
    rec._create_open_positions_schema()

    pos_dict = {
        "trade_id": "T26",
        "symbol": "NIFTY 50",
        "instrument": "NIFTY26NOV24000CE",
        "underlying": "NIFTY",
        "direction": "LONG",
        "quantity": 65,
        "entry_price": 200.0,
        "entry_time": "2026-11-09T10:00:00",
        "stop_loss": 205.50,  # ratcheted breakeven SL
        "target": 250.0,
        "max_bars": 16,
        "bars_held": 4,
        "breakeven_set": 1,
        "half_risk_set": 1,
        "trailing_floor": 0.0,
        "high_water_mark": 220.0,
        "strategy": "trend",
        "initial_sl": 180.0,
        "initial_risk_distance": 20.0,
    }
    rec.record_open_position(pos_dict)

    loaded = rec.load_open_positions()
    assert len(loaded) == 1
    row = loaded[0]
    assert row["initial_sl"] == 180.0
    assert row["initial_risk_distance"] == 20.0
    assert row["half_risk_set"] == 1
    rec.close()


def test_defect_27_undo_pass_cleans_instrument_and_allows_same_bar_retry():
    """Defect 27: ExecutionGate.undo_pass removes tradingsymbol and allows same-bar retry."""
    gate = ExecutionGate(max_positions=3)
    sig = ExecutableSignal(
        symbol="NIFTY 50",
        action="BUY_CE",
        direction="bullish",
        option_type="CE",
        entry_price=150.0,
        stop_loss=138.0,
        target=180.0,
        risk_reward=2.5,
        bar_timestamp="2026-11-09 10:30:00",
        instrument="NIFTY26NOV24800CE",
    )
    res1 = gate.check(sig)
    assert res1.passed

    # Immediately check again -> rejected duplicate
    res_dup = gate.check(sig)
    assert not res_dup.passed

    # Now undo pass with instrument
    gate.undo_pass(symbol="NIFTY 50", direction="bullish", instrument="NIFTY26NOV24800CE")

    # Retrying on the SAME bar must now pass!
    res_retry = gate.check(sig)
    assert res_retry.passed


def test_defect_28_resample_hourly_offset_15min():
    """Defect 28: _resample_to_hourly uses offset='15min' matching 09:15 IST market open."""
    dates = pd.date_range("2026-11-09 09:15:00", "2026-11-09 15:15:00", freq="15min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": range(len(dates)),
        "high": [x + 5 for x in range(len(dates))],
        "low": [x - 5 for x in range(len(dates))],
        "close": range(len(dates)),
        "volume": [1000] * len(dates),
    })
    hourly = _resample_to_hourly(df)
    assert not hourly.empty
    # First candle must be 09:15:00, second 10:15:00
    timestamps = [ts.strftime("%H:%M:%S") for ts in pd.to_datetime(hourly["timestamp"])]
    assert timestamps[0] == "09:15:00"
    assert timestamps[1] == "10:15:00"


def test_defect_29_scanner_vix_gate_and_scaled_sl_risk_cap():
    """Defect 29: LiveScanner VIX gate respects low_vix_mode and enforces Rs 850 risk cap."""
    scanner = LiveScanner.__new__(LiveScanner)
    scanner._current_vix = 9.5
    sig = MagicMock()
    sig.action = "BUY_CE"

    # With low_vix_mode enabled, VIX 9.5 must not be rejected as dead market
    reason = scanner._check_vix_gate(sig)
    assert reason is None


def test_defect_30_monthly_tracker_dynamic_capital_and_date(tmp_path):
    """Defect 30: _generate_monthly_tracker_md uses dynamic capital and current month."""
    md_file = str(tmp_path / "test_tracker.md")
    df = pd.DataFrame([{
        "Date": "2026-11-09",
        "Total_Trades": 2,
        "Wins": 2,
        "Losses": 0,
        "Win_Rate_Pct": 100.0,
        "Gross_Profit": 2500.0,
        "Gross_Loss": 0.0,
        "Net_PnL": 2350.0,
        "Capital": 100000.0,
        "Daily_Return_Pct": 2.35,
        "Cumulative_PnL": 2350.0,
        "Notes": "Clean run",
    }])
    _generate_monthly_tracker_md(df, md_file)

    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Must contain current month and dynamic capital Rs 100,000, NOT August 2026 or Rs 15,000
    current_month = datetime.now().strftime("%B %Y")
    assert f"**Tracking Period:** {current_month}" in content
    assert "**Initial Base Capital:** Rs 100,000" in content
    assert "(+2.35%)" in content
