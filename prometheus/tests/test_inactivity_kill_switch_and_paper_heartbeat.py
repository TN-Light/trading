import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import pytz

from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.types import Position, Direction, ExitReason, TradeSnapshot
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
from prometheus.papertrade.fill_simulator import FillSimulator, FillResult

IST = pytz.timezone("Asia/Kolkata")


class MockFeed:
    def __init__(self, prices=None):
        self.prices = prices or {}

    def get_ltp(self, instrument: str) -> float:
        return self.prices.get(instrument, 0.0)


def test_inactivity_kill_switch_requires_true_40min_elapsed_duration():
    """Verify that mid-candle entries are NOT killed after only 20-27 minutes despite bars_held >= 3."""
    feed = MockFeed({"NIFTY19OCT2623150CE": 80.90, "NIFTY 50": 23105.70})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    tracker = PositionTracker(
        fill_sim=sim,
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
        trend_aware=True,
    )

    now = datetime.now(IST)
    # Entry 27 minutes ago (1620s < 2400s threshold)
    entry_time = now - timedelta(minutes=27)

    pos = Position(
        trade_id="PAPER-20261016-TEST27M",
        symbol="NIFTY 50",
        instrument="NIFTY19OCT2623150CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=102.20,
        entry_time=entry_time,
        stop_loss=61.85,
        target=109.15,
        max_bars=16,
        strategy="Golden_Setup (1H+VWAP+ORB)",
        trade_mode="intraday",
        bars_held=3,  # 3 bars elapsed (e.g. 10:15, 10:30, 10:45)
        entry_spot=23131.55,
        atr=85.0,
        target_gain_pts=14.0,
        tier="C",
    )
    tracker.open_positions[pos.trade_id] = pos

    snap = TradeSnapshot(
        timestamp=now,
        symbol="NIFTY 50",
        instrument="",
        open=23110.0,
        high=23120.0,
        low=23100.0,
        close=23105.70,
        volume=15000,
        bar_interval="15minute",
    )

    # With only 27 minutes elapsed, inactivity kill switch must NOT trigger
    exit_px, exit_reason = tracker._evaluate_exit_via_feed(pos, snap, is_session_end=False, is_square_off=False)
    assert exit_reason != ExitReason.INACTIVITY_KILL_SWITCH, (
        f"Inactivity kill switch must NOT fire at 27 minutes! Got: {exit_reason}"
    )


def test_inactivity_kill_switch_triggers_after_true_40min_elapsed_duration():
    """Verify that after >= 40-45 minutes (2400s+) and 3+ bars of broken trend, inactivity kill switch fires."""
    feed = MockFeed({"NIFTY19OCT2623150CE": 80.90, "NIFTY 50": 23105.70})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    tracker = PositionTracker(
        fill_sim=sim,
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
        trend_aware=True,
    )

    now = datetime.now(IST)
    # Entry 45 minutes ago (2700s >= 2400s)
    entry_time = now - timedelta(minutes=45)

    pos = Position(
        trade_id="PAPER-20261016-TEST45M",
        symbol="NIFTY 50",
        instrument="NIFTY19OCT2623150CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=102.20,
        entry_time=entry_time,
        stop_loss=61.85,
        target=109.15,
        max_bars=16,
        strategy="Golden_Setup (1H+VWAP+ORB)",
        trade_mode="intraday",
        bars_held=3,
        entry_spot=23131.55,
        atr=85.0,
        target_gain_pts=14.0,
        tier="C",
    )
    tracker.open_positions[pos.trade_id] = pos

    snap = TradeSnapshot(
        timestamp=now,
        symbol="NIFTY 50",
        instrument="",
        open=23110.0,
        high=23120.0,
        low=23100.0,
        close=23105.70,
        volume=15000,
        bar_interval="15minute",
    )

    exit_px, exit_reason = tracker._evaluate_exit_via_feed(pos, snap, is_session_end=False, is_square_off=False)
    assert exit_reason == ExitReason.INACTIVITY_KILL_SWITCH
    assert exit_px == pytest.approx(80.90, 0.01)


def test_live_paper_capture_poll_open_positions_triggers_heartbeat():
    """Verify LivePaperCapture.poll_open_positions() updates LTP and fires 5-min heartbeat status alert."""
    mock_telegram = MagicMock()
    feed = MockFeed({"NIFTY19OCT2623100CE": 105.0})

    config = CaptureConfig(enabled=True, sqlite_path=":memory:", csv_path="reports/papertrade/test_ledger.csv")
    capture = LivePaperCapture(config=config, ltp_source=feed, telegram=mock_telegram)

    now = datetime.now(IST)
    # Position entered 310 seconds ago (milestone 1 = 5 minutes)
    entry_time = now - timedelta(seconds=310)

    pos = Position(
        trade_id="PAPER-HEARTBEAT-01",
        symbol="NIFTY 50",
        instrument="NIFTY19OCT2623100CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=95.0,
        entry_time=entry_time,
        stop_loss=75.0,
        target=125.0,
        max_bars=16,
        strategy="Golden_Setup",
        trade_mode="intraday",
        tier="C",
    )
    capture._tracker.open_positions[pos.trade_id] = pos

    # Fast poll cycle
    capture.poll_open_positions()

    # Verify LTP updated
    assert pos.current_price == 105.0

    # Verify 5-minute Telegram Heartbeat card was triggered
    assert mock_telegram.alert_trade_update.call_count == 1
    call_args = mock_telegram.alert_trade_update.call_args[0][0]
    assert call_args["trade_id"] == "PAPER-HEARTBEAT-01"
    assert call_args["milestone"] == 5  # 5th 60-second milestone (310s elapsed)
    assert call_args["holding_duration"] == "5 minutes"
    assert call_args["current_price"] == 105.0
    assert call_args["gross_pnl_pts"] == 10.0  # 105 - 95

    # Second poll immediately should NOT fire duplicate milestone 1
    capture.poll_open_positions()
    assert mock_telegram.alert_trade_update.call_count == 1

    # Advance entry time to 610s (milestone 10 = 10 minutes)
    pos.entry_time = now - timedelta(seconds=610)
    capture.poll_open_positions()
    assert mock_telegram.alert_trade_update.call_count == 2
    call_args2 = mock_telegram.alert_trade_update.call_args[0][0]
    assert call_args2["milestone"] == 10  # 10th 60-second milestone (~10 minutes)
    assert call_args2["holding_duration"] == "10 minutes"
