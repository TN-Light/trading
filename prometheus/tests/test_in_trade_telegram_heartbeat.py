"""
Automated Test Suite for Prometheus 5-Minute In-Trade Telegram Status Heartbeat.

Validates:
1. Heartbeat triggers at 5 min (300s) and 10 min (600s) milestones.
2. Strict deduplication / anti-spam within the same 5-minute milestone window.
3. Immediate lifecycle teardown and memory cleanup upon trade exit.
4. Telegram card formatting with all mandatory telemetry (LTP, Entry, Gross/Net P&L in pts & Rs,
   Active SL, Target, Elapsed Wall-Clock Duration, 8-Pillar Health Score).
5. Dual compatibility with Live (PositionMonitor) and Paper (PositionTracker / LivePaperCapture).
6. Credit spread support with multi-leg Kite search copy boxes and credit decay calculations.
7. 8-Pillar Health Score 1-line summary variants (runner, threat, neutral).
"""

from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional
from unittest.mock import MagicMock
import pytest

from prometheus.utils.indian_market import IST
from prometheus.papertrade.types import Position, Direction, ExitReason, TradeSnapshot
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.position_health import PositionHealthReport, format_health_summary
from prometheus.interface.telegram_bot import TelegramBot
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig


class MockTelegramTracker:
    """Mock Telegram bot collecting sent messages and trade updates."""

    def __init__(self):
        self.updates = []
        self.messages = []

    def alert_trade_update(self, update_info: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        info = dict(update_info) if update_info else {}
        info.update(kwargs)
        self.updates.append(info)
        return True

    def send_message(self, text: str, parse_mode: Optional[str] = "HTML") -> bool:
        self.messages.append(text)
        return True

    def send_message_async(self, text: str, parse_mode: str = "HTML") -> None:
        self.messages.append(text)


class MockPriceFeed:
    """Mock price feed for FillSimulator."""

    def __init__(self, ltp_dict: Optional[Dict[str, float]] = None):
        self.ltp_dict = ltp_dict or {}

    def get_ltp(self, symbol: str, instrument: Optional[str] = None) -> float:
        key = instrument or symbol
        return float(self.ltp_dict.get(key, 150.0))


def _create_mock_tracker(on_heartbeat=None, feed_dict=None):
    feed = MockPriceFeed(feed_dict)
    fill_sim = FillSimulator(feed=feed)
    return PositionTracker(
        fill_sim=fill_sim,
        cost_model=CostModel(),
        enable_trailing=True,
        on_heartbeat=on_heartbeat,
    )


def test_heartbeat_fires_at_5min_and_10min_milestones_paper():
    """Verify heartbeat triggers at 300s (5m) and 600s (10m) milestones in PositionTracker."""
    dispatched = []

    def on_heartbeat_cb(pos, price, elapsed_sec, milestone):
        dispatched.append({
            "pos": pos,
            "price": price,
            "elapsed_sec": elapsed_sec,
            "milestone": milestone,
        })

    tracker = _create_mock_tracker(on_heartbeat=on_heartbeat_cb, feed_dict={"NIFTY26OCT24000CE": 165.0})
    now = datetime.now(IST)

    # 1. Position held for 290s: Milestone 0 (no heartbeat)
    pos = Position(
        trade_id="PAPER-001",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT24000CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=75,
        entry_price=140.0,
        stop_loss=120.0,
        target=180.0,
        max_bars=10,
        entry_time=now - timedelta(seconds=290),
    )
    tracker.open_position(pos)

    fired = tracker.check_heartbeat(pos)
    assert not fired
    assert len(dispatched) == 0

    # 2. Position advances to 305s: Milestone 1 (5 minutes)
    pos.entry_time = now - timedelta(seconds=305)
    fired = tracker.check_heartbeat(pos)
    assert fired
    assert len(dispatched) == 1
    assert dispatched[0]["milestone"] == 1
    assert dispatched[0]["elapsed_sec"] >= 300
    assert dispatched[0]["price"] == 165.0

    # 3. Position advances to 605s: Milestone 2 (10 minutes)
    pos.entry_time = now - timedelta(seconds=605)
    fired = tracker.check_heartbeat(pos)
    assert fired
    assert len(dispatched) == 2
    assert dispatched[1]["milestone"] == 2
    assert dispatched[1]["elapsed_sec"] >= 600


def test_heartbeat_anti_spam_deduplication():
    """Verify ticks arriving at 305s, 320s, 400s, 590s do NOT emit duplicate alerts."""
    dispatched = []

    def on_heartbeat_cb(pos, price, elapsed_sec, milestone):
        dispatched.append(milestone)

    tracker = _create_mock_tracker(on_heartbeat=on_heartbeat_cb)
    now = datetime.now(IST)

    pos = Position(
        trade_id="DEDUP-001",
        symbol="BANK NIFTY",
        instrument="BANKNIFTY26OCT52000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=30,
        entry_price=250.0,
        stop_loss=210.0,
        target=330.0,
        max_bars=10,
        entry_time=now - timedelta(seconds=305),
    )
    tracker.open_position(pos)

    # First trigger at 305s
    assert tracker.check_heartbeat(pos) is True
    assert len(dispatched) == 1

    # Repeated checks within the same [300s, 599s] milestone window
    for sec_offset in [310, 350, 420, 500, 599]:
        pos.entry_time = now - timedelta(seconds=sec_offset)
        assert tracker.check_heartbeat(pos) is False
        assert len(dispatched) == 1  # Strictly 1 alert

    # Next milestone at 601s fires once
    pos.entry_time = now - timedelta(seconds=601)
    assert tracker.check_heartbeat(pos) is True
    assert len(dispatched) == 2
    assert dispatched == [1, 2]


def test_heartbeat_clean_teardown_on_trade_close():
    """Verify closed trades immediately dismantle timer state and never fire post-exit."""
    dispatched = []

    def on_heartbeat_cb(pos, price, elapsed_sec, milestone):
        dispatched.append(milestone)

    tracker = _create_mock_tracker(on_heartbeat=on_heartbeat_cb)
    now = datetime.now(IST)

    pos = Position(
        trade_id="TEARDOWN-001",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT24200CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=75,
        entry_price=100.0,
        stop_loss=80.0,
        target=140.0,
        max_bars=10,
        entry_time=now - timedelta(seconds=305),
    )
    tracker.open_position(pos)

    # Fires 5-min heartbeat
    assert tracker.check_heartbeat(pos) is True
    assert len(dispatched) == 1
    assert "TEARDOWN-001" in tracker._heartbeat_milestones

    # Close trade at 400s
    trade = tracker.close_position("TEARDOWN-001", now, exit_price=130.0, exit_reason=ExitReason.TARGET)
    assert trade is not None
    assert "TEARDOWN-001" not in tracker.open_positions
    assert "TEARDOWN-001" not in tracker._heartbeat_milestones

    # Advance time to 605s (Milestone 2): must NOT fire because position is closed
    pos.entry_time = now - timedelta(seconds=605)
    assert tracker.check_heartbeat(pos) is False
    assert len(dispatched) == 1  # No additional alerts sent after exit


def test_telegram_card_formatting_all_required_fields():
    """Verify TelegramBot.alert_trade_update formats all mandatory metrics into HTML card."""
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message = MagicMock(return_value=True)
    bot.send_message_async = MagicMock()

    payload = {
        "trade_id": "POS-20261007-001",
        "symbol": "NIFTY 50",
        "instrument": "NIFTY26JUL24500CE",
        "kite_search": "NIFTY 24500 CE",
        "direction": "BUY CE",
        "current_price": 145.50,
        "entry_price": 120.00,
        "gross_pnl_pts": 25.5,
        "gross_pnl": 1912.50,
        "net_pnl": 1855.50,
        "net_pnl_pts": 24.7,
        "stop_loss": 120.90,
        "target": 170.00,
        "trailing_stage": "BREAKEVEN",
        "holding_duration": "5 minutes",
        "health_score": 45.0,
        "health_summary": "Score: +45/100 [HEALTHY] | Momentum healthy, moving toward target",
        "quantity": 75,
        "async_send": False,
    }

    result = bot.alert_trade_update(payload)
    assert result is True
    bot.send_message.assert_called_once()

    msg = bot.send_message.call_args[0][0]

    # Header check
    assert "TRADE UPDATE — LIVE STATUS" in msg or "TRADE UPDATE" in msg
    # Symbol & Direction
    assert "NIFTY 50" in msg
    assert "BUY CE" in msg
    # Zerodha Kite contract search box & API
    assert "NIFTY 24500 CE" in msg
    assert "NIFTY26JUL24500CE" in msg
    # LTP vs Entry Price
    assert "Rs 145.50" in msg
    assert "Rs 120.00" in msg
    assert "+21.2%" in msg or "+21.3%" in msg
    # Running Gross & Net P&L in pts and Rs
    assert "+25.5 pts" in msg
    assert "+1,912.50 Rs" in msg
    assert "+24.7 pts" in msg
    assert "+1,855.50 Rs" in msg
    # Active Trailing SL & Target
    assert "Rs 120.90" in msg
    assert "BREAKEVEN" in msg
    assert "Rs 170.00" in msg
    # Elapsed Wall-Clock Duration
    assert "5 minutes" in msg
    # 8-Pillar Health Score Summary
    assert "8-Pillar Health Score" in msg
    assert "Momentum healthy, moving toward target" in msg
    # Trade ID
    assert "POS-20261007-001" in msg


def test_heartbeat_live_position_monitor_compatibility():
    """Verify live PositionMonitor evaluates cadence in _process_tick without blocking trailing."""
    dispatched = []

    def on_heartbeat_cb(state, current_price, elapsed_sec, milestone, health_report):
        dispatched.append({
            "state": state,
            "current_price": current_price,
            "elapsed_sec": elapsed_sec,
            "milestone": milestone,
            "health_report": health_report,
        })

    mock_broker = MagicMock()
    monitor = PositionMonitor(
        broker=mock_broker,
        poll_interval=1,
        on_heartbeat=on_heartbeat_cb,
    )

    now = datetime.now()
    entry_time_str = (now - timedelta(seconds=290)).strftime("%Y-%m-%d %H:%M:%S")

    state = TrailingState(
        position_id="LIVE-001",
        tradingsymbol="BANKNIFTY26OCT52500CE",
        symbol="BANK NIFTY",
        entry_premium=300.0,
        initial_sl=260.0,
        current_sl=260.0,
        target=400.0,
        direction="bullish",
        entry_time=entry_time_str,
    )
    monitor.add_position(state)

    # 1. At 290s: should NOT fire
    monitor._process_tick(state, current_price=315.0)
    assert len(dispatched) == 0

    # 2. Advance state entry_time to 305s ago: fires Milestone 1
    state.entry_time = (now - timedelta(seconds=305)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=320.0)
    assert len(dispatched) == 1
    assert dispatched[0]["milestone"] == 1
    assert dispatched[0]["elapsed_sec"] >= 300
    assert dispatched[0]["current_price"] == 320.0
    assert state._last_heartbeat_milestone == 1

    # 3. Repeated tick at 320s ago: no duplicate
    state.entry_time = (now - timedelta(seconds=320)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=322.0)
    assert len(dispatched) == 1

    # 4. Advance to 605s ago: fires Milestone 2
    state.entry_time = (now - timedelta(seconds=605)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=330.0)
    assert len(dispatched) == 2
    assert dispatched[1]["milestone"] == 2

    # 5. Clean teardown
    monitor.remove_position("LIVE-001")
    assert "LIVE-001" not in monitor._positions
    assert "LIVE-001" not in monitor._heartbeat_milestones


def test_heartbeat_live_credit_spread_position_monitor():
    """Verify live PositionMonitor emits 5-minute heartbeat for credit spread positions."""
    dispatched = []

    def on_heartbeat_cb(state, current_price, elapsed_sec, milestone, health_report=None):
        dispatched.append({
            "state": state,
            "current_price": current_price,
            "elapsed_sec": elapsed_sec,
            "milestone": milestone,
            "health_report": health_report,
        })

    mock_broker = MagicMock()
    monitor = PositionMonitor(
        broker=mock_broker,
        poll_interval=1,
        on_heartbeat=on_heartbeat_cb,
    )

    now = datetime.now()
    entry_time_str = (now - timedelta(seconds=290)).strftime("%Y-%m-%d %H:%M:%S")

    state = TrailingState(
        position_id="CS-LIVE-001",
        tradingsymbol="NIFTY26OCT24000CE/NIFTY26OCT24100CE",
        symbol="NIFTY",
        entry_premium=40.0,
        initial_sl=60.0,
        current_sl=60.0,
        target=12.0,
        direction="neutral_range",
        strategy="credit_spread",
        strategy_type="credit_spread",
        entry_time=entry_time_str,
    )
    monitor.add_position(state)

    # 1. At 290s: should NOT fire
    monitor._process_tick(state, current_price=35.0)
    assert len(dispatched) == 0

    # 2. Advance state entry_time to 305s ago: fires Milestone 1
    state.entry_time = (now - timedelta(seconds=305)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=35.0)
    assert len(dispatched) == 1
    assert dispatched[0]["milestone"] == 1
    assert dispatched[0]["elapsed_sec"] >= 300
    assert dispatched[0]["current_price"] == 35.0
    assert state._last_heartbeat_milestone == 1

    # 3. Repeated tick at 320s ago: deduplicated (no extra alert)
    state.entry_time = (now - timedelta(seconds=320)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=34.0)
    assert len(dispatched) == 1

    # 4. Advance to 605s ago with breakeven decay reached (e.g. LTP <= 20.0): fires Milestone 2
    state.entry_time = (now - timedelta(seconds=605)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=18.0)
    assert len(dispatched) == 2
    assert dispatched[1]["milestone"] == 2
    assert dispatched[1]["current_price"] == 18.0
    assert state.breakeven_set is True

    # 5. Clean teardown
    monitor.remove_position("CS-LIVE-001")
    assert "CS-LIVE-001" not in monitor._positions
    assert "CS-LIVE-001" not in monitor._heartbeat_milestones


def test_heartbeat_credit_spread_formatting():
    """Verify credit spread positions display both legs in Kite search box and decay P&L."""
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message = MagicMock(return_value=True)

    spread_payload = {
        "trade_id": "SPREAD-001",
        "symbol": "NIFTY 50",
        "instrument": "NIFTY26JUL24200PE/NIFTY26JUL24050PE",
        "direction": "Credit Spread: BULL PUT SPREAD",
        "current_price": 18.50,
        "entry_price": 32.00,
        "gross_pnl_pts": 13.50,
        "gross_pnl": 1012.50,
        "net_pnl": 955.50,
        "net_pnl_pts": 12.74,
        "stop_loss": 48.00,
        "target": 9.60,
        "trailing_stage": "INITIAL",
        "holding_duration": "15 minutes",
        "health_score": 28.0,
        "health_summary": "Score: +28/100 [HEALTHY] | Momentum healthy, moving toward target",
        "quantity": 75,
        "async_send": False,
    }

    result = bot.alert_trade_update(spread_payload)
    assert result is True
    bot.send_message.assert_called_once()

    msg = bot.send_message.call_args[0][0]

    assert "Credit Spread: BULL PUT SPREAD" in msg
    assert "Leg 1 (Short):" in msg
    assert "Leg 2 (Hedge):" in msg
    assert "Current Spread LTP:" in msg
    assert "Rs 18.50" in msg
    assert "Net Credit:" in msg
    assert "Rs 32.00" in msg
    assert "+42.2% decay" in msg
    assert "+13.5 pts" in msg
    assert "+1,012.50 Rs" in msg
    assert "15 minutes" in msg


def test_health_score_summary_variants():
    """Verify format_health_summary outputs appropriate diagnostics across regimes."""
    # 1. Parabolic runner (+75 score)
    runner_report = PositionHealthReport(
        position_id="P1",
        symbol="NIFTY",
        tradingsymbol="NIFTY24500CE",
        direction="bullish",
        current_price=180.0,
        entry_price=120.0,
        health_score=75.0,
        real_favor=True,
        favor_reasons=["High-velocity institutional volume surge", "Dealer short-gamma squeeze"],
    )
    runner_text = format_health_summary(runner_report)
    assert "+75" in runner_text
    assert "healthy" in runner_text.lower() or "runner" in runner_text.lower()

    # 2. Structural threat (-55 score)
    threat_report = PositionHealthReport(
        position_id="P2",
        symbol="NIFTY",
        tradingsymbol="NIFTY24500CE",
        direction="bullish",
        current_price=105.0,
        entry_price=120.0,
        health_score=-55.0,
        real_threat=True,
        threat_reasons=["Severe session VWAP breakdown", "Volume exhaustion"],
    )
    threat_text = format_health_summary(threat_report)
    assert "-55" in threat_text
    assert "warning" in threat_text.lower() or "critical" in threat_text.lower()

    # 3. Flat chop / consolidation (0 score)
    neutral_report = PositionHealthReport(
        position_id="P3",
        symbol="NIFTY",
        tradingsymbol="NIFTY24500CE",
        direction="bullish",
        current_price=121.0,
        entry_price=120.0,
        health_score=0.0,
    )
    neutral_text = format_health_summary(neutral_report)
    assert "0" in neutral_text
    assert "consolidating" in neutral_text.lower() or "structural bounds" in neutral_text.lower()

    # 4. None / missing report
    none_text = format_health_summary(None)
    assert "N/A" in none_text


def test_live_paper_capture_heartbeat_wiring(tmp_path):
    """Verify LivePaperCapture forwards tracker heartbeats to TelegramBot.alert_trade_update."""
    mock_tg = MockTelegramTracker()
    now = datetime.now(IST)

    config = CaptureConfig(
        enabled=True,
        sqlite_path=str(tmp_path / "test_ledger.sqlite"),
        csv_path=str(tmp_path / "test_ledger.csv"),
        enable_trailing=True,
    )
    feed_dict = {"NIFTY26OCT24500CE": 160.0}
    mock_feed = MockPriceFeed(feed_dict)

    capture = LivePaperCapture(
        config=config,
        ltp_source=mock_feed,
        telegram=mock_tg,
    )

    pos = Position(
        trade_id="BRIDGE-001",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT24500CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=75,
        entry_price=135.0,
        stop_loss=115.0,
        target=175.0,
        max_bars=10,
        entry_time=now - timedelta(seconds=305),
    )
    capture._engine.tracker.open_position(pos)

    # Trigger heartbeat check via tracker
    fired = capture._engine.tracker.check_heartbeat(pos)
    assert fired is True
    assert len(mock_tg.updates) == 1

    update = mock_tg.updates[0]
    assert update["trade_id"] == "BRIDGE-001"
    assert update["symbol"] == "NIFTY 50"
    assert update["instrument"] == "NIFTY26OCT24500CE"
    assert update["entry_price"] == 135.0
    assert update["current_price"] == 160.0
    assert update["gross_pnl_pts"] == 25.0
    assert update["gross_pnl"] == 1875.0
    assert update["holding_duration"] == "5 minutes"
    assert update["milestone"] == 1
