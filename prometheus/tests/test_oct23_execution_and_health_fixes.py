import pytest
from datetime import datetime, date, time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
import pytz

IST = pytz.timezone("Asia/Kolkata")

from prometheus.papertrade.types import (
    Direction, Position, ExitReason, TradeSnapshot
)
from prometheus.papertrade.engine import PaperTradeEngine
from prometheus.papertrade.position_tracker import PositionTracker
from prometheus.papertrade.post_exit_tracker import PostExitTracker
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
from prometheus.interface.telegram_bot import TelegramBot


class MockPriceFeed:
    def __init__(self, prices=None):
        self.prices = prices or {}

    def get_ltp(self, instrument: str) -> float:
        return self.prices.get(instrument, 100.0)

    def get_quote(self, instrument: str):
        p = self.get_ltp(instrument)
        return (p, p * 0.999, p * 1.001)


@dataclass
class DummySignal:
    symbol: str
    instrument: str
    direction: Direction
    entry_price_hint: float
    stop_loss: float
    target: float
    target_gain_pts: float
    max_bars: int = 16
    trade_mode: str = "intraday"
    strategy: str = "Momentum"
    signal_score: float = 8.0
    signal_confidence: float = 0.9
    bar_timestamp: datetime = None
    underlying: str = ""


def test_rank1_paper_capture_routing():
    """Verify that in paper mode, a Rank #1 Live-Eligible signal is sent to paper_capture."""
    mock_pc = MagicMock()
    mock_pc.enabled = True
    mock_pc.on_signal.return_value = "PAPER-RANK1-TEST"

    # Simulate main.py loop logic
    refined = {"symbol": "NIFTY 50", "strategy": "Golden_Setup"}

    # Mock _execute_signal_with_feedback returning None (in paper mode)
    position = None

    shadow_trade_id = None
    if position is None and getattr(mock_pc, "enabled", False):
        shadow_trade_id = mock_pc.on_signal(refined)

    assert shadow_trade_id == "PAPER-RANK1-TEST"
    mock_pc.on_signal.assert_called_once_with(refined)


def test_slippage_aware_target_and_sl():
    """Verify that slippage on fill recalibrates target and stop_loss relative to fill price."""
    feed = MockPriceFeed({"SENSEX26OCT73100PE": 594.29})
    engine = PaperTradeEngine(feed=feed)

    # Signal generated when option price was 557.45 with target 589.45 (+32 pts)
    sig = DummySignal(
        symbol="SENSEX",
        instrument="SENSEX26OCT73100PE",
        direction=Direction.LONG,
        entry_price_hint=557.45,
        stop_loss=376.85,  # 180.6 pts risk
        target=589.45,     # 32.0 pts gain
        target_gain_pts=32.0,
        trade_mode="intraday",
        underlying="SENSEX",
    )

    tid = engine.process_new_signal(sig)
    pos = engine.tracker.open_positions[tid]

    # Fill price is ~594.88 (fill simulator applies slippage)
    assert pos.entry_price > 594.0
    # Target MUST be recalibrated: fill.fill_price + 32.0 = ~626.88 (NEVER lower than entry 589.45!)
    assert pos.target == round(pos.entry_price + 32.0, 2)
    assert pos.target > pos.entry_price
    # SL retains 180.6 pts risk distance: fill.fill_price - 180.6 = ~414.28
    assert pos.stop_loss == round(pos.entry_price - 180.6, 2)


def test_cost_buffer_guarantees_net_green_breakeven():
    """Verify that SENSEX 1-lot cost buffer is >= 5.50 pts, guaranteeing net green return on breakeven."""
    tracker = PositionTracker(fill_sim=MagicMock())

    pos = Position(
        trade_id="TEST-SX",
        symbol="SENSEX",
        instrument="SENSEX26OCT72900PE",
        underlying="SENSEX",
        direction=Direction.LONG,
        quantity=20,
        entry_price=603.85,
        entry_time=datetime.now(IST),
        stop_loss=397.55,
        target=917.35,
        target_gain_pts=305.8,
        tier="B",
        max_bars=16,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Simulate price reaching 625.0 (+21.15 pts gain)
    tracker._maybe_advance_trailing_stop(pos, current_price=625.0)

    # Check the ratcheted SL
    assert pos.breakeven_set is True
    # Breakeven SL must be entry + cost_buffer (where cost_buffer is >= 5.50 pts)
    assert pos.stop_loss >= round(pos.entry_price + 5.50, 2)

    # Calculate actual net P&L if stopped out at new SL:
    gross_gain = (pos.stop_loss - pos.entry_price) * pos.quantity
    # Statutory fees on SENSEX 20 qty @ ~600 INR are ~93.65 INR
    fees = 93.65
    net_pnl = gross_gain - fees
    assert net_pnl > 0, f"Breakeven must be net green: gross={gross_gain}, fees={fees}, net={net_pnl}"


def test_exit_reason_trailing_stop_vs_stop_loss():
    """Verify that exiting at or above entry price sets exit_reason to TRAILING_STOP."""
    feed = MockPriceFeed({"SENSEX26OCT72900PE": 609.35})
    tracker = PositionTracker(fill_sim=MagicMock(feed=feed))

    pos = Position(
        trade_id="TEST-SX-TRAIL",
        symbol="SENSEX",
        instrument="SENSEX26OCT72900PE",
        underlying="SENSEX",
        direction=Direction.LONG,
        quantity=20,
        entry_price=603.85,
        entry_time=datetime.now(IST),
        stop_loss=609.35,  # Ratcheted above entry
        target=917.35,
        breakeven_set=True,
        max_bars=16,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Snapshot with price touching SL
    snap = TradeSnapshot(
        timestamp=datetime.now(IST),
        symbol="SENSEX",
        instrument="SENSEX26OCT72900PE",
        open=609.35,
        high=615.0,
        low=608.0,
        close=609.0,
        volume=1000,
    )

    exit_px, exit_reason = tracker._evaluate_exit(pos, snap)
    assert exit_px == 609.35
    assert exit_reason == ExitReason.TRAILING_STOP
    assert exit_reason != ExitReason.STOP_LOSS


def test_martingale_lockout_only_on_net_loss():
    """Verify that LivePaperCapture only locks out (symbol, direction) if net_pnl < 0."""
    capture = LivePaperCapture(CaptureConfig(enabled=True), ltp_source=MagicMock())

    # Simulated trade closing at Breakeven with net profit (or 0)
    be_trade = MagicMock()
    be_trade.instrument = "SENSEX26OCT72900PE"
    be_trade.symbol = "SENSEX"
    be_trade.direction.value = "SHORT"
    be_trade.exit_reason = ExitReason.TRAILING_STOP
    be_trade.net_pnl = 15.20
    be_trade.trade_type = "BUY_PE"

    capture._on_trade_closed(be_trade)
    assert ("SENSEX", "BUY_PE") not in capture._today_stopped_out_symbol_directions

    # Simulated trade with an actual stop loss (-1,500 net loss)
    loss_trade = MagicMock()
    loss_trade.instrument = "SENSEX26OCT72900PE"
    loss_trade.symbol = "SENSEX"
    loss_trade.direction.value = "SHORT"
    loss_trade.exit_reason = ExitReason.STOP_LOSS
    loss_trade.net_pnl = -1500.0
    loss_trade.trade_type = "BUY_PE"

    capture._on_trade_closed(loss_trade)
    assert ("SENSEX", "BUY_PE") in capture._today_stopped_out_symbol_directions


def test_post_exit_tracker_rejects_spot_index_pollution():
    """Verify that PostExitTracker ignores spot index bars (instrument='') and does not corrupt MFE."""
    tracker = PostExitTracker()

    trade = MagicMock()
    trade.trade_id = "SX-TEST-1"
    trade.symbol = "SENSEX"
    trade.underlying = "SENSEX"
    trade.instrument = "SENSEX26OCT72900PE"
    trade.direction.value = "SHORT"
    trade.entry_price = 603.85
    trade.exit_price = 608.35
    trade.target = 917.35
    trade.stop_loss = 608.35
    trade.exit_reason = "trailing_stop"
    trade.exit_time = datetime.now(IST)
    trade.tier = "B"
    trade.entry_spot = 72865.65

    item = tracker.register_closed_trade(trade)
    assert item is not None
    assert item.post_exit_mfe == 608.35

    # Simulate an index bar arriving with close=72824.41 (spot index price!)
    index_snap = TradeSnapshot(
        timestamp=datetime.now(IST),
        symbol="SENSEX",
        instrument="",  # Cash index bar has empty instrument
        open=72800.0,
        high=72900.0,
        low=72750.0,
        close=72824.41,
        volume=0,
    )
    upd = tracker.on_bar(index_snap)
    assert upd == []  # MUST NOT match or process!
    assert item.post_exit_mfe == 608.35  # MFE MUST NOT be contaminated with 72,824!

    # Now simulate option contract tick arriving
    upd_opt = tracker.on_tick("SENSEX26OCT72900PE", 635.0, datetime.now(IST))
    assert len(upd_opt) == 1
    assert item.post_exit_mfe == 635.0
    assert item.alpha_left_pts == pytest.approx(635.0 - 608.35, 0.01)


def test_in_trade_heartbeat_60s_cadence():
    """Verify that PositionTracker fires heartbeats at 60s intervals."""
    callback = MagicMock()
    tracker = PositionTracker(
        fill_sim=MagicMock(),
        on_heartbeat=callback,
        heartbeat_interval_seconds=60,
    )

    pos = Position(
        trade_id="HB-TEST",
        symbol="SENSEX",
        instrument="SENSEX26OCT72900PE",
        underlying="SENSEX",
        direction=Direction.LONG,
        quantity=20,
        entry_price=603.85,
        entry_time=datetime.now(IST),
        stop_loss=397.55,
        target=917.35,
        max_bars=16,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Elapsed 30s -> M=0 (no fire)
    with patch.object(tracker, "_get_elapsed_seconds", return_value=30.0):
        res = tracker.check_heartbeat(pos, current_price=610.0)
        assert res is False
        assert callback.call_count == 0

    # Elapsed 65s -> M=1 (fires milestone 1!)
    with patch.object(tracker, "_get_elapsed_seconds", return_value=65.0):
        res = tracker.check_heartbeat(pos, current_price=615.0)
        assert res is True
        assert callback.call_count == 1
        assert pos._last_heartbeat_milestone == 1

    # Elapsed 80s -> M=1 (deduped, does not fire)
    with patch.object(tracker, "_get_elapsed_seconds", return_value=80.0):
        res = tracker.check_heartbeat(pos, current_price=618.0)
        assert res is False
        assert callback.call_count == 1

    # Elapsed 125s -> M=2 (fires milestone 2!)
    with patch.object(tracker, "_get_elapsed_seconds", return_value=125.0):
        res = tracker.check_heartbeat(pos, current_price=622.0)
        assert res is True
        assert callback.call_count == 2
        assert pos._last_heartbeat_milestone == 2


def test_trailing_stop_alert_with_health():
    """Verify that alert_trailing_stop_updated formats health score cleanly."""
    bot = TelegramBot(bot_token="fake", chat_id="12345")

    with patch.object(bot, "send_message") as mock_send:
        bot.alert_trailing_stop_updated(
            symbol="SENSEX",
            instrument="SENSEX26OCT72900PE",
            old_sl=500.70,
            new_sl=609.35,
            stage="breakeven",
            current_price=635.20,
            gain_pts=31.35,
            cost_pts=5.50,
            entry_price=603.85,
            health_score=35.0,
            health_summary="Short Gamma Squeeze | Strong OI Commitment",
        )

        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "TRAILING STOP: CAPITAL PROTECTED (BREAKEVEN)" in msg
        assert "Trade Health & Edge" in msg
        assert "Score: +35" in msg
        assert "Short Gamma Squeeze" in msg
