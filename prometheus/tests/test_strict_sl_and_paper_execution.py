"""
Unit tests for Strict Stop-Loss Enforcement and Real-Time Paper Execution.

Verifies:
1. PositionMonitor immediately exits on stop loss from tick 1 / bar 1 (zero lag, no Phase 1/Phase 2 immunity).
2. PositionTracker evaluates option stop-loss / target on every bar when receiving underlying index bars.
"""

from datetime import datetime
from unittest.mock import MagicMock
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.papertrade.position_tracker import (
    PositionTracker,
    Position,
    Direction,
    CostModel,
    ExitReason,
)
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.papertrade.types import TradeSnapshot


class DummyFeed:
    def __init__(self, quotes=None):
        self.quotes = quotes or {}

    def get_ltp(self, instrument: str) -> float:
        return float(self.quotes.get(instrument, 0.0))


def test_position_monitor_strict_sl_bar_1():
    """Verify PositionMonitor exits IMMEDIATELY on bar 1 if LTP touches initial_sl (no Phase 1 immunity)."""
    exits = []
    def _on_exit(pos_id, price, reason):
        exits.append((pos_id, price, reason))

    monitor = PositionMonitor(broker=MagicMock(), on_exit=_on_exit)

    state = TrailingState(
        position_id="POS-TEST-SL-01",
        symbol="NIFTY 50",
        tradingsymbol="NIFTY26SEP24000CE",
        entry_premium=100.0,
        initial_sl=85.0,
        current_sl=85.0,
        target=130.0,
        direction="bullish",
        entry_bar_count=1,  # Bar 1 (previously Phase 1 had 80% loss floor immunity)
    )

    # LTP drops to 84.0 (breaching initial SL of 85.0)
    monitor._process_tick(state, 84.0)

    assert len(exits) == 1, "Exit must fire immediately on bar 1 without Phase 1 immunity"
    pos_id, price, reason = exits[0]
    assert pos_id == "POS-TEST-SL-01"
    assert price == 84.0
    assert reason == "stop_loss_hit"


def test_position_tracker_evaluates_sl_on_underlying_bar():
    """Verify PositionTracker checks option LTP feed and triggers SL immediately on underlying bar."""
    feed = DummyFeed({"NIFTY26SEP24000CE": 84.0})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    tracker = PositionTracker(fill_sim=sim, cost_model=CostModel(cost_per_side_bps=0.0), enable_trailing=False)

    pos = Position(
        trade_id="PAPER-TEST-SL-01",
        symbol="NIFTY 50",
        instrument="NIFTY26SEP24000CE",
        underlying="NIFTY 50",
        direction=Direction.LONG,
        quantity=50,
        entry_price=100.0,
        entry_time=datetime(2026, 9, 10, 9, 30),
        stop_loss=85.0,
        target=130.0,
        max_bars=16,
        strategy="PriceAction_Momentum",
        trade_mode="intraday",
    )
    tracker.open_positions[pos.trade_id] = pos

    # An intermediate underlying bar arrives (not session end, not square off)
    snap = TradeSnapshot(
        timestamp=datetime(2026, 9, 10, 9, 45),
        symbol="NIFTY 50",
        instrument="",  # Underlying bar has empty instrument
        open=24800.0,
        high=24850.0,
        low=24780.0,
        close=24790.0,
        volume=10000,
        bar_interval="15minute",
    )

    closed = tracker.on_bar(snap, is_session_end=False, is_square_off=False)

    assert len(closed) == 1, "Position must be stopped out immediately upon receiving intermediate bar"
    trade = closed[0]
    assert trade.trade_id == "PAPER-TEST-SL-01"
    assert trade.exit_reason == ExitReason.STOP_LOSS
    assert trade.exit_price == 85.0
    assert len(tracker.open_positions) == 0
