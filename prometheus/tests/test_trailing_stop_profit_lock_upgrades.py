"""
Unit tests for Trailing Stop & Profit Lock Upgrades (Credit Spread Ratchet + Option Buying Dead-Zone Bridge).

Verifies:
1. Credit Spread 2-Stage Percentage Decay Ratchet (PositionTracker & PositionMonitor):
   - Stage 1: Breakeven lock at >= 30% decay (SL = Entry - Cost Buffer).
   - Stage 2: 50% profit lock at >= 45% decay (SL = Entry * 0.50).
   - Exit execution on rebound crosses above ratcheted SL, booking guaranteed profit.
2. Option Buying Stage 0.5 Profit Lock (PositionTracker & PositionMonitor):
   - Stage 0.5: Locks 0.35R profit (+14 pts / +Rs 910) at >= 0.65R gain (+26 pts).
   - Bridges the historical 0.4R to 1.0R dead-zone that caused Oct 15 runner giveback.
   - Stage 1 (1.0R -> 0.50R lock), Stage 2 (1.75R -> 0.65R lock), Stage 3 (2.5R -> 0.75R lock).
3. Telegram Notification & Alert Formatting:
   - Emits structured alerts with custom headlines for spread locks and Stage 0.5.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from prometheus.utils.indian_market import IST
from prometheus.papertrade.types import (
    Position, Direction, ExitReason, TradeSnapshot,
)
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.execution.position_monitor import PositionMonitor, TrailingState


class MockFeed:
    """Mock price feed for FillSimulator."""
    def __init__(self, ltp_dict=None):
        self.ltp_dict = ltp_dict or {}

    def get_ltp(self, symbol, instrument=None):
        key = instrument or symbol
        return self.ltp_dict.get(key, 100.0)


class MockTelegram:
    def __init__(self):
        self.alerts = []

    def alert_trailing_stop_updated(self, symbol, instrument, old_sl, new_sl, stage, current_price, gain_pts, cost_pts, entry_price=0.0):
        self.alerts.append({
            "symbol": symbol,
            "instrument": instrument,
            "old_sl": old_sl,
            "new_sl": new_sl,
            "stage": stage,
            "current_price": current_price,
            "gain_pts": gain_pts,
            "cost_pts": cost_pts,
            "entry_price": entry_price,
        })

    def send_message(self, text):
        pass


# ==============================================================================
# 1. Credit Spread 2-Stage Percentage Decay Ratchet Tests (Paper Engine)
# ==============================================================================

def test_credit_spread_stage1_breakeven_lock_at_30pct_decay():
    """Verify Credit Spread locks breakeven (entry - cost buffer) at >= 30% decay."""
    mock_tg = MockTelegram()

    def on_sl_update(pos, old_sl, new_sl, stage, current_price, decay_pts, cost_buffer_pts):
        mock_tg.alert_trailing_stop_updated(
            pos.symbol, pos.instrument, old_sl, new_sl, stage, current_price, decay_pts, cost_buffer_pts, pos.entry_price
        )

    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
        on_sl_update=on_sl_update,
    )

    # Trade matching Oct 15 FIN NIFTY credit spread (Entry=71.71, Initial SL=106.80)
    pos = Position(
        trade_id="SPREAD-OCT15-01",
        symbol="NIFTY FIN SERVICE",
        instrument="FINNIFTY27OCT2625350CE/FINNIFTY27OCT2625500CE",
        underlying="FINNIFTY",
        direction=Direction.SHORT,
        quantity=65,
        entry_price=71.71,
        entry_time=datetime.now(IST),
        stop_loss=106.80,
        target=21.36,
        max_bars=16,
        strategy="Hedged_Credit_Spread",
        tier="C",
    )
    tracker.open_position(pos)

    # 1. Minor decay to 55.00 (23.3% decay < 30% threshold) -> No ratchet
    tracker._maybe_advance_trailing_stop(pos, 55.00)
    assert pos.stop_loss == 106.80
    assert pos.breakeven_set is False
    assert len(mock_tg.alerts) == 0

    # 2. Favorable decay to 50.00 (decay = 71.71 - 50.00 = 21.71 pts = 30.28% >= 30%)
    # Cost buffer for FINNIFTY is 1.80 pts
    # Expected new_sl = min(71.71 - 1.80, 71.71 * 0.90) = min(69.91, 64.54) = 64.54
    tracker._maybe_advance_trailing_stop(pos, 50.00)
    assert pos.breakeven_set is True
    assert pos.stop_loss < 71.71
    assert pos.stop_loss == pytest.approx(64.54, 0.01)

    # Verify Telegram alert
    assert len(mock_tg.alerts) == 1
    assert mock_tg.alerts[0]["stage"] == "spread_breakeven"
    assert mock_tg.alerts[0]["new_sl"] == pytest.approx(64.54, 0.01)


def test_credit_spread_stage2_50pct_profit_lock_at_45pct_decay():
    """Verify Credit Spread locks 50% of credit at >= 45% decay."""
    mock_tg = MockTelegram()

    def on_sl_update(pos, old_sl, new_sl, stage, current_price, decay_pts, cost_buffer_pts):
        mock_tg.alert_trailing_stop_updated(
            pos.symbol, pos.instrument, old_sl, new_sl, stage, current_price, decay_pts, cost_buffer_pts, pos.entry_price
        )

    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
        on_sl_update=on_sl_update,
    )

    pos = Position(
        trade_id="SPREAD-OCT15-02",
        symbol="NIFTY FIN SERVICE",
        instrument="FINNIFTY27OCT2625350CE/FINNIFTY27OCT2625500CE",
        underlying="FINNIFTY",
        direction=Direction.SHORT,
        quantity=65,
        entry_price=71.71,
        entry_time=datetime.now(IST),
        stop_loss=106.80,
        target=21.36,
        max_bars=16,
        strategy="Hedged_Credit_Spread",
        tier="C",
    )
    tracker.open_position(pos)

    # First trigger Stage 1 at 50.00 (30% decay)
    tracker._maybe_advance_trailing_stop(pos, 50.00)
    assert pos.breakeven_set is True

    # Advance to 38.00 (decay = 71.71 - 38.00 = 33.71 pts = 47.0% >= 45%)
    # Expected new_sl = 71.71 * 0.50 = 35.86 pts
    tracker._maybe_advance_trailing_stop(pos, 38.00)
    assert pos.trailing_floor == 0.50
    assert pos.stop_loss == pytest.approx(35.86, 0.01)

    # Verify Stage 2 alert emitted
    assert len(mock_tg.alerts) == 2
    assert mock_tg.alerts[1]["stage"] == "spread_lock_50pct"
    assert mock_tg.alerts[1]["new_sl"] == pytest.approx(35.86, 0.01)


def test_credit_spread_rebound_exits_at_ratcheted_sl_with_profit():
    """Verify that when spread price rebounds after 45% decay, it stops out at ratcheted SL with net profit."""
    feed = MockFeed()
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=feed, slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
    )

    pos = Position(
        trade_id="SPREAD-REBOUND-01",
        symbol="NIFTY FIN SERVICE",
        instrument="FINNIFTY27OCT2625350CE/FINNIFTY27OCT2625500CE",
        underlying="FINNIFTY",
        direction=Direction.SHORT,
        quantity=65,
        entry_price=71.71,
        entry_time=datetime.now(IST),
        stop_loss=106.80,
        target=21.36,
        max_bars=16,
        strategy="Hedged_Credit_Spread",
        tier="C",
    )
    tracker.open_position(pos)

    # Spread decays to 35.00 -> locks 50% profit (SL moves to 35.86)
    tracker._maybe_advance_trailing_stop(pos, 35.00)
    assert pos.stop_loss == pytest.approx(35.86, 0.01)

    # Rebound bar: spread jumps up to 40.00 (crosses above ratcheted SL 35.86)
    # Simulate on_bar exit check
    feed.ltp_dict["FINNIFTY27OCT2625350CE/FINNIFTY27OCT2625500CE"] = 40.00
    snap = TradeSnapshot(
        symbol="NIFTY FIN SERVICE",
        timestamp=datetime.now(IST),
        open=25000.0, high=25100.0, low=24950.0, close=25050.0, volume=1000,
        instrument=""
    )
    closed = tracker.on_bar(snap)

    assert len(closed) == 1
    trade = closed[0]
    assert trade.exit_reason == ExitReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(40.00, 0.01)
    # Realized gross PnL = (Entry - Exit) * Qty = (71.71 - 40.00) * 65 = +Rs 2,061.15
    assert trade.gross_pnl > 0
    assert trade.net_pnl > 0


# ==============================================================================
# 2. Option Buying Stage 0.5 Dead-Zone Bridge Tests (Paper Engine)
# ==============================================================================

def test_option_buying_stage0_5_bridges_dead_zone():
    """Verify Option Buying locks 0.35R profit at >= 0.65R gain, bridging the 0.4R to 1.0R dead-zone."""
    mock_tg = MockTelegram()

    def on_sl_update(pos, old_sl, new_sl, stage, current_price, gain_pts, cost_buffer_pts):
        mock_tg.alert_trailing_stop_updated(
            pos.symbol, pos.instrument, old_sl, new_sl, stage, current_price, gain_pts, cost_buffer_pts, pos.entry_price
        )

    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
        on_sl_update=on_sl_update,
    )

    # Trade matching Oct 15 NIFTY option buy (Entry=124.27, SL=84.27, Risk distance R=40.0)
    pos = Position(
        trade_id="BUY-OCT15-01",
        symbol="NIFTY 50",
        instrument="NIFTY19OCT2623000CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=124.27,
        entry_time=datetime.now(IST),
        stop_loss=84.27,
        target=194.90,
        max_bars=16,
        strategy="Golden_Setup",
        tier="B",
    )
    tracker.open_position(pos)

    # Step 1: Rally to 141.00 (+16.73 pts = 0.418R >= 0.40R, and >= 2.0 min_be_gain for NIFTY)
    # Trigger Breakeven: SL = Entry + 1.20 = 125.47
    tracker._maybe_advance_trailing_stop(pos, 141.00)
    assert pos.breakeven_set is True
    assert pos.stop_loss == pytest.approx(125.47, 0.01)
    assert len(mock_tg.alerts) == 1
    assert mock_tg.alerts[0]["stage"] == "breakeven"

    # Step 2: Price advances to 151.00 (+26.73 pts = 0.668R >= 0.65R)
    # Stage 0.5 MUST trigger! Lock 0.35R profit:
    # Expected new_sl = 124.27 + 0.35 * 40.0 = 138.27 (+14.0 pts profit locked!)
    tracker._maybe_advance_trailing_stop(pos, 151.00)
    assert pos.trailing_floor == 0.35
    assert pos.stop_loss == pytest.approx(138.27, 0.01)

    assert len(mock_tg.alerts) == 2
    assert mock_tg.alerts[1]["stage"] == "lock_35pct"
    assert mock_tg.alerts[1]["new_sl"] == pytest.approx(138.27, 0.01)

    # Step 3: Pullback to 135.00 stops out at 138.27 with +14.00 pts profit (+Rs 910 gross)
    # Instead of collapsing back to 125.47 (breakeven -Rs 4.21 net)!


def test_option_buying_multistage_progression():
    """Verify Option Buying full stage progression: Breakeven -> 0.35R -> 0.50R -> 0.65R -> 0.75R."""
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
    )

    pos = Position(
        trade_id="BUY-RUNNER-01",
        symbol="NIFTY 50",
        instrument="NIFTY19OCT2623000CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(IST),
        stop_loss=70.0,  # Risk R = 30.0
        target=200.0,
        max_bars=16,
        strategy="Golden_Setup",
        tier="B",
    )
    tracker.open_position(pos)

    # 1. Gain +13.0 pts (0.43R) -> Breakeven (100.0 + 1.20 = 101.20)
    tracker._maybe_advance_trailing_stop(pos, 113.0)
    assert pos.breakeven_set is True
    assert pos.stop_loss == pytest.approx(101.20, 0.01)

    # 2. Gain +20.0 pts (0.667R >= 0.65R) -> Stage 0.5 lock 0.35R (100 + 0.35*30 = 110.50)
    tracker._maybe_advance_trailing_stop(pos, 120.0)
    assert pos.trailing_floor == 0.35
    assert pos.stop_loss == pytest.approx(110.50, 0.01)

    # 3. Gain +32.0 pts (1.067R >= 1.0R) -> Stage 1 lock 0.50R (100 + 0.50*30 = 115.00)
    tracker._maybe_advance_trailing_stop(pos, 132.0)
    assert pos.trailing_floor == 0.50
    assert pos.stop_loss == pytest.approx(115.00, 0.01)

    # 4. Gain +54.0 pts (1.80R >= 1.75R) -> Stage 2 lock 0.65R (100 + 0.65*30 = 119.50)
    tracker._maybe_advance_trailing_stop(pos, 154.0)
    assert pos.trailing_floor == 0.65
    assert pos.stop_loss == pytest.approx(119.50, 0.01)

    # 5. Gain +76.0 pts (2.53R >= 2.5R) -> Stage 3 lock 0.75R (100 + 0.75*30 = 122.50)
    tracker._maybe_advance_trailing_stop(pos, 176.0)
    assert pos.trailing_floor == 0.75
    assert pos.stop_loss == pytest.approx(122.50, 0.01)


# ==============================================================================
# 3. Live PositionMonitor Parity Tests
# ==============================================================================

def test_position_monitor_credit_spread_parity():
    """Verify PositionMonitor mirrors credit spread 30% and 45% decay ratchets."""
    monitor = PositionMonitor(broker=MagicMock())

    state = TrailingState(
        position_id="LIVE-SPREAD-01",
        symbol="NIFTY FIN SERVICE",
        tradingsymbol="FINNIFTY27OCT2625350CE/FINNIFTY27OCT2625500CE",
        entry_premium=71.71,
        initial_sl=106.80,
        current_sl=106.80,
        target=21.36,
        direction="bearish",
        strategy_type="credit_spread",
        quantity=65,
    )

    # At 50.00 (30% decay) -> Breakeven Lock (new_sl = min(71.71 - 1.80, 71.71 * 0.85) = 60.95)
    monitor._process_tick(state, 50.00)
    assert state.breakeven_set is True
    assert state.current_sl < 71.71
    assert state.current_stage() == "SPREAD BREAKEVEN"

    # At 38.00 (45% decay) -> 50% Profit Lock (new_sl = 71.71 * 0.50 = 35.86)
    monitor._process_tick(state, 38.00)
    assert state.trailing_stage2 is True
    assert state.current_sl == pytest.approx(35.86, 0.01)
    assert state.current_stage() == "SPREAD LOCK 50%"


def test_position_monitor_option_buying_stage0_5_parity():
    """Verify PositionMonitor bridges the dead zone at >= 0.65R gain."""
    monitor = PositionMonitor(broker=MagicMock())

    state = TrailingState(
        position_id="LIVE-BUY-01",
        symbol="NIFTY 50",
        tradingsymbol="NIFTY19OCT2623000CE",
        entry_premium=100.0,
        initial_sl=70.0,  # Risk R = 30.0
        current_sl=70.0,
        target=200.0,
        direction="bullish",
        trade_mode="intraday",
        quantity=65,
    )

    # 1. Gain +13.0 pts -> Breakeven
    monitor._process_tick(state, 113.0)
    assert state.breakeven_set is True
    assert state.current_stage() == "BREAKEVEN"

    # 2. Gain +20.0 pts (0.667R >= 0.65R) -> Stage 0.5 Lock 0.35R (100 + 0.35*30 = 110.50)
    monitor._process_tick(state, 120.0)
    assert state.trailing_stage0_5 is True
    assert state.current_sl == pytest.approx(110.50, 0.01)
    assert state.current_stage() == "LOCK 35%"
