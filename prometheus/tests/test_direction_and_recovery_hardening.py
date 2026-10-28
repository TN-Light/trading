"""
Unit & Regression Tests: Direction Normalization, Crash Recovery Side Invariant, and Spread Fill Invariant Hardening.

Covers:
1. types.py: Direction.from_signal_direction maps 'buy_pe' to Direction.SHORT (not LONG).
2. order_manager.py: Crash recovery assigns OrderSide.BUY for PE option buying (not SELL),
   preventing duplicate buying on exit and PnL inversion. Credit spreads assign OrderSide.SELL.
3. position_monitor.py: TrailingState.is_bullish correctly recognizes 'LONG', 'bullish',
   'buy_ce', and '...CE' tokens as bullish, and 'SHORT', 'bearish', 'buy_pe', '...PE' as bearish.
4. engine.py & position_tracker.py: FillSimulator receives side='SELL' on spread entry and
   side='BUY' on spread exit; side='BUY' on option buying entry and side='SELL' on option exit.
"""
import pytest
from unittest.mock import MagicMock, patch

from prometheus.papertrade.types import Direction, ExitReason, Position
from prometheus.papertrade.signal_source import SignalNotification
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.execution.broker import OrderSide, OrderStatus, OrderType, Order
from prometheus.execution.order_manager import OrderManager
from prometheus.execution.position_monitor import TrailingState


def test_direction_from_signal_direction_mapping():
    """Verify Direction.from_signal_direction cleanly separates bullish and bearish variants."""
    # Bullish variants -> Direction.LONG
    assert Direction.from_signal_direction("bullish") == Direction.LONG
    assert Direction.from_signal_direction("buy_ce") == Direction.LONG
    assert Direction.from_signal_direction("long") == Direction.LONG
    assert Direction.from_signal_direction("buy") == Direction.LONG

    # Bearish variants -> Direction.SHORT
    assert Direction.from_signal_direction("buy_pe") == Direction.SHORT
    assert Direction.from_signal_direction("bearish") == Direction.SHORT
    assert Direction.from_signal_direction("short") == Direction.SHORT
    assert Direction.from_signal_direction("sell") == Direction.SHORT
    assert Direction.from_signal_direction("sell_call_spread") == Direction.SHORT
    assert Direction.from_signal_direction("sell_put_spread") == Direction.SHORT


def test_trailing_state_is_bullish_property():
    """Verify TrailingState.is_bullish handles strings, Enums, and tradingsymbol fallbacks."""
    # Bullish cases
    ts1 = TrailingState(
        position_id="P1", tradingsymbol="NIFTY26OCT22500CE", symbol="NIFTY 50",
        entry_premium=100.0, initial_sl=80.0, current_sl=80.0, target=140.0,
        direction="bullish", strategy="trend", entry_time="10:00:00"
    )
    assert ts1.is_bullish is True

    # Direction as "LONG" string
    ts2 = TrailingState(
        position_id="P2", tradingsymbol="NIFTY26OCT22500CE", symbol="NIFTY 50",
        entry_premium=100.0, initial_sl=80.0, current_sl=80.0, target=140.0,
        direction="LONG", strategy="trend", entry_time="10:00:00"
    )
    assert ts2.is_bullish is True

    # Direction as Direction.LONG enum
    ts3 = TrailingState(
        position_id="P3", tradingsymbol="NIFTY26OCT22500CE", symbol="NIFTY 50",
        entry_premium=100.0, initial_sl=80.0, current_sl=80.0, target=140.0,
        direction=Direction.LONG, strategy="trend", entry_time="10:00:00"
    )
    assert ts3.is_bullish is True

    # Bearish PE cases
    ts_pe = TrailingState(
        position_id="P4", tradingsymbol="NIFTY26OCT22500PE", symbol="NIFTY 50",
        entry_premium=100.0, initial_sl=80.0, current_sl=80.0, target=140.0,
        direction="bearish", strategy="trend", entry_time="10:00:00"
    )
    assert ts_pe.is_bullish is False

    ts_pe_short = TrailingState(
        position_id="P5", tradingsymbol="NIFTY26OCT22500PE", symbol="NIFTY 50",
        entry_premium=100.0, initial_sl=80.0, current_sl=80.0, target=140.0,
        direction="SHORT", strategy="trend", entry_time="10:00:00"
    )
    assert ts_pe_short.is_bullish is False

    # Spread cases
    ts_spread_bull = TrailingState(
        position_id="P6", tradingsymbol="NIFTY26OCT22500PE/NIFTY26OCT22400PE", symbol="NIFTY 50",
        entry_premium=25.0, initial_sl=40.0, current_sl=40.0, target=5.0,
        direction="neutral_bullish", strategy="bull_put_spread", entry_time="10:00:00"
    )
    assert ts_spread_bull.is_bullish is True

    ts_spread_bear = TrailingState(
        position_id="P7", tradingsymbol="NIFTY26OCT22500CE/NIFTY26OCT22600CE", symbol="NIFTY 50",
        entry_premium=25.0, initial_sl=40.0, current_sl=40.0, target=5.0,
        direction="neutral_bearish", strategy="bear_call_spread", entry_time="10:00:00"
    )
    assert ts_spread_bear.is_bullish is False


def test_crash_recovery_order_side_for_put_buying():
    """Verify that crash recovery for a Put buying position reconstructs side=OrderSide.BUY.
    On close, it must execute OrderSide.SELL (closing the long put), never OrderSide.BUY.
    """
    mock_broker = MagicMock()
    mock_risk = MagicMock()

    # Broker reports 0 active position or position of 50
    mock_pos = MagicMock()
    mock_pos.tradingsymbol = "NIFTY26OCT22500PE"
    mock_pos.quantity = 50
    mock_broker.get_positions.return_value = [mock_pos]

    om = OrderManager(broker=mock_broker, risk_manager=mock_risk)

    state = TrailingState(
        position_id="POS_PUT_RECOVER_01",
        tradingsymbol="NIFTY26OCT22500PE",
        symbol="NIFTY 50",
        entry_premium=150.0,
        initial_sl=120.0,
        current_sl=120.0,
        target=200.0,
        direction="bearish",  # Bearish put buying
        strategy="trend",
        entry_time="2026-10-28 10:00:00",
    )

    # Simulate exit of crash-recovered position
    mock_broker.place_order.side_effect = lambda ord: Order(
        order_id="EXIT_ORD_99",
        symbol=ord.symbol,
        tradingsymbol=ord.tradingsymbol,
        side=ord.side,
        order_type=ord.order_type,
        quantity=ord.quantity,
        filled_quantity=ord.quantity,
        average_price=180.0,  # profitable exit at 180 (entry 150)
        status=OrderStatus.COMPLETE,
    )

    pnl = om.close_position("POS_PUT_RECOVER_01", reason="target", state=state)

    # 1. Managed position dummy entry must have been OrderSide.BUY
    managed = om.managed_positions["POS_PUT_RECOVER_01"]
    assert managed.entry_orders[0].side == OrderSide.BUY

    # 2. Placed close order must be OrderSide.SELL (selling the contract)
    placed_order = mock_broker.place_order.call_args[0][0]
    assert placed_order.side == OrderSide.SELL

    # 3. PnL must be positive: (180 - 150) * 50 = +1500, NOT negative inverted!
    assert pnl == pytest.approx(1500.0)


def test_crash_recovery_order_side_for_credit_spread():
    """Verify that crash recovery for a credit spread reconstructs side=OrderSide.SELL.
    On close, it must execute OrderSide.BUY (buying back the short spread).
    """
    mock_broker = MagicMock()
    mock_risk = MagicMock()

    mock_pos = MagicMock()
    mock_pos.tradingsymbol = "NIFTY26OCT22500PE/NIFTY26OCT22400PE"
    mock_pos.quantity = -50
    mock_broker.get_positions.return_value = [mock_pos]

    om = OrderManager(broker=mock_broker, risk_manager=mock_risk)

    state = TrailingState(
        position_id="POS_SPREAD_RECOVER_01",
        tradingsymbol="NIFTY26OCT22500PE/NIFTY26OCT22400PE",
        symbol="NIFTY 50",
        entry_premium=30.0,
        initial_sl=50.0,
        current_sl=50.0,
        target=10.0,
        direction="neutral_bullish",
        strategy="bull_put_spread",
        strategy_type="credit_spread",
        entry_time="2026-10-28 10:00:00",
    )

    mock_broker.place_order.side_effect = lambda ord: Order(
        order_id="EXIT_ORD_CS",
        symbol=ord.symbol,
        tradingsymbol=ord.tradingsymbol,
        side=ord.side,
        order_type=ord.order_type,
        quantity=ord.quantity,
        filled_quantity=ord.quantity,
        average_price=10.0,  # profitable decay exit at 10 (entry 30)
        status=OrderStatus.COMPLETE,
    )

    pnl = om.close_position("POS_SPREAD_RECOVER_01", reason="target", state=state)

    # 1. Managed position dummy entry must have been OrderSide.SELL
    managed = om.managed_positions["POS_SPREAD_RECOVER_01"]
    assert managed.entry_orders[0].side == OrderSide.SELL

    # 2. Placed close order must be OrderSide.BUY (buying to close)
    placed_order = mock_broker.place_order.call_args[0][0]
    assert placed_order.side == OrderSide.BUY

    # 3. PnL for credit spread: (entry - exit) * qty = (30 - 10) * 50 = +1000
    assert pnl == pytest.approx(1000.0)


def test_spread_fill_side_in_paper_engine_and_tracker():
    """Verify that FillSimulator receives side='SELL' on spread entry and side='BUY' on spread exit."""
    from prometheus.papertrade.engine import PaperTradeEngine
    from prometheus.papertrade.position_tracker import PositionTracker

    mock_feed = MagicMock()
    mock_feed.get_quote.return_value = (30.0, 28.0, 32.0)
    mock_feed.get_ltp.return_value = 30.0

    engine = PaperTradeEngine(feed=mock_feed)
    tracker = engine.tracker

    # 1. Test Spread Entry in PaperTradeEngine
    spread_sig = SignalNotification(
        symbol="NIFTY 50",
        instrument="NIFTY26OCT22500CE/NIFTY26OCT22600CE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        strike=22500.0,
        option_type="CE",
        expiry="2026-10-29",
        entry_price_hint=30.0,
        stop_loss=50.0,
        target=10.0,
        strategy="credit_spread",
    )

    with patch.object(tracker.fill_sim, "fill", wraps=tracker.fill_sim.fill) as spy_fill:
        mock_feed.get_quote.side_effect = lambda inst: (40.0, 39.0, 41.0) if "22500" in inst else (10.0, 9.5, 10.5)
        trade_id = engine.process_new_signal(spread_sig)
        assert trade_id is not None
        # On spread entry, side must be "SELL"
        call_args = spy_fill.call_args_list[0]
        side_arg = call_args[1].get("side") if "side" in call_args[1] else call_args[0][3]
        assert side_arg == "SELL"

    # 2. Test Spread Exit in PositionTracker
    with patch.object(tracker.fill_sim, "fill", wraps=tracker.fill_sim.fill) as spy_exit_fill:
        mock_feed.get_quote.side_effect = lambda inst: (20.0, 19.0, 21.0) if "22500" in inst else (5.0, 4.5, 5.5)
        from datetime import datetime
        trade = tracker.close_position(trade_id, timestamp=datetime.now(), exit_price=0.0, exit_reason=ExitReason.TARGET)
        assert trade is not None
        call_args = spy_exit_fill.call_args_list[0]
        side_arg = call_args[1].get("side") if "side" in call_args[1] else call_args[0][3]
        assert side_arg == "BUY"


def test_directional_put_fill_side_in_paper_engine_and_tracker():
    """Verify directional Put option buying enters with side='BUY' and exits with side='SELL'."""
    from prometheus.papertrade.engine import PaperTradeEngine
    from prometheus.papertrade.position_tracker import PositionTracker

    mock_feed = MagicMock()
    mock_feed.get_quote.return_value = (150.0, 149.0, 151.0)
    mock_feed.get_ltp.return_value = 150.0

    engine = PaperTradeEngine(feed=mock_feed)
    tracker = engine.tracker

    put_sig = SignalNotification(
        symbol="NIFTY 50",
        instrument="NIFTY26OCT22500PE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        strike=22500.0,
        option_type="PE",
        expiry="2026-10-29",
        entry_price_hint=150.0,
        stop_loss=120.0,
        target=190.0,
        strategy="trend",
    )

    with patch.object(tracker.fill_sim, "fill", wraps=tracker.fill_sim.fill) as spy_fill:
        trade_id = engine.process_new_signal(put_sig)
        assert trade_id is not None
        # On single-leg put entry, side must be "BUY"
        call_args = spy_fill.call_args_list[0]
        side_arg = call_args[1].get("side") if "side" in call_args[1] else call_args[0][3]
        assert side_arg == "BUY"

    with patch.object(tracker.fill_sim, "fill", wraps=tracker.fill_sim.fill) as spy_exit_fill:
        from datetime import datetime
        trade = tracker.close_position(trade_id, timestamp=datetime.now(), exit_price=0.0, exit_reason=ExitReason.TARGET)
        assert trade is not None
        call_args = spy_exit_fill.call_args_list[0]
        side_arg = call_args[1].get("side") if "side" in call_args[1] else call_args[0][3]
        assert side_arg == "SELL"
