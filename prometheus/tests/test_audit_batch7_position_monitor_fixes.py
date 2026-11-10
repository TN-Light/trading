import pytest
from unittest.mock import MagicMock
from datetime import datetime, date
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.broker import Order, OrderStatus, OrderType


def test_bug1_stage2_stage3_sl_never_exceeds_ltp():
    """Verify that Stage 2 and Stage 3 ratchets never set SL above current LTP (Bug 1)."""
    broker = MagicMock()
    monitor = PositionMonitor(broker=broker)

    # NIFTY 50 option: entry 200.0, initial_sl 187.0 (rd = 13.0 pts)
    state = TrailingState(
        position_id="test_pos_1",
        tradingsymbol="NIFTY26N1024500CE",
        symbol="NIFTY 50",
        entry_premium=200.0,
        initial_sl=187.0,
        current_sl=187.0,
        target=250.0,
        direction="bullish",
    )
    monitor.add_position(state)

    # 1. At LTP = 219.50 (gain = +19.50 pts = +1.5R)
    monitor._process_tick(state, 219.50)
    assert state.trailing_stage2 is True
    # Crucial: SL must be strictly below current LTP!
    assert state.current_sl < 219.50
    # In legacy bug, new_sl was max(200 * 1.16, 200 + 8.45) = 232.00 (12.5 pts ABOVE LTP)
    assert state.current_sl <= 219.00
    assert state.current_sl >= 200.0 + 13.0 * 0.65  # at least +0.65R (208.45)

    # 2. At LTP = 226.00 (gain = +26.00 pts = +2.0R)
    monitor._process_tick(state, 226.00)
    assert state.trailing_stage3 is True
    # Crucial: SL must be strictly below current LTP!
    assert state.current_sl < 226.00
    # In legacy bug, new_sl was max(200 * 1.24, 200 + 9.75) = 248.00 (22 pts ABOVE LTP)
    assert state.current_sl <= 225.50
    assert state.current_sl >= 200.0 + 13.0 * 0.75  # at least +0.75R (209.75)


def test_bug2_modify_broker_sl_manual_updates_limit_price():
    """Verify that _modify_broker_sl_manual ratchets limit price alongside trigger (Bug 2)."""
    broker = MagicMock()
    fake_order = MagicMock()
    fake_order.status = OrderStatus.OPEN
    broker.get_order_status.return_value = fake_order

    monitor = PositionMonitor(broker=broker)
    state = TrailingState(
        position_id="test_pos_2",
        tradingsymbol="NIFTY26N1024500CE",
        symbol="NIFTY 50",
        entry_premium=200.0,
        initial_sl=187.0,
        current_sl=200.0,
        target=250.0,
        direction="bullish",
        sl_order_id="SL_12345",
    )

    monitor._modify_broker_sl_manual(state, 220.0)

    # Verify modify_order was called with aligned trigger AND aligned limit price
    broker.modify_order.assert_called_once_with(
        "SL_12345",
        trigger_price=220.0,
        price=209.0,  # 220.0 * 0.95 = 209.0
    )


def test_bug3_check_intraday_bar_increment_handles_iso_strings():
    """Verify _check_intraday_bar_increment uses _get_elapsed_seconds and handles ISO entry_time (Bug 3)."""
    monitor = PositionMonitor(broker=MagicMock())
    state = TrailingState(
        position_id="test_pos_3",
        tradingsymbol="NIFTY26N1024500CE",
        symbol="NIFTY 50",
        entry_premium=200.0,
        initial_sl=187.0,
        current_sl=187.0,
        target=250.0,
        direction="bullish",
    )
    # ISO entry time string with timezone offset
    state.entry_time = "2026-11-10T09:15:00+05:30"
    state.entry_bar_count = 0
    state._last_bar_ts = None
    state.bar_interval = "15minute"

    monitor.add_position(state)
    # Should not raise ValueError: time data does not match format
    monitor._check_intraday_bar_increment()
    assert state._last_bar_ts is not None


def test_bug4_tg_cmd_positions_quantity_scaling_and_credit_spread():
    """Verify _tg_cmd_positions uses quantity for rupee PnL and handles credit spreads & BFO (Bug 4)."""
    from prometheus.main import Prometheus
    bot = MagicMock(spec=Prometheus)
    bot.position_monitor = MagicMock()
    bot.position_monitor.active_count = 2

    # Option buying state
    buy_state = TrailingState(
        position_id="buy_pos",
        tradingsymbol="NIFTY26N1024500CE",
        symbol="NIFTY 50",
        entry_premium=100.0,
        initial_sl=90.0,
        current_sl=90.0,
        target=130.0,
        direction="bullish",
    )
    buy_state.quantity = 65

    # Credit spread state
    spread_state = TrailingState(
        position_id="spread_pos",
        tradingsymbol="SENSEX26NOV75000PE",
        symbol="SENSEX",
        entry_premium=50.0,
        initial_sl=70.0,
        current_sl=70.0,
        target=10.0,
        direction="bullish",
    )
    spread_state.quantity = 20
    spread_state.strategy_type = "credit_spread"

    bot.position_monitor.get_positions.return_value = {
        "buy_pos": buy_state,
        "spread_pos": spread_state,
    }

    def mock_get_ltp(tsym, exchange="NFO"):
        if "NIFTY" in tsym:
            assert exchange == "NFO"
            return 110.0  # +10 pts gain -> +650 Rs
        elif "SENSEX" in tsym:
            assert exchange == "BFO"
            return 40.0   # +10 pts gain for seller -> +200 Rs
        return 0.0

    bot.broker = MagicMock()
    bot.broker.get_ltp.side_effect = mock_get_ltp
    bot._make_kite_search_name = MagicMock(return_value="DISPLAY_NAME")

    output = Prometheus._tg_cmd_positions(bot)
    assert "+650.00 (+10.0%)" in output
    assert "+200.00 (+20.0%)" in output


def test_bug5_symbol_parsing_supports_monthly_and_angelone():
    """Verify _parse_expiry and _parse_strike support monthly and Angel One formats (Bug 5)."""
    from prometheus.main import Prometheus
    bot = MagicMock(spec=Prometheus)

    # 1. Monthly Kite symbol (preponed to Monday Nov 23 because Tuesday Nov 24 is a trading holiday)
    exp = Prometheus._parse_expiry_from_tradingsymbol(bot, "NIFTY26NOV24500CE", "NIFTY 50")
    strike, otype = Prometheus._parse_strike_otype_from_tradingsymbol(bot, "NIFTY26NOV24500CE", "NIFTY 50")
    assert exp == "2026-11-23"
    assert strike == 24500.0
    assert otype == "CE"

    # 2. Angel One DDMMMYY symbol
    exp_ao = Prometheus._parse_expiry_from_tradingsymbol(bot, "NIFTY10NOV2624500PE", "NIFTY 50")
    strike_ao, otype_ao = Prometheus._parse_strike_otype_from_tradingsymbol(bot, "NIFTY10NOV2624500PE", "NIFTY 50")
    assert exp_ao == "2026-11-10"
    assert strike_ao == 24500.0
    assert otype_ao == "PE"

    # 3. Weekly Kite symbol
    exp_wk = Prometheus._parse_expiry_from_tradingsymbol(bot, "NIFTY26N1024500CE", "NIFTY 50")
    strike_wk, otype_wk = Prometheus._parse_strike_otype_from_tradingsymbol(bot, "NIFTY26N1024500CE", "NIFTY 50")
    assert exp_wk == "2026-11-10"
    assert strike_wk == 24500.0
    assert otype_wk == "CE"
