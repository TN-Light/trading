import pytest
import json
from unittest.mock import MagicMock, patch
from prometheus.execution.broker import OrderSide, OrderType, OrderStatus, ProductType
from prometheus.execution.paper_trader import PaperTrader
from prometheus.execution.order_manager import OrderManager
from prometheus.execution.position_monitor import TrailingState, PositionMonitor


class MockController:
    def __init__(self):
        self.store = MagicMock()
        self.broker = PaperTrader(initial_capital=100000.0)
        self.risk_manager = MagicMock()
        self.order_manager = OrderManager(self.broker, self.risk_manager)
        self.position_monitor = MagicMock()
        self.telegram = MagicMock()
        self.multi_account = MagicMock()
        self.initial_capital = 100000.0

    def _restore_persisted_positions(self):
        # We inline the method from main.py using a patch or we import and bind it.
        # To test it cleanly, we can use the actual main.py code by calling the method
        # directly on our controller instance if main.py allows.
        # For this test, we import the function or implement the exact logic we wrote.
        pass


def test_restoration_logic():
    # Setup database load mock response
    entry_orders_data = {
        "multi_account_label": None,
        "entry_orders": [
            {
                "order_id": "ORD-001",
                "symbol": "SENSEX",
                "tradingsymbol": "SENSEX76600PE",
                "side": "BUY",
                "order_type": "MARKET",
                "product": "MIS",
                "quantity": 50,
                "filled_quantity": 50,
                "average_price": 230.0,
                "status": "COMPLETE",
                "timestamp": "2026-06-30 15:00:00",
                "tag": "ENTRY"
            }
        ]
    }
    
    mock_db_row = {
        "position_id": "POS-001",
        "tradingsymbol": "SENSEX76600PE",
        "symbol": "SENSEX",
        "direction": "bullish",
        "entry_premium": 230.0,
        "initial_sl": 200.0,
        "current_sl": 210.0,
        "target": 300.0,
        "sl_order_id": "SL-001",
        "strategy": "trend",
        "entry_time": "2026-06-30 15:00:00",
        "breakeven_set": 0,
        "trailing_activated": 1,
        "trailing_stage2": 0,
        "trailing_stage3": 0,
        "premium_hwm": 250.0,
        "entry_bar_count": 2,
        "max_bars": 7,
        "breakeven_ratio": 0.6,
        "risk_distance": 20.0,
        "bar_interval": "15minute",
        "trade_mode": "intraday",
        "entry_orders_json": json.dumps(entry_orders_data)
    }

    # Instantiate mock controller
    ctrl = MockController()
    ctrl.store.load_open_positions.return_value = [mock_db_row]
    ctrl.store.close_stale_positions.return_value = 0

    # Bind the actual main.py _restore_persisted_positions to our MockController
    from prometheus.main import Prometheus
    ctrl._restore_persisted_positions = Prometheus._restore_persisted_positions.__get__(ctrl, MockController)
    ctrl._parse_strike_otype_from_tradingsymbol = Prometheus._parse_strike_otype_from_tradingsymbol.__get__(ctrl, MockController)
    ctrl._parse_expiry_from_tradingsymbol = Prometheus._parse_expiry_from_tradingsymbol.__get__(ctrl, MockController)

    # Execute restoration
    ctrl._restore_persisted_positions()

    # Assertions
    # 1. TrailingState restored in position_monitor
    ctrl.position_monitor.restore_positions.assert_called_once()
    restored_states = ctrl.position_monitor.restore_positions.call_args[0][0]
    assert len(restored_states) == 1
    state = restored_states[0]
    assert state.position_id == "POS-001"
    assert state.tradingsymbol == "SENSEX76600PE"
    assert state.entry_orders_json == json.dumps(entry_orders_data)

    # 2. ManagedPosition restored in OrderManager
    assert "POS-001" in ctrl.order_manager.managed_positions
    managed = ctrl.order_manager.managed_positions["POS-001"]
    assert managed.position_id == "POS-001"
    assert managed.symbol == "SENSEX"
    assert managed.stop_loss == 200.0
    assert managed.trailing_stop == 210.0
    assert len(managed.entry_orders) == 1
    assert managed.entry_orders[0].order_id == "ORD-001"
    assert managed.entry_orders[0].quantity == 50

    # 3. Position restored in PaperTrader broker
    assert "SENSEX76600PE" in ctrl.broker.positions
    pos = ctrl.broker.positions["SENSEX76600PE"]
    assert pos.tradingsymbol == "SENSEX76600PE"
    assert pos.quantity == 50
    assert pos.average_price == 230.0


def test_daily_state_persistence():
    from prometheus.main import Prometheus
    from datetime import time as dtime_cls
    
    ctrl = MockController()
    
    # Store mocked state dictionary
    db_state = {}
    ctrl.store.load_state.side_effect = lambda key: db_state.get(key, "")
    ctrl.store.save_state.side_effect = lambda key, val: db_state.update({key: val})

    # Bind helper methods to ctrl
    ctrl._get_daily_state = Prometheus._get_daily_state.__get__(ctrl, MockController)
    ctrl._set_daily_state = Prometheus._set_daily_state.__get__(ctrl, MockController)

    # Test boolean
    ctrl._set_daily_state("bool_val", True)
    assert ctrl._get_daily_state("bool_val", False) is True

    # Test integer
    ctrl._set_daily_state("int_val", 42)
    assert ctrl._get_daily_state("int_val", 0) == 42

    # Test string set
    test_set = {"NIFTY", "BANKNIFTY"}
    ctrl._set_daily_state("set_val", test_set)
    assert ctrl._get_daily_state("set_val", set()) == test_set

    # Test datetime.time set (scans)
    time_set = {dtime_cls(13, 0), dtime_cls(15, 35)}
    ctrl._set_daily_state("index_scans", time_set)
    restored_time_set = ctrl._get_daily_state("index_scans", set())
    assert restored_time_set == time_set

