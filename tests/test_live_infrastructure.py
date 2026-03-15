#!/usr/bin/env python
"""
Comprehensive test suite for PROMETHEUS live infrastructure.

Tests:
1. TrailingState dataclass -- all fields, __post_init__, current_stage()
2. PositionMonitor -- 5-stage trailing stop (bullish)
3. PositionMonitor -- SL breach, target hit, time stop
4. PositionMonitor -- 5-stage trailing stop (bearish / PUT buying)
5. SQLite persistence -- save/load/close position state
6. OrderManager -- create_trailing_state
7. Telegram -- confirmation flow
8. PaperTrader -- modify_order, get_ltp, update_prices
9. Edge cases
10. PositionMonitor lifecycle
11. main.py formatting bug check

Run: python tests/test_live_infrastructure.py
"""

import sys
import os
import threading
import time
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prometheus.execution.position_monitor import TrailingState, PositionMonitor
from prometheus.execution.broker import (
    BrokerBase, Order, OrderType, OrderSide, ProductType, OrderStatus
)
from prometheus.execution.paper_trader import PaperTrader
from prometheus.data.store import DataStore


# ============================================================================
# Test helpers
# ============================================================================
passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    # Sanitize for Windows console output
    safe_name = name.encode("ascii", "replace").decode("ascii")
    if condition:
        passed += 1
        print(f"  PASS  {safe_name}")
    else:
        failed += 1
        safe_detail = detail.encode("ascii", "replace").decode("ascii") if detail else ""
        msg = f"  FAIL  {safe_name}"
        if safe_detail:
            msg += f" -- {safe_detail}"
        print(msg)
        errors.append(msg)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# 1. TrailingState Tests
# ============================================================================
def test_trailing_state():
    section("1. TrailingState Dataclass")

    # Basic construction
    ts = TrailingState(
        position_id="POS-001",
        tradingsymbol="NIFTY25MAR22500CE",
        symbol="NIFTY 50",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
    )
    test("Basic construction", ts.position_id == "POS-001")
    test("Direction set", ts.direction == "bullish")

    # __post_init__ computes risk_distance
    test(
        "risk_distance auto-computed",
        abs(ts.risk_distance - 60.0) < 0.01,
        f"expected 60.0, got {ts.risk_distance}"
    )

    # current_stage at INITIAL
    test("Initial stage", ts.current_stage() == "INITIAL")

    # current_stage progression
    ts.breakeven_set = True
    test("Stage after breakeven", ts.current_stage() == "BREAKEVEN")

    ts.trailing_activated = True
    test("Stage after lock 20%", ts.current_stage() == "LOCK 20%")

    ts.trailing_stage2 = True
    test("Stage after lock 50%", ts.current_stage() == "LOCK 50%")

    ts.trailing_stage3 = True
    test("Stage after runner", ts.current_stage() == "RUNNER (70%+)")

    # to_dict roundtrip
    d = ts.to_dict()
    test("to_dict has position_id", d["position_id"] == "POS-001")
    test("to_dict has premium_hwm", "premium_hwm" in d)
    test("to_dict has risk_distance", d["risk_distance"] == 60.0)

    # Edge case: risk_distance when initial_sl is 0
    ts2 = TrailingState(
        position_id="POS-002",
        tradingsymbol="TEST",
        symbol="TEST",
        entry_premium=100.0,
        initial_sl=0.0,
        current_sl=0.0,
        target=200.0,
        direction="bullish",
    )
    test(
        "risk_distance fallback (sl=0)",
        abs(ts2.risk_distance - 30.0) < 0.01,
        f"expected 30.0 (30% of entry), got {ts2.risk_distance}"
    )

    # Edge case: explicit risk_distance should not be overwritten
    ts3 = TrailingState(
        position_id="POS-003",
        tradingsymbol="TEST",
        symbol="TEST",
        entry_premium=100.0,
        initial_sl=80.0,
        current_sl=80.0,
        target=200.0,
        direction="bullish",
        risk_distance=25.0,  # explicit
    )
    test(
        "explicit risk_distance preserved",
        abs(ts3.risk_distance - 25.0) < 0.01,
        f"expected 25.0, got {ts3.risk_distance}"
    )


# ============================================================================
# 2. PositionMonitor -- 5-Stage Trailing Stop (Bullish)
# ============================================================================
def test_trailing_stop_bullish():
    section("2. 5-Stage Trailing Stop (Bullish)")

    exit_calls = []
    trail_calls = []
    state_calls = []

    def on_exit(pid, price, reason):
        exit_calls.append((pid, price, reason))

    def on_trail(state, old_sl):
        trail_calls.append((state.position_id, state.current_stage(), old_sl, state.current_sl))

    def on_state(state):
        state_calls.append(state.position_id)

    broker = PaperTrader(100000)
    pm = PositionMonitor(
        broker=broker,
        poll_interval=1,
        on_exit=on_exit,
        on_trailing_update=on_trail,
        on_state_changed=on_state,
    )

    # Create a position: entry=200, SL=140, target=500 (set high to not interfere)
    # risk_distance = 200 - 140 = 60
    ts = TrailingState(
        position_id="TEST-BULL-001",
        tradingsymbol="NIFTY25MAR22500CE",
        symbol="NIFTY 50",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=600.0,  # high enough to not interfere with trailing tests
        direction="bullish",
    )
    test("Setup: risk_distance=60", abs(ts.risk_distance - 60.0) < 0.01)

    # Stage 0: Breakeven at entry + rd * 0.6 = 200 + 36 = 236
    # New SL = entry + rd * 0.10 = 200 + 6 = 206
    pm._process_tick(ts, 235.0)  # below trigger
    test("Stage 0 not triggered below 236", not ts.breakeven_set)

    pm._process_tick(ts, 236.0)  # at trigger
    test("Stage 0 triggered at 236", ts.breakeven_set)
    test(
        "Stage 0 SL = 206",
        abs(ts.current_sl - 206.0) < 0.01,
        f"got {ts.current_sl}"
    )
    test("Stage 0 trail callback fired", len(trail_calls) == 1)

    # Stage 1: Lock 20% at entry + rd * 1.0 = 200 + 60 = 260
    # New SL = entry + rd * 0.20 = 200 + 12 = 212
    pm._process_tick(ts, 259.0)
    test("Stage 1 not triggered below 260", not ts.trailing_activated)

    pm._process_tick(ts, 260.0)
    test("Stage 1 triggered at 260", ts.trailing_activated)
    test(
        "Stage 1 SL = 212",
        abs(ts.current_sl - 212.0) < 0.01,
        f"got {ts.current_sl}"
    )

    # Stage 2: Lock 50% at entry + rd * 2.0 = 200 + 120 = 320
    # New SL = entry + rd * 0.50 = 200 + 30 = 230
    pm._process_tick(ts, 319.0)
    test("Stage 2 not triggered below 320", not ts.trailing_stage2)

    pm._process_tick(ts, 320.0)
    test("Stage 2 triggered at 320", ts.trailing_stage2)
    test(
        "Stage 2 SL = 230",
        abs(ts.current_sl - 230.0) < 0.01,
        f"got {ts.current_sl}"
    )

    # Stage 3: Lock 70% at entry + rd * 3.0 = 200 + 180 = 380
    # New SL = entry + rd * 0.70 = 200 + 42 = 242, HWM = 380
    pm._process_tick(ts, 379.0)
    test("Stage 3 not triggered below 380", not ts.trailing_stage3)

    pm._process_tick(ts, 380.0)
    test("Stage 3 triggered at 380", ts.trailing_stage3)
    test(
        "Stage 3 SL = 242",
        abs(ts.current_sl - 242.0) < 0.01,
        f"got {ts.current_sl}"
    )
    test(
        "Stage 3 HWM = 380",
        abs(ts.premium_hwm - 380.0) < 0.01,
        f"got {ts.premium_hwm}"
    )

    # Stage 4: Dynamic trail
    # Price goes to 400 -> HWM = 400
    # floor_sl = 200 + 60*0.70 = 242
    # trail_offset = (400 - 200) * 0.30 = 60
    # dynamic_sl = 400 - 60 = 340
    # new_sl = max(242, 340) = 340
    pm._process_tick(ts, 400.0)
    test(
        "Stage 4 HWM updated to 400",
        abs(ts.premium_hwm - 400.0) < 0.01,
        f"got {ts.premium_hwm}"
    )
    test(
        "Stage 4 SL = 340",
        abs(ts.current_sl - 340.0) < 0.01,
        f"got {ts.current_sl}"
    )

    # Price goes higher to 500 -> HWM = 500
    # trail_offset = (500 - 200) * 0.30 = 90
    # dynamic_sl = 500 - 90 = 410
    pm._process_tick(ts, 500.0)
    test(
        "Stage 4 SL ratchets up to 410",
        abs(ts.current_sl - 410.0) < 0.01,
        f"got {ts.current_sl}"
    )

    # Price drops to 450 -> HWM stays 500
    # dynamic_sl stays max(242, 500-90=410) = 410, no change
    old_sl = ts.current_sl
    pm._process_tick(ts, 450.0)
    test(
        "Stage 4 SL doesn't drop when price drops",
        abs(ts.current_sl - old_sl) < 0.01,
        f"SL moved from {old_sl} to {ts.current_sl}"
    )
    test("HWM stays at peak", abs(ts.premium_hwm - 500.0) < 0.01)

    # Verify no spurious exit calls
    test("No exit triggered during normal trailing", len(exit_calls) == 0)

    # All state persist callbacks fired (stages 0-3 + at least 1 stage 4)
    test("State persist callbacks fired", len(state_calls) >= 5, f"got {len(state_calls)}")


# ============================================================================
# 3. Exit Conditions (SL Breach, Target, Time Stop)
# ============================================================================
def test_exit_conditions():
    section("3. Exit Conditions (SL Breach, Target, Time Stop)")

    exit_calls = []

    def on_exit(pid, price, reason):
        exit_calls.append((pid, price, reason))

    broker = PaperTrader(100000)
    pm = PositionMonitor(broker=broker, poll_interval=1, on_exit=on_exit)

    # --- SL Breach (bullish) --- premium drops below SL
    ts = TrailingState(
        position_id="SL-BULL",
        tradingsymbol="TEST",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
    )
    pm._process_tick(ts, 139.0)
    test(
        "SL breach (bullish) triggers exit",
        len(exit_calls) == 1 and exit_calls[-1][2] == "stop_loss",
    )

    # --- SL Breach (bearish / PUT buying) --- premium drops below SL
    ts2 = TrailingState(
        position_id="SL-BEAR",
        tradingsymbol="TEST2",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bearish",
    )
    pm._process_tick(ts2, 139.0)  # premium drops below SL
    test(
        "SL breach (bearish/PUT) triggers exit",
        len(exit_calls) == 2 and exit_calls[-1][2] == "stop_loss",
    )

    # --- Target Hit (bullish) --- premium rises above target
    ts3 = TrailingState(
        position_id="TGT-BULL",
        tradingsymbol="TEST3",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
    )
    pm._process_tick(ts3, 380.0)
    test(
        "Target hit (bullish) triggers exit",
        len(exit_calls) == 3 and exit_calls[-1][2] == "target",
    )

    # --- Target Hit (bearish / PUT buying) --- premium rises above target
    ts4 = TrailingState(
        position_id="TGT-BEAR",
        tradingsymbol="TEST4",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bearish",
    )
    pm._process_tick(ts4, 380.0)  # premium rises above target
    test(
        "Target hit (bearish/PUT) triggers exit",
        len(exit_calls) == 4 and exit_calls[-1][2] == "target",
    )

    # --- Time Stop ---
    ts5 = TrailingState(
        position_id="TIME-STOP",
        tradingsymbol="TEST5",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
        entry_bar_count=7,  # at max_bars
        max_bars=7,
    )
    pm._process_tick(ts5, 210.0)  # price is fine, but time expired
    test(
        "Time stop triggers exit",
        len(exit_calls) == 5 and exit_calls[-1][2] == "time_stop",
    )

    # --- Time stop NOT triggered when bar_count < max ---
    ts6 = TrailingState(
        position_id="TIME-OK",
        tradingsymbol="TEST6",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
        entry_bar_count=6,
        max_bars=7,
    )
    pm._process_tick(ts6, 210.0)
    test(
        "No time stop when bar_count < max_bars",
        len(exit_calls) == 5,
        f"expected 5, got {len(exit_calls)}"
    )


# ============================================================================
# 4. 5-Stage Trailing Stop (Bearish / PUT Buying)
# ============================================================================
def test_trailing_stop_bearish():
    section("4. 5-Stage Trailing Stop (Bearish / PUT Buying)")

    exit_calls = []
    trail_calls = []

    def on_exit(pid, price, reason):
        exit_calls.append((pid, price, reason))

    def on_trail(state, old_sl):
        trail_calls.append((state.position_id, state.current_sl))

    broker = PaperTrader(100000)
    pm = PositionMonitor(broker=broker, poll_interval=1, on_exit=on_exit, on_trailing_update=on_trail)

    # Bearish: buying PUTs -- premium INCREASES when underlying drops
    # Trailing logic is same as bullish: premium going up = profit
    ts = TrailingState(
        position_id="BEAR-001",
        tradingsymbol="NIFTY25MAR22500PE",
        symbol="NIFTY 50",
        entry_premium=150.0,
        initial_sl=100.0,
        current_sl=100.0,
        target=400.0,  # high to not interfere
        direction="bearish",
    )
    # risk_distance = 150 - 100 = 50
    test("Bearish risk_distance", abs(ts.risk_distance - 50.0) < 0.01)

    # Stage 0 at entry + rd*0.6 = 150 + 30 = 180
    # SL -> entry + rd*0.10 = 150 + 5 = 155
    pm._process_tick(ts, 180.0)
    test("Bearish Stage 0 breakeven", ts.breakeven_set)
    test("Bearish Stage 0 SL", abs(ts.current_sl - 155.0) < 0.01, f"got {ts.current_sl}")

    # Stage 1 at entry + rd*1.0 = 150 + 50 = 200
    # SL -> entry + rd*0.20 = 150 + 10 = 160
    pm._process_tick(ts, 200.0)
    test("Bearish Stage 1 lock 20%", ts.trailing_activated)
    test("Bearish Stage 1 SL", abs(ts.current_sl - 160.0) < 0.01, f"got {ts.current_sl}")

    # Stage 2 at entry + rd*2.0 = 150 + 100 = 250
    pm._process_tick(ts, 250.0)
    test("Bearish Stage 2 lock 50%", ts.trailing_stage2)
    test("Bearish Stage 2 SL", abs(ts.current_sl - 175.0) < 0.01, f"got {ts.current_sl}")

    # Stage 3 at entry + rd*3.0 = 150 + 150 = 300
    pm._process_tick(ts, 300.0)
    test("Bearish Stage 3 runner", ts.trailing_stage3)
    test("Bearish Stage 3 SL", abs(ts.current_sl - 185.0) < 0.01, f"got {ts.current_sl}")

    # No false exit during trailing
    test("Bearish: no false exit during trailing", len(exit_calls) == 0)

    # SL breach: premium drops below current_sl of 185
    ts_sl = TrailingState(
        position_id="BEAR-SL",
        tradingsymbol="TEST-BEAR-SL",
        symbol="TEST",
        entry_premium=150.0,
        initial_sl=100.0,
        current_sl=185.0,  # After stage 3
        target=400.0,
        direction="bearish",
        breakeven_set=True,
        trailing_activated=True,
        trailing_stage2=True,
        trailing_stage3=True,
    )
    pm._process_tick(ts_sl, 180.0)  # premium drops below SL
    test(
        "Bearish SL breach on premium drop",
        len(exit_calls) == 1 and exit_calls[-1][2] == "stop_loss",
    )

    # No false exit when premium is above SL
    exit_before = len(exit_calls)
    ts_ok = TrailingState(
        position_id="BEAR-OK",
        tradingsymbol="TEST-BEAR-OK",
        symbol="TEST",
        entry_premium=150.0,
        initial_sl=100.0,
        current_sl=155.0,
        target=400.0,
        direction="bearish",
        breakeven_set=True,
    )
    pm._process_tick(ts_ok, 190.0)  # premium above SL -- should NOT exit
    test(
        "Bearish: no false exit when premium above SL",
        len(exit_calls) == exit_before,
        f"got {len(exit_calls)} exits (expected {exit_before})"
    )


# ============================================================================
# 5. SQLite Persistence
# ============================================================================
def test_sqlite_persistence():
    section("5. SQLite Persistence")

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_prometheus.db")

    try:
        store = DataStore(db_path=db_path)

        state_dict = {
            "position_id": "POS-TEST-001",
            "tradingsymbol": "NIFTY25MAR22500CE",
            "symbol": "NIFTY 50",
            "direction": "bullish",
            "strategy": "trend",
            "entry_premium": 200.0,
            "initial_sl": 140.0,
            "current_sl": 206.0,
            "target": 380.0,
            "sl_order_id": "SL-123",
            "entry_time": "2026-03-15 10:00:00",
            "status": "open",
            "breakeven_set": True,
            "trailing_activated": False,
            "trailing_stage2": False,
            "trailing_stage3": False,
            "premium_hwm": 0.0,
            "entry_bar_count": 2,
            "max_bars": 7,
            "breakeven_ratio": 0.6,
            "risk_distance": 60.0,
            "entry_orders_json": "[]",
        }
        store.save_position_state(state_dict)
        test("Save position state", True)

        loaded = store.load_open_positions()
        test("Load returns 1 position", len(loaded) == 1, f"got {len(loaded)}")

        row = loaded[0]
        test("Position ID matches", row["position_id"] == "POS-TEST-001")
        test("Entry premium matches", abs(row["entry_premium"] - 200.0) < 0.01)
        test("Current SL matches", abs(row["current_sl"] - 206.0) < 0.01)
        test("Breakeven set matches", row["breakeven_set"] == 1)
        test("Risk distance matches", abs(row["risk_distance"] - 60.0) < 0.01)
        test("Entry bar count matches", row["entry_bar_count"] == 2)

        # Update (simulate trailing advance)
        state_dict["current_sl"] = 212.0
        state_dict["trailing_activated"] = True
        store.save_position_state(state_dict)

        loaded2 = store.load_open_positions()
        test("Updated SL persisted", abs(loaded2[0]["current_sl"] - 212.0) < 0.01)
        test("Stage 1 persisted", loaded2[0]["trailing_activated"] == 1)

        # Close
        store.close_position_state("POS-TEST-001", pnl=1500.0)
        loaded3 = store.load_open_positions()
        test("Closed position not in open list", len(loaded3) == 0)

        # Multiple positions
        for i in range(3):
            s = state_dict.copy()
            s["position_id"] = f"POS-MULTI-{i:03d}"
            s["status"] = "open"
            store.save_position_state(s)

        loaded4 = store.load_open_positions()
        test("Multiple positions saved", len(loaded4) == 3, f"got {len(loaded4)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# 6. OrderManager -- create_trailing_state
# ============================================================================
def test_order_manager():
    section("6. OrderManager Integration")

    from prometheus.execution.order_manager import OrderManager, ManagedPosition
    from prometheus.risk.manager import RiskManager

    broker = PaperTrader(100000)
    risk_limits = {
        "max_daily_loss": 5000,
        "max_weekly_loss": 10000,
        "max_monthly_loss": 25000,
        "max_position_size_pct": 40,
        "max_open_positions": 3,
        "drawdown_halt_pct": 10,
        "max_trade_risk_pct": 2,
    }
    risk = RiskManager(risk_limits, 100000)
    om = OrderManager(broker, risk, "paper")

    managed = ManagedPosition(
        position_id="OM-TEST-001",
        symbol="NIFTY 50",
        strategy="trend",
        direction="bullish",
        entry_orders=[],
        exit_orders=[],
        stop_loss=140.0,
        target=380.0,
        trailing_stop=0,
        entry_time="2026-03-15 10:00:00",
        tradingsymbol="NIFTY25MAR22500CE",
        entry_premium=200.0,
        sl_order_id="SL-001",
        max_bars=7,
        breakeven_ratio=0.6,
    )
    om.managed_positions["OM-TEST-001"] = managed

    ts = om.create_trailing_state("OM-TEST-001")
    test("create_trailing_state returns TrailingState", ts is not None)
    test("TrailingState position_id", ts.position_id == "OM-TEST-001")
    test("TrailingState tradingsymbol", ts.tradingsymbol == "NIFTY25MAR22500CE")
    test("TrailingState entry_premium", abs(ts.entry_premium - 200.0) < 0.01)
    test("TrailingState initial_sl", abs(ts.initial_sl - 140.0) < 0.01)
    test("TrailingState target", abs(ts.target - 380.0) < 0.01)
    test("TrailingState direction", ts.direction == "bullish")
    test("TrailingState sl_order_id", ts.sl_order_id == "SL-001")
    test("TrailingState max_bars", ts.max_bars == 7)

    ts2 = om.create_trailing_state("NONEXISTENT")
    test("Non-existent position returns None", ts2 is None)

    managed.status = "closed"
    ts3 = om.create_trailing_state("OM-TEST-001")
    test("Closed position returns None", ts3 is None)


# ============================================================================
# 7. Telegram Confirmation Flow
# ============================================================================
def test_telegram_confirmation():
    section("7. Telegram Confirmation Flow")

    from prometheus.interface.telegram_bot import TelegramBot

    bot = TelegramBot()  # empty token/chat_id -> _enabled=False

    result = bot.handle_confirm()
    test("No pending -> confirm returns info", "No pending" in result)

    result2 = bot.handle_reject()
    test("No pending -> reject returns info", "No pending" in result2)

    signal = {"action": "BUY_CE", "symbol": "NIFTY 50", "confidence": 0.85}

    # Test confirm path
    bot._pending_confirmation = signal
    bot._confirmation_event.clear()
    bot._confirmation_result = None

    def do_confirm():
        time.sleep(0.3)
        bot.handle_confirm()

    t = threading.Thread(target=do_confirm)
    t.start()
    bot._confirmation_event.wait(timeout=2)
    result = bot._confirmation_result
    test("Confirm sets result to True", result == True)
    t.join()

    # Test reject path
    bot._pending_confirmation = signal
    bot._confirmation_event.clear()
    bot._confirmation_result = None

    def do_reject():
        time.sleep(0.3)
        bot.handle_reject()

    t2 = threading.Thread(target=do_reject)
    t2.start()
    bot._confirmation_event.wait(timeout=2)
    result2 = bot._confirmation_result
    test("Reject sets result to False", result2 == False)
    t2.join()

    # Test timeout path
    bot._pending_confirmation = signal
    bot._confirmation_event.clear()
    bot._confirmation_result = None
    bot._confirmation_event.wait(timeout=0.5)
    test("Timeout leaves result as None", bot._confirmation_result is None)


# ============================================================================
# 8. PaperTrader -- modify_order, get_ltp
# ============================================================================
def test_paper_trader():
    section("8. PaperTrader (modify_order, get_ltp, update_prices)")

    broker = PaperTrader(100000)

    ltp = broker.get_ltp("NIFTY25MAR22500CE")
    test("get_ltp returns 0 for unknown", abs(ltp) < 0.01)

    broker.update_prices({"NIFTY25MAR22500CE": 200.0})
    ltp = broker.get_ltp("NIFTY25MAR22500CE")
    test("get_ltp returns 200 after update", abs(ltp - 200.0) < 0.01)

    # Place and modify an SL order
    order = Order(
        tradingsymbol="NIFTY25MAR22500CE",
        exchange="NFO",
        side=OrderSide.SELL,
        order_type=OrderType.SL_M,
        product=ProductType.MIS,
        quantity=25,
        trigger_price=140.0,
        tag="TEST-SL",
    )
    placed = broker.place_order(order)
    test("SL order placed", placed.status == OrderStatus.TRIGGER_PENDING)
    test("SL order has ID", len(placed.order_id) > 0)

    # Modify the trigger price
    modified = broker.modify_order(placed.order_id, trigger_price=206.0)
    test("SL order modified", modified.trigger_price == 206.0 if hasattr(modified, 'trigger_price') else False)

    # Verify modification persists
    order_check = broker.get_order_status(placed.order_id)
    test(
        "Modified trigger persisted",
        abs(order_check.trigger_price - 206.0) < 0.01,
        f"got {order_check.trigger_price}"
    )

    # Modify non-existent order
    bad = broker.modify_order("NONEXISTENT")
    test("Modify non-existent -> rejected", bad.status == OrderStatus.REJECTED)

    # Test SL trigger with updated price
    broker.update_prices({"NIFTY25MAR22500CE": 207.0})
    order_after = broker.get_order_status(placed.order_id)
    test("SL not triggered above trigger price", order_after.status == OrderStatus.TRIGGER_PENDING)

    broker.update_prices({"NIFTY25MAR22500CE": 206.0})
    order_after2 = broker.get_order_status(placed.order_id)
    test(
        "SL triggered at trigger price",
        order_after2.status == OrderStatus.COMPLETE,
        f"status={order_after2.status.value}"
    )


# ============================================================================
# 9. Edge Cases
# ============================================================================
def test_edge_cases():
    section("9. Edge Cases")

    broker = PaperTrader(100000)
    exit_calls = []

    def on_exit(pid, price, reason):
        exit_calls.append((pid, price, reason))

    pm = PositionMonitor(broker=broker, poll_interval=1, on_exit=on_exit)

    # Zero risk_distance -> should skip trailing entirely
    ts = TrailingState(
        position_id="EDGE-ZERO-RD",
        tradingsymbol="TEST",
        symbol="TEST",
        entry_premium=0.0,
        initial_sl=0.0,
        current_sl=0.0,
        target=100.0,
        direction="bullish",
    )
    pm._process_tick(ts, 50.0)
    test("Zero risk_distance: no crash", True)
    test("Zero risk_distance: no trailing", not ts.breakeven_set)

    # Target = 0 -> no target check
    ts2 = TrailingState(
        position_id="EDGE-NO-TGT",
        tradingsymbol="TEST2",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=0.0,
        direction="bullish",
    )
    pm._process_tick(ts2, 500.0)  # way above entry but no target
    test("No target: no target exit", len(exit_calls) == 0)

    # max_bars = 0 -> no time stop
    ts3 = TrailingState(
        position_id="EDGE-NO-TIME",
        tradingsymbol="TEST3",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
        entry_bar_count=100,
        max_bars=0,
    )
    pm._process_tick(ts3, 210.0)
    test("max_bars=0: no time stop", len(exit_calls) == 0)

    # SL exactly at current price (boundary)
    ts4 = TrailingState(
        position_id="EDGE-SL-EXACT",
        tradingsymbol="TEST4",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
    )
    pm._process_tick(ts4, 140.0)  # price = current_sl exactly
    test(
        "SL at exact boundary triggers exit",
        len(exit_calls) == 1 and exit_calls[-1][2] == "stop_loss",
    )


# ============================================================================
# 10. PositionMonitor thread lifecycle
# ============================================================================
def test_monitor_lifecycle():
    section("10. PositionMonitor Lifecycle")

    broker = PaperTrader(100000)
    pm = PositionMonitor(broker=broker, poll_interval=1)

    pm.start()
    test("Monitor running after start", pm._running)
    test("Monitor thread alive", pm._thread is not None and pm._thread.is_alive())

    ts = TrailingState(
        position_id="LIFE-001",
        tradingsymbol="TEST",
        symbol="TEST",
        entry_premium=200.0,
        initial_sl=140.0,
        current_sl=140.0,
        target=380.0,
        direction="bullish",
    )
    pm.add_position(ts)
    test("Active count = 1", pm.active_count == 1)

    positions = pm.get_positions()
    test("get_positions returns copy", "LIFE-001" in positions)

    pm.remove_position("LIFE-001")
    test("Active count = 0 after remove", pm.active_count == 0)

    pm.stop()
    time.sleep(0.5)
    test("Monitor stopped", not pm._running)

    pm.stop()
    test("Double stop doesn't crash", True)

    pm2 = PositionMonitor(broker=broker, poll_interval=1)
    pm2.start()
    pm2.start()  # second call should be no-op
    test("Double start no-op", pm2._running)
    pm2.stop()


# ============================================================================
# 11. Verify _handle_position_exit fix
# ============================================================================
def test_handle_position_exit_formatting():
    section("11. main.py _handle_position_exit formatting")

    # Read the actual code to verify the fix
    import inspect
    main_path = os.path.join(os.path.dirname(__file__), "..", "prometheus", "main.py")
    with open(main_path, "r", encoding="utf-8") as f:
        code = f.read()

    # The fixed code should have pnl_text = ... on a separate line
    has_fix = "pnl_text" in code and "_handle_position_exit" in code
    test(
        "Fixed f-string ternary in _handle_position_exit",
        has_fix,
        "Should use pnl_text variable instead of inline ternary"
    )


# ============================================================================
# 12. Verify position_monitor.py SL/target fix
# ============================================================================
def test_sl_target_direction_agnostic():
    section("12. SL/Target Direction-Agnostic (Premium-Based)")

    # Read position_monitor.py to verify fix
    pm_path = os.path.join(os.path.dirname(__file__), "..", "prometheus", "execution", "position_monitor.py")
    with open(pm_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Should NOT have direction-specific SL/target checks
    has_old_bullish_sl = 'state.direction == "bullish" and current_price <= state.current_sl' in code
    has_old_bearish_sl = 'state.direction == "bearish" and current_price >= state.current_sl' in code
    has_old_bearish_tgt = 'state.direction == "bearish" and state.target > 0 and current_price <= state.target' in code

    test("No direction-specific SL check (bullish)", not has_old_bullish_sl)
    test("No direction-specific SL check (bearish)", not has_old_bearish_sl)
    test("No direction-specific target check (bearish)", not has_old_bearish_tgt)

    # Should have direction-agnostic checks
    has_unified_sl = "current_price <= state.current_sl" in code
    has_unified_tgt = "current_price >= state.target" in code
    test("Unified SL check exists", has_unified_sl)
    test("Unified target check exists", has_unified_tgt)


# ============================================================================
# Run all tests
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PROMETHEUS Live Infrastructure Test Suite")
    print("=" * 60)

    test_trailing_state()
    test_trailing_stop_bullish()
    test_exit_conditions()
    test_trailing_stop_bearish()
    test_sqlite_persistence()
    test_order_manager()
    test_telegram_confirmation()
    test_paper_trader()
    test_edge_cases()
    test_monitor_lifecycle()
    test_handle_position_exit_formatting()
    test_sl_target_direction_agnostic()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(f"  {e}")

    sys.exit(0 if failed == 0 else 1)
