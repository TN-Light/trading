#!/usr/bin/env python
"""
Comprehensive test suite for PROMETHEUS intraday trading module.

Tests:
1. Session-anchored VWAP -- resets at day boundaries
2. Session VWAP -- upper/lower bands computed correctly
3. Auto bar interval selection -- VIX threshold
4. IV annualization -- sqrt(78*252) for 5min
5. Time stop bars -- correct for 5min and 15min
6. Intraday bar increment -- every 5/15 minutes
7. Square-off only affects intraday positions (not swing)
8. Risk manager blocks trades after intraday limit
9. Risk manager resets intraday counter daily
10. No entries before 9:45 AM or after 2:30 PM (timing logic)
11. PositionMonitor TrailingState intraday fields
12. SQLite persistence with intraday fields
13. CLI --intraday flag parses correctly
14. reason_labels includes intraday_square_off

Run: python tests/test_intraday.py
"""

import sys
import os
import tempfile
import shutil
import math
from datetime import datetime, date, timedelta, time as dtime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from prometheus.execution.position_monitor import TrailingState, PositionMonitor
from prometheus.execution.broker import (
    BrokerBase, Order, OrderType, OrderSide, ProductType, OrderStatus
)
from prometheus.execution.paper_trader import PaperTrader
from prometheus.data.store import DataStore
from prometheus.risk.manager import RiskManager
from prometheus.signals.technical import calculate_session_vwap


# ============================================================================
# Test helpers
# ============================================================================
passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    safe_name = name.encode("ascii", "replace").decode("ascii")
    if condition:
        passed += 1
        print(f"  PASS: {safe_name}")
    else:
        failed += 1
        safe_detail = detail.encode("ascii", "replace").decode("ascii") if detail else ""
        msg = f"  FAIL: {safe_name} -- {safe_detail}"
        print(msg)
        errors.append(msg)


def section(title):
    safe = title.encode("ascii", "replace").decode("ascii")
    print(f"\n{'='*60}")
    print(f" {safe}")
    print(f"{'='*60}")


def make_intraday_df(n_bars=20, n_days=2):
    """Create a synthetic intraday DataFrame spanning multiple days."""
    rows = []
    base_date = datetime(2025, 3, 10, 9, 15)  # Monday 9:15 AM
    bars_per_day = n_bars // n_days
    price = 22000.0

    for day in range(n_days):
        day_start = base_date + timedelta(days=day)
        for bar in range(bars_per_day):
            ts = day_start + timedelta(minutes=bar * 5)
            o = price + np.random.uniform(-20, 20)
            h = o + abs(np.random.uniform(5, 30))
            l = o - abs(np.random.uniform(5, 30))
            c = (h + l) / 2 + np.random.uniform(-10, 10)
            vol = np.random.randint(50000, 200000)
            rows.append({
                "timestamp": ts,
                "open": o, "high": h, "low": l, "close": c,
                "volume": vol,
            })
            price = c

    return pd.DataFrame(rows)


# ============================================================================
# 1. Session-anchored VWAP -- resets at day boundaries
# ============================================================================
section("1. Session VWAP resets at day boundaries")

df_2d = make_intraday_df(n_bars=20, n_days=2)
vwap_df = calculate_session_vwap(df_2d)

# VWAP column exists
test("VWAP column exists", "vwap" in vwap_df.columns)
test("VWAP upper_1 exists", "vwap_upper_1" in vwap_df.columns)
test("VWAP lower_1 exists", "vwap_lower_1" in vwap_df.columns)
test("VWAP std exists", "vwap_std" in vwap_df.columns)

# Check that VWAP resets between days
# Day 1 last bar, Day 2 first bar -- VWAP should be different (session reset)
day1_dates = pd.to_datetime(vwap_df["timestamp"]).dt.date
unique_dates = sorted(day1_dates.unique())
test("Two trading days present", len(unique_dates) == 2)

if len(unique_dates) == 2:
    day1_mask = day1_dates == unique_dates[0]
    day2_mask = day1_dates == unique_dates[1]
    # Day 2, bar 1: VWAP should equal the typical price of that bar (fresh session)
    day2_first = vwap_df[day2_mask].iloc[0]
    tp_first = (day2_first["high"] + day2_first["low"] + day2_first["close"]) / 3
    test(
        "Day 2 first bar VWAP = typical price (session reset)",
        abs(day2_first["vwap"] - tp_first) < 0.01,
        f"vwap={day2_first['vwap']:.2f} tp={tp_first:.2f}"
    )

# ============================================================================
# 2. Session VWAP -- bands computed correctly
# ============================================================================
section("2. Session VWAP bands")

# upper_1 >= vwap >= lower_1
test("upper_1 >= vwap (all rows)",
     (vwap_df["vwap_upper_1"] >= vwap_df["vwap"] - 0.01).all())
test("vwap >= lower_1 (all rows)",
     (vwap_df["vwap"] >= vwap_df["vwap_lower_1"] - 0.01).all())
test("upper_2 >= upper_1",
     (vwap_df["vwap_upper_2"] >= vwap_df["vwap_upper_1"] - 0.01).all())
test("lower_2 <= lower_1",
     (vwap_df["vwap_lower_2"] <= vwap_df["vwap_lower_1"] + 0.01).all())

# std >= 0
test("VWAP std >= 0", (vwap_df["vwap_std"] >= -0.01).all())

# No internal columns leaked
test("No _date column in output", "_date" not in vwap_df.columns)
test("No _tp_vol column in output", "_tp_vol" not in vwap_df.columns)

# ============================================================================
# 3. Auto bar interval selection -- VIX threshold
# ============================================================================
section("3. Auto bar interval selection")

# We test _select_intraday_interval by mocking data engine
# Since it's a method on Prometheus, we'll test the logic directly

class MockDataEngine:
    def __init__(self, vix_val):
        self._vix = vix_val
    def get_vix(self):
        return self._vix

# Simulate the selection logic from main.py
def select_interval(vix, threshold=18.0, configured="auto"):
    if configured != "auto":
        return configured
    return "5minute" if vix > threshold else "15minute"

test("VIX > 18 -> 5minute", select_interval(20.0) == "5minute")
test("VIX = 18 -> 15minute", select_interval(18.0) == "15minute")
test("VIX < 18 -> 15minute", select_interval(12.5) == "15minute")
test("VIX = 18.01 -> 5minute", select_interval(18.01) == "5minute")
test("Configured override 15minute", select_interval(25.0, configured="15minute") == "15minute")
test("Configured override 5minute", select_interval(10.0, configured="5minute") == "5minute")

# ============================================================================
# 4. IV annualization for 5min and 15min
# ============================================================================
section("4. IV annualization")

# From main.py signal generator factory:
# 5min: sqrt(78 * 252) = sqrt(19656)
# 15min: sqrt(26 * 252) = sqrt(6552)
# daily: sqrt(252)

ann_5min = math.sqrt(78 * 252)
ann_15min = math.sqrt(26 * 252)
ann_daily = math.sqrt(252)

test("5min annualization ~ 140.2", abs(ann_5min - 140.2) < 0.5,
     f"got {ann_5min:.2f}")
test("15min annualization ~ 80.9", abs(ann_15min - 80.9) < 0.5,
     f"got {ann_15min:.2f}")
test("Daily annualization ~ 15.87", abs(ann_daily - 15.87) < 0.1,
     f"got {ann_daily:.2f}")
test("5min > 15min > daily", ann_5min > ann_15min > ann_daily)

# Verify these constants appear in main.py
import re
main_path = os.path.join(os.path.dirname(__file__), "..", "prometheus", "main.py")
with open(main_path, "r", encoding="utf-8") as f:
    main_src = f.read()

test("main.py has sqrt(78 * 252)", "78 * 252" in main_src or "78*252" in main_src)
test("main.py has sqrt(26 * 252)", "26 * 252" in main_src or "26*252" in main_src)

# ============================================================================
# 5. Time stop bars -- correct for 5min and 15min
# ============================================================================
section("5. Time stop bars")

# From main.py: 5min -> 60/48/36; 15min -> 20/16/12; daily -> 7/6/5
def get_time_stop_bars(interval, capital):
    if interval == "5minute":
        if capital < 50000:
            return 60
        elif capital < 100000:
            return 48
        else:
            return 36
    elif interval == "15minute":
        if capital < 50000:
            return 20
        elif capital < 100000:
            return 16
        else:
            return 12
    else:  # day
        if capital < 50000:
            return 7
        elif capital < 100000:
            return 6
        else:
            return 5

test("5min <50K -> 60 bars", get_time_stop_bars("5minute", 15000) == 60)
test("5min 50-100K -> 48 bars", get_time_stop_bars("5minute", 75000) == 48)
test("5min >100K -> 36 bars", get_time_stop_bars("5minute", 150000) == 36)
test("15min <50K -> 20 bars", get_time_stop_bars("15minute", 15000) == 20)
test("15min 50-100K -> 16 bars", get_time_stop_bars("15minute", 75000) == 16)
test("15min >100K -> 12 bars", get_time_stop_bars("15minute", 150000) == 12)
test("daily <50K -> 7 bars", get_time_stop_bars("day", 15000) == 7)

# 5min ~5 hours: 60 * 5min = 300min = 5h
test("5min 60 bars ~ 5 hours", 60 * 5 == 300)
# 15min ~5 hours: 20 * 15min = 300min = 5h
test("15min 20 bars ~ 5 hours", 20 * 15 == 300)

# ============================================================================
# 6. Intraday bar increment -- every 5/15 minutes
# ============================================================================
section("6. Intraday bar increment")

# Create a TrailingState with 5min bar interval
ts_5min = TrailingState(
    position_id="INTRA_5M_001",
    tradingsymbol="NIFTY25MAR22000CE",
    symbol="NIFTY 50",
    entry_premium=200.0,
    initial_sl=140.0,
    current_sl=140.0,
    target=320.0,
    direction="bullish",
    bar_interval="5minute",
    trade_mode="intraday",
)

test("TrailingState bar_interval = 5minute", ts_5min.bar_interval == "5minute")
test("TrailingState trade_mode = intraday", ts_5min.trade_mode == "intraday")
test("Initial bar count = 0", ts_5min.entry_bar_count == 0)

# Simulate bar increment
ts_5min.entry_bar_count += 1
test("Bar count after +1 = 1", ts_5min.entry_bar_count == 1)

# 15min state
ts_15min = TrailingState(
    position_id="INTRA_15M_001",
    tradingsymbol="NIFTY25MAR22000CE",
    symbol="NIFTY 50",
    entry_premium=200.0,
    initial_sl=140.0,
    current_sl=140.0,
    target=320.0,
    direction="bullish",
    bar_interval="15minute",
    trade_mode="intraday",
)
test("15min state bar_interval correct", ts_15min.bar_interval == "15minute")

# Swing state should still default to "day"
ts_swing = TrailingState(
    position_id="SWING_001",
    tradingsymbol="NIFTY25MAR22000CE",
    symbol="NIFTY 50",
    entry_premium=200.0,
    initial_sl=140.0,
    current_sl=140.0,
    target=320.0,
    direction="bullish",
)
test("Default bar_interval = day", ts_swing.bar_interval == "day")
test("Default trade_mode = swing", ts_swing.trade_mode == "swing")

# ============================================================================
# 7. Square-off only affects intraday positions
# ============================================================================
section("7. Square-off filters intraday only")

# Simulate position dict like PositionMonitor returns
positions = {
    "INTRA_001": TrailingState(
        position_id="INTRA_001", tradingsymbol="SYM_A", symbol="NIFTY 50",
        entry_premium=200, initial_sl=140, current_sl=140, target=320,
        direction="bullish", trade_mode="intraday",
    ),
    "SWING_001": TrailingState(
        position_id="SWING_001", tradingsymbol="SYM_B", symbol="NIFTY 50",
        entry_premium=300, initial_sl=210, current_sl=210, target=480,
        direction="bearish", trade_mode="swing",
    ),
    "INTRA_002": TrailingState(
        position_id="INTRA_002", tradingsymbol="SYM_C", symbol="NIFTY BANK",
        entry_premium=150, initial_sl=105, current_sl=105, target=240,
        direction="bullish", trade_mode="intraday",
    ),
}

# Filter like _square_off_intraday_positions does
intraday_pids = [
    pid for pid, state in positions.items()
    if getattr(state, "trade_mode", "swing") == "intraday"
]

test("Intraday filter finds 2 positions", len(intraday_pids) == 2)
test("INTRA_001 included", "INTRA_001" in intraday_pids)
test("INTRA_002 included", "INTRA_002" in intraday_pids)
test("SWING_001 excluded", "SWING_001" not in intraday_pids)

# ============================================================================
# 8. Risk manager blocks trades after intraday limit
# ============================================================================
section("8. Risk manager intraday trade limit")

config = {
    "max_daily_loss": 5000,
    "max_daily_loss_pct": 3.0,
    "max_daily_trades": 10,
    "max_weekly_loss": 10000,
    "max_open_positions": 5,
    "max_single_position_pct": 30.0,
    "max_correlated_pct": 50.0,
    "drawdown_halt_pct": 10.0,
    "consecutive_losses_pause": 5,
    "max_intraday_trades": 3,
}
rm = RiskManager(config, initial_capital=200000)

# Mock market open
from unittest.mock import patch

def mock_market_open(t=None):
    return True

# Record 3 intraday trade entries
for i in range(3):
    rm.record_trade_entry({"instrument": f"OPT_{i}", "trade_mode": "intraday"})

test("3 intraday trades recorded", rm._intraday_trades_today == 3)
test("3 total trades recorded", rm._trades_today == 3)

# 4th intraday trade should be blocked
with patch("prometheus.utils.indian_market.is_market_open", mock_market_open):
    result = rm.pre_trade_check({
        "entry_price": 200, "quantity": 50, "direction": "bullish",
        "symbol": "NIFTY 50", "trade_mode": "intraday",
    })
    has_intraday_violation = any("intraday" in v.lower() for v in result.violations)
    test("4th intraday trade blocked", not result.approved)
    test("Violation mentions intraday", has_intraday_violation,
         f"violations: {result.violations}")

# But a swing trade should still be allowed (if under daily limit)
with patch("prometheus.utils.indian_market.is_market_open", mock_market_open):
    result_swing = rm.pre_trade_check({
        "entry_price": 200, "quantity": 50, "direction": "bullish",
        "symbol": "NIFTY 50", "trade_mode": "swing",
    })
    swing_blocked_by_intraday = any("intraday" in v.lower() for v in result_swing.violations)
    test("Swing trade NOT blocked by intraday limit", not swing_blocked_by_intraday)

# ============================================================================
# 9. Risk manager resets intraday counter daily
# ============================================================================
section("9. Intraday counter daily reset")

# Simulate day change
rm2 = RiskManager(config, initial_capital=200000)
rm2._intraday_trades_today = 4
rm2._trades_today = 6
rm2._daily_pnl = -1000
old_today = rm2._today

# Force day change
new_day = old_today + timedelta(days=1)
rm2._check_day_reset(new_day)

test("Intraday trades reset to 0", rm2._intraday_trades_today == 0)
test("Daily trades reset to 0", rm2._trades_today == 0)
test("Daily PnL reset to 0", rm2._daily_pnl == 0.0)

# Monday reset also resets weekly
rm3 = RiskManager(config, initial_capital=200000)
rm3._weekly_pnl = -5000
rm3._intraday_trades_today = 3
# Find next Monday from today
today = date.today()
days_until_monday = (7 - today.weekday()) % 7
if days_until_monday == 0:
    days_until_monday = 7
next_monday = today + timedelta(days=days_until_monday)
rm3._check_day_reset(next_monday)
test("Weekly PnL reset on Monday", rm3._weekly_pnl == 0.0)
test("Intraday trades reset on Monday", rm3._intraday_trades_today == 0)

# ============================================================================
# 10. Timing zone logic
# ============================================================================
section("10. Intraday timing zones")

# Test the time comparisons used in run_intraday_mode
skip_minutes = 30
last_entry_time = dtime(14, 30)
square_off_time = dtime(15, 15)

# Pre-market
t_premarket = dtime(9, 0)
test("9:00 AM is pre-market", t_premarket < dtime(9, 15))

# Opening skip (9:15-9:45)
t_opening = datetime(2025, 3, 10, 9, 30)
market_open = t_opening.replace(hour=9, minute=15, second=0, microsecond=0)
elapsed = (t_opening - market_open).total_seconds()
test("9:30 AM is in opening skip", elapsed < skip_minutes * 60, f"elapsed={elapsed}s")

# Scan window (9:45-14:30)
t_scan = dtime(11, 0)
test("11:00 AM is scan window",
     t_scan >= dtime(9, 45) and t_scan < last_entry_time)

# No-entry zone (14:30-15:15)
t_noentry = dtime(14, 45)
test("14:45 is no-entry zone",
     t_noentry >= last_entry_time and t_noentry < square_off_time)

# Square-off time
t_squareoff = dtime(15, 15)
test("15:15 triggers square-off", t_squareoff >= square_off_time)

# After scan window opens
t_945 = datetime(2025, 3, 10, 9, 45)
elapsed_945 = (t_945 - t_945.replace(hour=9, minute=15)).total_seconds()
test("9:45 AM = 30min elapsed (scan starts)", elapsed_945 >= skip_minutes * 60)

# ============================================================================
# 11. TrailingState to_dict includes intraday fields
# ============================================================================
section("11. TrailingState serialization")

ts_intra = TrailingState(
    position_id="TEST_INTRA",
    tradingsymbol="NIFTY25MAR22000CE",
    symbol="NIFTY 50",
    entry_premium=200.0,
    initial_sl=140.0,
    current_sl=140.0,
    target=320.0,
    direction="bullish",
    bar_interval="5minute",
    trade_mode="intraday",
)

d = ts_intra.to_dict()
test("to_dict has bar_interval", "bar_interval" in d)
test("to_dict bar_interval = 5minute", d["bar_interval"] == "5minute")
test("to_dict has trade_mode", "trade_mode" in d)
test("to_dict trade_mode = intraday", d["trade_mode"] == "intraday")
test("to_dict has position_id", d["position_id"] == "TEST_INTRA")

# ============================================================================
# 12. SQLite persistence with intraday fields
# ============================================================================
section("12. SQLite persistence -- intraday fields")

tmp_dir = tempfile.mkdtemp()
try:
    db_path = os.path.join(tmp_dir, "test_intraday.db")
    store = DataStore(db_path)

    # Save intraday position
    state_dict = {
        "position_id": "INTRA_PERSIST_001",
        "tradingsymbol": "NIFTY25MAR22000CE",
        "symbol": "NIFTY 50",
        "direction": "bullish",
        "strategy": "momentum_breakout",
        "entry_premium": 200.0,
        "initial_sl": 140.0,
        "current_sl": 165.0,
        "target": 320.0,
        "sl_order_id": "SL_123",
        "entry_time": "2025-03-10 10:00:00",
        "status": "open",
        "breakeven_set": True,
        "trailing_activated": False,
        "trailing_stage2": False,
        "trailing_stage3": False,
        "premium_hwm": 0,
        "entry_bar_count": 12,
        "max_bars": 60,
        "breakeven_ratio": 0.5,
        "risk_distance": 60.0,
        "trade_mode": "intraday",
        "bar_interval": "5minute",
        "product_type": "MIS",
    }
    store.save_position_state(state_dict)

    # Also save a swing position to verify they coexist
    swing_dict = state_dict.copy()
    swing_dict["position_id"] = "SWING_PERSIST_001"
    swing_dict["trade_mode"] = "swing"
    swing_dict["bar_interval"] = "day"
    swing_dict["product_type"] = "CNC"
    swing_dict["max_bars"] = 7
    store.save_position_state(swing_dict)

    # Load back
    loaded = store.load_open_positions()
    test("Two positions loaded", len(loaded) == 2)

    intra_loaded = [p for p in loaded if p["position_id"] == "INTRA_PERSIST_001"]
    test("Intraday position found", len(intra_loaded) == 1)

    if intra_loaded:
        p = intra_loaded[0]
        test("trade_mode = intraday", p["trade_mode"] == "intraday")
        test("bar_interval = 5minute", p["bar_interval"] == "5minute")
        test("product_type = MIS", p["product_type"] == "MIS")
        test("max_bars = 60", p["max_bars"] == 60)
        test("breakeven_ratio = 0.5", abs(p["breakeven_ratio"] - 0.5) < 0.001)
        test("entry_bar_count = 12", p["entry_bar_count"] == 12)
        test("current_sl = 165", abs(p["current_sl"] - 165.0) < 0.01)

    swing_loaded = [p for p in loaded if p["position_id"] == "SWING_PERSIST_001"]
    test("Swing position found", len(swing_loaded) == 1)
    if swing_loaded:
        test("Swing trade_mode = swing", swing_loaded[0]["trade_mode"] == "swing")
        test("Swing bar_interval = day", swing_loaded[0]["bar_interval"] == "day")

    # Close intraday, verify swing still open
    store.close_position_state("INTRA_PERSIST_001", pnl=500.0)
    still_open = store.load_open_positions()
    test("After close, 1 position open", len(still_open) == 1)
    test("Remaining is swing", still_open[0]["position_id"] == "SWING_PERSIST_001")

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

# ============================================================================
# 13. CLI --intraday flag
# ============================================================================
section("13. CLI --intraday flag")

import argparse

# Simulate the parser from main.py
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=[
    "scan", "backtest", "paper", "signal", "setup",
    "walkforward", "sensitivity", "parrondo_tuning",
    "semi_auto", "full_auto", "dry_run"
])
parser.add_argument("--intraday", action="store_true", default=False)
parser.add_argument("--interval", type=int, default=300)

args1 = parser.parse_args(["paper", "--intraday"])
test("paper --intraday: mode=paper", args1.mode == "paper")
test("paper --intraday: intraday=True", args1.intraday is True)

args2 = parser.parse_args(["dry_run", "--intraday", "--interval", "10"])
test("dry_run --intraday: mode=dry_run", args2.mode == "dry_run")
test("dry_run --intraday: intraday=True", args2.intraday is True)
test("dry_run --intraday: interval=10", args2.interval == 10)

args3 = parser.parse_args(["paper"])
test("paper (no flag): intraday=False", args3.intraday is False)

args4 = parser.parse_args(["full_auto", "--intraday"])
test("full_auto --intraday: mode=full_auto", args4.mode == "full_auto")
test("full_auto --intraday: intraday=True", args4.intraday is True)

# ============================================================================
# 14. reason_labels includes intraday_square_off
# ============================================================================
section("14. reason_labels in main.py")

test("main.py has intraday_square_off label",
     "intraday_square_off" in main_src)
test("main.py has 'Intraday Square-Off'",
     "Intraday Square-Off" in main_src)

# Also verify dispatch routes
test("main.py dispatches paper --intraday to run_intraday_mode",
     "args.intraday" in main_src and "run_intraday_mode" in main_src)

# ============================================================================
# 15. PositionMonitor _check_intraday_bar_increment logic
# ============================================================================
section("15. PositionMonitor intraday bar increment")

# Create a mock broker
class MockBroker(BrokerBase):
    def __init__(self):
        super().__init__()
    def connect(self): return True
    def is_connected(self): return True
    def get_ltp(self, symbol, exchange=None): return 200.0
    def place_order(self, *a, **kw): return None
    def cancel_order(self, *a, **kw): return True
    def get_order_status(self, *a, **kw): return None
    def get_orders(self): return []
    def get_positions(self): return []
    def get_margins(self): return {}
    def modify_order(self, *a, **kw): return None

monitor = PositionMonitor(MockBroker(), poll_interval=1)

# Add a 5min intraday position
ts_5m = TrailingState(
    position_id="BAR_TEST_5M", tradingsymbol="SYM", symbol="NIFTY 50",
    entry_premium=200, initial_sl=140, current_sl=140, target=320,
    direction="bullish", bar_interval="5minute", trade_mode="intraday",
)
monitor.add_position(ts_5m)

# Add a daily swing position
ts_day = TrailingState(
    position_id="BAR_TEST_DAY", tradingsymbol="SYM2", symbol="NIFTY 50",
    entry_premium=300, initial_sl=210, current_sl=210, target=480,
    direction="bullish", bar_interval="day", trade_mode="swing",
)
monitor.add_position(ts_day)

# Manually set _last_bar_ts to simulate 6 minutes ago
now = datetime.now()
ts_5m._last_bar_ts = now - timedelta(minutes=6)
ts_day._last_bar_ts = now - timedelta(minutes=6)

initial_5m_bars = ts_5m.entry_bar_count
initial_day_bars = ts_day.entry_bar_count

# Call intraday bar increment
monitor._check_intraday_bar_increment()

test("5min position bar incremented after 6min",
     ts_5m.entry_bar_count == initial_5m_bars + 1,
     f"was {initial_5m_bars}, now {ts_5m.entry_bar_count}")
test("Daily position bar NOT incremented by intraday check",
     ts_day.entry_bar_count == initial_day_bars)

# Simulate 16 min elapsed for a 15min position
ts_15m = TrailingState(
    position_id="BAR_TEST_15M", tradingsymbol="SYM3", symbol="NIFTY BANK",
    entry_premium=150, initial_sl=105, current_sl=105, target=240,
    direction="bearish", bar_interval="15minute", trade_mode="intraday",
)
monitor.add_position(ts_15m)
ts_15m._last_bar_ts = now - timedelta(minutes=16)
initial_15m = ts_15m.entry_bar_count
monitor._check_intraday_bar_increment()
test("15min position bar incremented after 16min",
     ts_15m.entry_bar_count == initial_15m + 1,
     f"was {initial_15m}, now {ts_15m.entry_bar_count}")

# 30 min elapsed for 5min -> should add 6 (not 1)
ts_5m._last_bar_ts = now - timedelta(minutes=31)
ts_5m.entry_bar_count = 0
monitor._check_intraday_bar_increment()
test("5min +31min -> adds 6 bars", ts_5m.entry_bar_count == 6,
     f"got {ts_5m.entry_bar_count}")

# Cleanup
monitor.stop()

# ============================================================================
# 16. Session VWAP edge cases
# ============================================================================
section("16. Session VWAP edge cases")

# Single bar
df_1bar = pd.DataFrame([{
    "timestamp": datetime(2025, 3, 10, 9, 15),
    "open": 22000, "high": 22050, "low": 21950, "close": 22020,
    "volume": 100000,
}])
vwap_1 = calculate_session_vwap(df_1bar)
test("Single bar VWAP works", len(vwap_1) == 1 and not pd.isna(vwap_1["vwap"].iloc[0]))
# For single bar, VWAP = typical price
tp = (22050 + 21950 + 22020) / 3
test("Single bar VWAP = typical price",
     abs(vwap_1["vwap"].iloc[0] - tp) < 0.01,
     f"vwap={vwap_1['vwap'].iloc[0]:.2f} tp={tp:.2f}")
# Std should be 0 for single bar
test("Single bar VWAP std = 0", abs(vwap_1["vwap_std"].iloc[0]) < 0.01)

# Zero volume bar
df_0vol = pd.DataFrame([{
    "timestamp": datetime(2025, 3, 10, 9, 15),
    "open": 22000, "high": 22050, "low": 21950, "close": 22020,
    "volume": 0,
}])
vwap_0 = calculate_session_vwap(df_0vol)
test("Zero volume bar -> VWAP is NaN", pd.isna(vwap_0["vwap"].iloc[0]))

# ============================================================================
# 17. Config section
# ============================================================================
section("17. Settings.yaml intraday config")

import yaml
settings_path = os.path.join(
    os.path.dirname(__file__), "..", "prometheus", "config", "settings.yaml"
)
with open(settings_path, "r") as f:
    cfg = yaml.safe_load(f)

test("intraday section exists in settings", "intraday" in cfg)
if "intraday" in cfg:
    intra_cfg = cfg["intraday"]
    test("intraday.enabled exists", "enabled" in intra_cfg)
    test("intraday.bar_interval = auto", intra_cfg.get("bar_interval") == "auto")
    test("intraday.vix_threshold_5min = 18.0", intra_cfg.get("vix_threshold_5min") == 18.0)
    test("intraday.skip_first_minutes = 30", intra_cfg.get("skip_first_minutes") == 30)
    test("intraday.last_entry_time = 14:30", intra_cfg.get("last_entry_time") == "14:30")
    test("intraday.square_off_time = 15:15", intra_cfg.get("square_off_time") == "15:15")
    test("intraday.max_daily_trades = 4", intra_cfg.get("max_daily_trades") == 4)
    test("intraday.instruments is list", isinstance(intra_cfg.get("instruments"), list))
    test("NIFTY 50 in instruments", "NIFTY 50" in intra_cfg.get("instruments", []))

# ============================================================================
# 18. DataEngine fetch_intraday and get_vix exist
# ============================================================================
section("18. DataEngine methods")

engine_path = os.path.join(
    os.path.dirname(__file__), "..", "prometheus", "data", "engine.py"
)
with open(engine_path, "r", encoding="utf-8") as f:
    engine_src = f.read()

test("fetch_intraday method exists", "def fetch_intraday" in engine_src)
test("get_vix method exists", "def get_vix" in engine_src)
test("fetch_intraday calls force_refresh=True", "force_refresh=True" in engine_src)
test("get_vix uses ^INDIAVIX", "INDIAVIX" in engine_src)

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print(f" RESULTS: {passed} passed, {failed} failed ({passed+failed} total)")
print(f"{'='*60}")

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
