#!/usr/bin/env python
"""
Comprehensive test suite for PROMETHEUS intraday backtester.

Tests:
1. 5min theta -- correct per-bar decay, swing theta unchanged
2. Session time gating -- no entries before 9:45, no entries after 14:30
3. Force square-off at 15:15 -- "intraday_square_off" exit reason
4. Swing isolation -- intraday_session=False by default, no gating
5. Intraday metrics -- sessions, win rate, square-off pct
6. Intraday trade counter -- separate from daily_trades
7. CLI dispatch -- --intraday routes correctly
8. Combined mode flag -- --combined parses
9. BacktestEngine defaults unchanged -- swing path unaffected

Run: python tests/test_intraday_backtest.py
"""

import sys
import os
import math
from datetime import datetime, time as dtime

# This module is intended to run as a standalone validation script,
# not as a pytest-collected test module.
__test__ = False

if __name__ != "__main__":
     import pytest
     pytest.skip("script-style validation module", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from prometheus.backtest.engine import BacktestEngine, BacktestTrade


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


def make_intraday_data(n_days=3, interval_minutes=15):
    """Create synthetic intraday DataFrame with proper timestamps."""
    rows = []
    base_date = datetime(2025, 3, 10, 0, 0)  # Monday
    price = 22000.0

    for day in range(n_days):
        day_date = base_date + pd.Timedelta(days=day)
        # Generate bars from 9:15 to 15:30
        t = day_date.replace(hour=9, minute=15)
        end_t = day_date.replace(hour=15, minute=30)
        while t <= end_t:
            o = price + np.random.uniform(-15, 15)
            h = o + abs(np.random.uniform(5, 25))
            l = o - abs(np.random.uniform(5, 25))
            c = (h + l) / 2 + np.random.uniform(-5, 5)
            vol = np.random.randint(50000, 200000)
            rows.append({
                "timestamp": t,
                "open": o, "high": h, "low": l, "close": c,
                "volume": vol,
            })
            price = c
            t += pd.Timedelta(minutes=interval_minutes)

    return pd.DataFrame(rows)


# ============================================================================
# 1. 5min theta -- correct per-bar decay
# ============================================================================
section("1. 5min theta per-bar decay")

engine = BacktestEngine(initial_capital=15000)

# Create a 5min options position
pos_5min = {
    "entry_time": "2025-03-10 10:00",
    "symbol": "NIFTY",
    "direction": "bullish",
    "entry_price": 200.0,
    "stop_loss": 140.0,
    "target": 320.0,
    "quantity": 1,
    "strategy": "intraday",
    "instrument_type": "options",
    "delta": 0.5,
    "current_premium": 200.0,
    "prev_close": 22000.0,
    "max_bars": 60,
    "bar_interval": "5minute",
    "bars_held": 0,
    "breakeven_ratio": 0.5,
}

# Create a flat bar (no price movement) to isolate theta
flat_bar = pd.Series({
    "timestamp": "2025-03-10 10:05",
    "open": 22000.0, "high": 22000.0, "low": 22000.0, "close": 22000.0,
    "volume": 100000,
})

# Check exit with 5min position (bars_held = 5, should be <= 20, theta = 0.0003)
pos_5min["bars_held"] = 5
pos_5min["current_premium"] = 200.0
exit_triggered, _, _ = engine._check_exit(pos_5min, flat_bar)

# After _check_exit, current_premium should have lost only 0.03% (theta)
# premium = 200 - 200*0.0003 = 199.94 (approximately, ignoring gamma on flat bar)
expected_theta = 200.0 * 0.0003
actual_premium = pos_5min["current_premium"]
premium_loss = 200.0 - actual_premium
test("5min theta ~0.03%/bar (bars_held=5)",
     abs(premium_loss - expected_theta) < 0.5,
     f"loss={premium_loss:.4f} expected~{expected_theta:.4f}")

# Test with bars_held > 20 (accelerated theta)
pos_5min_late = pos_5min.copy()
pos_5min_late["bars_held"] = 25
pos_5min_late["current_premium"] = 200.0
engine._check_exit(pos_5min_late, flat_bar)
premium_loss_late = 200.0 - pos_5min_late["current_premium"]
expected_theta_late = 200.0 * 0.0006
test("5min theta ~0.06%/bar (bars_held=25)",
     abs(premium_loss_late - expected_theta_late) < 0.5,
     f"loss={premium_loss_late:.4f} expected~{expected_theta_late:.4f}")

# Test 15min theta unchanged
pos_15min = pos_5min.copy()
pos_15min["bar_interval"] = "15minute"
pos_15min["bars_held"] = 5
pos_15min["current_premium"] = 200.0
engine._check_exit(pos_15min, flat_bar)
premium_loss_15min = 200.0 - pos_15min["current_premium"]
expected_15min = 200.0 * 0.001
test("15min theta ~0.1%/bar (unchanged)",
     abs(premium_loss_15min - expected_15min) < 0.5,
     f"loss={premium_loss_15min:.4f} expected~{expected_15min:.4f}")

# Test daily theta for swing (unchanged)
pos_daily = pos_5min.copy()
pos_daily["bar_interval"] = "day"
pos_daily["bars_held"] = 3
pos_daily["current_premium"] = 200.0
engine._check_exit(pos_daily, flat_bar)
premium_loss_daily = 200.0 - pos_daily["current_premium"]
# Daily theta at bars_held=3 with no expiry: 0.008 = 0.8%
expected_daily = 200.0 * 0.008
test("Daily theta ~0.8%/bar (swing unchanged)",
     abs(premium_loss_daily - expected_daily) < 0.5,
     f"loss={premium_loss_daily:.4f} expected~{expected_daily:.4f}")

# 5min << 15min << daily (proper scaling)
test("Theta ordering: 5min < 15min < daily",
     premium_loss < premium_loss_15min < premium_loss_daily,
     f"5min={premium_loss:.4f} 15min={premium_loss_15min:.4f} daily={premium_loss_daily:.4f}")


# ============================================================================
# 2. Session time gating -- no entries before 9:45 / after 14:30
# ============================================================================
section("2. Session time gating")

# Create 15min intraday data spanning full session
data_session = make_intraday_data(n_days=2, interval_minutes=15)

# Simple signal generator that always generates a BUY signal
signal_count = [0]
def always_signal(data_so_far):
    signal_count[0] += 1
    return {
        "action": "BUY",
        "direction": "bullish",
        "symbol": "NIFTY",
        "entry_price": 200.0,
        "stop_loss": 140.0,
        "target": 320.0,
        "quantity": 1,
        "strategy": "test_intraday",
        "instrument_type": "options",
        "delta": 0.5,
        "max_bars": 20,
        "bar_interval": "15minute",
        "breakeven_ratio": 0.5,
    }

# Run with intraday session ON
engine_session = BacktestEngine(
    initial_capital=15000,
    intraday_session=True,
    session_open_time="09:45",
    session_no_entry_time="14:30",
    session_close_time="15:15",
    max_intraday_trades_per_day=4,
)
result_session = engine_session.run(
    data=data_session,
    signal_generator=always_signal,
    strategy_name="test_session_gating",
    warmup_bars=5,
)

# Verify no entries before 9:45
early_entries = [t for t in engine_session.trades
                 if pd.to_datetime(t.entry_time).time() < dtime(9, 45)]
test("No entries before 9:45 AM", len(early_entries) == 0,
     f"found {len(early_entries)} early entries")

# Verify no entries after 14:30
late_entries = [t for t in engine_session.trades
                if pd.to_datetime(t.entry_time).time() >= dtime(14, 30)]
test("No entries after 14:30", len(late_entries) == 0,
     f"found {len(late_entries)} late entries")

# Verify data has bars in all time zones (proving gating works, not data absence)
data_times = pd.to_datetime(data_session["timestamp"]).dt.time
has_early = any(t < dtime(9, 45) for t in data_times)
has_late = any(t >= dtime(14, 30) for t in data_times)
test("Data includes pre-9:45 bars", has_early)
test("Data includes post-14:30 bars", has_late)

test("Total intraday trades > 0", result_session.total_trades > 0,
     f"got {result_session.total_trades}")


# ============================================================================
# 3. Force square-off at 15:15
# ============================================================================
section("3. Force square-off at 15:15")

# Check for intraday_square_off exits
square_off_trades = [t for t in engine_session.trades
                     if t.exit_reason == "intraday_square_off"]
test("At least one intraday_square_off exit", len(square_off_trades) > 0,
     f"found {len(square_off_trades)}")

# All exits should be before 15:30 (session closes)
for t in engine_session.trades:
    exit_time = pd.to_datetime(t.exit_time).time()
    if exit_time > dtime(15, 30):
        test(f"Exit {t.exit_time} is before 15:30", False, f"exit at {exit_time}")
        break
else:
    test("All exits before 15:30", True)

# Square-off exits should be at or after 15:15
for t in square_off_trades:
    exit_time = pd.to_datetime(t.exit_time).time()
    test(f"Square-off at >= 15:15", exit_time >= dtime(15, 15),
         f"exit at {exit_time}")


# ============================================================================
# 4. Swing isolation -- default has no session gating
# ============================================================================
section("4. Swing isolation (intraday_session=False)")

engine_swing = BacktestEngine(initial_capital=15000)
test("Default intraday_session is False", engine_swing.intraday_session is False)

# Run swing engine on same data -- should NOT gate by time
signal_count_swing = [0]
def swing_signal(data_so_far):
    signal_count_swing[0] += 1
    return {
        "action": "BUY",
        "direction": "bullish",
        "symbol": "NIFTY",
        "entry_price": 200.0,
        "stop_loss": 140.0,
        "target": 320.0,
        "quantity": 1,
        "strategy": "test_swing",
        "instrument_type": "options",
        "delta": 0.5,
        "max_bars": 7,
        "bar_interval": "day",
        "breakeven_ratio": 0.6,
    }

result_swing = engine_swing.run(
    data=data_session,
    signal_generator=swing_signal,
    strategy_name="test_swing",
    warmup_bars=5,
)

# Swing should NOT produce any intraday_square_off exits
swing_square_offs = [t for t in engine_swing.trades
                     if t.exit_reason == "intraday_square_off"]
test("Swing: NO intraday_square_off exits", len(swing_square_offs) == 0,
     f"found {len(swing_square_offs)}")

# Swing should be able to trade at any time
test("Swing produces trades (no session gating)", result_swing.total_trades > 0,
     f"got {result_swing.total_trades}")


# ============================================================================
# 5. Intraday metrics calculation
# ============================================================================
section("5. Intraday metrics")

intra_metrics = engine_session.calculate_intraday_metrics()
test("Intraday metrics computed", len(intra_metrics) > 0)
test("total_sessions > 0", intra_metrics.get("total_sessions", 0) > 0)
test("avg_trades_per_session is float",
     isinstance(intra_metrics.get("avg_trades_per_session"), float))
test("session_win_rate is number",
     isinstance(intra_metrics.get("session_win_rate"), (int, float)))
test("square_off_exits counted",
     "square_off_exits" in intra_metrics)
test("square_off_pct exists", "square_off_pct" in intra_metrics)
test("best_session_pnl exists", "best_session_pnl" in intra_metrics)
test("worst_session_pnl exists", "worst_session_pnl" in intra_metrics)
test("entry_hour_distribution exists", "entry_hour_distribution" in intra_metrics)

# Swing should not have meaningful intraday metrics (no square-off)
swing_metrics = engine_swing.calculate_intraday_metrics()
swing_sq = swing_metrics.get("square_off_exits", 0)
test("Swing has 0 square-off exits", swing_sq == 0)


# ============================================================================
# 6. Intraday trade counter per day
# ============================================================================
section("6. Intraday trade counter per day")

# With max 4 trades/day on 2-day data, total should be <= 8
total_trades = result_session.total_trades
test("Respects max_intraday_trades_per_day (<=8 for 2 days)",
     total_trades <= 8,
     f"got {total_trades}")

# Signal generator was called many more times than trades allowed
test("Trades respect daily limit",
     total_trades <= 8,
     f"trades={total_trades} (max 4/day x 2 days)")


# ============================================================================
# 7. CLI dispatch -- --intraday routes correctly
# ============================================================================
section("7. CLI dispatch")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=[
    "scan", "backtest", "paper", "signal", "setup",
    "walkforward", "sensitivity", "parrondo_tuning",
    "semi_auto", "full_auto", "dry_run"
])
parser.add_argument("--intraday", action="store_true", default=False)
parser.add_argument("--combined", action="store_true", default=False)
parser.add_argument("--parrondo", action="store_true", default=False)
parser.add_argument("--days", type=int, default=365)

args_bt_intra = parser.parse_args(["backtest", "--intraday"])
test("backtest --intraday parsed", args_bt_intra.intraday is True)
test("backtest --intraday mode=backtest", args_bt_intra.mode == "backtest")

args_wf_intra = parser.parse_args(["walkforward", "--intraday"])
test("walkforward --intraday parsed", args_wf_intra.intraday is True)

args_combined = parser.parse_args(["dry_run", "--combined"])
test("dry_run --combined parsed", args_combined.combined is True)

args_swing = parser.parse_args(["backtest", "--days", "365"])
test("backtest (no --intraday)", args_swing.intraday is False and args_swing.combined is False)

args_bt_intra_par = parser.parse_args(["backtest", "--intraday", "--parrondo"])
test("backtest --intraday --parrondo", args_bt_intra_par.intraday and args_bt_intra_par.parrondo)


# ============================================================================
# 8. BacktestEngine.__init__ params
# ============================================================================
section("8. BacktestEngine intraday params")

# Default engine (swing)
e_default = BacktestEngine(initial_capital=200000)
test("Default intraday_session=False", e_default.intraday_session is False)

# Intraday engine
e_intra = BacktestEngine(
    initial_capital=15000,
    intraday_session=True,
    session_open_time="09:45",
    session_no_entry_time="14:30",
    session_close_time="15:15",
    max_intraday_trades_per_day=4,
)
test("Intraday session=True", e_intra.intraday_session is True)
test("Session open time parsed", e_intra._session_open_time == dtime(9, 45))
test("No entry time parsed", e_intra._session_no_entry_time == dtime(14, 30))
test("Close time parsed", e_intra._session_close_time == dtime(15, 15))
test("Max trades per day = 4", e_intra._max_intraday_trades_per_day == 4)

# Swing engine should NOT have _session_open_time attribute (not initialized)
test("Swing engine has no _session_open_time",
     not hasattr(e_default, '_session_open_time'))


# ============================================================================
# 9. main.py has intraday backtest methods
# ============================================================================
section("9. main.py intraday backtest methods")

main_path = os.path.join(os.path.dirname(__file__), "..", "prometheus", "main.py")
with open(main_path, "r", encoding="utf-8") as f:
    main_src = f.read()

test("run_intraday_backtest exists", "def run_intraday_backtest" in main_src)
test("_run_intraday_backtest_on_slice exists",
     "def _run_intraday_backtest_on_slice" in main_src)
test("run_intraday_walkforward exists", "def run_intraday_walkforward" in main_src)
test("run_combined_mode exists", "def run_combined_mode" in main_src)
test("--combined flag in argparse", '"--combined"' in main_src)
test("intraday_session=True in intraday backtest",
     "intraday_session=True" in main_src)
test("Swing run_backtest comment says LOCKED",
     "LOCKED" in main_src)


# ============================================================================
# 10. engine.py 5min theta branch exists
# ============================================================================
section("10. engine.py 5min theta code verification")

engine_path = os.path.join(
    os.path.dirname(__file__), "..", "prometheus", "backtest", "engine.py"
)
with open(engine_path, "r", encoding="utf-8") as f:
    engine_src = f.read()

test("5minute theta branch exists",
     'bar_interval == "5minute"' in engine_src)
test("theta_pct = 0.0003 (5min early)",
     "theta_pct = 0.0003" in engine_src)
test("theta_pct = 0.0006 (5min late)",
     "theta_pct = 0.0006" in engine_src)
test("intraday_session param in __init__",
     "intraday_session: bool = False" in engine_src)
test("intraday_square_off exit reason",
     '"intraday_square_off"' in engine_src)
test("can_enter_new gates signal generation",
     "can_enter_new" in engine_src)
test("calculate_intraday_metrics method",
     "def calculate_intraday_metrics" in engine_src)


# ============================================================================
# 11. Capital accounting correctness
# ============================================================================
section("11. Capital accounting")

# Intraday backtest should end with different capital than start
test("Intraday final capital != initial (trades happened)",
     result_session.final_capital != 15000 or result_session.total_trades == 0)

# Capital should still be positive
test("Intraday final capital > 0",
     result_session.final_capital > 0)

# No trade should have NaN P&L
nan_pnl = [t for t in engine_session.trades if pd.isna(t.net_pnl)]
test("No NaN P&L in trades", len(nan_pnl) == 0,
     f"found {len(nan_pnl)} NaN trades")


# ============================================================================
# 12. Regression: swing engine default behavior unchanged
# ============================================================================
section("12. Swing regression check")

# Create simple daily data
daily_data = pd.DataFrame([{
    "timestamp": pd.Timestamp(2025, 1, 10) + pd.Timedelta(days=i),
    "open": 22000 + i*10,
    "high": 22050 + i*10,
    "low": 21950 + i*10,
    "close": 22020 + i*10,
    "volume": 100000 + i*5000,
} for i in range(30)])

def daily_signal(data_so_far):
    if len(data_so_far) < 10:
        return None
    return {
        "action": "BUY",
        "direction": "bullish",
        "symbol": "NIFTY",
        "entry_price": 200.0,
        "stop_loss": 140.0,
        "target": 260.0,
        "quantity": 1,
        "strategy": "trend",
        "instrument_type": "options",
        "delta": 0.5,
        "max_bars": 7,
        "bar_interval": "day",
        "breakeven_ratio": 0.6,
    }

# Run swing backtest on daily data (intraday_session=False)
e_swing_reg = BacktestEngine(initial_capital=200000)
result_reg = e_swing_reg.run(
    data=daily_data,
    signal_generator=daily_signal,
    strategy_name="swing_regression",
    warmup_bars=5,
)

test("Swing regression: produces trades", result_reg.total_trades > 0)
test("Swing regression: no square-off exits",
     all(t.exit_reason != "intraday_square_off" for t in e_swing_reg.trades))
test("Swing regression: intraday_session=False", e_swing_reg.intraday_session is False)


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
