#!/usr/bin/env python3
"""
Breakeven Ratio Sensitivity Check

Runs 3 backtests with breakeven_ratio = {0.4, 0.5, 0.6}.
All other parameters locked to the optimizer's winning values:
  - session_open_time = "09:30"
  - max_intraday_trades = 3
  - mr_min_score = 2.0 (irrelevant but kept for consistency)

Each backtest runs on BOTH in-sample (2015-2022) and out-of-sample (2023+)
independently so we can compare stability across both periods.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Stub out network-dependent modules before any Prometheus import
import prometheus.data.angelone_fetcher
prometheus.data.angelone_fetcher.create_angelone_fetcher = lambda *a, **kw: None
import prometheus.data.angelone_options
prometheus.data.angelone_options.create_angelone_option_chain = lambda *a, **kw: None
import prometheus.interface.telegram_bot
prometheus.interface.telegram_bot.TelegramBot.connect = lambda self: None

from prometheus.config import get
from prometheus.backtest.engine import BacktestEngine
from prometheus.main import Prometheus
from prometheus.signals.technical import calculate_vwap

# ── Constants ──────────────────────────────────────────────────────────
SYMBOL = "NIFTY 50"
INITIAL_CAPITAL = 15_000
CSV_PATH = PROJECT_ROOT / "dataset" / f"{SYMBOL}_15minute.csv"

# Locked parameters
LOCKED = {
    "session_open_time": "09:30",
    "max_intraday_trades": 3,
    "mr_min_score": 2.0,
}

# The single variable under test
TEST_VALUES = [0.4, 0.5, 0.6]

TRAIN_END = 2022
TEST_START = 2023


def load_data(path):
    df = pd.read_csv(path)
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "datetime", "time", "timestamp"):
            col_map[c] = "timestamp"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl in ("volume", "vol"):
            col_map[c] = "volume"
        elif cl == "vix":
            col_map[c] = "vix"
    df = df.rename(columns=col_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "vix" in df.columns:
        df["vix"] = df["vix"].ffill().fillna(14.0)
    else:
        df["vix"] = 14.0
    return df


def build_bias_map(data):
    bias_map = {}
    if len(data) < 10:
        return bias_map
    hourly_vwap = calculate_vwap(data.copy())
    for i in range(10, len(data)):
        chunk = data.iloc[max(0, i - 20):i + 1]
        date_key = str(chunk["timestamp"].iloc[-1])[:10]
        recent_h = chunk.tail(5)
        highs = recent_h["high"].values
        lows = recent_h["low"].values
        hh = sum(1 for j in range(1, len(highs)) if highs[j] > highs[j - 1])
        hl = sum(1 for j in range(1, len(lows)) if lows[j] > lows[j - 1])
        lh = sum(1 for j in range(1, len(highs)) if highs[j] < highs[j - 1])
        ll = sum(1 for j in range(1, len(lows)) if lows[j] < lows[j - 1])
        close_val = recent_h["close"].iloc[-1]
        vwap_val = hourly_vwap["vwap"].iloc[i] if "vwap" in hourly_vwap.columns and i < len(hourly_vwap) else close_val
        bull = hh + hl + (1 if close_val > vwap_val else 0)
        bear = lh + ll + (1 if close_val < vwap_val else 0)
        if bull >= 4 and bull > bear + 1:
            bias_map[date_key] = "bullish"
        elif bear >= 4 and bear > bull + 1:
            bias_map[date_key] = "bearish"
        else:
            bias_map[date_key] = "neutral"
    return bias_map


def run_single_backtest(data, regime_state, bias_map, breakeven_ratio, label):
    """Run one backtest and return a metrics dict."""
    p = Prometheus(mode_override="backtest")
    capital_tracker = {"capital": INITIAL_CAPITAL, "peak": INITIAL_CAPITAL}

    gen_params = {
        "mr_min_score": LOCKED["mr_min_score"],
        "breakeven_ratio": breakeven_ratio,
    }

    signal_generator = p._make_signal_generator(
        regime_state=regime_state,
        hourly_bias_map=bias_map,
        capital=INITIAL_CAPITAL,
        primary_interval="15minute",
        symbol=SYMBOL,
        param_overrides=gen_params,
        parrondo=False,
        capital_tracker=capital_tracker,
    )

    cost_cfg = dict(get("backtest.costs", {}))
    cost_cfg["slippage_pct"] = 0.30

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        cost_config=cost_cfg,
        capital_tracker=capital_tracker,
        max_positions=1,
        intraday_session=True,
        session_open_time=LOCKED["session_open_time"],
        max_intraday_trades_per_day=LOCKED["max_intraday_trades"],
    )

    result = engine.run(
        data=data,
        signal_generator=signal_generator,
        strategy_name="sensitivity",
        warmup_bars=30,
    )

    if not result or result.total_trades < 5:
        return {
            "label": label,
            "breakeven_ratio": breakeven_ratio,
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "cagr": 0,
            "max_dd_pct": 0,
            "sharpe": 0,
            "net_pnl": 0,
        }

    return {
        "label": label,
        "breakeven_ratio": breakeven_ratio,
        "trades": result.total_trades,
        "win_rate": round(result.win_rate, 2),
        "profit_factor": round(result.profit_factor, 2),
        "cagr": round(result.annualized_return_pct, 2),
        "max_dd_pct": round(result.max_drawdown_pct, 2),
        "sharpe": round(result.sharpe_ratio, 2),
        "net_pnl": round(result.final_capital - INITIAL_CAPITAL, 2),
    }


def main():
    start = time.time()
    print("=" * 70)
    print("  BREAKEVEN RATIO SENSITIVITY CHECK")
    print("  Locked: session_open=09:30, max_trades=3, mr_min_score=2.0")
    print("  Testing: breakeven_ratio = {0.4, 0.5, 0.6}")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data(CSV_PATH)
    train = df[df["timestamp"].dt.year <= TRAIN_END].copy().reset_index(drop=True)
    test = df[df["timestamp"].dt.year >= TEST_START].copy().reset_index(drop=True)
    print(f"  Train: {len(train):,} bars (2015-{TRAIN_END})")
    print(f"  Test:  {len(test):,} bars ({TEST_START}+)")

    # Pre-compute regimes and bias maps for both slices
    print("\nPre-computing regimes and bias maps...")
    p_dummy = Prometheus(mode_override="backtest")

    train_regime = p_dummy.regime_detector.detect(train)
    train_bias = build_bias_map(train)
    print("  Train regime + bias: done")

    test_regime = p_dummy.regime_detector.detect(test)
    test_bias = build_bias_map(test)
    print("  Test regime + bias: done")

    results = []

    for bev in TEST_VALUES:
        print(f"\n{'─' * 50}")
        print(f"  breakeven_ratio = {bev}")
        print(f"{'─' * 50}")

        # In-Sample
        print(f"  Running IS backtest (2015-{TRAIN_END})...")
        t0 = time.time()
        is_res = run_single_backtest(train, train_regime, train_bias, bev, f"IS_{bev}")
        print(f"    Done in {time.time() - t0:.0f}s — {is_res['trades']} trades, "
              f"WR={is_res['win_rate']}%, PF={is_res['profit_factor']}, "
              f"CAGR={is_res['cagr']}%, DD={is_res['max_dd_pct']}%")
        results.append(is_res)

        # Out-of-Sample
        print(f"  Running OOS backtest ({TEST_START}+)...")
        t0 = time.time()
        oos_res = run_single_backtest(test, test_regime, test_bias, bev, f"OOS_{bev}")
        print(f"    Done in {time.time() - t0:.0f}s — {oos_res['trades']} trades, "
              f"WR={oos_res['win_rate']}%, PF={oos_res['profit_factor']}, "
              f"CAGR={oos_res['cagr']}%, DD={oos_res['max_dd_pct']}%")
        results.append(oos_res)

    # Save results
    out_path = PROJECT_ROOT / "reports" / "breakeven_sensitivity_check.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("  SENSITIVITY COMPARISON TABLE")
    print("=" * 70)
    df_r = pd.DataFrame(results)
    print(df_r.to_string(index=False))
    print("=" * 70)

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
