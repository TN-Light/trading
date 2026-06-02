#!/usr/bin/env python3
"""
Quick 3-scenario backtest for fast iteration.
Runs ONLY the critical comparison scenarios:
  1. base/full  (direct comparison to May 4 baseline)
  2. base/10y   (recent decade)
  3. slip_0.30/full (realistic slippage)

~45 min total vs ~6+ hours for full 22-scenario suite.
"""

import os
import sys
import time
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from prometheus.config import get
from prometheus.backtest.engine import BacktestEngine

SYMBOL = "NIFTY 50"
INITIAL_CAPITAL = 15_000
CSV_PATH = PROJECT_ROOT / "dataset" / f"{SYMBOL}_15minute.csv"


def load_data():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
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
    df = df.rename(columns=col_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} bars: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    return df


def run_scenario(df, label, slippage_pct=0.0):
    """Run one backtest scenario."""
    from prometheus.main import Prometheus
    from prometheus.signals.technical import calculate_vwap

    t0 = time.time()
    p = Prometheus()
    p.initial_capital = INITIAL_CAPITAL
    p.capital = INITIAL_CAPITAL

    cost_cfg = dict(get("backtest.costs", {}))
    cost_cfg["slippage_pct"] = slippage_pct

    capital_tracker = {"capital": INITIAL_CAPITAL, "peak": INITIAL_CAPITAL}

    regime_state = None
    if len(df) >= 50:
        regime_state = p.regime_detector.detect(df)

    hourly_bias_map = {}
    if len(df) >= 10:
        hourly_vwap = calculate_vwap(df.copy())
        for i in range(10, len(df)):
            chunk = df.iloc[max(0, i - 20):i + 1]
            date_key = str(chunk["timestamp"].iloc[-1])[:10]
            recent_h = chunk.tail(5)
            highs = recent_h["high"].values
            lows = recent_h["low"].values
            hh_count = sum(1 for j in range(1, len(highs)) if highs[j] > highs[j - 1])
            hl_count = sum(1 for j in range(1, len(lows)) if lows[j] > lows[j - 1])
            lh_count = sum(1 for j in range(1, len(highs)) if highs[j] < highs[j - 1])
            ll_count = sum(1 for j in range(1, len(lows)) if lows[j] < lows[j - 1])
            close_val = recent_h["close"].iloc[-1]
            vwap_val = hourly_vwap["vwap"].iloc[i] if "vwap" in hourly_vwap.columns and i < len(hourly_vwap) else close_val
            bull_points = hh_count + hl_count + (1 if close_val > vwap_val else 0)
            bear_points = lh_count + ll_count + (1 if close_val < vwap_val else 0)
            if bull_points >= 4 and bull_points > bear_points + 1:
                hourly_bias_map[date_key] = "bullish"
            elif bear_points >= 4 and bear_points > bull_points + 1:
                hourly_bias_map[date_key] = "bearish"
            else:
                hourly_bias_map[date_key] = "neutral"

    p.regime_detector.reset_cache()
    signal_generator = p._make_signal_generator(
        regime_state=regime_state,
        hourly_bias_map=hourly_bias_map,
        capital=INITIAL_CAPITAL,
        primary_interval="15minute",
        symbol=SYMBOL,
        param_overrides={"mr_min_score": 2.5},
        parrondo=False,
        capital_tracker=capital_tracker,
    )

    max_pos = 1 if INITIAL_CAPITAL < 30000 else 2
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        cost_config=cost_cfg,
        capital_tracker=capital_tracker,
        max_positions=max_pos,
    )

    result = engine.run(
        data=df,
        signal_generator=signal_generator,
        strategy_name=f"swing_15m_{label}",
        warmup_bars=30,
    )

    elapsed = time.time() - t0
    return result, elapsed


def main():
    print("=" * 70)
    print(f"  QUICK BACKTEST  |  {SYMBOL}  |  Rs {INITIAL_CAPITAL:,}")
    print("=" * 70)

    df = load_data()

    # 10-year slice
    end = df["timestamp"].iloc[-1]
    start_10y = end - pd.DateOffset(years=10)
    df_10y = df[df["timestamp"] >= start_10y].reset_index(drop=True)

    scenarios = [
        ("base_full", df, 0.0),
        ("base_10y", df_10y, 0.0),
        ("slip_0.30_full", df, 0.30),
    ]

    # May 4 baseline for comparison
    baseline = {
        "base_full": {"PF": 12.55, "Sharpe": 1.92, "DD": 9.39, "Trades": 1159, "Final": 1027000},
        "base_10y": {"PF": 12.55, "Sharpe": 1.92, "DD": 9.39, "Trades": 1159, "Final": 1027000},
        "slip_0.30_full": {"PF": 11.20, "Sharpe": 1.90, "DD": 9.50, "Trades": 1159, "Final": 995000},
    }

    results = []
    for label, data, slip in scenarios:
        n_bars = len(data)
        print(f"\n{'─' * 50}")
        print(f"  {label}  ({n_bars:,} bars, slip={slip}%)")
        print(f"{'─' * 50}")

        result, elapsed = run_scenario(data, label, slip)
        if result and result.total_trades > 0:
            r = result
            print(f"  ✓ {r.total_trades} trades | PF {r.profit_factor:.2f} | "
                  f"Sharpe {r.sharpe_ratio:.2f} | DD {r.max_drawdown_pct:.1f}% | "
                  f"Final Rs {r.final_capital:,.0f} | {elapsed:.0f}s")
            results.append({
                "label": label,
                "trades": r.total_trades,
                "pf": r.profit_factor,
                "sharpe": r.sharpe_ratio,
                "sortino": r.sortino_ratio,
                "calmar": r.calmar_ratio,
                "dd": r.max_drawdown_pct,
                "cagr": r.annualized_return_pct,
                "final": r.final_capital,
                "wr": r.win_rate,
            })
        else:
            print(f"  ✗ No trades | {elapsed:.0f}s")

    # Summary comparison
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON vs MAY 4 BASELINE")
    print(f"{'=' * 80}")
    print(f"{'Metric':<12} | {'May4 Base/Full':>14} | {'New Base/Full':>14} | {'Delta':>10}")
    print(f"{'─' * 60}")

    if results:
        r = results[0]  # base_full
        b = baseline["base_full"]
        comparisons = [
            ("Trades", b["Trades"], r["trades"]),
            ("PF", b["PF"], r["pf"]),
            ("Sharpe", b["Sharpe"], r["sharpe"]),
            ("Sortino", 2.44, r["sortino"]),
            ("Calmar", 4.97, r["calmar"]),
            ("Max DD%", b["DD"], r["dd"]),
            ("CAGR%", 46.66, r["cagr"]),
            ("Win Rate%", 61.3, r["wr"]),
            ("Final Rs", b["Final"], r["final"]),
        ]
        for name, old, new in comparisons:
            if isinstance(new, float):
                delta = new - old
                arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
                # For DD, lower is better
                if name == "Max DD%":
                    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
                    color = "WORSE" if delta > 0 else "BETTER"
                else:
                    color = "BETTER" if delta > 0 else "WORSE"
                print(f"{name:<12} | {old:>14.2f} | {new:>14.2f} | {arrow} {abs(delta):>7.2f} {color}")
            else:
                delta = new - old
                arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
                print(f"{name:<12} | {old:>14,} | {new:>14,} | {arrow} {abs(delta):>7,}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
