#!/usr/bin/env python3
"""
Multi-Index Apex Hunter Backtest
================================
Runs a full Apex Hunter backtest on every tradeable NSE F&O index
using the optimized parameters from the grid optimizer.

Generates:
  - Per-index metrics CSV
  - Per-index yearly breakdown markdown
  - Consolidated cross-index comparison summary
"""

import os
import sys
import time
import csv
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from prometheus.config import get
from prometheus.backtest.engine import BacktestEngine

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 15_000
SLIPPAGE_PCT = 0.30  # Realistic slippage

# Tradeable NSE F&O indices with dataset files
INDICES = [
    "NIFTY 50",
    "NIFTY BANK",
    "NIFTY FIN SERVICE",
]

# Optimized parameters from grid optimizer (2026-06-29)
OPTIMIZED_PARAMS = {
    "breakeven_ratio": 0.4,
    "session_open_time": "09:30",
    "max_intraday_trades": 3,
    "mr_min_score": 2.0,
}

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_csv_15m(path: Path) -> pd.DataFrame:
    """Load 15-minute OHLCV CSV with timestamp parsing."""
    print(f"  Loading CSV: {path.name}")
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

    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column in {path}. Columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0
    if "vix" in df.columns:
        df["vix"] = df["vix"].ffill().fillna(14.0)
    else:
        df["vix"] = 14.0

    print(f"    {len(df):,} bars: {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# SINGLE INDEX BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def run_backtest_for_index(symbol: str, data: pd.DataFrame) -> dict:
    """Run a full backtest for a single index using optimized params."""
    from prometheus.main import Prometheus
    from prometheus.signals.technical import calculate_vwap

    p = Prometheus(mode_override="backtest")

    capital_tracker = {"capital": INITIAL_CAPITAL, "peak": INITIAL_CAPITAL}

    # Compute regime
    regime_state = None
    if len(data) >= 50:
        regime_state = p.regime_detector.detect(data)

    # Compute VWAP bias map
    hourly_bias_map = {}
    if len(data) >= 10:
        hourly_vwap = calculate_vwap(data.copy())
        for i in range(10, len(data)):
            chunk = data.iloc[max(0, i - 20):i + 1]
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

    # Build signal generator with optimized params
    gen_params = {
        "mr_min_score": OPTIMIZED_PARAMS["mr_min_score"],
        "breakeven_ratio": OPTIMIZED_PARAMS["breakeven_ratio"],
    }

    signal_generator = p._make_signal_generator(
        regime_state=regime_state,
        hourly_bias_map=hourly_bias_map,
        capital=INITIAL_CAPITAL,
        primary_interval="15minute",
        symbol=symbol,
        param_overrides=gen_params,
        parrondo=False,
        capital_tracker=capital_tracker,
    )

    # Setup engine
    cost_cfg = dict(get("backtest.costs", {}))
    cost_cfg["slippage_pct"] = SLIPPAGE_PCT

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        cost_config=cost_cfg,
        capital_tracker=capital_tracker,
        max_positions=1 if INITIAL_CAPITAL < 30000 else 2,
        intraday_session=True,
        session_open_time=OPTIMIZED_PARAMS["session_open_time"],
        max_intraday_trades_per_day=OPTIMIZED_PARAMS["max_intraday_trades"],
    )

    result = engine.run(
        data=data,
        signal_generator=signal_generator,
        strategy_name=f"apex_{symbol.lower().replace(' ', '_')}",
        warmup_bars=30,
    )

    if not result or result.total_trades == 0:
        return None

    # Save individual trade list
    safe_name = symbol.lower().replace(" ", "_")
    trades_path = REPORT_DIR / f"apex_trades_{safe_name}.csv"
    trade_dicts = []
    for t in engine.trades:
        trade_dicts.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "gross_pnl": t.gross_pnl,
            "costs": t.costs,
            "net_pnl": t.net_pnl,
            "strategy": t.strategy,
            "exit_reason": t.exit_reason,
        })
    df_trades = pd.DataFrame(trade_dicts)
    df_trades.to_csv(trades_path, index=False)
    print(f"    Saved {len(df_trades)} trades -> {trades_path.name}")

    # Buy & hold reference
    bh_start = data["close"].iloc[30]
    bh_end = data["close"].iloc[-1]
    bh_return = (bh_end / bh_start - 1) * 100
    years = max((data["timestamp"].iloc[-1] - data["timestamp"].iloc[30]).days / 365.25, 0.01)
    bh_cagr = ((bh_end / bh_start) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        "symbol": symbol,
        "trades": result.total_trades,
        "wins": result.winning_trades,
        "losses": result.losing_trades,
        "win_rate": round(result.win_rate, 1),
        "net_pnl": round(sum(t.net_pnl for t in engine.trades), 2),
        "avg_profit": round(result.avg_win, 2),
        "avg_loss": round(result.avg_loss, 2),
        "profit_factor": round(result.profit_factor, 2),
        "total_return_pct": round(result.total_return_pct, 2),
        "cagr": round(result.annualized_return_pct, 2),
        "max_dd_pct": round(result.max_drawdown_pct, 2),
        "sharpe": round(result.sharpe_ratio, 2),
        "sortino": round(result.sortino_ratio, 2),
        "calmar": round(result.calmar_ratio, 2),
        "alpha_pct": round(result.total_return_pct - bh_return, 2),
        "psr_pct": round(result.psr_pct, 1),
        "min_trl": result.min_track_record_len,
        "max_dd_duration_days": result.max_drawdown_duration_days,
        "avg_trade_pnl": round(result.avg_trade_pnl, 2),
        "avg_hold_min": round(result.avg_hold_duration_min, 1),
        "bh_return_pct": round(bh_return, 2),
        "bh_cagr": round(bh_cagr, 2),
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": round(result.final_capital, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# YEARLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════

def generate_yearly_breakdown(symbol: str, trades_path: Path, output_md_path: Path):
    """Generate per-year performance table for an index."""
    if not trades_path.exists():
        return
    df = pd.read_csv(trades_path)
    if df.empty:
        return

    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year

    equity = INITIAL_CAPITAL
    equity_curve = [INITIAL_CAPITAL]
    for pnl in df["net_pnl"]:
        equity += pnl
        equity_curve.append(equity)

    df["equity"] = equity_curve[1:]

    yearly_stats = []
    for yr, group in df.groupby("year"):
        yr_idx = group.index
        start_idx = yr_idx[0]
        start_cap = equity_curve[start_idx]
        end_cap = equity_curve[yr_idx[-1] + 1]

        yr_return = (end_cap - start_cap) / start_cap * 100 if start_cap > 0 else 0

        yr_equity = np.array(equity_curve[start_idx: yr_idx[-1] + 2])
        peak = np.maximum.accumulate(yr_equity)
        dd = (peak - yr_equity) / peak * 100
        max_dd = dd.max()

        wins = group[group["net_pnl"] > 0]
        losses = group[group["net_pnl"] <= 0]
        win_rate = len(wins) / len(group) * 100 if len(group) > 0 else 0

        avg_win = wins["net_pnl"].mean() if not wins.empty else 0
        avg_loss = losses["net_pnl"].mean() if not losses.empty else 0

        pf = wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if losses["net_pnl"].sum() != 0 else float("inf")

        yearly_stats.append({
            "Year": yr,
            "Trades": len(group),
            "Win Rate%": round(win_rate, 1),
            "PnL": round(end_cap - start_cap, 0),
            "Start Cap": round(start_cap, 0),
            "End Cap": round(end_cap, 0),
            "Return%": round(yr_return, 1),
            "Max DD%": round(max_dd, 2),
            "PF": round(pf, 2),
            "Avg Win": round(avg_win, 0),
            "Avg Loss": round(avg_loss, 0),
        })

    df_y = pd.DataFrame(yearly_stats)

    headers = list(df_y.columns)
    table_lines = []
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df_y.iterrows():
        table_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    md_content = []
    md_content.append(f"# {symbol} — Yearly Breakdown\n")
    md_content.append(f"Optimized Parameters: breakeven_ratio={OPTIMIZED_PARAMS['breakeven_ratio']}, "
                      f"max_trades={OPTIMIZED_PARAMS['max_intraday_trades']}, "
                      f"session_open={OPTIMIZED_PARAMS['session_open_time']}\n")
    md_content.extend(table_lines)

    with open(output_md_path, "w") as f:
        f.write("\n".join(md_content))
    print(f"    Yearly breakdown -> {output_md_path.name}")
    print("\n".join(table_lines))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    print("=" * 70)
    print("  PROMETHEUS Multi-Index Apex Hunter Backtest")
    print(f"  Date: {today}  |  Capital: Rs {INITIAL_CAPITAL:,}")
    print(f"  Indices: {', '.join(INDICES)}")
    print(f"  Params: {OPTIMIZED_PARAMS}")
    print("=" * 70)

    all_results = []

    for symbol in INDICES:
        safe_name = symbol.lower().replace(" ", "_")
        csv_path = PROJECT_ROOT / "dataset" / f"{symbol}_15minute.csv"

        print(f"\n{'-' * 70}")
        print(f"  >> {symbol}")
        print(f"{'-' * 70}")

        if not csv_path.exists():
            print(f"    [!] Dataset not found: {csv_path.name}. Skipping.")
            continue

        data = load_csv_15m(csv_path)

        result = run_backtest_for_index(symbol, data)

        if result:
            all_results.append(result)
            print(f"\n    OK {result['trades']} trades | WR {result['win_rate']}% | "
                  f"PF {result['profit_factor']:.2f} | Sharpe {result['sharpe']:.2f} | "
                  f"DD {result['max_dd_pct']:.1f}% | CAGR {result['cagr']:.1f}% | "
                  f"Final Rs {result['final_capital']:,.0f}")

            # Generate yearly breakdown
            trades_path = REPORT_DIR / f"apex_trades_{safe_name}.csv"
            md_path = REPORT_DIR / f"apex_yearly_{safe_name}_{today}.md"
            try:
                generate_yearly_breakdown(symbol, trades_path, md_path)
            except Exception as e:
                print(f"    [!] Yearly breakdown failed: {e}")
        else:
            print(f"    XX No trades generated for {symbol}")

    # Save consolidated metrics
    if all_results:
        metrics_path = REPORT_DIR / f"apex_multi_index_metrics_{today}.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[REPORT] Consolidated metrics -> {metrics_path.name}")

        # Print comparison table
        print("\n" + "=" * 70)
        print("  CROSS-INDEX COMPARISON")
        print("=" * 70)
        header = f"{'Index':<22} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'CAGR%':>8} {'MaxDD%':>7} {'PSR%':>6} {'Final Cap':>14}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(f"{r['symbol']:<22} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} "
                  f"{r['sharpe']:>7.2f} {r['cagr']:>7.1f}% {r['max_dd_pct']:>6.1f}% {r['psr_pct']:>5.1f}% "
                  f"Rs {r['final_capital']:>12,.0f}")
        print("=" * 70)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Complete in {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
