#!/usr/bin/env python3
"""
Fast Apex Hunter Backtest -- Full metrics suite for comparison.

Reproduces the same metrics as the May 4th baseline test:
  - Base (0% slippage) + Slip (0.30% slippage)
  - Full-run, 10y, 5y, Train(2015-2019), Test(2020+)
  - Crisis months: 2020-03..06, 2022-02, 2024-06
  - Regime + Volatility summaries

Optimized for speed:
  - Parallel scenario execution via ProcessPoolExecutor
  - Numpy-vectorized metrics
  - CSV data pre-loaded once, shared across runs
"""

import os
import sys
import time
import csv
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

warnings.filterwarnings("ignore")

# -- Project root --
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from prometheus.config import get
from prometheus.backtest.engine import BacktestEngine, BacktestResult


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

SYMBOL = "NIFTY 50"
INITIAL_CAPITAL = 15_000
CSV_15M_PATH = PROJECT_ROOT / "dataset" / f"{SYMBOL}_15minute.csv"

SLIPPAGE_SCENARIOS = [
    ("slip_0.30", 0.30),
]

FULL_RUN_PERIODS = ["full"]
HOLDOUT_PERIODS = []
CRISIS_MONTHS = []

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_csv_15m(path: Path) -> pd.DataFrame:
    """Load 15-minute OHLCV CSV with timestamp parsing."""
    print(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "datetime", "time", "timestamp"):
            col_map[c] = "timestamp"
        elif cl in ("open",):
            col_map[c] = "open"
        elif cl in ("high",):
            col_map[c] = "high"
        elif cl in ("low",):
            col_map[c] = "low"
        elif cl in ("close",):
            col_map[c] = "close"
        elif cl in ("volume", "vol"):
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column found in {path}. Columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0
    print(f"  Loaded {len(df):,} bars: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    return df


def slice_data(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    """Slice data by date range."""
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["timestamp"] >= pd.Timestamp(start)
    if end:
        mask &= df["timestamp"] <= pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[mask].reset_index(drop=True)


def slice_by_month(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    """Slice data for a specific YYYY-MM month."""
    year, month = int(ym[:4]), int(ym[5:7])
    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1)
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df[mask].reset_index(drop=True)


def slice_last_n_years(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Slice the last N years of data."""
    end = df["timestamp"].iloc[-1]
    start = end - pd.DateOffset(years=n)
    return df[df["timestamp"] >= start].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST RUNNER (single scenario)
# ═══════════════════════════════════════════════════════════════════════

def run_single_backtest(
    data: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    slippage_pct: float,
    scenario_label: str,
    period_label: str,
) -> dict:
    """Run one backtest scenario and return metrics dict."""
    if data.empty or len(data) < 100:
        return None

    from prometheus.main import Prometheus

    p = Prometheus()
    p.initial_capital = initial_capital
    p.capital = initial_capital

    cost_cfg = dict(get("backtest.costs", {}))
    cost_cfg["slippage_pct"] = slippage_pct

    capital_tracker = {"capital": initial_capital, "peak": initial_capital}

    # Build signal generator - use p.regime_detector (already initialized)
    regime_state = None
    if len(data) >= 50:
        regime_state = p.regime_detector.detect(data)

    hourly_bias_map = {}
    from prometheus.signals.technical import calculate_vwap
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

    p.regime_detector.reset_cache()
    signal_generator = p._make_signal_generator(
        regime_state=regime_state,
        hourly_bias_map=hourly_bias_map,
        capital=initial_capital,
        primary_interval="15minute",
        symbol=symbol,
        param_overrides={"mr_min_score": 2.5},
        parrondo=False,
        capital_tracker=capital_tracker,
    )

    max_pos = 1 if initial_capital < 30000 else 2
    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_config=cost_cfg,
        capital_tracker=capital_tracker,
        max_positions=max_pos,
        intraday_session=True,  # INTRADAY OVERRIDE
    )

    result = engine.run(
        data=data,
        signal_generator=signal_generator,
        strategy_name=f"apex_hunter_{scenario_label}_{period_label}",
        warmup_bars=30,
    )

    if not result or result.total_trades == 0:
        return None

    # Save trades list to CSV
    trades_path = REPORT_DIR / f"apex_hunter_trades_{scenario_label}_{period_label}.csv"
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
    print(f"  Saved {len(df_trades)} trades to {trades_path.name}")

    # Buy & hold reference
    bh_start = data["close"].iloc[30]
    bh_end = data["close"].iloc[-1]
    bh_return = (bh_end / bh_start - 1) * 100
    years = max((data["timestamp"].iloc[-1] - data["timestamp"].iloc[30]).days / 365.25, 0.01)
    bh_cagr = ((bh_end / bh_start) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        "scenario": scenario_label,
        "period": period_label,
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
        "initial_capital": initial_capital,
        "final_capital": round(result.final_capital, 2),
        "evaluation_mode": "full_run_slice" if period_label in FULL_RUN_PERIODS else "isolated_holdout",
        "note": f"csv_15m|{'full_run_slice' if period_label in FULL_RUN_PERIODS else 'isolated'}",
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"  PROMETHEUS Apex Hunter Backtest -- {today}")
    print(f"  Symbol: {SYMBOL}  |  Capital: Rs {INITIAL_CAPITAL:,}")
    print("=" * 70)

    # Load data once
    if not CSV_15M_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_15M_PATH}")
        sys.exit(1)

    df_full = load_csv_15m(CSV_15M_PATH)

    # Build all test tasks
    tasks = []
    for scenario_label, slip in SLIPPAGE_SCENARIOS:
        # Full run periods
        for period in FULL_RUN_PERIODS:
            if period == "full":
                data_slice = df_full
            elif period == "10y":
                data_slice = slice_last_n_years(df_full, 10)
            elif period == "5y":
                data_slice = slice_last_n_years(df_full, 5)
            else:
                data_slice = df_full
            tasks.append((data_slice, scenario_label, period, slip))

        # Train/test holdout
        for period_label, start, end in HOLDOUT_PERIODS:
            data_slice = slice_data(df_full, start, end)
            tasks.append((data_slice, scenario_label, period_label, slip))

        # Crisis months (isolated: fresh capital each time)
        for ym in CRISIS_MONTHS:
            data_slice = slice_by_month(df_full, ym)
            tasks.append((data_slice, scenario_label, ym, slip))

    print(f"\n{len(tasks)} test scenarios queued. Running...")

    metrics_path = REPORT_DIR / f"apex_hunter_metrics_{today}.csv"
    completed_scenarios = set()
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed_scenarios.add(f"{row.get('scenario', '')}_{row.get('period', '')}")
            print(f"\n[RESUME] Found {len(completed_scenarios)} completed scenarios in {metrics_path.name}. Skipping them...")
        except Exception as e:
            print(f"[RESUME] Error reading existing CSV: {e}")
        
    # Filter tasks to only run uncompleted scenarios
    tasks = [t for t in tasks if f"{t[1]}_{t[2]}" not in completed_scenarios]
    if not tasks:
        print("\nAll scenarios are already completed!")
        # Self-destruct startup script if we finished
        startup_bat = Path(os.getenv('APPDATA', '')) / r"Microsoft\Windows\Start Menu\Programs\Startup\resume_apex_test.bat"
        if startup_bat.exists():
            try:
                startup_bat.unlink()
                print("Cleaned up startup script.")
            except:
                pass
        sys.exit(0)
    else:
        print(f"\n{len(tasks)} scenarios left to run...")
    # Run sequentially (Prometheus is not pickle-safe for multiprocessing)
    all_results = []
    
    while tasks:
        # Re-filter tasks inside the loop to pop the completed ones (and ensure safe resume)
        tasks = [t for t in tasks if f"{t[1]}_{t[2]}" not in completed_scenarios]
        if not tasks:
            break
            
        current_task = tasks[0]
        data_slice, scenario_label, period_label, slip = current_task
        
        label = f"[Remaining: {len(tasks)}] {scenario_label}/{period_label}"
        print(f"\n{'-' * 50}")
        print(f"  {label}  ({len(data_slice):,} bars, slip={slip}%)")
        print(f"{'-' * 50}")
        
        try:
            result = run_single_backtest(
                data=data_slice,
                symbol=SYMBOL,
                initial_capital=INITIAL_CAPITAL,
                slippage_pct=slip,
                scenario_label=scenario_label,
                period_label=period_label,
            )
            if result:
                all_results.append(result)
                print(f"  OK {result['trades']} trades | PF {result['profit_factor']:.2f} "
                      f"| Sharpe {result['sharpe']:.2f} | DD {result['max_dd_pct']:.1f}% "
                      f"| Final Rs {result['final_capital']:,.0f}")
                      
                # Incremental Save
                file_exists = metrics_path.exists()
                with open(metrics_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=result.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result)
            else:
                print(f"  !! No trades or insufficient data")
                
            # Successfully finished this scenario, add to completed
            completed_scenarios.add(f"{scenario_label}_{period_label}")
            
        except Exception as e:
            print(f"  X ERROR (Likely API/Internet cut): {e}")
            print(f"  ... Sleeping for 60 seconds and auto-retrying...")
            time.sleep(60)
            continue

    # -- Generate yearly breakdown --
    for scenario_label, _ in SLIPPAGE_SCENARIOS:
        for period in FULL_RUN_PERIODS:
            t_path = REPORT_DIR / f"apex_hunter_trades_{scenario_label}_{period}.csv"
            md_path = REPORT_DIR / f"apex_hunter_yearly_breakdown_{scenario_label}_{period}_{today}.md"
            if t_path.exists():
                try:
                    generate_yearly_breakdown(t_path, md_path)
                except Exception as e:
                    print(f"Failed to generate yearly breakdown: {e}")

    # -- Metrics already incrementally saved in loop --
    if all_results:
        print(f"\nOK Metrics saved: {metrics_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    print(f"  Results: {len(all_results)} scenarios")
    print(f"{'=' * 70}")
    
    # Self-destruct startup script after completion
    startup_bat = Path(os.getenv('APPDATA', '')) / r"Microsoft\Windows\Start Menu\Programs\Startup\resume_apex_test.bat"
    if startup_bat.exists():
        try:
            startup_bat.unlink()
        except:
            pass


def generate_yearly_breakdown(trades_path: Path, output_md_path: Path):
    if not trades_path.exists():
        return
    df = pd.read_csv(trades_path)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    
    # Reconstruct equity curve
    equity = INITIAL_CAPITAL
    equity_curve = [INITIAL_CAPITAL]
    for pnl in df["net_pnl"]:
        equity += pnl
        equity_curve.append(equity)
    
    # Attach equity to trades (shift by 1 since equity_curve[0] is start)
    df["equity"] = equity_curve[1:]
    
    yearly_stats = []
    for yr, group in df.groupby("year"):
        # Initial capital at start of year
        yr_idx = group.index
        start_idx = yr_idx[0]
        start_cap = equity_curve[start_idx] # equity before first trade of year
        end_cap = equity_curve[yr_idx[-1] + 1]
        
        yr_return = (end_cap - start_cap) / start_cap * 100 if start_cap > 0 else 0
        
        # Calculate drawdown within the year
        yr_equity = np.array(equity_curve[start_idx : yr_idx[-1] + 2])
        peak = np.maximum.accumulate(yr_equity)
        dd = (peak - yr_equity) / peak * 100
        max_dd = dd.max()
        
        wins = group[group["net_pnl"] > 0]
        losses = group[group["net_pnl"] <= 0]
        win_rate = len(wins) / len(group) * 100 if len(group) > 0 else 0
        
        avg_win = wins["net_pnl"].mean() if not wins.empty else 0
        avg_loss = losses["net_pnl"].mean() if not losses.empty else 0
        
        pf = wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if losses["net_pnl"].sum() != 0 else float('inf')
        
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
    
    # Write custom markdown table to avoid tabulate dependency
    headers = list(df_y.columns)
    table_lines = []
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df_y.iterrows():
        table_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        
    md_content = []
    md_content.append(f"# Yearly Breakdown for {trades_path.name}\n")
    md_content.extend(table_lines)
    
    with open(output_md_path, "w") as f:
        f.write("\n".join(md_content))
    print(f"\nYearly breakdown saved to {output_md_path.name}")
    print("\n".join(table_lines))


if __name__ == "__main__":
    main()
