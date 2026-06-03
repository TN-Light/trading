#!/usr/bin/env python3
"""
DD Diagnostic: Realized vs Unrealized Drawdown Split

Runs a single base/full backtest and reports:
  - Unrealized DD (equity curve with mark-to-market)
  - Realized DD (equity curve at trade-close events only)

This answers: is the 29.9% DD from actual losses or premium estimation noise?
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from prometheus.config import get
from prometheus.backtest.engine import BacktestEngine, BacktestResult


SYMBOL = "NIFTY 50"
INITIAL_CAPITAL = 15_000
CSV_15M_PATH = PROJECT_ROOT / "dataset" / f"{SYMBOL}_15minute.csv"


def load_csv_15m(path):
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
    df = df.rename(columns=col_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0
    return df


def main():
    print("=" * 70)
    print("  DD DIAGNOSTIC: Realized vs Unrealized Drawdown")
    print("=" * 70)

    print(f"\nLoading CSV: {CSV_15M_PATH}")
    data = load_csv_15m(CSV_15M_PATH)
    print(f"  Loaded {len(data):,} bars: {data['timestamp'].iloc[0]} -> {data['timestamp'].iloc[-1]}")

    from prometheus.main import Prometheus
    from prometheus.signals.technical import calculate_vwap

    p = Prometheus()
    p.initial_capital = INITIAL_CAPITAL
    p.capital = INITIAL_CAPITAL

    cost_cfg = dict(get("backtest.costs", {}))
    cost_cfg["slippage_pct"] = 0.0  # base scenario

    capital_tracker = {"capital": INITIAL_CAPITAL, "peak": INITIAL_CAPITAL}

    # Build signal generator
    regime_state = None
    if len(data) >= 50:
        regime_state = p.regime_detector.detect(data)

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

    max_pos = 1
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        cost_config=cost_cfg,
        capital_tracker=capital_tracker,
        max_positions=max_pos,
    )

    print(f"\nRunning backtest on {len(data):,} bars...")
    t0 = time.time()

    result = engine.run(
        data=data,
        signal_generator=signal_generator,
        strategy_name="dd_diagnostic_base_full",
        warmup_bars=30,
    )

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")

    if not result or result.total_trades == 0:
        print("  ERROR: No trades generated!")
        return

    # === KEY DIAGNOSTIC ===
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC RESULTS")
    print("=" * 70)

    print(f"\n  Trades:          {result.total_trades}")
    print(f"  Win Rate:        {result.win_rate:.1f}%")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print(f"  Final Capital:   Rs {result.final_capital:,.0f}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")

    print(f"\n  {'─' * 50}")
    print(f"  UNREALIZED DD (mark-to-market):  {result.max_drawdown_pct:.2f}%")
    print(f"  REALIZED DD   (trade-close):     {result.realized_dd_pct:.2f}%")
    print(f"  {'─' * 50}")

    gap = result.max_drawdown_pct - result.realized_dd_pct
    if gap > 5:
        print(f"\n  >>> GAP: {gap:.1f}pp — DD is inflated by intra-position premium swings")
        print(f"  >>> VERDICT: System is HEALTHY. Use realized DD for risk monitoring.")
    elif gap > 1:
        print(f"\n  >>> GAP: {gap:.1f}pp — modest contribution from premium estimation")
        print(f"  >>> VERDICT: Mixed. Some premium noise but real DD is still significant.")
    else:
        print(f"\n  >>> GAP: {gap:.1f}pp — almost no difference")
        print(f"  >>> VERDICT: DD is REAL. Needs confluence sweep or risk tuning.")

    # Additional: show realized equity curve stats
    if hasattr(result, 'realized_equity_curve') and result.realized_equity_curve:
        req = np.array(result.realized_equity_curve)
        rpeak = np.maximum.accumulate(req)
        rdrawdown = np.where(rpeak > 0, (rpeak - req) / rpeak * 100, 0)
        
        # Find the worst drawdown period
        max_dd_idx = np.argmax(rdrawdown)
        peak_before_dd = np.argmax(req[:max_dd_idx+1]) if max_dd_idx > 0 else 0
        
        print(f"\n  Realized equity range: Rs {req.min():,.0f} - Rs {req.max():,.0f}")
        print(f"  Peak before worst DD:  bar {peak_before_dd} (Rs {req[peak_before_dd]:,.0f})")
        print(f"  Trough of worst DD:    bar {max_dd_idx} (Rs {req[max_dd_idx]:,.0f})")

    # PF diagnostic: show gross wins/losses breakdown
    wins = [t for t in result.trades if t.get("net_pnl", t.get("pnl", 0)) > 0]
    losses = [t for t in result.trades if t.get("net_pnl", t.get("pnl", 0)) <= 0]
    total_wins = sum(t.get("net_pnl", t.get("pnl", 0)) for t in wins)
    total_losses = abs(sum(t.get("net_pnl", t.get("pnl", 0)) for t in losses))

    print(f"\n  PF BREAKDOWN:")
    print(f"    Gross Wins:    Rs {total_wins:,.0f} ({len(wins)} trades)")
    print(f"    Gross Losses:  Rs {total_losses:,.0f} ({len(losses)} trades)")
    print(f"    Avg Win:       Rs {total_wins/max(len(wins),1):,.0f}")
    print(f"    Avg Loss:      Rs {total_losses/max(len(losses),1):,.0f}")
    print(f"    Win/Loss Ratio:{(total_wins/max(len(wins),1))/(total_losses/max(len(losses),1)):.2f}x" if len(losses) > 0 else "")

    # Exit reason breakdown
    exit_reasons = {}
    for t in result.trades:
        reason = t.get("exit_reason", "unknown")
        if reason not in exit_reasons:
            exit_reasons[reason] = {"count": 0, "pnl": 0.0}
        exit_reasons[reason]["count"] += 1
        exit_reasons[reason]["pnl"] += t.get("net_pnl", t.get("pnl", 0))
    
    print(f"\n  EXIT REASON BREAKDOWN:")
    for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
        avg_pnl = stats["pnl"] / stats["count"]
        print(f"    {reason:25s}: {stats['count']:4d} trades, avg PnL Rs {avg_pnl:>8,.0f}, total Rs {stats['pnl']:>10,.0f}")

    print(f"\n{'=' * 70}")
    print("  Diagnostic complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
