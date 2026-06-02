#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20-Day Swing-15m Backtest -- Detailed Trade Analysis
Runs on NIFTY 50 and NIFTY BANK using yfinance 15-minute data.
Prints every trade's entry/exit, direction, PnL, signals, and exit reason.
"""

import os
import sys
import warnings
import io
import pandas as pd
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    _sys.stderr = io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from prometheus.config import get
from prometheus.backtest.engine import BacktestEngine


def run_swing_15m_backtest(symbol: str, days: int = 20):
    """Run swing backtest on 15m bars and return (result, engine, data)."""
    from prometheus.main import Prometheus

    p = Prometheus()
    initial_capital = p.initial_capital

    # Fetch 15m bars (primary), hourly (bias), daily (regime)
    data_15m = p.data.fetch_historical(symbol, days=days, interval="15minute", force_refresh=True)
    data_hourly = p.data.fetch_historical(symbol, days=days, interval="60minute", force_refresh=True)
    data_daily = p.data.fetch_historical(symbol, days=max(days, 120), interval="day", force_refresh=True)

    if data_15m.empty:
        print(f"  ERROR: No 15-minute data for {symbol}")
        return None, None, None

    print(f"  Data: {len(data_15m)} x 15min bars | {data_15m['timestamp'].iloc[0]} → {data_15m['timestamp'].iloc[-1]}")

    # Regime detection
    regime_state = p.regime_detector.detect(data_daily) if len(data_daily) >= 50 else None
    if regime_state:
        print(f"  Regime: {regime_state.regime.value} | Conf: {regime_state.confidence:.2f} | Trend: {regime_state.trend_strength:+.2f}")

    # Hourly bias
    from prometheus.signals.technical import calculate_vwap
    hourly_bias_map = {}
    if not data_hourly.empty and len(data_hourly) >= 10:
        hourly_vwap = calculate_vwap(data_hourly.copy())
        for i in range(10, len(data_hourly)):
            chunk = data_hourly.iloc[max(0, i - 20):i + 1]
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

    # Signal generator
    capital_tracker = {"capital": initial_capital, "peak": initial_capital}
    p.regime_detector.reset_cache()
    signal_gen = p._make_signal_generator(
        regime_state=regime_state,
        hourly_bias_map=hourly_bias_map,
        capital=initial_capital,
        primary_interval="15minute",
        symbol=symbol,
        param_overrides={"mr_min_score": 2.5},
        parrondo=False,
        capital_tracker=capital_tracker,
    )

    # Engine
    cost_cfg = dict(get("backtest.costs", {}))
    max_pos = 1 if initial_capital < 30000 else 2
    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_config=cost_cfg,
        capital_tracker=capital_tracker,
        max_positions=max_pos,
    )

    result = engine.run(
        data=data_15m,
        signal_generator=signal_gen,
        strategy_name=f"swing_15m_{symbol.replace(' ', '_')}",
        warmup_bars=30,
    )

    return result, engine, data_15m


def print_trade_details(engine, symbol):
    """Print detailed info for each trade."""
    trades = engine.trades
    if not trades:
        print(f"\n  No trades found for {symbol}.")
        return

    print(f"\n  {'─' * 90}")
    print(f"  INDIVIDUAL TRADE DETAILS — {symbol}")
    print(f"  {'─' * 90}")

    for i, t in enumerate(trades, 1):
        direction = getattr(t, 'direction', 'N/A')
        entry_time = getattr(t, 'entry_time', 'N/A')
        exit_time = getattr(t, 'exit_time', 'N/A')
        entry_price = getattr(t, 'entry_price', 0)
        exit_price = getattr(t, 'exit_price', 0)
        net_pnl = getattr(t, 'net_pnl', 0)
        gross_pnl = getattr(t, 'gross_pnl', 0)
        costs = getattr(t, 'costs', 0)
        exit_reason = getattr(t, 'exit_reason', 'N/A')
        hold_min = getattr(t, 'hold_duration_minutes', 0)
        strategy = getattr(t, 'strategy', 'N/A')
        quantity = getattr(t, 'quantity', 0)

        # Signal features
        features = []
        for feat in ['liqsweep', 'fvg', 'vp', 'ote', 'rsi_div', 'vol_surge', 'vol_confirm', 'vwap', 'bias']:
            if getattr(t, f'signal_{feat}', False):
                features.append(feat)

        bull_score = getattr(t, 'bull_score', 0)
        bear_score = getattr(t, 'bear_score', 0)
        atr_entry = getattr(t, 'atr_at_entry', 0)
        regime_entry = getattr(t, 'regime_at_entry', 'N/A')

        result_emoji = "✅" if net_pnl > 0 else "❌"

        print(f"\n  Trade #{i} {result_emoji}")
        print(f"    Direction:    {direction}")
        print(f"    Entry:        {entry_time} @ Rs {entry_price:,.2f}")
        print(f"    Exit:         {exit_time} @ Rs {exit_price:,.2f}")
        print(f"    Quantity:     {quantity}")
        print(f"    Gross PnL:    Rs {gross_pnl:+,.2f}")
        print(f"    Costs:        Rs {costs:,.2f}")
        print(f"    Net PnL:      Rs {net_pnl:+,.2f}")
        print(f"    Hold:         {hold_min:.0f} min ({hold_min/60:.1f} hrs)")
        print(f"    Exit Reason:  {exit_reason}")
        print(f"    Strategy:     {strategy}")
        print(f"    Regime:       {regime_entry}")
        print(f"    Scores:       Bull={bull_score:.2f} | Bear={bear_score:.2f}")
        print(f"    ATR at entry: {atr_entry:.2f}")
        print(f"    Signals:      {', '.join(features) if features else 'none'}")


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print("=" * 90)
    print(f"  PROMETHEUS Swing-15m Backtest — Last 20 Trading Days")
    print(f"  Date: {today}")
    print("=" * 90)

    symbols = ["NIFTY 50", "NIFTY BANK"]

    all_trades = {}

    for symbol in symbols:
        print(f"\n{'━' * 90}")
        print(f"  {symbol}")
        print(f"{'━' * 90}")

        result, engine, data = run_swing_15m_backtest(symbol, days=20)

        if result and result.total_trades > 0:
            print(f"\n  SUMMARY: {result.total_trades} trades | WR {result.win_rate:.1f}% | "
                  f"PF {result.profit_factor:.2f} | Sharpe {result.sharpe_ratio:.2f} | "
                  f"Max DD {result.max_drawdown_pct:.1f}% | "
                  f"Final Rs {result.final_capital:,.0f} ({result.total_return_pct:+.1f}%)")

            print_trade_details(engine, symbol)
            all_trades[symbol] = engine.trades
        else:
            print(f"\n  No trades generated in this 20-day window.")
            all_trades[symbol] = []

    # Summary
    print(f"\n\n{'=' * 90}")
    print(f"  COMBINED SUMMARY")
    print(f"{'=' * 90}")
    total_trades = sum(len(t) for t in all_trades.values())
    total_pnl = sum(sum(getattr(t, 'net_pnl', 0) for t in trades) for trades in all_trades.values())
    winners = sum(sum(1 for t in trades if getattr(t, 'net_pnl', 0) > 0) for trades in all_trades.values())
    wr = (winners / total_trades * 100) if total_trades > 0 else 0

    print(f"  Total trades:   {total_trades}")
    print(f"  Winners:        {winners} ({wr:.1f}%)")
    print(f"  Net PnL:        Rs {total_pnl:+,.2f}")
    print(f"  Per trade avg:  Rs {total_pnl/total_trades:+,.2f}" if total_trades > 0 else "")

    # Eligible trades for today
    print(f"\n{'=' * 90}")
    print(f"  TRADE ELIGIBILITY ASSESSMENT")
    print(f"{'=' * 90}")

    if total_trades == 0:
        print("  ⚠️  No trades in the last 20 days — system is very selective.")
    elif total_pnl > 0:
        print(f"  ✅ System found {total_trades} eligible trades with net positive PnL Rs {total_pnl:+,.2f}")
        print(f"     The swing logic IS generating signals on 15m charts.")
    else:
        print(f"  ⚠️  {total_trades} trades found but net PnL is negative Rs {total_pnl:+,.2f}")
        print(f"     Consider tightening filters or reviewing signal quality.")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
