#!/usr/bin/env python3
"""
Walk-Forward Validation for Tuned Parrondo Best Combo
Train: 2007-2020 | Test: 2021-2026
Symbols: NIFTY BANK + NIFTY 50
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from prometheus.main import Prometheus

# Best combo from 243-sweep
BEST_REGIME_OVERRIDES = {
    "trend_strength_strong": 0.40,
    "trend_strength_sideways": 0.30,
    "vol_expanding_mult": 1.15,
    "hurst_accumulation": 0.40,
}
BEST_MR_MIN_SCORE = 3.0

PARAM_OVERRIDES = {
    "regime_overrides": BEST_REGIME_OVERRIDES,
    "mr_min_score": BEST_MR_MIN_SCORE,
}

def run_walkforward(symbol, train_days=6750, test_days=1825):
    """Run walk-forward: train on full history, test on last 5 years."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION: {symbol}")
    print(f"  Best Combo: TS=0.40, SS=0.30, VE=1.15, HA=0.40, MR=3.0")
    print(f"{'='*70}")

    p = Prometheus()

    # Fetch all data (max ~18.5yr)
    print(f"\n  Fetching {train_days} days of data for {symbol}...")
    data_all = p.data.fetch_historical(symbol, days=train_days, interval="day", force_refresh=True)

    if data_all.empty or len(data_all) < 200:
        print(f"  ERROR: Only {len(data_all)} bars available")
        return

    print(f"  Total bars: {len(data_all)} ({str(data_all['timestamp'].iloc[0])[:10]} to {str(data_all['timestamp'].iloc[-1])[:10]})")

    # Split: train = everything before 2021, test = 2021 onwards
    train_mask = data_all["timestamp"].astype(str) < "2021-01-01"
    test_mask = data_all["timestamp"].astype(str) >= "2021-01-01"

    data_train = data_all[train_mask].copy().reset_index(drop=True)
    data_test = data_all[test_mask].copy().reset_index(drop=True)

    print(f"  Train: {len(data_train)} bars ({str(data_train['timestamp'].iloc[0])[:10]} to {str(data_train['timestamp'].iloc[-1])[:10]})")
    print(f"  Test:  {len(data_test)} bars ({str(data_test['timestamp'].iloc[0])[:10]} to {str(data_test['timestamp'].iloc[-1])[:10]})")

    # ---- IN-SAMPLE (Train 2007-2020) ----
    print(f"\n  --- IN-SAMPLE (Train) ---")
    result_train, _ = p._run_backtest_on_slice(
        data_train, symbol, f"TRAIN_tuned_{symbol}",
        param_overrides=PARAM_OVERRIDES,
        verbose=True,
        parrondo=True,
    )

    # ---- OUT-OF-SAMPLE (Test 2021-2026) ----
    print(f"\n  --- OUT-OF-SAMPLE (Test) ---")
    result_test, _ = p._run_backtest_on_slice(
        data_test, symbol, f"TEST_tuned_{symbol}",
        param_overrides=PARAM_OVERRIDES,
        verbose=True,
        parrondo=True,
    )

    # ---- BASELINE (Test without tuning) ----
    print(f"\n  --- BASELINE (Test, no tuning) ---")
    result_baseline, _ = p._run_backtest_on_slice(
        data_test, symbol, f"TEST_baseline_{symbol}",
        param_overrides=None,
        verbose=True,
        parrondo=True,
    )

    # ---- COMPARISON ----
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD RESULTS: {symbol}")
    print(f"{'='*70}")

    def wr(r):
        return r.win_rate * 100 if r.win_rate <= 1 else r.win_rate

    print(f"\n  {'Metric':<20} {'IS (Train)':<15} {'OOS (Tuned)':<15} {'OOS (Baseline)':<15} {'Tuned vs Base':<15}")
    print(f"  {'-'*75}")
    print(f"  {'PF':<20} {result_train.profit_factor:<15.2f} {result_test.profit_factor:<15.2f} {result_baseline.profit_factor:<15.2f} {result_test.profit_factor - result_baseline.profit_factor:+.2f}")
    print(f"  {'Sharpe':<20} {result_train.sharpe_ratio:<15.2f} {result_test.sharpe_ratio:<15.2f} {result_baseline.sharpe_ratio:<15.2f} {result_test.sharpe_ratio - result_baseline.sharpe_ratio:+.2f}")
    print(f"  {'Win Rate':<20} {wr(result_train):<14.0f}% {wr(result_test):<14.0f}% {wr(result_baseline):<14.0f}% {wr(result_test) - wr(result_baseline):+.0f}pp")
    print(f"  {'Max DD':<20} {result_train.max_drawdown_pct:<14.1f}% {result_test.max_drawdown_pct:<14.1f}% {result_baseline.max_drawdown_pct:<14.1f}% {result_test.max_drawdown_pct - result_baseline.max_drawdown_pct:+.1f}pp")
    print(f"  {'Return':<20} {result_train.total_return_pct:<14.0f}% {result_test.total_return_pct:<14.0f}% {result_baseline.total_return_pct:<14.0f}% {result_test.total_return_pct - result_baseline.total_return_pct:+.0f}%")
    print(f"  {'Trades':<20} {result_train.total_trades:<15} {result_test.total_trades:<15} {result_baseline.total_trades:<15}")

    # Validation checks
    print(f"\n  --- VALIDATION CHECKS ---")
    checks = 0
    total = 5

    # Check 1: OOS PF > 1.0
    if result_test.profit_factor > 1.0:
        checks += 1
        print(f"  [PASS] OOS PF > 1.0: {result_test.profit_factor:.2f}")
    else:
        print(f"  [FAIL] OOS PF > 1.0: {result_test.profit_factor:.2f}")

    # Check 2: OOS PF >= 60% of IS PF
    ratio = result_test.profit_factor / max(result_train.profit_factor, 0.01)
    if ratio >= 0.6:
        checks += 1
        print(f"  [PASS] OOS/IS PF ratio >= 60%: {ratio:.0%}")
    else:
        print(f"  [FAIL] OOS/IS PF ratio >= 60%: {ratio:.0%}")

    # Check 3: OOS Sharpe > 0.5
    if result_test.sharpe_ratio > 0.5:
        checks += 1
        print(f"  [PASS] OOS Sharpe > 0.5: {result_test.sharpe_ratio:.2f}")
    else:
        print(f"  [FAIL] OOS Sharpe > 0.5: {result_test.sharpe_ratio:.2f}")

    # Check 4: Tuned beats baseline OOS
    if result_test.profit_factor >= result_baseline.profit_factor:
        checks += 1
        print(f"  [PASS] Tuned PF >= Baseline PF: {result_test.profit_factor:.2f} >= {result_baseline.profit_factor:.2f}")
    else:
        print(f"  [FAIL] Tuned PF >= Baseline PF: {result_test.profit_factor:.2f} < {result_baseline.profit_factor:.2f}")

    # Check 5: OOS DD < 60%
    if result_test.max_drawdown_pct < 60:
        checks += 1
        print(f"  [PASS] OOS DD < 60%: {result_test.max_drawdown_pct:.1f}%")
    else:
        print(f"  [FAIL] OOS DD < 60%: {result_test.max_drawdown_pct:.1f}%")

    print(f"\n  VERDICT: {checks}/{total} checks passed")
    if checks >= 4:
        print(f"  >>> VALIDATED — Tuned combo works on unseen data!")
    elif checks >= 3:
        print(f"  >>> PARTIAL — Some degradation but still profitable")
    else:
        print(f"  >>> FAILED — Combo may be overfit to training data")

    print(f"{'='*70}\n")
    return result_train, result_test, result_baseline


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  PARRONDO TUNING — WALK-FORWARD VALIDATION")
    print("  Best Combo from 243-Sweep")
    print("="*70)

    # Run on NIFTY BANK first (primary)
    bank_results = run_walkforward("NIFTY BANK")

    # Run on NIFTY 50 (secondary validation)
    nifty_results = run_walkforward("NIFTY 50")

    print("\n" + "="*70)
    print("  ALL VALIDATIONS COMPLETE")
    print("="*70)
