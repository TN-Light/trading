#!/usr/bin/env python3
"""
Phase 2: Train Regression Signal Weights from Historical Trades.

1. Run 15yr backtest on NIFTY BANK (with Parrondo best combo)
2. Extract signal features from each trade
3. Train Ridge regression on 2007-2020 trades
4. Test on 2021-2026 trades
5. Compare learned weights vs hardcoded baseline
6. Save weights to config file
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from prometheus.main import Prometheus
from prometheus.intelligence.signal_regression import SignalRegressionTrainer
import pandas as pd
import numpy as np

# Best Parrondo params from sweep
REGIME_OVERRIDES = {
    "trend_strength_strong": 0.40,
    "trend_strength_sideways": 0.30,
    "vol_expanding_mult": 1.15,
    "hurst_accumulation": 0.40,
}

PARAM_OVERRIDES = {
    "regime_overrides": REGIME_OVERRIDES,
    "mr_min_score": 3.0,
}


def extract_trades_with_features(symbol="NIFTY BANK", days=5475):
    """Run backtest and extract trades with signal features."""
    print(f"\n{'='*70}")
    print(f"  EXTRACTING TRADE FEATURES: {symbol} ({days} days)")
    print(f"{'='*70}")

    p = Prometheus()

    # Fetch all data
    data_all = p.data.fetch_historical(symbol, days=days, interval="day", force_refresh=True)
    print(f"  Data: {len(data_all)} bars")

    if data_all.empty or len(data_all) < 200:
        print("  ERROR: Insufficient data")
        return [], []

    # Split into train (2007-2020) and test (2021+)
    train_mask = data_all["timestamp"].astype(str) < "2021-01-01"
    test_mask = data_all["timestamp"].astype(str) >= "2021-01-01"

    data_train = data_all[train_mask].copy().reset_index(drop=True)
    data_test = data_all[test_mask].copy().reset_index(drop=True)

    print(f"  Train: {len(data_train)} bars | Test: {len(data_test)} bars")

    # Run backtest on TRAIN period
    print(f"\n  Running TRAIN backtest...")
    result_train, engine_train = p._run_backtest_on_slice(
        data_train, symbol, f"TRAIN_{symbol}",
        param_overrides=PARAM_OVERRIDES, verbose=False, parrondo=True,
    )
    train_trades = engine_train.trades if hasattr(engine_train, 'trades') else []
    print(f"  TRAIN trades: {len(train_trades)}")

    # Run backtest on TEST period
    print(f"  Running TEST backtest...")
    result_test, engine_test = p._run_backtest_on_slice(
        data_test, symbol, f"TEST_{symbol}",
        param_overrides=PARAM_OVERRIDES, verbose=False, parrondo=True,
    )
    test_trades = engine_test.trades if hasattr(engine_test, 'trades') else []
    print(f"  TEST trades: {len(test_trades)}")

    return train_trades, test_trades


def train_and_compare():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("  PHASE 2: REGRESSION SIGNAL WEIGHTS TRAINING")
    print("="*70)

    # Extract trades with features
    train_trades, test_trades = extract_trades_with_features("NIFTY BANK", 5475)

    if not train_trades:
        print("  ERROR: No train trades extracted")
        return

    # Initialize trainer
    trainer = SignalRegressionTrainer()

    # Extract features from train trades
    print(f"\n  Extracting features from {len(train_trades)} train trades...")
    X_train, y_pnl_train, y_win_train = trainer.extract_features(train_trades)

    if X_train is None or len(X_train) < 20:
        print(f"  ERROR: Only {0 if X_train is None else len(X_train)} trades have signal features")
        print("  Signal features may not be flowing through correctly.")
        print("  Checking first trade attributes...")
        if train_trades:
            t = train_trades[0]
            print(f"    signal_liqsweep: {getattr(t, 'signal_liqsweep', 'MISSING')}")
            print(f"    atr_at_entry: {getattr(t, 'atr_at_entry', 'MISSING')}")
            print(f"    regime_at_entry: {getattr(t, 'regime_at_entry', 'MISSING')}")
        return

    print(f"  Feature matrix: {X_train.shape[0]} trades × {X_train.shape[1]} features")

    # Show feature activation rates
    print(f"\n  Signal Activation Rates (Train):")
    for col in trainer.feature_names:
        rate = X_train[col].mean() * 100
        print(f"    {col:<25} {rate:>5.1f}% active")

    # Train Ridge regression (P&L target)
    print(f"\n  Training Ridge regression (alpha=1.0)...")
    result = trainer.train_regression(X_train, y_pnl_train, method="ridge", alpha=1.0, test_size=0.2)

    if result is None:
        print("  ERROR: Regression failed")
        return

    print(f"  R² train: {result['r2_train']:.4f} | R² test: {result['r2_test']:.4f}")

    # Show learned weights
    print(f"\n  Learned Signal Weights:")
    print(f"  {'Signal':<25} {'Learned':<10} {'Baseline':<10} {'Diff':<10}")
    print(f"  {'-'*55}")

    baseline = {
        "signal_liqsweep": 1.5, "signal_fvg": 1.5, "signal_vp": 1.0,
        "signal_ote": 1.0, "signal_rsi_div": 1.5, "signal_vol_surge": 0.5,
        "signal_vol_confirm": 0.5, "signal_vwap": 0.5, "signal_bias": 0.5,
    }

    normalized = result.get("normalized_weights", {})
    for signal_name in trainer.feature_names:
        learned = normalized.get(signal_name, 0.0)
        base = baseline.get(signal_name, 1.0)
        # Scale learned weights to match baseline range
        learned_scaled = learned * max(baseline.values())
        print(f"  {signal_name:<25} {learned_scaled:>8.2f}   {base:>8.2f}   {learned_scaled - base:>+8.2f}")

    # Extract features from test trades
    print(f"\n  Extracting features from {len(test_trades)} test trades...")
    X_test, y_pnl_test, y_win_test = trainer.extract_features(test_trades)

    if X_test is not None and len(X_test) > 0:
        # Evaluate on test set
        X_test_scaled = trainer.scaler.transform(X_test)
        r2_oos = trainer.model.score(X_test_scaled, y_pnl_test)
        print(f"\n  OOS R² (2021-2026): {r2_oos:.4f}")

        # Win rate comparison
        train_wr = (y_pnl_train > 0).mean() * 100
        test_wr = (y_pnl_test > 0).mean() * 100
        print(f"  Train WR: {train_wr:.1f}% | Test WR: {test_wr:.1f}%")

    # Save weights
    weights_to_save = {
        "method": "ridge",
        "r2_train": result["r2_train"],
        "r2_test": result["r2_test"],
        "raw_weights": result["weights"],
        "normalized_weights": normalized,
        "baseline_weights": baseline,
    }
    trainer.save_weights(weights_to_save, "prometheus/config/regression_weights.json")

    print(f"\n{'='*70}")
    print(f"  REGRESSION TRAINING COMPLETE")
    print(f"  Weights saved to: prometheus/config/regression_weights.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    train_and_compare()
