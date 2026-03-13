# ============================================================================
# PROMETHEUS — Signal Regression Trainer
# ============================================================================
"""
Train regression models to learn optimal signal weights from historical trades.

Instead of hardcoded weights (LiqSweep=1.5, FVG=1.5, etc.),
fit data-driven weights from actual trade outcomes.

Scikit-learn LinearRegression with Ridge regularization to avoid overfitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import pickle
from pathlib import Path

from prometheus.utils.logger import logger


class SignalRegressionTrainer:
    """
    Train regression models on historical trade data to learn signal weights.

    Workflow:
    1. Extract features from BacktestTrade objects
    2. Create design matrix X (signal activations) and target y (P&L or win/loss)
    3. Fit regression: LinearRegression or Ridge or LogisticRegression
    4. Output learned weights to replace hardcoded confluence points
    5. Walk-forward validation to ensure no meta-overfitting
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.weights = None
        self.feature_names = [
            "signal_liqsweep",
            "signal_fvg",
            "signal_vp",
            "signal_ote",
            "signal_rsi_div",
            "signal_vol_surge",
            "signal_vol_confirm",
            "signal_vwap",
            "signal_bias",
        ]
        self.context_features = ["atr_at_entry", "regime_accumulation", "regime_distribution"]

    def extract_features(self, trades: List) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract features from BacktestTrade objects.

        Args:
            trades: List of BacktestTrade dataclass objects

        Returns:
            X: Design matrix (features), shape (N, n_features)
            y_pnl: P&L target (continuous), shape (N,)
            y_win: Win/loss target (binary), shape (N,)
        """
        data = []

        for trade in trades:
            # Skip trades with incomplete signal data
            if not hasattr(trade, 'signal_liqsweep'):
                continue

            # One-hot encode regime
            regime_accum = 1.0 if trade.regime_at_entry == "accumulation" else 0.0
            regime_dist = 1.0 if trade.regime_at_entry == "distribution" else 0.0

            row = {
                # Signal features (binary or 0-1)
                "signal_liqsweep": float(trade.signal_liqsweep),
                "signal_fvg": float(trade.signal_fvg),
                "signal_vp": float(trade.signal_vp),
                "signal_ote": float(trade.signal_ote),
                "signal_rsi_div": float(trade.signal_rsi_div),
                "signal_vol_surge": float(trade.signal_vol_surge),
                "signal_vol_confirm": float(trade.signal_vol_confirm),
                "signal_vwap": float(trade.signal_vwap),
                "signal_bias": float(trade.signal_bias),

                # Context features
                "atr_at_entry": float(trade.atr_at_entry) if trade.atr_at_entry > 0 else 1.0,
                "regime_accumulation": regime_accum,
                "regime_distribution": regime_dist,

                # Targets
                "net_pnl": float(trade.net_pnl),
                "is_win": 1.0 if trade.net_pnl > 0 else 0.0,
            }
            data.append(row)

        if not data:
            logger.warning("No trades with signal features found for regression training")
            return None, None, None

        df = pd.DataFrame(data)

        # Design matrix and targets
        feature_cols = self.feature_names + self.context_features
        X = df[feature_cols].fillna(0)
        y_pnl = df["net_pnl"]
        y_win = df["is_win"]

        logger.info(f"Extracted {len(X)} trades with signal features for regression training")
        return X, y_pnl, y_win

    def train_regression(
        self,
        X: pd.DataFrame,
        y_pnl: pd.Series,
        method: str = "ridge",
        alpha: float = 1.0,
        test_size: float = 0.2,
    ) -> Dict:
        """
        Train regression model on extracted features.

        Args:
            X: Design matrix (features)
            y_pnl: P&L target (continuous)
            method: "linear", "ridge", or "logistic"
            alpha: Regularization strength (for Ridge)
            test_size: Train/test split ratio

        Returns:
            results: Dict with model, weights, r2 scores, etc.
        """
        if X is None or len(X) < 20:
            logger.warning("Insufficient data for regression training")
            return None

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_pnl, test_size=test_size, random_state=42
        )

        logger.info(f"Train: {len(X_train)} trades | Test: {len(X_test)} trades")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit model
        if method == "linear":
            self.model = LinearRegression()
        elif method == "ridge":
            self.model = Ridge(alpha=alpha)
        elif method == "logistic":
            # For binary classification (win/loss)
            y_binary = (y_train > 0).astype(int)
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_train_scaled, y_binary)
            y_pred = self.model.predict(X_test_scaled)
            accuracy = (y_pred == (y_test > 0).astype(int)).mean()
            logger.info(f"Logistic regression accuracy: {accuracy:.2%}")
            return {"model": self.model, "method": "logistic", "accuracy": accuracy}
        else:
            raise ValueError(f"Unknown method: {method}")

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        r2_train = self.model.score(X_train_scaled, y_train)
        r2_test = self.model.score(X_test_scaled, y_test)

        logger.info(f"R² train: {r2_train:.4f} | R² test: {r2_test:.4f}")

        # Extract coefficients as weights
        coef = self.model.coef_
        self.weights = dict(zip(self.feature_names, coef[: len(self.feature_names)]))

        # Normalize weights to be comparable to confluence points (0.5-1.5 range typically)
        min_coef = np.min(np.abs(coef[coef != 0])) if np.any(coef != 0) else 1.0
        max_coef = np.max(np.abs(coef)) if np.any(coef != 0) else 1.0
        scale_factor = 1.0 / max(max_coef, 0.01)

        normalized_weights = {}
        for name, coef_val in self.weights.items():
            normalized_weights[name] = abs(coef_val) * scale_factor

        logger.info(f"Learned weights (normalized): {normalized_weights}")

        return {
            "model": self.model,
            "weights": self.weights,
            "normalized_weights": normalized_weights,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "method": method,
        }

    def compare_weights(self, learned: Dict, baseline_fixed: Dict = None) -> Dict:
        """
        Compare learned weights vs hardcoded baseline weights.

        Args:
            learned: Dict of learned signal weights
            baseline_fixed: Dict of hardcoded weights (defaults to fixed values)

        Returns:
            comparison: Dict with differences and insights
        """
        if baseline_fixed is None:
            # Hardcoded Prometheus weights (from confluence scoring)
            baseline_fixed = {
                "signal_liqsweep": 1.5,
                "signal_fvg": 1.5,
                "signal_vp": 1.0,
                "signal_ote": 1.0,
                "signal_rsi_div": 1.5,
                "signal_vol_surge": 0.5,
                "signal_vol_confirm": 0.5,
                "signal_vwap": 0.5,
                "signal_bias": 0.5,
            }

        comparison = {}
        for signal_name in self.feature_names:
            learned_val = learned.get(signal_name, 0.0)
            fixed_val = baseline_fixed.get(signal_name, 1.0)
            diff = learned_val - fixed_val
            diff_pct = (diff / fixed_val * 100) if fixed_val != 0 else 0

            comparison[signal_name] = {
                "learned": learned_val,
                "baseline": fixed_val,
                "diff": diff,
                "diff_pct": diff_pct,
            }

        logger.info(f"Weight comparison:\n{json.dumps(comparison, indent=2)}")
        return comparison

    def save_weights(self, weights: Dict, output_file: str = None) -> str:
        """Save learned weights to JSON config file."""
        if output_file is None:
            output_file = "prometheus/config/regression_weights.json"

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(weights, f, indent=2)

        logger.info(f"Weights saved to: {output_file}")
        return output_file

    def load_weights(self, input_file: str = "prometheus/config/regression_weights.json") -> Dict:
        """Load learned weights from JSON config file."""
        try:
            with open(input_file, 'r') as f:
                weights = json.load(f)
            logger.info(f"Loaded weights from: {input_file}")
            return weights
        except FileNotFoundError:
            logger.warning(f"No weights file found at {input_file}, using defaults")
            return None


# ============================================================================
# Integration Helper: Convert BacktestTrade features to confluence scores
# ============================================================================

def confluence_score_from_learned_weights(
    signal_dict: Dict,
    learned_weights: Dict,
) -> Tuple[float, float]:
    """
    Compute bull and bear confluence scores using learned weights
    instead of hardcoded points.

    Args:
        signal_dict: Dict with "sweep_direction", "fvg_direction", etc.
        learned_weights: Dict with learned weight for each signal

    Returns:
        (bull_score, bear_score)
    """
    bull_score = 0.0
    bear_score = 0.0

    # Map signal_dict keys to weight keys
    mappings = {
        "sweep_direction": "signal_liqsweep",
        "fvg_direction": "signal_fvg",
        "vp_direction": "signal_vp",
        "ote_direction": "signal_ote",
        "div_direction": "signal_rsi_div",
    }

    for signal_key, weight_key in mappings.items():
        if signal_key in signal_dict:
            direction = signal_dict[signal_key]
            weight = learned_weights.get(weight_key, 1.0)

            if direction == "bullish":
                bull_score += weight
            elif direction == "bearish":
                bear_score += weight

    # Add context signals
    if signal_dict.get("volume_surge"):
        bull_score += learned_weights.get("signal_vol_surge", 0.5)

    if signal_dict.get("vol_confirm_dir") == "bullish":
        bull_score += learned_weights.get("signal_vol_confirm", 0.5)
    elif signal_dict.get("vol_confirm_dir") == "bearish":
        bear_score += learned_weights.get("signal_vol_confirm", 0.5)

    if signal_dict.get("hourly_bias") == "bullish":
        bull_score += learned_weights.get("signal_bias", 0.5)
    elif signal_dict.get("hourly_bias") == "bearish":
        bear_score += learned_weights.get("signal_bias", 0.5)

    return bull_score, bear_score
