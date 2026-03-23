# ============================================================================
# PROMETHEUS — Pattern Mining Engine
# ============================================================================
"""
Multi-layer loss pattern discovery engine.

5 Analysis Layers:
  1. FrequencyPatternMiner    — Single-attribute loss frequency scoring
  2. CombinationPatternMiner  — Apriori algorithm for multi-attribute combos
  3. TemporalClusterAnalyzer  — Loss heatmaps by time dimensions
  4. LossArchetypeClassifier  — Classify into 8 loss archetypes
  5. HiddenPatternDetector    — XGBoost + SHAP for invisible patterns

Input:  loss_database.csv (from LossDNATagger)
Output: loss_patterns_report.json + loss_heatmap (interactive HTML)
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from collections import defaultdict
from pathlib import Path


# ============================================================================
# Layer 1 — Frequency Pattern Miner
# ============================================================================

class FrequencyPatternMiner:
    """
    Find single attributes that appear in losses significantly
    more than in wins.

    Loss Frequency Score = (% of losses with attribute)
                         - (% of wins with attribute)

    Thresholds:
      > 20% → Loss Pattern
      > 40% → Critical Loss Pattern
      > 60% → Lethal Loss Pattern
    """

    THRESHOLD_LOSS = 20
    THRESHOLD_CRITICAL = 40
    THRESHOLD_LETHAL = 60

    def __init__(self, min_sample_size: int = 10):
        self.min_sample_size = min_sample_size
        self.patterns: List[Dict] = []

    def mine(self, df: pd.DataFrame, attribute_columns: List[str]) -> List[Dict]:
        """
        Mine single-attribute patterns.

        Args:
            df: Tagged trade database with 'win' column
            attribute_columns: Columns to analyze

        Returns:
            List of pattern dicts sorted by loss_frequency_score
        """
        self.patterns = []
        losers = df[~df['win']]
        winners = df[df['win']]
        n_losses = len(losers)
        n_wins = len(winners)

        if n_losses < self.min_sample_size:
            return []

        for col in attribute_columns:
            if col not in df.columns:
                continue

            if df[col].dtype == bool or df[col].nunique() <= 20:
                # Categorical / boolean column
                for val in df[col].unique():
                    loss_pct = (losers[col] == val).sum() / n_losses * 100
                    win_pct = (winners[col] == val).sum() / n_wins * 100 if n_wins > 0 else 0
                    score = loss_pct - win_pct

                    count_in_losses = (losers[col] == val).sum()
                    if count_in_losses < 3:
                        continue

                    severity = 'normal'
                    if score > self.THRESHOLD_LETHAL:
                        severity = 'lethal'
                    elif score > self.THRESHOLD_CRITICAL:
                        severity = 'critical'
                    elif score > self.THRESHOLD_LOSS:
                        severity = 'loss_pattern'

                    if score > 10:  # Only record meaningful patterns
                        self.patterns.append({
                            'attribute': col,
                            'value': str(val),
                            'loss_frequency_score': round(score, 1),
                            'loss_pct': round(loss_pct, 1),
                            'win_pct': round(win_pct, 1),
                            'count_in_losses': int(count_in_losses),
                            'severity': severity,
                            'type': 'single_attribute',
                        })

        self.patterns.sort(key=lambda x: x['loss_frequency_score'], reverse=True)
        return self.patterns

    def get_summary(self) -> Dict:
        """Get summary statistics of discovered patterns."""
        lethal = [p for p in self.patterns if p['severity'] == 'lethal']
        critical = [p for p in self.patterns if p['severity'] == 'critical']
        loss = [p for p in self.patterns if p['severity'] == 'loss_pattern']
        return {
            'total_patterns': len(self.patterns),
            'lethal_patterns': len(lethal),
            'critical_patterns': len(critical),
            'loss_patterns': len(loss),
            'top_5': self.patterns[:5],
        }


# ============================================================================
# Layer 2 — Combination Pattern Miner (Apriori-inspired)
# ============================================================================

class CombinationPatternMiner:
    """
    Discover hidden loss patterns in COMBINATIONS of attributes.

    Single attributes may look normal, but together they create
    lethal loss clusters. Uses an Apriori-inspired approach:

    Depth 2: All 2-attribute combinations
    Depth 3: Promising 2-combos extended to 3
    Depth 4: Top 3-combos extended to 4

    A combination is flagged if its loss rate exceeds a threshold
    AND the sample size is sufficient.
    """

    def __init__(
        self,
        min_support: int = 5,
        min_loss_rate: float = 0.70,
        max_depth: int = 4
    ):
        self.min_support = min_support
        self.min_loss_rate = min_loss_rate
        self.max_depth = max_depth
        self.patterns: List[Dict] = []

    def mine(
        self,
        df: pd.DataFrame,
        attribute_columns: List[str],
        max_unique_per_col: int = 8
    ) -> List[Dict]:
        """
        Mine combination patterns up to max_depth.

        Args:
            df: Tagged trade database
            attribute_columns: Columns to combine
            max_unique_per_col: Skip columns with too many unique values
        """
        self.patterns = []

        # Filter to manageable categorical columns
        usable_cols = []
        for col in attribute_columns:
            if col not in df.columns:
                continue
            n_unique = df[col].nunique()
            if n_unique <= max_unique_per_col and n_unique >= 2:
                usable_cols.append(col)

        # Create binary feature matrix for each (col, value) pair
        conditions = {}
        for col in usable_cols:
            for val in df[col].unique():
                if pd.isna(val):
                    continue
                key = f"{col}={val}"
                conditions[key] = (df[col] == val)

        condition_keys = list(conditions.keys())
        n_total = len(df)
        n_losses = (~df['win']).sum()
        base_loss_rate = n_losses / n_total if n_total > 0 else 0

        # Depth 2: All pairs
        depth2_promising = []
        for i, k1 in enumerate(condition_keys):
            for k2 in condition_keys[i+1:]:
                # Skip same-column combinations
                col1 = k1.split('=')[0]
                col2 = k2.split('=')[0]
                if col1 == col2:
                    continue

                mask = conditions[k1] & conditions[k2]
                n = mask.sum()
                if n < self.min_support:
                    continue

                loss_rate = (~df.loc[mask, 'win']).sum() / n
                if loss_rate >= self.min_loss_rate:
                    total_loss = df.loc[mask & ~df['win'], 'net_pnl'].sum()
                    pattern = {
                        'combination': [k1, k2],
                        'depth': 2,
                        'count': int(n),
                        'loss_rate': round(loss_rate * 100, 1),
                        'base_loss_rate': round(base_loss_rate * 100, 1),
                        'lift': round(loss_rate / base_loss_rate, 2) if base_loss_rate > 0 else 0,
                        'total_loss_pnl': round(total_loss, 0),
                        'type': 'combination',
                    }
                    self.patterns.append(pattern)
                    if loss_rate >= 0.60:  # Promising for extension
                        depth2_promising.append((k1, k2, mask))

        # Depth 3: Extend promising pairs
        depth3_promising = []
        if self.max_depth >= 3:
            for k1, k2, mask_pair in depth2_promising:
                col1 = k1.split('=')[0]
                col2 = k2.split('=')[0]
                for k3 in condition_keys:
                    col3 = k3.split('=')[0]
                    if col3 in (col1, col2):
                        continue
                    mask = mask_pair & conditions[k3]
                    n = mask.sum()
                    if n < self.min_support:
                        continue
                    loss_rate = (~df.loc[mask, 'win']).sum() / n
                    if loss_rate >= self.min_loss_rate:
                        total_loss = df.loc[mask & ~df['win'], 'net_pnl'].sum()
                        pattern = {
                            'combination': [k1, k2, k3],
                            'depth': 3,
                            'count': int(n),
                            'loss_rate': round(loss_rate * 100, 1),
                            'base_loss_rate': round(base_loss_rate * 100, 1),
                            'lift': round(loss_rate / base_loss_rate, 2) if base_loss_rate > 0 else 0,
                            'total_loss_pnl': round(total_loss, 0),
                            'type': 'combination',
                        }
                        self.patterns.append(pattern)
                        if loss_rate >= 0.75:
                            depth3_promising.append((k1, k2, k3, mask))

        # Depth 4: Extend top triples
        if self.max_depth >= 4:
            for k1, k2, k3, mask_triple in depth3_promising[:50]:  # Cap at 50
                col1, col2, col3 = k1.split('=')[0], k2.split('=')[0], k3.split('=')[0]
                for k4 in condition_keys:
                    col4 = k4.split('=')[0]
                    if col4 in (col1, col2, col3):
                        continue
                    mask = mask_triple & conditions[k4]
                    n = mask.sum()
                    if n < self.min_support:
                        continue
                    loss_rate = (~df.loc[mask, 'win']).sum() / n
                    if loss_rate >= self.min_loss_rate:
                        total_loss = df.loc[mask & ~df['win'], 'net_pnl'].sum()
                        self.patterns.append({
                            'combination': [k1, k2, k3, k4],
                            'depth': 4,
                            'count': int(n),
                            'loss_rate': round(loss_rate * 100, 1),
                            'base_loss_rate': round(base_loss_rate * 100, 1),
                            'lift': round(loss_rate / base_loss_rate, 2) if base_loss_rate > 0 else 0,
                            'total_loss_pnl': round(total_loss, 0),
                            'type': 'combination',
                        })

        self.patterns.sort(key=lambda x: (-x['loss_rate'], -x['count']))
        return self.patterns


# ============================================================================
# Layer 3 — Temporal Cluster Analyzer
# ============================================================================

class TemporalClusterAnalyzer:
    """
    Find if losses cluster in specific time conditions.

    Outputs a LOSS HEATMAP showing the riskiest time windows.
    """

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal clustering of losses.

        Returns dict with heatmaps and high-risk windows.
        """
        losers = df[~df['win']]
        n_total = len(df)
        n_losses = len(losers)

        results = {
            'total_trades': n_total,
            'total_losses': n_losses,
            'base_loss_rate': round(n_losses / n_total * 100, 1) if n_total > 0 else 0,
            'heatmaps': {},
            'high_risk_windows': [],
        }

        # 1. Loss rate by day of week
        if 'day_name' in df.columns:
            day_stats = self._loss_rate_by(df, 'day_name')
            results['heatmaps']['day_of_week'] = day_stats
            for day, stats in day_stats.items():
                if stats['loss_rate'] > results['base_loss_rate'] + 10:
                    results['high_risk_windows'].append({
                        'dimension': 'day_of_week',
                        'value': day,
                        'loss_rate': stats['loss_rate'],
                        'excess_loss_rate': round(stats['loss_rate'] - results['base_loss_rate'], 1),
                        'sample_size': stats['count'],
                    })

        # 2. Loss rate by month
        if 'month_name' in df.columns:
            results['heatmaps']['month'] = self._loss_rate_by(df, 'month_name')

        # 3. Loss rate by week of month
        if 'week_of_month' in df.columns:
            results['heatmaps']['week_of_month'] = self._loss_rate_by(df, 'week_of_month')

        # 4. Loss rate by regime
        if 'regime_at_entry' in df.columns:
            results['heatmaps']['regime'] = self._loss_rate_by(df, 'regime_at_entry')

        # 5. Loss rate by expiry proximity
        if 'days_to_weekly_expiry' in df.columns:
            results['heatmaps']['expiry_proximity'] = self._loss_rate_by(df, 'days_to_weekly_expiry')

        # 6. Loss rate by hold duration bucket
        if 'hold_bucket' in df.columns:
            results['heatmaps']['hold_duration'] = self._loss_rate_by(df, 'hold_bucket')

        # 7. Loss rate by score bucket
        if 'score_bucket' in df.columns:
            results['heatmaps']['score_bucket'] = self._loss_rate_by(df, 'score_bucket')

        # 8. Loss rate by direction + regime cross-tab
        if 'dir_regime_combo' in df.columns:
            combo_stats = self._loss_rate_by(df, 'dir_regime_combo')
            results['heatmaps']['direction_x_regime'] = combo_stats
            for combo, stats in combo_stats.items():
                if stats['loss_rate'] > results['base_loss_rate'] + 15 and stats['count'] >= 5:
                    results['high_risk_windows'].append({
                        'dimension': 'direction_x_regime',
                        'value': combo,
                        'loss_rate': stats['loss_rate'],
                        'excess_loss_rate': round(stats['loss_rate'] - results['base_loss_rate'], 1),
                        'sample_size': stats['count'],
                    })

        # 9. Year-over-year
        if 'year' in df.columns:
            results['heatmaps']['year'] = self._loss_rate_by(df, 'year')

        # Sort high risk windows by excess loss rate
        results['high_risk_windows'].sort(
            key=lambda x: x['excess_loss_rate'], reverse=True
        )

        return results

    def _loss_rate_by(self, df: pd.DataFrame, column: str) -> Dict:
        """Calculate loss rate grouped by a column."""
        stats = {}
        for val in sorted(df[column].dropna().unique(), key=str):
            sub = df[df[column] == val]
            n = len(sub)
            if n < 3:
                continue
            losses = (~sub['win']).sum()
            total_pnl = sub['net_pnl'].sum()
            avg_loss = sub[~sub['win']]['net_pnl'].mean() if losses > 0 else 0
            stats[str(val)] = {
                'count': int(n),
                'losses': int(losses),
                'wins': int(n - losses),
                'loss_rate': round(losses / n * 100, 1),
                'total_pnl': round(total_pnl, 0),
                'avg_loss': round(avg_loss, 0),
            }
        return stats


# ============================================================================
# Layer 4 — Loss Archetype Classifier
# ============================================================================

class LossArchetypeClassifier:
    """
    Classify every loss into one of 8 archetypes.

    Each archetype has a specific cause and a specific surgical fix.
    """

    ARCHETYPES = {
        'false_signal': {
            'name': 'The False Signal Loss',
            'description': 'Setup looked perfect, all signals aligned, but price immediately reversed',
            'cause': 'Structural signal failure',
            'fix': 'Improve signal quality filters',
        },
        'stop_hunt': {
            'name': 'The Stop Hunt Loss',
            'description': 'Price hit stop loss exactly then reversed in original direction',
            'cause': 'Stop placed at obvious retail level',
            'fix': 'Adaptive stop placement away from obvious levels',
        },
        'chop_grind': {
            'name': 'The Chop Grind Loss',
            'description': 'No clear direction, market oscillated until stop/time-stop hit',
            'cause': 'Entered during ranging/choppy regime',
            'fix': 'Regime detector blocks entries in choppy conditions',
        },
        'overextension': {
            'name': 'The Overextension Loss',
            'description': 'Entered after price already moved significantly',
            'cause': 'Late entry, chasing price',
            'fix': 'Entry timing filter — max distance from zone origin',
        },
        'oversizing': {
            'name': 'The Oversizing Loss',
            'description': 'Trade direction was right but loss was too large relative to account',
            'cause': 'Position sizing failure',
            'fix': 'Dynamic Kelly sizing enforced at execution layer',
        },
        'sequence': {
            'name': 'The Sequence Loss',
            'description': 'Loss came after sequence of wins/losses (overconfidence or revenge)',
            'cause': 'Psychological state affecting discipline',
            'fix': 'Automated circuit breaker after N consecutive losses',
        },
        'cost_kill': {
            'name': 'The Cost Kill Loss',
            'description': 'Trade was actually profitable but costs turned it into a loss',
            'cause': 'Transaction costs exceeding edge',
            'fix': 'Minimum expected move filter before entry',
        },
        'regime_mismatch': {
            'name': 'The Regime Mismatch Loss',
            'description': 'Strategy was incompatible with current market regime',
            'cause': 'Wrong strategy for the regime',
            'fix': 'Regime gate blocks incompatible strategy/regime combos',
        },
    }

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify every losing trade into its archetype.

        Returns DataFrame with 'loss_archetype' column.
        """
        df = df.copy()
        df['loss_archetype'] = 'unclassified'

        # Only classify losses
        loss_mask = ~df['win']

        # 1. Cost Kill — gross profit but net loss
        cost_kill = loss_mask & df.get('killed_by_costs', pd.Series(False, index=df.index))
        df.loc[cost_kill, 'loss_archetype'] = 'cost_kill'

        # 2. Sequence — after 3+ consecutive losses or 5+ consecutive wins
        sequence = loss_mask & (
            df.get('after_3_losses', pd.Series(False, index=df.index)) |
            df.get('after_5_wins', pd.Series(False, index=df.index))
        )
        df.loc[sequence & (df['loss_archetype'] == 'unclassified'), 'loss_archetype'] = 'sequence'

        # 3. Overextension — entered far from EMA / overextended
        overext = loss_mask & (
            df.get('overextended_entry', pd.Series(False, index=df.index)) |
            (df.get('dist_from_ema21_at_entry', pd.Series(0, index=df.index)).abs() > 3.0)
        )
        df.loc[overext & (df['loss_archetype'] == 'unclassified'), 'loss_archetype'] = 'overextension'

        # 4. Chop Grind — time_stop exit with small loss (market went nowhere)
        chop = loss_mask & (
            df.get('is_time_stop', pd.Series(False, index=df.index)) &
            (df['net_pnl'].abs() < 150)
        )
        df.loc[chop & (df['loss_archetype'] == 'unclassified'), 'loss_archetype'] = 'chop_grind'

        # 5. False Signal — immediate reversal (stopped out in <= 1 day) with signals present
        false_sig = loss_mask & (
            df.get('immediate_reversal', pd.Series(False, index=df.index)) &
            (df.get('signal_count', pd.Series(0, index=df.index)) >= 2) &
            df.get('is_stop_loss', pd.Series(False, index=df.index))
        )
        df.loc[false_sig & (df['loss_archetype'] == 'unclassified'), 'loss_archetype'] = 'false_signal'

        # 6. Regime Mismatch — trend strategy in choppy regime or vice versa
        regime_mm = loss_mask & (
            (df.get('is_trend_strategy', pd.Series(False, index=df.index)) &
             df['regime_at_entry'].isin(['accumulation', 'distribution'])) |
            (df.get('is_mr_strategy', pd.Series(False, index=df.index)) &
             df['regime_at_entry'].isin(['markup', 'markdown']))
        )
        df.loc[regime_mm & (df['loss_archetype'] == 'unclassified'), 'loss_archetype'] = 'regime_mismatch'

        # 7. Oversizing — catastrophic loss (> 300 absolute)
        oversize = loss_mask & (df['net_pnl'] < -300)
        df.loc[oversize & (df['loss_archetype'] == 'unclassified'), 'loss_archetype'] = 'oversizing'

        # 8. Stop Hunt — full stop hit; remaining unclassified stop losses
        stop_hunt = loss_mask & df.get('full_stop_hit', pd.Series(False, index=df.index))
        df.loc[stop_hunt & (df['loss_archetype'] == 'unclassified'), 'loss_archetype'] = 'stop_hunt'

        return df

    def get_archetype_breakdown(self, df: pd.DataFrame) -> Dict:
        """Get percentage breakdown of loss archetypes."""
        losses = df[~df['win']]
        if len(losses) == 0:
            return {}

        breakdown = {}
        for archetype in list(self.ARCHETYPES.keys()) + ['unclassified']:
            count = (losses['loss_archetype'] == archetype).sum()
            if count > 0:
                pnl = losses[losses['loss_archetype'] == archetype]['net_pnl'].sum()
                info = self.ARCHETYPES.get(archetype, {})
                breakdown[archetype] = {
                    'count': int(count),
                    'pct_of_losses': round(count / len(losses) * 100, 1),
                    'total_pnl': round(pnl, 0),
                    'avg_pnl': round(pnl / count, 0),
                    'name': info.get('name', archetype),
                    'fix': info.get('fix', 'Manual review'),
                }

        return dict(sorted(breakdown.items(), key=lambda x: x[1]['count'], reverse=True))


# ============================================================================
# Layer 5 — Hidden Pattern Detector (XGBoost + SHAP)
# ============================================================================

class HiddenPatternDetector:
    """
    Use machine learning to find patterns too complex for human eyes.

    Trains XGBoost classifier:
      Input:  All trade attributes at entry
      Output: Probability this trade becomes a loss (0-100%)

    After training, extracts SHAP values to identify:
      - Which attributes have highest predictive power for losses
      - Which combinations create non-linear loss risk
      - Threshold values where loss probability spikes
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.feature_names: List[str] = []
        self.shap_importance: Dict = {}

    def train(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Dict:
        """
        Train XGBoost classifier to predict losses.

        Args:
            df: Tagged trade database
            feature_columns: Columns to use as features (auto-detected if None)

        Returns:
            Training results including accuracy, feature importance
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return {'error': 'xgboost or sklearn not installed'}

        # Auto-select numeric + encodeable features
        if feature_columns is None:
            feature_columns = self._auto_select_features(df)

        self.feature_names = feature_columns.copy()

        # Prepare features
        X, encoders = self._prepare_features(df, feature_columns)
        y = (~df['win']).astype(int)  # 1 = loss, 0 = win

        if len(X) < 50:
            return {'error': 'Insufficient data for ML training'}

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')

        # Full fit for SHAP and feature importance
        self.model.fit(X, y)

        # Feature importance (built-in)
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # SHAP values (if available)
        shap_results = self._compute_shap(X)

        results = {
            'cv_accuracy': round(np.mean(cv_scores) * 100, 1),
            'cv_std': round(np.std(cv_scores) * 100, 1),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'loss_rate': round(y.mean() * 100, 1),
            'feature_importance_top20': dict(list(importance.items())[:20]),
            'shap_importance': shap_results,
        }

        return results

    def predict_loss_probability(self, trade_features: Dict) -> float:
        """Predict loss probability for a single trade."""
        if self.model is None:
            return 0.5

        # Build feature vector
        features = []
        for col in self.feature_names:
            val = trade_features.get(col, 0)
            if isinstance(val, bool):
                val = int(val)
            elif isinstance(val, str):
                val = hash(val) % 1000  # Simple encoding
            features.append(float(val) if not pd.isna(val) else 0.0)

        X = np.array([features])
        prob = self.model.predict_proba(X)[0][1]  # Prob of class 1 (loss)
        return round(prob * 100, 1)

    def _auto_select_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-select features suitable for ML."""
        exclude = {
            'entry_time', 'exit_time', 'entry_dt', 'exit_dt',
            'net_pnl', 'gross_pnl', 'win', 'loss_archetype',
            'cumulative_pnl', 'running_peak', 'drawdown_at_entry',
            'exit_price', 'price_move_pct', 'hold_to_loss_ratio',
            'actual_rr', 'loss_atr_multiple',
        }

        features = []
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype in [np.float64, np.int64, np.bool_]:
                if df[col].nunique() > 1 and df[col].isna().sum() < len(df) * 0.5:
                    features.append(col)
            elif df[col].dtype == 'object' and df[col].nunique() <= 15:
                features.append(col)

        return features

    def _prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Encode features for XGBoost."""
        from sklearn.preprocessing import LabelEncoder

        X = pd.DataFrame()
        encoders = {}

        for col in feature_columns:
            if col not in df.columns:
                continue

            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                le = LabelEncoder()
                vals = df[col].fillna('missing').astype(str)
                X[col] = le.fit_transform(vals)
                encoders[col] = le
            elif df[col].dtype == bool:
                X[col] = df[col].astype(int)
            else:
                X[col] = df[col].fillna(0).astype(float)

        # Update feature names to match actual columns
        self.feature_names = list(X.columns)

        return X.values, encoders

    def _compute_shap(self, X: np.ndarray) -> Dict:
        """Compute SHAP values for feature importance."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(
                self.feature_names,
                [round(float(v), 4) for v in mean_abs_shap]
            ))
            shap_importance = dict(sorted(
                shap_importance.items(), key=lambda x: x[1], reverse=True
            ))
            return dict(list(shap_importance.items())[:20])
        except Exception:
            return {'note': 'SHAP not available — install with: pip install shap'}


# ============================================================================
# Master Pattern Mining Orchestrator
# ============================================================================

class PatternMiningEngine:
    """
    Orchestrates all 5 analysis layers and produces a unified report.
    """

    # Columns to use for pattern mining (from LossDNATagger output)
    CATEGORICAL_ATTRS = [
        'direction', 'regime_at_entry', 'exit_reason', 'exit_category',
        'day_name', 'is_monday', 'is_friday',
        'month_name', 'quarter', 'week_of_month',
        'is_expiry_day', 'is_day_before_expiry', 'is_expiry_week',
        'confluence_bucket', 'score_bucket', 'hold_bucket',
        'is_trend_strategy', 'is_mr_strategy', 'is_expiry_strategy',
        'is_reactive_entry', 'loss_severity',
        'dir_regime_combo', 'strat_type',
        'vol_expanding_at_entry', 'atr_above_avg_at_entry',
        'trend_aligned_at_entry', 'overextended_entry',
        'immediate_reversal', 'slow_grind_loss', 'full_stop_hit',
        'killed_by_costs', 'after_3_losses', 'after_5_wins', 'in_drawdown',
    ]

    COMBO_ATTRS = [
        'direction', 'regime_at_entry', 'exit_category',
        'day_name', 'is_monday', 'is_friday',
        'is_expiry_day', 'is_expiry_week',
        'confluence_bucket', 'score_bucket',
        'is_trend_strategy', 'is_expiry_strategy',
        'strat_type', 'hold_bucket',
        'vol_expanding_at_entry', 'atr_above_avg_at_entry',
        'trend_aligned_at_entry', 'in_drawdown',
    ]

    def run_full_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run all 5 analysis layers and return unified results.

        Args:
            df: Output from LossDNATagger.tag_trades()

        Returns:
            Complete analysis report dictionary
        """
        results = {
            'summary': {
                'total_trades': len(df),
                'wins': int(df['win'].sum()),
                'losses': int((~df['win']).sum()),
                'win_rate': round(df['win'].mean() * 100, 1),
                'total_pnl': round(df['net_pnl'].sum(), 0),
            },
            'layers': {}
        }

        # Layer 1: Frequency Mining
        print("  → Layer 1: Frequency Pattern Mining...")
        freq_miner = FrequencyPatternMiner()
        freq_patterns = freq_miner.mine(df, self.CATEGORICAL_ATTRS)
        results['layers']['frequency'] = {
            'patterns': freq_patterns[:30],  # Top 30
            'summary': freq_miner.get_summary(),
        }

        # Layer 2: Combination Mining
        print("  → Layer 2: Combination Pattern Mining (Apriori)...")
        combo_miner = CombinationPatternMiner(min_support=5, min_loss_rate=0.70)
        combo_patterns = combo_miner.mine(df, self.COMBO_ATTRS)
        results['layers']['combination'] = {
            'patterns': combo_patterns[:50],  # Top 50
            'total_found': len(combo_patterns),
        }

        # Layer 3: Temporal Clustering
        print("  → Layer 3: Temporal Cluster Analysis...")
        temporal = TemporalClusterAnalyzer()
        results['layers']['temporal'] = temporal.analyze(df)

        # Layer 4: Archetype Classification
        print("  → Layer 4: Loss Archetype Classification...")
        archetype_clf = LossArchetypeClassifier()
        df = archetype_clf.classify(df)
        results['layers']['archetypes'] = archetype_clf.get_archetype_breakdown(df)

        # Layer 5: Hidden Patterns (ML)
        print("  → Layer 5: Hidden Pattern Detection (XGBoost)...")
        hidden = HiddenPatternDetector()
        results['layers']['hidden_ml'] = hidden.train(df)

        return results, df

    def save_report(self, results: Dict, output_path: str = 'loss_patterns_report.json'):
        """Save the full analysis report as JSON."""
        # Convert non-serializable values
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj

        report_str = json.dumps(results, indent=2, default=make_serializable)
        Path(output_path).write_text(report_str)
        print(f"\n  ✓ Pattern report saved: {output_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================
if __name__ == '__main__':
    from prometheus.analysis.loss_dna_tagger import LossDNATagger

    csv_files = [
        'trade_analysis_NIFTY_50_full.csv',
        'trade_analysis_NIFTY_BANK_full.csv',
        'trade_analysis_SENSEX_full.csv',
    ]

    for csv_file in csv_files:
        path = Path(csv_file)
        if not path.exists():
            print(f"  ✗ Not found: {csv_file}")
            continue

        symbol = csv_file.replace('trade_analysis_', '').replace('_full.csv', '')
        print(f"\n{'#'*70}")
        print(f"  PATTERN MINING: {symbol}")
        print(f"{'#'*70}")

        # Step 1: Tag trades
        trades = pd.read_csv(csv_file)
        tagger = LossDNATagger()
        tagged = tagger.tag_trades(trades)

        # Step 2: Run full analysis
        engine = PatternMiningEngine()
        results, tagged = engine.run_full_analysis(tagged)

        # Step 3: Save
        engine.save_report(results, f'loss_patterns_report_{symbol}.json')

        # Print key findings
        freq = results['layers']['frequency']['summary']
        print(f"\n  FREQUENCY: {freq['total_patterns']} patterns found")
        print(f"    Lethal: {freq['lethal_patterns']}, Critical: {freq['critical_patterns']}")
        if freq['top_5']:
            print(f"    Top pattern: {freq['top_5'][0]['attribute']}={freq['top_5'][0]['value']} "
                  f"(score: {freq['top_5'][0]['loss_frequency_score']})")

        combos = results['layers']['combination']
        print(f"\n  COMBINATIONS: {combos['total_found']} toxic combos found")

        archetypes = results['layers']['archetypes']
        if archetypes:
            top_arch = list(archetypes.items())[0]
            print(f"\n  TOP ARCHETYPE: {top_arch[1]['name']} — "
                  f"{top_arch[1]['pct_of_losses']}% of losses")

        ml = results['layers']['hidden_ml']
        if 'cv_accuracy' in ml:
            print(f"\n  ML MODEL: {ml['cv_accuracy']}% accuracy (±{ml['cv_std']}%)")
