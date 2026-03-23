# ============================================================================
# PROMETHEUS — Loss DNA Tagger
# ============================================================================
"""
Tags every historical trade with 40+ attributes to build the Loss DNA Database.

Every attribute recorded here is used downstream by the PatternMiner to discover
which combinations of conditions predict losses. The richer the attribute set,
the more hidden patterns the system can find.

Input:  trade_analysis_*.csv files (from analyze_losses.py)
Output: loss_database.csv — enriched with all derived attributes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class LossDNATagger:
    """
    Enriches every trade with 40+ contextual attributes for pattern mining.

    Attributes are grouped into:
      1. MARKET CONDITION — regime, volatility, day-of-week, time-of-day, expiry distance
      2. TRADE SETUP — strategy, signals, confluence, zone strength, RR ratio
      3. PRICE ACTION — entry candle, volume, extension from VWAP, gap presence
      4. POST-TRADE — speed of loss, stop hunt detection, loss archetype hints
    """

    # Indian market expiry schedule (approximate)
    WEEKLY_EXPIRY_DAY = 3  # Thursday (0=Mon)
    MONTHLY_EXPIRY_WEEK = -1  # Last week of month

    # Time buckets for Indian market (IST)
    TIME_BUCKETS = {
        'open_15min': ('09:15', '09:30'),
        'morning_hour': ('09:30', '10:30'),
        'mid_morning': ('10:30', '12:00'),
        'afternoon': ('12:00', '14:00'),
        'closing_hour': ('14:00', '15:00'),
        'last_30min': ('15:00', '15:30'),
    }

    def __init__(self, daily_data: Optional[pd.DataFrame] = None):
        """
        Args:
            daily_data: Full OHLCV daily data for the symbol (used for
                       ATR, VWAP, volatility context calculations).
        """
        self.daily_data = daily_data
        self._precompute_daily_metrics()

    def _precompute_daily_metrics(self):
        """Pre-compute rolling metrics from daily data for fast lookups."""
        if self.daily_data is None or len(self.daily_data) == 0:
            self._daily_metrics = None
            return

        df = self.daily_data.copy()
        df.index = pd.to_datetime(df.index) if not pd.api.types.is_datetime64_any_dtype(df.index) else df.index

        # Rolling volatility (20-day realized vol annualized)
        returns = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252) * 100
        df['realized_vol_5'] = returns.rolling(5).std() * np.sqrt(252) * 100
        df['vol_expanding'] = df['realized_vol_5'] > df['realized_vol_20'] * 1.2

        # ATR 14
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_20_avg'] = df['atr_14'].rolling(20).mean()
        df['atr_above_avg'] = df['atr_14'] > df['atr_20_avg']

        # EMA alignment
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()

        # Price distance from key MAs (%)
        df['dist_from_ema21_pct'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
        df['dist_from_ema50_pct'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100

        # Volume relative to 20-period average
        if 'volume' in df.columns:
            df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        else:
            df['vol_ratio'] = 1.0

        # Daily returns and magnitude
        df['daily_return_pct'] = returns * 100
        df['abs_return_pct'] = df['daily_return_pct'].abs()

        # Gap detection
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100

        # Range position (where in day's range did it close)
        df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

        # Trend alignment (all timeframes)
        df['trend_ema_aligned_bull'] = (
            (df['close'] > df['ema_9']) &
            (df['ema_9'] > df['ema_21']) &
            (df['ema_21'] > df['ema_50'])
        )
        df['trend_ema_aligned_bear'] = (
            (df['close'] < df['ema_9']) &
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50'])
        )

        # RSI 14
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss_val = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss_val.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_overbought'] = df['rsi_14'] > 70
        df['rsi_oversold'] = df['rsi_14'] < 30

        # Consecutive up/down days
        df['up_day'] = df['daily_return_pct'] > 0
        streak = df['up_day'].ne(df['up_day'].shift()).cumsum()
        df['streak_length'] = df.groupby(streak).cumcount() + 1
        df['streak_direction'] = df['up_day'].map({True: 'up', False: 'down'})

        self._daily_metrics = df

    def tag_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tag every trade with all 40+ loss DNA attributes.

        Args:
            trades_df: DataFrame from analyze_losses.py with columns:
                entry_time, exit_time, direction, net_pnl, gross_pnl, costs,
                exit_reason, regime_at_entry, bull_score, bear_score,
                atr_at_entry, strategy, entry_price, exit_price, hold_minutes,
                signal_* columns, entry_type, win, year, strat_type

        Returns:
            Enriched DataFrame with all DNA attributes appended.
        """
        df = trades_df.copy()

        # Parse datetime
        df['entry_dt'] = pd.to_datetime(df['entry_time'])
        df['exit_dt'] = pd.to_datetime(df['exit_time'])

        # === MARKET CONDITION ATTRIBUTES ===
        df = self._tag_market_conditions(df)

        # === TRADE SETUP ATTRIBUTES ===
        df = self._tag_trade_setup(df)

        # === PRICE ACTION ATTRIBUTES ===
        df = self._tag_price_action(df)

        # === POST-TRADE ATTRIBUTES ===
        df = self._tag_post_trade(df)

        # === SEQUENCE ATTRIBUTES ===
        df = self._tag_sequence(df)

        return df

    def _tag_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag market condition attributes at time of entry."""

        # Day of week (0=Mon, 4=Fri)
        df['day_of_week'] = df['entry_dt'].dt.dayofweek
        df['day_name'] = df['entry_dt'].dt.day_name()
        df['is_monday'] = df['day_of_week'] == 0
        df['is_friday'] = df['day_of_week'] == 4

        # Month
        df['month'] = df['entry_dt'].dt.month
        df['month_name'] = df['entry_dt'].dt.month_name()

        # Week of month (1-5)
        df['week_of_month'] = (df['entry_dt'].dt.day - 1) // 7 + 1

        # Quarter
        df['quarter'] = df['entry_dt'].dt.quarter

        # Days to weekly expiry (Thursday)
        df['days_to_weekly_expiry'] = df['entry_dt'].apply(self._days_to_expiry)

        # Is it expiry day?
        df['is_expiry_day'] = df['days_to_weekly_expiry'] == 0
        df['is_day_before_expiry'] = df['days_to_weekly_expiry'] == 1

        # Monthly expiry proximity (last Thursday of month)
        df['days_to_monthly_expiry'] = df['entry_dt'].apply(self._days_to_monthly_expiry)
        df['is_expiry_week'] = df['days_to_monthly_expiry'] <= 4

        # Volatility context from daily data
        if self._daily_metrics is not None:
            df = self._merge_daily_context(df)
        else:
            # Defaults when no daily data available
            df['vol_expanding_at_entry'] = False
            df['atr_above_avg_at_entry'] = False
            df['gap_at_entry_pct'] = 0.0
            df['vol_ratio_at_entry'] = 1.0
            df['rsi_at_entry'] = 50.0
            df['dist_from_ema21_at_entry'] = 0.0
            df['dist_from_ema50_at_entry'] = 0.0
            df['trend_aligned_at_entry'] = False
            df['streak_at_entry'] = 0
            df['streak_dir_at_entry'] = 'none'

        return df

    def _tag_trade_setup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag trade setup attributes."""

        # Signal count (number of confluent signals)
        signal_cols = [c for c in df.columns if c.startswith('signal_')]
        df['signal_count'] = df[signal_cols].sum(axis=1)

        # Signal confidence buckets
        df['confluence_bucket'] = pd.cut(
            df['signal_count'],
            bins=[-1, 0, 1, 2, 3, 100],
            labels=['zero', 'one', 'two', 'three', 'four_plus']
        )

        # Strategy type classification
        df['is_trend_strategy'] = df['strategy'].str.contains('pro_', na=False)
        df['is_mr_strategy'] = df['strategy'].str.contains('mr_', na=False)
        df['is_expiry_strategy'] = df['strategy'].str.contains('expiry_', na=False)

        # Score at entry (using appropriate score for direction)
        df['relevant_score'] = np.where(
            df['direction'] == 'bullish',
            df['bull_score'],
            df['bear_score']
        )

        # Score buckets
        df['score_bucket'] = pd.cut(
            df['relevant_score'],
            bins=[-0.1, 1.0, 2.0, 3.0, 4.0, 100],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )

        # RR ratio (estimated from entry/exit/costs)
        df['actual_rr'] = np.where(
            df['net_pnl'] > 0,
            df['net_pnl'] / df['costs'].clip(lower=1),
            df['net_pnl'] / df['costs'].clip(lower=1)
        )

        # Whether entry was proactive (planned) vs reactive (immediate)
        df['is_reactive_entry'] = df['entry_type'] == 'immediate'

        # Hold duration buckets
        df['hold_hours'] = df['hold_minutes'] / 60
        df['hold_days'] = df['hold_minutes'] / 1440
        df['hold_bucket'] = pd.cut(
            df['hold_hours'],
            bins=[-1, 6, 24, 72, 168, 10000],
            labels=['intraday', 'overnight', '1_3_days', '3_7_days', '7_plus_days']
        )

        # Direction + regime combo (key cross-tab)
        df['dir_regime_combo'] = df['direction'] + '_' + df['regime_at_entry']

        return df

    def _tag_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag price action attributes at entry."""

        # Price move magnitude during trade
        df['price_move_pct'] = (
            (df['exit_price'] - df['entry_price']) / df['entry_price'].clip(lower=0.01) * 100
        )

        # Was stop loss the exit?
        df['is_stop_loss'] = df['exit_reason'] == 'stop_loss'
        df['is_time_stop'] = df['exit_reason'] == 'time_stop'
        df['is_target_hit'] = df['exit_reason'] == 'target'

        # Cost impact analysis
        df['cost_pct_of_entry'] = df['costs'] / df['entry_price'].clip(lower=0.01) * 100
        df['killed_by_costs'] = (df['gross_pnl'] > 0) & (df['net_pnl'] < 0)

        # Loss severity (none, small, medium, large, catastrophic)
        df['loss_severity'] = 'none'
        df.loc[df['net_pnl'] < 0, 'loss_severity'] = 'small'
        df.loc[df['net_pnl'] < -100, 'loss_severity'] = 'medium'
        df.loc[df['net_pnl'] < -200, 'loss_severity'] = 'large'
        df.loc[df['net_pnl'] < -400, 'loss_severity'] = 'catastrophic'

        # Normalized loss (as multiple of ATR)
        df['loss_atr_multiple'] = np.where(
            df['atr_at_entry'] > 0,
            df['net_pnl'].abs() / df['atr_at_entry'],
            0
        )

        return df

    def _tag_post_trade(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag post-trade analysis attributes."""

        # Speed of loss (fast = bad signal, slow = chop grind)
        df['hold_to_loss_ratio'] = np.where(
            df['net_pnl'] < 0,
            df['hold_minutes'],  # Lower = faster loss
            np.nan
        )

        # Immediate reversal indicator (lost money in < 1 bar / 1 day)
        df['immediate_reversal'] = (df['net_pnl'] < 0) & (df['hold_minutes'] <= 1440)

        # Slow grind loss (held for days but still lost)
        df['slow_grind_loss'] = (df['net_pnl'] < 0) & (df['hold_minutes'] > 4320)

        # Stop hunt indicator (full stop hit = likely stop hunt if loss is close to SL)
        df['full_stop_hit'] = df['exit_reason'] == 'stop_loss'

        # Exit reason categories
        df['exit_category'] = df['exit_reason'].map({
            'stop_loss': 'stopped_out',
            'target': 'target_hit',
            'time_stop': 'timed_out',
        }).fillna('other')

        return df

    def _tag_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag sequence-dependent attributes (loss streaks, psychology markers)."""

        # Sort by entry time
        df = df.sort_values('entry_dt').reset_index(drop=True)

        # Consecutive win/loss tracking
        df['prev_trade_win'] = df['win'].shift(1).fillna(True)
        df['prev_2_trades_lost'] = (~df['win'].shift(1).fillna(True)) & (~df['win'].shift(2).fillna(True))

        # Loss streak counter
        streak_counter = []
        current_streak = 0
        for _, row in df.iterrows():
            if not row['win']:
                current_streak += 1
            else:
                current_streak = 0
            streak_counter.append(current_streak)
        df['loss_streak_position'] = streak_counter

        # Win streak counter (for overconfidence detection)
        win_streak = []
        current_win = 0
        for _, row in df.iterrows():
            if row['win']:
                current_win += 1
            else:
                current_win = 0
            win_streak.append(current_win)
        df['win_streak_position'] = win_streak

        # After-streak flags
        df['after_3_losses'] = df['loss_streak_position'] >= 3
        df['after_5_wins'] = df['win_streak_position'] >= 5

        # Running PnL (equity curve position)
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        df['running_peak'] = df['cumulative_pnl'].cummax()
        df['drawdown_at_entry'] = df['running_peak'] - df['cumulative_pnl']
        df['drawdown_pct_at_entry'] = np.where(
            df['running_peak'] > 0,
            df['drawdown_at_entry'] / df['running_peak'] * 100,
            0
        )

        # In-drawdown flag (already losing money when entering this trade)
        df['in_drawdown'] = df['drawdown_pct_at_entry'] > 5

        return df

    # === HELPER METHODS ===

    def _days_to_expiry(self, dt: datetime) -> int:
        """Calculate days to next Thursday (weekly expiry)."""
        day = dt.weekday()
        if day <= 3:
            return 3 - day
        else:
            return 7 - day + 3

    def _days_to_monthly_expiry(self, dt: datetime) -> int:
        """Calculate days to last Thursday of the month."""
        import calendar
        year, month = dt.year, dt.month
        last_day = calendar.monthrange(year, month)[1]

        # Find last Thursday
        last_date = datetime(year, month, last_day)
        while last_date.weekday() != 3:
            last_date -= timedelta(days=1)

        delta = (last_date - dt).days
        if delta < 0:
            # Already past this month's expiry, look at next month
            month += 1
            if month > 12:
                month = 1
                year += 1
            last_day = calendar.monthrange(year, month)[1]
            last_date = datetime(year, month, last_day)
            while last_date.weekday() != 3:
                last_date -= timedelta(days=1)
            delta = (last_date - dt).days

        return delta

    def _merge_daily_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge daily market context metrics into trade data."""
        metrics = self._daily_metrics.copy()
        metrics.index = metrics.index.normalize()

        # For each trade, find the closest daily bar on or before entry
        entry_dates = df['entry_dt'].dt.normalize()

        merge_cols = [
            'vol_expanding', 'atr_above_avg', 'gap_pct', 'vol_ratio',
            'rsi_14', 'dist_from_ema21_pct', 'dist_from_ema50_pct',
            'trend_ema_aligned_bull', 'trend_ema_aligned_bear',
            'streak_length', 'streak_direction', 'realized_vol_20',
            'range_position'
        ]

        # Use merge_asof for time-based matching
        df_sorted = df.sort_values('entry_dt').copy()
        df_sorted['_merge_date'] = df_sorted['entry_dt'].dt.normalize()

        metrics_for_merge = metrics[merge_cols].copy()
        metrics_for_merge['_merge_date'] = metrics_for_merge.index

        merged = pd.merge_asof(
            df_sorted.sort_values('_merge_date'),
            metrics_for_merge.sort_values('_merge_date'),
            on='_merge_date',
            direction='backward'
        )

        # Rename merged columns
        rename_map = {
            'vol_expanding': 'vol_expanding_at_entry',
            'atr_above_avg': 'atr_above_avg_at_entry',
            'gap_pct': 'gap_at_entry_pct',
            'vol_ratio': 'vol_ratio_at_entry',
            'rsi_14': 'rsi_at_entry',
            'dist_from_ema21_pct': 'dist_from_ema21_at_entry',
            'dist_from_ema50_pct': 'dist_from_ema50_at_entry',
            'trend_ema_aligned_bull': 'trend_bull_aligned_at_entry',
            'trend_ema_aligned_bear': 'trend_bear_aligned_at_entry',
            'streak_length': 'streak_at_entry',
            'streak_direction': 'streak_dir_at_entry',
            'realized_vol_20': 'realized_vol_at_entry',
            'range_position': 'range_position_at_entry',
        }
        merged = merged.rename(columns=rename_map)

        # Trend alignment (direction-aware)
        merged['trend_aligned_at_entry'] = np.where(
            merged['direction'] == 'bullish',
            merged.get('trend_bull_aligned_at_entry', False),
            merged.get('trend_bear_aligned_at_entry', False)
        )

        # Extended from EMA (overextended entry detector)
        merged['overextended_entry'] = merged['dist_from_ema21_at_entry'].abs() > 3.0

        # Drop temp column
        merged = merged.drop(columns=['_merge_date'], errors='ignore')

        return merged.sort_values('entry_dt').reset_index(drop=True)

    def save_loss_database(
        self,
        tagged_df: pd.DataFrame,
        output_path: str = 'loss_database.csv'
    ):
        """Save the fully tagged trade database."""
        tagged_df.to_csv(output_path, index=False)
        n_losses = (~tagged_df['win']).sum()
        n_wins = tagged_df['win'].sum()
        n_attrs = len([c for c in tagged_df.columns
                      if c not in ['entry_time', 'exit_time']])
        print(f"\n  ✓ Loss DNA Database saved: {output_path}")
        print(f"    {len(tagged_df)} trades ({n_wins} wins, {n_losses} losses)")
        print(f"    {n_attrs} attributes per trade")

    @classmethod
    def from_csv(
        cls,
        trades_csv: str,
        daily_csv: Optional[str] = None
    ) -> 'LossDNATagger':
        """
        Factory: create tagger and tag trades from CSV files.

        Args:
            trades_csv: Path to trade_analysis_*.csv
            daily_csv: Optional path to daily OHLCV data for context

        Returns:
            LossDNATagger instance (call .tag_trades() with loaded data)
        """
        daily_data = None
        if daily_csv and Path(daily_csv).exists():
            daily_data = pd.read_csv(daily_csv, index_col='date', parse_dates=True)

        return cls(daily_data=daily_data)


# ============================================================================
# CLI Entry Point
# ============================================================================
if __name__ == '__main__':
    import sys
    from pathlib import Path

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

        print(f"\n{'='*70}")
        print(f"  TAGGING: {csv_file}")
        print(f"{'='*70}")

        trades = pd.read_csv(csv_file)
        tagger = LossDNATagger()
        tagged = tagger.tag_trades(trades)

        symbol = csv_file.replace('trade_analysis_', '').replace('_full.csv', '')
        output = f'loss_database_{symbol}.csv'
        tagger.save_loss_database(tagged, output)

        # Preview
        losses = tagged[~tagged['win']]
        print(f"\n  Top loss attributes (losses only):")
        for col in ['day_name', 'regime_at_entry', 'confluence_bucket',
                     'score_bucket', 'exit_category', 'loss_severity',
                     'hold_bucket', 'dir_regime_combo']:
            if col in losses.columns:
                top = losses[col].value_counts().head(3)
                print(f"    {col}: {dict(top)}")
