"""
Shared test fixtures for the pipeline test suite.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def make_15min_bars(n_bars: int = 200, start_date: str = "2026-01-05 09:15:00",
                     base_price: float = 24000.0, trend: float = 0.0) -> pd.DataFrame:
    """Generate realistic 15-minute OHLCV bars for testing.
    
    Args:
        n_bars: Number of bars to generate
        start_date: Start timestamp
        base_price: Starting price
        trend: Price drift per bar (positive = uptrend)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    timestamps = []
    dt = pd.to_datetime(start_date)
    for i in range(n_bars):
        # Skip non-market hours: 9:15 - 15:30
        if dt.time() >= pd.Timestamp("15:30").time():
            dt = dt.replace(hour=9, minute=15) + timedelta(days=1)
            # Skip weekends
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
        timestamps.append(dt)
        dt += timedelta(minutes=15)
    
    np.random.seed(42)
    prices = [base_price]
    for i in range(1, n_bars):
        change = np.random.normal(trend, base_price * 0.003)
        prices.append(prices[-1] + change)
    
    data = {
        "timestamp": timestamps,
        "open": prices,
        "high": [p + abs(np.random.normal(0, p * 0.002)) for p in prices],
        "low": [p - abs(np.random.normal(0, p * 0.002)) for p in prices],
        "close": [p + np.random.normal(0, p * 0.001) for p in prices],
        "volume": [int(np.random.exponential(100000)) for _ in prices],
    }
    
    df = pd.DataFrame(data)
    # Ensure high >= open/close and low <= open/close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def make_daily_bars(n_bars: int = 365, start_date: str = "2025-01-05",
                     base_price: float = 24000.0) -> pd.DataFrame:
    """Generate daily OHLCV bars."""
    timestamps = pd.bdate_range(start=start_date, periods=n_bars)
    np.random.seed(123)
    prices = [base_price]
    for i in range(1, n_bars):
        change = np.random.normal(0, base_price * 0.01)
        prices.append(prices[-1] + change)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + abs(np.random.normal(0, p * 0.005)) for p in prices],
        "low": [p - abs(np.random.normal(0, p * 0.005)) for p in prices],
        "close": [p + np.random.normal(0, p * 0.003) for p in prices],
        "volume": [int(np.random.exponential(500000)) for _ in prices],
    })


@pytest.fixture
def sample_15min_data():
    """200 bars of 15-minute data."""
    return make_15min_bars(200)


@pytest.fixture
def sample_daily_data():
    """365 bars of daily data."""
    return make_daily_bars(365)


@pytest.fixture
def stale_15min_data():
    """15-minute data that is 2 hours old."""
    return make_15min_bars(200, start_date="2026-01-03 09:15:00")


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def missing_columns_data():
    """Data with missing required columns."""
    df = make_15min_bars(100)
    return df.drop(columns=["close"])


@pytest.fixture
def mock_data_engine(sample_15min_data, sample_daily_data):
    """Mock data engine that returns test data."""
    engine = MagicMock()
    
    def fetch_historical(symbol, days=60, interval="day"):
        if interval == "15minute":
            return sample_15min_data.copy()
        elif interval == "60minute":
            return sample_15min_data.copy()  # Simplified
        elif interval == "day":
            return sample_daily_data.copy()
        return pd.DataFrame()
    
    engine.fetch_historical = MagicMock(side_effect=fetch_historical)
    engine.get_vix = MagicMock(return_value=15.0)
    return engine


@pytest.fixture
def mock_telegram():
    """Mock telegram bot."""
    tg = MagicMock()
    tg.send_message = MagicMock(return_value=True)
    tg._enabled = True
    return tg


@pytest.fixture
def sample_raw_signal():
    """A raw signal dict as produced by the signal generator."""
    return {
        "direction": "bullish",
        "strategy": "trend",
        "entry_price": 150.0,
        "stop_loss": 120.0,
        "target": 210.0,
        "bull_score": 4.5,
        "bear_score": 1.0,
        "strike": 24000,
        "option_expiry_date": "2026-01-08",
        "option_type": "CE",
        "lot_size": 75,
        "quantity": 75,
        "regime": "markup",
        "reasons": ["LiqSweep", "FVG", "VP", "VWAP"],
        "symbol": "NIFTY 50",
        "bar_timestamp": "2026-01-05 10:30:00",
    }


@pytest.fixture
def sample_bearish_signal():
    """A raw bearish signal."""
    return {
        "direction": "bearish",
        "strategy": "trend",
        "entry_price": 180.0,
        "stop_loss": 210.0,
        "target": 120.0,
        "bear_score": 3.8,
        "bull_score": 0.5,
        "strike": 24000,
        "option_expiry_date": "2026-01-08",
        "option_type": "PE",
        "lot_size": 75,
        "quantity": 75,
        "regime": "markdown",
        "reasons": ["LiqSweep", "OTE", "Dist"],
        "symbol": "NIFTY 50",
        "bar_timestamp": "2026-01-05 11:00:00",
    }


@pytest.fixture
def sample_expiry_signal():
    """A signal with expiry strategy (should be rejected)."""
    return {
        "direction": "bullish",
        "strategy": "expiry_trend",
        "entry_price": 50.0,
        "stop_loss": 30.0,
        "target": 80.0,
        "bull_score": 3.0,
        "strike": 24000,
        "lot_size": 75,
        "quantity": 75,
        "symbol": "NIFTY 50",
    }


@pytest.fixture
def sample_no_direction_signal():
    """A signal with no direction (should be rejected)."""
    return {
        "strategy": "trend",
        "entry_price": 150.0,
        "bull_score": 2.0,
        "symbol": "NIFTY 50",
    }
