"""
Tests for DataBridge — data fetching and validation.

Covers:
- Successful fetch with valid data
- Empty data returns EMPTY status
- Stale data returns STALE status
- Missing columns returns MISSING_COLUMNS status
- Fetch exception returns FETCH_ERROR status
- Fallback: empty hourly/daily uses primary data
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from prometheus.pipeline.data_bridge import DataBridge
from prometheus.pipeline.types import DataStatus


class TestDataBridgeFetch:
    """Test data fetching and validation."""
    
    def test_successful_fetch(self):
        """Valid data returns OK status with correct metadata."""
        from datetime import datetime, timedelta
        # Use recent timestamps so data is not stale
        recent_data = pd.DataFrame({
            "timestamp": [datetime.now() - timedelta(minutes=i*15) for i in range(200, 0, -1)],
            "open": range(200), "high": range(200),
            "low": range(200), "close": range(200), "volume": range(200),
        })
        engine = MagicMock()
        engine.fetch_historical = MagicMock(return_value=recent_data)
        
        bridge = DataBridge(engine, max_staleness_seconds=999999)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.OK
        assert result.symbol == "NIFTY 50"
        assert not result.primary.empty
        assert result.bar_count > 0
        assert result.error_message == ""
    
    def test_empty_primary_data(self):
        """Empty primary data returns EMPTY status."""
        engine = MagicMock()
        engine.fetch_historical = MagicMock(return_value=pd.DataFrame())
        
        bridge = DataBridge(engine)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.EMPTY
        assert "empty" in result.error_message.lower() or "no" in result.error_message.lower()
    
    def test_none_primary_data(self):
        """None returned for primary data treated as empty."""
        engine = MagicMock()
        engine.fetch_historical = MagicMock(return_value=None)
        
        bridge = DataBridge(engine)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.EMPTY
    
    def test_missing_columns(self, mock_data_engine):
        """Data missing required columns returns MISSING_COLUMNS."""
        # Override to return data without 'close'
        bad_data = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-05", periods=100, freq="15min"),
            "open": range(100),
            "high": range(100),
            "low": range(100),
            # no 'close' column
        })
        mock_data_engine.fetch_historical = MagicMock(return_value=bad_data)
        
        bridge = DataBridge(mock_data_engine)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.MISSING_COLUMNS
        assert "close" in result.error_message.lower()
    
    def test_stale_data(self):
        """Data older than max_staleness returns STALE status."""
        from datetime import datetime, timedelta
        now = datetime.now()
        # Generate 100 15-minute bars ending 40 minutes ago (same day)
        timestamps = [now - timedelta(minutes=40 + i*15) for i in range(100)]
        timestamps.reverse()  # chronologically ascending
        old_data = pd.DataFrame({
            "timestamp": timestamps,
            "open": range(100),
            "high": range(100),
            "low": range(100),
            "close": range(100),
            "volume": range(100),
        })
        engine = MagicMock()
        engine.fetch_historical = MagicMock(return_value=old_data)
        
        bridge = DataBridge(engine, max_staleness_seconds=1800)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.STALE
        assert result.staleness_seconds > 1800
    
    def test_fetch_exception(self):
        """Exception during fetch returns FETCH_ERROR status."""
        engine = MagicMock()
        engine.fetch_historical = MagicMock(side_effect=ConnectionError("API timeout"))
        
        bridge = DataBridge(engine)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.FETCH_ERROR
        assert "API timeout" in result.error_message
    
    def test_empty_hourly_falls_back_to_primary(self, mock_data_engine):
        """When hourly data is empty, it falls back to primary."""
        def custom_fetch(symbol, days=60, interval="day", force_refresh=False):
            if interval == "15minute":
                return pd.DataFrame({
                    "timestamp": pd.date_range(datetime.now() - timedelta(hours=5),
                                               periods=100, freq="15min"),
                    "open": range(100), "high": range(100),
                    "low": range(100), "close": range(100), "volume": range(100),
                })
            elif interval == "60minute":
                return pd.DataFrame()  # Empty!
            return pd.DataFrame({
                "timestamp": pd.bdate_range("2025-01-01", periods=50),
                "open": range(50), "high": range(50),
                "low": range(50), "close": range(50), "volume": range(50),
            })
        
        mock_data_engine.fetch_historical = MagicMock(side_effect=custom_fetch)
        bridge = DataBridge(mock_data_engine, max_staleness_seconds=999999)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.OK
        # hourly should have been filled with primary
        assert not result.hourly.empty


class TestDataBridgeStaleness:
    """Test staleness calculation."""
    
    def test_fresh_data_is_not_stale(self):
        """Recent data within threshold is OK."""
        recent_data = pd.DataFrame({
            "timestamp": pd.date_range(
                datetime.now() - timedelta(minutes=20),
                periods=100, freq="15min"
            )[:10],  # Only use 10 to stay within ~20 min
            "open": range(10), "high": range(10),
            "low": range(10), "close": range(10), "volume": range(10),
        })
        # Use last 10 timestamps starting 20 min ago
        recent_data = pd.DataFrame({
            "timestamp": [datetime.now() - timedelta(minutes=i*15) for i in range(10, 0, -1)],
            "open": range(10), "high": range(10),
            "low": range(10), "close": range(10), "volume": range(10),
        })
        
        engine = MagicMock()
        engine.fetch_historical = MagicMock(return_value=recent_data)
        
        bridge = DataBridge(engine, max_staleness_seconds=1800)
        result = bridge.fetch_scan_data("NIFTY 50")
        
        assert result.status == DataStatus.OK
        assert result.staleness_seconds < 1800
