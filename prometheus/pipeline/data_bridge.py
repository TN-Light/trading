"""DataBridge: Fetches and validates market data for live scanning."""

import pandas as pd
from datetime import datetime
from typing import Optional
from prometheus.pipeline.types import ScanData, DataStatus
from prometheus.utils.logger import logger

REQUIRED_COLUMNS = {'timestamp', 'open', 'high', 'low', 'close'}
MAX_STALENESS_SECONDS = 1800  # 30 minutes

class DataBridge:
    """Fetches market data for live scanning with validation."""
    
    def __init__(self, data_engine, max_staleness_seconds: int = MAX_STALENESS_SECONDS):
        self._data = data_engine
        self._max_staleness = max_staleness_seconds
    
    def fetch_scan_data(self, symbol: str, interval: str = '15minute') -> ScanData:
        """Fetch and validate data for a symbol.
        
        Returns ScanData with status indicating success/failure.
        Never raises - all errors are captured in the status field.
        """
        fetch_time = datetime.now()
        
        try:
            primary = self._data.fetch_historical(symbol, days=60, interval=interval)
            hourly = self._data.fetch_historical(symbol, days=60, interval='60minute')
            daily = self._data.fetch_historical(symbol, days=365, interval='day')
        except Exception as e:
            logger.error(f"DataBridge: fetch error for {symbol}: {e}")
            return ScanData(
                symbol=symbol,
                primary=pd.DataFrame(),
                hourly=pd.DataFrame(),
                daily=pd.DataFrame(),
                status=DataStatus.FETCH_ERROR,
                fetch_time=fetch_time,
                error_message=str(e),
            )
        
        # Validate primary data
        if primary is None or primary.empty:
            logger.warning(f"DataBridge: empty primary data for {symbol}")
            return ScanData(
                symbol=symbol,
                primary=pd.DataFrame(),
                hourly=hourly if hourly is not None else pd.DataFrame(),
                daily=daily if daily is not None else pd.DataFrame(),
                status=DataStatus.EMPTY,
                fetch_time=fetch_time,
                error_message=f"No {interval} data returned for {symbol}",
            )
        
        # Check required columns
        missing = REQUIRED_COLUMNS - set(c.lower() for c in primary.columns)
        if missing:
            logger.warning(f"DataBridge: missing columns {missing} for {symbol}")
            return ScanData(
                symbol=symbol,
                primary=primary,
                hourly=hourly if hourly is not None else pd.DataFrame(),
                daily=daily if daily is not None else pd.DataFrame(),
                status=DataStatus.MISSING_COLUMNS,
                fetch_time=fetch_time,
                error_message=f"Missing columns: {missing}",
            )
        
        # Fill defaults for optional frames
        if hourly is None or hourly.empty:
            hourly = primary
        if daily is None or daily.empty:
            daily = primary
        
        scan_data = ScanData(
            symbol=symbol,
            primary=primary,
            hourly=hourly,
            daily=daily,
            status=DataStatus.OK,
            fetch_time=fetch_time,
        )
        
        # Check staleness
        if scan_data.staleness_seconds > self._max_staleness:
            scan_data.status = DataStatus.STALE
            scan_data.error_message = (
                f"Data is {scan_data.staleness_seconds:.0f}s old "
                f"(max {self._max_staleness}s)"
            )
            logger.warning(f"DataBridge: stale data for {symbol}: {scan_data.error_message}")
        else:
            logger.info(
                f"DataBridge: {symbol} OK — {scan_data.bar_count} bars, "
                f"last bar {scan_data.last_bar_time}, "
                f"staleness {scan_data.staleness_seconds:.0f}s"
            )
        
        return scan_data
