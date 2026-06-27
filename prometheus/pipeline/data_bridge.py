"""DataBridge: Fetches and validates market data for live scanning."""

import pandas as pd
from datetime import datetime
from typing import Optional
from prometheus.pipeline.types import ScanData, DataStatus
from prometheus.utils.logger import logger

REQUIRED_COLUMNS = {'timestamp', 'open', 'high', 'low', 'close'}
MAX_STALENESS_SECONDS = 1800  # 30 minutes


def _resample_to_hourly(df_15min: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute OHLCV data to 60-minute bars.
    
    This eliminates a separate API call for hourly data, saving ~1.7s per symbol.
    """
    if df_15min is None or df_15min.empty:
        return pd.DataFrame()
    
    try:
        df = df_15min.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            return df  # Can't resample without datetime index
        
        # OHLCV resampling
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }
        if 'volume' in df.columns:
            agg['volume'] = 'sum'
        
        hourly = df.resample('1h').agg(agg).dropna(subset=['open'])
        hourly = hourly.reset_index()
        hourly.rename(columns={'index': 'timestamp'}, inplace=True)
        
        return hourly
    except Exception as e:
        logger.debug(f"DataBridge: resample failed: {e}")
        return df_15min


class DataBridge:
    """Fetches market data for live scanning with validation."""
    
    def __init__(self, data_engine, max_staleness_seconds: int = MAX_STALENESS_SECONDS):
        self._data = data_engine
        self._max_staleness = max_staleness_seconds
    
    def fetch_scan_data(self, symbol: str, interval: str = '15minute') -> ScanData:
        """Fetch and validate data for a symbol.
        
        Returns ScanData with status indicating success/failure.
        Never raises - all errors are captured in the status field.
        
        Optimization: resamples 15min → 60min instead of separate API call.
        """
        fetch_time = datetime.now()
        
        try:
            # Primary MUST be fresh for live scanning — bypass cache
            primary = self._data.fetch_historical(
                symbol, days=60, interval=interval, force_refresh=True
            )
            # Resample 15min → 60min (saves one API call ~1.7s per symbol)
            hourly = _resample_to_hourly(primary)
            # Daily can use cache — doesn't change intraday
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
        
        # Check if data is from a previous day (market closed/ad-hoc holiday)
        if scan_data.last_bar_time and scan_data.last_bar_time.date() < fetch_time.date():
            scan_data.status = DataStatus.MARKET_CLOSED
            scan_data.error_message = f"Data is from previous day {scan_data.last_bar_time.date()}"
            logger.info(f"DataBridge: {symbol} market appears closed (last bar: {scan_data.last_bar_time.date()})")
        # Check staleness
        elif scan_data.staleness_seconds > self._max_staleness:
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
