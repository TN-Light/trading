# ============================================================================
# PROMETHEUS — Angel One SmartAPI Data Fetcher
# ============================================================================
"""
Fetches historical candle data from Angel One SmartAPI.
Supports 5min/15min intraday data up to ~5 years back.
Used for intraday backtesting with much larger sample than yfinance (60d).
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional

from prometheus.utils.logger import logger


class AngelOneFetcher:
    """Angel One SmartAPI historical data fetcher."""

    # Angel One symbol tokens for NSE indices
    INDEX_TOKENS = {
        "NIFTY 50": "99926000",
        "NIFTY BANK": "99926009",
        "SENSEX": "99919000",       # BSE
        "NIFTY FIN SERVICE": "99926037",
        "NIFTY MIDCAP SELECT": "99926074",
        "INDIA VIX": "99926004",
    }

    # Angel One symbol tokens for NSE equities (F&O stocks)
    STOCK_TOKENS = {
        "HDFCBANK": "1333",
        "RELIANCE": "2885",
        "SBIN": "3045",
        "TATAMOTORS": "3456",
        "INFY": "1594",
        "ICICIBANK": "4963",
    }

    # Exchange mapping
    INDEX_EXCHANGES = {
        "NIFTY 50": "NSE",
        "NIFTY BANK": "NSE",
        "SENSEX": "BSE",
        "NIFTY FIN SERVICE": "NSE",
        "NIFTY MIDCAP SELECT": "NSE",
        "INDIA VIX": "NSE",
    }

    # Interval mapping
    INTERVAL_MAP = {
        "5minute": "FIVE_MINUTE",
        "5m": "FIVE_MINUTE",
        "15minute": "FIFTEEN_MINUTE",
        "15m": "FIFTEEN_MINUTE",
        "60minute": "ONE_HOUR",
        "1h": "ONE_HOUR",
        "day": "ONE_DAY",
        "1d": "ONE_DAY",
    }

    # Max candles per request (Angel One limit ~2000)
    MAX_CANDLES_PER_REQUEST = 2000

    # Candles per day by interval (approx, 9:15 to 15:30)
    CANDLES_PER_DAY = {
        "FIVE_MINUTE": 75,
        "FIFTEEN_MINUTE": 25,
        "ONE_HOUR": 7,
        "ONE_DAY": 1,
    }

    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self._obj = None
        self._auth_token = None
        self._login_time = None
        import threading
        self._lock = threading.Lock()

    def _login(self) -> bool:
        """Login to Angel One SmartAPI with TOTP."""
        try:
            from SmartApi import SmartConnect
            import pyotp

            totp = pyotp.TOTP(self.totp_secret).now()
            self._obj = SmartConnect(api_key=self.api_key)
            data = self._obj.generateSession(self.client_code, self.password, totp)

            if data.get("status"):
                self._auth_token = data["data"]["jwtToken"]
                self._login_time = datetime.now()
                logger.info(f"Angel One login successful: {data['data'].get('name', 'unknown')}")
                return True
            else:
                logger.error(f"Angel One login failed: {data}")
                return False
        except Exception as e:
            logger.error(f"Angel One login error: {e}")
            return False

    def _ensure_connected(self) -> bool:
        """Ensure we have a valid session (re-login if >6 hours old)."""
        with self._lock:
            if self._obj is None or self._login_time is None:
                return self._login()
            if (datetime.now() - self._login_time).total_seconds() > 6 * 3600:
                return self._login()
            return True

    def fetch_historical(
        self,
        symbol: str,
        days: int = 365,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """
        Fetch historical candle data from Angel One.

        Fetches in 30-day chunks to stay within API limits.
        Returns DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        if not self._ensure_connected():
            return pd.DataFrame()

        token = self.INDEX_TOKENS.get(symbol) or self.STOCK_TOKENS.get(symbol)
        exchange = self.INDEX_EXCHANGES.get(symbol, "NSE")
        ao_interval = self.INTERVAL_MAP.get(interval)

        if not token:
            logger.error(f"Angel One: unknown symbol '{symbol}'. Known: {list(self.INDEX_TOKENS.keys()) + list(self.STOCK_TOKENS.keys())}")
            return pd.DataFrame()
        if not ao_interval:
            logger.error(f"Angel One: unknown interval '{interval}'. Known: {list(self.INTERVAL_MAP.keys())}")
            return pd.DataFrame()

        # Calculate chunk size (days per request)
        cpd = self.CANDLES_PER_DAY.get(ao_interval, 75)
        days_per_chunk = max(1, self.MAX_CANDLES_PER_REQUEST // cpd)
        # Cap at 30 days per chunk for reliability
        days_per_chunk = min(days_per_chunk, 30)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        all_candles = []
        chunk_start = start_date
        request_count = 0

        logger.info(f"Angel One: fetching {symbol} {interval} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)")

        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=days_per_chunk), end_date)

            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": ao_interval,
                "fromdate": chunk_start.strftime("%Y-%m-%d 09:15"),
                "todate": chunk_end.strftime("%Y-%m-%d 15:30"),
            }

            # Retry with exponential backoff for rate limiting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self._obj.getCandleData(params)

                    if result and result.get("status") and result.get("data"):
                        candles = result["data"]
                        all_candles.extend(candles)
                        request_count += 1

                        if request_count % 20 == 0:
                            logger.info(f"  ... fetched {len(all_candles)} candles so far ({chunk_start.strftime('%Y-%m-%d')})")
                        break  # Success, exit retry loop

                    # Rate limiting: ~3 requests/sec to be safe
                    time.sleep(0.35)
                    break  # No data but no error, don't retry

                except Exception as e:
                    err_msg = str(e)
                    if "exceeding access rate" in err_msg.lower() or "access denied" in err_msg.lower():
                        wait = (2 ** attempt)  # 1s, 2s, 4s
                        logger.warning(f"Angel One rate limited at {chunk_start.strftime('%Y-%m-%d')} (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                        time.sleep(wait)
                        if attempt == max_retries - 1:
                            logger.error(f"Angel One: chunk {chunk_start.strftime('%Y-%m-%d')} failed after {max_retries} retries")
                    else:
                        logger.warning(f"Angel One fetch error at {chunk_start.strftime('%Y-%m-%d')}: {e}")
                        time.sleep(1)
                        break  # Non-rate-limit errors don't benefit from retry

            # Small delay between chunks to avoid rate limits
            time.sleep(0.35)
            chunk_start = chunk_end

        if not all_candles:
            logger.warning(f"Angel One: no data returned for {symbol} {interval}")
            return pd.DataFrame()

        # Convert to DataFrame
        # Angel One candle format: [timestamp_str, open, high, low, close, volume]
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Remove timezone info to match yfinance format
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # Remove duplicates (overlapping chunks)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Angel One: fetched {len(df)} candles for {symbol} {interval} ({request_count} API calls)")

        return df

    def is_available(self) -> bool:
        """Check if Angel One credentials are configured."""
        return bool(self.api_key and self.client_code and self.password and self.totp_secret)


def create_angelone_fetcher() -> Optional[AngelOneFetcher]:
    """Create an AngelOneFetcher from credentials.yaml config."""
    try:
        import yaml
        from pathlib import Path

        cred_path = Path(__file__).parent.parent / "config" / "credentials.yaml"
        if not cred_path.exists():
            return None

        with open(cred_path) as f:
            creds = yaml.safe_load(f)

        ao = creds.get("angelone", {})
        api_key = ao.get("api_key", "")
        client_code = ao.get("client_code", "")
        password = ao.get("password", "")
        totp_secret = ao.get("totp_secret", "")

        if not all([api_key, client_code, password, totp_secret]):
            return None
        if "your_" in client_code or "your_" in password:
            return None

        return AngelOneFetcher(api_key, client_code, password, totp_secret)
    except Exception as e:
        logger.warning(f"Angel One fetcher init failed: {e}")
        return None
