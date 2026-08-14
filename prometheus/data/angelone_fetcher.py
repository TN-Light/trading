# ============================================================================
# PROMETHEUS — Angel One SmartAPI Data Fetcher
# ============================================================================
"""
Fetches historical candle data from Angel One SmartAPI.
Supports 5min/15min intraday data up to ~5 years back.
Used for intraday backtesting with much larger sample than yfinance (60d).

Rate-limit strategy (Aug 2026):
  Angel One enforces ~3 req/sec and 180 req/min on getCandleData.
  The SmartAPI SDK has its own internal urllib3 retry that escalates to
  131-second waits on HTTP 429.  To prevent that cascade we:
    1. Gate every API call through a thread-safe SmartAPIRateLimiter
       (token-bucket style, 1.0s minimum between calls).
    2. On AB1021 / 429, set a global cooldown timestamp that ALL threads
       respect before their next attempt.
    3. Suppress the SDK's internal retry count so control returns to our
       code quickly on failure.
    4. Use aggressive backoff (5s / 10s / 20s) and fail after 3 retries
       per chunk — then fall back to yfinance in the DataEngine layer.
"""

import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from typing import Optional

from prometheus.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════
# Thread-safe rate limiter — enforces minimum interval between API calls
# ═══════════════════════════════════════════════════════════════════════════
class SmartAPIRateLimiter:
    """Centralised sliding-window rate limiter for SmartAPI calls.

    Guarantees a minimum ``delay_between_calls`` gap between successive
    calls across ALL threads / symbols.  Any thread calling ``wait()``
    will block until at least ``delay`` seconds have elapsed since the
    last API call.  Thread-safe via a ``threading.Lock``.
    """

    def __init__(self, delay_between_calls: float = 1.0):
        self.delay = delay_between_calls
        self._last_call: float = 0.0
        self._lock = threading.Lock()
        # Global cooldown: when a 429/AB1021 is received, ALL threads
        # must wait until this monotonic timestamp before retrying.
        self._cooldown_until: float = 0.0

    def wait(self):
        """Block until the next API call is safe to make."""
        with self._lock:
            now = time.monotonic()
            # Respect global cooldown first
            if now < self._cooldown_until:
                sleep_for = self._cooldown_until - now
                logger.debug(f"[RateLimiter] global cooldown: sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
                now = time.monotonic()
            # Enforce per-call delay
            elapsed = now - self._last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_call = time.monotonic()

    def set_cooldown(self, seconds: float):
        """Set a global cooldown after a rate-limit hit.

        All threads will wait until ``seconds`` from now before making
        their next API call.
        """
        with self._lock:
            target = time.monotonic() + seconds
            # Only extend, never shorten, an existing cooldown
            if target > self._cooldown_until:
                self._cooldown_until = target
                logger.info(f"[RateLimiter] global cooldown set: {seconds:.0f}s")


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
        self._lock = threading.Lock()
        # Shared rate limiter — 1.0s minimum gap between getCandleData calls
        self._rate_limiter = SmartAPIRateLimiter(delay_between_calls=1.0)

    def _login(self) -> bool:
        """Login to Angel One SmartAPI with TOTP."""
        try:
            from SmartApi import SmartConnect
            import pyotp

            totp = pyotp.TOTP(self.totp_secret).now()
            self._obj = SmartConnect(api_key=self.api_key)

            # ── Suppress SDK internal urllib3 retries ──────────────────
            # The SDK's HTTP session uses urllib3 Retry with aggressive
            # escalation (up to 131s waits on 429).  We override it so
            # control returns to OUR retry loop on the first failure.
            try:
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry

                low_retry = Retry(
                    total=1,                    # 1 retry max inside the SDK
                    backoff_factor=0.5,         # 0.5s between retries
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "POST"],
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(max_retries=low_retry)
                # SmartConnect stores its session in self._obj.session
                if hasattr(self._obj, 'session'):
                    self._obj.session.mount("https://", adapter)
                    self._obj.session.mount("http://", adapter)
                    logger.debug("Angel One: SDK internal retries capped at 1")
                # Some SDK versions use requestInstance / _session
                elif hasattr(self._obj, 'requestInstance'):
                    sess = getattr(self._obj.requestInstance, 'session', None) or \
                           getattr(self._obj.requestInstance, 's', None)
                    if sess:
                        sess.mount("https://", adapter)
                        sess.mount("http://", adapter)
                        logger.debug("Angel One: SDK internal retries capped at 1 (requestInstance)")
            except Exception as e:
                logger.debug(f"Angel One: could not override SDK retries: {e}")

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
        chunk_failed = False

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

            # ── Retry with aggressive backoff for rate limiting ───────
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    # Gate through the centralised rate limiter
                    self._rate_limiter.wait()
                    result = self._obj.getCandleData(params)

                    if result and result.get("status") and result.get("data"):
                        candles = result["data"]
                        all_candles.extend(candles)
                        request_count += 1
                        success = True

                        if request_count % 20 == 0:
                            logger.info(f"  ... fetched {len(all_candles)} candles so far ({chunk_start.strftime('%Y-%m-%d')})")
                        break  # Success, exit retry loop

                    # Check for rate limit returned as a dict (status=False, errorcode=AB1021)
                    if result and not result.get("status"):
                        err_code = str(result.get("errorcode", ""))
                        err_msg = str(result.get("message", ""))
                        if err_code == "AB1021" or "too many requests" in err_msg.lower():
                            # Aggressive backoff: 5s, 10s, 20s
                            wait = min(5 * (2 ** attempt), 60)
                            # Set global cooldown so other threads/symbols also back off
                            self._rate_limiter.set_cooldown(wait)
                            logger.warning(
                                f"Angel One rate limited (AB1021) on {symbol} "
                                f"at {chunk_start.strftime('%Y-%m-%d')} "
                                f"(attempt {attempt+1}/{max_retries}), waiting {wait}s..."
                            )
                            time.sleep(wait)
                            if attempt == max_retries - 1:
                                logger.error(
                                    f"Angel One: chunk {chunk_start.strftime('%Y-%m-%d')} "
                                    f"failed after {max_retries} retries (AB1021)"
                                )
                            continue  # Retry!

                    # No data but no error, don't retry
                    break

                except Exception as e:
                    err_msg = str(e)
                    is_rate_limit = (
                        "exceeding access rate" in err_msg.lower() or
                        "access denied" in err_msg.lower() or
                        "ab1021" in err_msg.lower() or
                        "429" in err_msg or
                        "too many requests" in err_msg.lower()
                    )
                    if is_rate_limit:
                        wait = min(5 * (2 ** attempt), 60)  # 5s, 10s, 20s (cap 60s)
                        self._rate_limiter.set_cooldown(wait)
                        logger.warning(
                            f"Angel One rate limited on {symbol} "
                            f"at {chunk_start.strftime('%Y-%m-%d')} "
                            f"(attempt {attempt+1}/{max_retries}), waiting {wait}s..."
                        )
                        time.sleep(wait)
                        if attempt == max_retries - 1:
                            logger.error(
                                f"Angel One: chunk {chunk_start.strftime('%Y-%m-%d')} "
                                f"failed after {max_retries} retries"
                            )
                    else:
                        logger.warning(f"Angel One fetch error at {chunk_start.strftime('%Y-%m-%d')}: {e}")
                        time.sleep(1)
                        break  # Non-rate-limit errors don't benefit from retry

            if not success:
                chunk_failed = True
                break

            # Inter-chunk delay — keeps us well under the 180 req/min ceiling
            time.sleep(1.5)
            chunk_start = chunk_end

        if chunk_failed:
            logger.error(f"Angel One: data fetch failed because one or more chunks failed for {symbol}")
            return pd.DataFrame()

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
