# ============================================================================
# PROMETHEUS — High-Fidelity Mock SmartConnect Server & Reference Harness
# ============================================================================
"""
MockSmartConnectServer simulates the official Angel One SmartAPI server
with strict rate-limiting quotas, sliding window enforcement, payload
constraints (50 tokens max), and HTTP 429 / AB1021 error emulation.

Also provides reference interface contracts for progressive testability:
- PriorityTier
- EndpointCategory
- MicroCacheEntry
- ReferencePrioritizedTokenBucketLimiter
- ReferenceBatchMarketDataEngine
- ReferenceSharedPriceMicroCache
- ReferenceFallbackPolicy
"""

import time
import threading
from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union

from prometheus.utils.logger import logger


class SmartAPIRateLimitError(Exception):
    """Exception raised when SmartAPI quota ceiling is breached."""
    def __init__(self, message: str = "AB1021: Rate Limit Exceeded", error_code: str = "AB1021"):
        super().__init__(message)
        self.error_code = error_code


class MockSmartConnectServer:
    """High-fidelity mock simulating Angel One SmartAPI endpoint quotas,
    sliding window rate limits, payload batch limits, network latency, and error responses.

    Official Angel One Quotas:
    - Quotes / Market Data: 10.0 req/sec (1.0s sliding window)
    - Historical Data:       3.0 req/sec (1.0s sliding window)
    - Orders:               10.0 req/sec (1.0s sliding window)
    """

    def __init__(
        self,
        max_quote_rate: float = 10.0,
        max_hist_rate: float = 3.0,
        max_order_rate: float = 10.0,
        max_tokens_per_batch: int = 50,
        simulated_latency_sec: float = 0.0,
        raise_on_quota_exceeded: bool = False,
    ):
        self.max_quote_rate = float(max_quote_rate)
        self.max_hist_rate = float(max_hist_rate)
        self.max_order_rate = float(max_order_rate)
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.simulated_latency_sec = float(simulated_latency_sec)
        self.raise_on_quota_exceeded = bool(raise_on_quota_exceeded)

        # Sliding window timestamp logs
        self._quote_timestamps: List[float] = []
        self._hist_timestamps: List[float] = []
        self._order_timestamps: List[float] = []
        self._lock = threading.Lock()

        # Telemetry metrics
        self.total_calls: int = 0
        self.quotes_calls: int = 0
        self.hist_calls: int = 0
        self.order_calls: int = 0
        self.batch_calls: int = 0
        self.total_tokens_requested: int = 0
        self.http_429_count: int = 0
        self.ab1021_count: int = 0
        self.payload_rejected_count: int = 0

        # Fault injection toggles
        self.inject_timeout: bool = False
        self.inject_ab1021: bool = False
        self.custom_prices: Dict[str, float] = {}

    def reset_metrics(self) -> None:
        """Reset all call counters and sliding window history."""
        with self._lock:
            self._quote_timestamps.clear()
            self._hist_timestamps.clear()
            self._order_timestamps.clear()
            self.total_calls = 0
            self.quotes_calls = 0
            self.hist_calls = 0
            self.order_calls = 0
            self.batch_calls = 0
            self.total_tokens_requested = 0
            self.http_429_count = 0
            self.ab1021_count = 0
            self.payload_rejected_count = 0
            self.inject_timeout = False
            self.inject_ab1021 = False

    def _check_rate_limit(
        self,
        timestamps: List[float],
        max_rate: float,
        endpoint_name: str
    ) -> Optional[Dict[str, Any]]:
        """Prune timestamps older than 1.0s and verify against max_rate."""
        now = time.monotonic()
        window_start = now - 1.0
        while timestamps and timestamps[0] <= window_start:
            timestamps.pop(0)

        if len(timestamps) >= max_rate or self.inject_ab1021:
            self.ab1021_count += 1
            self.http_429_count += 1
            if self.raise_on_quota_exceeded:
                raise SmartAPIRateLimitError(
                    f"HTTP 429 / AB1021: Too many requests on {endpoint_name}. "
                    f"Exceeded quota {max_rate:.1f} req/sec.",
                    error_code="AB1021"
                )
            return {
                "status": False,
                "errorcode": "AB1021",
                "message": f"Too many requests / Exceeding access rate limit on {endpoint_name}",
                "data": None,
            }

        timestamps.append(now)
        return None

    def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
        """Simulate Angel One getMarketData('FULL') endpoint.

        Payload format:
            {"NFO": ["35001", "35002"], "BFO": ["845120"], "NSE": ["99926000"]}
        Maximum allowed total tokens across all exchanges is 50.
        """
        if self.inject_timeout:
            raise TimeoutError("Simulated broker socket read timeout")

        with self._lock:
            self.total_calls += 1
            self.quotes_calls += 1

            # 1. Enforce sliding-window rate limit (10.0 req/s)
            rate_err = self._check_rate_limit(self._quote_timestamps, self.max_quote_rate, "getMarketData")
            if rate_err is not None:
                return rate_err

            # 2. Extract and validate all tokens in payload
            all_tokens: List[tuple] = []
            if isinstance(payload, dict):
                segments = payload.get("exchangeTokens", payload) if "exchangeTokens" in payload else payload
                for exch, tok_list in segments.items():
                    if isinstance(tok_list, list):
                        for t in tok_list:
                            all_tokens.append((exch, str(t)))

            total_token_count = len(all_tokens)
            self.total_tokens_requested += total_token_count

            # 3. Enforce maximum 50 tokens per batch constraint
            if total_token_count > self.max_tokens_per_batch:
                self.payload_rejected_count += 1
                return {
                    "status": False,
                    "errorcode": "AB1010",
                    "message": f"Payload token count ({total_token_count}) exceeds maximum allowed ({self.max_tokens_per_batch})",
                    "data": None,
                }

            self.batch_calls += 1

        if self.simulated_latency_sec > 0:
            time.sleep(self.simulated_latency_sec)

        # 4. Generate response
        fetched = []
        for exch, tok in all_tokens:
            try:
                tok_int = abs(int(tok))
            except ValueError:
                tok_int = sum(ord(c) for c in tok)

            base_price = self.custom_prices.get(tok, 150.0 + (tok_int % 100))
            fetched.append({
                "exchange": exch,
                "tradingSymbol": f"MOCK_{exch}_{tok}",
                "symbolToken": tok,
                "ltp": float(base_price),
                "open": float(base_price - 5.0),
                "high": float(base_price + 10.0),
                "low": float(base_price - 8.0),
                "close": float(base_price),
                "bestBidPrice": round(base_price - 0.25, 2),
                "bestAskPrice": round(base_price + 0.25, 2),
                "opnInterest": 1500000 + tok_int * 100,
                "tradeVolume": 75000 + tok_int * 20,
            })

        return {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": {
                "fetched": fetched,
                "unfetched": []
            }
        }

    def ltpData(self, exchange: str, tradingsymbol: str, symboltoken: str) -> Dict[str, Any]:
        """Simulate Angel One ltpData endpoint."""
        if self.inject_timeout:
            raise TimeoutError("Simulated broker socket read timeout")

        with self._lock:
            self.total_calls += 1
            self.quotes_calls += 1
            self.total_tokens_requested += 1

            rate_err = self._check_rate_limit(self._quote_timestamps, self.max_quote_rate, "ltpData")
            if rate_err is not None:
                return rate_err

        if self.simulated_latency_sec > 0:
            time.sleep(self.simulated_latency_sec)

        tok_str = str(symboltoken)
        try:
            tok_int = abs(int(tok_str))
        except ValueError:
            tok_int = sum(ord(c) for c in tok_str)

        price = self.custom_prices.get(tok_str, 150.0 + (tok_int % 100))
        return {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": {
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "symboltoken": tok_str,
                "open": price - 5.0,
                "high": price + 10.0,
                "low": price - 8.0,
                "close": price,
                "ltp": float(price),
            }
        }

    def getCandleData(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Angel One getCandleData endpoint (Historical 3.0 req/s quota)."""
        if self.inject_timeout:
            raise TimeoutError("Simulated broker socket read timeout")

        with self._lock:
            self.total_calls += 1
            self.hist_calls += 1

            rate_err = self._check_rate_limit(self._hist_timestamps, self.max_hist_rate, "getCandleData")
            if rate_err is not None:
                return rate_err

        if self.simulated_latency_sec > 0:
            time.sleep(self.simulated_latency_sec)

        candles = [
            ["2026-10-08T09:15:00+05:30", 25000.0, 25050.0, 24980.0, 25020.0, 100000],
            ["2026-10-08T09:30:00+05:30", 25020.0, 25080.0, 25010.0, 25060.0, 120000],
            ["2026-10-08T09:45:00+05:30", 25060.0, 25100.0, 25040.0, 25080.0, 95000],
            ["2026-10-08T10:00:00+05:30", 25080.0, 25110.0, 25050.0, 25070.0, 110000],
            ["2026-10-08T10:15:00+05:30", 25070.0, 25120.0, 25060.0, 25115.0, 130000],
        ]
        return {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": candles,
        }

    def placeOrder(self, orderparams: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order placement (Order 10.0 req/s quota)."""
        if self.inject_timeout:
            raise TimeoutError("Simulated broker socket read timeout")

        with self._lock:
            self.total_calls += 1
            self.order_calls += 1

            rate_err = self._check_rate_limit(self._order_timestamps, self.max_order_rate, "placeOrder")
            if rate_err is not None:
                return rate_err

        if self.simulated_latency_sec > 0:
            time.sleep(self.simulated_latency_sec)

        return {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": {"orderid": f"ORD_{int(time.monotonic()*1000)}"}
        }

    def modifyOrder(self, orderparams: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order modification (Order 10.0 req/s quota)."""
        if self.inject_timeout:
            raise TimeoutError("Simulated broker socket read timeout")

        with self._lock:
            self.total_calls += 1
            self.order_calls += 1

            rate_err = self._check_rate_limit(self._order_timestamps, self.max_order_rate, "modifyOrder")
            if rate_err is not None:
                return rate_err

        if self.simulated_latency_sec > 0:
            time.sleep(self.simulated_latency_sec)

        return {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": {"orderid": orderparams.get("orderid", "ORD_MODIFIED")}
        }


# ============================================================================
# Interface Contracts & Reference Implementations for Progressive Testing
# ============================================================================

class PriorityTier(IntEnum):
    P1_ORDER_EXECUTION = 1
    P2_POSITION_MONITOR = 2
    P3_HEALTH_EVALUATION = 3
    P4_BACKGROUND_SCANNER = 4


class EndpointCategory(Enum):
    HISTORICAL = "historical"  # 2.5 req/s (broker limit: 3.0 req/s)
    QUOTES = "quotes"          # 8.5 req/s (broker limit: 10.0 req/s)
    ORDERS = "orders"          # 8.5 req/s (broker limit: 10.0 req/s)


@dataclass
class MicroCacheEntry:
    price: float
    bid: float
    ask: float
    open_interest: int
    volume: int
    timestamp: float
    source: str  # "websocket" | "batch_rest" | "fallback"
    is_stale: bool


class ReferencePrioritizedTokenBucketLimiter:
    """Contract 1 Reference Implementation.
    Endpoint-differentiated token buckets capped at 85% broker quotas.
    Sleeping occurs strictly OUTSIDE mutex locks.
    """

    def __init__(self):
        # 85% caps:
        # Historical: 2.5 req/s (capacity 3)
        # Quotes: 8.5 req/s (capacity 10)
        # Orders: 8.5 req/s (capacity 5)
        self.configs = {
            EndpointCategory.HISTORICAL: {"rate": 2.5, "cap": 3.0, "reserved": 0.0, "max_window": 3},
            EndpointCategory.QUOTES: {"rate": 8.5, "cap": 10.0, "reserved": 2.0, "max_window": 8},
            EndpointCategory.ORDERS: {"rate": 8.5, "cap": 5.0, "reserved": 1.0, "max_window": 8},
        }

        self.tokens = {k: cfg["cap"] for k, cfg in self.configs.items()}
        self.last_refill = {k: time.monotonic() for k in self.configs}
        self.cooldown_until = {k: 0.0 for k in self.configs}
        self._dispatch_history = {k: [] for k in self.configs}
        self._lock = threading.Lock()

    def set_cooldown(self, endpoint: EndpointCategory, seconds: float) -> None:
        with self._lock:
            target = time.monotonic() + seconds
            if target > self.cooldown_until[endpoint]:
                self.cooldown_until[endpoint] = target

    def acquire(self, endpoint: EndpointCategory, priority: PriorityTier = PriorityTier.P2_POSITION_MONITOR, timeout: float = 10.0) -> bool:
        """Acquire 1 token. Sleeping strictly occurs OUTSIDE the mutex lock."""
        deadline = time.monotonic() + timeout
        cfg = self.configs[endpoint]
        rate = cfg["rate"]
        cap = cfg["cap"]
        reserved = cfg["reserved"]
        max_window = cfg.get("max_window", int(rate))

        while True:
            now = time.monotonic()
            if now >= deadline:
                return False

            sleep_time = 0.0
            with self._lock:
                # 1. Check cooldown
                if now < self.cooldown_until[endpoint]:
                    sleep_time = min(self.cooldown_until[endpoint] - now, 0.05)
                else:
                    # 2. Prune 1.0-second sliding window
                    history = self._dispatch_history[endpoint]
                    window_start = now - 1.0
                    while history and history[0] <= window_start:
                        history.pop(0)

                    # 3. Refill tokens
                    elapsed = now - self.last_refill[endpoint]
                    self.tokens[endpoint] = min(cap, self.tokens[endpoint] + elapsed * rate)
                    self.last_refill[endpoint] = now

                    # 4. Check sliding window cap (85% of broker limit)
                    if len(history) >= max_window:
                        wait_for_window = (history[0] + 1.0) - now + 0.005
                        sleep_time = max(0.005, min(0.05, wait_for_window))
                    else:
                        # 5. Token check based on priority tier
                        min_required = 0.0 if priority == PriorityTier.P1_ORDER_EXECUTION else reserved
                        if self.tokens[endpoint] >= (min_required + 1.0):
                            self.tokens[endpoint] -= 1.0
                            history.append(now)
                            return True
                        else:
                            deficit = (min_required + 1.0) - self.tokens[endpoint]
                            sleep_time = max(0.005, min(0.05, deficit / rate))

            # Sleep OUTSIDE mutex lock
            time.sleep(sleep_time)


class ReferenceBatchMarketDataEngine:
    """Contract 2 Reference Implementation.
    Coalesces multi-token quote requests into <=50 token chunks,
    calling client getMarketData('FULL').
    """

    def __init__(self, client: Any, limiter: Optional[ReferencePrioritizedTokenBucketLimiter] = None):
        self.client = client
        self.limiter = limiter or ReferencePrioritizedTokenBucketLimiter()

    def get_quotes_batch(
        self,
        requests: List[Dict[str, str]],
        priority: PriorityTier = PriorityTier.P2_POSITION_MONITOR,
        timeout: float = 2.0
    ) -> Dict[str, Dict[str, Any]]:
        """Coalesces requests into <=50 token chunks and calls getMarketData('FULL')."""
        if not requests:
            return {}

        # Deduplicate while preserving exchange
        unique_reqs: Dict[str, str] = {}
        for r in requests:
            tok = str(r.get("symboltoken", r.get("symbol_token", r.get("token", "")))).strip()
            exch = str(r.get("exchange", "NFO")).strip().upper()
            if tok:
                unique_reqs[tok] = exch

        if not unique_reqs:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        tokens_list = list(unique_reqs.items())

        # Chunk into slices of max 50 tokens
        chunk_size = 50
        for i in range(0, len(tokens_list), chunk_size):
            chunk = tokens_list[i : i + chunk_size]

            # Partition by exchange
            payload: Dict[str, List[str]] = {}
            for tok, exch in chunk:
                payload.setdefault(exch, []).append(tok)

            # Acquire permission from limiter
            if not self.limiter.acquire(EndpointCategory.QUOTES, priority=priority, timeout=timeout):
                logger.warning("BatchMarketDataEngine rate limiter timeout")
                continue

            resp = self.client.getMarketData("FULL", payload)
            if resp and resp.get("status") and resp.get("data"):
                fetched = resp["data"].get("fetched", [])
                for item in fetched:
                    tok = str(item.get("symbolToken", item.get("symboltoken", "")))
                    results[tok] = {
                        "ltp": float(item.get("ltp", 0.0)),
                        "bid": float(item.get("bestBidPrice", 0.0) or 0.0),
                        "ask": float(item.get("bestAskPrice", 0.0) or 0.0),
                        "open_interest": int(item.get("opnInterest", 0) or 0),
                        "volume": int(item.get("tradeVolume", 0) or 0),
                        "exchange": item.get("exchange", "NFO"),
                        "tradingsymbol": item.get("tradingSymbol", ""),
                    }

        return results


class ReferenceSharedPriceMicroCache:
    """Contract 3 Reference Implementation.
    Lock-free read-optimized micro-cache with sub-second TTL (default 750ms).
    """

    def __init__(self, default_ttl: float = 0.750):
        self.default_ttl = float(default_ttl)
        self._cache: Dict[str, MicroCacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, token: str, max_age: Optional[float] = None) -> Optional[MicroCacheEntry]:
        """Returns non-stale cached quote if age <= max_age, else None."""
        ttl = self.default_ttl if max_age is None else max_age
        entry = self._cache.get(str(token))
        if entry is None:
            return None
        now = time.monotonic()
        if (now - entry.timestamp) <= ttl:
            return entry
        return None

    def put(self, token: str, entry: MicroCacheEntry) -> None:
        """Atomically updates cached entry."""
        self._cache[str(token)] = entry

    def get_or_fetch(self, tokens: List[str], fetch_func: Callable[[List[str]], Dict[str, MicroCacheEntry]]) -> Dict[str, MicroCacheEntry]:
        """Returns cached entries; fetches missing/stale entries in a single batch."""
        res: Dict[str, MicroCacheEntry] = {}
        missing: List[str] = []
        now = time.monotonic()

        for t in tokens:
            t_str = str(t)
            entry = self._cache.get(t_str)
            if entry is not None and (now - entry.timestamp) <= self.default_ttl:
                res[t_str] = entry
            else:
                missing.append(t_str)

        if missing:
            fetched = fetch_func(missing)
            for t_str, entry in fetched.items():
                self.put(t_str, entry)
                res[t_str] = entry

        return res


class ReferenceFallbackPolicy:
    """Contract 4 Reference Implementation.
    Non-hallucinatory fallback: logs [STALE_DATA_ALERT], freezes trailing ratchets,
    and resolves secondary spot feeds without price fabrication.
    """

    stale_alerts: List[Dict[str, Any]] = []

    @classmethod
    def reset_alerts(cls) -> None:
        cls.stale_alerts.clear()

    @classmethod
    def handle_option_stale(cls, symbol: str, token: str, last_known_price: float, age_seconds: float) -> None:
        """Logs [STALE_DATA_ALERT], freezes trailing ratchets, keeps hard exchange SL intact without price fabrication."""
        alert = {
            "symbol": symbol,
            "token": token,
            "last_known_price": last_known_price,
            "age_seconds": age_seconds,
            "timestamp": time.monotonic(),
        }
        cls.stale_alerts.append(alert)
        logger.warning(
            f"[STALE_DATA_ALERT] Quote stale for {symbol} (token {token}, age {age_seconds:.1f}s). "
            f"Freezing trailing ratchet at last known LTP {last_known_price:.2f}. "
            f"Zero synthetic prices permitted."
        )

    _secondary_spot_feed: Optional[Callable[[str], Optional[float]]] = None

    @classmethod
    def set_secondary_spot_feed(cls, feed_func: Optional[Callable[[str], Optional[float]]]) -> None:
        """Register an authenticated secondary spot feed."""
        cls._secondary_spot_feed = feed_func

    @classmethod
    def get_secondary_spot_price(cls, symbol: str) -> Optional[float]:
        """Secondary verified index spot lookup (never fabricates synthetic prices)."""
        if cls._secondary_spot_feed is not None:
            try:
                price = cls._secondary_spot_feed(symbol)
                if price is not None and float(price) > 0.0:
                    return float(price)
                alert_type = "SECONDARY_SPOT_EMPTY"
                err_msg = f"returned invalid price ({price})"
            except Exception as e:
                alert_type = "SECONDARY_SPOT_FAILED"
                err_msg = f"feed exception: {e}"
        else:
            alert_type = "SECONDARY_SPOT_UNCONFIGURED"
            err_msg = "no secondary feed registered"

        alert = {
            "symbol": symbol,
            "token": "SPOT",
            "last_known_price": 0.0,
            "age_seconds": 0.0,
            "type": alert_type,
            "timestamp": time.monotonic(),
        }
        cls.stale_alerts.append(alert)
        logger.warning(
            f"[STALE_DATA_ALERT] Secondary spot price unavailable for {symbol} ({err_msg}). "
            f"Zero synthetic prices permitted."
        )
        return None
