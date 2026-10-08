# ============================================================================
# PROMETHEUS — Batch Market Data Engine & Multi-Token Coalescer (Milestone 2)
# ============================================================================
"""
Unified 50-token batch market data engine with 25ms debounce coalescing.

Eliminates single-token round trips and rate-limit contention against Angel One SmartAPI:
  - 25ms Debounce Coalescer: Aggregates concurrent queries across threads into unified batches.
  - Priority Inheritance: Batches inherit the highest priority tier among queued callers.
  - 50-Token Chunking: Partitions requests into slices of <= 50 tokens across all segment keys (AB1010 protection).
  - Cross-Segment Partitioning: Formats payloads across NFO, BFO, and NSE segments.
  - Out-of-Lock Concurrency: Mutex locks are held <5µs; rate limiter waiting and broker I/O execute strictly outside locks.
  - Future Demultiplexing: Demultiplexes responses back to individual caller futures.
  - Zero Price Synthesis: Unfetched or missing tokens are omitted from result dicts; broker timeouts propagate cleanly.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from prometheus.data.rate_limiter import (
    EndpointCategory,
    PrioritizedTokenBucketLimiter,
    PriorityTier,
    _normalize_priority,
)

logger = logging.getLogger(__name__)


@dataclass
class _BatchCallerWait:
    """Descriptor representing a waiting caller in get_quotes_batch."""
    tokens: List[str]
    token_exchanges: Dict[str, str]
    priority: int
    future: concurrent.futures.Future
    enqueued_at: float
    deadline: float


@dataclass
class _TokenRequestMeta:
    """Metadata tracking a unique token and its interested callers."""
    token: str
    exchange: str
    highest_priority: int
    callers: List[_BatchCallerWait] = field(default_factory=list)


class BatchMarketDataEngine:
    """Unified 50-token batch market data engine with 25ms debounce coalescing."""

    def __init__(
        self,
        client: Any,
        limiter: Optional[Any] = None,
        debounce_ms: float = 25.0,
        max_batch_size: int = 50,
        auto_start: bool = True,
        micro_cache: Optional[Any] = None,
    ) -> None:
        self.client = client
        self.limiter = limiter or PrioritizedTokenBucketLimiter()
        self.debounce_ms = float(debounce_ms)
        self.max_batch_size = int(max_batch_size)
        self.micro_cache = micro_cache

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._pending_callers: List[_BatchCallerWait] = []
        self._pending_tokens: Set[str] = set()

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Telemetry metrics
        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens_requested": 0,
            "total_tokens_coalesced": 0,
            "total_batches_dispatched": 0,
            "total_broker_calls": 0,
            "total_timeouts": 0,
            "total_exceptions": 0,
            "cut_through_flushes": 0,
            "debounce_flushes": 0,
        }

        if auto_start and self.debounce_ms > 0:
            self.start()

    def start(self) -> None:
        """Start the background coalescer worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._coalescer_worker_loop,
                name="BatchMarketDataCoalescer",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Gracefully stop the background worker thread."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._cv.notify_all()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def __enter__(self) -> BatchMarketDataEngine:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    # -------------------------------------------------------------------------
    # Public Synchronous & Asynchronous APIs
    # -------------------------------------------------------------------------

    def get_quotes_batch(
        self,
        requests: Optional[List[Dict[str, str]]],
        priority: Union[PriorityTier, int] = PriorityTier.P2_POSITION_MONITOR,
        timeout: float = 2.0,
    ) -> Dict[str, Dict[str, Any]]:
        """Synchronously fetch batch quotes with 25ms coalescing & priority inheritance.

        Args:
            requests: List of request dicts, e.g. [{"exchange": "NFO", "symboltoken": "35001"}]
            priority: PriorityTier enum or int (P1 to P4). Default P2.
            timeout: Maximum wait time in seconds.

        Returns:
            Dict mapping symbol token to quote data dictionary.
        """
        if not requests:
            return {}

        fut = self.get_quotes_batch_async(requests, priority=priority, timeout=timeout)
        try:
            return fut.result(timeout=max(0.0, float(timeout)))
        except concurrent.futures.TimeoutError:
            with self._lock:
                self.stats["total_timeouts"] += 1
            fut.cancel()
            return {}
        except Exception:
            with self._lock:
                self.stats["total_exceptions"] += 1
            return {}

    def get_quotes_batch_async(
        self,
        requests: Optional[List[Dict[str, str]]],
        priority: Union[PriorityTier, int] = PriorityTier.P2_POSITION_MONITOR,
        timeout: float = 2.0,
    ) -> concurrent.futures.Future:
        """Asynchronously enqueue requests and return a Future."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        if not requests:
            fut.set_result({})
            return fut

        # Extract, sanitize, and deduplicate tokens for this caller
        token_exchanges: Dict[str, str] = {}
        for r in requests:
            if not isinstance(r, dict):
                continue
            tok = str(r.get("symboltoken") or r.get("symbol_token") or r.get("token") or "").strip()
            exch = str(r.get("exchange", "NFO") or "NFO").strip().upper()
            if not exch:
                exch = "NFO"
            if tok:
                token_exchanges[tok] = exch

        token_list = list(token_exchanges.keys())
        if not token_list:
            fut.set_result({})
            return fut

        pri_val = _normalize_priority(priority)
        now = time.monotonic()
        caller_wait = _BatchCallerWait(
            tokens=token_list,
            token_exchanges=token_exchanges,
            priority=pri_val,
            future=fut,
            enqueued_at=now,
            deadline=now + timeout,
        )

        # Fast direct path if debounce is disabled or worker not running
        if self.debounce_ms <= 0 or not self._running:
            self._execute_direct_batch(caller_wait, timeout)
            return fut

        # Enqueue into the 25ms debounce accumulator under lock (<5µs hold)
        with self._lock:
            self.stats["total_requests"] += 1
            self.stats["total_tokens_requested"] += len(token_list)
            self._pending_callers.append(caller_wait)
            for tok in token_list:
                self._pending_tokens.add(tok)

            # Wake worker if cut-through capacity reached or this is the first caller
            if len(self._pending_tokens) >= self.max_batch_size:
                self._cv.notify()
            elif len(self._pending_callers) == 1:
                self._cv.notify()

        return fut

    # -------------------------------------------------------------------------
    # Internal Debounce Coalescer Loop
    # -------------------------------------------------------------------------

    def _coalescer_worker_loop(self) -> None:
        """Background worker thread executing the 25ms sliding/tick debounce window."""
        while True:
            with self._lock:
                while not self._pending_callers and self._running:
                    self._cv.wait()

                if not self._running and not self._pending_callers:
                    break

                # Debounce window evaluation
                window_start = time.monotonic()
                debounce_sec = self.debounce_ms / 1000.0
                cut_through = False

                while True:
                    if len(self._pending_tokens) >= self.max_batch_size:
                        cut_through = True
                        break

                    elapsed = time.monotonic() - window_start
                    remaining = debounce_sec - elapsed
                    if remaining <= 0:
                        break

                    self._cv.wait(timeout=remaining)

                if cut_through:
                    self.stats["cut_through_flushes"] += 1
                else:
                    self.stats["debounce_flushes"] += 1

                # Atomically detach pending batch under lock (<5µs)
                batch_callers = list(self._pending_callers)
                self._pending_callers.clear()
                self._pending_tokens.clear()

            # Process detached batch completely OUTSIDE the lock
            if batch_callers:
                self._process_coalesced_batch(batch_callers)

    def _process_coalesced_batch(self, batch_callers: List[_BatchCallerWait]) -> None:
        """Deduplicates, chunks, acquires tokens, and dispatches broker calls outside locks."""
        # Deduplicate and group requests
        token_meta_map: Dict[str, _TokenRequestMeta] = {}
        for caller in batch_callers:
            for tok in caller.tokens:
                exch = caller.token_exchanges.get(tok, "NFO")
                if tok not in token_meta_map:
                    token_meta_map[tok] = _TokenRequestMeta(
                        token=tok,
                        exchange=exch,
                        highest_priority=caller.priority,
                        callers=[caller],
                    )
                else:
                    if caller.priority < token_meta_map[tok].highest_priority:
                        token_meta_map[tok].highest_priority = caller.priority
                    token_meta_map[tok].callers.append(caller)

        tokens_list = list(token_meta_map.keys())
        with self._lock:
            self.stats["total_tokens_coalesced"] += len(tokens_list)
            self.stats["total_batches_dispatched"] += 1

        master_quotes: Dict[str, Dict[str, Any]] = {}
        batch_exception: Optional[Exception] = None

        # Slice into chunks of max 50 tokens
        chunk_size = self.max_batch_size
        for i in range(0, len(tokens_list), chunk_size):
            chunk_tokens = tokens_list[i : i + chunk_size]

            # Priority Inheritance: chunk priority is highest (min value) in this chunk
            chunk_priority_val = min(token_meta_map[t].highest_priority for t in chunk_tokens)
            try:
                chunk_priority = PriorityTier(chunk_priority_val)
            except (ValueError, TypeError):
                chunk_priority = PriorityTier.P2_POSITION_MONITOR

            # Build exchange-segregated payload
            payload: Dict[str, List[str]] = {}
            for tok in chunk_tokens:
                exch = token_meta_map[tok].exchange or "NFO"
                payload.setdefault(exch, []).append(tok)

            # Acquire permission from PrioritizedTokenBucketLimiter outside lock
            chunk_timeout = 2.0
            if not self.limiter.acquire(EndpointCategory.QUOTES, priority=chunk_priority, timeout=chunk_timeout):
                logger.warning(f"BatchMarketDataEngine: rate limiter timeout for chunk {i // chunk_size + 1}")
                continue

            with self._lock:
                self.stats["total_broker_calls"] += 1

            # Dispatch outbound network call to broker
            try:
                resp = self.client.getMarketData("FULL", payload)
                if resp and resp.get("status") and resp.get("data"):
                    fetched = resp["data"].get("fetched", [])
                    if isinstance(fetched, list):
                        for item in fetched:
                            if not isinstance(item, dict):
                                continue
                            tok = str(item.get("symbolToken") or item.get("symboltoken") or item.get("token") or "").strip()
                            if not tok:
                                continue
                            ltp = float(item.get("ltp", 0.0) or 0.0)
                            bid = float(item.get("bestBidPrice", item.get("bid", 0.0)) or 0.0)
                            ask = float(item.get("bestAskPrice", item.get("ask", 0.0)) or 0.0)
                            spread = round(max(0.0, ask - bid), 4) if (ask > 0 and bid > 0) else 0.0
                            oi = int(item.get("opnInterest", item.get("open_interest", item.get("oi", 0))) or 0)
                            vol = int(item.get("tradeVolume", item.get("volume", 0)) or 0)
                            master_quotes[tok] = {
                                "ltp": ltp,
                                "bid": bid,
                                "ask": ask,
                                "spread": spread,
                                "open_interest": oi,
                                "volume": vol,
                                "open": float(item.get("open", 0.0) or 0.0),
                                "high": float(item.get("high", 0.0) or 0.0),
                                "low": float(item.get("low", 0.0) or 0.0),
                                "close": float(item.get("close", 0.0) or 0.0),
                                "exchange": str(item.get("exchange", "NFO") or "NFO").strip().upper(),
                                "tradingsymbol": str(item.get("tradingSymbol", item.get("tradingsymbol", "")) or "").strip(),
                                "symboltoken": tok,
                            }
                elif resp and (resp.get("errorcode") == "AB1021" or resp.get("status") is False):
                    if resp.get("errorcode") == "AB1021":
                        try:
                            self.limiter.set_cooldown(20.0, endpoint=EndpointCategory.QUOTES)
                        except Exception:
                            try:
                                self.limiter.set_cooldown(EndpointCategory.QUOTES, 20.0)
                            except Exception:
                                pass
                        logger.error("BatchMarketDataEngine: AB1021 received; set 20s cooldown on quotes")
            except Exception as ex:
                logger.warning(f"BatchMarketDataEngine: broker call exception: {ex}")
                batch_exception = ex
                with self._lock:
                    self.stats["total_exceptions"] += 1

        # Update shared micro-cache if present
        if getattr(self, "micro_cache", None) is not None and master_quotes:
            try:
                self.micro_cache.put_batch(master_quotes)
            except Exception as e:
                logger.debug(f"BatchMarketDataEngine: could not update micro_cache: {e}")

        # Demultiplex results to each caller's Future
        for caller in batch_callers:
            if caller.future.done():
                continue
            if batch_exception and not master_quotes:
                caller.future.set_exception(batch_exception)
            else:
                caller_res = {t: master_quotes[t] for t in caller.tokens if t in master_quotes}
                caller.future.set_result(caller_res)

    def _execute_direct_batch(
        self,
        caller: _BatchCallerWait,
        timeout: float,
    ) -> None:
        """Direct execution path when debounce_ms is 0 or worker stopped."""
        tokens_list = caller.tokens
        results: Dict[str, Dict[str, Any]] = {}
        chunk_size = self.max_batch_size
        batch_exception: Optional[Exception] = None

        with self._lock:
            self.stats["total_requests"] += 1
            self.stats["total_tokens_requested"] += len(tokens_list)
            self.stats["total_tokens_coalesced"] += len(tokens_list)
            self.stats["total_batches_dispatched"] += 1

        try:
            try:
                caller_priority = PriorityTier(caller.priority)
            except (ValueError, TypeError):
                caller_priority = PriorityTier.P2_POSITION_MONITOR

            for i in range(0, len(tokens_list), chunk_size):
                chunk = tokens_list[i : i + chunk_size]
                payload: Dict[str, List[str]] = {}
                for tok in chunk:
                    exch = caller.token_exchanges.get(tok, "NFO")
                    payload.setdefault(exch, []).append(tok)

                if not self.limiter.acquire(EndpointCategory.QUOTES, priority=caller_priority, timeout=timeout):
                    logger.warning(f"BatchMarketDataEngine: direct path rate limiter timeout")
                    continue

                with self._lock:
                    self.stats["total_broker_calls"] += 1

                resp = self.client.getMarketData("FULL", payload)
                if resp and resp.get("status") and resp.get("data"):
                    fetched = resp["data"].get("fetched", [])
                    if isinstance(fetched, list):
                        for item in fetched:
                            if not isinstance(item, dict):
                                continue
                            tok = str(item.get("symbolToken") or item.get("symboltoken") or item.get("token") or "").strip()
                            if not tok:
                                continue
                            ltp = float(item.get("ltp", 0.0) or 0.0)
                            bid = float(item.get("bestBidPrice", item.get("bid", 0.0)) or 0.0)
                            ask = float(item.get("bestAskPrice", item.get("ask", 0.0)) or 0.0)
                            spread = round(max(0.0, ask - bid), 4) if (ask > 0 and bid > 0) else 0.0
                            oi = int(item.get("opnInterest", item.get("open_interest", item.get("oi", 0))) or 0)
                            vol = int(item.get("tradeVolume", item.get("volume", 0)) or 0)
                            results[tok] = {
                                "ltp": ltp,
                                "bid": bid,
                                "ask": ask,
                                "spread": spread,
                                "open_interest": oi,
                                "volume": vol,
                                "open": float(item.get("open", 0.0) or 0.0),
                                "high": float(item.get("high", 0.0) or 0.0),
                                "low": float(item.get("low", 0.0) or 0.0),
                                "close": float(item.get("close", 0.0) or 0.0),
                                "exchange": str(item.get("exchange", "NFO") or "NFO").strip().upper(),
                                "tradingsymbol": str(item.get("tradingSymbol", item.get("tradingsymbol", "")) or "").strip(),
                                "symboltoken": tok,
                            }
                elif resp and resp.get("errorcode") == "AB1021":
                    try:
                        self.limiter.set_cooldown(20.0, endpoint=EndpointCategory.QUOTES)
                    except Exception:
                        try:
                            self.limiter.set_cooldown(EndpointCategory.QUOTES, 20.0)
                        except Exception:
                            pass
                    logger.error("BatchMarketDataEngine: AB1021 in direct batch; set cooldown")

            if getattr(self, "micro_cache", None) is not None and results:
                try:
                    self.micro_cache.put_batch(results)
                except Exception as e:
                    logger.debug(f"BatchMarketDataEngine: could not update micro_cache: {e}")

            caller.future.set_result(results)
        except Exception as e:
            with self._lock:
                self.stats["total_exceptions"] += 1
            caller.future.set_exception(e)


__all__ = [
    "BatchMarketDataEngine",
]
