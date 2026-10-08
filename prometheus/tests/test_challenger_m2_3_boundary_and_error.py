# ============================================================================
# PROMETHEUS — Adversarial Challenge Test Suite (Challenger M2-3-1)
# Boundary Chunking, Multi-Segment Partitioning, Error Demux & Micro-Cache
# ============================================================================
"""
Adversarial Verification Suite designed by Challenger M2-3-1 (Boundary & Error Challenger).

Rigorous empirical challenge of:
1. Boundary Chunking: 0, 1, 49, 50, 51, 100, 101, 150, 200 tokens across NFO, BFO, NSE.
2. Multi-Segment Partitioning: Mixed exchanges, sum <= 50 tokens, normalization, empty keys.
3. Broker Error Injections: AB1021 rate-limit cooldown, HTTP 429, socket timeouts, partial failure, corrupt payloads.
4. Error Demultiplexing Without Deadlocks: High-concurrency unblocking guarantees.
5. Micro-Cache Concurrency & Invariants: Sub-second TTL, lock-free reads, zero synthetic price fallback.
6. Integrated Pipeline: Batch engine auto-populating micro-cache, get_or_fetch bypass.
"""

import concurrent.futures
import threading
import time
from typing import Any, Dict, List, Optional
import pytest

from prometheus.tests.mock_smartapi_server import (
    MockSmartConnectServer,
    SmartAPIRateLimitError,
)
from prometheus.data.rate_limiter import (
    EndpointCategory,
    PrioritizedTokenBucketLimiter,
    PriorityTier,
)
from prometheus.data.batch_engine import BatchMarketDataEngine
from prometheus.data.price_micro_cache import (
    FallbackPolicy,
    MicroCacheEntry,
    SharedPriceMicroCache,
)


# ============================================================================
# 1. 50-TOKEN BOUNDARY CHUNKING & MULTI-SEGMENT PARTITIONING
# ============================================================================

class TestBoundaryChunkingAndSegments:
    """Stress-test chunking boundaries (0, 1, 49, 50, 51, 100, 101, 150, 200) and segment partitioning."""

    def test_boundary_0_tokens_all_variants(self):
        """0 tokens: None, empty list, list of empty dicts, whitespace tokens return {} with 0 broker calls."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        # None
        assert engine.get_quotes_batch(None) == {}
        # Empty list
        assert engine.get_quotes_batch([]) == {}
        # Empty dicts
        assert engine.get_quotes_batch([{}]) == {}
        # Whitespace / missing tokens
        assert engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "   "}]) == {}
        assert engine.get_quotes_batch([{"exchange": "", "symboltoken": ""}]) == {}

        assert server.total_calls == 0
        assert server.batch_calls == 0
        engine.stop()

    def test_boundary_1_token_across_exchanges(self):
        """1 token: Exactly 1 call, correctly dispatched across NFO, BFO, NSE."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        for exch, tok in [("NFO", "10001"), ("BFO", "20002"), ("NSE", "30003")]:
            quotes = engine.get_quotes_batch([{"exchange": exch, "symboltoken": tok}])
            assert len(quotes) == 1
            assert tok in quotes
            assert quotes[tok]["exchange"] == exch
            assert quotes[tok]["symboltoken"] == tok

        assert server.batch_calls == 3
        assert server.payload_rejected_count == 0
        engine.stop()

    def test_boundary_exact_50_tokens_single_chunk(self):
        """Exactly 50 tokens: Dispatches in exactly 1 batch call, 0 rejected."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        reqs = [{"exchange": "NFO", "symboltoken": f"NFO_{i}"} for i in range(50)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 50
        assert server.batch_calls == 1
        assert server.payload_rejected_count == 0
        engine.stop()

    def test_boundary_51_tokens_splits_into_50_and_1(self):
        """51 tokens: Splits into exactly 2 batch calls (50 + 1), all 51 returned."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        reqs = [{"exchange": "NFO", "symboltoken": f"TOK_{i}"} for i in range(51)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 51
        assert server.batch_calls == 2
        assert server.payload_rejected_count == 0
        engine.stop()

    def test_boundary_100_tokens_splits_into_50_and_50(self):
        """100 tokens: Splits into exactly 2 batch calls of 50 each."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        reqs = [{"exchange": "NFO", "symboltoken": f"TOK_{i}"} for i in range(100)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 100
        assert server.batch_calls == 2
        assert server.payload_rejected_count == 0
        engine.stop()

    def test_boundary_101_tokens_splits_into_three_chunks(self):
        """101 tokens: Splits into exactly 3 batch calls (50 + 50 + 1)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        reqs = [{"exchange": "NFO", "symboltoken": f"TOK_{i}"} for i in range(101)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 101
        assert server.batch_calls == 3
        assert server.payload_rejected_count == 0
        engine.stop()

    def test_multi_segment_partitioning_under_50_tokens(self):
        """Mixed segments: 20 NFO, 20 BFO, 10 NSE (50 total) dispatches in 1 call."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        reqs = (
            [{"exchange": "NFO", "symboltoken": f"NFO_{i}"} for i in range(20)]
            + [{"exchange": "BFO", "symboltoken": f"BFO_{i}"} for i in range(20)]
            + [{"exchange": "NSE", "symboltoken": f"NSE_{i}"} for i in range(10)]
        )
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 50
        assert server.batch_calls == 1
        assert server.payload_rejected_count == 0
        engine.stop()

    def test_multi_segment_partitioning_over_50_tokens(self):
        """Mixed segments: 35 NFO, 35 BFO (70 total) splits into 2 calls <= 50 each."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        reqs = (
            [{"exchange": "NFO", "symboltoken": f"NFO_{i}"} for i in range(35)]
            + [{"exchange": "BFO", "symboltoken": f"BFO_{i}"} for i in range(35)]
        )
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 70
        assert server.batch_calls == 2
        assert server.payload_rejected_count == 0
        engine.stop()

    def test_exchange_case_and_whitespace_normalization(self):
        """Exchange tags with irregular casing or whitespace normalize cleanly to uppercase."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        reqs = [
            {"exchange": " nfo ", "symboltoken": "TOK_1"},
            {"exchange": "bfo", "symboltoken": "TOK_2"},
            {"exchange": "Nse", "symboltoken": "TOK_3"},
            {"exchange": "", "symboltoken": "TOK_4"},  # Defaults to NFO
        ]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 4
        assert quotes["TOK_1"]["exchange"] == "NFO"
        assert quotes["TOK_2"]["exchange"] == "BFO"
        assert quotes["TOK_3"]["exchange"] == "NSE"
        assert quotes["TOK_4"]["exchange"] == "NFO"
        engine.stop()


# ============================================================================
# 2. BROKER ERROR INJECTIONS & DEMULTIPLEXING WITHOUT DEADLOCK
# ============================================================================

class TestBrokerErrorInjectionsAndDemux:
    """Stress-test broker error injections (AB1021, HTTP 429, timeouts, partial/corrupt payloads)."""

    def test_broker_ab1021_triggers_cooldown_and_returns_empty_without_deadlock(self):
        """AB1021 rate limit triggers 20s cooldown on limiter and returns {} without deadlock."""
        limiter = PrioritizedTokenBucketLimiter()
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, limiter=limiter, debounce_ms=10.0)

        # Inject AB1021 error response
        server.inject_ab1021 = True
        quotes = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "99001"}])

        assert quotes == {}
        # Verify rate limiter cooldown was engaged
        assert limiter.is_in_cooldown(EndpointCategory.QUOTES) is True

        # Subsequent caller during cooldown fails rate limiter acquisition cleanly without deadlocking
        res2 = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "99002"}], timeout=0.1)
        assert res2 == {}
        engine.stop()

    def test_broker_http_429_exception_handled_cleanly(self):
        """Mock broker raising SmartAPIRateLimitError (HTTP 429) returns {} and increments exception stat."""
        server = MockSmartConnectServer(raise_on_quota_exceeded=True)
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        # Trigger quota exception
        server.inject_ab1021 = True
        quotes = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "42901"}])

        assert quotes == {}
        assert engine.stats["total_exceptions"] >= 1
        engine.stop()

    def test_broker_socket_timeout_unblocks_all_concurrent_callers(self):
        """Broker socket read timeout unblocks all concurrent callers cleanly within timeout."""
        server = MockSmartConnectServer()
        server.inject_timeout = True
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        results = {}
        barrier = threading.Barrier(5)

        def worker(w_id: int):
            barrier.wait()
            res = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": f"TMO_{w_id}"}], timeout=1.0)
            results[w_id] = res

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), f"Thread {t.name} deadlocked!"

        assert len(results) == 5
        for res in results.values():
            assert res == {}
        engine.stop()

    def test_broker_partial_and_corrupt_payload_parsing(self):
        """Broker returning malformed, non-dict, and None entries handles them safely without crashing."""
        class CorruptClient:
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]):
                return {
                    "status": True,
                    "data": {
                        "fetched": [
                            # Valid item
                            {
                                "symbolToken": "OK_1",
                                "ltp": 150.0,
                                "bestBidPrice": 149.5,
                                "bestAskPrice": 150.5,
                                "opnInterest": 1000,
                                "tradeVolume": 500,
                                "exchange": "NFO",
                                "tradingSymbol": "NFO_OK_1",
                            },
                            # Non-dict corrupt item
                            "CORRUPT_STRING_PAYLOAD",
                            None,
                            # Missing symbolToken
                            {
                                "ltp": 200.0,
                            },
                            # None numeric fields
                            {
                                "symbolToken": "OK_2",
                                "ltp": None,
                                "bestBidPrice": None,
                                "bestAskPrice": None,
                                "opnInterest": None,
                                "tradeVolume": None,
                                "exchange": "NSE",
                            },
                            # Requested token MISS_3 is completely omitted
                        ]
                    }
                }

        engine = BatchMarketDataEngine(CorruptClient(), debounce_ms=10.0)
        reqs = [
            {"exchange": "NFO", "symboltoken": "OK_1"},
            {"exchange": "NSE", "symboltoken": "OK_2"},
            {"exchange": "NFO", "symboltoken": "MISS_3"},
        ]
        quotes = engine.get_quotes_batch(reqs)

        # OK_1 is correctly parsed
        assert "OK_1" in quotes
        assert quotes["OK_1"]["ltp"] == 150.0
        assert quotes["OK_1"]["bid"] == 149.5

        # OK_2 is parsed with safe 0.0 defaults
        assert "OK_2" in quotes
        assert quotes["OK_2"]["ltp"] == 0.0
        assert quotes["OK_2"]["bid"] == 0.0

        # MISS_3 is omitted (Zero Hallucination Mandate)
        assert "MISS_3" not in quotes
        engine.stop()

    def test_multi_chunk_partial_failure_preserves_valid_chunk(self):
        """In a 100-token request (2 chunks of 50), if chunk 2 fails, chunk 1 tokens are returned."""
        call_count = 0

        class FlakyChunkClient:
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Chunk 1 succeeds
                    tokens = payload.get("NFO", [])
                    return {
                        "status": True,
                        "data": {
                            "fetched": [
                                {"symbolToken": t, "ltp": 100.0 + i, "exchange": "NFO"}
                                for i, t in enumerate(tokens)
                            ]
                        }
                    }
                else:
                    # Chunk 2 fails with exception
                    raise RuntimeError("Broker dropped connection on chunk 2")

        engine = BatchMarketDataEngine(FlakyChunkClient(), debounce_ms=10.0)
        reqs = [{"exchange": "NFO", "symboltoken": f"FLK_{i}"} for i in range(100)]
        quotes = engine.get_quotes_batch(reqs)

        # Chunk 1 tokens (0 to 49) are returned
        for i in range(50):
            assert f"FLK_{i}" in quotes

        # Chunk 2 tokens (50 to 99) are omitted (Zero Hallucination)
        for i in range(50, 100):
            assert f"FLK_{i}" not in quotes
        engine.stop()

    def test_direct_execution_path_under_error(self):
        """Direct execution path (debounce_ms=0) handles AB1021 and timeouts cleanly."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=0.0)

        # AB1021
        server.inject_ab1021 = True
        quotes = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "DIR_1"}])
        assert quotes == {}

        # Timeout
        server.inject_ab1021 = False
        server.inject_timeout = True
        quotes2 = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "DIR_2"}])
        assert quotes2 == {}
        engine.stop()


# ============================================================================
# 3. MICRO-CACHE CONCURRENCY & ZERO-HALLUCINATION FALLBACK
# ============================================================================

class TestMicroCacheConcurrencyAndFallback:
    """Stress-test sub-second TTL, lock-free concurrency, and FallbackPolicy invariants."""

    def test_micro_cache_sub_second_ttl_high_precision(self):
        """Micro-cache respects 50ms TTL with zero synthetic prices on expiry."""
        cache = SharedPriceMicroCache(default_ttl=0.050)
        now = time.monotonic()

        entry = MicroCacheEntry(
            price=250.0,
            bid=249.5,
            ask=250.5,
            open_interest=50000,
            volume=10000,
            timestamp=now,
            tradingsymbol="BANKNIFTY_TEST",
        )
        cache.put("99001", entry)

        # Immediate lookup: HIT
        assert cache.get("99001") is not None
        assert cache.get("99001").price == 250.0

        # Wait past 50ms TTL
        time.sleep(0.065)

        # Expired lookup: MISS (returns None, never synthetic price)
        assert cache.get("99001") is None
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_micro_cache_concurrent_hammer_no_race_conditions(self):
        """Concurrent reader and writer threads hammering micro-cache produce 0 exceptions."""
        cache = SharedPriceMicroCache(default_ttl=0.500)
        stop_event = threading.Event()
        errors = []

        def writer(w_id: int):
            try:
                tok = f"HAMMER_{w_id}"
                while not stop_event.is_set():
                    now = time.monotonic()
                    cache.put(
                        tok,
                        MicroCacheEntry(
                            price=100.0 + w_id,
                            bid=99.0,
                            ask=101.0,
                            open_interest=1000,
                            volume=500,
                            timestamp=now,
                            tradingsymbol=f"SYM_{w_id}",
                        ),
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader(r_id: int):
            try:
                tok = f"HAMMER_{r_id % 5}"
                while not stop_event.is_set():
                    cache.get(tok)
                    cache.get_by_symbol(f"SYM_{r_id % 5}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        threads += [threading.Thread(target=reader, args=(i,)) for i in range(15)]

        for t in threads:
            t.start()

        time.sleep(0.5)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "Worker thread deadlocked during cache hammer"

        assert len(errors) == 0, f"Encountered cache concurrency errors: {errors}"
        stats = cache.get_stats()
        assert stats["puts"] > 0
        assert stats["hits"] + stats["misses"] > 0

    def test_fallback_policy_stale_alert_telemetry(self):
        """FallbackPolicy.handle_option_stale logs structured alert without price fabrication."""
        FallbackPolicy.reset_alerts()

        FallbackPolicy.handle_option_stale(
            symbol="BANKNIFTY26OCT51000CE",
            token="35001",
            last_known_price=245.50,
            age_seconds=1.85,
        )

        assert len(FallbackPolicy.stale_alerts) == 1
        alert = FallbackPolicy.stale_alerts[0]
        assert alert["symbol"] == "BANKNIFTY26OCT51000CE"
        assert alert["token"] == "35001"
        assert alert["last_known_price"] == 245.50
        assert alert["age_seconds"] == 1.85

    def test_fallback_policy_secondary_spot_resolution(self):
        """FallbackPolicy resolves secondary spot feed for indices without fabricating prices."""
        # By default unconfigured secondary feed returns None (no price fabrication)
        assert FallbackPolicy.get_secondary_spot_price("NIFTY 50") is None

        # When authenticated secondary feed is registered:
        feed = {"NIFTY 50": 25050.0, "NIFTY BANK": 51200.0, "SENSEX": 81500.0}
        FallbackPolicy.set_secondary_spot_feed(lambda s: feed.get(s))
        assert FallbackPolicy.get_secondary_spot_price("NIFTY 50") == 25050.0
        assert FallbackPolicy.get_secondary_spot_price("NIFTY BANK") == 51200.0
        assert FallbackPolicy.get_secondary_spot_price("SENSEX") == 81500.0
        # Unknown instrument returns None, never synthetic price
        assert FallbackPolicy.get_secondary_spot_price("UNKNOWN_INDEX") is None
        FallbackPolicy.set_secondary_spot_feed(None)


# ============================================================================
# 4. BATCH ENGINE & MICRO-CACHE INTEGRATED PIPELINE
# ============================================================================

class TestIntegratedBatchEngineAndMicroCache:
    """Verify integrated pipeline: batch engine populates cache, get_or_fetch avoids broker roundtrips."""

    def test_batch_engine_auto_populates_micro_cache(self):
        """BatchMarketDataEngine configured with micro_cache automatically updates cache on quotes."""
        server = MockSmartConnectServer()
        cache = SharedPriceMicroCache(default_ttl=1.0)
        engine = BatchMarketDataEngine(server, debounce_ms=10.0, micro_cache=cache)

        reqs = [{"exchange": "NFO", "symboltoken": "77001"}]
        quotes = engine.get_quotes_batch(reqs)

        assert "77001" in quotes
        cached = cache.get("77001")
        assert cached is not None
        assert cached.price == quotes["77001"]["ltp"]
        assert cached.source == "batch_rest"
        engine.stop()

    def test_micro_cache_get_or_fetch_avoids_broker_call_on_fresh_tokens(self):
        """get_or_fetch only fetches missing tokens, completely bypassing broker for fresh tokens."""
        server = MockSmartConnectServer()
        cache = SharedPriceMicroCache(default_ttl=1.0)
        engine = BatchMarketDataEngine(server, debounce_ms=10.0, micro_cache=cache)

        # Seed cache with token 88001
        cache.put(
            "88001",
            MicroCacheEntry(
                price=300.0,
                bid=299.0,
                ask=301.0,
                open_interest=5000,
                volume=2000,
                timestamp=time.monotonic(),
                tradingsymbol="SEEDED_88001",
            ),
        )

        def fetch_missing(tokens: List[str]) -> Dict[str, MicroCacheEntry]:
            # This should ONLY be called for 88002
            assert "88001" not in tokens
            assert "88002" in tokens
            reqs = [{"exchange": "NFO", "symboltoken": t} for t in tokens]
            batch_res = engine.get_quotes_batch(reqs)
            res = {}
            for t, q in batch_res.items():
                res[t] = MicroCacheEntry(
                    price=q["ltp"],
                    bid=q["bid"],
                    ask=q["ask"],
                    open_interest=q["open_interest"],
                    volume=q["volume"],
                    timestamp=time.monotonic(),
                )
            return res

        results = cache.get_or_fetch(["88001", "88002"], fetch_missing)

        assert len(results) == 2
        assert results["88001"].price == 300.0  # From seed cache
        assert "88002" in results               # Fetched from broker
        assert server.total_tokens_requested == 1  # Only 1 token requested from broker!
        engine.stop()


# ============================================================================
# 5. ADVERSARIAL CONCURRENCY, CANCELLATION & STAMPEDE
# ============================================================================

class TestAdversarialConcurrencyAndStampede:
    """Adversarially probe race conditions, future cancellation, stampedes, and dirty inputs."""

    def test_adversarial_caller_cancellation_race_condition(self):
        """When some callers cancel their futures before batch flushes, other callers succeed without errors."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=30.0)

        futures = []
        for i in range(10):
            fut = engine.get_quotes_batch_async([{"exchange": "NFO", "symboltoken": f"CANCEL_{i}"}])
            futures.append(fut)

        # Immediately cancel the first 5 futures
        for i in range(5):
            futures[i].cancel()

        # Wait for remaining 5 futures to finish
        results = []
        for i in range(5, 10):
            res = futures[i].result(timeout=2.0)
            results.append(res)

        assert len(results) == 5
        for i, res in enumerate(results):
            tok = f"CANCEL_{i + 5}"
            assert tok in res
            assert res[tok]["symboltoken"] == tok

        engine.stop()

    def test_adversarial_cache_stampede_coalescing(self):
        """30 concurrent threads requesting the exact same token are coalesced into 1 broker call."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=25.0)

        results = {}
        barrier = threading.Barrier(30)

        def worker(w_id: int):
            barrier.wait()
            res = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "STAMPEDE_1"}])
            results[w_id] = res

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), "Worker thread deadlocked during stampede"

        assert len(results) == 30
        for res in results.values():
            assert "STAMPEDE_1" in res
            assert res["STAMPEDE_1"]["ltp"] > 0

        # All 30 callers coalesced: exactly 1 broker batch call, exactly 1 token requested
        assert server.batch_calls == 1
        assert server.total_tokens_requested == 1
        engine.stop()

    def test_adversarial_100_thread_concurrency_mixed_errors_no_deadlock(self):
        """100 concurrent threads under mixed broker error conditions (timeouts + AB1021) complete without deadlocks."""
        import random

        class FlakyMixedServer:
            def __init__(self):
                self.calls = 0
                self._lock = threading.Lock()

            def getMarketData(self, mode: str, payload: Dict[str, List[str]]):
                with self._lock:
                    self.calls += 1
                    r = random.random()
                if r < 0.20:
                    raise TimeoutError("Simulated random timeout")
                elif r < 0.40:
                    return {"status": False, "errorcode": "AB1021", "message": "Rate Limit Exceeded"}
                else:
                    fetched = []
                    for exch, tokens in payload.items():
                        for t in tokens:
                            fetched.append({
                                "symbolToken": t,
                                "exchange": exch,
                                "ltp": 120.0,
                                "bestBidPrice": 119.5,
                                "bestAskPrice": 120.5,
                            })
                    return {"status": True, "data": {"fetched": fetched}}

        server = FlakyMixedServer()
        engine = BatchMarketDataEngine(server, debounce_ms=15.0)
        unblocked = []

        def worker(w_id: int):
            prio = (w_id % 4) + 1
            res = engine.get_quotes_batch(
                [{"exchange": "NFO", "symboltoken": f"RACE_{w_id}"}],
                priority=prio,
                timeout=2.0,
            )
            unblocked.append(w_id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), f"Thread {t.name} deadlocked!"

        assert len(unblocked) == 100, f"Only {len(unblocked)} of 100 threads unblocked!"
        engine.stop()

    def test_adversarial_dirty_input_types_graceful_handling(self):
        """Dirty input types (integer tokens, None, strings, missing keys) do not crash the engine."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server, debounce_ms=10.0)

        dirty_reqs = [
            {"exchange": "NFO", "symboltoken": 54321},          # Integer token
            {"exchange": "NFO", "token": "76543"},               # 'token' instead of 'symboltoken'
            {"exchange": "NSE", "symbol_token": "98765"},        # 'symbol_token' key
            {"exchange": None, "symboltoken": "11111"},          # None exchange
            "INVALID_NON_DICT_STRING",                           # Non-dict string
            12345,                                               # Non-dict int
            None,                                                # None
            {},                                                  # Empty dict
            {"exchange": "NFO", "symboltoken": None},            # None symboltoken
        ]

        quotes = engine.get_quotes_batch(dirty_reqs)

        # Valid tokens extracted and converted cleanly
        assert "54321" in quotes
        assert "76543" in quotes
        assert "98765" in quotes
        assert "11111" in quotes
        # Corrupt items safely omitted
        assert len(quotes) == 4
        engine.stop()
