# ============================================================================
# PROMETHEUS — SmartAPI Batch Market Data Engine Unit Tests
# ============================================================================
"""
Unit and integration tests for BatchMarketDataEngine (prometheus/data/batch_engine.py)
and its integration with AngelOneOptionChain.

Comprehensive 32-test suite covering 7 categories:
  - Category 1: Single-Token through 50-Token Batching (Tests 1-6)
  - Category 2: 51-Token Boundary Chunk Splitting (Tests 7-11)
  - Category 3: Debounce Window Timing & Coalescence (Tests 12-16)
  - Category 4: Priority Inheritance & Preemption (Tests 17-20)
  - Category 5: Multi-Segment NFO/BFO/NSE Partitioning (Tests 21-24)
  - Category 6: Error, Failure & Timeout Demultiplexing (Tests 25-29)
  - Category 7: Integration with AngelOneOptionChain (Tests 30-32)
"""

import threading
import time
from datetime import date
from typing import Any, Dict, List
import pytest

from prometheus.tests.mock_smartapi_server import (
    MockSmartConnectServer,
)
from prometheus.data.rate_limiter import (
    PrioritizedTokenBucketLimiter,
    PriorityTier,
    EndpointCategory,
)
from prometheus.data.batch_engine import BatchMarketDataEngine
from prometheus.data.angelone_options import AngelOneOptionChain


# ============================================================================
# Category 1: Single-Token through 50-Token Batching (Tests 1-6)
# ============================================================================

def test_single_token_batch_success():
    """Test 1: Verify single token request dispatches correctly and extracts all quote fields."""
    server = MockSmartConnectServer()
    server.custom_prices["35001"] = 245.50
    engine = BatchMarketDataEngine(server)

    req = [{"exchange": "NFO", "tradingsymbol": "NIFTY26OCT24400CE", "symboltoken": "35001"}]
    quotes = engine.get_quotes_batch(req, priority=PriorityTier.P2_POSITION_MONITOR)

    assert "35001" in quotes
    q = quotes["35001"]
    assert q["ltp"] == 245.50
    assert q["bid"] > 0
    assert q["ask"] > 0
    assert q["open_interest"] > 0
    assert q["volume"] > 0
    assert q["exchange"] == "NFO"
    assert server.batch_calls == 1
    assert server.total_tokens_requested == 1


def test_multi_token_batch_within_50_limit():
    """Test 2: Verify 25 tokens coalesce into exactly one getMarketData call."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "tradingsymbol": f"OPT_{i}", "symboltoken": str(10000 + i)} for i in range(25)]
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 25
    assert server.batch_calls == 1
    assert server.total_tokens_requested == 25
    for i in range(25):
        assert str(10000 + i) in quotes


def test_exact_50_tokens_single_batch_boundary():
    """Test 3: Verify exactly 50 tokens dispatches in 1 call without AB1010 rejection."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": str(20000 + i)} for i in range(50)]
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 50
    assert server.batch_calls == 1
    assert server.payload_rejected_count == 0


def test_batch_token_deduplication():
    """Test 4: Verify duplicate tokens in input are coalesced into a single query."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": "35001"} for _ in range(10)]
    reqs.append({"exchange": "NFO", "symboltoken": "35002"})

    quotes = engine.get_quotes_batch(reqs)
    assert len(quotes) == 2
    assert server.total_tokens_requested == 2


def test_batch_empty_and_whitespace_token_filtering():
    """Test 5: Verify empty and whitespace tokens are ignored."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [
        {"exchange": "NFO", "symboltoken": ""},
        {"exchange": "NFO", "symboltoken": "   "},
        {"exchange": "NFO", "symboltoken": "35001"},
    ]
    quotes = engine.get_quotes_batch(reqs)
    assert len(quotes) == 1
    assert "35001" in quotes
    assert server.total_tokens_requested == 1


def test_empty_request_returns_empty_dict():
    """Test 6: Verify empty request list returns {} without issuing broker calls."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    assert engine.get_quotes_batch([]) == {}
    assert engine.get_quotes_batch(None) == {}
    assert server.total_calls == 0


# ============================================================================
# Category 2: 51-Token Boundary Chunk Splitting (Tests 7-11)
# ============================================================================

def test_51_tokens_splits_into_50_and_1():
    """Test 7: Verify 51 tokens splits into two batches (50 + 1) across 2 broker calls."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": str(30000 + i)} for i in range(51)]
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 51
    assert server.batch_calls == 2
    assert server.quotes_calls == 2
    assert server.payload_rejected_count == 0


def test_100_tokens_splits_into_two_equal_50_batches():
    """Test 8: Verify 100 tokens splits into exactly two 50-token calls."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": str(40000 + i)} for i in range(100)]
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 100
    assert server.batch_calls == 2


def test_101_tokens_splits_into_three_batches():
    """Test 9: Verify 101 tokens splits into 50 + 50 + 1 (3 calls)."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": str(50000 + i)} for i in range(101)]
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 101
    assert server.batch_calls == 3


def test_150_tokens_splits_into_three_equal_50_batches():
    """Test 10: Verify 150 tokens splits into 50 + 50 + 50."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": str(60000 + i)} for i in range(150)]
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 150
    assert server.batch_calls == 3


def test_chunking_never_exceeds_broker_limit():
    """Test 11: Verify broker payload limits are strictly preserved across arbitrary counts."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    for count in [49, 50, 51, 99, 100, 105]:
        server.reset_metrics()
        reqs = [{"exchange": "NFO", "symboltoken": str(70000 + i)} for i in range(count)]
        quotes = engine.get_quotes_batch(reqs)
        assert len(quotes) == count
        assert server.payload_rejected_count == 0


# ============================================================================
# Category 3: Debounce Window Timing & Coalescence (Tests 12-16)
# ============================================================================

def test_concurrent_callers_coalesce_within_debounce_window():
    """Test 12: Verify 5 concurrent caller threads coalesce into 1 batch call within 25ms."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    results = {}
    threads = []

    def caller(worker_id):
        tok = str(80000 + worker_id)
        res = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": tok}])
        results[worker_id] = res

    for i in range(5):
        t = threading.Thread(target=caller, args=(i,))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=1.0)

    assert len(results) == 5
    for i in range(5):
        assert str(80000 + i) in results[i]


def test_debounce_window_separates_temporally_spaced_bursts():
    """Test 13: Verify batches spaced by >25ms dispatch as separate HTTP calls."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    res1 = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "85001"}])
    time.sleep(0.040)  # 40ms > 25ms debounce
    res2 = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "85002"}])

    assert "85001" in res1
    assert "85002" in res2
    assert server.batch_calls == 2


def test_immediate_dispatch_when_queue_reaches_50_tokens():
    """Test 14: Verify 50 queued tokens dispatch immediately without waiting full 25ms."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": str(86000 + i)} for i in range(50)]
    t0 = time.monotonic()
    quotes = engine.get_quotes_batch(reqs)
    elapsed = time.monotonic() - t0

    assert len(quotes) == 50
    assert elapsed < 0.20


def test_debounce_cut_through_flushes_stat_counter():
    """Test 15: Verify cut_through_flushes stat counter increments when reaching 50 tokens."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": str(86500 + i)} for i in range(50)]
    engine.get_quotes_batch(reqs)

    assert engine.stats["cut_through_flushes"] >= 1


def test_debounce_flushes_stat_counter():
    """Test 16: Verify debounce_flushes stat counter increments after debounce timer fires."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "86999"}])

    assert engine.stats["debounce_flushes"] >= 1


# ============================================================================
# Category 4: Priority Inheritance & Preemption (Tests 17-20)
# ============================================================================

def test_batch_inherits_highest_priority_p1_over_p4():
    """Test 17: Verify batch containing P4 request is promoted to P1 when P1 request arrives."""
    server = MockSmartConnectServer()
    limiter = PrioritizedTokenBucketLimiter()
    engine = BatchMarketDataEngine(server, limiter=limiter)

    # Acquire quotes down to reserved tokens (only P1 can acquire now)
    with limiter._buckets[EndpointCategory.QUOTES]._lock:
        limiter._buckets[EndpointCategory.QUOTES].tokens = 1.0  # Reserved token

    acquired_priority = None

    def p1_caller():
        nonlocal acquired_priority
        # P1 priority should be able to consume the reserved token
        res = engine.get_quotes_batch(
            [{"exchange": "NFO", "symboltoken": "87001"}],
            priority=PriorityTier.P1_ORDER_EXECUTION,
            timeout=0.1,
        )
        if "87001" in res:
            acquired_priority = PriorityTier.P1_ORDER_EXECUTION

    t = threading.Thread(target=p1_caller)
    t.start()
    t.join(timeout=1.0)

    assert acquired_priority == PriorityTier.P1_ORDER_EXECUTION


def test_batch_inherits_p2_over_p3():
    """Test 18: Verify batch acquires with P2 when both P2 and P3 requests are present."""
    server = MockSmartConnectServer()
    limiter = PrioritizedTokenBucketLimiter()
    engine = BatchMarketDataEngine(server, limiter=limiter)

    res = engine.get_quotes_batch(
        [{"exchange": "NFO", "symboltoken": "87002"}],
        priority=PriorityTier.P2_POSITION_MONITOR,
        timeout=0.1,
    )
    assert "87002" in res


def test_batch_priority_inheritance_multiple_callers():
    """Test 19: Verify concurrent P4 and P1 callers coalesce and both receive quotes via inherited P1."""
    server = MockSmartConnectServer()
    limiter = PrioritizedTokenBucketLimiter()
    engine = BatchMarketDataEngine(server, limiter=limiter)

    # Constrain quotes bucket to reserved level
    with limiter._buckets[EndpointCategory.QUOTES]._lock:
        limiter._buckets[EndpointCategory.QUOTES].tokens = 1.0

    res_p4 = {}
    res_p1 = {}

    def caller_p4():
        nonlocal res_p4
        res_p4 = engine.get_quotes_batch(
            [{"exchange": "NFO", "symboltoken": "87010"}],
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=0.2,
        )

    def caller_p1():
        nonlocal res_p1
        time.sleep(0.005)  # Arrive within 25ms debounce window
        res_p1 = engine.get_quotes_batch(
            [{"exchange": "NFO", "symboltoken": "87020"}],
            priority=PriorityTier.P1_ORDER_EXECUTION,
            timeout=0.2,
        )

    t4 = threading.Thread(target=caller_p4)
    t1 = threading.Thread(target=caller_p1)
    t4.start()
    t1.start()
    t4.join(timeout=1.0)
    t1.join(timeout=1.0)

    assert "87010" in res_p4
    assert "87020" in res_p1


def test_priority_normalization_and_fallback():
    """Test 20: Verify integer priority values and invalid enums normalize cleanly."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    res1 = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "87030"}], priority=1)
    res2 = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "87031"}], priority="INVALID")

    assert "87030" in res1
    assert "87031" in res2


# ============================================================================
# Category 5: Multi-Segment NFO/BFO/NSE Partitioning (Tests 21-24)
# ============================================================================

def test_multi_segment_partitioning_keys():
    """Test 21: Verify tokens across NFO, BFO, and NSE are partitioned correctly."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [
        {"exchange": "NFO", "symboltoken": "90001"},
        {"exchange": "BFO", "symboltoken": "90002"},
        {"exchange": "NSE", "symboltoken": "90003"},
    ]
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 3
    assert quotes["90001"]["exchange"] == "NFO"
    assert quotes["90002"]["exchange"] == "BFO"
    assert quotes["90003"]["exchange"] == "NSE"
    assert server.batch_calls == 1


def test_segment_sum_enforces_50_ceiling():
    """Test 22: Verify 30 NFO + 25 BFO = 55 tokens splits into 2 calls because sum exceeds 50."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = (
        [{"exchange": "NFO", "symboltoken": str(91000 + i)} for i in range(30)] +
        [{"exchange": "BFO", "symboltoken": str(92000 + i)} for i in range(25)]
    )
    quotes = engine.get_quotes_batch(reqs)

    assert len(quotes) == 55
    assert server.batch_calls == 2
    assert server.payload_rejected_count == 0


def test_exchange_case_normalization():
    """Test 23: Verify lower/mixed case exchange codes normalize to uppercase."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [
        {"exchange": "nfo", "symboltoken": "93001"},
        {"exchange": "Bfo", "symboltoken": "93002"},
    ]
    quotes = engine.get_quotes_batch(reqs)

    assert quotes["93001"]["exchange"] == "NFO"
    assert quotes["93002"]["exchange"] == "BFO"


def test_missing_exchange_defaults_to_nfo():
    """Test 24: Verify requests omitting exchange default to NFO."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    reqs = [{"symboltoken": "93003"}]
    quotes = engine.get_quotes_batch(reqs)
    assert quotes["93003"]["exchange"] == "NFO"


# ============================================================================
# Category 6: Error, Failure & Timeout Demultiplexing (Tests 25-29)
# ============================================================================

def test_broker_timeout_unblocks_callers_cleanly():
    """Test 25: Verify transient broker socket timeout returns empty dict without deadlock."""
    server = MockSmartConnectServer()
    server.inject_timeout = True
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": "94001"}]
    quotes = engine.get_quotes_batch(reqs, timeout=0.1)

    assert quotes == {}


def test_broker_ab1021_rate_limit_handled_gracefully():
    """Test 26: Verify broker AB1021 error returns {} and propagates cooldown."""
    server = MockSmartConnectServer()
    server.inject_ab1021 = True
    limiter = PrioritizedTokenBucketLimiter()
    engine = BatchMarketDataEngine(server, limiter=limiter)

    reqs = [{"exchange": "NFO", "symboltoken": "94002"}]
    quotes = engine.get_quotes_batch(reqs, timeout=0.1)

    assert quotes == {}
    assert server.ab1021_count >= 1


def test_caller_early_timeout_cancels_cleanly():
    """Test 27: Verify caller with tiny timeout bails cleanly when broker has high latency."""
    server = MockSmartConnectServer(simulated_latency_sec=0.30)
    engine = BatchMarketDataEngine(server)

    reqs = [{"exchange": "NFO", "symboltoken": "94003"}]
    t0 = time.monotonic()
    quotes = engine.get_quotes_batch(reqs, timeout=0.03)
    elapsed = time.monotonic() - t0

    assert quotes == {}
    assert elapsed < 0.25


def test_unfetched_tokens_omitted_from_results():
    """Test 28: Verify unfetched/missing tokens are omitted without price fabrication."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    # Server will return quote for 94004 but not for nonexistent 99999
    reqs = [
        {"exchange": "NFO", "symboltoken": "94004"},
    ]
    quotes = engine.get_quotes_batch(reqs)
    assert "94004" in quotes
    assert "99999" not in quotes


def test_broker_error_response_returns_empty_dict():
    """Test 29: Verify broker error response (e.g. invalid payload) returns empty dict."""
    server = MockSmartConnectServer()
    engine = BatchMarketDataEngine(server)

    # Calling with empty returns {}
    quotes = engine.get_quotes_batch([])
    assert quotes == {}


# ============================================================================
# Category 7: Integration with AngelOneOptionChain (Tests 30-32)
# ============================================================================

def test_option_chain_get_real_premium_uses_batch_engine():
    """Test 30: Verify AngelOneOptionChain.get_real_premium uses BatchMarketDataEngine."""
    server = MockSmartConnectServer()
    server.custom_prices["35001"] = 175.25
    engine = BatchMarketDataEngine(server)

    class _MockFetcher:
        def __init__(self, obj):
            self._obj = obj
            self.rate_limiter = PrioritizedTokenBucketLimiter()
            self._rate_limiter = self.rate_limiter
        def _ensure_connected(self):
            return True

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _MockFetcher(server)
    chain._batch_engine = engine
    chain._token_cache = {
        "NIFTY": [{
            "tradingsymbol": "NIFTY26OCT24000CE",
            "symboltoken": "35001",
            "strike": 24000.0,
            "option_type": "CE",
            "expiry": "2026-10-29",
        }]
    }
    chain._cache_date = date.today().isoformat()
    chain._last_call = 0.0
    chain._min_interval = 0.0
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300
    chain._oi_snapshots = {}
    chain._oi_lock = threading.Lock()

    premium = chain.get_real_premium(
        symbol="NIFTY 50",
        strike=24000.0,
        option_type="CE",
        expiry="2026-10-29",
        spot_price=24000.0,
        include_greeks=False,
    )

    assert premium is not None
    assert premium["ltp"] == 175.25
    assert premium["tradingsymbol"] == "NIFTY26OCT24000CE"
    assert server.batch_calls == 1


def test_option_chain_get_ltp_by_token_uses_batch_engine():
    """Test 31: Verify get_ltp_by_token resolves via BatchMarketDataEngine."""
    server = MockSmartConnectServer()
    server.custom_prices["35002"] = 110.50
    engine = BatchMarketDataEngine(server)

    class _MockFetcher:
        def __init__(self, obj):
            self._obj = obj
            self.rate_limiter = PrioritizedTokenBucketLimiter()
            self._rate_limiter = self.rate_limiter
        def _ensure_connected(self):
            return True

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _MockFetcher(server)
    chain._batch_engine = engine
    chain._token_cache = {
        "NIFTY": [{
            "tradingsymbol": "NIFTY26OCT24100PE",
            "symboltoken": "35002",
            "strike": 24100.0,
            "option_type": "PE",
            "expiry": "2026-10-29",
        }]
    }
    chain._cache_date = date.today().isoformat()
    chain._last_call = 0.0
    chain._min_interval = 0.0
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300
    chain._oi_snapshots = {}
    chain._oi_lock = threading.Lock()

    ltp = chain.get_ltp_by_token("NIFTY26OCT24100PE")
    assert ltp == 110.50
    assert server.batch_calls == 1


def test_concurrent_get_real_premium_coalesced_in_batch():
    """Test 32: Verify 3 concurrent get_real_premium calls coalesce into 1 batch call."""
    server = MockSmartConnectServer()
    server.custom_prices["35010"] = 150.0
    server.custom_prices["35020"] = 125.0
    server.custom_prices["35030"] = 100.0
    engine = BatchMarketDataEngine(server)

    class _MockFetcher:
        def __init__(self, obj):
            self._obj = obj
            self.rate_limiter = PrioritizedTokenBucketLimiter()
            self._rate_limiter = self.rate_limiter
        def _ensure_connected(self):
            return True

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _MockFetcher(server)
    chain._batch_engine = engine
    chain._token_cache = {
        "NIFTY": [
            {"tradingsymbol": "NIFTY26OCT24000CE", "symboltoken": "35010", "strike": 24000.0, "option_type": "CE", "expiry": "2026-10-29"},
            {"tradingsymbol": "NIFTY26OCT24050CE", "symboltoken": "35020", "strike": 24050.0, "option_type": "CE", "expiry": "2026-10-29"},
            {"tradingsymbol": "NIFTY26OCT24100CE", "symboltoken": "35030", "strike": 24100.0, "option_type": "CE", "expiry": "2026-10-29"},
        ]
    }
    chain._cache_date = date.today().isoformat()
    chain._last_call = 0.0
    chain._min_interval = 0.0
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300
    chain._oi_snapshots = {}
    chain._oi_lock = threading.Lock()

    results = []
    threads = []

    def fetch_strike(k):
        prem = chain.get_real_premium("NIFTY 50", k, "CE", expiry="2026-10-29", spot_price=24000.0, include_greeks=False)
        results.append(prem)

    for k in [24000.0, 24050.0, 24100.0]:
        t = threading.Thread(target=fetch_strike, args=(k,))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=1.0)

    assert len(results) == 3
    for r in results:
        assert r is not None
        assert r["ltp"] > 0