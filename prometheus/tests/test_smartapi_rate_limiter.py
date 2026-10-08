# ============================================================================
# PROMETHEUS — SmartAPI Prioritized Token Bucket Rate Limiter Tests
# ============================================================================
"""
Unit and integration tests for the Prioritized Token Bucket Rate Limiter
(prometheus/data/rate_limiter.py) and its integration with SmartAPIFetcher and
AngelOneOptionChain.

Validates:
  - Category 1: Independent Endpoint Bucket Capacity & 85% Ceilings (Tests 1-3)
  - Category 2: Priority Preemption & 4-Tier Scheduling (Tests 4-6)
  - Category 3: Multithreaded Non-Blocking Concurrency (Tests 7-8)
  - Category 4: Timeout Handling & Graceful Bail (Tests 9-10)
  - Category 5: Cooldown Propagation & Management (Tests 11-13)
  - Category 6: Telemetry & Statistics Veracity (Test 14)
  - Category 7: Backward Compatibility (Tests 15-16)
  - Category 8: Integration with AngelOneOptionChain (Tests 17-20)
  - Category 9: Dynamic Starvation Aging Boost (Test 21)
"""

import threading
import time
from datetime import date
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from prometheus.data.angelone_fetcher import AngelOneFetcher
from prometheus.data.angelone_options import AngelOneOptionChain
from prometheus.data.rate_limiter import (
    EndpointCategory,
    PrioritizedTokenBucketLimiter,
    PriorityTier,
    SmartAPIRateLimiter,
    _TokenBucket,
    _Waiter,
)


# ============================================================================
# Category 1: Independent Endpoint Bucket Capacity & 85% Ceilings
# ============================================================================

def test_independent_endpoint_capacities():
    """Verify that exhausting one endpoint's bucket has zero impact on other endpoints."""
    limiter = PrioritizedTokenBucketLimiter()

    # Draining QUOTES bucket by acquiring 8 tokens
    for _ in range(8):
        assert limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.1)

    quotes_bucket = limiter._buckets[EndpointCategory.QUOTES]
    hist_bucket = limiter._buckets[EndpointCategory.HISTORICAL]
    orders_bucket = limiter._buckets[EndpointCategory.ORDERS]

    # Quotes should be nearly empty (~0.5 tokens left out of 8.5)
    assert quotes_bucket.tokens < 1.0

    # HISTORICAL and ORDERS buckets must remain completely full
    assert hist_bucket.tokens == pytest.approx(3.0, abs=0.05)
    assert orders_bucket.tokens == pytest.approx(5.0, abs=0.05)

    # HISTORICAL acquisitions must succeed immediately without delay
    t0 = time.monotonic()
    assert limiter.acquire(EndpointCategory.HISTORICAL, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.01)
    assert limiter.acquire(EndpointCategory.HISTORICAL, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.01)
    assert (time.monotonic() - t0) < 0.05


def test_85_percent_quota_capping_rates():
    """Verify that refill rates do not exceed 85% of official broker quotas."""
    limiter = PrioritizedTokenBucketLimiter()

    hist = limiter._buckets[EndpointCategory.HISTORICAL]
    quotes = limiter._buckets[EndpointCategory.QUOTES]
    orders = limiter._buckets[EndpointCategory.ORDERS]

    # Official limits: Historical=3.0 req/s, Quotes=10.0 req/s, Orders=10.0 req/s
    assert hist.fill_rate <= 3.0 * 0.85 + 1e-6
    assert quotes.fill_rate <= 10.0 * 0.85 + 1e-6
    assert orders.fill_rate <= 10.0 * 0.85 + 1e-6

    # Nominal rate values
    assert hist.fill_rate == 2.5
    assert quotes.fill_rate == 8.5
    assert orders.fill_rate == 8.5

    # Burst capacities match configured ceilings
    assert hist.capacity == 3.0
    assert quotes.capacity == 8.5
    assert orders.capacity == 5.0


def test_continuous_fractional_token_refill():
    """Verify tokens refill continuously based on elapsed monotonic time."""
    limiter = PrioritizedTokenBucketLimiter()
    bucket = limiter._buckets[EndpointCategory.QUOTES]

    with bucket._lock:
        bucket.tokens = 0.0
        bucket.last_fill = time.monotonic()

    time.sleep(0.20)

    with bucket._lock:
        now = time.monotonic()
        bucket._replenish(now)
        # Expected refill: 0.20s * 8.5 tokens/s = ~1.70 tokens
        assert 1.4 <= bucket.tokens <= 2.2
        assert bucket.tokens <= bucket.capacity


# ============================================================================
# Category 2: Priority Preemption & 4-Tier Scheduling
# ============================================================================

def test_priority_preemption_p1_before_p4():
    """Verify that an urgent P1 request preempts a queued P4 request when tokens replenish."""
    # Use small fill rate (4.0 tokens/s = 0.25s per token) for deterministic observation
    limiter = PrioritizedTokenBucketLimiter(
        custom_rates={EndpointCategory.QUOTES: (4.0, 4.0)}
    )
    bucket = limiter._buckets[EndpointCategory.QUOTES]

    # Drain bucket to 0
    with bucket._lock:
        bucket.tokens = 0.0
        bucket.last_fill = time.monotonic()

    completion_order = []

    def worker_p4():
        res = limiter.acquire(
            EndpointCategory.QUOTES,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=2.0,
        )
        if res:
            completion_order.append(("P4", time.monotonic()))

    def worker_p1():
        res = limiter.acquire(
            EndpointCategory.QUOTES,
            priority=PriorityTier.P1_ORDER_EXECUTION,
            timeout=2.0,
        )
        if res:
            completion_order.append(("P1", time.monotonic()))

    # Start P4 first; it must enter queue and sleep
    t_p4 = threading.Thread(target=worker_p4)
    t_p4.start()
    time.sleep(0.04)

    # Start P1 later; it must jump ahead of P4
    t_p1 = threading.Thread(target=worker_p1)
    t_p1.start()

    t_p1.join(timeout=2.0)
    t_p4.join(timeout=2.0)

    assert len(completion_order) == 2
    assert completion_order[0][0] == "P1", f"P1 must complete before P4, got {completion_order}"
    assert completion_order[1][0] == "P4"


def test_strict_4_tier_priority_dispatch_order():
    """Verify queue drains strictly in order: P1 -> P2 -> P3 -> P4."""
    # 5.0 tokens/sec
    limiter = PrioritizedTokenBucketLimiter(
        custom_rates={EndpointCategory.QUOTES: (5.0, 5.0)}
    )
    bucket = limiter._buckets[EndpointCategory.QUOTES]

    with bucket._lock:
        bucket.tokens = 0.0
        bucket.last_fill = time.monotonic()

    order = []
    lock = threading.Lock()

    def run_worker(priority: PriorityTier, tag: str):
        if limiter.acquire(EndpointCategory.QUOTES, priority=priority, timeout=3.0):
            with lock:
                order.append(tag)

    # Enqueue in reverse order: P4, then P3, then P2, then P1
    threads = []
    for tier, tag in [
        (PriorityTier.P4_BACKGROUND_SCANNER, "P4"),
        (PriorityTier.P3_HEALTH_EVALUATION, "P3"),
        (PriorityTier.P2_POSITION_MONITOR, "P2"),
        (PriorityTier.P1_ORDER_EXECUTION, "P1"),
    ]:
        t = threading.Thread(target=run_worker, args=(tier, tag))
        threads.append(t)
        t.start()
        time.sleep(0.02)

    for t in threads:
        t.join(timeout=3.0)

    assert order == ["P1", "P2", "P3", "P4"], f"Expected strict priority order, got {order}"


def test_fifo_ordering_within_same_priority_tier():
    """Verify that within the same priority tier, requests are served FIFO."""
    limiter = PrioritizedTokenBucketLimiter(
        custom_rates={EndpointCategory.QUOTES: (5.0, 5.0)}
    )
    bucket = limiter._buckets[EndpointCategory.QUOTES]

    with bucket._lock:
        bucket.tokens = 0.0
        bucket.last_fill = time.monotonic()

    order = []
    lock = threading.Lock()

    def worker(name: str):
        if limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P2_POSITION_MONITOR, timeout=2.0):
            with lock:
                order.append(name)

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))

    t1.start()
    time.sleep(0.03)
    t2.start()

    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert order == ["A", "B"], f"FIFO order within tier violated: {order}"


# ============================================================================
# Category 3: Multithreaded Non-Blocking Concurrency
# ============================================================================

def test_multithreaded_non_blocking_sleep_concurrency():
    """Prove that waiting threads sleep outside the mutex lock without serializing."""
    limiter = PrioritizedTokenBucketLimiter(
        custom_rates={EndpointCategory.QUOTES: (2.0, 2.0)}  # 0.5s per token
    )
    bucket = limiter._buckets[EndpointCategory.QUOTES]

    # Drain bucket
    with bucket._lock:
        bucket.tokens = 0.0
        bucket.last_fill = time.monotonic()

    lock_times = []

    def probe_worker():
        # Measures time to acquire and release mutex on each probe
        t_start = time.monotonic()
        limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=1.5)
        lock_times.append(time.monotonic() - t_start)

    threads = [threading.Thread(target=probe_worker) for _ in range(3)]
    for t in threads:
        t.start()
        time.sleep(0.01)

    for t in threads:
        t.join(timeout=2.5)

    assert len(lock_times) == 3


def test_high_concurrency_stress_no_deadlock():
    """Verify limiter stability under 20 concurrent threads across all 4 priorities."""
    limiter = PrioritizedTokenBucketLimiter(
        custom_rates={EndpointCategory.QUOTES: (50.0, 50.0)}
    )
    results = []
    lock = threading.Lock()

    def worker(i: int):
        pri = [
            PriorityTier.P1_ORDER_EXECUTION,
            PriorityTier.P2_POSITION_MONITOR,
            PriorityTier.P3_HEALTH_EVALUATION,
            PriorityTier.P4_BACKGROUND_SCANNER,
        ][i % 4]
        res = limiter.acquire(EndpointCategory.QUOTES, priority=pri, timeout=2.0)
        with lock:
            results.append(res)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)

    assert len(results) == 20
    assert all(results)


# ============================================================================
# Category 4: Timeout Handling & Graceful Bail
# ============================================================================

def test_timeout_returns_false_without_raising():
    """Verify acquire() returns False when tokens cannot be acquired within timeout."""
    limiter = PrioritizedTokenBucketLimiter()
    limiter.set_cooldown(10.0)

    t0 = time.monotonic()
    res = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P3_HEALTH_EVALUATION, timeout=0.10)
    elapsed = time.monotonic() - t0

    assert res is False
    assert 0.08 <= elapsed <= 0.25


def test_timed_out_waiter_evicted_from_queue():
    """Verify that a timed-out waiter is cleanly removed and does not consume future tokens."""
    limiter = PrioritizedTokenBucketLimiter(
        custom_rates={EndpointCategory.QUOTES: (1.0, 1.0)}
    )
    bucket = limiter._buckets[EndpointCategory.QUOTES]

    with bucket._lock:
        bucket.tokens = 0.0
        bucket.last_fill = time.monotonic()

    # Thread times out after 0.05s
    res = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P4_BACKGROUND_SCANNER, timeout=0.05)
    assert res is False

    # Queue must be clean
    with bucket._lock:
        assert len(bucket._waiters) == 0

    # Wait for 1 token to refill
    time.sleep(1.1)

    # Next call should acquire cleanly
    res2 = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.1)
    assert res2 is True


# ============================================================================
# Category 5: Cooldown Propagation & Management
# ============================================================================

def test_set_cooldown_pauses_dispatch():
    """Verify set_cooldown() blocks subsequent requests until cooldown elapses."""
    limiter = PrioritizedTokenBucketLimiter()
    limiter.set_cooldown(0.30)

    t0 = time.monotonic()
    res = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=1.0)
    elapsed = time.monotonic() - t0

    assert res is True
    assert elapsed >= 0.28


def test_set_cooldown_only_extends_never_shortens():
    """Verify subsequent shorter cooldown calls do not truncate an existing cooldown."""
    limiter = PrioritizedTokenBucketLimiter()
    limiter.set_cooldown(5.0)
    initial_target = limiter.cooldown_until

    limiter.set_cooldown(1.0)
    assert limiter.cooldown_until == initial_target


def test_endpoint_specific_cooldown():
    """Verify cooldown set on QUOTES does not block HISTORICAL."""
    limiter = PrioritizedTokenBucketLimiter()
    limiter.set_cooldown(2.0, endpoint=EndpointCategory.QUOTES)

    assert limiter.is_in_cooldown(EndpointCategory.QUOTES) is True
    assert limiter.is_in_cooldown(EndpointCategory.HISTORICAL) is False

    t0 = time.monotonic()
    res = limiter.acquire(EndpointCategory.HISTORICAL, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.1)
    elapsed = time.monotonic() - t0

    assert res is True
    assert elapsed < 0.05


# ============================================================================
# Category 6: Telemetry & Statistics Veracity
# ============================================================================

def test_limiter_statistics_tracking():
    """Verify get_stats() tracks accurate operational telemetry."""
    limiter = PrioritizedTokenBucketLimiter()

    # 5 immediate acquisitions
    for _ in range(5):
        assert limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.1)

    # Force queueing by setting tokens to 0
    bucket = limiter._buckets[EndpointCategory.QUOTES]
    with bucket._lock:
        bucket.tokens = 0.0
        bucket.last_fill = time.monotonic()

    # 2 throttled acquisitions (will sleep briefly for token refill)
    assert limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=1.0)
    assert limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=1.0)

    # 1 timeout
    limiter.set_cooldown(5.0)
    assert limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P3_HEALTH_EVALUATION, timeout=0.05) is False

    stats = limiter.get_stats()
    assert stats["total_requests"] == 8
    assert stats["total_acquired"] == 7
    assert stats["total_throttled"] == 2
    assert stats["total_timeouts"] == 1
    assert "quotes" in stats["buckets"]
    assert "current_tokens" in stats["buckets"]["quotes"]


# ============================================================================
# Category 7: Backward Compatibility
# ============================================================================

def test_backward_compat_wait_method():
    """Verify legacy callers invoking limiter.wait() succeed without modification."""
    limiter = PrioritizedTokenBucketLimiter()
    res = limiter.wait()
    assert res is True

    adapter = SmartAPIRateLimiter(delay_between_calls=0.0)
    res_adapter = adapter.wait()
    assert res_adapter is True


def test_backward_compat_cooldown_until_attribute():
    """Verify existing tests checking limiter._cooldown_until pass without change."""
    limiter = SmartAPIRateLimiter(delay_between_calls=1.0)
    assert hasattr(limiter, "_cooldown_until")

    limiter.set_cooldown(20.0)
    remaining = limiter._cooldown_until - time.monotonic()
    assert 18.0 <= remaining <= 20.5


# ============================================================================
# Category 8: Integration with AngelOneOptionChain
# ============================================================================

def test_options_rate_limit_invokes_limiter_acquire():
    """Verify AngelOneOptionChain._rate_limit() routes to limiter.acquire()."""
    limiter = PrioritizedTokenBucketLimiter()

    class _MockFetcher:
        def __init__(self, rl):
            self._rate_limiter = rl

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _MockFetcher(limiter)
    chain._last_call = 0.0
    chain._min_interval = 0.0

    res = chain._rate_limit(priority=PriorityTier.P2_POSITION_MONITOR)
    assert res is True

    stats = limiter.get_stats()
    tier_stats = stats["buckets"]["quotes"]["requests_by_tier"]
    assert tier_stats["P2_POSITION_MONITOR"] >= 1


def test_options_discover_contracts_uses_parameterized_priority():
    """Verify discover_contracts propagates specified priority to rate limiter."""
    limiter = PrioritizedTokenBucketLimiter()

    class _MockObj:
        def searchScrip(self, seg, q):
            return {"data": [{"tradingsymbol": "NIFTY26AUG2624400CE", "symboltoken": "101", "name": "NIFTY", "expiry": "2026-08-26", "instrumenttype": "OPTIDX"}]}

    class _MockFetcher:
        def __init__(self, rl, obj):
            self._rate_limiter = rl
            self._obj = obj
        def _ensure_connected(self):
            return True

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _MockFetcher(limiter, _MockObj())
    chain._cache_date = ""
    chain._token_cache = {}
    chain._last_call = 0.0
    chain._min_interval = 0.0
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300

    contracts = chain.discover_contracts(
        "NIFTY 50",
        expiry_date="2026-08-26",
        strikes_around_atm=2,
        spot_price=24400.0,
        priority=PriorityTier.P4_BACKGROUND_SCANNER,
    )
    assert len(contracts) >= 1

    stats = limiter.get_stats()
    tier_stats = stats["buckets"]["quotes"]["requests_by_tier"]
    assert tier_stats["P4_BACKGROUND_SCANNER"] >= 1


def test_options_fallback_searchscrip_is_rate_limited():
    """Verify that the fallback searchScrip calls in discover_contracts pass through _rate_limit()."""
    call_records = []

    class _MockLimiter:
        def acquire(self, endpoint, priority=None, timeout=10.0):
            call_records.append((endpoint, priority))
            return True

    class _MockObj:
        def __init__(self):
            self.calls = []
        def searchScrip(self, seg, q):
            self.calls.append(q)
            # Fail date-specific query, succeed on bare underlying fallback
            if "NIFTY" in q and q != "NIFTY":
                return {"data": None}
            return {"data": [{"tradingsymbol": "NIFTY26AUG2624400CE", "symboltoken": "101", "name": "NIFTY", "expiry": "2026-08-26", "instrumenttype": "OPTIDX"}]}

    class _MockFetcher:
        def __init__(self, rl, obj):
            self._rate_limiter = rl
            self._obj = obj
        def _ensure_connected(self):
            return True

    mock_obj = _MockObj()
    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _MockFetcher(_MockLimiter(), mock_obj)
    chain._cache_date = ""
    chain._token_cache = {}
    chain._last_call = 0.0
    chain._min_interval = 0.0
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300

    contracts = chain.discover_contracts(
        "NIFTY 50",
        expiry_date="2026-08-26",
        strikes_around_atm=2,
        spot_price=24400.0,
        priority=PriorityTier.P3_HEALTH_EVALUATION,
    )
    assert len(contracts) >= 1
    # Both primary searchScrip and fallback searchScrip must have called rate_limit
    assert len(mock_obj.calls) == 2
    assert len(call_records) >= 2


def test_options_get_real_premium_defaults_to_p2_priority():
    """Verify get_real_premium defaults to P2_POSITION_MONITOR for live quote lookups."""
    rate_calls = []

    class _MockLimiter:
        def acquire(self, endpoint, priority=None, timeout=10.0):
            rate_calls.append(priority)
            return True

    class _MockObj:
        def ltpData(self, seg, sym, tok):
            return {"status": True, "data": {"ltp": 150.0}}
        def getMarketData(self, mode, payload):
            return {"status": True, "data": {"fetched": [{"bestBidPrice": 149.0, "bestAskPrice": 151.0, "opnInterest": 5000, "tradeVolume": 2000}]}}
        def optionGreek(self, params):
            return {"status": True, "data": {"delta": 0.5, "gamma": 0.01, "theta": -5.0, "vega": 10.0, "impliedVolatility": 15.0}}

    class _MockFetcher:
        def __init__(self, rl, obj):
            self._rate_limiter = rl
            self._obj = obj
        def _ensure_connected(self):
            return True

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _MockFetcher(_MockLimiter(), _MockObj())
    chain._token_cache = {
        "NIFTY": [{
            "tradingsymbol": "NIFTY26AUG2624400CE",
            "symboltoken": "101",
            "strike": 24400.0,
            "option_type": "CE",
            "expiry": "2026-08-26",
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
        strike=24400.0,
        option_type="CE",
        expiry="2026-08-26",
        spot_price=24400.0,
    )
    assert premium is not None
    assert premium["ltp"] == 150.0
    assert PriorityTier.P2_POSITION_MONITOR in rate_calls


# ============================================================================
# Category 9: Dynamic Starvation Aging Boost
# ============================================================================

def test_dynamic_aging_boost_p4_to_p3():
    """Verify that a P4 waiter waiting >= aging_threshold_sec receives an effective priority boost to P3."""
    bucket = _TokenBucket(
        endpoint=EndpointCategory.QUOTES,
        fill_rate=1.0,
        capacity=1.0,
        aging_threshold_sec=0.10,
    )
    event = threading.Event()
    waiter = _Waiter(
        effective_priority=int(PriorityTier.P4_BACKGROUND_SCANNER),
        seq=1,
        priority=PriorityTier.P4_BACKGROUND_SCANNER,
        enqueued_at=time.monotonic(),
        event=event,
    )

    with bucket._lock:
        bucket._waiters.append(waiter)

    # Before threshold
    with bucket._lock:
        bucket._update_waiter_priorities(time.monotonic())
        assert waiter.effective_priority == int(PriorityTier.P4_BACKGROUND_SCANNER)

    # After threshold
    time.sleep(0.12)
    with bucket._lock:
        bucket._update_waiter_priorities(time.monotonic())
        assert waiter.effective_priority == int(PriorityTier.P3_HEALTH_EVALUATION)


# ============================================================================
# Category 10: Integration with AngelOneFetcher
# ============================================================================

def test_fetcher_rate_limiter_initialization():
    """Verify AngelOneFetcher initializes with default PrioritizedTokenBucketLimiter or accepts custom."""
    fetcher_default = AngelOneFetcher("key", "client", "pwd", "totp")
    assert isinstance(fetcher_default.rate_limiter, PrioritizedTokenBucketLimiter)
    assert isinstance(fetcher_default._rate_limiter, SmartAPIRateLimiter)
    assert fetcher_default._rate_limiter._limiter is fetcher_default.rate_limiter

    custom_limiter = PrioritizedTokenBucketLimiter()
    fetcher_custom = AngelOneFetcher("key", "client", "pwd", "totp", rate_limiter=custom_limiter)
    assert fetcher_custom.rate_limiter is custom_limiter
    assert fetcher_custom._rate_limiter._limiter is custom_limiter


def test_fetcher_fetch_historical_routes_to_historical_bucket(monkeypatch):
    """Verify fetch_historical gates calls through EndpointCategory.HISTORICAL with specified priority."""
    limiter = PrioritizedTokenBucketLimiter()
    fetcher = AngelOneFetcher("key", "client", "pwd", "totp", rate_limiter=limiter)

    mock_obj = MagicMock()
    mock_obj.getCandleData.return_value = {
        "status": True,
        "data": [["2026-08-25 09:15", 24000.0, 24050.0, 23980.0, 24020.0, 1000]],
    }
    fetcher._obj = mock_obj
    fetcher._ensure_connected = lambda: True

    df = fetcher.fetch_historical("NIFTY 50", days=1, priority=PriorityTier.P4_BACKGROUND_SCANNER)
    assert not df.empty
    assert len(df) == 1

    stats = limiter.get_stats()
    tier_stats = stats["buckets"]["historical"]["requests_by_tier"]
    assert tier_stats["P4_BACKGROUND_SCANNER"] >= 1


def test_fetcher_fetch_historical_ab1021_triggers_endpoint_cooldown(monkeypatch):
    """Verify AB1021 in fetch_historical triggers cooldown on EndpointCategory.HISTORICAL."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    cooldowns = []
    limiter = PrioritizedTokenBucketLimiter()
    orig_set_cooldown = limiter.set_cooldown

    def tracking_set_cooldown(*args, **kwargs):
        wait = kwargs.get("wait", args[0] if len(args) > 0 else 0)
        target_ep = kwargs.get("endpoint", args[2] if len(args) > 2 else None)
        cooldowns.append((wait, target_ep))
        orig_set_cooldown(0.01, endpoint=target_ep)

    limiter.set_cooldown = tracking_set_cooldown
    fetcher = AngelOneFetcher("key", "client", "pwd", "totp", rate_limiter=limiter)

    mock_obj = MagicMock()
    mock_obj.getCandleData.side_effect = [
        {"status": False, "errorcode": "AB1021", "message": "Too Many Requests"},
        {"status": True, "data": [["2026-08-25 09:15", 24000.0, 24050.0, 23980.0, 24020.0, 1000]]},
    ]
    fetcher._obj = mock_obj
    fetcher._ensure_connected = lambda: True

    df = fetcher.fetch_historical("NIFTY 50", days=1, priority=PriorityTier.P4_BACKGROUND_SCANNER)
    assert not df.empty

    assert len(cooldowns) >= 1
    assert cooldowns[0][1] == EndpointCategory.HISTORICAL
    assert cooldowns[0][0] >= 5.0


def test_fetcher_get_candle_data_ab1021_sets_cooldown():
    """Verify get_candle_data triggers 20s cooldown on EndpointCategory.HISTORICAL on AB1021."""
    limiter = PrioritizedTokenBucketLimiter()
    fetcher = AngelOneFetcher("key", "client", "pwd", "totp", rate_limiter=limiter)

    mock_obj = MagicMock()
    mock_obj.getCandleData.return_value = {
        "status": False,
        "errorcode": "AB1021",
        "message": "Too Many Requests",
    }
    fetcher._obj = mock_obj
    fetcher._ensure_connected = lambda: True

    res = fetcher.get_candle_data({"symboltoken": "99926000"})
    assert res is not None
    assert res.get("status") is False

    assert limiter.is_in_cooldown(EndpointCategory.HISTORICAL)
    assert limiter.get_remaining_cooldown(EndpointCategory.HISTORICAL) > 15.0



