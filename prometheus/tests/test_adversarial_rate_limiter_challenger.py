"""
Adversarial Stress Test Suite for PrioritizedTokenBucketLimiter.
Author: Challenger M1-1
Mission: Adversarial verification of:
  1. High-concurrency thread flood (50+ threads simultaneously acquiring across all 3 buckets).
  2. Priority preemption under heavy load (P1 latency <50ms while P4 queue is saturated).
  3. Sliding window boundary enforcement (no single 1.0-second window exceeds broker limits).
"""

import time
import threading
from typing import List, Tuple
import pytest

from prometheus.data.rate_limiter import (
    PrioritizedTokenBucketLimiter,
    PriorityTier,
    EndpointCategory,
)


class TestAdversarialRateLimiterChallenger:
    """Adversarial stress testing suite directly challenging PrioritizedTokenBucketLimiter."""

    def test_high_concurrency_thread_flood_all_buckets(self):
        """Stress vector 1: Flood 60 concurrent threads across all 3 buckets simultaneously.

        - 20 threads on HISTORICAL
        - 20 threads on QUOTES
        - 20 threads on ORDERS
        - Mix of priorities P1, P2, P3, P4
        - Asserts: No deadlocks, 100% token accounting, non-negative tokens, clean teardown.
        """
        limiter = PrioritizedTokenBucketLimiter()
        num_threads = 60
        results: List[Tuple[bool, float, EndpointCategory, PriorityTier]] = []
        lock = threading.Lock()

        endpoints = [
            EndpointCategory.HISTORICAL,
            EndpointCategory.QUOTES,
            EndpointCategory.ORDERS,
        ]
        priorities = [
            PriorityTier.P1_ORDER_EXECUTION,
            PriorityTier.P2_POSITION_MONITOR,
            PriorityTier.P3_HEALTH_EVALUATION,
            PriorityTier.P4_BACKGROUND_SCANNER,
        ]

        start_gate = threading.Event()

        def worker_task(thread_id: int):
            ep = endpoints[thread_id % len(endpoints)]
            pri = priorities[thread_id % len(priorities)]
            start_gate.wait(timeout=5.0)

            t0 = time.perf_counter()
            # Use 3.0s timeout to allow queued threads to process under contention
            acquired = limiter.acquire(endpoint=ep, priority=pri, timeout=3.0)
            elapsed = time.perf_counter() - t0

            with lock:
                results.append((acquired, elapsed, ep, pri))

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()

        # Trigger simultaneous start
        start_gate.set()

        # Join all threads with strict timeout to detect deadlocks
        for t in threads:
            t.join(timeout=6.0)
            assert not t.is_alive(), "Deadlock detected: thread failed to finish within timeout"

        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"

        # Validate token accounting across buckets
        total_acquired = sum(1 for r in results if r[0])
        total_timeouts = sum(1 for r in results if not r[0])

        assert total_acquired > 0, "No threads acquired tokens during flood"
        stats_acquired = sum(b.stats["total_acquired"] for b in limiter._buckets.values())
        stats_timeouts = sum(b.stats["total_timeouts"] for b in limiter._buckets.values())

        assert stats_acquired == total_acquired, (
            f"Stats total_acquired ({stats_acquired}) does not match thread results ({total_acquired})"
        )
        assert stats_timeouts == total_timeouts, (
            f"Stats total_timeouts ({stats_timeouts}) does not match thread results ({total_timeouts})"
        )

        for cat, bucket in limiter._buckets.items():
            assert bucket.tokens >= 0.0, f"Bucket {cat.value} tokens dropped below 0: {bucket.tokens}"

    def test_priority_preemption_under_heavy_p4_queue_load(self):
        """Stress vector 2: Saturate queue with P4 requests, then verify P1 latency is <50ms.

        Adversarial Scenario:
        1. Drain initial quote tokens down to reservation limit.
        2. Flood the Quotes queue with 20 P4 threads blocked waiting for tokens.
        3. Dispatch P1 request (order execution / trailing SL trigger).
        4. Measure P1 latency: assert P1 preempts P4 queue and completes in <50ms.
        """
        limiter = PrioritizedTokenBucketLimiter()
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        # Drain fast-path tokens down to reserve level (2.0 tokens)
        while bucket.tokens > bucket.reserve_tokens:
            acquired = limiter.acquire(
                endpoint=EndpointCategory.QUOTES,
                priority=PriorityTier.P1_ORDER_EXECUTION,
                timeout=0.01,
            )
            if not acquired:
                break

        # Squeezed to reserve level. Now any P4 request MUST queue up.
        p4_started_gate = threading.Barrier(21)  # 20 workers + main thread
        p4_hold_event = threading.Event()
        p4_results = []
        p4_lock = threading.Lock()

        def p4_worker():
            p4_started_gate.wait(timeout=5.0)
            # P4 requests wait in queue
            acq = limiter.acquire(
                endpoint=EndpointCategory.QUOTES,
                priority=PriorityTier.P4_BACKGROUND_SCANNER,
                timeout=2.0,
            )
            with p4_lock:
                p4_results.append(acq)

        p4_threads = [threading.Thread(target=p4_worker) for _ in range(20)]
        for t in p4_threads:
            t.start()

        # Release barrier so all 20 P4 threads enter limiter.acquire simultaneously
        p4_started_gate.wait(timeout=5.0)

        # Allow threads to enqueue into _waiters
        time.sleep(0.05)

        with bucket._lock:
            p4_waiting_count = len(bucket._waiters)

        assert p4_waiting_count >= 10, f"Expected at least 10 P4 waiters in queue, found {p4_waiting_count}"

        # Now issue P1 request under heavy P4 saturation
        t_p1_start = time.perf_counter()
        p1_acquired = limiter.acquire(
            endpoint=EndpointCategory.QUOTES,
            priority=PriorityTier.P1_ORDER_EXECUTION,
            timeout=1.0,
        )
        p1_latency_ms = (time.perf_counter() - t_p1_start) * 1000.0

        assert p1_acquired is True, "P1 request failed to acquire token under P4 heavy load"
        assert p1_latency_ms < 50.0, f"P1 latency {p1_latency_ms:.2f}ms exceeded 50ms requirement"

        # Teardown: let P4 threads finish
        for t in p4_threads:
            t.join(timeout=3.0)

    def test_priority_preemption_multiple_trials(self):
        """Verify P1 preemption consistently beats 50ms requirement over 5 independent trials."""
        latencies = []
        for trial in range(5):
            limiter = PrioritizedTokenBucketLimiter()
            bucket = limiter._buckets[EndpointCategory.QUOTES]

            # Drain to reservation level
            with bucket._lock:
                bucket.tokens = 1.5  # Below P4 requirement (3.0), above P1 requirement (1.0)
                bucket._dispatch_history.clear()

            # Enqueue 15 P4 requests
            barrier = threading.Barrier(16)
            def p4_worker():
                barrier.wait(timeout=2.0)
                limiter.acquire(
                    endpoint=EndpointCategory.QUOTES,
                    priority=PriorityTier.P4_BACKGROUND_SCANNER,
                    timeout=1.0,
                )

            threads = [threading.Thread(target=p4_worker) for _ in range(15)]
            for t in threads:
                t.start()

            barrier.wait(timeout=2.0)
            time.sleep(0.02)  # Ensure enqueued

            t0 = time.perf_counter()
            p1_ok = limiter.acquire(
                endpoint=EndpointCategory.QUOTES,
                priority=PriorityTier.P1_ORDER_EXECUTION,
                timeout=0.5,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0

            assert p1_ok is True
            assert lat_ms < 50.0, f"Trial {trial}: P1 latency {lat_ms:.2f}ms >= 50ms"
            latencies.append(lat_ms)

            for t in threads:
                t.join(timeout=1.5)

        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        assert max_lat < 50.0, f"Max P1 latency {max_lat:.2f}ms exceeded 50ms"
        assert avg_lat < 25.0, f"Average P1 latency {avg_lat:.2f}ms unusually high"

    def test_sliding_window_boundary_quotes_broker_ceiling(self):
        """Stress vector 3: Sliding window boundary testing for QUOTES endpoint.

        Broker limit: 10.0 req/s.
        Limiter cap: max_window = 8 req / 1.0s (85% cap).
        Continuously acquire 25 tokens across multiple seconds.
        Verify that for EVERY 1.0-second window, count <= 8 (and strictly < 10 broker quota).
        """
        limiter = PrioritizedTokenBucketLimiter()
        timestamps = []

        total_requests = 20
        for _ in range(total_requests):
            ok = limiter.acquire(
                endpoint=EndpointCategory.QUOTES,
                priority=PriorityTier.P1_ORDER_EXECUTION,
                timeout=5.0,
            )
            assert ok is True
            timestamps.append(time.monotonic())

        # Verify sliding window invariant across all sub-windows
        for i, t_ref in enumerate(timestamps):
            # Count how many requests occurred in [t_ref - 1.0 + 1e-4, t_ref]
            window_requests = [
                t for t in timestamps if (t_ref - 1.0 + 1e-4) <= t <= (t_ref + 1e-4)
            ]
            count = len(window_requests)
            assert count <= 8, (
                f"Sliding window violation on QUOTES: {count} requests dispatched in 1.0s window ending at {t_ref:.4f} (Max allowed: 8)"
            )
            assert count < 10, f"QUOTES exceeded broker limit 10.0 req/s: {count} requests in 1.0s"

    def test_sliding_window_boundary_historical_broker_ceiling(self):
        """Stress vector 3 (cont): Sliding window boundary testing for HISTORICAL endpoint.

        Broker limit: 3.0 req/s.
        Limiter cap: max_window = 3 req / 1.0s (83.3% cap).
        Continuously acquire 8 tokens across multiple seconds.
        Verify that for EVERY 1.0-second window, count <= 3.
        """
        limiter = PrioritizedTokenBucketLimiter()
        timestamps = []

        total_requests = 8
        for _ in range(total_requests):
            ok = limiter.acquire(
                endpoint=EndpointCategory.HISTORICAL,
                priority=PriorityTier.P1_ORDER_EXECUTION,
                timeout=5.0,
            )
            assert ok is True
            timestamps.append(time.monotonic())

        for i, t_ref in enumerate(timestamps):
            window_requests = [
                t for t in timestamps if (t_ref - 1.0 + 1e-4) <= t <= (t_ref + 1e-4)
            ]
            count = len(window_requests)
            assert count <= 3, (
                f"Sliding window violation on HISTORICAL: {count} requests dispatched in 1.0s window ending at {t_ref:.4f} (Max allowed: 3)"
            )

    def test_sliding_window_boundary_orders_broker_ceiling(self):
        """Stress vector 3 (cont): Sliding window boundary testing for ORDERS endpoint.

        Broker limit: 10.0 req/s.
        Limiter cap: max_window = 8 req / 1.0s.
        Continuously acquire 16 tokens across multiple seconds.
        Verify that for EVERY 1.0-second window, count <= 8.
        """
        limiter = PrioritizedTokenBucketLimiter()
        timestamps = []

        total_requests = 16
        for _ in range(total_requests):
            ok = limiter.acquire(
                endpoint=EndpointCategory.ORDERS,
                priority=PriorityTier.P1_ORDER_EXECUTION,
                timeout=5.0,
            )
            assert ok is True
            timestamps.append(time.monotonic())

        for i, t_ref in enumerate(timestamps):
            window_requests = [
                t for t in timestamps if (t_ref - 1.0 + 1e-4) <= t <= (t_ref + 1e-4)
            ]
            count = len(window_requests)
            assert count <= 8, (
                f"Sliding window violation on ORDERS: {count} requests dispatched in 1.0s window ending at {t_ref:.4f} (Max allowed: 8)"
            )


    def test_100_thread_flood_contention(self):
        """Stress vector 4: Massive 100-thread flood across all buckets.

        Asserts that under extreme OS thread contention:
        - No deadlocks occur.
        - Token accounting is 100% exact.
        - Statistics match observed reality.
        """
        limiter = PrioritizedTokenBucketLimiter()
        num_threads = 100
        barrier = threading.Barrier(num_threads + 1)
        results = []
        lock = threading.Lock()

        endpoints = [EndpointCategory.HISTORICAL, EndpointCategory.QUOTES, EndpointCategory.ORDERS]
        priorities = [
            PriorityTier.P1_ORDER_EXECUTION,
            PriorityTier.P2_POSITION_MONITOR,
            PriorityTier.P3_HEALTH_EVALUATION,
            PriorityTier.P4_BACKGROUND_SCANNER,
        ]

        def task(i: int):
            ep = endpoints[i % 3]
            pri = priorities[i % 4]
            barrier.wait(timeout=5.0)
            ok = limiter.acquire(endpoint=ep, priority=pri, timeout=4.0)
            with lock:
                results.append((i, ep, pri, ok))

        threads = [threading.Thread(target=task, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()

        barrier.wait(timeout=5.0)

        for t in threads:
            t.join(timeout=8.0)
            assert not t.is_alive(), f"Thread failed to terminate under 100-thread flood"

        assert len(results) == num_threads
        acq_count = sum(1 for r in results if r[3])
        timeout_count = sum(1 for r in results if not r[3])
        stats_acq = sum(b.stats["total_acquired"] for b in limiter._buckets.values())
        stats_timeouts = sum(b.stats["total_timeouts"] for b in limiter._buckets.values())

        assert stats_acq == acq_count
        assert stats_timeouts == timeout_count
        assert acq_count > 0

    def test_rapid_ab1021_cooldown_injection_under_concurrency(self):
        """Stress vector 5: Inject AB1021 cooldown while threads are actively competing.

        Asserts:
        - Cooldown immediately halts new dispatches.
        - Zero requests leak through during cooldown duration.
        - System automatically resumes full throughput after cooldown elapses.
        """
        limiter = PrioritizedTokenBucketLimiter()
        dispatches_during_cooldown = []
        lock = threading.Lock()
        stop_event = threading.Event()

        def worker():
            while not stop_event.is_set():
                ok = limiter.acquire(EndpointCategory.QUOTES, PriorityTier.P2_POSITION_MONITOR, timeout=0.1)
                now = time.monotonic()
                if ok:
                    with lock:
                        if limiter.is_in_cooldown(EndpointCategory.QUOTES):
                            dispatches_during_cooldown.append(now)
                time.sleep(0.01)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()

        time.sleep(0.1)  # Let threads run

        # Trigger AB1021 cooldown of 0.3 seconds
        cooldown_duration = 0.3
        limiter.set_cooldown(cooldown_duration, endpoint=EndpointCategory.QUOTES)
        assert limiter.is_in_cooldown(EndpointCategory.QUOTES) is True

        time.sleep(cooldown_duration + 0.05)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        assert len(dispatches_during_cooldown) == 0, (
            f"Leaked {len(dispatches_during_cooldown)} dispatches during active cooldown!"
        )
        assert limiter.is_in_cooldown(EndpointCategory.QUOTES) is False

    def test_mass_timeout_waiter_eviction_under_concurrency(self):
        """Stress vector 6: Squeeze bucket and cause 30 threads to time out simultaneously.

        Asserts:
        - Timed out waiters are evicted from `_waiters`.
        - `_waiters` queue length returns to 0.
        - Subsequent P1 request can acquire without being blocked by dead waiters.
        """
        limiter = PrioritizedTokenBucketLimiter()
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        # Drain bucket completely
        with bucket._lock:
            bucket.tokens = 0.0
            bucket.last_fill = time.monotonic()

        num_waiters = 30
        barrier = threading.Barrier(num_waiters + 1)
        timeouts = []
        lock = threading.Lock()

        def short_waiter():
            barrier.wait(timeout=2.0)
            ok = limiter.acquire(EndpointCategory.QUOTES, PriorityTier.P4_BACKGROUND_SCANNER, timeout=0.04)
            with lock:
                timeouts.append(ok)

        threads = [threading.Thread(target=short_waiter) for _ in range(num_waiters)]
        for t in threads:
            t.start()

        barrier.wait(timeout=2.0)

        for t in threads:
            t.join(timeout=2.0)

        assert len(timeouts) == num_waiters
        assert all(ok is False for ok in timeouts)

        # Check queue eviction
        with bucket._lock:
            assert len(bucket._waiters) == 0, f"Leaked {len(bucket._waiters)} waiters in queue"

        # Refill tokens and verify fresh acquisition works immediately
        with bucket._lock:
            bucket.tokens = 5.0
            bucket._dispatch_history.clear()

        ok = limiter.acquire(EndpointCategory.QUOTES, PriorityTier.P1_ORDER_EXECUTION, timeout=0.1)
        assert ok is True
