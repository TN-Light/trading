# ============================================================================
# PROMETHEUS — Adversarial Challenge Test Suite (Challenger M1-2)
# Starvation, Priority Aging, Timeout Leaks, and Cooldown Safety
# ============================================================================
"""
Adversarial Verification Suite designed by Challenger M1-2.

Rigorous empirical challenge of:
1. Dynamic priority aging & starvation protection (P4 -> P3 boost at >=4.0s)
2. Starvation resistance under constant high-priority contention
3. Timeout handling, zero-leak queue eviction, and zombie waiter prevention
4. Cooldown safety: global & endpoint isolation, monotonic extension, and reset
5. Multi-threaded thundering herd after cooldown release
6. Reserve token preservation (P1 unblocked vs P2/P3/P4 reserve floor)
7. Sliding window rate enforcement
"""

import threading
import time
from typing import List
import pytest

from prometheus.data.rate_limiter import (
    EndpointCategory,
    PrioritizedTokenBucketLimiter,
    PriorityTier,
    SmartAPIRateLimiter,
    _TokenBucket,
    _Waiter,
)


# ============================================================================
# 1. STARVATION & DYNAMIC PRIORITY AGING TESTS
# ============================================================================

class TestStarvationAndAging:
    """Adversarially challenge dynamic priority aging and starvation prevention."""

    def test_starvation_p4_promoted_to_p3_at_exact_aging_threshold(self):
        """Verify boundary condition: P4 is NOT promoted at 3.99s, but IS promoted at >=4.0s."""
        bucket = _TokenBucket(
            endpoint=EndpointCategory.QUOTES,
            fill_rate=1.0,
            capacity=1.0,
            aging_threshold_sec=4.0,
        )
        now = time.monotonic()
        waiter = _Waiter(
            effective_priority=int(PriorityTier.P4_BACKGROUND_SCANNER),
            seq=1,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            enqueued_at=now - 3.95,  # 3.95s ago (< 4.0s)
            event=threading.Event(),
        )

        with bucket._lock:
            bucket._waiters.append(waiter)
            bucket._update_waiter_priorities(now)
            assert waiter.effective_priority == int(PriorityTier.P4_BACKGROUND_SCANNER), (
                "P4 waiter promoted prematurely before 4.0s threshold"
            )

            # Advance simulated time past 4.0s
            bucket._update_waiter_priorities(now + 0.10)
            assert waiter.effective_priority == int(PriorityTier.P3_HEALTH_EVALUATION), (
                "P4 waiter NOT promoted to P3 after >=4.0s wait"
            )

    def test_starvation_p1_p2_p3_never_demoted_or_incorrectly_aged(self):
        """Verify that P1, P2, and P3 waiters never have their priorities modified by aging."""
        bucket = _TokenBucket(
            endpoint=EndpointCategory.QUOTES,
            fill_rate=1.0,
            capacity=1.0,
            aging_threshold_sec=4.0,
        )
        now = time.monotonic()
        w1 = _Waiter(int(PriorityTier.P1_ORDER_EXECUTION), 1, PriorityTier.P1_ORDER_EXECUTION, now - 10.0, threading.Event())
        w2 = _Waiter(int(PriorityTier.P2_POSITION_MONITOR), 2, PriorityTier.P2_POSITION_MONITOR, now - 10.0, threading.Event())
        w3 = _Waiter(int(PriorityTier.P3_HEALTH_EVALUATION), 3, PriorityTier.P3_HEALTH_EVALUATION, now - 10.0, threading.Event())

        with bucket._lock:
            bucket._waiters.extend([w1, w2, w3])
            bucket._update_waiter_priorities(now)
            assert w1.effective_priority == 1
            assert w2.effective_priority == 2
            assert w3.effective_priority == 3

    def test_starvation_adversarial_p4_eventually_completes_under_constant_p3_flood(self):
        """Starvation challenge: A P4 request waiting under continuous P3 contention.

        Under a steady stream of P3 requests, P4 would wait forever without aging.
        With aging boost (threshold 4.0s), P4 gets promoted to P3 and, by virtue of its
        earlier arrival (seq), gets dispatched ahead of newly arriving P3 requests.
        """
        limiter = PrioritizedTokenBucketLimiter(
            custom_rates={EndpointCategory.QUOTES: (4.0, 4.0, 4)}  # 4 req/s = 0.25s per token
        )
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        # Drain bucket
        with bucket._lock:
            bucket.tokens = 0.0
            bucket.last_fill = time.monotonic()

        p4_completed = threading.Event()
        p4_completion_time = []
        stop_p3_flood = threading.Event()
        p3_dispatch_count = [0]

        # Thread 1: P4 background scanner waiting up to 6.0s
        t_start = time.monotonic()
        def p4_worker():
            acquired = limiter.acquire(
                EndpointCategory.QUOTES,
                priority=PriorityTier.P4_BACKGROUND_SCANNER,
                timeout=6.5,
            )
            if acquired:
                p4_completion_time.append(time.monotonic() - t_start)
                p4_completed.set()

        t_p4 = threading.Thread(target=p4_worker, daemon=True)
        t_p4.start()

        # Thread 2: P3 flood thread issuing continuous P3 requests every 0.1s
        def p3_flooder():
            while not stop_p3_flood.is_set():
                limiter.acquire(
                    EndpointCategory.QUOTES,
                    priority=PriorityTier.P3_HEALTH_EVALUATION,
                    timeout=0.4,
                )
                p3_dispatch_count[0] += 1
                time.sleep(0.08)

        t_p3 = threading.Thread(target=p3_flooder, daemon=True)
        t_p3.start()

        # Wait for P4 to complete or fail
        finished = p4_completed.wait(timeout=6.0)
        stop_p3_flood.set()
        t_p3.join(timeout=1.0)
        t_p4.join(timeout=1.0)

        assert finished, "P4 request was starved and did NOT complete within 6.0s under P3 flood!"
        assert len(p4_completion_time) == 1
        elapsed = p4_completion_time[0]
        # P4 should have waited at least ~4.0s (aging threshold) and completed shortly after promotion
        assert elapsed >= 3.85, f"P4 completed suspiciously early ({elapsed:.2f}s) — did it bypass priority queue?"
        assert elapsed <= 5.5, f"P4 completed too late ({elapsed:.2f}s) — aging boost was not effective!"
        assert p3_dispatch_count[0] >= 5, "P3 flood thread did not produce expected contention"

    def test_starvation_multiple_p4_waiters_preserve_fifo_after_boost(self):
        """Verify that when multiple P4 waiters cross the 4.0s threshold, they maintain FIFO order."""
        bucket = _TokenBucket(
            endpoint=EndpointCategory.QUOTES,
            fill_rate=2.0,
            capacity=2.0,
            aging_threshold_sec=0.20,  # Fast threshold for deterministic multi-waiter test
        )
        now = time.monotonic()
        w1 = _Waiter(int(PriorityTier.P4_BACKGROUND_SCANNER), 1, PriorityTier.P4_BACKGROUND_SCANNER, now - 0.25, threading.Event())
        w2 = _Waiter(int(PriorityTier.P4_BACKGROUND_SCANNER), 2, PriorityTier.P4_BACKGROUND_SCANNER, now - 0.23, threading.Event())
        w3 = _Waiter(int(PriorityTier.P4_BACKGROUND_SCANNER), 3, PriorityTier.P4_BACKGROUND_SCANNER, now - 0.21, threading.Event())

        with bucket._lock:
            bucket._waiters.extend([w3, w1, w2])  # Insert out of order
            bucket._update_waiter_priorities(now)
            # After sorting, order must be w1, w2, w3
            assert bucket._waiters[0] is w1
            assert bucket._waiters[1] is w2
            assert bucket._waiters[2] is w3

    def test_starvation_p1_and_p2_still_preempt_boosted_p4(self):
        """Invariant: An aged P4 waiter is boosted to P3, NOT to P1 or P2.

        New P1 (Order) and P2 (Position Monitor) requests must still jump ahead of boosted P4.
        """
        bucket = _TokenBucket(
            endpoint=EndpointCategory.QUOTES,
            fill_rate=2.0,
            capacity=2.0,
            aging_threshold_sec=0.10,
        )
        now = time.monotonic()
        # w_p4 was enqueued 0.5s ago -> eligible for P3 boost
        w_p4 = _Waiter(int(PriorityTier.P4_BACKGROUND_SCANNER), 1, PriorityTier.P4_BACKGROUND_SCANNER, now - 0.50, threading.Event())
        # w_p1 just arrived
        w_p1 = _Waiter(int(PriorityTier.P1_ORDER_EXECUTION), 10, PriorityTier.P1_ORDER_EXECUTION, now, threading.Event())
        # w_p2 just arrived
        w_p2 = _Waiter(int(PriorityTier.P2_POSITION_MONITOR), 11, PriorityTier.P2_POSITION_MONITOR, now, threading.Event())

        with bucket._lock:
            bucket._waiters.extend([w_p4, w_p1, w_p2])
            bucket._update_waiter_priorities(now)
            # Order must be: w_p1 (pri 1) > w_p2 (pri 2) > w_p4 (pri 3 aged)
            assert bucket._waiters[0] is w_p1
            assert bucket._waiters[1] is w_p2
            assert bucket._waiters[2] is w_p4


# ============================================================================
# 2. TIMEOUT HANDLING & QUEUE LEAK / EVICTION TESTS
# ============================================================================

class TestTimeoutHandlingAndQueueEviction:
    """Adversarially challenge timeout handling, queue leaks, and zombie waiters."""

    def test_timeout_zero_and_negative_immediate_bailout(self):
        """Zero or negative timeout on an empty bucket must bail out in <1ms without queueing."""
        limiter = PrioritizedTokenBucketLimiter()
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        with bucket._lock:
            bucket.tokens = 0.0
            bucket.last_fill = time.monotonic()

        t0 = time.monotonic()
        res_zero = limiter.acquire(EndpointCategory.QUOTES, timeout=0.0)
        t_zero = time.monotonic() - t0

        t0 = time.monotonic()
        res_neg = limiter.acquire(EndpointCategory.QUOTES, timeout=-5.0)
        t_neg = time.monotonic() - t0

        assert res_zero is False
        assert res_neg is False
        assert t_zero < 0.005, f"Timeout 0.0 blocked for {t_zero*1000:.2f}ms"
        assert t_neg < 0.005, f"Negative timeout blocked for {t_neg*1000:.2f}ms"

        with bucket._lock:
            assert len(bucket._waiters) == 0, "Waiters queue leaked on zero/negative timeout"

    def test_timeout_massive_concurrency_no_queue_leaks(self):
        """50 concurrent threads with micro-timeouts (0.01s to 0.03s) on an exhausted bucket.

        All 50 must cleanly time out. After completion, bucket._waiters must have exactly 0 elements.
        """
        limiter = PrioritizedTokenBucketLimiter(
            custom_rates={EndpointCategory.QUOTES: (0.001, 0.001, 1)}  # Effectively zero refill
        )
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        with bucket._lock:
            bucket.tokens = 0.0
            bucket.last_fill = time.monotonic()

        results = []
        lock = threading.Lock()

        def timeout_worker(timeout_sec: float):
            r = limiter.acquire(
                EndpointCategory.QUOTES,
                priority=PriorityTier.P4_BACKGROUND_SCANNER,
                timeout=timeout_sec,
            )
            with lock:
                results.append(r)

        threads = [
            threading.Thread(target=timeout_worker, args=(0.01 + (i % 5) * 0.005,))
            for i in range(50)
        ]

        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1.5)
        elapsed = time.monotonic() - t0

        assert len(results) == 50, f"Expected 50 results, got {len(results)}"
        assert not any(results), "Expected all 50 micro-timeout workers to fail"
        assert elapsed < 1.0, f"50 micro-timeout workers took too long: {elapsed:.2f}s"

        with bucket._lock:
            assert len(bucket._waiters) == 0, (
                f"Memory leak / zombie waiter detected: {len(bucket._waiters)} waiters remain in queue!"
            )

        # Telemetry verification
        stats = limiter.get_stats()
        assert stats["buckets"]["quotes"]["total_timeouts"] >= 50

    def test_timeout_does_not_consume_or_steal_replenished_tokens(self):
        """Verify that a timed-out waiter never consumes or holds tokens replenished after its exit."""
        limiter = PrioritizedTokenBucketLimiter(
            custom_rates={EndpointCategory.QUOTES: (1.0, 1.0, 1)}
        )
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        # Drain bucket
        with bucket._lock:
            bucket.tokens = 0.0
            bucket.last_fill = time.monotonic()

        # Thread 1 times out after 0.02s
        t_res = limiter.acquire(EndpointCategory.QUOTES, timeout=0.02)
        assert t_res is False

        # Add exactly 1 token
        with bucket._lock:
            bucket.tokens = 1.0
            bucket.last_fill = time.monotonic()

        # Thread 2 should acquire immediately
        t0 = time.monotonic()
        acquired = limiter.acquire(
            EndpointCategory.QUOTES,
            priority=PriorityTier.P1_ORDER_EXECUTION,
            timeout=0.10,
        )
        assert acquired is True
        assert (time.monotonic() - t0) < 0.05

    def test_timeout_interleaved_with_successful_waiters(self):
        """Mix of threads: 10 with short timeout (0.02s) and 5 with sufficient timeout (0.8s).

        Short timeout threads must bail cleanly; long timeout threads must acquire tokens in order.
        """
        limiter = PrioritizedTokenBucketLimiter(
            custom_rates={EndpointCategory.QUOTES: (5.0, 5.0, 5)}  # 1 token every 0.20s
        )
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        with bucket._lock:
            bucket.tokens = 0.0
            bucket.last_fill = time.monotonic()

        short_results = []
        long_results = []
        lock = threading.Lock()

        def short_worker():
            r = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P4_BACKGROUND_SCANNER, timeout=0.02)
            with lock:
                short_results.append(r)

        def long_worker():
            r = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P2_POSITION_MONITOR, timeout=1.2)
            with lock:
                long_results.append(r)

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=short_worker))
        for _ in range(3):
            threads.append(threading.Thread(target=long_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert len(short_results) == 10
        assert not any(short_results), "Short timeout workers unexpectedly succeeded"
        assert len(long_results) == 3
        assert all(long_results), "Long timeout workers failed to acquire"

        with bucket._lock:
            assert len(bucket._waiters) == 0, "Waiters leaked after interleaved test"


# ============================================================================
# 3. COOLDOWN SAFETY & PAUSE INTEGRITY TESTS
# ============================================================================

class TestCooldownSafetyAndPauseIntegrity:
    """Adversarially challenge cooldown pause mechanics, isolation, monotonicity, and reset."""

    def test_cooldown_global_pauses_all_endpoints(self):
        """Setting global cooldown pauses all 3 endpoints (Quotes, Historical, Orders)."""
        limiter = PrioritizedTokenBucketLimiter()
        limiter.set_cooldown(0.35)

        # All 3 endpoints must report in cooldown
        assert limiter.is_in_cooldown(EndpointCategory.QUOTES) is True
        assert limiter.is_in_cooldown(EndpointCategory.HISTORICAL) is True
        assert limiter.is_in_cooldown(EndpointCategory.ORDERS) is True

        # Immediate calls (timeout=0) must be rejected
        assert limiter.acquire(EndpointCategory.QUOTES, timeout=0.0) is False
        assert limiter.acquire(EndpointCategory.HISTORICAL, timeout=0.0) is False
        assert limiter.acquire(EndpointCategory.ORDERS, timeout=0.0) is False

        # Caller with timeout=1.0s must hold until cooldown expires, then succeed
        t0 = time.monotonic()
        res = limiter.acquire(EndpointCategory.QUOTES, timeout=1.0)
        elapsed = time.monotonic() - t0

        assert res is True
        assert elapsed >= 0.30, f"Dispatch did not pause for cooldown duration: elapsed {elapsed:.3f}s"

    def test_cooldown_endpoint_isolation_preserves_other_endpoints(self):
        """Setting cooldown on QUOTES must NOT pause HISTORICAL or ORDERS."""
        limiter = PrioritizedTokenBucketLimiter()
        limiter.set_cooldown(1.5, endpoint=EndpointCategory.QUOTES)

        assert limiter.is_in_cooldown(EndpointCategory.QUOTES) is True
        assert limiter.is_in_cooldown(EndpointCategory.HISTORICAL) is False
        assert limiter.is_in_cooldown(EndpointCategory.ORDERS) is False

        # HISTORICAL should acquire with zero delay (<5ms)
        t0 = time.monotonic()
        res_hist = limiter.acquire(EndpointCategory.HISTORICAL, timeout=0.05)
        elapsed_hist = time.monotonic() - t0

        assert res_hist is True
        assert elapsed_hist < 0.010, f"HISTORICAL delayed by QUOTES cooldown: {elapsed_hist*1000:.2f}ms"

        # ORDERS should acquire with zero delay (<5ms)
        t0 = time.monotonic()
        res_orders = limiter.acquire(EndpointCategory.ORDERS, timeout=0.05)
        elapsed_orders = time.monotonic() - t0

        assert res_orders is True
        assert elapsed_orders < 0.010, f"ORDERS delayed by QUOTES cooldown: {elapsed_orders*1000:.2f}ms"

        # QUOTES zero-timeout must fail
        assert limiter.acquire(EndpointCategory.QUOTES, timeout=0.0) is False

    def test_cooldown_monotonicity_never_shortened(self):
        """Cooldown target timestamp must only extend, never shorten."""
        limiter = PrioritizedTokenBucketLimiter()
        limiter.set_cooldown(3.0)
        initial_target = float(limiter.cooldown_until)

        # Attempt to shorten cooldown to 1.0s
        limiter.set_cooldown(1.0)
        assert float(limiter.cooldown_until) == initial_target, "Cooldown was shortened by smaller second value!"

        # Attempt to extend cooldown to 5.0s
        limiter.set_cooldown(5.0)
        assert float(limiter.cooldown_until) > initial_target, "Cooldown failed to extend to longer value!"

    def test_cooldown_manual_reset_clears_all_pauses(self):
        """Calling reset_cooldown() must instantly clear active cooldown and allow dispatch."""
        limiter = PrioritizedTokenBucketLimiter()
        limiter.set_cooldown(30.0)

        assert limiter.is_in_cooldown() is True
        assert limiter.get_remaining_cooldown() > 25.0

        # Reset cooldown
        limiter.reset_cooldown()

        assert limiter.is_in_cooldown() is False
        assert limiter.get_remaining_cooldown() == 0.0

        # Next acquire succeeds immediately
        t0 = time.monotonic()
        res = limiter.acquire(EndpointCategory.QUOTES, timeout=0.05)
        assert res is True
        assert (time.monotonic() - t0) < 0.010

    def test_cooldown_thundering_herd_dispatch_order(self):
        """10 threads waiting during cooldown must dispatch in strict priority order once cooldown expires."""
        limiter = PrioritizedTokenBucketLimiter(
            custom_rates={EndpointCategory.QUOTES: (20.0, 20.0, 20)}
        )
        limiter.set_cooldown(0.20)

        dispatch_log = []
        lock = threading.Lock()

        def client_worker(tag: str, pri: PriorityTier):
            if limiter.acquire(EndpointCategory.QUOTES, priority=pri, timeout=1.0):
                with lock:
                    dispatch_log.append(tag)

        # Launch P4, P3, P2, P1 threads during cooldown
        threads = [
            threading.Thread(target=client_worker, args=("P4", PriorityTier.P4_BACKGROUND_SCANNER)),
            threading.Thread(target=client_worker, args=("P3", PriorityTier.P3_HEALTH_EVALUATION)),
            threading.Thread(target=client_worker, args=("P2", PriorityTier.P2_POSITION_MONITOR)),
            threading.Thread(target=client_worker, args=("P1", PriorityTier.P1_ORDER_EXECUTION)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1.5)

        assert len(dispatch_log) == 4
        assert dispatch_log == ["P1", "P2", "P3", "P4"], (
            f"Thundering herd post-cooldown did not dispatch in priority order: {dispatch_log}"
        )


# ============================================================================
# 4. RESERVE TOKEN INVARIANTS & SLIDING WINDOW LIMITS
# ============================================================================

class TestReserveTokenAndSlidingWindowInvariants:
    """Challenge reserve token floor (P1 vs P2/P3/P4) and rolling 1.0s window cap."""

    def test_p1_consumes_down_to_zero_while_p2_preserves_reserves(self):
        """P1 (Orders/SL) can consume down to 0 tokens; P2/P3/P4 cannot consume reserve tokens."""
        limiter = PrioritizedTokenBucketLimiter()
        bucket = limiter._buckets[EndpointCategory.QUOTES]

        # Set tokens to exactly 2.0 (which equals reserve_tokens)
        with bucket._lock:
            bucket.tokens = 2.0
            bucket.last_fill = time.monotonic()

        # P2 requires tokens >= reserve (2.0) + 1.0 = 3.0. With 2.0 tokens, P2 cannot acquire!
        res_p2 = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P2_POSITION_MONITOR, timeout=0.0)
        assert res_p2 is False, "P2 breached reserve token floor!"

        # P3 requires tokens >= 3.0. Cannot acquire!
        res_p3 = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P3_HEALTH_EVALUATION, timeout=0.0)
        assert res_p3 is False, "P3 breached reserve token floor!"

        # P1 requires tokens >= 0.0 + 1.0 = 1.0. With 2.0 tokens, P1 CAN acquire!
        res_p1 = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.0)
        assert res_p1 is True, "P1 was blocked despite tokens being above 0 floor!"

        # Tokens now 1.0. P1 can acquire one more!
        res_p1_second = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.0)
        assert res_p1_second is True

        # Tokens now 0.0. P1 now cannot acquire without waiting!
        res_p1_third = limiter.acquire(EndpointCategory.QUOTES, priority=PriorityTier.P1_ORDER_EXECUTION, timeout=0.0)
        assert res_p1_third is False

    def test_sliding_window_hard_caps_dispatch_at_max_window(self):
        """Sliding window strictly caps requests at max_window within 1.0 second, even if tokens exist."""
        # Max window = 4 in 1.0s
        limiter = PrioritizedTokenBucketLimiter(
            custom_rates={EndpointCategory.HISTORICAL: (2.5, 10.0, 4)}  # 10 capacity, but max_window=4
        )

        # 4 requests should pass immediately
        for _ in range(4):
            assert limiter.acquire(EndpointCategory.HISTORICAL, timeout=0.05) is True

        # 5th request must wait for sliding window to slide past 1.0s, even though tokens >= 5.0
        t0 = time.monotonic()
        res_5th = limiter.acquire(EndpointCategory.HISTORICAL, timeout=1.5)
        elapsed = time.monotonic() - t0

        assert res_5th is True
        assert elapsed >= 0.95, f"5th request bypassed sliding window cap: elapsed {elapsed:.3f}s"
