# ============================================================================
# PROMETHEUS — Remediation Challenger 2: Empirical Verification & Stress Suite
# ============================================================================
"""
Independent empirical test suite executed by Challenger Remediation 2
to adversarially verify the Four Core Acceptance Criteria:
1. Zero HTTP 429 / AB1021 errors under high concurrency.
2. P95 quote latency < 1,000ms.
3. Broker call volume reduction >= 60%.
4. Zero synthetic price fabrication (GEMINI.md).
Plus:
5. Out-of-lock sleep & virtual scheduling in angelone_options.py.
6. Execution freeze invariants across PositionMonitor & PositionTracker.
"""

import os
import re
import time
import threading
from typing import Dict, List, Optional
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


class TestChallengerFourCoreAcceptanceCriteria:
    """Rigorous empirical validation of the Four Core Acceptance Criteria."""

    def test_criterion_1_and_2_and_3_high_concurrency_stress(self):
        """Stress harness: 100 concurrent workers hammering broker endpoints.
        Evaluates Criteria 1 (0 AB1021/429), 2 (P95 < 1000ms), and 3 (Call reduction >= 60%).
        """
        # Strict mock server enforcing 10.0 req/sec limit on quotes
        server = MockSmartConnectServer(max_quote_rate=10.0, simulated_latency_sec=0.002)
        limiter = PrioritizedTokenBucketLimiter()
        engine = BatchMarketDataEngine(server, limiter, debounce_ms=20.0, max_batch_size=50)
        cache = SharedPriceMicroCache(default_ttl=0.750)

        # 100 symbols across NSE, NFO, BFO
        symbols = (
            [{"exchange": "NSE", "symboltoken": str(10000 + i)} for i in range(15)] +
            [{"exchange": "NFO", "symboltoken": str(20000 + i)} for i in range(50)] +
            [{"exchange": "BFO", "symboltoken": str(30000 + i)} for i in range(35)]
        )
        all_tokens = [s["symboltoken"] for s in symbols]

        latencies_ms: List[float] = []
        client_read_requests: int = 0
        lock = threading.Lock()
        stop_event = threading.Event()
        errors: List[Exception] = []

        def cache_fetch_func(missing_tokens):
            reqs = [s for s in symbols if s["symboltoken"] in missing_tokens]
            quotes = engine.get_quotes_batch(reqs, priority=PriorityTier.P2_POSITION_MONITOR, timeout=1.5)
            now = time.monotonic()
            return {
                t: MicroCacheEntry(
                    price=quotes[t]["ltp"], bid=quotes[t]["bid"], ask=quotes[t]["ask"],
                    open_interest=quotes[t]["open_interest"], volume=quotes[t]["volume"],
                    timestamp=now, source="batch_rest", is_stale=False, tradingsymbol=f"SYM_{t}"
                )
                for t in quotes if t in quotes
            }

        def worker_active_monitor(worker_id: int):
            nonlocal client_read_requests
            # Each monitor queries 20 tokens
            slice_start = (worker_id * 5) % 80
            pos_tokens = all_tokens[slice_start:slice_start + 20]
            while not stop_event.is_set():
                t0 = time.monotonic()
                try:
                    res = cache.get_or_fetch(pos_tokens, cache_fetch_func)
                    dt = (time.monotonic() - t0) * 1000.0
                    with lock:
                        client_read_requests += len(pos_tokens)
                        latencies_ms.append(dt)
                except Exception as e:
                    with lock:
                        errors.append(e)
                time.sleep(0.02)

        def worker_order_router():
            while not stop_event.is_set():
                try:
                    if limiter.acquire(EndpointCategory.ORDERS, PriorityTier.P1_ORDER_EXECUTION, timeout=0.2):
                        server.modifyOrder({"orderid": "SL_TRAIL_ACTIVE"})
                except Exception as e:
                    with lock:
                        errors.append(e)
                time.sleep(0.04)

        # Spawn 20 concurrent monitoring workers + 2 order routing workers
        threads = [threading.Thread(target=worker_active_monitor, args=(i,)) for i in range(20)]
        threads.extend([threading.Thread(target=worker_order_router) for _ in range(2)])

        for t in threads:
            t.start()

        time.sleep(3.0)  # Sustained burst
        stop_event.set()

        for t in threads:
            t.join(timeout=3.0)
        engine.stop()

        # Check for unhandled exceptions
        assert len(errors) == 0, f"Encountered unexpected worker exceptions: {errors[:5]}"

        # --- CRITERION 1: Zero HTTP 429 / AB1021 Errors ---
        assert server.ab1021_count == 0, f"CRITERION 1 FAILED: Expected 0 AB1021 errors, got {server.ab1021_count}"
        assert server.http_429_count == 0, f"CRITERION 1 FAILED: Expected 0 HTTP 429 errors, got {server.http_429_count}"

        # --- CRITERION 2: P95 quote latency < 1,000ms ---
        assert len(latencies_ms) > 100, f"Insufficient samples: {len(latencies_ms)}"
        sorted_latencies = sorted(latencies_ms)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p90 = sorted_latencies[int(len(sorted_latencies) * 0.90)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        print(f"\n[LATENCY BENCHMARK] P50={p50:.2f}ms, P90={p90:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms (Samples: {len(latencies_ms)})")
        assert p95 < 1000.0, f"CRITERION 2 FAILED: P95 latency {p95:.2f}ms exceeds 1,000ms limit"

        # --- CRITERION 3: Broker call volume reduction >= 60% ---
        baseline_calls = client_read_requests
        actual_broker_calls = server.quotes_calls
        reduction_pct = ((baseline_calls - actual_broker_calls) / baseline_calls) * 100.0
        print(f"[CALL REDUCTION BENCHMARK] Baseline={baseline_calls}, Broker Calls={actual_broker_calls}, Reduction={reduction_pct:.2f}%")
        assert reduction_pct >= 60.0, f"CRITERION 3 FAILED: Reduction {reduction_pct:.2f}% is below 60.0%"

    def test_criterion_4_zero_synthetic_price_fabrication(self):
        """CRITERION 4: Rigorous empirical verification of Zero Price Fabrication (GEMINI.md)."""
        # 1. Static codebase scan: No hardcoded spot maps or 25050.0 in prometheus/
        found_matches = []
        for root, _, files in os.walk("prometheus"):
            for f in files:
                if f.endswith(".py") and "test" not in root and "test" not in f:
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                        content = fp.read()
                        if "25050.0" in content:
                            found_matches.append((path, "25050.0"))
                        if re.search(r"spot_map\s*=\s*\{", content):
                            found_matches.append((path, "spot_map"))

        assert len(found_matches) == 0, f"CRITERION 4 FAILED: Hardcoded synthetic price structures found in production code: {found_matches}"

        # 2. FallbackPolicy runtime behavior: No synthetic price when unconfigured
        FallbackPolicy.reset_alerts()
        FallbackPolicy.set_secondary_spot_feed(None)
        spot_res = FallbackPolicy.get_secondary_spot_price("NIFTY 50")
        assert spot_res is None, f"Expected None for unconfigured secondary feed, got {spot_res}"
        assert len(FallbackPolicy.stale_alerts) == 1
        assert FallbackPolicy.stale_alerts[0]["type"] == "SECONDARY_SPOT_UNCONFIGURED"

        # 3. FallbackPolicy runtime behavior: No synthetic price on feed exception
        def failing_feed(symbol: str):
            raise ConnectionError("Feed unreachable")

        FallbackPolicy.set_secondary_spot_feed(failing_feed)
        spot_res2 = FallbackPolicy.get_secondary_spot_price("NIFTY 50")
        assert spot_res2 is None, f"Expected None on failing secondary feed, got {spot_res2}"
        assert len(FallbackPolicy.stale_alerts) == 2
        assert FallbackPolicy.stale_alerts[1]["type"] == "SECONDARY_SPOT_FAILED"

        # Clean up
        FallbackPolicy.set_secondary_spot_feed(None)
        FallbackPolicy.reset_alerts()


class TestChallengerConcurrencyAndInvariants:
    """Adversarial challenge of concurrency mechanics and state invariants."""

    def test_out_of_lock_sleep_in_angelone_options(self):
        """Verify that AngelOneOptionChain._rate_limit NEVER holds _oi_lock while sleeping."""
        from prometheus.data.angelone_options import AngelOneOptionChain

        chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
        chain._min_interval = 0.200  # 200ms pacing
        chain._last_call = time.time()  # force full 200ms sleep
        chain._oi_lock = threading.Lock()
        chain._fetcher = None

        lock_acquired_during_sleep = threading.Event()
        lock_acquire_latency = []

        def sleeping_thread():
            chain._rate_limit(timeout=1.0)

        def probing_thread():
            time.sleep(0.030)  # Wait for sleeping_thread to enter _rate_limit and start sleeping
            t0 = time.monotonic()
            acquired = chain._oi_lock.acquire(timeout=0.010)
            t1 = time.monotonic()
            if acquired:
                lock_acquired_during_sleep.set()
                lock_acquire_latency.append((t1 - t0) * 1000.0)
                chain._oi_lock.release()

        t1 = threading.Thread(target=sleeping_thread)
        t2 = threading.Thread(target=probing_thread)

        t1.start()
        t2.start()

        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        assert lock_acquired_during_sleep.is_set(), (
            "DEFECT: _oi_lock was held during time.sleep! "
            "Probing thread failed to acquire mutex lock while rate limiter was sleeping."
        )
        print(f"\n[MUTEX CONTENTION] _oi_lock acquired in {lock_acquire_latency[0]:.3f}ms while caller was sleeping.")

    def test_position_monitor_freezes_ratchet_on_missing_quote(self):
        """Verify PositionMonitor strictly freezes current_sl and premium_hwm when ltp <= 0."""
        from unittest.mock import MagicMock
        from prometheus.execution.position_monitor import PositionMonitor, TrailingState

        mock_broker = MagicMock()
        mock_broker.get_ltp.return_value = 0.0  # Broker quote returns 0.0 (timeout or failure)
        on_trail = MagicMock()
        on_exit = MagicMock()

        monitor = PositionMonitor(broker=mock_broker, poll_interval=0.01, on_trailing_update=on_trail, on_exit=on_exit)
        state = TrailingState(
            position_id="POS_FROZEN_CHALLENGER",
            tradingsymbol="NIFTY26OCT25000CE",
            symbol="NIFTY",
            entry_premium=100.0,
            initial_sl=80.0,
            current_sl=110.0,  # Already ratcheted above entry
            target=180.0,
            direction="bullish",
            risk_distance=20.0,
            premium_hwm=130.0,
            token="35001",
        )
        monitor.add_position(state)

        # Trigger process tick with non-positive price
        monitor._process_tick(state, 0.0)
        assert state.current_sl == 110.0, "current_sl must remain frozen"
        assert state.premium_hwm == 130.0, "premium_hwm must remain frozen"
        on_trail.assert_not_called()
        on_exit.assert_not_called()

        # Trigger process tick with negative price
        monitor._process_tick(state, -5.0)
        assert state.current_sl == 110.0
        assert state.premium_hwm == 130.0
        on_trail.assert_not_called()
        on_exit.assert_not_called()

    def test_position_tracker_freezes_ratchet_on_non_positive_price(self):
        """Verify PositionTracker._maybe_advance_trailing_stop freezes on current_price <= 0.0."""
        from datetime import datetime
        from prometheus.papertrade.position_tracker import PositionTracker
        from prometheus.papertrade.fill_simulator import FillSimulator
        from unittest.mock import MagicMock
        from prometheus.papertrade.types import Position, Direction
        tracker = PositionTracker(fill_sim=FillSimulator(feed=MagicMock()))
        pos = Position(
            trade_id="POS_TRACKER_001",
            symbol="BANKNIFTY",
            instrument="BANKNIFTY26OCT52000CE",
            underlying="BANKNIFTY",
            direction=Direction.LONG,
            quantity=15,
            entry_price=200.0,
            entry_time=datetime.now(),
            stop_loss=160.0,
            target=300.0,
            max_bars=10,
            initial_sl=160.0,
            high_water_mark=200.0,
        )
        tracker.open_positions["POS_TRACKER_001"] = pos

        tracker._maybe_advance_trailing_stop(pos, 0.0)
        assert pos.stop_loss == 160.0
        assert pos.high_water_mark == 200.0

        tracker._maybe_advance_trailing_stop(pos, -10.0)
        assert pos.stop_loss == 160.0
        assert pos.high_water_mark == 200.0
