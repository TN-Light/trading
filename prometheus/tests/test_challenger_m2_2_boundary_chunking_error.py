# ============================================================================
# PROMETHEUS — Adversarial Challenge Test Suite (Challenger M2-2)
# Boundary Chunking, Cross-Segment Partitioning, Error Demux & Zero-Hallucination
# ============================================================================
"""
Adversarial Verification Suite designed by Challenger M2-2.

Rigorous empirical challenge of:
1. 50-token payload boundary: 1, 49, 50, 51, 99, 100, 101, 105, 150, 151 tokens.
2. Cross-segment partitioning: {"NFO": [...], "BFO": [...], "NSE": [...]} <= 50 tokens combined.
3. Broker rejection & error demultiplexing: AB1021 cooldown, AB1010 rejection, timeout propagation, partial chunk failure.
4. Zero-hallucination mandate (GEMINI.md): unfetched/missing tokens omitted, zero synthetic prices fabricated.
5. Priority inheritance & high-concurrency caller demultiplexing.
"""

import threading
import time
from typing import Any, Dict, List, Optional
import pytest

from prometheus.tests.mock_smartapi_server import MockSmartConnectServer
from prometheus.data.rate_limiter import (
    EndpointCategory,
    PrioritizedTokenBucketLimiter,
    PriorityTier,
)
from prometheus.data.batch_engine import BatchMarketDataEngine


# ============================================================================
# 1. 50-TOKEN PAYLOAD BOUNDARY CHUNKING TESTS
# ============================================================================

class TestBoundaryTokenChunking:
    """Adversarially challenge chunking boundaries: 1, 49, 50, 51, 100, 105, 150, 151 tokens."""

    def test_boundary_1_token_single_call(self):
        """Boundary: 1 token dispatches in exactly 1 batch call."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": "10001"}]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 1
        assert "10001" in quotes
        assert server.batch_calls == 1
        assert server.payload_rejected_count == 0

    def test_boundary_49_tokens_single_call(self):
        """Boundary: 49 tokens (just below limit) dispatches in exactly 1 batch call."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10100 + i)} for i in range(49)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 49
        assert server.batch_calls == 1
        assert server.payload_rejected_count == 0

    def test_boundary_50_tokens_exact_single_call(self):
        """Boundary: Exactly 50 tokens dispatches in exactly 1 batch call."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10200 + i)} for i in range(50)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 50
        assert server.batch_calls == 1
        assert server.payload_rejected_count == 0

    def test_boundary_51_tokens_two_calls_50_and_1(self):
        """Boundary: 51 tokens (limit + 1) splits into exactly 2 calls (50 + 1)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10300 + i)} for i in range(51)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 51
        assert server.batch_calls == 2
        assert server.quotes_calls == 2
        assert server.payload_rejected_count == 0

    def test_boundary_99_tokens_two_calls_50_and_49(self):
        """Boundary: 99 tokens splits into exactly 2 calls (50 + 49)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10400 + i)} for i in range(99)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 99
        assert server.batch_calls == 2
        assert server.payload_rejected_count == 0

    def test_boundary_100_tokens_two_calls_50_and_50(self):
        """Boundary: Exactly 100 tokens splits into exactly 2 calls (50 + 50)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10500 + i)} for i in range(100)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 100
        assert server.batch_calls == 2
        assert server.payload_rejected_count == 0

    def test_boundary_101_tokens_three_calls_50_50_1(self):
        """Boundary: 101 tokens splits into exactly 3 calls (50 + 50 + 1)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10600 + i)} for i in range(101)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 101
        assert server.batch_calls == 3
        assert server.payload_rejected_count == 0

    def test_boundary_105_tokens_three_calls_50_50_5(self):
        """Boundary: 105 tokens splits into exactly 3 calls (50 + 50 + 5)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10700 + i)} for i in range(105)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 105
        assert server.batch_calls == 3
        assert server.payload_rejected_count == 0

    def test_boundary_150_tokens_three_calls_50_50_50(self):
        """Boundary: Exactly 150 tokens splits into exactly 3 calls (50 + 50 + 50)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10800 + i)} for i in range(150)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 150
        assert server.batch_calls == 3
        assert server.payload_rejected_count == 0

    def test_boundary_151_tokens_four_calls_50_50_50_1(self):
        """Boundary: 151 tokens splits into exactly 4 calls (50 + 50 + 50 + 1)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [{"exchange": "NFO", "symboltoken": str(10900 + i)} for i in range(151)]
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 151
        assert server.batch_calls == 4
        assert server.payload_rejected_count == 0

    def test_no_token_loss_or_corruption_across_chunk_boundaries(self):
        """Verify all tokens across chunk boundaries match requested keys exactly."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        requested_tokens = [str(20000 + i) for i in range(105)]
        reqs = [{"exchange": "NFO", "symboltoken": t} for t in requested_tokens]
        quotes = engine.get_quotes_batch(reqs)

        assert set(quotes.keys()) == set(requested_tokens)
        for t in requested_tokens:
            q = quotes[t]
            assert q["symboltoken"] == t
            assert q["ltp"] > 0
            assert q["bid"] > 0
            assert q["ask"] > 0


# ============================================================================
# 2. CROSS-SEGMENT PARTITIONING TESTS
# ============================================================================

class TestCrossSegmentPartitioning:
    """Adversarially challenge cross-segment partitioning: NFO, BFO, NSE <= 50 tokens combined."""

    def test_cross_segment_50_tokens_split_across_three_exchanges(self):
        """Verify 20 NFO + 20 BFO + 10 NSE = 50 tokens combined executes in 1 call."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = (
            [{"exchange": "NFO", "symboltoken": str(30000 + i)} for i in range(20)] +
            [{"exchange": "BFO", "symboltoken": str(40000 + i)} for i in range(20)] +
            [{"exchange": "NSE", "symboltoken": str(50000 + i)} for i in range(10)]
        )
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 50
        assert server.batch_calls == 1
        assert server.payload_rejected_count == 0

    def test_cross_segment_51_tokens_splits_enforcing_ceiling_per_call(self):
        """Verify 20 NFO + 20 BFO + 11 NSE = 51 tokens combined splits into 2 calls."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = (
            [{"exchange": "NFO", "symboltoken": str(30100 + i)} for i in range(20)] +
            [{"exchange": "BFO", "symboltoken": str(40100 + i)} for i in range(20)] +
            [{"exchange": "NSE", "symboltoken": str(50100 + i)} for i in range(11)]
        )
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 51
        assert server.batch_calls == 2
        assert server.payload_rejected_count == 0

    def test_cross_segment_all_dispatched_payloads_strictly_under_or_equal_50(self):
        """Interceptor test: Inspect every outbound payload passed to client.getMarketData."""
        recorded_payloads: List[Dict[str, List[str]]] = []

        class InterceptingServer(MockSmartConnectServer):
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
                recorded_payloads.append(payload)
                return super().getMarketData(mode, payload)

        server = InterceptingServer()
        engine = BatchMarketDataEngine(server)

        # 48 NFO + 26 BFO + 31 NSE = 105 tokens
        reqs = (
            [{"exchange": "NFO", "symboltoken": str(60000 + i)} for i in range(48)] +
            [{"exchange": "BFO", "symboltoken": str(70000 + i)} for i in range(26)] +
            [{"exchange": "NSE", "symboltoken": str(80000 + i)} for i in range(31)]
        )
        quotes = engine.get_quotes_batch(reqs)

        assert len(quotes) == 105
        assert len(recorded_payloads) == 3

        # Every outbound payload must have total tokens across all exchange segments <= 50
        for idx, payload in enumerate(recorded_payloads):
            total_tokens_in_payload = sum(len(toks) for toks in payload.values())
            assert total_tokens_in_payload <= 50, (
                f"Payload {idx + 1} violated ceiling: total tokens = {total_tokens_in_payload} > 50! "
                f"Payload content: {payload}"
            )

    def test_cross_segment_exchange_tag_preservation(self):
        """Verify returned quote dictionaries retain the correct exchange tag."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [
            {"exchange": "NFO", "symboltoken": "90001"},
            {"exchange": "BFO", "symboltoken": "90002"},
            {"exchange": "NSE", "symboltoken": "90003"},
        ]
        quotes = engine.get_quotes_batch(reqs)

        assert quotes["90001"]["exchange"] == "NFO"
        assert quotes["90002"]["exchange"] == "BFO"
        assert quotes["90003"]["exchange"] == "NSE"

    def test_cross_segment_case_insensitivity_and_default(self):
        """Verify case normalization ('nfo' -> 'NFO') and missing exchange defaults to 'NFO'."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)
        reqs = [
            {"exchange": "nfo", "symboltoken": "91001"},
            {"exchange": "BfO", "symboltoken": "91002"},
            {"exchange": "nse", "symboltoken": "91003"},
            {"symboltoken": "91004"},  # Missing exchange
        ]
        quotes = engine.get_quotes_batch(reqs)

        assert quotes["91001"]["exchange"] == "NFO"
        assert quotes["91002"]["exchange"] == "BFO"
        assert quotes["91003"]["exchange"] == "NSE"
        assert quotes["91004"]["exchange"] == "NFO"


# ============================================================================
# 3. BROKER REJECTION & ERROR DEMULTIPLEXING TESTS
# ============================================================================

class TestErrorHandlingAndDemultiplexing:
    """Adversarially challenge broker rejection codes, timeout propagation, and demux."""

    def test_broker_ab1021_triggers_limiter_cooldown_and_returns_empty(self):
        """Broker AB1021 error sets 20s cooldown on limiter and returns {} without crashing."""
        server = MockSmartConnectServer()
        server.inject_ab1021 = True
        limiter = PrioritizedTokenBucketLimiter()
        engine = BatchMarketDataEngine(server, limiter=limiter)

        quotes = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "92001"}], timeout=0.1)
        assert quotes == {}
        assert limiter.is_in_cooldown(EndpointCategory.QUOTES) is True
        assert limiter.get_remaining_cooldown(EndpointCategory.QUOTES) > 15.0

    def test_broker_ab1010_payload_rejection_returns_empty(self):
        """Broker AB1010 payload rejection is handled safely returning {}."""
        class RejectionServer(MockSmartConnectServer):
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
                return {
                    "status": False,
                    "errorcode": "AB1010",
                    "message": "Invalid token count",
                    "data": None,
                }

        engine = BatchMarketDataEngine(RejectionServer())
        quotes = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "92002"}])
        assert quotes == {}

    def test_broker_socket_timeout_unblocks_cleanly_without_hanging(self):
        """Transient broker socket timeout unblocks caller promptly returning {}."""
        server = MockSmartConnectServer()
        server.inject_timeout = True
        engine = BatchMarketDataEngine(server)

        t0 = time.monotonic()
        quotes = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "92003"}], timeout=0.2)
        elapsed = time.monotonic() - t0

        assert quotes == {}
        assert elapsed < 0.5  # Did not hang or block indefinitely

    def test_broker_network_exception_propagates_cleanly(self):
        """Uncaught network socket exception does not deadlock engine; callers receive {}."""
        class ExplodingServer:
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
                raise ConnectionResetError("Broker socket forcibly closed by remote host")

        engine = BatchMarketDataEngine(ExplodingServer())
        quotes = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "92004"}], timeout=0.2)
        assert quotes == {}
        assert engine.stats["total_exceptions"] >= 1

    def test_partial_chunk_failure_preserves_successful_chunks_and_omits_failed(self):
        """60 tokens (50 in chunk 1, 10 in chunk 2). Chunk 1 succeeds, Chunk 2 fails.
        Callers must receive the 50 valid tokens; failed 10 tokens must be omitted with 0 synthesis."""
        class PartialFailureServer(MockSmartConnectServer):
            def __init__(self):
                super().__init__()
                self.call_idx = 0

            def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
                self.call_idx += 1
                if self.call_idx == 2:
                    raise ConnectionError("Chunk 2 socket failure")
                return super().getMarketData(mode, payload)

        server = PartialFailureServer()
        engine = BatchMarketDataEngine(server)

        # 60 tokens -> Chunk 1: 50 tokens (93000..93049), Chunk 2: 10 tokens (93050..93059)
        reqs = [{"exchange": "NFO", "symboltoken": str(93000 + i)} for i in range(60)]
        quotes = engine.get_quotes_batch(reqs, timeout=0.5)

        # 50 tokens from Chunk 1 are present
        assert len(quotes) == 50
        for i in range(50):
            assert str(93000 + i) in quotes

        # 10 tokens from failed Chunk 2 are strictly omitted (Zero Hallucination)
        for i in range(50, 60):
            assert str(93000 + i) not in quotes

    def test_concurrent_callers_during_broker_failure_all_unblock(self):
        """Verify multiple concurrent callers during a broker outage all unblock cleanly."""
        class DownServer:
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
                raise TimeoutError("Broker unreachable")

        engine = BatchMarketDataEngine(DownServer())
        results = {}
        threads = []

        def worker(wid):
            res = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": str(94000 + wid)}], timeout=0.2)
            results[wid] = res

        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=1.0)

        assert len(results) == 10
        for wid in range(10):
            assert results[wid] == {}


# ============================================================================
# 4. ZERO-HALLUCINATION MANDATE (GEMINI.md)
# ============================================================================

class TestZeroHallucinationMandate:
    """Adversarially verify that unfetched/missing tokens are omitted and no synthetic prices fabricated."""

    def test_missing_tokens_omitted_no_synthetic_prices(self):
        """When broker returns data for only a subset of requested tokens, missing tokens must be omitted."""
        class PartialDataServer(MockSmartConnectServer):
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
                # Only return quote for "95001", omit "95002" and "95003"
                return {
                    "status": True,
                    "data": {
                        "fetched": [{
                            "exchange": "NFO",
                            "tradingSymbol": "NIFTY26OCT24400CE",
                            "symbolToken": "95001",
                            "ltp": 250.0,
                            "open": 240.0,
                            "high": 255.0,
                            "low": 235.0,
                            "close": 245.0,
                            "bestBidPrice": 249.8,
                            "bestAskPrice": 250.2,
                            "opnInterest": 100000,
                            "tradeVolume": 50000,
                        }]
                    }
                }

        engine = BatchMarketDataEngine(PartialDataServer())
        reqs = [
            {"exchange": "NFO", "symboltoken": "95001"},
            {"exchange": "NFO", "symboltoken": "95002"},
            {"exchange": "NFO", "symboltoken": "95003"},
        ]
        quotes = engine.get_quotes_batch(reqs)

        # "95001" must be present with authentic price
        assert "95001" in quotes
        assert quotes["95001"]["ltp"] == 250.0

        # "95002" and "95003" must be strictly absent (Zero Hallucination)
        assert "95002" not in quotes
        assert "95003" not in quotes

    def test_completely_empty_broker_response_returns_empty_dict(self):
        """When broker returns empty fetched array, engine returns {} with zero synthetic records."""
        class EmptyDataServer(MockSmartConnectServer):
            def getMarketData(self, mode: str, payload: Dict[str, List[str]]) -> Dict[str, Any]:
                return {"status": True, "data": {"fetched": []}}

        engine = BatchMarketDataEngine(EmptyDataServer())
        reqs = [{"exchange": "NFO", "symboltoken": "95004"}]
        quotes = engine.get_quotes_batch(reqs)
        assert quotes == {}

    def test_demultiplexing_isolation_callers_only_receive_their_requested_tokens(self):
        """Verify Caller A only receives Token A, Caller B only receives Token B (no cross-leakage)."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)

        res_a = None
        res_b = None

        def caller_a():
            nonlocal res_a
            res_a = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": "96001"}])

        def caller_b():
            nonlocal res_b
            res_b = engine.get_quotes_batch([{"exchange": "BFO", "symboltoken": "96002"}])

        ta = threading.Thread(target=caller_a)
        tb = threading.Thread(target=caller_b)
        ta.start()
        tb.start()
        ta.join(timeout=1.0)
        tb.join(timeout=1.0)

        assert "96001" in res_a
        assert "96002" not in res_a

        assert "96002" in res_b
        assert "96001" not in res_b


# ============================================================================
# 5. PRIORITY INHERITANCE & CONCURRENCY TESTS
# ============================================================================

class TestPriorityInheritanceAndHighConcurrency:
    """Adversarially challenge priority inheritance and high concurrency coalescing."""

    def test_priority_inheritance_p4_promoted_when_coalesced_with_p1(self):
        """When a P4 caller is queued, arriving P1 caller causes chunk to inherit P1 priority."""
        server = MockSmartConnectServer()
        limiter = PrioritizedTokenBucketLimiter()
        engine = BatchMarketDataEngine(server, limiter=limiter)

        # Deplete quotes bucket down to reserved level (1.0 token)
        with limiter._buckets[EndpointCategory.QUOTES]._lock:
            limiter._buckets[EndpointCategory.QUOTES].tokens = 1.0

        p4_res = None
        p1_res = None

        def p4_caller():
            nonlocal p4_res
            p4_res = engine.get_quotes_batch(
                [{"exchange": "NFO", "symboltoken": "97001"}],
                priority=PriorityTier.P4_BACKGROUND_SCANNER,
                timeout=0.3,
            )

        def p1_caller():
            nonlocal p1_res
            time.sleep(0.005)  # Arrive within 25ms debounce window
            p1_res = engine.get_quotes_batch(
                [{"exchange": "NFO", "symboltoken": "97002"}],
                priority=PriorityTier.P1_ORDER_EXECUTION,
                timeout=0.3,
            )

        t4 = threading.Thread(target=p4_caller)
        t1 = threading.Thread(target=p1_caller)
        t4.start()
        t1.start()
        t4.join(timeout=1.0)
        t1.join(timeout=1.0)

        # Because batch inherited P1, it successfully consumed the reserved token!
        assert p4_res is not None and "97001" in p4_res
        assert p1_res is not None and "97002" in p1_res

    def test_high_concurrency_stress_50_callers_no_token_loss_or_deadlock(self):
        """50 concurrent threads requesting tokens simultaneously within debounce window."""
        server = MockSmartConnectServer()
        engine = BatchMarketDataEngine(server)

        results = {}
        threads = []

        def worker(idx):
            tok = str(98000 + idx)
            res = engine.get_quotes_batch([{"exchange": "NFO", "symboltoken": tok}], timeout=1.0)
            results[idx] = res

        for i in range(50):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert len(results) == 50
        for i in range(50):
            tok = str(98000 + i)
            assert tok in results[i], f"Token {tok} missing for caller {i}!"
            assert results[i][tok]["ltp"] > 0
