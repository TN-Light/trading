# E2E Test Infrastructure: Angel One SmartAPI Optimization & Fortification

## 1. Test Philosophy
- **Opaque-Box & Requirement-Driven**: Tests derive directly from authoritative specifications in `ORIGINAL_REQUEST.md` (header `## 2026-10-08T10:37:29Z`), `PROJECT.md`, and system integrity guidelines in `GEMINI.md`.
- **Zero Live Broker Dependency**: All broker interactions are mediated via high-fidelity in-memory mock servers (`MockSmartConnectServer`) enforcing real Angel One SmartAPI rate limits, sliding window quotas, and payload constraints without requiring internet connectivity or live API keys.
- **Strict Mathematical Invariance & Telemetry**:
  - Zero tolerance for rate limit breaches: exactly 0 HTTP 429 / AB1021 errors under continuous multi-threaded load.
  - Strict latency SLA: End-to-end signal processing and quote refresh latency < 1,000ms across active positions.
  - Proven efficiency: Outbound broker call reduction >= 60% compared to unbatched sequential baselines.
  - Zero Price Hallucination: Frozen trailing ratchets and `[STALE_DATA_ALERT]` telemetry under broker timeouts; synthetic prices strictly prohibited.
- **Methodology**: Category-Partition + Boundary Value Analysis (BVA) + Multi-Threaded Concurrency Testing + 100-Symbol / 10-Minute Workload Stress Simulation.

---

## 2. Feature Inventory & Test Mapping

| # | Feature | Requirement Source | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Real-World) |
|---|---------|-------------------|:-----------------:|:-----------------:|:---------------------:|:-------------------:|
| F1 | Endpoint-Differentiated Token Buckets | `ORIGINAL_REQUEST §R3` / `PROJECT.md #1` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F2 | 4-Tier Request Priority Scheduling | `ORIGINAL_REQUEST §R3` / `PROJECT.md #2` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F3 | Non-Blocking Mutex Sleeping | `ORIGINAL_REQUEST §R1, R3` / `PROJECT.md #3` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F4 | 50-Token Batch Coalescer & Debounce | `ORIGINAL_REQUEST §R3` / `PROJECT.md #4` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F5 | Cross-Segment Multi-Key Partitioning | `ORIGINAL_REQUEST §R3` / `PROJECT.md #5` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F6 | Unified Quote Extraction (`get_real_premium`) | `ORIGINAL_REQUEST §R3` / `PROJECT.md #6` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F7 | Shared Non-Blocking Price Micro-Cache | `ORIGINAL_REQUEST §R4` / `PROJECT.md #7` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F8 | Non-Hallucinatory Fallback & Freeze | `ORIGINAL_REQUEST §R4` / `GEMINI.md` | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F9 | Mock SmartConnect Rate-Limiting Server | `PROJECT.md #11` / E2E Track | ≥ 5 | ≥ 5 | ✓ | ✓ |
| F10 | 100-Symbol / 10-Minute Stress Harness | `ORIGINAL_REQUEST §Acceptance Criteria` | ≥ 5 | ≥ 5 | ✓ | ✓ |

---

## 3. Test Architecture

### 3.1 Architecture Overview
```
+--------------------------------------------------------------------------------------------------+
|                                    E2E TEST HARNESS ARCHITECTURE                                 |
|                                                                                                  |
|   [Worker: PositionMonitor]   [Worker: PositionTracker]   [Worker: HealthEngine]   [Worker: Scanner] |
|        (Priority 2)                (Priority 2)                (Priority 3)          (Priority 4)    |
|             │                           │                           │                     │          |
|             └───────────────────────────┼───────────────────────────┴─────────────────────┘          |
|                                         ▼                                                            |
|                        ┌──────────────────────────────────────────────┐                              |
|                        │      SharedPriceMicroCache (TTL 750ms)       │                              |
|                        │         (Atomic, Lock-Free Lookups)          │                              |
|                        └────────────────┬─────────────────────────────┘                              |
|                                         │ (Cache Miss / Stale)                                       |
|                                         ▼                                                            |
|                        ┌──────────────────────────────────────────────┐                              |
|                        │    BatchMarketDataEngine (Max 50 Tokens)     │                              |
|                        │      (25ms Debounce Window, Segments)        │                              |
|                        └────────────────┬─────────────────────────────┘                              |
|                                         ▼                                                            |
|                        ┌──────────────────────────────────────────────┐                              |
|                        │   PrioritizedTokenBucketLimiter (85% Cap)    │                              |
|                        │   P1: Orders/SL (8.5/s)  P2: Quotes (8.5/s)  │                              |
|                        │   P3: Health (8.5/s)     P4: Scanners (2.5/s)│                              |
|                        └────────────────┬─────────────────────────────┘                              |
|                                         │                                                            |
+-----------------------------------------┼------------------------------------------------------------+
                                          ▼
                         ┌──────────────────────────────────────────────┐
                         │            MockSmartConnectServer            │
                         │   - Quotes Quota: 10.0 req/s (Sliding 1s)    │
                         │   - Hist Quota:    3.0 req/s (Sliding 1s)    │
                         │   - HTTP 429 / AB1021 Emulation on Breach    │
                         │   - Max Payload: 50 Tokens Enforcement       │
                         │   - Realistic Network Latency & Jitter       │
                         └──────────────────────────────────────────────┘
```

### 3.2 Mock Infrastructure (`MockSmartConnectServer`)
Located in `prometheus/tests/mock_smartapi_server.py`:
- **Endpoint Quota Enforcement**:
  - Quotes / Market Data: 10 req/s sliding window.
  - Historical Data: 3 req/s sliding window.
  - Orders: 10 req/s sliding window.
- **Quota Exceeded Emulation**:
  - Raises or returns `{"status": False, "errorcode": "AB1021", "message": "Too many requests / Exceeding access rate"}` when calls exceed ceiling.
- **Payload Verification**:
  - Rejects payloads exceeding 50 total tokens across all exchanges (`AB1010`).
- **Telemetry Counters**:
  - Tracks total calls, endpoint calls, batch calls, tokens requested, HTTP 429/AB1021 events, and per-call timestamps.

---

## 4. Test Tier Hierarchy & Coverage Thresholds

### Tier 1: Feature Coverage (≥ 5 test cases per feature)
- **Token Bucket Replenishment & Capacity**: Continuous token regeneration, capacity bounds, initial token availability.
- **4 Priority Tiers**: Priority 1 preemption and token reservation, Priority 2 position polling, Priority 3 health checking, Priority 4 background scanner scheduling.
- **Batch Coalescer**: 25ms debounce grouping, coalescing multiple callers into single batch, demultiplexing results to individual futures.
- **Micro-Cache TTL**: Fresh read-through hits within 750ms, stale cache invalidation after 750ms, atomic concurrent dictionary updates.
- **Non-Hallucinatory Fallbacks**: Secondary feed resolution for index spot, freeze option trailing ratchet with `[STALE_DATA_ALERT]` telemetry.

### Tier 2: Boundary & Corner Cases (≥ 5 test cases per feature)
- **Zero & Empty Inputs**: 0 tokens requested, empty exchange token dictionary, empty string tokens.
- **Exact Batch Boundaries**: Exactly 50 tokens in a single request (valid single batch).
- **Chunk Splitting**: 51 tokens (must split into 50 + 1 chunks across 2 batch requests).
- **Extreme Burst & Quota Saturation**: Rapid concurrent requests saturating client bucket; client sleeps outside mutex without 429 error.
- **Broker Timeouts & Socket Errors**: Transient broker network timeout emulation; cache serves last known quote or raises clean non-fabricating error.
- **Cross-Segment Multi-Exchange**: Mixed payloads containing `NFO`, `BFO`, and `NSE` tokens in varying proportions.

### Tier 3: Cross-Feature Interactions
- **Concurrent Tri-Tier Contention**: Background scanner (P4) fetching 50 tokens + Position Monitor (P2) checking 20 positions + Trailing SL Order modification (P1).
- **Batch Coalescer + Micro-Cache Pipeline**: Multiple consumer threads query overlapping tokens; only missing/stale tokens trigger batch REST.
- **Broker Stale Feed Failover + Trailing Freeze**: When broker returns stale/timeout ticks, trailing stop remains frozen at high-water mark without synthesizing artificial prices.

### Tier 4: Real-World Workload & Concurrency Stress Harness
- **Target File**: `prometheus/tests/test_smartapi_stress_and_concurrency.py`
- **Simulation Parameters**:
  - 100 concurrent symbols/positions (10 index spots + 90 option contracts).
  - 10-minute simulated trading session (600 seconds wall-clock or accelerated deterministic virtual-time equivalent of 600s).
  - Multiple concurrent asynchronous worker threads simulating:
    * `PositionMonitor`: 50 active positions polled every 5.0s.
    * `PositionTracker`: 50 open positions polled every 5.0s.
    * `PositionHealthEngine`: 100 positions evaluated every 10.0s.
    * `LiveScanner`: 100 symbols swept every 30.0s.
    * `OrderRouter`: Periodic trailing stop order updates at Priority 1.
- **Quantitative Pass/Fail Criteria**:
  1. `server.ab1021_count == 0` and `server.http_429_count == 0` (Zero rate limit breaches).
  2. End-to-end quote refresh and processing latency P95 and P99 < 1,000ms.
  3. Total outbound broker calls reduced by ≥ 60% compared to unbatched baseline ($C_{\text{batched}} \le 0.40 \times C_{\text{baseline}}$).
  4. 0 synthetic/fabricated prices; 100% data traceability.

---

## 5. Test Directory Layout

```
prometheus/
├── tests/
│   ├── mock_smartapi_server.py                 # MockSmartConnectServer (quota & latency emulation)
│   ├── test_smartapi_stress_and_concurrency.py # Tiers 1-4 comprehensive E2E test suite & stress harness
│   ├── test_smartapi_rate_limiter.py          # Limiter unit tests (Milestone 1)
│   ├── test_smartapi_batch_engine.py          # Batch engine unit tests (Milestone 2)
│   ├── test_smartapi_micro_cache.py           # Micro-cache unit tests (Milestone 3)
│   └── test_smartapi_smartstream.py           # SmartStream hybrid tests (Milestone 4)
TEST_INFRA.md                                   # This test infrastructure specification
TEST_READY.md                                   # Test suite readiness & execution verification
```

---

## 6. Execution Command & Verification

To run the full E2E test suite:
```powershell
python -m pytest prometheus/tests/test_smartapi_stress_and_concurrency.py -v
```

To run individual tiers:
```powershell
# Tier 1: Feature Coverage
python -m pytest prometheus/tests/test_smartapi_stress_and_concurrency.py -k "Tier1" -v

# Tier 2: Boundary & Corner Cases
python -m pytest prometheus/tests/test_smartapi_stress_and_concurrency.py -k "Tier2" -v

# Tier 3: Cross-Feature Interactions
python -m pytest prometheus/tests/test_smartapi_stress_and_concurrency.py -k "Tier3" -v

# Tier 4: Real-World Stress Harness
python -m pytest prometheus/tests/test_smartapi_stress_and_concurrency.py -k "Tier4" -v
```
