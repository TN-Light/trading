# Project: Angel One SmartAPI Rate Limit Optimization and Fortification

## Architecture
Prometheus quantitative execution and telemetry architecture fortified against Angel One SmartAPI rate limits (HTTP 429 / AB1021). The architecture decouples outbound calls into endpoint-specific token buckets at 85% broker ceilings, coalesces multi-token market queries into 50-token `getMarketData("FULL")` batches, shares atomic tick snapshots across consumers via a sub-second non-blocking micro-cache, and supports an optional hardened SmartStream WebSocket accelerator with seamless 1,500ms REST failover.

```
+------------------------------------------------------------------------------------------------+
|                                    PROMETHEUS TRADING SYSTEM                                   |
|                                                                                                |
|   [PositionMonitor]       [PositionTracker]       [PositionHealthEngine]       [LiveScanner]   |
|      (Priority 2)            (Priority 2)              (Priority 3)             (Priority 4)   |
|           │                       │                         │                        │         |
|           └───────────────┬───────┴─────────────────────────┴────────────────────────┘         |
|                           ▼                                                                    |
|           ┌──────────────────────────────────────────────┐                                     |
|           │         SharedPriceMicroCache (TTL 750ms)     │ <─── [SmartStreamHybridClient]     |
|           │ (Thread-Safe, Non-Blocking, Stale-Data Alert)│      (Active Positions Push Feed,   |
|           └───────────────────────┬──────────────────────┘       Auto REST Failover @ 1500ms)  |
|                                   │ (Cache Miss / Stale)                                       |
|                                   ▼                                                            |
|           ┌──────────────────────────────────────────────┐                                     |
|           │      BatchMarketDataEngine (Max 50 Tokens)   │                                     |
|           │   (25ms Debounce Coalescer, NFO/BFO Split)   │                                     |
|           └───────────────────────┬──────────────────────┘                                     |
|                                   ▼                                                            |
|           ┌──────────────────────────────────────────────┐                                     |
|           │     PrioritizedTokenBucketLimiter (85% Cap)  │                                     |
|           │   P1: Orders/SL (8.5/s)  P2: Positions (8.5/s)│                                     |
|           │   P3: Health (8.5/s)     P4: Scanners (2.5/s) │                                     |
|           │      (Lock-Free Sleep Outside Mutex)         │                                     |
|           └───────────────────────┬──────────────────────┘                                     |
|                                   │                                                            |
+-----------------------------------┼────────────────────────────────────────────────────────────+
                                    ▼
                  [Angel One SmartAPI Gateway / Broker]
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Endpoint-Differentiated Token Buckets | Independent token buckets for Historical (2.5 req/s), Quotes (8.5 req/s), Orders (8.5 req/s) at 85% broker quota ceilings | M1 | Survey E1, E3 |
| 2 | 4-Tier Request Priority Scheduling | Strict prioritization: P1 Orders/SL > P2 Position Monitor > P3 8-Pillar Health > P4 Background Scanners | M1 | Survey E1, E3 |
| 3 | Non-Blocking Mutex Sleeping | Relocate all rate-limit sleep calls outside mutex locks to eliminate thread serialization | M1 | Survey E1, E3 |
| 4 | Token Batch Coalescer (50-Token Chunks) | Asynchronous 25ms debounce queue coalescing single/multi-token queries into 50-token `getMarketData("FULL")` batches | M2 | Survey E1, E3 |
| 5 | Cross-Segment Multi-Key Batching | Partition batched tokens across `{"NFO": [...], "BFO": [...], "NSE": [...]}` within the 50-token payload limit | M2 | Survey E3 |
| 6 | Refactored `get_real_premium` | Eliminate redundant `ltpData` single-token calls; extract LTP, bid, ask, and OI directly from full market batch response | M2 | Survey E1, E3 |
| 7 | Shared Non-Blocking Price Micro-Cache | Thread-safe in-memory cache with sub-second TTL (750ms) serving `PositionMonitor`, `PositionTracker`, `PositionHealthEngine`, and `LiveBridge` | M3 | Survey E1, E3 |
| 8 | Non-Hallucinatory Spot & Option Fallbacks | Secondary feed fallback for index spot (Kite -> NSE Direct Web Feed); freeze option trailing stops with `[STALE_DATA_ALERT]` telemetry under broker timeout | M3 | Survey E1, E3, GEMINI.md |
| 9 | Hardened SmartStream WebSocket Client | Wrap `SmartWebSocketV2`: fix buy/sell array inversion, add `ping_timeout=5s`, robust multi-retry reconnect, dynamic JWT refresh | M4 | Survey E2 |
| 10 | Hybrid Streamer with 1,500ms REST Failover | Stream active positions into micro-cache; automatically fall back to Priority 2 Batch REST if tick timestamp exceeds 1,500ms | M4 | Survey E2 |
| 11 | Mock SmartAPI Rate-Limiting Server | Test harness mock simulating official Angel One rate limit ceilings (3 req/s historical, 10 req/s quotes, HTTP 429 / AB1021) | E2E Track | Survey E3 |
| 12 | 100-Symbol / 10-Minute Concurrency Stress Harness | Stress test simulating 100 concurrent symbols/positions over 10 minutes asserting 0 rate-limit errors, P95/P99 latency <1,000ms, and call reduction >=60% | E2E Track | Survey E3, ORIGINAL_REQUEST |
| 13 | Complete Repository Test Verification | 100% pass rate across `prometheus/tests` and `prometheus/pipeline_tests` (baseline 510 tests) | M5 | Survey E1, E3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Prioritized Token-Bucket Rate Limiter | Implement `PrioritizedTokenBucketLimiter` with 3 endpoint buckets (Historical, Quotes, Orders) capped at 85%, 4 priority tiers, lock-free sleeping outside mutex, and integrate into `angelone_fetcher.py` and `angelone_options.py`. | None | DONE |
| M2 | Unified 50-Token Batch Engine | Implement `BatchMarketDataEngine` with 25ms debounce coalescer, up to 50 tokens per `getMarketData("FULL")` call, multi-segment NFO/BFO partitioning, and refactor `get_real_premium` to use batch data. | M1 | IN_PROGRESS |
| M3 | Shared Non-Blocking Price Micro-Cache & Non-Hallucinatory Fallback | Implement `SharedPriceMicroCache` (750ms TTL) thread-safe read-optimized cache, wire into `PositionMonitor`, `LiveBridge`, and `PositionHealthEngine`; enforce spot secondary feed and option trailing freeze with `[STALE_DATA_ALERT]`. | M2 | PLANNED |
| M4 | SmartStream WebSocket Hybrid Streamer & Hardened Failover | Implement `SmartStreamHybridClient` fixing SDK bugs (inverted depth, ping_timeout, reconnect loop, dynamic JWT), streaming active positions to micro-cache with 1,500ms automatic REST failover. | M3 | PLANNED |
| M5 | Final Milestone: 100% E2E Test Suite Pass & Adversarial Hardening | Phase 1: Pass 100% of E2E test suite (Tiers 1-4) published by E2E Testing Track. Phase 2: Tier 5 white-box adversarial coverage hardening and full repository test verification (510+ tests). | M1, M2, M3, M4, E2E Track | PLANNED |

## E2E Testing Track Scope
Independent parallel track deriving tests from requirements:
- **Test Infrastructure**: `MockSmartConnectServer` with rate-limiting enforcement (3 req/s historical, 10 req/s quotes, HTTP 429 emulation).
- **Tier 1 (Feature Coverage, >=5 per feature)**: Rate limiter token replenishment, priority preemption, batch chunking, cache TTL expiry, fallback telemetry.
- **Tier 2 (Boundary & Corner Cases, >=5 per feature)**: Empty batches, exactly 50 tokens, 51 tokens (chunk splitting), broker network timeouts, mixed NFO/BFO segments, zero-division/invalid tokens.
- **Tier 3 (Cross-Feature Combinations)**: Concurrent scanner background fetches (P4) colliding with open position monitoring (P2) and trailing stop modification (P1).
- **Tier 4 (Real-World Application Scenarios)**: 100 concurrent symbols/positions simulated over 10 minutes asserting 0 HTTP 429/AB1021 errors, latency <1,000ms, and call reduction >=60%.
- **Publishes**: `TEST_INFRA.md` and `TEST_READY.md`.

## Interface Contracts

### Contract 1: Rate Limiter (`PrioritizedTokenBucketLimiter`)
```python
class PriorityTier(IntEnum):
    P1_ORDER_EXECUTION = 1
    P2_POSITION_MONITOR = 2
    P3_HEALTH_EVALUATION = 3
    P4_BACKGROUND_SCANNER = 4

class EndpointCategory(Enum):
    HISTORICAL = "historical"  # 2.5 req/s (broker limit: 3.0 req/s)
    QUOTES = "quotes"          # 8.5 req/s (broker limit: 10.0 req/s)
    ORDERS = "orders"          # 8.5 req/s (broker limit: 10.0 req/s)

class PrioritizedTokenBucketLimiter:
    def acquire(self, endpoint: EndpointCategory, priority: PriorityTier, timeout: float = 10.0) -> bool:
        """Acquire permission to dispatch a request. Sleeps outside mutex. Returns True if acquired, False on timeout."""
        ...
```

### Contract 2: Batch Engine (`BatchMarketDataEngine`)
```python
class BatchMarketDataEngine:
    def get_quotes_batch(
        self,
        requests: List[Dict[str, str]], # [{"exchange": "NFO", "tradingsymbol": "...", "symboltoken": "..."}]
        priority: PriorityTier = PriorityTier.P2_POSITION_MONITOR,
        timeout: float = 2.0
    ) -> Dict[str, Dict[str, Any]]:
        """Coalesces requests into <=50 token chunks, calls getMarketData('FULL'), returns token-keyed quote dict."""
        ...
```

### Contract 3: Price Micro-Cache (`SharedPriceMicroCache`)
```python
@dataclass
class MicroCacheEntry:
    price: float
    bid: float
    ask: float
    open_interest: int
    volume: int
    timestamp: float
    source: str # "websocket" | "batch_rest" | "fallback"
    is_stale: bool

class SharedPriceMicroCache:
    def get(self, token: str, max_age: float = 0.750) -> Optional[MicroCacheEntry]:
        """Returns non-stale cached quote if age <= max_age, else None."""
        ...
    def put(self, token: str, entry: MicroCacheEntry) -> None:
        """Atomically updates cached entry."""
        ...
    def get_or_fetch(self, tokens: List[str], fetch_func: Callable) -> Dict[str, MicroCacheEntry]:
        """Returns cached entries; fetches missing/stale entries in a single batch."""
        ...
```

### Contract 4: Non-Hallucinatory Fallback (`StaleDataAlert`)
```python
class FallbackPolicy:
    @staticmethod
    def handle_option_stale(symbol: str, token: str, last_known_price: float, age_seconds: float) -> None:
        """Logs [STALE_DATA_ALERT], freezes trailing ratchets, keeps hard exchange SL intact without price fabrication."""
        ...
```

## Code Layout
- `prometheus/data/rate_limiter.py` — PrioritizedTokenBucketLimiter implementation
- `prometheus/data/batch_engine.py` — BatchMarketDataEngine implementation
- `prometheus/data/price_micro_cache.py` — SharedPriceMicroCache implementation
- `prometheus/data/smartstream_streamer.py` — Hardened SmartStream WebSocket hybrid client
- `prometheus/data/angelone_fetcher.py` — Integrated with rate limiter
- `prometheus/data/angelone_options.py` — Integrated with batch engine & micro-cache
- `prometheus/execution/position_monitor.py` — Uses micro-cache & batch quotes
- `prometheus/paper_executor/live_bridge.py` — Uses micro-cache & batch quotes
- `prometheus/tests/test_smartapi_rate_limiter.py` — Limiter unit tests
- `prometheus/tests/test_smartapi_batch_engine.py` — Batch engine unit tests
- `prometheus/tests/test_smartapi_micro_cache.py` — Micro-cache unit tests
- `prometheus/tests/test_smartapi_smartstream.py` — SmartStream hybrid tests
- `prometheus/tests/test_smartapi_stress_and_concurrency.py` — 100-symbol stress test harness
