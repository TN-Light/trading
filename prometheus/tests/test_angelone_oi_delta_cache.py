# ============================================================================
# PROMETHEUS — Angel One OI Delta Cache Unit Tests
# ============================================================================
"""
Tests for the in-memory ContractOISnapshot cache in AngelOneOptionChain.
Verifies:
1. ContractOISnapshot dataclass properties (session baseline vs poll delta).
2. Initial poll sets baseline with delta_oi_session = 0 and delta_oi_poll = 0.
3. Subsequent polls with positive/negative OI shift compute accurate session and poll deltas.
4. Date roll flushes the cache cleanly and establishes a new session baseline.
5. Thread safety under concurrent multi-threaded fetches.
6. get_real_premium attaches oi_change and delta_oi using the snapshot cache.
7. Downstream OIAnalyzer receives non-zero oi_change to compute commitment_ratio > 0
   and trigger institutional buildup/unwinding signals.
8. reset_oi_cache clears all snapshots.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import pandas as pd
import pytest

from prometheus.data.angelone_options import AngelOneOptionChain, ContractOISnapshot
from prometheus.signals.oi_analyzer import OIAnalyzer


class _FakeSmartConnect:
    """Mock SmartConnect object providing configurable getMarketData and ltpData."""

    def __init__(self):
        self.market_data_responses = []
        self.ltp_data_map = {}
        self.market_data_calls = []

    def getMarketData(self, mode, payload):
        self.market_data_calls.append((mode, payload))
        if self.market_data_responses:
            return self.market_data_responses.pop(0)
        # Default empty response
        return {"status": True, "data": {"fetched": []}}

    def ltpData(self, seg, tradingsymbol, token):
        ltp = self.ltp_data_map.get(str(token), 100.0)
        return {
            "status": True,
            "data": {
                "ltp": ltp,
                "tradingsymbol": tradingsymbol,
                "symboltoken": str(token),
            },
        }

    def optionGreek(self, params):
        return {
            "status": True,
            "data": {
                "delta": 0.5,
                "gamma": 0.002,
                "theta": -5.0,
                "vega": 12.0,
                "impliedVolatility": 15.5,
            },
        }


class _FakeFetcher:
    """Mock AngelOneFetcher."""

    def __init__(self, obj):
        self._obj = obj

    def _ensure_connected(self):
        return True

    @property
    def obj(self):
        return self._obj


@pytest.fixture
def fake_chain():
    """Create an AngelOneOptionChain instance with fake session."""
    obj = _FakeSmartConnect()
    fetcher = _FakeFetcher(obj)
    chain = AngelOneOptionChain(fetcher)
    chain._min_interval = 0.0  # Disable rate-limiting sleeps for fast unit tests
    return chain, obj


# ============================================================================
# 1. ContractOISnapshot Dataclass Unit Tests
# ============================================================================

def test_contract_oi_snapshot_properties():
    """Verify ContractOISnapshot baseline, poll tracking, and delta calculations."""
    snap = ContractOISnapshot(
        token="35001",
        tradingsymbol="NIFTY26OCT24000CE",
        session_baseline_oi=50000,
        prev_poll_oi=50000,
        current_oi=50000,
        last_poll_time=time.time(),
        poll_count=1,
    )

    # Initial state: no change
    assert snap.delta_oi_session == 0
    assert snap.delta_oi_poll == 0
    assert snap.poll_count == 1

    # Poll 2: OI increases to 55,000 (+5,000)
    snap.update(55000)
    assert snap.current_oi == 55000
    assert snap.prev_poll_oi == 50000
    assert snap.session_baseline_oi == 50000
    assert snap.delta_oi_session == 5000
    assert snap.delta_oi_poll == 5000
    assert snap.poll_count == 2

    # Poll 3: OI increases further to 58,000 (+3,000)
    snap.update(58000)
    assert snap.current_oi == 58000
    assert snap.prev_poll_oi == 55000
    assert snap.session_baseline_oi == 50000
    assert snap.delta_oi_session == 8000
    assert snap.delta_oi_poll == 3000
    assert snap.poll_count == 3

    # Poll 4: OI decreases to 52,000 (-6,000 unwinding)
    snap.update(52000)
    assert snap.current_oi == 52000
    assert snap.prev_poll_oi == 58000
    assert snap.session_baseline_oi == 50000
    assert snap.delta_oi_session == 2000
    assert snap.delta_oi_poll == -6000
    assert snap.poll_count == 4

    # Poll 5: OI falls below baseline to 45,000 (-7,000)
    snap.update(45000)
    assert snap.current_oi == 45000
    assert snap.prev_poll_oi == 52000
    assert snap.session_baseline_oi == 50000
    assert snap.delta_oi_session == -5000
    assert snap.delta_oi_poll == -7000
    assert snap.poll_count == 5


# ============================================================================
# 2. Initial Poll Baseline Setting (Delta = 0)
# ============================================================================

def test_initial_poll_sets_baseline_with_zero_deltas(fake_chain):
    """Verify first poll of day initializes snapshot baseline with delta_oi = 0."""
    chain, obj = fake_chain

    contracts = [
        {"symboltoken": "35001", "tradingsymbol": "NIFTY26OCT24000CE", "strike": 24000, "option_type": "CE", "expiry": "2026-10-29"},
        {"symboltoken": "35002", "tradingsymbol": "NIFTY26OCT24000PE", "strike": 24000, "option_type": "PE", "expiry": "2026-10-29"},
    ]

    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [
                    {
                        "symbolToken": "35001",
                        "tradingSymbol": "NIFTY26OCT24000CE",
                        "ltp": 150.0,
                        "open": 140.0,
                        "high": 160.0,
                        "low": 135.0,
                        "close": 150.0,
                        "tradeVolume": 25000,
                        "opnInterest": 100000,
                        "bestBidPrice": 149.5,
                        "bestAskPrice": 150.5,
                    },
                    {
                        "symbolToken": "35002",
                        "tradingSymbol": "NIFTY26OCT24000PE",
                        "ltp": 120.0,
                        "open": 125.0,
                        "high": 130.0,
                        "low": 115.0,
                        "close": 120.0,
                        "tradeVolume": 30000,
                        "opnInterest": 80000,
                        "bestBidPrice": 119.5,
                        "bestAskPrice": 120.5,
                    },
                ]
            },
        }
    ]

    df = chain.fetch_market_data(contracts, underlying="NIFTY", trading_date="2026-10-01")

    assert len(df) == 2
    assert "oi_change" in df.columns
    assert "delta_oi" in df.columns

    ce_row = df[df["symboltoken"] == "35001"].iloc[0]
    pe_row = df[df["symboltoken"] == "35002"].iloc[0]

    # First poll: both cumulative and incremental delta must be exactly 0
    assert ce_row["oi"] == 100000
    assert ce_row["oi_change"] == 0
    assert ce_row["delta_oi"] == 0

    assert pe_row["oi"] == 80000
    assert pe_row["oi_change"] == 0
    assert pe_row["delta_oi"] == 0

    # Verify snapshot cache state
    snap_ce = chain.get_oi_snapshot("35001")
    assert snap_ce is not None
    assert snap_ce.session_baseline_oi == 100000
    assert snap_ce.prev_poll_oi == 100000
    assert snap_ce.current_oi == 100000
    assert snap_ce.poll_count == 1

    snap_pe = chain.get_oi_snapshot("35002")
    assert snap_pe is not None
    assert snap_pe.session_baseline_oi == 80000
    assert snap_pe.prev_poll_oi == 80000
    assert snap_pe.current_oi == 80000
    assert snap_pe.poll_count == 1


# ============================================================================
# 3. Subsequent Polls with Positive/Negative Shifts
# ============================================================================

def test_subsequent_polls_with_positive_and_negative_shifts(fake_chain):
    """Verify subsequent polls calculate accurate cumulative and incremental deltas."""
    chain, obj = fake_chain

    contracts = [
        {"symboltoken": "35001", "tradingsymbol": "NIFTY26OCT24000CE", "strike": 24000, "option_type": "CE", "expiry": "2026-10-29"},
    ]

    # Poll 1: baseline 100,000
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "ltp": 150.0,
                    "tradeVolume": 10000,
                    "opnInterest": 100000,
                }]
            },
        }
    ]
    df1 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df1.iloc[0]["oi_change"] == 0
    assert df1.iloc[0]["delta_oi"] == 0

    # Poll 2: OI rises to 110,000 (+10,000 session, +10,000 poll)
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "ltp": 155.0,
                    "tradeVolume": 25000,
                    "opnInterest": 110000,
                }]
            },
        }
    ]
    df2 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df2.iloc[0]["oi"] == 110000
    assert df2.iloc[0]["oi_change"] == 10000
    assert df2.iloc[0]["delta_oi"] == 10000

    # Poll 3: OI rises further to 118,000 (+18,000 session, +8,000 poll)
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "ltp": 160.0,
                    "tradeVolume": 40000,
                    "opnInterest": 118000,
                }]
            },
        }
    ]
    df3 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df3.iloc[0]["oi"] == 118000
    assert df3.iloc[0]["oi_change"] == 18000
    assert df3.iloc[0]["delta_oi"] == 8000

    # Poll 4: OI unwinds to 105,000 (+5,000 session, -13,000 poll)
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "ltp": 145.0,
                    "tradeVolume": 65000,
                    "opnInterest": 105000,
                }]
            },
        }
    ]
    df4 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df4.iloc[0]["oi"] == 105000
    assert df4.iloc[0]["oi_change"] == 5000
    assert df4.iloc[0]["delta_oi"] == -13000

    # Poll 5: OI falls below session open to 92,000 (-8,000 session, -13,000 poll)
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "ltp": 130.0,
                    "tradeVolume": 90000,
                    "opnInterest": 92000,
                }]
            },
        }
    ]
    df5 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df5.iloc[0]["oi"] == 92000
    assert df5.iloc[0]["oi_change"] == -8000
    assert df5.iloc[0]["delta_oi"] == -13000

    snap = chain.get_oi_snapshot("35001")
    assert snap.poll_count == 5
    assert snap.session_baseline_oi == 100000
    assert snap.prev_poll_oi == 105000
    assert snap.current_oi == 92000


# ============================================================================
# 4. Date Roll Cache Invalidation
# ============================================================================

def test_date_roll_flushes_cache_cleanly(fake_chain):
    """Verify that date change flushes the snapshot cache and starts new baseline."""
    chain, obj = fake_chain

    contracts = [
        {"symboltoken": "35001", "tradingsymbol": "NIFTY26OCT24000CE", "strike": 24000, "option_type": "CE", "expiry": "2026-10-29"},
    ]

    # Day 1 Poll 1: Baseline 100,000 on 2026-10-01
    obj.market_data_responses = [
        {"status": True, "data": {"fetched": [{"symbolToken": "35001", "opnInterest": 100000, "tradeVolume": 1000}]}}
    ]
    df_d1_p1 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df_d1_p1.iloc[0]["oi_change"] == 0

    # Day 1 Poll 2: Increases to 110,000 on 2026-10-01
    obj.market_data_responses = [
        {"status": True, "data": {"fetched": [{"symbolToken": "35001", "opnInterest": 110000, "tradeVolume": 5000}]}}
    ]
    df_d1_p2 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df_d1_p2.iloc[0]["oi_change"] == 10000
    assert len(chain.get_all_oi_snapshots()) == 1

    # Day 2 Roll: New date 2026-10-02, overnight OI is 112,000
    obj.market_data_responses = [
        {"status": True, "data": {"fetched": [{"symbolToken": "35001", "opnInterest": 112000, "tradeVolume": 200}]}}
    ]
    df_d2_p1 = chain.fetch_market_data(contracts, trading_date="2026-10-02")

    # On new day, 112,000 becomes the new session baseline, deltas must be 0!
    assert df_d2_p1.iloc[0]["oi"] == 112000
    assert df_d2_p1.iloc[0]["oi_change"] == 0
    assert df_d2_p1.iloc[0]["delta_oi"] == 0

    snap_d2 = chain.get_oi_snapshot("35001")
    assert snap_d2.session_baseline_oi == 112000
    assert snap_d2.current_oi == 112000
    assert snap_d2.poll_count == 1

    # Day 2 Poll 2: Increases to 115,000 (+3,000 above Day 2 baseline)
    obj.market_data_responses = [
        {"status": True, "data": {"fetched": [{"symbolToken": "35001", "opnInterest": 115000, "tradeVolume": 2000}]}}
    ]
    df_d2_p2 = chain.fetch_market_data(contracts, trading_date="2026-10-02")
    assert df_d2_p2.iloc[0]["oi_change"] == 3000
    assert df_d2_p2.iloc[0]["delta_oi"] == 3000
    assert snap_d2.poll_count == 2


# ============================================================================
# 5. Thread Safety Under Concurrent Fetches
# ============================================================================

def test_thread_safety_under_concurrent_fetches(fake_chain):
    """Verify thread-safe updates to snapshot cache with concurrent worker threads."""
    chain, _ = fake_chain

    tokens = [f"token_{i}" for i in range(10)]
    polls_per_thread = 25
    num_threads = 8

    errors = []

    def worker(thread_idx: int):
        try:
            for step in range(polls_per_thread):
                token = tokens[step % len(tokens)]
                # Unique deterministic OI value per update
                oi_val = 10000 + (thread_idx * 1000) + step * 50
                chain._update_oi_snapshot(
                    token=token,
                    tradingsymbol=f"SYM_{token}",
                    current_oi=oi_val,
                    trading_date="2026-10-01",
                )
        except Exception as ex:
            errors.append(ex)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent updates produced exceptions: {errors}"

    snapshots = chain.get_all_oi_snapshots()
    assert len(snapshots) == len(tokens)

    total_polls = sum(snap.poll_count for snap in snapshots.values())
    expected_polls = num_threads * polls_per_thread
    assert total_polls == expected_polls, f"Expected {expected_polls} polls, got {total_polls}"

    # Verify each snapshot is mathematically consistent
    for token, snap in snapshots.items():
        assert snap.delta_oi_session == snap.current_oi - snap.session_baseline_oi
        assert snap.delta_oi_poll == snap.current_oi - snap.prev_poll_oi
        assert snap.poll_count >= 1


# ============================================================================
# 6. Single-Contract Lookup: get_real_premium
# ============================================================================

def test_get_real_premium_attaches_oi_change_and_delta_oi(fake_chain):
    """Verify get_real_premium populates oi, oi_change, and delta_oi via snapshot cache."""
    chain, obj = fake_chain

    # Mock contract discovery cache so get_real_premium finds the target
    chain._token_cache["NIFTY"] = [
        {"tradingsymbol": "NIFTY26OCT24000CE", "symboltoken": "35001", "strike": 24000, "option_type": "CE", "expiry": "2026-10-29"}
    ]
    chain._cache_date = "2026-10-01"

    obj.ltp_data_map["35001"] = 155.0

    # First call: baseline 50,000
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "bestBidPrice": 154.5,
                    "bestAskPrice": 155.5,
                    "tradeVolume": 10000,
                    "opnInterest": 50000,
                }]
            },
        }
    ]

    p1 = chain.get_real_premium("NIFTY 50", 24000, "CE", expiry="2026-10-29")
    assert p1 is not None
    assert p1["oi"] == 50000
    assert p1["oi_change"] == 0
    assert p1["delta_oi"] == 0
    assert p1["volume"] == 10000

    # Second call: OI jumps to 54,000 (+4,000)
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "bestBidPrice": 156.0,
                    "bestAskPrice": 157.0,
                    "tradeVolume": 25000,
                    "opnInterest": 54000,
                }]
            },
        }
    ]

    p2 = chain.get_real_premium("NIFTY 50", 24000, "CE", expiry="2026-10-29")
    assert p2 is not None
    assert p2["oi"] == 54000
    assert p2["oi_change"] == 4000
    assert p2["delta_oi"] == 4000
    assert p2["volume"] == 25000

    # Third call: OI unwinds to 51,500 (-2,500 poll, +1,500 session)
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [{
                    "symbolToken": "35001",
                    "bestBidPrice": 152.0,
                    "bestAskPrice": 153.0,
                    "tradeVolume": 45000,
                    "opnInterest": 51500,
                }]
            },
        }
    ]

    p3 = chain.get_real_premium("NIFTY 50", 24000, "CE", expiry="2026-10-29")
    assert p3 is not None
    assert p3["oi"] == 51500
    assert p3["oi_change"] == 1500
    assert p3["delta_oi"] == -2500
    assert p3["volume"] == 45000


# ============================================================================
# 7. Downstream OIAnalyzer Integration & Buildup/Unwinding Signals
# ============================================================================

def test_downstream_oi_analyzer_receives_oi_change_and_computes_commitment_ratio(fake_chain):
    """
    End-to-end integration test:
    Verify that option chain DataFrame with real oi_change produces true non-zero
    commitment_ratio and triggers institutional buildup/unwinding signals.
    """
    chain, obj = fake_chain
    analyzer = OIAnalyzer()

    spot = 24000.0
    contracts = [
        {"symboltoken": "1001", "tradingsymbol": "NIFTY24000CE", "strike": 24000, "option_type": "CE", "expiry": "2026-10-29"},
        {"symboltoken": "1002", "tradingsymbol": "NIFTY24000PE", "strike": 24000, "option_type": "PE", "expiry": "2026-10-29"},
    ]

    # Poll 1: Initial baseline
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [
                    {"symbolToken": "1001", "ltp": 150.0, "tradeVolume": 20000, "opnInterest": 100000},
                    {"symbolToken": "1002", "ltp": 140.0, "tradeVolume": 20000, "opnInterest": 100000},
                ]
            },
        }
    ]
    df1 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    res1 = analyzer.analyze(df1, spot_price=spot)

    # First poll: baseline OI means delta = 0, commitment_ratio = 0
    assert res1["metrics"]["commitment_ratio"] == 0.0

    # Poll 2: Heavy Call OI buildup (+20,000) and moderate Put OI buildup (+10,000)
    # Total volume = 30,000 (CE) + 30,000 (PE) = 60,000
    # Expected commitment ratio = (20,000 + 10,000) / 60,000 = 0.500
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [
                    {"symbolToken": "1001", "ltp": 145.0, "tradeVolume": 30000, "opnInterest": 120000},
                    {"symbolToken": "1002", "ltp": 145.0, "tradeVolume": 30000, "opnInterest": 110000},
                ]
            },
        }
    ]
    df2 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    res2 = analyzer.analyze(df2, spot_price=spot)

    # Commitment ratio is non-zero and mathematically exact
    assert res2["metrics"]["commitment_ratio"] == 0.500
    assert res2["metrics"]["commitment_ratio"] > 0.0

    # Verify institutional buildup signal was triggered
    sig_types = [s.signal_type for s in res2["signals"]]
    assert "call_oi_buildup" in sig_types
    call_sig = next(s for s in res2["signals"] if s.signal_type == "call_oi_buildup")
    assert call_sig.direction == "bearish"
    assert "Call OI +20,000 near ATM" in call_sig.details

    # Poll 3: Put unwinding (-15,000 shift on PE)
    # Baseline for PE was 100,000. If current PE OI drops to 85,000, oi_change = -15,000.
    obj.market_data_responses = [
        {
            "status": True,
            "data": {
                "fetched": [
                    {"symbolToken": "1001", "ltp": 145.0, "tradeVolume": 35000, "opnInterest": 120000},
                    {"symbolToken": "1002", "ltp": 130.0, "tradeVolume": 35000, "opnInterest": 85000},
                ]
            },
        }
    ]
    df3 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    res3 = analyzer.analyze(df3, spot_price=spot)

    sig_types_3 = [s.signal_type for s in res3["signals"]]
    assert "put_oi_unwinding" in sig_types_3
    put_sig = next(s for s in res3["signals"] if s.signal_type == "put_oi_unwinding")
    assert put_sig.direction == "bearish"
    assert "Put OI -15,000 near ATM" in put_sig.details


# ============================================================================
# 8. Reset Cache
# ============================================================================

def test_reset_oi_cache(fake_chain):
    """Verify reset_oi_cache flushes all snapshots and resets date."""
    chain, _ = fake_chain

    chain._update_oi_snapshot("123", "SYM123", 5000, trading_date="2026-10-01")
    assert len(chain.get_all_oi_snapshots()) == 1

    chain.reset_oi_cache()
    assert len(chain.get_all_oi_snapshots()) == 0
    assert chain.get_oi_snapshot("123") is None
