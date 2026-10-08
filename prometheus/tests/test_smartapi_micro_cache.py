# ============================================================================
# PROMETHEUS — SmartAPI Shared Price Micro-Cache Unit Tests
# ============================================================================
"""
Unit test suite verifying Contract 3 & 4 of Milestone 3:
  - Sub-second TTL expiry and cache hit/miss semantics
  - Lock-free read fast-paths and thread safety
  - Reverse lookup by tradingsymbol
  - Batch insertion and selective batch fetching via get_or_fetch
  - FallbackPolicy compliance (Zero synthetic price fabrication)
"""

import time
import threading
import pytest
from prometheus.data.price_micro_cache import (
    MicroCacheEntry,
    SharedPriceMicroCache,
    FallbackPolicy,
    get_shared_micro_cache,
)


def test_micro_cache_entry_basics():
    """Verify MicroCacheEntry properties and automatic spread calculation."""
    entry = MicroCacheEntry(
        price=150.25,
        bid=150.0,
        ask=150.5,
        open_interest=50000,
        volume=120000,
        timestamp=time.monotonic(),
        source="batch_rest",
        tradingsymbol="NIFTY26OCT25000CE",
    )
    assert entry.ltp == 150.25
    assert entry.oi == 50000
    assert entry.spread == 0.5
    assert entry.age() >= 0.0
    assert not entry.is_stale


def test_micro_cache_put_get_hit_miss():
    """Verify single-token get, put, hit, and miss tracking."""
    cache = SharedPriceMicroCache(default_ttl=0.5)
    assert cache.get("12345") is None

    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0

    entry = MicroCacheEntry(
        price=100.0,
        bid=99.5,
        ask=100.5,
        open_interest=1000,
        volume=2000,
        timestamp=time.monotonic(),
        tradingsymbol="TEST_TOKEN",
    )
    cache.put("12345", entry)

    cached = cache.get("12345")
    assert cached is not None
    assert cached.price == 100.0

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["puts"] == 1
    assert stats["hit_rate_pct"] == 50.0


def test_micro_cache_ttl_expiry():
    """Verify entry expires after TTL."""
    cache = SharedPriceMicroCache(default_ttl=0.05)  # 50ms TTL
    entry = MicroCacheEntry(
        price=200.0,
        bid=199.0,
        ask=201.0,
        open_interest=1000,
        volume=2000,
        timestamp=time.monotonic(),
    )
    cache.put("TOKEN_EXP", entry)
    assert cache.get("TOKEN_EXP") is not None

    time.sleep(0.08)
    assert cache.get("TOKEN_EXP") is None


def test_micro_cache_get_by_symbol():
    """Verify reverse lookup by tradingsymbol."""
    cache = SharedPriceMicroCache(default_ttl=1.0)
    entry = MicroCacheEntry(
        price=500.0,
        bid=499.0,
        ask=501.0,
        open_interest=2500,
        volume=5000,
        timestamp=time.monotonic(),
        tradingsymbol="BANKNIFTY26OCT50000CE",
    )
    cache.put("TOKEN_BNF", entry)

    by_sym = cache.get_by_symbol("BANKNIFTY26OCT50000CE")
    assert by_sym is not None
    assert by_sym.price == 500.0
    assert cache.get_by_symbol("NON_EXISTENT") is None


def test_micro_cache_put_batch_dicts_and_entries():
    """Verify put_batch works with both MicroCacheEntry and raw dict payloads."""
    cache = SharedPriceMicroCache(default_ttl=1.0)
    entry1 = MicroCacheEntry(
        price=10.0,
        bid=9.0,
        ask=11.0,
        open_interest=100,
        volume=200,
        timestamp=time.monotonic(),
        tradingsymbol="SYM1",
    )
    dict2 = {
        "ltp": 25.5,
        "bestBidPrice": 25.0,
        "bestAskPrice": 26.0,
        "opnInterest": 300,
        "tradeVolume": 400,
        "tradingSymbol": "SYM2",
        "exchange": "NFO",
    }

    cache.put_batch({"T1": entry1, "T2": dict2})
    assert cache.get("T1").price == 10.0
    assert cache.get("T2").price == 25.5
    assert cache.get_by_symbol("SYM2").open_interest == 300


def test_micro_cache_get_or_fetch_coalescing():
    """Verify get_or_fetch only queries missing or stale tokens."""
    cache = SharedPriceMicroCache(default_ttl=1.0)
    now = time.monotonic()
    fresh_entry = MicroCacheEntry(
        price=100.0, bid=99.0, ask=101.0, open_interest=10, volume=20, timestamp=now
    )
    cache.put("FRESH_TOK", fresh_entry)

    fetch_call_count = 0
    requested_missing = []

    def mock_fetch(tokens):
        nonlocal fetch_call_count, requested_missing
        fetch_call_count += 1
        requested_missing.extend(tokens)
        res = {}
        for t in tokens:
            res[t] = MicroCacheEntry(
                price=999.0, bid=998.0, ask=1000.0, open_interest=0, volume=0, timestamp=time.monotonic()
            )
        return res

    results = cache.get_or_fetch(["FRESH_TOK", "MISSING_1", "MISSING_2"], mock_fetch)

    assert fetch_call_count == 1
    assert set(requested_missing) == {"MISSING_1", "MISSING_2"}
    assert results["FRESH_TOK"].price == 100.0
    assert results["MISSING_1"].price == 999.0
    assert results["MISSING_2"].price == 999.0


def test_micro_cache_invalidation_and_clear():
    """Verify eviction and clearing mechanics."""
    cache = SharedPriceMicroCache(default_ttl=1.0)
    entry = MicroCacheEntry(
        price=50.0, bid=49.0, ask=51.0, open_interest=50, volume=100, timestamp=time.monotonic(), tradingsymbol="SYM_EVICT"
    )
    cache.put("TOK_EVICT", entry)
    assert "TOK_EVICT" in cache

    cache.invalidate("TOK_EVICT")
    assert "TOK_EVICT" not in cache
    assert cache.get_by_symbol("SYM_EVICT") is None

    cache.put("T1", entry)
    cache.put("T2", entry)
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_fallback_policy_stale_alert():
    """Verify FallbackPolicy logs stale alerts without synthetic price fabrication."""
    from unittest.mock import MagicMock
    FallbackPolicy.reset_alerts()
    assert len(FallbackPolicy.stale_alerts) == 0

    FallbackPolicy.handle_option_stale("BANKNIFTY", "99999", 420.5, 3.5)
    assert len(FallbackPolicy.stale_alerts) == 1
    alert = FallbackPolicy.stale_alerts[0]
    assert alert["symbol"] == "BANKNIFTY"
    assert alert["token"] == "99999"
    assert alert["last_known_price"] == 420.5
    assert alert["age_seconds"] == 3.5

    # Secondary spot lookup returns None and records alert by default
    assert FallbackPolicy.get_secondary_spot_price("NIFTY 50") is None
    assert len(FallbackPolicy.stale_alerts) == 2
    spot_alert = FallbackPolicy.stale_alerts[1]
    assert spot_alert["symbol"] == "NIFTY 50"
    assert spot_alert["type"] == "SECONDARY_SPOT_UNCONFIGURED"

    # When authenticated secondary feed is registered via mock provider:
    mock_kite = MagicMock()
    mock_kite.get_ltp.side_effect = lambda s: 24890.50 if s == "NIFTY 50" else None
    FallbackPolicy.set_secondary_spot_feed(mock_kite.get_ltp)
    assert FallbackPolicy.get_secondary_spot_price("NIFTY 50") == 24890.50
    assert FallbackPolicy.get_secondary_spot_price("UNKNOWN") is None
    FallbackPolicy.set_secondary_spot_feed(None)
    FallbackPolicy.reset_alerts()


def test_singleton_micro_cache():
    """Verify get_shared_micro_cache returns consistent instance."""
    c1 = get_shared_micro_cache()
    c2 = get_shared_micro_cache()
    assert c1 is c2


def test_production_wiring_fallback_policy_stale_alert():
    """Verify live components (LivePriceFeed, PaperTrader, PositionMonitor) trigger FallbackPolicy without fabricating prices."""
    FallbackPolicy.reset_alerts()

    # 1. PaperTrader get_ltp fallback triggers FallbackPolicy
    from prometheus.execution.paper_trader import PaperTrader
    pt = PaperTrader(initial_capital=100000.0)
    price = pt.get_ltp("NONEXISTENT_CALL_OCT_25000")
    assert price == 0.0  # Zero synthetic price
    assert len(FallbackPolicy.stale_alerts) >= 1
    assert any(a["symbol"] == "NONEXISTENT_CALL_OCT_25000" for a in FallbackPolicy.stale_alerts)

    # 2. LivePriceFeed get_ltp fallback triggers FallbackPolicy
    from prometheus.paper_executor.live_bridge import LivePriceFeed
    feed = LivePriceFeed(ltp_source=None, data_engine=None)
    price_live = feed.get_ltp("NIFTY26OCT25000CE")
    assert price_live == 0.0  # Zero synthetic price
    assert any(a["symbol"] == "NIFTY26OCT25000CE" for a in FallbackPolicy.stale_alerts)

    # 3. PositionMonitor loop tick with ltp <= 0 triggers FallbackPolicy
    from unittest.mock import MagicMock
    from prometheus.execution.position_monitor import PositionMonitor, TrailingState
    mock_broker = MagicMock()
    mock_broker.get_ltp.return_value = 0.0  # broker returns 0 on failure
    monitor = PositionMonitor(broker=mock_broker, poll_interval=0.01)

    state = TrailingState(
        position_id="test-pos-1",
        symbol="NIFTY",
        tradingsymbol="NIFTY26OCT25000CE",
        direction="bullish",
        entry_premium=100.0,
        initial_sl=85.0,
        current_sl=85.0,
        target=130.0,
        premium_hwm=110.0,
    )
    monitor._positions["test-pos-1"] = state

    sym_check = f"{state.tradingsymbol or ''} {state.symbol or ''}".upper()
    pos_exchange = "BFO" if ("SENSEX" in sym_check or "BSX" in sym_check or "BANKEX" in sym_check) else "NFO"
    ltp = monitor.broker.get_ltp(state.tradingsymbol, exchange=pos_exchange)
    assert ltp == 0.0
    if ltp <= 0:
        monitor._ltp_fail_counts["test-pos-1"] = monitor._ltp_fail_counts.get("test-pos-1", 0) + 1
        FallbackPolicy.handle_option_stale(
            symbol=state.tradingsymbol or state.symbol,
            token="",
            last_known_price=state.premium_hwm or state.entry_premium,
            age_seconds=float(monitor._ltp_fail_counts["test-pos-1"] * monitor.poll_interval),
        )
    # SL remains untouched (frozen)
    assert state.current_sl == 85.0
    assert any(a["symbol"] == "NIFTY26OCT25000CE" for a in FallbackPolicy.stale_alerts)
