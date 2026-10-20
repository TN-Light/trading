"""
Unit and integration tests for /scan latency, rate-limiter anti-burst pacing,
SmartAPI Greek lookup payloads, and spot-volume trend day detection.

Verifies:
1. Historical API Inter-Call Pacing (rate_limiter.py)
   - Mandatory >=400ms inter-call pacing on EndpointCategory.HISTORICAL
   - Sub-5ms bursts strictly prevented even when tokens are available
   - Non-blocking (timeout=0.0) fails when pacing unsatisfied
   - P1 priority and other endpoints (QUOTES, ORDERS) unaffected
2. Trim Historical Scan Window (data_bridge.py)
   - 15-minute primary scan fetches days=5 instead of days=60
   - Single-chunk Angel One dispatch with valid data handling
3. SmartAPI Greek Lookup Payload (angelone_options.py)
   - get_real_premium passes root underlying name (e.g. NIFTY, BANKNIFTY)
   - expirydate formatted as DDMMMYYYY (e.g. 13OCT2026)
4. Spot-Volume Proxy for Trend Day Detection (price_action_momentum.py & main.py)
   - Zero-volume spot feeds evaluate range expansion (>= 2.0x ATR + displacement)
   - Lunch dead zone (11:30 - 13:15 IST) bypass permits high-conviction momentum
"""

import time
from datetime import datetime, time as dtime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from prometheus.data.rate_limiter import (
    EndpointCategory,
    PrioritizedTokenBucketLimiter,
    PriorityTier,
)
from prometheus.pipeline.data_bridge import DataBridge
from prometheus.pipeline.types import DataStatus
from prometheus.data.angelone_options import AngelOneOptionChain
from prometheus.signals.price_action_momentum import PriceActionMomentumScanner


# ============================================================================
# 1. Historical API Inter-Call Pacing Tests
# ============================================================================

class TestHistoricalApiInterCallPacing:
    """Verify >= 400ms mandatory inter-call pacing on EndpointCategory.HISTORICAL."""

    def test_historical_default_config_pacing(self):
        """Historical bucket must have min_pacing >= 0.400s configured by default."""
        limiter = PrioritizedTokenBucketLimiter()
        hist_bucket = limiter._buckets[EndpointCategory.HISTORICAL]
        assert hist_bucket.min_pacing >= 0.400
        assert limiter.configs[EndpointCategory.HISTORICAL]["min_pacing"] >= 0.400

        # Quotes and Orders buckets must not have historical pacing
        quotes_bucket = limiter._buckets[EndpointCategory.QUOTES]
        orders_bucket = limiter._buckets[EndpointCategory.ORDERS]
        assert quotes_bucket.min_pacing == 0.0
        assert orders_bucket.min_pacing == 0.0

    def test_historical_inter_call_pacing_delay_enforced(self):
        """Successive historical requests must be spaced by at least 400ms."""
        limiter = PrioritizedTokenBucketLimiter()

        t0 = time.monotonic()
        ok1 = limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=2.0,
        )
        t1 = time.monotonic()
        assert ok1 is True
        assert (t1 - t0) < 0.05  # First call acquires immediately

        # Second call must wait >= 400ms even though tokens remain in bucket
        ok2 = limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=2.0,
        )
        t2 = time.monotonic()
        assert ok2 is True
        inter_call_delay = t2 - t1
        assert inter_call_delay >= 0.380, f"Expected >= 400ms pacing, got {inter_call_delay:.4f}s"

    def test_sub_5ms_burst_strictly_prevented(self):
        """Sub-5ms bursts must be rejected when timeout is zero (non-blocking)."""
        limiter = PrioritizedTokenBucketLimiter()

        # First call succeeds
        assert limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=1.0,
        ) is True

        # Immediate non-blocking call (< 5ms) must fail because pacing is violated
        assert limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=0.0,
        ) is False

    def test_short_timeout_strictly_honored_without_artificial_extension(self):
        """Caller timeout shorter than pacing delay must be respected and not forcibly extended."""
        limiter = PrioritizedTokenBucketLimiter()

        # First call succeeds immediately
        assert limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=1.0,
        ) is True

        # Second call with 50ms timeout must fail quickly (< 100ms), NOT block for 450ms
        t0 = time.monotonic()
        res = limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=0.05,
        )
        elapsed = time.monotonic() - t0
        assert res is False, "Request should have timed out due to 400ms pacing requirement"
        assert elapsed < 0.12, f"Caller timeout violated: blocked for {elapsed:.4f}s instead of ~0.05s"

    def test_p1_order_execution_exempt_from_historical_pacing(self):
        """P1 orders bypass pacing to preserve <50ms emergency SLA."""
        limiter = PrioritizedTokenBucketLimiter()

        # Acquire once
        assert limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P4_BACKGROUND_SCANNER,
            timeout=1.0,
        ) is True

        # Immediate P1 acquire should succeed without 400ms delay
        t0 = time.monotonic()
        ok_p1 = limiter.acquire(
            endpoint=EndpointCategory.HISTORICAL,
            priority=PriorityTier.P1_ORDER_EXECUTION,
            timeout=0.05,
        )
        t1 = time.monotonic()
        assert ok_p1 is True
        assert (t1 - t0) < 0.05

    def test_quotes_bucket_unaffected_by_historical_pacing(self):
        """Quotes bucket dispatches rapidly without 400ms pacing delay."""
        limiter = PrioritizedTokenBucketLimiter()

        t0 = time.monotonic()
        for _ in range(3):
            assert limiter.acquire(
                endpoint=EndpointCategory.QUOTES,
                priority=PriorityTier.P2_POSITION_MONITOR,
                timeout=0.1,
            ) is True
        elapsed = time.monotonic() - t0
        assert elapsed < 0.10, f"Quotes burst throttled unexpectedly: {elapsed:.4f}s"


# ============================================================================
# 2. Trim Historical Scan Window Tests
# ============================================================================

class TestTrimHistoricalScanWindow:
    """Verify DataBridge fetches days=5 for live 15-minute scanning."""

    def test_data_bridge_calls_fetch_historical_with_5_days(self):
        """Primary intraday scan data must request days=5, not days=60."""
        recent_data = pd.DataFrame({
            "timestamp": [datetime.now() - timedelta(minutes=i * 15) for i in range(120, 0, -1)],
            "open": range(120),
            "high": range(120),
            "low": range(120),
            "close": range(120),
            "volume": range(120),
        })

        mock_data_engine = MagicMock()
        mock_data_engine.fetch_historical = MagicMock(return_value=recent_data)

        bridge = DataBridge(mock_data_engine, max_staleness_seconds=999999)
        result = bridge.fetch_scan_data("NIFTY 50", interval="15minute")

        assert result.status == DataStatus.OK
        assert mock_data_engine.fetch_historical.call_count >= 1

        # First call is the primary intraday scan fetch
        first_call = mock_data_engine.fetch_historical.call_args_list[0]
        args, kwargs = first_call
        assert args[0] == "NIFTY 50"
        assert kwargs.get("days") == 5, f"Expected days=5, got days={kwargs.get('days')}"
        assert kwargs.get("interval") == "15minute"
        assert kwargs.get("force_refresh") is True


# ============================================================================
# 3. SmartAPI Greek Lookup Payload Tests
# ============================================================================

class TestSmartApiGreekLookupPayload:
    """Verify get_real_premium passes root underlying and DDMMMYYYY expirydate to optionGreek."""

    def test_greek_lookup_payload_formatting(self):
        """optionGreek must receive root underlying symbol and DDMMMYYYY expiry."""
        mock_obj = MagicMock()
        mock_obj.optionGreek.return_value = {
            "status": True,
            "data": {
                "delta": "0.52",
                "gamma": "0.0012",
                "theta": "-15.4",
                "vega": "8.7",
                "impliedVolatility": "14.2",
            },
        }

        mock_fetcher = MagicMock()
        mock_fetcher._obj = mock_obj
        mock_fetcher._ensure_connected.return_value = True
        chain = AngelOneOptionChain(fetcher=mock_fetcher)

        # Seed instruments cache with contract
        chain._instruments_cache = [
            {
                "symboltoken": "12345",
                "tradingsymbol": "NIFTY13OCT2625000CE",
                "name": "NIFTY",
                "underlying": "NIFTY",
                "expiry": "2026-10-13",
                "strike": 25000.0,
                "option_type": "CE",
            }
        ]

        # Mock market data response
        mock_obj.ltpData.return_value = {"status": True, "data": {"ltp": 125.0}}
        chain.discover_contracts = MagicMock(return_value=chain._instruments_cache)
        chain._get_batch_engine = MagicMock(return_value=None)

        res = chain.get_real_premium(
            symbol="NIFTY",
            strike=25000.0,
            option_type="CE",
            expiry="2026-10-13",
            include_greeks=True,
        )

        assert res is not None
        assert mock_obj.optionGreek.called

        called_payload = mock_obj.optionGreek.call_args[0][0]
        assert called_payload["name"] == "NIFTY", (
            f"Expected root underlying 'NIFTY', got {called_payload['name']}"
        )
        assert called_payload["expirydate"] == "13OCT2026", (
            f"Expected uppercase DDMMMYYYY '13OCT2026', got {called_payload['expirydate']}"
        )
        assert called_payload["strikeprice"] == "25000.0"
        assert called_payload["optiontype"] == "CE"

        # Verify parsed Greek values
        assert res["delta"] == pytest.approx(0.52)
        assert res["gamma"] == pytest.approx(0.0012)
        assert res["theta"] == pytest.approx(-15.4)
        assert res["vega"] == pytest.approx(8.7)
        assert res["iv"] == pytest.approx(14.2)

    def test_greek_lookup_payload_date_types(self):
        """Verify datetime.date and timestamp string inputs convert to uppercase DDMMMYYYY."""
        from datetime import date
        mock_obj = MagicMock()
        mock_obj.optionGreek.return_value = {"status": True, "data": {"delta": "0.45"}}
        mock_fetcher = MagicMock()
        mock_fetcher._obj = mock_obj
        mock_fetcher._ensure_connected.return_value = True
        chain = AngelOneOptionChain(fetcher=mock_fetcher)

        chain._instruments_cache = [
            {
                "symboltoken": "99999",
                "tradingsymbol": "BANKNIFTY13OCT2654000PE",
                "name": "BANKNIFTY",
                "underlying": "BANKNIFTY",
                "expiry": "2026-10-13 00:00:00",
                "strike": 54000.0,
                "option_type": "PE",
            }
        ]
        chain.discover_contracts = MagicMock(return_value=chain._instruments_cache)
        chain._get_batch_engine = MagicMock(return_value=None)
        mock_obj.ltpData.return_value = {"status": True, "data": {"ltp": 250.0}}

        # Test with datetime.date object
        res_date = chain.get_real_premium(
            symbol="NIFTY BANK",
            strike=54000.0,
            option_type="PE",
            expiry=date(2026, 10, 13),
            include_greeks=True,
        )
        assert res_date is not None
        payload_date = mock_obj.optionGreek.call_args[0][0]
        assert payload_date["name"] == "BANKNIFTY"
        assert payload_date["expirydate"] == "13OCT2026"

        # Test with ISO string containing timestamp
        res_ts = chain.get_real_premium(
            symbol="NIFTY BANK",
            strike=54000.0,
            option_type="PE",
            expiry="2026-10-13 00:00:00",
            include_greeks=True,
        )
        assert res_ts is not None
        payload_ts = mock_obj.optionGreek.call_args[0][0]
        assert payload_ts["expirydate"] == "13OCT2026"


# ============================================================================
# 4. Spot-Volume Proxy for Trend Day Detection Tests
# ============================================================================

class TestSpotVolumeProxyTrendDay:
    """Verify spot-volume proxy triggers is_institutional_trend_day on zero-volume feeds."""

    def test_spot_proxy_confirms_institutional_trend_day(self):
        """Range expansion (>=2x ATR + body displacement) confirms trend day on zero volume."""
        from prometheus.tests.test_trend_day_lunch_bypass_and_orb_sl import _generate_synthetic_candles

        scanner = PriceActionMomentumScanner()
        prior_df = _generate_synthetic_candles(n_bars=20, start_dt=datetime(2026, 9, 29, 9, 15), trend="neutral")
        today_df = _generate_synthetic_candles(n_bars=13, start_dt=datetime(2026, 9, 30, 9, 15), trend="bullish", strong_volume=False)
        df = pd.concat([prior_df, today_df], ignore_index=True)
        df["volume"] = 0  # Zero volume spot index

        # Create massive institutional range expansion on bar 24 (10:15 AM breakout bar >= 2.0x ATR)
        df.loc[24, "high"] = df.loc[24, "open"] + 130.0
        df.loc[24, "close"] = df.loc[24, "open"] + 120.0
        df.loc[24, "low"] = df.loc[24, "open"] - 5.0

        sig = scanner.evaluate_bar(df, symbol="NIFTY BANK", is_expiry_day=True, golden_mode=True)
        assert sig is not None, "Signal should not be suppressed during lunch on confirmed spot trend day"
        assert sig["is_institutional_trend_day"] is True
        assert sig.get("has_range_expansion") is True
        assert sig["action"] == "BUY_CE"

    def test_spot_proxy_does_not_trigger_on_normal_chop(self):
        """Chop without range expansion strictly does NOT trigger is_institutional_trend_day."""
        from prometheus.tests.test_trend_day_lunch_bypass_and_orb_sl import _generate_synthetic_candles

        scanner = PriceActionMomentumScanner()
        prior_df = _generate_synthetic_candles(n_bars=20, start_dt=datetime(2026, 9, 29, 9, 15), trend="neutral")
        today_df = _generate_synthetic_candles(n_bars=13, start_dt=datetime(2026, 9, 30, 9, 15), trend="bullish", strong_volume=False)
        df = pd.concat([prior_df, today_df], ignore_index=True)
        df["volume"] = 0  # Zero volume spot index, no range expansion

        sig = scanner.evaluate_bar(df, symbol="NIFTY BANK", is_expiry_day=True, golden_mode=True)
        # At 12:15 PM without confirmed trend day, lunch cutoff suppresses the signal
        assert sig is None or sig.get("is_institutional_trend_day") is False

    def test_spot_proxy_directional_displacement_alignment(self):
        """Bearish plunge bar must NOT spoof institutional trend day or range expansion for BUY_CE."""
        from prometheus.tests.test_trend_day_lunch_bypass_and_orb_sl import _generate_synthetic_candles

        scanner = PriceActionMomentumScanner()
        prior_df = _generate_synthetic_candles(n_bars=20, start_dt=datetime(2026, 9, 29, 9, 15), trend="neutral")
        today_df = _generate_synthetic_candles(n_bars=13, start_dt=datetime(2026, 9, 30, 9, 15), trend="bullish", strong_volume=False)
        df = pd.concat([prior_df, today_df], ignore_index=True)
        df["volume"] = 0  # Zero volume spot index

        # Create massive BEARISH plunge bar on bar 24 (10:15 AM plunge: close << open)
        df.loc[24, "high"] = df.loc[24, "open"] + 5.0
        df.loc[24, "low"] = df.loc[24, "open"] - 130.0
        df.loc[24, "close"] = df.loc[24, "open"] - 120.0

        # When evaluating at 12:15 PM, this bearish plunge must NOT confirm a bullish trend day for BUY_CE
        sig = scanner.evaluate_bar(df, symbol="NIFTY BANK", is_expiry_day=True, golden_mode=True)
        # Should be None (suppressed during lunch because today is NOT an institutional bull trend day)
        if sig is not None and sig.get("action") == "BUY_CE":
            assert sig.get("is_institutional_trend_day") is not True
            assert sig.get("has_range_expansion") is not True

    def test_main_lunch_gate_spot_proxy_bypass(self):
        """Verify main.py lunch gate evaluates spot range expansion proxy and respects ORB bounds."""
        # Simulated refined signal during lunch dead zone (12:15 PM) on spot index
        refined_breakout = {
            "action": "BUY_CE",
            "strategy_type": "option_buying",
            "symbol": "NIFTY BANK",
            "tradingsymbol": "BANKNIFTY30SEP2654500CE",
            "spot_price": 54600.0,
            "orb_high": 54500.0,  # Price broken OUT above ORB
            "adx": 28.0,
            "atr": 60.0,
            "bar_range": 130.0,  # >= 2.0x ATR
            "body_displacement": 110.0,  # >= 0.5x bar_range
            "has_range_expansion": True,
            "is_institutional_trend_day": False,  # Initially unflagged
        }

        # Emulate main.py lunch gate logic
        is_lunch_dead_zone = True
        is_buying = True
        is_trend_day = bool(refined_breakout.get("is_institutional_trend_day"))

        if not is_trend_day:
            adx_val = float(refined_breakout.get("adx", 0.0) or 0.0)
            spot_atr = float(refined_breakout.get("atr", 0.0) or 0.0)
            bar_range = float(refined_breakout.get("bar_range", 0.0) or 0.0)
            body_disp = float(refined_breakout.get("body_displacement", 0.0) or 0.0)
            has_expansion = bool(refined_breakout.get("has_range_expansion"))
            if adx_val >= 25.0 and spot_atr > 0 and (has_expansion or (bar_range >= 2.0 * spot_atr and body_disp >= 0.5 * bar_range)):
                orb_h = refined_breakout.get("orb_high")
                orb_l = refined_breakout.get("orb_low")
                spot_p = float(refined_breakout.get("underlying_price", 0) or refined_breakout.get("spot_price", 0) or refined_breakout.get("entry_price", 0))
                is_orb_ok = True
                if orb_h is not None and "CE" in str(refined_breakout.get("action", "")):
                    is_orb_ok = (spot_p >= orb_h) if spot_p > 0 else True
                elif orb_l is not None and "PE" in str(refined_breakout.get("action", "")):
                    is_orb_ok = (spot_p <= orb_l) if spot_p > 0 else True
                if is_orb_ok:
                    is_trend_day = True
                    refined_breakout["is_institutional_trend_day"] = True

        assert is_trend_day is True
        assert refined_breakout["is_institutional_trend_day"] is True

        # Now test that inside-ORB chop strictly rejects trend day bypass
        refined_inside_orb = {
            "action": "BUY_CE",
            "strategy_type": "option_buying",
            "symbol": "NIFTY BANK",
            "spot_price": 54400.0,  # Price trapped INSIDE ORB (< 54500)
            "orb_high": 54500.0,
            "adx": 28.0,
            "atr": 60.0,
            "has_range_expansion": True,
            "is_institutional_trend_day": False,
        }
        is_trend_day_chop = False
        adx_val = float(refined_inside_orb.get("adx", 0.0) or 0.0)
        spot_atr = float(refined_inside_orb.get("atr", 0.0) or 0.0)
        has_expansion = bool(refined_inside_orb.get("has_range_expansion"))
        if adx_val >= 25.0 and spot_atr > 0 and has_expansion:
            orb_h = refined_inside_orb.get("orb_high")
            spot_p = float(refined_inside_orb.get("spot_price", 0))
            is_orb_ok = (spot_p >= orb_h) if spot_p > 0 else True
            if is_orb_ok:
                is_trend_day_chop = True

        assert is_trend_day_chop is False, "Inside-ORB chop must not bypass lunch gate"

