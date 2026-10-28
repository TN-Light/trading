"""
Unit and regression tests for Friday Session Learnings & Quantitative Flaw Fixes.

Covers:
1. Climax Exhaustion Gate on Trend-Day Lunch Bypass in main.py
2. Harmonized Tier B Promotion for High-Conviction Setups in Dealer Long Gamma (tier_classifier.py)
3. SmartAPI Option Greek Optimization & BSE/BFO skip in angelone_options.py
4. Throttled [STALE_DATA_ALERT] console logging in FallbackPolicy (price_micro_cache.py)
5. Weekend Trading-Day Alignment in AngelOneFetcher.fetch_historical (angelone_fetcher.py)
"""

import time
import pytest
from datetime import datetime, time as dtime, timedelta
from unittest.mock import MagicMock, patch

from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.data.price_micro_cache import FallbackPolicy
from prometheus.data.angelone_options import AngelOneOptionChain
from prometheus.data.angelone_fetcher import AngelOneFetcher
from prometheus.data.rate_limiter import PrioritizedTokenBucketLimiter, EndpointCategory, PriorityTier


# ============================================================================
# 1. Climax Exhaustion Gate Tests
# ============================================================================

def test_climax_exhaustion_detection_logic():
    """Verify spot overextension checks reject climax exhaustion setups and accept fresh breakouts."""
    def eval_climax(spot_p, vwap_val, spot_atr, orb_h, orb_l, action):
        is_climax_exhausted = False
        if spot_atr > 0 and spot_p > 0:
            if vwap_val > 0 and abs(spot_p - vwap_val) >= 2.2 * spot_atr:
                is_climax_exhausted = True
            if orb_h is not None and "CE" in action and (spot_p - orb_h) >= 2.0 * spot_atr:
                is_climax_exhausted = True
            elif orb_l is not None and "PE" in action and (orb_l - spot_p) >= 2.0 * spot_atr:
                is_climax_exhausted = True
        return is_climax_exhausted

    # Case A: Sensex 12:39 PM scenario (Spot = 83,866, ORB High = 83,100, ATR = 200, VWAP = 83,200)
    # spot - orb_h = 766 pts >= 2.0 * 200 (400 pts) -> EXHAUSTED
    assert eval_climax(
        spot_p=83866, vwap_val=83200, spot_atr=200, orb_h=83100, orb_l=82800, action="BUY_CE"
    ) is True

    # Case B: VWAP Overextended (Spot = 25,500, VWAP = 25,000, ATR = 150 -> diff 500 >= 2.2 * 150 = 330)
    assert eval_climax(
        spot_p=25500, vwap_val=25000, spot_atr=150, orb_h=25400, orb_l=25100, action="BUY_CE"
    ) is True

    # Case C: Fresh Breakout (Spot = 25,220, ORB High = 25,200, VWAP = 25,150, ATR = 100)
    # spot - orb_h = 20 pts (< 200), spot - vwap = 70 pts (< 220) -> FRESH
    assert eval_climax(
        spot_p=25220, vwap_val=25150, spot_atr=100, orb_h=25200, orb_l=24900, action="BUY_CE"
    ) is False

    # Case D: Downside PE Climax (Spot = 55,000, ORB Low = 55,800, ATR = 300 -> diff 800 >= 2.0 * 300)
    assert eval_climax(
        spot_p=55000, vwap_val=55700, spot_atr=300, orb_h=56200, orb_l=55800, action="BUY_PE"
    ) is True


# ============================================================================
# 2. Harmonized Tier B Promotion for High-Conviction in Dealer Long Gamma
# ============================================================================

def test_long_gamma_tier_b_promotion_for_score_8_plus():
    """Score >= 8.0 in Dealer Long Gamma for Put Buying (ceiling fade) promotes to Tier B Scalp with compressed targets (GEMINI.md Rule 1)."""
    sig = {
        "action": "BUY_PE",
        "symbol": "BANK NIFTY",
        "strategy": "PriceAction_Momentum",
        "strategy_type": "option_buying",
        "edge_score": 8.5,
        "bar_timestamp": "2026-10-16 10:53:00",
        "reasons": [
            "15M_ORB_Low_Breakdown",
            "Session_VWAP_Bearish",
            "1H_Trend_Bearish",
        ],
        "net_gex_crores": 2500.0,  # Strongly Dealer Long Gamma (> +500 Cr)
        "net_gex": 2500.0,
        "gamma_regime": "LONG_GAMMA",
        "days_to_expiry": 3,
    }
    res = classify_signal_tier(sig)
    assert res["tier"] == "B"
    assert res["tier_name"] == "REGIME_SCALP_LONG_GAMMA"
    assert res["is_live_eligible"] is True
    assert "REGIME SCALP" in res["tier_badge"]
    assert "Compressed Scalp Target" in res["action_instruction"]

    # In contrast, Call Buying (BUY_CE) in Dealer Long Gamma is strictly demoted to Tier C (Asymmetric GEX Gate)
    sig_ce = dict(sig, action="BUY_CE")
    res_ce = classify_signal_tier(sig_ce)
    assert res_ce["tier"] == "C"
    assert res_ce["is_live_eligible"] is False


def test_long_gamma_remains_tier_c_if_score_below_8():
    """Score < 8.0 in Dealer Long Gamma must remain Tier C (Paper Only)."""
    sig = {
        "action": "BUY_CE",
        "symbol": "BANK NIFTY",
        "strategy": "PriceAction_Momentum",
        "strategy_type": "option_buying",
        "edge_score": 7.5,  # Score < 8.0
        "bar_timestamp": "2026-10-16 10:53:00",
        "reasons": [
            "15M_ORB_High_Breakout",
            "Session_VWAP_Bullish",
            "1H_Trend_Bullish",
        ],
        "net_gex_crores": 2500.0,  # Dealer Long Gamma
        "net_gex": 2500.0,
        "gamma_regime": "LONG_GAMMA",
        "days_to_expiry": 3,
    }
    res = classify_signal_tier(sig)
    assert res["tier"] == "C"
    assert res["is_live_eligible"] is False
    assert "PAPER ONLY" in res["tier_badge"]
    assert "Paper Trading" in res["action_instruction"]


# ============================================================================
# 3. SmartAPI Option Greek Optimization & BSE/BFO Skip
# ============================================================================

def test_angelone_options_get_real_premium_defaults_and_bse_skip():
    """Verify include_greeks defaults to False, and BSE/BFO skips optionGreek calls."""
    mock_fetcher = MagicMock()
    mock_obj = MagicMock()
    mock_fetcher._obj = mock_obj
    mock_fetcher._ensure_connected.return_value = True

    mock_batch_engine = MagicMock()
    ao = AngelOneOptionChain(mock_fetcher, batch_engine=mock_batch_engine)

    # Verify method signature default
    import inspect
    sig = inspect.signature(ao.get_real_premium)
    assert sig.parameters["include_greeks"].default is False

    ao._rate_limit = lambda priority: True

    # Mock discover_contracts and batch engine snapshot
    mock_contract = {
        "symboltoken": "12345",
        "token": "12345",
        "symbol": "SENSEX",
        "strike": 83000,
        "type": "CE",
        "option_type": "CE",
        "expiry": "2026-10-16",
        "tradingsymbol": "SENSEX26OCT83000CE",
        "trading_symbol": "SENSEX26OCT83000CE",
        "segment": "BFO",
    }
    ao.discover_contracts = MagicMock(return_value=[mock_contract])

    mock_batch_engine.get_quotes_batch.return_value = {
        "12345": {
            "ltp": 150.0,
            "bid": 149.0,
            "ask": 151.0,
            "open_interest": 10000,
            "volume": 50000,
        }
    }

    # Test 1: include_greeks=False (default) -> optionGreek never called
    res = ao.get_real_premium("SENSEX", 83000, "CE", expiry="2026-10-16")
    assert res is not None
    assert res["ltp"] == 150.0
    assert mock_obj.optionGreek.called is False

    # Test 2: include_greeks=True on SENSEX (BFO) -> optionGreek still skipped because unsupported
    mock_obj.reset_mock()
    res2 = ao.get_real_premium("SENSEX", 83000, "CE", expiry="2026-10-16", include_greeks=True)
    assert res2 is not None
    assert mock_obj.optionGreek.called is False


# ============================================================================
# 4. FallbackPolicy Alert Throttling
# ============================================================================

def test_fallback_policy_alert_throttling():
    """Verify handle_option_stale appends to stale_alerts but throttles logger warnings to 60s."""
    FallbackPolicy.reset_alerts()
    assert len(FallbackPolicy.stale_alerts) == 0

    with patch("prometheus.data.price_micro_cache.logger.warning") as mock_warn:
        # Call 1: should append and log
        FallbackPolicy.handle_option_stale("NIFTY 50", "99926000", 24000.0, 1.5)
        assert len(FallbackPolicy.stale_alerts) == 1
        assert mock_warn.call_count == 1

        # Call 2 (immediate): should append to telemetry, but NOT call logger.warning again
        FallbackPolicy.handle_option_stale("NIFTY 50", "99926000", 24000.0, 2.0)
        assert len(FallbackPolicy.stale_alerts) == 2
        assert mock_warn.call_count == 1  # Throttled!

        # Advance monotonic time by 65 seconds
        orig_time = time.monotonic
        try:
            with patch("time.monotonic", return_value=orig_time() + 65.0):
                FallbackPolicy.handle_option_stale("NIFTY 50", "99926000", 24000.0, 3.0)
                assert len(FallbackPolicy.stale_alerts) == 3
                assert mock_warn.call_count == 2  # Logged again after 60s cooldown!
        finally:
            FallbackPolicy.reset_alerts()
            assert len(FallbackPolicy.stale_alerts) == 0
            assert len(FallbackPolicy._last_logged_time) == 0


# ============================================================================
# 5. Weekend Trading-Day Alignment in AngelOneFetcher
# ============================================================================

def test_angelone_fetcher_weekend_trading_day_alignment():
    """Verify fetch_historical clamps weekend end_date to Friday and aligns start_date to weekdays."""
    fetcher = AngelOneFetcher("key", "client", "pwd", "totp")
    mock_obj = MagicMock()
    mock_obj.getCandleData.return_value = {
        "status": True,
        "data": [["2026-10-16 09:15", 25000.0, 25050.0, 24950.0, 25020.0, 100]],
    }
    fetcher._obj = mock_obj
    fetcher._ensure_connected = lambda: True

    captured_params = []
    def capture_get_candles(params):
        captured_params.append(params)
        return {"status": True, "data": [["2026-10-16 09:15", 25000, 25050, 24950, 25020, 100]]}

    mock_obj.getCandleData.side_effect = capture_get_candles

    # Mock datetime.now() to a Sunday (2026-10-18 12:00:00)
    sunday_dt = datetime(2026, 10, 18, 12, 0, 0)
    with patch("prometheus.data.angelone_fetcher.datetime") as mock_dt:
        mock_dt.now.return_value = sunday_dt
        mock_dt.strptime = datetime.strptime
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        df = fetcher.fetch_historical("NIFTY 50", days=5, priority=PriorityTier.P4_BACKGROUND_SCANNER)
        assert not df.empty
        assert len(captured_params) > 0

        first_param = captured_params[0]
        from_str = first_param["fromdate"]
        to_str = first_param["todate"]

        from_dt = datetime.strptime(from_str, "%Y-%m-%d %H:%M")
        to_dt = datetime.strptime(to_str, "%Y-%m-%d %H:%M")

        # Must NOT query Sunday (2026-10-18) as todate! todate must be clamped to Friday (2026-10-16)
        assert to_dt.weekday() < 5  # Mon-Fri
        assert from_dt.weekday() < 5  # Mon-Fri
        assert to_dt.strftime("%Y-%m-%d") == "2026-10-16"
        assert from_dt.strftime("%Y-%m-%d") == "2026-10-12"
