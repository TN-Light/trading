import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime

from prometheus.strategies.credit_spread import CreditSpreadStrategy
from prometheus.signals.price_action_momentum import PriceActionMomentumScanner
from prometheus.signals.tier_classifier import classify_signal_tier


class MockOptionChainWithMissingStrike:
    """Mock option chain where a primary hedge strike (23150 PE) is missing/None, but fallback 23100 PE exists."""
    def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
        if strike == 23300.0 and option_type == "PE":
            return {"ltp": 25.0, "tradingsymbol": "NIFTY22SEP2623300PE"}
        elif strike == 23150.0 and option_type == "PE":
            return None  # Simulates omitted strike in broker API
        elif strike == 23100.0 and option_type == "PE":
            return {"ltp": 3.0, "tradingsymbol": "NIFTY22SEP2623100PE"}
        return None

    def get_option_chain(self, symbol, spot_price=None, expiry_date=None):
        return pd.DataFrame()


def _build_synthetic_bars(trend="neutral"):
    """Helper to build >= 25 realistic 15M candles spanning prior day and today."""
    # 15 bars yesterday
    ts_yesterday = pd.date_range("2026-09-21 11:30", periods=15, freq="15min")
    rows = []
    price = 23400.0
    for ts in ts_yesterday:
        rows.append({
            "timestamp": ts,
            "open": price,
            "high": price + 15,
            "low": price - 15,
            "close": price,
            "volume": 100000,
        })

    # Today's bars from 09:15 to 10:30 (6 bars)
    today_times = ["09:15", "09:30", "09:45", "10:00", "10:15", "10:30"]
    for i, t_str in enumerate(today_times):
        ts = pd.to_datetime(f"2026-09-22 {t_str}")
        if i == 0:  # 09:15 ORB
            o, h, l, c = 23450.0, 23470.0, 23430.0, 23455.0
        elif i == 1:  # 09:30 ORB
            o, h, l, c = 23455.0, 23465.0, 23435.0, 23460.0
        elif i == 5 and trend == "breakdown":  # 10:30 Breakdown below ORB Low (23430)
            o, h, l, c = 23440.0, 23442.0, 23370.0, 23380.0
        else:
            o, h, l, c = 23450.0, 23460.0, 23440.0, 23450.0
        rows.append({
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 250000 if (i == 5 and trend == "breakdown") else 120000,
        })
    return pd.DataFrame(rows)


def _build_1h_bars(trend="NEUTRAL"):
    """Helper to build 1-Hour candles with defined HTF EMA alignment."""
    ts = pd.date_range("2026-09-15 09:15", periods=30, freq="1h")
    rows = []
    for i, t in enumerate(ts):
        if trend == "BEARISH":
            c = 23800.0 - (i * 25)  # Close well below EMA20 and EMA50
        elif trend == "BULLISH":
            c = 23000.0 + (i * 25)  # Close well above EMA20 and EMA50
        else:
            # Steady uptrend, but last bar pulls back below EMA20 while staying above EMA50 (NEUTRAL)
            c = 23000.0 + (i * 20)
            if i == 29:
                c = 23300.0  # EMA20 ~ 23370, EMA50 ~ 23230 -> Close between EMAs -> NEUTRAL
        rows.append({
            "timestamp": t,
            "open": c - 5,
            "high": c + 15,
            "low": c - 15,
            "close": c,
            "volume": 500000,
        })
    return pd.DataFrame(rows)


def test_credit_spread_dynamic_hedge_discovery_fallback():
    """Verify CreditSpreadStrategy dynamically probes and selects fallback hedge strike when primary is unlisted."""
    strat = CreditSpreadStrategy(max_days_to_expiry=1)
    df = _build_synthetic_bars(trend="neutral")
    mock_chain = MockOptionChainWithMissingStrike()

    sig = strat.evaluate_spread(
        df,
        symbol="NIFTY 50",
        capital=100000,
        option_chain=mock_chain,
    )

    assert sig is not None, "Credit spread should NOT be None despite missing 23150 strike!"
    assert sig["strategy_type"] == "credit_spread"
    assert sig["spread_type"] == "BULL_PUT_SPREAD"
    # Long strike must be dynamically adjusted to the available fallback strike 23100
    assert sig["short_strike"] == 23300.0
    assert sig["long_strike"] == 23100.0
    assert sig["strike_width"] == 200.0  # 23300 - 23100
    assert sig["net_credit"] == 22.0     # 25.0 - 3.0
    assert sig["legs"][0]["strike"] == 23100.0
    assert sig["legs"][0]["tradingsymbol"] == "NIFTY22SEP2623100PE"
    assert sig["legs"][1]["strike"] == 23300.0
    assert sig["legs"][1]["tradingsymbol"] == "NIFTY22SEP2623300PE"


def test_price_action_momentum_neutral_1h_trend_passes_to_tier_c():
    """Verify that a valid ORB breakdown with NEUTRAL 1H trend emits a signal that Tier Classifier gates to Tier C."""
    scanner = PriceActionMomentumScanner()
    df = _build_synthetic_bars(trend="breakdown")
    df_1h_neutral = _build_1h_bars(trend="NEUTRAL")

    # Evaluate bar at 10:30 (breakdown bar) with golden_mode=True
    sig = scanner.evaluate_bar(
        df,
        symbol="NIFTY 50",
        is_expiry_day=True,
        df_1h=df_1h_neutral,
        golden_mode=True,
    )

    assert sig is not None, "10:30 breakdown should emit a signal with NEUTRAL 1H trend!"
    assert sig["action"] == "BUY_PE"
    assert sig["is_golden_setup"] is False, "NEUTRAL 1H trend must NOT be marked as golden setup!"
    assert sig["edge_score"] >= 5.0

    # Classify institutional tier
    tier_info = classify_signal_tier(sig)
    # Tier Classifier must assign to Tier C (Paper Trading Only), blocking live broker execution
    assert tier_info["tier"] == "C"
    assert tier_info["is_live_eligible"] is False
    assert any("NEUTRAL" in r for r in tier_info["classification_reasons"])


def test_price_action_momentum_strict_1h_trend_qualifies_for_live():
    """Verify that a breakdown with strictly BEARISH 1H trend marks is_golden_setup=True and qualifies for Tier B/S."""
    scanner = PriceActionMomentumScanner()
    df = _build_synthetic_bars(trend="breakdown")
    df_1h_bearish = _build_1h_bars(trend="BEARISH")

    sig = scanner.evaluate_bar(
        df,
        symbol="NIFTY 50",
        is_expiry_day=False,  # Non-0DTE to test Tier B live eligibility
        df_1h=df_1h_bearish,
        golden_mode=True,
    )

    assert sig is not None
    assert sig["action"] == "BUY_PE"
    assert sig["is_golden_setup"] is True, "Strict BEARISH 1H trend must be marked as golden setup!"

    tier_info = classify_signal_tier(sig)
    # Strict 1H alignment allows live eligibility (Tier B or S)
    assert tier_info["tier"] in ("S", "B")
    assert tier_info["is_live_eligible"] is True


def test_price_action_momentum_conflicting_1h_trend_strictly_rejected():
    """Verify that a breakdown (PE) with BULLISH 1H trend is strictly rejected (returns None)."""
    scanner = PriceActionMomentumScanner()
    df = _build_synthetic_bars(trend="breakdown")
    df_1h_bullish = _build_1h_bars(trend="BULLISH")

    sig = scanner.evaluate_bar(
        df,
        symbol="NIFTY 50",
        is_expiry_day=True,
        df_1h=df_1h_bullish,
        golden_mode=True,
    )

    assert sig is None, "PE breakdown against BULLISH 1H trend must be strictly vetoed!"
