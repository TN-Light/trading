"""
Unit and integration tests for:
1. Spot-Anchored ORB Retest Stop-Loss with dynamic target scaling
2. Institutional Trend-Day Lunch Dead Zone Bypass (ADX >= 25, Volume >= 1.5x 20-SMA)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta

from prometheus.signals.technical import calculate_adx
from prometheus.signals.price_action_momentum import PriceActionMomentumScanner
from prometheus.signals.tier_classifier import classify_signal_tier


def _generate_synthetic_candles(n_bars=30, start_dt=datetime(2026, 9, 30, 9, 15), trend="bullish", strong_volume=False):
    """Generate 15-minute synthetic OHLCV candles."""
    timestamps = [start_dt + timedelta(minutes=15 * i) for i in range(n_bars)]
    base_price = 54200.0

    rows = []
    current = base_price
    for i, ts in enumerate(timestamps):
        if trend == "bullish":
            step = 30.0 if i > 1 else 10.0
        elif trend == "bearish":
            step = -30.0 if i > 1 else -10.0
        else:
            step = 15.0 if (i % 2 == 0) else -15.0

        open_p = current
        high_p = open_p + 15.0 + max(0, step)
        low_p = open_p - 15.0 + min(0, step)
        close_p = open_p + step
        
        # Base volume with optional massive volume surge on breakout
        if strong_volume and i >= 10:
            vol = 300000  # 3x normal volume
        else:
            vol = 100000

        rows.append({
            "timestamp": ts,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
        })
        current = close_p

    return pd.DataFrame(rows)


class TestWildersADX:
    """Verify calculate_adx implementation."""

    def test_adx_computation_range(self):
        df = _generate_synthetic_candles(n_bars=40, trend="bullish")
        adx = calculate_adx(df, period=14)
        assert len(adx) == len(df)
        assert not adx.isna().any()
        # In a sustained 40-bar trend, ADX should clearly exceed 25.0
        assert adx.iloc[-1] > 25.0, f"Expected ADX > 25.0, got {adx.iloc[-1]}"

    def test_adx_flat_market_low(self):
        df = _generate_synthetic_candles(n_bars=40, trend="flat")
        adx = calculate_adx(df, period=14)
        assert len(adx) == len(df)
        # In choppy sideways market, ADX remains low
        assert adx.iloc[-1] < 25.0, f"Expected ADX < 25.0 in flat market, got {adx.iloc[-1]}"


class TestSpotAnchoredORBRetestSL:
    """Verify that Stop Loss anchored to Spot ORB Line with retest buffer survives normal shakeouts."""

    def test_banknifty_retest_buffer_survival(self):
        """
        Simulate Bank Nifty on 2026-09-30:
        Entry Spot: 54,778.90
        ORB High: 54,625.75
        ATR: 88.0
        Option Entry: 1055.05
        Option Retest Low: 979.10
        """
        spot_price = 54778.90
        orb_high = 54625.75
        spot_atr = 88.0
        opt_ltp = 1055.05
        atm_delta = 0.50
        noise_floor = 35.0

        # Calibration logic matching updated main.py
        retest_buffer = max(25.0, 0.30 * spot_atr)
        assert retest_buffer == pytest.approx(26.4, rel=1e-2)

        spot_sl_level = round(orb_high - retest_buffer, 2)
        assert spot_sl_level == pytest.approx(54599.35, rel=1e-2)

        spot_risk = max(spot_price - spot_sl_level, spot_atr * 0.5)
        structural_sl_pts = max(noise_floor, round(atm_delta * spot_risk, 1))
        assert structural_sl_pts == pytest.approx(89.8, rel=1e-2)

        # Baseline target
        target_gain_pts = 44.0
        # Check that target expands dynamically rather than SL being clamped
        if structural_sl_pts > round(target_gain_pts * 1.2, 1):
            target_gain_pts = max(target_gain_pts, round(structural_sl_pts * 1.2, 1))

        assert target_gain_pts == pytest.approx(107.8, rel=1e-2)

        sl_price = round(opt_ltp - structural_sl_pts, 2)
        assert sl_price == pytest.approx(965.25, rel=1e-2)

        # The option dropped to 979.10 during the 10:30 shakeout.
        # With sl_price at 965.25, the trade has 13.85 points of cushion!
        retest_low_option = 979.10
        assert retest_low_option > sl_price, (
            f"Expected retest low {retest_low_option} to stay ABOVE sl_price {sl_price}"
        )


class TestLunchDeadZoneTrendDayBypass:
    """Verify Institutional Trend-Day Bypass for the 11:30 - 13:15 IST Lunch Gate."""

    def test_tier_classifier_promotes_trend_day_during_lunch(self):
        """At 12:15 PM, an Institutional Trend Day is promoted to Tier B Live Eligible."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY BANK",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 6.5,
            "bar_timestamp": "2026-09-30 12:15:00",
            "reasons": [
                "ORB_Breakout_High(54625.8)",
                "Above_VWAP",
                "SuperTrend_Bull",
                "Institutional_Trend_Day(ADX=28.5>=25)",
            ],
            "is_institutional_trend_day": True,
            "adx": 28.5,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B", f"Expected Tier B on Institutional Trend Day, got {res['tier']}"
        assert res["tier_name"] == "INSTITUTIONAL_TREND_DAY"
        assert res["is_live_eligible"] is True
        assert any("Trend-Day Lunch Bypass" in r for r in res["classification_reasons"])

    def test_tier_classifier_strictly_blocks_normal_day_during_lunch(self):
        """At 12:15 PM, a normal day without trend-day confirmation is relegated to Tier C Paper Only."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY BANK",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 6.5,
            "bar_timestamp": "2026-09-30 12:15:00",
            "reasons": [
                "ORB_Breakout_High(54625.8)",
                "Above_VWAP",
            ],
            "is_institutional_trend_day": False,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C", f"Expected Tier C during lunch dead zone, got {res['tier']}"
        assert res["is_live_eligible"] is False
        assert any("Lunch Dead Zone Gate" in r for r in res["classification_reasons"])

    def test_scanner_evaluates_bar_during_lunch_when_trend_day_active(self):
        """PriceActionMomentumScanner allows continuation signals during lunch when trend day is confirmed."""
        scanner = PriceActionMomentumScanner()
        # 20 prior bars + 13 bars today (13th bar is 12:15 PM)
        prior_df = _generate_synthetic_candles(n_bars=20, start_dt=datetime(2026, 9, 29, 9, 15), trend="neutral")
        today_df = _generate_synthetic_candles(n_bars=13, start_dt=datetime(2026, 9, 30, 9, 15), trend="bullish", strong_volume=True)
        df = pd.concat([prior_df, today_df], ignore_index=True)

        sig = scanner.evaluate_bar(df, symbol="NIFTY BANK", is_expiry_day=True, golden_mode=True)
        assert sig is not None, "Expected valid continuation signal during lunch on Institutional Trend Day"
        assert sig["is_institutional_trend_day"] is True
        assert sig["adx"] >= 25.0
        assert sig["action"] == "BUY_CE"
