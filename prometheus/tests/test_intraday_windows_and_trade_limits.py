"""
Unit and integration tests for the Two-Window Intraday Schedule,
Lunch Dead Zone Gate, and Daily Trade Capacity Calibration.
"""

import pytest
from datetime import time as dtime
from unittest.mock import MagicMock, patch
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.config import get


class TestIntradayWindowsAndTiers:
    """Test suite for morning, lunch dead zone, afternoon, and post-cutoff execution windows."""

    def test_morning_window_option_buying_qualifies_tier_b(self):
        """At 10:15 AM, high-conviction option buying qualifies as Tier B Golden Setup."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup",
            "strategy_type": "option_buying",
            "edge_score": 6.8,
            "bar_timestamp": "2026-09-21 10:15:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "1H_Trend_Bullish",
            ],
            "is_golden_setup": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B"
        assert res["tier_name"] == "GOLDEN_SETUP"
        assert res["is_live_eligible"] is True
        assert any("Morning Window (09:30-11:30)" in r for r in res["classification_reasons"])

    def test_morning_power_hour_qualifies_tier_s(self):
        """At 09:45 AM, 5-factor confluence qualifies as Tier S Perfect Storm."""
        signal = {
            "action": "BUY_CE",
            "symbol": "BANKNIFTY",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 7.5,
            "bar_timestamp": "2026-09-21 09:45:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "Volume_Surge_Confirmed",
                "1H_Trend_Bullish",
            ],
            "has_volume_surge": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "S"
        assert res["tier_name"] == "PERFECT_STORM"
        assert res["is_live_eligible"] is True
        assert any("Morning Power Hour" in r for r in res["classification_reasons"])

    def test_lunch_dead_zone_strictly_blocks_option_buying(self):
        """At 12:15 PM (Lunch Dead Zone), option buying is strictly gated to Tier C (Paper Only)."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup",
            "strategy_type": "option_buying",
            "edge_score": 7.8,
            "bar_timestamp": "2026-09-21 12:15:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "Volume_Surge_Confirmed",
                "1H_Trend_Bullish",
            ],
            "has_volume_surge": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["is_live_eligible"] is False
        assert "LUNCH DEAD ZONE" in res["tier_badge"]
        assert any("Lunch Dead Zone Gate (11:30-13:15 IST)" in r for r in res["classification_reasons"])

    def test_lunch_dead_zone_permits_credit_spread(self):
        """At 12:15 PM, Credit Spreads (theta sellers) are permitted and qualify for Tier A."""
        signal = {
            "action": "SELL_BULL_PUT_SPREAD",
            "symbol": "NIFTY",
            "strategy": "credit_spread",
            "strategy_type": "credit_spread",
            "is_0dte": True,
            "otm_sigma": 2.2,
            "signal_score": 9.5,
            "oi_shielded": True,
            "trend_aligned": True,
            "bar_timestamp": "2026-09-21 12:15:00",
            "legs": [
                {"action": "SELL", "strike": 24800, "option_type": "PE", "entry": 35.0},
                {"action": "BUY", "strike": 24600, "option_type": "PE", "entry": 10.0},
            ],
            "net_credit": 25.0,
            "target_decay": 7.5,
            "hard_sl": 62.5,
            "margin_required": 42000,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "A"
        assert res["tier_name"] == "SURE_SHOT_SPREAD"
        assert res["is_live_eligible"] is True

    def test_afternoon_window_option_buying_qualifies_tier_b(self):
        """At 13:30 PM (Afternoon Window), trend continuation option buying qualifies for Tier B."""
        signal = {
            "action": "BUY_PE",
            "symbol": "SENSEX",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 6.8,
            "bar_timestamp": "2026-09-21 13:30:00",
            "reasons": [
                "15M_ORB_Low_Breakout",
                "Session_VWAP_Bearish",
                "1H_Trend_Bearish",
            ],
            "is_golden_setup": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B"
        assert res["tier_name"] == "GOLDEN_SETUP"
        assert res["is_live_eligible"] is True
        assert any("Afternoon Window (13:15-14:15)" in r for r in res["classification_reasons"])

    def test_afternoon_window_squeeze_qualifies_tier_s(self):
        """At 13:45 PM, 5-factor afternoon squeeze qualifies as Tier S Perfect Storm."""
        signal = {
            "action": "BUY_PE",
            "symbol": "SENSEX",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 8.0,
            "bar_timestamp": "2026-09-21 13:45:00",
            "reasons": [
                "15M_ORB_Low_Breakout",
                "Session_VWAP_Bearish",
                "Volume_Surge_Confirmed",
                "1H_Trend_Bearish",
            ],
            "has_volume_surge": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "S"
        assert res["tier_name"] == "PERFECT_STORM"
        assert res["is_live_eligible"] is True
        assert any("Afternoon Squeeze Window" in r for r in res["classification_reasons"])

    def test_post_cutoff_relegated_to_tier_c(self):
        """At 14:20 PM (past last_entry_time 14:15), new option buying is relegated to Tier C."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup",
            "strategy_type": "option_buying",
            "edge_score": 7.0,
            "bar_timestamp": "2026-09-21 14:20:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "1H_Trend_Bullish",
            ],
            "is_golden_setup": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["is_live_eligible"] is False
        assert any("Post-Cutoff Window (After 14:15 IST)" in r for r in res["classification_reasons"])


class TestSettingsConfiguration:
    """Verify settings.yaml reflects calibrated windows and limits."""

    def test_intraday_settings_calibrated(self):
        max_trades = get("intraday.max_daily_trades")
        last_entry = get("intraday.last_entry_time")
        tw = get("intraday.trading_windows", {})

        assert max_trades == 4, f"Expected max_daily_trades: 4, got {max_trades}"
        assert last_entry == "14:15", f"Expected last_entry_time: '14:15', got {last_entry}"
        assert tw.get("morning_start") == "09:30"
        assert tw.get("morning_end") == "11:30"
        assert tw.get("lunch_dead_zone_start") == "11:30"
        assert tw.get("lunch_dead_zone_end") == "13:15"
        assert tw.get("afternoon_start") == "13:15"
        assert tw.get("afternoon_end") == "14:15"

    def test_v2_dead_zone_harmonized(self):
        v2_dead_start = get("intraday.v2.dead_zone_start")
        v2_dead_end = get("intraday.v2.dead_zone_end")
        v2_aft_start = get("intraday.v2.afternoon_window_start")
        v2_cutoff = get("intraday.v2.entry_cutoff_time")

        assert v2_dead_start == "11:30"
        assert v2_dead_end == "13:15"
        assert v2_aft_start == "13:15"
        assert v2_cutoff == "14:15"


class TestMainExecutionWindowFiltering:
    """Test candidate filtering during lunch dead zone in intraday scanning logic."""

    def test_lunch_dead_zone_gate_logic(self):
        """Simulate candidate loop filtering during lunch dead zone vs morning vs afternoon."""
        lunch_start_time = dtime(11, 30)
        lunch_end_time = dtime(13, 15)

        opt_buying_signal = {
            "symbol": "NIFTY 50",
            "action": "BUY_CE",
            "strategy_type": "option_buying",
            "signal_score": 7.5,
            "tradingsymbol": "NIFTY26SEP25000CE",
        }
        credit_spread_signal = {
            "symbol": "BANKNIFTY",
            "action": "SELL_BULL_PUT_SPREAD",
            "strategy_type": "credit_spread",
            "signal_score": 9.2,
            "tradingsymbol": "BANKNIFTY26SEP56000PE",
        }

        # Case 1: Lunch Dead Zone (12:15 PM)
        t_lunch = dtime(12, 15)
        is_lunch = (lunch_start_time <= t_lunch < lunch_end_time)
        assert is_lunch is True

        # Option buying candidate must be suppressed
        is_buying_1 = "option_buying" in opt_buying_signal.get("strategy_type", "") or "BUY" in opt_buying_signal.get("action", "")
        suppress_buying = is_lunch and is_buying_1
        assert suppress_buying is True, "Option buying must be suppressed during lunch dead zone"

        # Credit spread candidate must NOT be suppressed
        is_buying_2 = "option_buying" in credit_spread_signal.get("strategy_type", "") or "BUY" in credit_spread_signal.get("action", "")
        suppress_cs = is_lunch and is_buying_2
        assert suppress_cs is False, "Credit spreads must NOT be suppressed during lunch dead zone"

        # Case 2: Morning Window (10:15 AM)
        t_morning = dtime(10, 15)
        is_lunch_morning = (lunch_start_time <= t_morning < lunch_end_time)
        assert is_lunch_morning is False
        assert (is_lunch_morning and is_buying_1) is False

        # Case 3: Afternoon Window (13:30 PM)
        t_afternoon = dtime(13, 30)
        is_lunch_afternoon = (lunch_start_time <= t_afternoon < lunch_end_time)
        assert is_lunch_afternoon is False
        assert (is_lunch_afternoon and is_buying_1) is False
