"""
Unit tests for the Prometheus 5-Tier Signal Classification and Execution Gating.
"""

import pytest
from unittest.mock import MagicMock
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.interface.telegram_bot import TelegramBot


class TestTierClassifier:
    """Tests for 5-tier pyramid signal classification."""

    def test_tier_s_perfect_storm(self):
        """Tier S requires 15M ORB + VWAP + Volume Surge + Strict 1H Trend + Score >= 7.0 + Morning Power Hour."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 8.2,
            "bar_timestamp": "2026-09-11 09:45:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "Volume_Surge_Confirmed (1.35x)",
                "1H_Trend_Bullish",
            ],
            "has_volume_surge": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "S"
        assert res["tier_name"] == "PERFECT_STORM"
        assert res["is_live_eligible"] is True
        assert "TIER S" in res["tier_badge"]
        assert "Live Trade" in res["action_instruction"]

    def test_tier_s_falls_back_to_tier_c_if_htf_neutral(self):
        """If 1H trend is NEUTRAL (chop), signal cannot be Golden Setup (Tier B); it is safely relegated to Tier C (Paper Only)."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 7.5,
            "bar_timestamp": "2026-09-11 09:45:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "Volume_Surge_Confirmed",
                "1H_Trend_Neutral",
            ],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["tier_name"] == "STANDARD_MOMENTUM"
        assert res["is_live_eligible"] is False

    def test_tier_s_falls_back_to_tier_b_outside_power_hour(self):
        """After 10:35 AM, an impulse signal is classified as Tier B, not Tier S."""
        signal = {
            "action": "BUY_CE",
            "symbol": "BANKNIFTY",
            "strategy": "PriceAction_Momentum",
            "strategy_type": "option_buying",
            "edge_score": 7.8,
            "bar_timestamp": "2026-09-11 11:15:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "Volume_Surge_Confirmed",
                "1H_Trend_Bullish",
            ],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B"
        assert res["tier_name"] == "GOLDEN_SETUP"
        assert res["is_live_eligible"] is True

    def test_tier_a_sure_shot_credit_spread(self):
        """Tier A requires 0-DTE + >=1.95σ OTM + Trend Aligned + Score >= 9.0."""
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
        assert "TIER A" in res["tier_badge"]

    def test_tier_c_credit_spread_non_0dte(self):
        """Non-0DTE spreads are Tier C (paper trading only)."""
        signal = {
            "action": "SELL_BEAR_CALL_SPREAD",
            "symbol": "BANKNIFTY",
            "strategy": "credit_spread",
            "strategy_type": "credit_spread",
            "is_0dte": False,
            "otm_sigma": 2.1,
            "signal_score": 7.8,
            "trend_aligned": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["tier_name"] == "STANDARD_SPREAD"
        assert res["is_live_eligible"] is False
        assert "TIER C" in res["tier_badge"]

    def test_tier_b_golden_setup(self):
        """Tier B Golden Setup requires strict 1H trend alignment, morning window, and >=6.5 score."""
        signal = {
            "action": "BUY_PE",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "edge_score": 6.8,
            "bar_timestamp": "2026-09-11 10:00:00",
            "reasons": [
                "15M_ORB_Low_Breakout",
                "Session_VWAP_Bearish",
                "1H_Trend_Bearish",
            ],
            "is_0dte": False,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B"
        assert res["tier_name"] == "GOLDEN_SETUP"
        assert res["is_live_eligible"] is True

    def test_0dte_option_buying_gated_to_tier_c_unless_tier_s(self):
        """On 0-DTE expiry sessions, moderate/Tier-B option buying is gated to Tier C (Paper Only) to prevent theta burn."""
        signal = {
            "action": "BUY_CE",
            "symbol": "SENSEX",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "edge_score": 6.8,
            "bar_timestamp": "2026-09-17 10:00:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "1H_Trend_Bullish",
            ],
            "is_0dte": True,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["tier_name"] == "STANDARD_MOMENTUM"
        assert res["is_live_eligible"] is False
        assert any("0-DTE Expiry Option Buying gated" in r for r in res["classification_reasons"])

    def test_tier_c_standard_momentum(self):
        """Score >= 3.5 but missing primary ORB or VWAP."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "edge_score": 3.8,
            "reasons": ["RSI_Oversold", "Fast_MACD"],
            "bar_timestamp": "2026-09-11 10:00:00",
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["tier_name"] == "STANDARD_MOMENTUM"
        assert res["is_live_eligible"] is False

    def test_tier_d_observe_only(self):
        """Score < 3.5 is filtered / observe only."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "edge_score": 2.1,
            "reasons": ["Minor_Tick"],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "D"
        assert res["tier_name"] == "OBSERVE_ONLY"
        assert res["is_live_eligible"] is False

    def test_hold_action_is_tier_d(self):
        """Action HOLD is always Tier D."""
        signal = {"action": "HOLD", "symbol": "NIFTY 50"}
        res = classify_signal_tier(signal)
        assert res["tier"] == "D"
        assert res["is_live_eligible"] is False


class TestTelegramAlertFormatting:
    """Test that Telegram alert rendering correctly displays the 5-tier pyramid."""

    def test_directional_alert_contains_tier_badge(self):
        bot = TelegramBot()
        bot.send_message = MagicMock()

        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "edge_score": 8.0,
            "bar_timestamp": "2026-09-11 09:45:00",
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "Volume_Surge_Confirmed",
                "1H_Trend_Bullish",
            ],
            "has_volume_surge": True,
            "entry": 145.0,
            "sl": 125.0,
            "target": 185.0,
            "strike": 24900,
            "option_type": "CE",
            "expiry": "2026-09-17",
            "leaderboard_rank": 1,
        }

        bot.alert_new_signal(signal)
        assert bot.send_message.called
        msg = bot.send_message.call_args[0][0]
        assert "RANK #1 SIGNAL" in msg
        assert "TIER S: PERFECT STORM" in msg
        assert "Live Trade" in msg

    def test_credit_spread_alert_contains_tier_badge(self):
        bot = TelegramBot()
        bot.send_message = MagicMock()

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
            "legs": [
                {"action": "SELL", "strike": 24800, "option_type": "PE", "entry": 35.0},
                {"action": "BUY", "strike": 24600, "option_type": "PE", "entry": 10.0},
            ],
            "net_credit": 25.0,
            "target_decay": 7.5,
            "hard_sl": 62.5,
            "margin_required": 42000,
            "leaderboard_rank": 1,
        }

        bot.alert_new_signal(signal)
        assert bot.send_message.called
        msg = bot.send_message.call_args[0][0]
        assert "TIER A: HIGH CONVICTION SPREAD" in msg
        assert "Live Trade" in msg


class TestPrometheusRuntimeInitialization:
    """Regression test: Ensure Prometheus __init__ does not truncate prematurely."""

    def test_prometheus_runtime_attributes_initialized(self):
        from prometheus.main import Prometheus
        p = Prometheus(mode_override="paper")
        assert hasattr(p, "_scan_lock"), "Prometheus must have _scan_lock initialized"
        assert p._scan_lock is not None
        assert hasattr(p, "_alerted_signals"), "Prometheus must have _alerted_signals initialized"
        assert isinstance(p._alerted_signals, dict)
        assert hasattr(p, "_last_trade_reject_alerts")
        assert hasattr(p, "gamma_ambush_log_file")
        assert hasattr(p, "_intraday_guardrail_audit")
        assert hasattr(p, "paper_capture")


class TestCommitmentRatioTelemetry:
    """Test suite for Option C: Institutional Commitment Ratio (|ΔOI| / Volume) shadow telemetry."""

    def test_oi_analyzer_calculates_commitment_ratio(self):
        from prometheus.signals.oi_analyzer import OIAnalyzer
        import pandas as pd

        analyzer = OIAnalyzer()
        chain_df = pd.DataFrame([
            {"option_type": "CE", "strike": 24000.0, "oi": 50000, "oi_change": 12000, "volume": 30000},
            {"option_type": "PE", "strike": 24000.0, "oi": 60000, "oi_change": 8000, "volume": 20000},
        ])
        res = analyzer.analyze(chain_df, spot_price=24000.0)
        metrics = res.get("metrics", {})
        assert "commitment_ratio" in metrics
        # (|12000| + |8000|) / (30000 + 20000) = 20000 / 50000 = 0.40
        assert metrics["commitment_ratio"] == 0.40

    def test_tier_classifier_preserves_commitment_ratio(self):
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "edge_score": 8.5,
            "bar_timestamp": "2026-09-15 09:45:00",
            "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "Volume_Surge_Confirmed", "1H_Trend_Bullish"],
            "has_volume_surge": True,
            "commitment_ratio": 0.45,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "S"
        assert res["commitment_ratio"] == 0.45
        assert any("Institutional Commitment: 0.45" in r for r in res["classification_reasons"])

    def test_telegram_alert_displays_commitment_ratio(self):
        bot = TelegramBot()
        bot.send_message = MagicMock()

        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "edge_score": 8.0,
            "bar_timestamp": "2026-09-15 09:45:00",
            "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish"],
            "entry": 145.0,
            "sl": 125.0,
            "target": 185.0,
            "strike": 24900,
            "option_type": "CE",
            "expiry": "2026-09-17",
            "commitment_ratio": 0.42,
            "leaderboard_rank": 1,
        }
        bot.alert_new_signal(signal)
        assert bot.send_message.called
        msg = bot.send_message.call_args[0][0]
        assert "Commitment: 0.42" in msg


