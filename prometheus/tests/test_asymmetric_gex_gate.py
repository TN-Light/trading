"""
Unit and regression tests for the Asymmetric GEX × Direction Gate and gex_regime telemetry.
"""

import pytest
from datetime import datetime
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.papertrade.types import PaperTrade, Position, Direction, ExitReason
from prometheus.papertrade.recorder import TradeRecorder, TRADE_COLUMNS


class TestAsymmetricGEXGate:
    """Verify that Call Buying (LONG) in Positive GEX is demoted to Tier C,
    while Put Buying (SHORT) in Positive GEX remains Tier B eligible."""

    def test_call_buying_in_positive_gex_demoted_to_tier_c(self):
        """Call buying (BUY_CE) into positive GEX (Long Gamma) must be demoted to Tier C (Paper Only)."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "edge_score": 8.0,
            "bar_timestamp": "2026-10-27 10:16:00",
            "net_gex": 102068171427.54,  # +102k Cr (Positive GEX)
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "1H_Trend_Bullish",
            ],
            "is_0dte": False,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["tier_name"] == "STANDARD_MOMENTUM"
        assert res["is_live_eligible"] is False
        assert any("Asymmetric GEX Gate" in r for r in res["classification_reasons"])
        assert "TIER C" in res["tier_badge"]

    def test_put_buying_in_positive_gex_remains_tier_b_eligible(self):
        """Put buying (BUY_PE) into positive GEX (fade off resistance) remains Tier B Live Eligible."""
        signal = {
            "action": "BUY_PE",
            "symbol": "SENSEX",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "edge_score": 8.0,
            "bar_timestamp": "2026-10-08 10:30:00",
            "net_gex": 183916900000.0,  # +183k Cr (Positive GEX)
            "reasons": [
                "15M_ORB_Low_Breakout",
                "Session_VWAP_Bearish",
                "1H_Trend_Bearish",
            ],
            "is_0dte": False,
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B"
        assert res["tier_name"] == "REGIME_SCALP_LONG_GAMMA"
        assert res["is_live_eligible"] is True
        assert "TIER B: REGIME SCALP" in res["tier_badge"]

    def test_call_buying_in_negative_gex_remains_tier_s_eligible(self):
        """Call buying in negative GEX (Short Gamma Squeeze) during power hour is Tier S eligible."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "edge_score": 8.2,
            "bar_timestamp": "2026-09-11 09:45:00",
            "net_gex": -203901255170.21,  # Short Gamma
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

    def test_gex_regime_tagging_in_papertrade_and_position(self):
        """PaperTrade and Position dataclasses must automatically infer gex_regime."""
        pos_long = Position(
            trade_id="TEST-1",
            symbol="NIFTY 50",
            instrument="NIFTY27OCT2622750CE",
            underlying="NIFTY",
            direction=Direction.LONG,
            quantity=65,
            entry_price=81.63,
            entry_time=datetime.now(),
            stop_loss=69.23,
            target=107.43,
            max_bars=20,
            net_gex=102068171427.54,
        )
        assert pos_long.gex_regime == "LONG_GAMMA"

        pos_short = Position(
            trade_id="TEST-2",
            symbol="NIFTY 50",
            instrument="NIFTY27OCT2622600PE",
            underlying="NIFTY",
            direction=Direction.SHORT,
            quantity=65,
            entry_price=85.0,
            entry_time=datetime.now(),
            stop_loss=70.0,
            target=110.0,
            max_bars=20,
            net_gex=-5000000000.0,
        )
        assert pos_short.gex_regime == "SHORT_GAMMA"

        trade = PaperTrade(
            trade_id="TEST-3",
            symbol="NIFTY 50",
            instrument="NIFTY27OCT2622750CE",
            underlying="NIFTY",
            direction=Direction.LONG,
            quantity=65,
            entry_price=81.63,
            exit_price=69.23,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            exit_reason=ExitReason.STOP_LOSS,
            gross_pnl=-806.10,
            costs=72.50,
            net_pnl=-878.60,
            return_pct=-16.56,
            holding_duration_seconds=278,
            net_gex=102068171427.54,
        )
        assert trade.gex_regime == "LONG_GAMMA"
        assert "gex_regime" in TRADE_COLUMNS
