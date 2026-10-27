"""
Unit and invariant tests for the Asymmetric GEX x Direction Gating Policy in Tier Classifier.

Verifies:
1. In Dealer Long Gamma (Net GEX > +1.5 Cr):
   - Call Buying (LONG) is strictly gated to Tier C (Paper Only), even with Score >= 8.0.
   - Put Buying (SHORT) is permitted for Tier B (Live Eligible) when Score >= 8.0 (Ceiling Fade Scalp).
2. In Dealer Short/Neutral Gamma (Net GEX <= +1.5 Cr):
   - Both Call Buying (LONG) and Put Buying (SHORT) are permitted for Tier B Golden Setups.
3. Telegram HTML safety: No unescaped raw '<' characters in classification reasons.
"""

import pytest
from prometheus.signals.tier_classifier import classify_signal_tier


class TestAsymmetricGEXDirectionGate:
    """Empirical invariant tests for Asymmetric GEX x Direction gating."""

    def test_call_buying_in_strong_long_gamma_gated_to_tier_c(self):
        """Today's exact setup (Oct 27 NIFTY 10:16 CE with +102k Cr GEX) must be gated to Tier C."""
        signal = {
            "action": "BUY_CE",
            "direction": "LONG",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "signal_score": 8.0,
            "bar_timestamp": "2026-10-27 10:15:00",
            "net_gex": 1.02e11,  # +102,000 Cr
            "net_gex_cr": 10200.0,
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "1H_Trend_Bullish",
            ],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C", f"Expected Tier C for call buying in Long Gamma, got {res['tier']}"
        assert res["is_live_eligible"] is False, "Call buying in Long Gamma must NOT be live-eligible"
        assert res["tier_name"] == "STANDARD_MOMENTUM"
        assert any("Asymmetric GEX Gate: Call Buying (LONG) strictly gated" in r for r in res["classification_reasons"])

    def test_sensex_call_buying_in_long_gamma_gated_to_tier_c(self):
        """Today's exact setup (Oct 27 SENSEX 10:16 CE with +16.5k Cr GEX) must be gated to Tier C."""
        signal = {
            "action": "BUY_CE",
            "direction": "LONG",
            "symbol": "SENSEX",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "signal_score": 8.0,
            "bar_timestamp": "2026-10-27 10:15:00",
            "net_gex": 1.65e10,  # +16,500 Cr
            "net_gex_cr": 1650.0,
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "1H_Trend_Bullish",
            ],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["is_live_eligible"] is False
        assert any("Asymmetric GEX Gate: Call Buying (LONG) strictly gated" in r for r in res["classification_reasons"])

    def test_put_buying_in_strong_long_gamma_promoted_to_tier_b(self):
        """Oct 8 setup (SENSEX/NIFTY PE with +50k to +184k Cr GEX) must be permitted for Tier B in valid entry window."""
        signal = {
            "action": "BUY_PE",
            "direction": "SHORT",
            "symbol": "SENSEX",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "signal_score": 8.0,
            "bar_timestamp": "2026-10-08 10:15:00",
            "net_gex": 1.83e11,  # +183,000 Cr
            "net_gex_cr": 18300.0,
            "reasons": [
                "15M_ORB_Low_Breakout",
                "Session_VWAP_Bearish",
                "1H_Trend_Bearish",
            ],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B", f"Expected Tier B for put buying in Long Gamma, got {res['tier']}"
        assert res["is_live_eligible"] is True, "Put buying fading ceiling in Long Gamma must be live-eligible"
        assert res["tier_name"] == "REGIME_SCALP_LONG_GAMMA"
        assert any("Asymmetric GEX Gate: Put Buying (SHORT) permitted" in r for r in res["classification_reasons"])

    def test_put_buying_in_long_gamma_requires_score_8(self):
        """Put buying in Long Gamma with score < 8.0 is relegated to Tier C paper observation."""
        signal = {
            "action": "BUY_PE",
            "direction": "SHORT",
            "symbol": "NIFTY 50",
            "strategy": "PriceAction_Momentum",
            "signal_score": 7.0,  # Below 8.0
            "bar_timestamp": "2026-10-08 10:00:00",
            "net_gex": 5.0e10,
            "net_gex_cr": 5000.0,
            "reasons": [
                "15M_ORB_Low_Breakout",
                "Session_VWAP_Bearish",
                "1H_Trend_Bearish",
            ],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["is_live_eligible"] is False
        assert any("Requires Score >= 8.0 for Tier B Put Scalp" in r for r in res["classification_reasons"])

    def test_call_buying_in_short_gamma_promoted_to_tier_b(self):
        """Call buying during Short Gamma (Oct 26 Squeeze @ -200k Cr GEX) is allowed for Tier B."""
        signal = {
            "action": "BUY_CE",
            "direction": "LONG",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup (1H+VWAP+ORB)",
            "is_golden_setup": True,
            "signal_score": 8.0,
            "bar_timestamp": "2026-10-26 10:30:00",
            "net_gex": -2.0e11,  # Short Gamma
            "net_gex_cr": -20000.0,
            "reasons": [
                "15M_ORB_High_Breakout",
                "Session_VWAP_Bullish",
                "1H_Trend_Bullish",
            ],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "B"
        assert res["is_live_eligible"] is True
        assert res["tier_name"] == "GOLDEN_SETUP"

    def test_reasons_contain_no_raw_less_than_symbols(self):
        """Verify HTML Telegram safety: No raw '<' in classification reasons."""
        call_signal = {
            "action": "BUY_CE",
            "direction": "LONG",
            "symbol": "NIFTY 50",
            "signal_score": 8.0,
            "bar_timestamp": "2026-10-27 10:15:00",
            "net_gex": 1.0e11,
            "net_gex_cr": 10000.0,
            "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        }
        res_call = classify_signal_tier(call_signal)
        for r in res_call["classification_reasons"]:
            assert "<" not in r, f"Found raw '<' in reason: {r}"

        put_signal = {
            "action": "BUY_PE",
            "direction": "SHORT",
            "symbol": "NIFTY 50",
            "signal_score": 7.0,
            "bar_timestamp": "2026-10-27 10:15:00",
            "net_gex": 1.0e11,
            "net_gex_cr": 10000.0,
            "reasons": ["15M_ORB_Low_Breakout", "Session_VWAP_Bearish", "1H_Trend_Bearish"],
        }
        res_put = classify_signal_tier(put_signal)
        for r in res_put["classification_reasons"]:
            assert "<" not in r, f"Found raw '<' in reason: {r}"
