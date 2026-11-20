"""Tests for Phase 2: Trend Day Adaptive Microstructure Gates.

Verifies:
1. Dynamic VWAP Gate: 60 bps ceiling on normal days/weak closes vs 160 bps on confirmed Trend Days / strong closes (CLV >= 0.70).
2. Cutoff Extension: 14:15 cutoff on normal days vs 15:00 on non-expiry confirmed Trend Days.
3. Ignition Lockout Reset: Permits exactly 1 secondary re-entry on confirmed ignition bar in Tier C paper sandbox.
"""

from datetime import time as dtime
import pandas as pd
import pytest

from prometheus.signals.price_action_momentum import PriceActionMomentumScanner
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig


def _create_mock_df(close_p: float, vwap_p: float, clv: float = 0.80, adx: float = 28.0) -> pd.DataFrame:
    """Create a synthetic 15-minute DataFrame with controlled VWAP and CLV."""
    high_p = close_p + 20.0
    low_p = high_p - (20.0 / max(clv, 0.01))
    open_p = low_p + 5.0
    
    rows = []
    # 20 historical bars
    for i in range(20):
        t_str = f"2026-11-20 09:{15 + i*15 if i < 3 else 45}:00"
        rows.append({
            "timestamp": pd.to_datetime(f"2026-11-20 10:{i:02d}:00"),
            "open": 63000.0,
            "high": 63100.0,
            "low": 62900.0,
            "close": 63050.0,
            "volume": 1000,
        })
    # Add current bar
    rows.append({
        "timestamp": pd.to_datetime("2026-11-20 12:00:00"),
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": 2500,
    })
    return pd.DataFrame(rows)


class TestDynamicVWAPClimaxGate:
    """Verify dynamic 60 bps -> 160 bps expansion on strong closes / trend days."""

    def test_normal_weak_close_vetoed_at_70bps(self):
        """On weak close (CLV < 0.70) without trend day, 70 bps (> 60 bps) is vetoed."""
        scanner = PriceActionMomentumScanner()
        # Close 63,450 vs VWAP 63,000 is +71 bps (> 60 bps)
        # Low CLV = 0.40 (weak close near low)
        bar_range = 100.0
        high_p = 63510.0
        low_p = high_p - bar_range
        close_p = low_p + (0.40 * bar_range)  # CLV = 0.40
        open_p = low_p + 80.0  # Red candle / weak close
        
        df = pd.DataFrame([
            {"timestamp": pd.to_datetime(f"2026-11-20 09:{15+i}:00"), "open": 63000, "high": 63050, "low": 62950, "close": 63000, "volume": 1000}
            for i in range(20)
        ] + [
            {"timestamp": pd.to_datetime("2026-11-20 10:30:00"), "open": open_p, "high": high_p, "low": low_p, "close": close_p, "volume": 1000}
        ])
        sig = scanner.evaluate_bar(df, symbol="NIFTY BANK", is_expiry_day=False, golden_mode=False)
        # Should be None because of weak close + vwap distance
        assert sig is None or sig.get("action") != "BUY_CE"

    def test_strong_close_permitted_at_100bps_on_trend_day(self):
        """On strong close (CLV >= 0.70), 100 bps extension is permitted (under 160 bps)."""
        scanner = PriceActionMomentumScanner()
        # High CLV = 0.90
        bar_range = 250.0
        high_p = 63650.0
        low_p = high_p - bar_range
        close_p = high_p - 10.0  # CLV = 240/250 = 0.96
        open_p = low_p + 20.0
        
        # 25 bars with strong trend
        rows = [
            {"timestamp": pd.to_datetime(f"2026-11-20 09:15:00"), "open": 62650, "high": 63000, "low": 62600, "close": 62900, "volume": 2000},
            {"timestamp": pd.to_datetime(f"2026-11-20 09:30:00"), "open": 62900, "high": 63150, "low": 62850, "close": 63100, "volume": 2500},
        ]
        for i in range(2, 22):
            rows.append({
                "timestamp": pd.to_datetime(f"2026-11-20 10:{i:02d}:00"),
                "open": 63100 + i*10, "high": 63150 + i*10, "low": 63050 + i*10, "close": 63140 + i*10, "volume": 1500
            })
        rows.append({
            "timestamp": pd.to_datetime("2026-11-20 13:30:00"),
            "open": open_p, "high": high_p, "low": low_p, "close": close_p, "volume": 4000
        })
        df = pd.DataFrame(rows)
        # Create 1H trend alignment
        df_1h = pd.DataFrame([
            {"timestamp": pd.to_datetime(f"2026-11-20 {h:02d}:00:00"), "open": 62500 + h*100, "high": 62700 + h*100, "low": 62450 + h*100, "close": 62650 + h*100, "volume": 10000}
            for h in range(9, 14)
        ])
        sig = scanner.evaluate_bar(df, symbol="NIFTY BANK", is_expiry_day=False, df_1h=df_1h, golden_mode=True)
        # Should generate BUY_CE because 100 bps is within the 160 bps threshold on strong close
        assert sig is not None
        assert sig.get("action") == "BUY_CE"


class TestCutoffExtension:
    """Verify 14:15 normal cutoff vs 15:00 on non-expiry trend days."""

    def test_trend_day_permits_entry_at_1445(self):
        """Confirmed trend day with non-0DTE permits entry up to 15:00."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup",
            "strategy_type": "option_buying",
            "edge_score": 8.0,
            "bar_timestamp": "2026-11-20 14:45:00",
            "expiry": "2026-11-23",  # 3 DTE (non-0DTE)
            "is_institutional_trend_day": True,
            "is_golden_setup": True,
            "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] in ("S", "B")
        assert res["is_live_eligible"] is True

    def test_0dte_expiry_blocks_entry_at_1445(self):
        """0-DTE expiry strictly blocks entry at 14:45 to prevent terminal theta decay."""
        signal = {
            "action": "BUY_CE",
            "symbol": "NIFTY 50",
            "strategy": "Golden_Setup",
            "strategy_type": "option_buying",
            "edge_score": 8.0,
            "bar_timestamp": "2026-11-20 14:45:00",
            "expiry": "2026-11-20",  # 0-DTE!
            "is_institutional_trend_day": True,
            "is_golden_setup": True,
            "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        }
        res = classify_signal_tier(signal)
        assert res["tier"] == "C"
        assert res["is_live_eligible"] is False


class TestIgnitionLockoutReset:
    """Verify conditional 1-time re-entry on ignition bar for Tier C paper sandbox."""

    def test_ignition_bar_allows_single_reentry_in_tier_c(self):
        class DummyFeed:
            def get_ltp(self, inst): return 100.0

        cfg = CaptureConfig(enabled=True)
        capture = LivePaperCapture(ltp_source=DummyFeed(), config=cfg)

        # Simulate stop-out recorded on NIFTY BANK BUY_CE
        capture._today_stopped_out_symbol_directions.add(("NIFTY BANK", "BUY_CE"))

        # Normal signal without ignition is locked out
        normal_sig = {"symbol": "NIFTY BANK", "action": "BUY_CE", "tier": "C", "is_ignition_bar": False}
        assert capture.is_direction_stopped_out("NIFTY BANK", refined_signal=normal_sig, tier="C") is True

        # Signal with ignition bar permits 1 re-entry
        ignition_sig = {"symbol": "NIFTY BANK", "action": "BUY_CE", "tier": "C", "is_ignition_bar": True}
        assert capture.is_direction_stopped_out("NIFTY BANK", refined_signal=ignition_sig, tier="C") is False

        # Simulate 1 re-entry taken
        capture._today_reentry_counts[("NIFTY BANK", "BUY_CE")] = 1

        # Second re-entry is strictly blocked
        assert capture.is_direction_stopped_out("NIFTY BANK", refined_signal=ignition_sig, tier="C") is True
