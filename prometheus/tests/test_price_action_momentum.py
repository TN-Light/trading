"""
Unit tests for PriceActionMomentumScanner (Prometheus 2.0 Momentum Engine).
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta
import pytest

from prometheus.signals.price_action_momentum import PriceActionMomentumScanner


def _generate_sample_candles(n_bars=30, start_dt=datetime(2026, 8, 24, 9, 15), trend="bullish"):
    """Generate realistic 15-minute synthetic OHLCV candles."""
    timestamps = [start_dt + timedelta(minutes=15 * i) for i in range(n_bars)]
    base_price = 24000.0
    
    rows = []
    current = base_price
    for i, ts in enumerate(timestamps):
        if trend == "bullish":
            step = 15.0 if i > 2 else 5.0
        elif trend == "bearish":
            step = -15.0 if i > 2 else -5.0
        else:
            step = (i % 3 - 1) * 5.0
            
        open_p = current
        high_p = open_p + 10.0 + max(0, step)
        low_p = open_p - 10.0 + min(0, step)
        close_p = open_p + step
        vol = 100000 + i * 5000
        
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


def test_pa_scanner_open_hours_gate():
    """Verify scanner skips the 09:15 - 09:45 morning chop."""
    scanner = PriceActionMomentumScanner()
    # Candle at 09:30 AM
    df_early = _generate_sample_candles(n_bars=2, start_dt=datetime(2026, 8, 24, 9, 15), trend="bullish")
    sig = scanner.evaluate_bar(df_early, symbol="NIFTY 50")
    assert sig is None, "Expected None before 09:50 AM open gate"


def test_pa_scanner_bullish_orb_breakout():
    """Verify scanner detects clean ORB high breakout after 09:50 AM."""
    scanner = PriceActionMomentumScanner()
    # Prior days bars (20 bars) + today's 6 bars (09:15 to 10:30)
    prior_df = _generate_sample_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), trend="neutral")
    today_df = _generate_sample_candles(n_bars=6, start_dt=datetime(2026, 8, 24, 9, 15), trend="bullish")
    df = pd.concat([prior_df, today_df], ignore_index=True)
    
    sig = scanner.evaluate_bar(df, symbol="NIFTY 50")
    assert sig is not None, "Expected valid BUY_CE signal on bullish ORB breakout"
    assert sig["action"] == "BUY_CE"
    assert sig["direction"] == "bullish"
    assert sig["stop_loss"] < sig["entry_price"]
    assert sig["target"] > sig["entry_price"]
    assert sig["confidence"] >= 0.60


def test_pa_scanner_bearish_orb_breakdown():
    """Verify scanner detects clean ORB low breakdown after 09:50 AM."""
    scanner = PriceActionMomentumScanner()
    # Prior days bars (20 bars) + today's 6 bars (09:15 to 10:30)
    prior_df = _generate_sample_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), trend="neutral")
    today_df = _generate_sample_candles(n_bars=6, start_dt=datetime(2026, 8, 24, 9, 15), trend="bearish")
    df = pd.concat([prior_df, today_df], ignore_index=True)
    
    sig = scanner.evaluate_bar(df, symbol="NIFTY 50")
    assert sig is not None, "Expected valid BUY_PE signal on bearish ORB breakdown"
    assert sig["action"] == "BUY_PE"
    assert sig["direction"] == "bearish"
    assert sig["stop_loss"] > sig["entry_price"]
    assert sig["target"] < sig["entry_price"]
    assert sig["confidence"] >= 0.60


def test_golden_setup_1h_trend_veto():
    """Verify 1H higher timeframe trend alignment vetoes counter-trend setups."""
    scanner = PriceActionMomentumScanner()
    prior_df = _generate_sample_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), trend="neutral")
    today_df = _generate_sample_candles(n_bars=6, start_dt=datetime(2026, 8, 24, 9, 15), trend="bearish")
    df = pd.concat([prior_df, today_df], ignore_index=True)

    # 1H candles starting Aug 21 showing strong BULLISH trend (price 24500 >> EMA20)
    h_rows = [
        {"timestamp": datetime(2026, 8, 21, 9, 15) + timedelta(hours=i), "open": 24000 + i*50, "high": 24050 + i*50, "low": 23990 + i*50, "close": 24040 + i*50, "volume": 10000}
        for i in range(30)
    ]
    df_1h_bull = pd.DataFrame(h_rows)

    # In a bullish 1H trend, bearish 15m PE breakdown MUST be vetoed
    sig = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h_bull, golden_mode=True)
    assert sig is None, "Expected bearish PE setup to be vetoed by Bullish 1-Hour HTF trend"


def test_golden_setup_1300_cutoff_on_non_expiry():
    """Verify 13:00 hard cutoff eliminates afternoon mean-reversion trades on non-expiry days."""
    scanner = PriceActionMomentumScanner()
    # 20 bars prior + 18 bars today (ends at 13:30 PM)
    prior_df = _generate_sample_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), trend="neutral")
    today_df = _generate_sample_candles(n_bars=18, start_dt=datetime(2026, 8, 24, 9, 15), trend="bearish")
    df = pd.concat([prior_df, today_df], ignore_index=True)

    sig = scanner.evaluate_bar(df, symbol="NIFTY 50", is_expiry_day=False, golden_mode=True)
    assert sig is None, "Expected trade after 13:00 to be blocked on non-expiry day"


def test_golden_setup_confluence_tag():
    """Verify Golden Setup tag when 1H Trend + VWAP + 15M ORB align."""
    scanner = PriceActionMomentumScanner()
    prior_df = _generate_sample_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), trend="neutral")
    today_df = _generate_sample_candles(n_bars=6, start_dt=datetime(2026, 8, 24, 9, 15), trend="bearish")
    df = pd.concat([prior_df, today_df], ignore_index=True)

    # 1H candles starting Aug 21 showing matching BEARISH trend
    h_rows = [
        {"timestamp": datetime(2026, 8, 21, 9, 15) + timedelta(hours=i), "open": 24500 - i*50, "high": 24510 - i*50, "low": 24440 - i*50, "close": 24450 - i*50, "volume": 10000}
        for i in range(30)
    ]
    df_1h_bear = pd.DataFrame(h_rows)

    sig = scanner.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h_bear, golden_mode=True)
    assert sig is not None, "Expected valid Golden Setup signal"
    assert sig["is_golden_setup"] is True
    assert "Golden_Setup" in sig["strategy"]

