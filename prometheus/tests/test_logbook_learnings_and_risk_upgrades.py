"""
Unit tests for the 3 Logbook-driven risk and execution upgrades:
1. Low-VIX Dynamic Conviction Filter (VIX < 11.5 demands score >= 4.5)
2. Max Nominal Premium Exposure Cap (Lot cost <= Rs 15,000)
3. Same-Strike Lockout & Profit-Locked Pyramiding (+10% profit requirement OR score >= 5.0 override)
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from prometheus.main import Prometheus


class DummyPosState:
    def __init__(self, tradingsymbol, entry_premium):
        self.tradingsymbol = tradingsymbol
        self.entry_premium = entry_premium


def _create_dummy_df():
    dates = pd.date_range("2026-08-27 09:15", periods=20, freq="15min")
    return pd.DataFrame({
        "timestamp": dates,
        "open": [24200.0] * 20,
        "high": [24220.0] * 20,
        "low": [24180.0] * 20,
        "close": [24190.0] * 20,
        "volume": [10000] * 20,
    })


def test_low_vix_gate_suppresses_weak_momentum():
    """When VIX < 11.5, Option Buying signals with edge_score < 4.5 must be suppressed."""
    p = Prometheus()
    p.mode = "paper"
    p.data.get_vix = MagicMock(return_value=10.5)  # Extreme low-VIX
    p.data.fetch_intraday = MagicMock(return_value=_create_dummy_df())
    
    # Mock credit spread strategy to return None so we test Option Buying suppression
    p._credit_spread_strategy = MagicMock()
    p._credit_spread_strategy.evaluate_spread.return_value = None
    
    # Mock Angel One option chain to return a valid LTP
    p.data.angelone_options = MagicMock()
    p.data.angelone_options.get_real_premium.return_value = {
        "ltp": 85.0,
        "tradingsymbol": "NIFTY01SEP2624200PE",
    }
    
    # PA Momentum signal with low score 3.8 (< 4.5)
    pa_sig = {
        "symbol": "NIFTY 50",
        "action": "BUY_PE",
        "direction": "bearish",
        "strike": 24200,
        "option_type": "PE",
        "edge_score": 3.8,
        "underlying_price": 24180.0,
    }
    
    p._pa_momentum_scanner = MagicMock()
    p._pa_momentum_scanner.evaluate_bar.return_value = pa_sig
    
    signal = p._get_intraday_signal_for_execution("NIFTY 50", "15minute", False)
    # Should be suppressed (None) because VIX=10.5 and score=3.8 < 4.5
    assert signal is None


def test_low_vix_gate_allows_high_conviction_breakout():
    """When VIX < 11.5, Option Buying signals with edge_score >= 4.5 are allowed."""
    p = Prometheus()
    p.mode = "paper"
    p.data.get_vix = MagicMock(return_value=10.5)  # Low-VIX
    p.data.fetch_intraday = MagicMock(return_value=_create_dummy_df())
    
    p._credit_spread_strategy = MagicMock()
    p._credit_spread_strategy.evaluate_spread.return_value = None
    
    p.data.angelone_options = MagicMock()
    p.data.angelone_options.get_real_premium.return_value = {
        "ltp": 85.0,
        "tradingsymbol": "NIFTY01SEP2624200PE",
    }
    
    # PA Momentum signal with high score 5.0 (>= 4.5)
    pa_sig = {
        "symbol": "NIFTY 50",
        "action": "BUY_PE",
        "direction": "bearish",
        "strike": 24200,
        "option_type": "PE",
        "edge_score": 5.0,
        "underlying_price": 24180.0,
    }
    
    p._pa_momentum_scanner = MagicMock()
    p._pa_momentum_scanner.evaluate_bar.return_value = pa_sig
    
    signal = p._get_intraday_signal_for_execution("NIFTY 50", "15minute", False)
    assert signal is not None
    assert signal.get("action") == "BUY_PE"
    assert signal.get("signal_score") == 5.0


def test_max_nominal_capital_cap_blocks_expensive_contracts():
    """Contracts with total single-lot cost > Rs 15,000 must be skipped to avoid oversize risk."""
    p = Prometheus()
    p.mode = "paper"
    p.data.get_vix = MagicMock(return_value=14.0)
    p.data.fetch_intraday = MagicMock(return_value=_create_dummy_df())
    
    p._credit_spread_strategy = MagicMock()
    p._credit_spread_strategy.evaluate_spread.return_value = None
    
    p.data.angelone_options = MagicMock()
    # BankNifty lot size 30 * LTP 900 = Rs 27,000 (> Rs 15,000 cap)
    p.data.angelone_options.get_real_premium.return_value = {
        "ltp": 900.0,
        "tradingsymbol": "BANKNIFTY29SEP2657900CE",
    }
    
    pa_sig = {
        "symbol": "NIFTY BANK",
        "action": "BUY_CE",
        "direction": "bullish",
        "strike": 57900,
        "option_type": "CE",
        "edge_score": 4.5,
        "underlying_price": 57800.0,
    }
    
    p._pa_momentum_scanner = MagicMock()
    p._pa_momentum_scanner.evaluate_bar.return_value = pa_sig
    
    signal = p._get_intraday_signal_for_execution("NIFTY BANK", "15minute", False)
    assert signal is None


def test_same_strike_lockout_blocks_repeat_entries_in_loss_with_moderate_score():
    """Repeat entry on same strike must be blocked if active position is not in >= +10% profit and score < 5.0."""
    p = Prometheus()
    p.mode = "paper"
    p.data.get_vix = MagicMock(return_value=14.0)
    p.data.fetch_intraday = MagicMock(return_value=_create_dummy_df())
    
    p._credit_spread_strategy = MagicMock()
    p._credit_spread_strategy.evaluate_spread.return_value = None
    
    # Mark instrument as traded today
    p._today_traded_instruments = {"NIFTY01SEP2624200PE"}
    
    # Position monitor has active position entered at 86.0, current LTP is 87.0 (only +1.1% profit, < 10%)
    p.position_monitor = MagicMock()
    p.position_monitor.get_positions.return_value = {
        "pos_1": DummyPosState("NIFTY01SEP2624200PE", entry_premium=86.0)
    }
    
    p.data.angelone_options = MagicMock()
    p.data.angelone_options.get_real_premium.return_value = {
        "ltp": 87.0,
        "tradingsymbol": "NIFTY01SEP2624200PE",
    }
    
    # Moderate score 4.0 (< 5.0)
    pa_sig = {
        "symbol": "NIFTY 50",
        "action": "BUY_PE",
        "direction": "bearish",
        "strike": 24200,
        "option_type": "PE",
        "edge_score": 4.0,
        "underlying_price": 24180.0,
    }
    
    p._pa_momentum_scanner = MagicMock()
    p._pa_momentum_scanner.evaluate_bar.return_value = pa_sig
    
    signal = p._get_intraday_signal_for_execution("NIFTY 50", "15minute", False)
    # Should be blocked because active position is only at 87.0 and score is 4.0 (< 5.0)
    assert signal is None


def test_strong_signal_override_permits_repeat_entry():
    """Repeat entry on same strike IS allowed when incoming signal has high conviction score >= 5.0."""
    p = Prometheus()
    p.mode = "paper"
    p.data.get_vix = MagicMock(return_value=14.0)
    p.data.fetch_intraday = MagicMock(return_value=_create_dummy_df())
    
    p._credit_spread_strategy = MagicMock()
    p._credit_spread_strategy.evaluate_spread.return_value = None
    
    p._today_traded_instruments = {"NIFTY01SEP2624200PE"}
    
    # Position monitor has active position entered at 86.0, current LTP is 87.0
    p.position_monitor = MagicMock()
    p.position_monitor.get_positions.return_value = {
        "pos_1": DummyPosState("NIFTY01SEP2624200PE", entry_premium=86.0)
    }
    
    p.data.angelone_options = MagicMock()
    p.data.angelone_options.get_real_premium.return_value = {
        "ltp": 87.0,
        "tradingsymbol": "NIFTY01SEP2624200PE",
    }
    
    # Very Strong signal with edge_score 5.2 (>= 5.0)
    pa_sig = {
        "symbol": "NIFTY 50",
        "action": "BUY_PE",
        "direction": "bearish",
        "strike": 24200,
        "option_type": "PE",
        "edge_score": 5.2,
        "underlying_price": 24180.0,
    }
    
    p._pa_momentum_scanner = MagicMock()
    p._pa_momentum_scanner.evaluate_bar.return_value = pa_sig
    
    signal = p._get_intraday_signal_for_execution("NIFTY 50", "15minute", False)
    # Allowed due to strong signal override
    assert signal is not None
    assert signal.get("action") == "BUY_PE"
    assert signal.get("signal_score") == 5.2


def test_same_strike_pyramiding_allowed_when_profit_locked():
    """Repeat entry on same strike IS allowed when active position is locked in >= +10% profit."""
    p = Prometheus()
    p.mode = "paper"
    p.data.get_vix = MagicMock(return_value=14.0)
    p.data.fetch_intraday = MagicMock(return_value=_create_dummy_df())
    
    p._credit_spread_strategy = MagicMock()
    p._credit_spread_strategy.evaluate_spread.return_value = None
    
    p._today_traded_instruments = {"NIFTY01SEP2624200PE"}
    
    # Position monitor has active position entered at 86.0, current LTP is 98.0 (+13.9% profit, >= 10%)
    p.position_monitor = MagicMock()
    p.position_monitor.get_positions.return_value = {
        "pos_1": DummyPosState("NIFTY01SEP2624200PE", entry_premium=86.0)
    }
    
    p.data.angelone_options = MagicMock()
    p.data.angelone_options.get_real_premium.return_value = {
        "ltp": 98.0,
        "tradingsymbol": "NIFTY01SEP2624200PE",
    }
    
    pa_sig = {
        "symbol": "NIFTY 50",
        "action": "BUY_PE",
        "direction": "bearish",
        "strike": 24200,
        "option_type": "PE",
        "edge_score": 4.0,
        "underlying_price": 24180.0,
    }
    
    p._pa_momentum_scanner = MagicMock()
    p._pa_momentum_scanner.evaluate_bar.return_value = pa_sig
    
    signal = p._get_intraday_signal_for_execution("NIFTY 50", "15minute", False)
    assert signal is not None
    assert signal.get("action") == "BUY_PE"
    assert signal.get("entry_price") == 98.0
