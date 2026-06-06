"""
Tests for SignalEvaluator — persistent signal generator wrapper.

Covers:
- Returns no signal for insufficient data
- Returns signal for valid data with enough confluence
- Persists generator across calls (not recreated)
- Refresh forces re-initialization
- Handles generator errors gracefully
- Diagnostics attached to result
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from prometheus.pipeline.types import ScanData, DataStatus, SignalResult
from prometheus.pipeline.signal_evaluator import SignalEvaluator
from prometheus.pipeline_tests.conftest import make_15min_bars, make_daily_bars


def _make_mock_prometheus(signal_to_return=None, raises=None):
    """Create a mock Prometheus instance that returns controlled signals."""
    mock = MagicMock()
    mock.initial_capital = 100000
    
    # Mock regime detector
    mock.regime_detector = MagicMock()
    mock.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime=MagicMock(value="markup"),
        confidence=0.8,
    ))
    mock.regime_detector.reset_cache = MagicMock()
    
    # Mock bias computation
    mock._compute_intraday_bias = MagicMock(return_value={"2026-01-05": "bullish"})
    mock._compute_daily_bias = MagicMock(return_value={"2026-01-05": "bullish"})
    
    # Mock data
    mock.data = MagicMock()
    mock.data.get_vix = MagicMock(return_value=15.0)
    
    # The critical mock: _make_signal_generator returns a callable
    def fake_gen(data_so_far):
        if raises:
            raise raises
        return signal_to_return
    
    mock._make_signal_generator = MagicMock(return_value=fake_gen)
    
    return mock


def _make_scan_data(symbol: str = "NIFTY 50", n_bars: int = 200) -> ScanData:
    """Create ScanData with enough bars for evaluation."""
    primary = make_15min_bars(n_bars)
    daily = make_daily_bars(100)
    return ScanData(
        symbol=symbol,
        primary=primary,
        hourly=primary,
        daily=daily,
        status=DataStatus.OK,
    )


class TestSignalEvaluatorInit:
    """Test evaluator initialization."""
    
    def test_initialize_creates_generator(self):
        """First evaluate call initializes the generator."""
        raw_signal = {"direction": "bullish", "bull_score": 4.0, "strategy": "trend"}
        mock_prom = _make_mock_prometheus(signal_to_return=raw_signal)
        
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        scan_data = _make_scan_data()
        
        result = evaluator.evaluate(scan_data)
        
        # _make_signal_generator should have been called once
        mock_prom._make_signal_generator.assert_called_once()
    
    def test_generator_persists_across_calls(self):
        """Generator is NOT recreated on subsequent evaluations."""
        raw_signal = {"direction": "bullish", "bull_score": 4.0, "strategy": "trend"}
        mock_prom = _make_mock_prometheus(signal_to_return=raw_signal)
        
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        scan_data = _make_scan_data()
        
        evaluator.evaluate(scan_data)
        evaluator.evaluate(scan_data)
        evaluator.evaluate(scan_data)
        
        # Should only be created ONCE, not 3 times
        assert mock_prom._make_signal_generator.call_count == 1
    
    def test_refresh_forces_reinitialization(self):
        """refresh() causes generator to be recreated on next evaluate."""
        raw_signal = {"direction": "bullish", "bull_score": 4.0, "strategy": "trend"}
        mock_prom = _make_mock_prometheus(signal_to_return=raw_signal)
        
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        scan_data = _make_scan_data()
        
        evaluator.evaluate(scan_data)
        assert mock_prom._make_signal_generator.call_count == 1
        
        evaluator.refresh()
        evaluator.evaluate(scan_data)
        assert mock_prom._make_signal_generator.call_count == 2


class TestSignalEvaluatorEvaluate:
    """Test signal evaluation."""
    
    def test_valid_signal_returns_signal_result(self):
        """Valid signal produces a SignalResult with has_signal=True."""
        raw_signal = {
            "direction": "bullish", "bull_score": 4.0, "strategy": "trend",
            "regime": "markup", "reasons": ["LiqSweep", "FVG"],
        }
        mock_prom = _make_mock_prometheus(signal_to_return=raw_signal)
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        result = evaluator.evaluate(_make_scan_data())
        
        assert result.has_signal
        assert result.direction == "bullish"
        assert result.confluence_score == 4.0
        assert result.bar_timestamp != ""
    
    def test_no_signal_returns_empty_result(self):
        """No signal returns SignalResult with has_signal=False."""
        mock_prom = _make_mock_prometheus(signal_to_return=None)
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        result = evaluator.evaluate(_make_scan_data())
        
        assert not result.has_signal
        assert result.diagnostics.get("reason") == "no_confluence"
    
    def test_insufficient_data_returns_no_signal(self):
        """Less than 50 bars returns no signal with diagnostic."""
        mock_prom = _make_mock_prometheus(signal_to_return=None)
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        scan_data = _make_scan_data(n_bars=30)  # Only 30 bars
        result = evaluator.evaluate(scan_data)
        
        assert not result.has_signal
        assert result.diagnostics.get("reason") == "insufficient_data"
    
    def test_generator_error_handled_gracefully(self):
        """Exception in generator is caught and logged."""
        mock_prom = _make_mock_prometheus(raises=ValueError("NaN in data"))
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        result = evaluator.evaluate(_make_scan_data())
        
        assert not result.has_signal
        assert "error" in result.diagnostics
    
    def test_bar_timestamp_attached(self):
        """Signal result includes bar timestamp from primary data."""
        raw_signal = {"direction": "bullish", "bull_score": 3.5, "strategy": "trend"}
        mock_prom = _make_mock_prometheus(signal_to_return=raw_signal)
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        scan_data = _make_scan_data()
        result = evaluator.evaluate(scan_data)
        
        assert result.has_signal
        assert result.bar_timestamp != ""
    
    def test_bearish_signal(self):
        """Bearish signal detected correctly."""
        raw_signal = {
            "direction": "bearish", "bear_score": 3.8, "bull_score": 0.5,
            "strategy": "trend", "reasons": ["LiqSweep", "OTE"],
        }
        mock_prom = _make_mock_prometheus(signal_to_return=raw_signal)
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        result = evaluator.evaluate(_make_scan_data())
        
        assert result.has_signal
        assert result.direction == "bearish"


class TestSignalEvaluatorDiagnostics:
    """Test diagnostic information in results."""
    
    def test_no_signal_has_reason(self):
        """No-signal result includes reason in diagnostics."""
        mock_prom = _make_mock_prometheus(signal_to_return=None)
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        result = evaluator.evaluate(_make_scan_data())
        
        assert "reason" in result.diagnostics
    
    def test_error_has_message(self):
        """Error result includes error message in diagnostics."""
        mock_prom = _make_mock_prometheus(raises=RuntimeError("test error"))
        evaluator = SignalEvaluator(mock_prom, "NIFTY 50", "15minute")
        
        result = evaluator.evaluate(_make_scan_data())
        
        assert "error" in result.diagnostics
        assert "test error" in str(result.diagnostics["error"])
