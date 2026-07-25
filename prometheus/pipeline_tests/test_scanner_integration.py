"""
Integration test for the full scan pipeline.

Tests the complete flow: data fetch → evaluate → convert → gate → notify.
Uses mock Prometheus instance to simulate the full pipeline without real APIs.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from prometheus.pipeline.types import (
    ScanCycle, DataStatus, GateVerdict,
)
from prometheus.pipeline.data_bridge import DataBridge
from prometheus.pipeline.signal_evaluator import SignalEvaluator
from prometheus.pipeline.signal_converter import SignalConverter
from prometheus.pipeline.execution_gate import ExecutionGate
from prometheus.pipeline.notifier import Notifier
from prometheus.pipeline.scanner import LiveScanner
from prometheus.pipeline_tests.conftest import make_15min_bars, make_daily_bars


def _make_mock_prometheus(symbols=None, signal_map=None):
    """Create a fully mocked Prometheus for integration testing.
    
    Args:
        symbols: List of symbols to scan
        signal_map: Dict[symbol, raw_signal_dict] - controls what signal
                     each symbol generates. None = no signal.
    """
    symbols = symbols or ["NIFTY 50"]
    signal_map = signal_map or {}
    
    mock = MagicMock()
    mock.symbols = symbols
    mock.initial_capital = 100000
    
    # Data engine
    primary = make_15min_bars(200, start_date=(datetime.now() - timedelta(hours=50)).strftime("%Y-%m-%d 09:15:00"))
    daily = make_daily_bars(100)
    
    def fetch_historical(symbol, days=60, interval="day", force_refresh=False):
        if interval in ("15minute", "60minute"):
            return primary.copy()
        return daily.copy()
    
    mock.data = MagicMock()
    mock.data.fetch_historical = MagicMock(side_effect=fetch_historical)
    mock.data.get_vix = MagicMock(return_value=15.0)
    
    # Regime detector
    mock.regime_detector = MagicMock()
    mock.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime=MagicMock(value="markup"),
        confidence=0.8,
    ))
    mock.regime_detector.reset_cache = MagicMock()
    
    # Bias computation
    mock._compute_intraday_bias = MagicMock(return_value={})
    mock._compute_daily_bias = MagicMock(return_value={})
    
    # Signal generator factory
    def make_gen(**kwargs):
        sym = kwargs.get("symbol", "")
        raw = signal_map.get(sym)
        
        def gen(data_so_far):
            return raw.copy() if raw else None
        
        return gen
    
    mock._make_signal_generator = MagicMock(side_effect=make_gen)
    
    # Order manager
    mock.order_manager = MagicMock()
    mock.order_manager.managed_positions = {}
    position_mock = MagicMock()
    position_mock.position_id = "PAPER-001"
    mock.order_manager.execute_signal = MagicMock(return_value=position_mock)
    mock.order_manager.create_trailing_state = MagicMock(return_value=None)
    
    # Telegram
    mock.telegram = MagicMock()
    mock.telegram.send_message = MagicMock(return_value=True)

    # Bug #2 fix (2026-07-22): LivePaperCapture bypass guard reads
    # `_paper_capture` and `.enabled` from the prometheus instance. The
    # scanner integration tests exercise the legacy OrderManager path
    # (paper_capture is disabled), so explicitly mark the mock as
    # bypass-disabled. Without this, MagicMock auto-creates a child
    # mock for the missing attribute and the bypass guard kicks in
    # incorrectly.
    mock._paper_capture = None

    return mock


class TestScanCycleIntegration:
    """Full pipeline integration tests."""
    
    def test_no_signal_pipeline(self):
        """Complete cycle with no signals works end-to-end."""
        mock_prom = _make_mock_prometheus(
            symbols=["NIFTY 50", "BANKNIFTY"],
            signal_map={},  # No signals for any symbol
        )
        
        scanner = LiveScanner(
            mock_prom, symbols=["NIFTY 50", "BANKNIFTY"],
            scan_interval_seconds=900,
        )
        
        cycle = scanner.run_scan_cycle()
        
        assert cycle.total_symbols == 2
        assert cycle.signals_found == 0
        assert cycle.signals_executed == 0
        # Telegram should have been called (scan summary)
        mock_prom.telegram.send_message.assert_called()
    
    def test_bullish_signal_executes(self):
        """Bullish signal flows through entire pipeline and executes."""
        signal = {
            "direction": "bullish",
            "strategy": "trend",
            "entry_price": 150.0,
            "stop_loss": 120.0,
            "target": 210.0,
            "bull_score": 4.5,
            "strike": 24000,
            "option_expiry_date": "2026-01-08",
            "option_type": "CE",
            "lot_size": 75,
            "quantity": 75,
            "regime": "markup",
            "reasons": ["LiqSweep", "FVG", "VP"],
            "symbol": "NIFTY 50",
        }
        
        mock_prom = _make_mock_prometheus(
            symbols=["NIFTY 50"],
            signal_map={"NIFTY 50": signal},
        )
        
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50"])
        cycle = scanner.run_scan_cycle()
        
        assert cycle.signals_found == 1
        assert cycle.signals_executed == 1
        # Order manager should have been called
        mock_prom.order_manager.execute_signal.assert_called_once()
    
    def test_multi_symbol_mixed_signals(self):
        """Multiple symbols: some signal, some don't."""
        nifty_signal = {
            "direction": "bullish", "strategy": "trend",
            "entry_price": 150.0, "stop_loss": 120.0, "target": 210.0,
            "bull_score": 4.0, "symbol": "NIFTY 50",
            "strike": 24000, "lot_size": 75, "quantity": 75,
        }
        
        mock_prom = _make_mock_prometheus(
            symbols=["NIFTY 50", "BANKNIFTY", "FINNIFTY"],
            signal_map={"NIFTY 50": nifty_signal},  # Only NIFTY signals
        )
        
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50", "BANKNIFTY", "FINNIFTY"])
        cycle = scanner.run_scan_cycle()
        
        assert cycle.total_symbols == 3
        assert cycle.signals_found == 1
    
    def test_duplicate_symbol_rejected_on_second_cycle(self):
        """Same symbol on second scan cycle is rejected by gate."""
        signal = {
            "direction": "bullish", "strategy": "trend",
            "entry_price": 150.0, "stop_loss": 120.0, "target": 210.0,
            "bull_score": 4.0, "symbol": "NIFTY 50",
            "strike": 24000, "lot_size": 75, "quantity": 75,
        }
        
        mock_prom = _make_mock_prometheus(
            symbols=["NIFTY 50"],
            signal_map={"NIFTY 50": signal},
        )
        
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50"])
        
        cycle1 = scanner.run_scan_cycle()
        assert cycle1.signals_executed == 1
        
        cycle2 = scanner.run_scan_cycle()
        # Signal should be found but rejected by gate (symbol already traded today)
        assert cycle2.signals_found == 1
        assert cycle2.signals_executed == 0
    
    def test_execution_failure_handled(self):
        """When order_manager rejects execution, pipeline handles gracefully."""
        signal = {
            "direction": "bearish", "strategy": "trend",
            "entry_price": 180.0, "stop_loss": 210.0, "target": 120.0,
            "bear_score": 3.8, "symbol": "NIFTY 50",
            "strike": 24000, "lot_size": 75, "quantity": 75,
        }
        
        mock_prom = _make_mock_prometheus(
            symbols=["NIFTY 50"],
            signal_map={"NIFTY 50": signal},
        )
        # Order manager rejects
        mock_prom.order_manager.execute_signal = MagicMock(return_value=None)
        mock_prom.order_manager.last_execution_error = "Insufficient margin"
        
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50"])
        cycle = scanner.run_scan_cycle()
        
        assert cycle.signals_found == 1
        assert cycle.signals_executed == 0
        # Should have error in result
        result = cycle.results[0]
        assert result.execution_error != ""
    
    def test_data_fetch_error_handled(self):
        """Data fetch failure for one symbol doesn't crash the pipeline."""
        mock_prom = _make_mock_prometheus(symbols=["NIFTY 50", "BANKNIFTY"])
        
        # Make NIFTY fetch fail
        call_count = [0]
        original_fetch = mock_prom.data.fetch_historical.side_effect
        
        def failing_fetch(symbol, days=60, interval="day", force_refresh=False):
            if symbol == "NIFTY 50" and interval == "15minute":
                raise ConnectionError("API timeout")
            return original_fetch(symbol, days=days, interval=interval, force_refresh=force_refresh)
        
        mock_prom.data.fetch_historical = MagicMock(side_effect=failing_fetch)
        
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50", "BANKNIFTY"])
        
        # Should not raise
        cycle = scanner.run_scan_cycle()
        
        assert cycle.total_symbols == 2
        # NIFTY should have data error
        nifty_result = [r for r in cycle.results if r.symbol == "NIFTY 50"][0]
        assert nifty_result.data_status == DataStatus.FETCH_ERROR


class TestScanCycleDiagnostics:
    """Test diagnostic information in scan cycles."""
    
    def test_scan_summary_includes_all_info(self):
        """ScanCycle.summary() includes count of symbols, signals, executions."""
        mock_prom = _make_mock_prometheus(
            symbols=["NIFTY 50"],
            signal_map={},
        )
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50"])
        cycle = scanner.run_scan_cycle()
        
        summary = cycle.summary()
        assert "1 symbols" in summary
        assert "0 signals" in summary
    
    def test_telegram_always_called(self):
        """Telegram is called for every scan cycle, even with 0 signals."""
        mock_prom = _make_mock_prometheus(symbols=["NIFTY 50"])
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50"])
        cycle = scanner.run_scan_cycle()
        
        # At minimum, the scan summary should have been sent
        assert mock_prom.telegram.send_message.called


class TestEvaluatorPersistence:
    """Test that evaluators persist across scan cycles."""
    
    def test_evaluator_reuses_generator(self):
        """Signal generator is created once and reused across cycles."""
        mock_prom = _make_mock_prometheus(symbols=["NIFTY 50"])
        scanner = LiveScanner(mock_prom, symbols=["NIFTY 50"])
        
        scanner.run_scan_cycle()
        scanner.run_scan_cycle()
        scanner.run_scan_cycle()
        
        # _make_signal_generator should be called once per symbol, not per cycle
        assert mock_prom._make_signal_generator.call_count == 1
