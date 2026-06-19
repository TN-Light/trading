"""
Tests for Notifier — Telegram scan result notifications.

Covers:
- Zero signals → shows diagnostic breakdown
- One signal → shows signal details
- Rejection reasons displayed
- Signal alert formatting
- Execution result notifications
- Telegram failure handled gracefully
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, call

from prometheus.pipeline.types import (
    ScanCycle, SymbolScanResult, ExecutableSignal,
    SignalResult, GateResult, GateVerdict, DataStatus,
)
from prometheus.pipeline.notifier import Notifier


def _make_symbol_result(symbol: str, has_signal: bool = False,
                         executed: bool = False,
                         data_error: str = "",
                         gate_verdict: GateVerdict = None) -> SymbolScanResult:
    """Helper to create SymbolScanResult for testing."""
    result = SymbolScanResult(symbol=symbol)
    
    if data_error:
        result.data_status = DataStatus.EMPTY
        result.data_error = data_error
        return result
    
    if has_signal:
        result.signal = SignalResult(
            raw_signal={
                "direction": "bullish", "bull_score": 4.0,
                "bar_timestamp": "2026-01-05 10:30:00",
                "reasons": ["LiqSweep", "FVG"],
            },
            symbol=symbol,
        )
        result.executable = ExecutableSignal(
            symbol=symbol, action="BUY_CE", direction="bullish",
            option_type="CE", entry_price=150.0, stop_loss=120.0,
            target=210.0, risk_reward=2.0,
        )
        
        if gate_verdict and gate_verdict != GateVerdict.PASS:
            result.gate = GateResult(
                verdict=gate_verdict,
                reason=f"{symbol} already traded today",
            )
        elif executed:
            result.gate = GateResult(verdict=GateVerdict.PASS)
            result.executed = True
    
    return result


class TestNotifierScanResult:
    """Test scan result notifications."""
    
    def test_zero_signals_sends_message(self, mock_telegram):
        """Zero signals sends a diagnostic message."""
        notifier = Notifier(mock_telegram)
        cycle = ScanCycle(
            results=[
                _make_symbol_result("NIFTY 50"),
                _make_symbol_result("BANKNIFTY"),
            ]
        )
        
        notifier.notify_scan_result(cycle)
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "0" in msg  # "Eligible signals: 0"
        assert "scan complete" in msg.lower()
    
    def test_signal_found_shows_details(self, mock_telegram):
        """Signal found shows signal details in message."""
        notifier = Notifier(mock_telegram)
        cycle = ScanCycle(
            results=[
                _make_symbol_result("NIFTY 50", has_signal=True, executed=True),
                _make_symbol_result("BANKNIFTY"),
            ]
        )
        
        notifier.notify_scan_result(cycle)
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "1" in msg  # signals found
        assert "NIFTY" in msg
    
    def test_rejection_shows_reason(self, mock_telegram):
        """Rejected signal shows rejection reason."""
        notifier = Notifier(mock_telegram)
        cycle = ScanCycle(
            results=[
                _make_symbol_result("NIFTY 50", has_signal=True,
                                     gate_verdict=GateVerdict.REJECT_DUPLICATE_SYMBOL),
            ]
        )
        
        notifier.notify_scan_result(cycle)
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "NIFTY" in msg
    
    def test_data_error_in_rejections(self, mock_telegram):
        """Data errors appear in rejection list."""
        notifier = Notifier(mock_telegram)
        cycle = ScanCycle(
            results=[
                _make_symbol_result("NIFTY 50", data_error="API timeout"),
            ]
        )
        
        notifier.notify_scan_result(cycle)
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "0" in msg


class TestNotifierSignalAlert:
    """Test individual signal alerts."""
    
    def test_bullish_signal_alert(self, mock_telegram):
        """Bullish signal shows green emoji and BUY_CE."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY 50", action="BUY_CE", direction="bullish",
            option_type="CE", entry_price=150.0, stop_loss=120.0,
            target=210.0, risk_reward=2.0, confidence=0.75,
            regime="markup", reasons=["LiqSweep", "FVG"],
        )
        
        notifier.notify_signal_alert(signal)
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "NIFTY" in msg
        assert "BUY_CE" in msg
        assert "150" in msg
    
    def test_bearish_signal_alert(self, mock_telegram):
        """Bearish signal shows red emoji and BUY_PE."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY 50", action="BUY_PE", direction="bearish",
            option_type="PE", entry_price=180.0, stop_loss=210.0,
            target=120.0, risk_reward=2.0, confidence=0.60,
        )
        
        notifier.notify_signal_alert(signal)
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "BUY_PE" in msg


class TestNotifierExecutionResult:
    """Test execution result notifications."""
    
    def test_successful_execution(self, mock_telegram):
        """Successful execution sends confirmation."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY 50", action="BUY_CE", direction="bullish",
            option_type="CE",
        )
        position = MagicMock()
        position.position_id = "POS-001"
        
        notifier.notify_execution_result(signal, position)
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "OPENED" in msg or "opened" in msg.lower()
        assert "POS-001" in msg
    
    def test_failed_execution(self, mock_telegram):
        """Failed execution sends error."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY 50", action="BUY_CE", direction="bullish",
            option_type="CE",
        )
        
        notifier.notify_execution_result(signal, None, error="Insufficient margin")
        
        mock_telegram.send_message.assert_called_once()
        msg = mock_telegram.send_message.call_args[0][0]
        assert "NOT EXECUTED" in msg
        assert "Insufficient margin" in msg


class TestNotifierTelegramFailure:
    """Test graceful handling of Telegram failures."""
    
    def test_telegram_exception_handled(self):
        """Telegram send failure doesn't crash the notifier."""
        tg = MagicMock()
        tg.send_message = MagicMock(side_effect=Exception("Network error"))
        
        notifier = Notifier(tg)
        cycle = ScanCycle(results=[_make_symbol_result("NIFTY 50")])
        
        # Should not raise
        notifier.notify_scan_result(cycle)
    
    def test_no_telegram_uses_logging(self):
        """When telegram is None, notifier still works (logging only)."""
        notifier = Notifier(None)
        cycle = ScanCycle(results=[_make_symbol_result("NIFTY 50")])
        
        # Should not raise
        notifier.notify_scan_result(cycle)


class TestKiteSearchName:
    """Test Kite-searchable contract name generation."""
    
    def test_nifty_bank_ce(self, mock_telegram):
        """NIFTY BANK maps to BANKNIFTY in Kite."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY BANK", action="BUY_CE", direction="bullish",
            option_type="CE", strike=45000.0, expiry="2026-06-26",
        )
        name = notifier._make_kite_search_name(signal)
        assert name == "BANKNIFTY JUN 45000 CE"
    
    def test_finnifty_pe(self, mock_telegram):
        """NIFTY FIN SERVICE maps to FINNIFTY."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY FIN SERVICE", action="BUY_PE", direction="bearish",
            option_type="PE", strike=26350.0, expiry="2026-06-24",
        )
        name = notifier._make_kite_search_name(signal)
        assert name == "FINNIFTY JUN 26350 PE"
    
    def test_stock_uses_symbol_directly(self, mock_telegram):
        """Stock symbols pass through as-is (e.g., HDFCBANK)."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="HDFCBANK", action="BUY_CE", direction="bullish",
            option_type="CE", strike=1900.0, expiry="2026-06-26",
        )
        name = notifier._make_kite_search_name(signal)
        assert name == "HDFCBANK JUN 1900 CE"
    
    def test_no_strike_returns_empty(self, mock_telegram):
        """No strike means no Kite name."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY 50", action="BUY_CE", direction="bullish",
            option_type="CE",
        )
        name = notifier._make_kite_search_name(signal)
        assert name == ''
    
    def test_signal_alert_includes_kite_name(self, mock_telegram):
        """Signal alert message contains the Kite-searchable name."""
        notifier = Notifier(mock_telegram)
        signal = ExecutableSignal(
            symbol="NIFTY 50", action="BUY_CE", direction="bullish",
            option_type="CE", entry_price=200.0, stop_loss=170.0,
            target=260.0, risk_reward=2.0, confidence=0.70,
            strike=24500.0, expiry="2026-06-26",
        )
        notifier.notify_signal_alert(signal)
        msg = mock_telegram.send_message.call_args[0][0]
        assert "NIFTY JUN 24500 CE" in msg
        assert "Kite" in msg
