"""
Tests for SignalConverter — backtest signal → executable format.

Covers:
- Bullish signal → BUY_CE with correct fields
- Bearish signal → BUY_PE with correct fields
- Missing direction → rejected
- Expiry strategy → rejected
- No signal → rejected
- Risk/reward calculation
- Confidence calculation from score
"""

import pytest
from prometheus.pipeline.types import SignalResult, ExecutableSignal
from prometheus.pipeline.signal_converter import SignalConverter


class TestSignalConverterValid:
    """Test successful signal conversions."""
    
    def test_bullish_signal_becomes_buy_ce(self, sample_raw_signal):
        """Bullish signal converts to BUY_CE."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_raw_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable is not None
        assert executable.action == "BUY_CE"
        assert executable.direction == "bullish"
        assert executable.option_type == "CE"
        assert executable.symbol == "NIFTY 50"
    
    def test_bearish_signal_becomes_buy_pe(self, sample_bearish_signal):
        """Bearish signal converts to BUY_PE."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_bearish_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable is not None
        assert executable.action == "BUY_PE"
        assert executable.direction == "bearish"
        assert executable.option_type == "PE"
    
    def test_entry_sl_target_preserved(self, sample_raw_signal):
        """Entry, SL, target values are correctly transferred."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_raw_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable.entry_price == 150.0
        assert executable.stop_loss == 120.0
        assert executable.target == 210.0
    
    def test_risk_reward_calculated(self, sample_raw_signal):
        """Risk/reward ratio is correctly calculated."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_raw_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        # risk = |150 - 120| = 30, reward = |210 - 150| = 60
        assert executable.risk_reward == 2.0
    
    def test_confidence_from_score(self, sample_raw_signal):
        """Confidence is calculated from bull_score/6.0."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_raw_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        # bull_score = 4.5, confidence = min(1.0, 4.5/6.0) = 0.75
        assert executable.confidence == pytest.approx(0.75, abs=0.01)
    
    def test_lot_size_preserved(self, sample_raw_signal):
        """Lot size from raw signal is preserved."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_raw_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable.lot_size == 75
    
    def test_bar_timestamp_preserved(self, sample_raw_signal):
        """Bar timestamp from signal is preserved."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_raw_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable.bar_timestamp == "2026-01-05 10:30:00"
    
    def test_raw_dict_attached(self, sample_raw_signal):
        """Original raw signal dict is attached for execution."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_raw_signal, symbol="NIFTY 50")
        
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable.raw is not None
        assert executable.raw["direction"] == "bullish"


class TestSignalConverterRejections:
    """Test signal rejections with logged reasons."""
    
    def test_no_signal_returns_none(self):
        """None signal result returns None."""
        converter = SignalConverter()
        result = converter.convert(None, "NIFTY 50")
        assert result is None
    
    def test_empty_signal_returns_none(self):
        """Signal with no raw data returns None."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=None, symbol="NIFTY 50")
        result = converter.convert(signal_result, "NIFTY 50")
        assert result is None
    
    def test_expiry_strategy_rejected(self, sample_expiry_signal):
        """Expiry strategy signals are rejected."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_expiry_signal, symbol="NIFTY 50")
        result = converter.convert(signal_result, "NIFTY 50")
        assert result is None
    
    def test_missing_direction_rejected(self, sample_no_direction_signal):
        """Signal without bullish/bearish direction is rejected."""
        converter = SignalConverter()
        signal_result = SignalResult(raw_signal=sample_no_direction_signal, symbol="NIFTY 50")
        result = converter.convert(signal_result, "NIFTY 50")
        assert result is None
    
    def test_invalid_direction_rejected(self):
        """Signal with invalid direction ('neutral') is rejected."""
        converter = SignalConverter()
        raw = {"direction": "neutral", "strategy": "trend", "entry_price": 100}
        signal_result = SignalResult(raw_signal=raw, symbol="NIFTY 50")
        result = converter.convert(signal_result, "NIFTY 50")
        assert result is None


class TestSignalConverterEdgeCases:
    """Edge cases in signal conversion."""
    
    def test_zero_risk_gives_zero_rr(self):
        """When entry == stop_loss, RR is 0."""
        converter = SignalConverter()
        raw = {
            "direction": "bullish", "strategy": "trend",
            "entry_price": 100.0, "stop_loss": 100.0, "target": 150.0,
            "bull_score": 3.0, "symbol": "NIFTY 50",
        }
        signal_result = SignalResult(raw_signal=raw, symbol="NIFTY 50")
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable is not None
        assert executable.risk_reward == 0.0
    
    def test_zero_score_gives_zero_confidence(self):
        """Zero score produces zero confidence."""
        converter = SignalConverter()
        raw = {
            "direction": "bullish", "strategy": "trend",
            "entry_price": 100.0, "stop_loss": 80.0, "target": 150.0,
            "bull_score": 0, "bear_score": 0, "symbol": "NIFTY 50",
        }
        signal_result = SignalResult(raw_signal=raw, symbol="NIFTY 50")
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable is not None
        assert executable.confidence == 0.0
    
    def test_high_score_caps_confidence_at_1(self):
        """Score > 6 caps confidence at 1.0."""
        converter = SignalConverter()
        raw = {
            "direction": "bullish", "strategy": "trend",
            "entry_price": 100.0, "stop_loss": 80.0, "target": 150.0,
            "bull_score": 8.0, "symbol": "NIFTY 50",
        }
        signal_result = SignalResult(raw_signal=raw, symbol="NIFTY 50")
        executable = converter.convert(signal_result, "NIFTY 50")
        
        assert executable.confidence == 1.0
