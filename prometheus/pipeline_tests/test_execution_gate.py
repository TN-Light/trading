"""
Tests for ExecutionGate — pre-execution filtering.

Covers:
- First signal for symbol passes
- Duplicate symbol same day rejected
- Same bar timestamp rejected
- Different bar timestamp passes
- Max positions reached → rejected
- Daily loss limit exceeded → rejected
- Daily reset clears all state
- undo_pass allows retry
"""

import pytest
from datetime import date
from prometheus.pipeline.types import ExecutableSignal, GateVerdict
from prometheus.pipeline.execution_gate import ExecutionGate


def _make_executable(symbol: str = "NIFTY 50", action: str = "BUY_CE",
                      bar_ts: str = "2026-01-05 10:30:00") -> ExecutableSignal:
    """Helper to create an ExecutableSignal for testing."""
    return ExecutableSignal(
        symbol=symbol,
        action=action,
        direction="bullish" if "CE" in action else "bearish",
        option_type="CE" if "CE" in action else "PE",
        entry_price=150.0,
        stop_loss=120.0,
        target=210.0,
        risk_reward=2.0,
        bar_timestamp=bar_ts,
    )


class TestExecutionGatePass:
    """Test cases where signals should pass the gate."""
    
    def test_first_signal_passes(self):
        """First signal for a symbol passes."""
        gate = ExecutionGate(max_positions=3)
        signal = _make_executable("NIFTY 50")
        result = gate.check(signal)
        
        assert result.passed
        assert result.verdict == GateVerdict.PASS
    
    def test_different_symbols_pass(self):
        """Different symbols on the same day both pass."""
        gate = ExecutionGate(max_positions=3)
        
        result1 = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:30:00"))
        result2 = gate.check(_make_executable("BANKNIFTY", bar_ts="2026-01-05 10:30:00"))
        
        assert result1.passed
        assert result2.passed
    
    def test_different_bar_ts_for_same_symbol_after_reset(self):
        """Same symbol with different bar_ts passes after daily reset."""
        gate = ExecutionGate(max_positions=3)
        
        result1 = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:30:00"))
        assert result1.passed
        
        # Reset daily state (simulating new day)
        gate.reset_daily()
        
        result2 = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-06 10:30:00"))
        assert result2.passed


class TestExecutionGateReject:
    """Test cases where signals should be rejected."""
    
    def test_duplicate_symbol_rejected(self):
        """Same symbol on same day rejected."""
        gate = ExecutionGate(max_positions=3)
        
        result1 = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:30:00"))
        assert result1.passed
        
        result2 = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:45:00"))
        assert not result2.passed
        assert result2.verdict == GateVerdict.REJECT_DUPLICATE_SYMBOL
    
    def test_same_bar_ts_rejected(self):
        """Same bar timestamp for same symbol rejected."""
        gate = ExecutionGate(max_positions=3)
        # Initialize gate state by doing a check that triggers auto-reset
        gate._current_date = date.today()
        # Manually set bar_ts to simulate a previous pass (without adding to today_traded)
        gate._last_bar_ts["NIFTY 50"] = "2026-01-05 10:30:00"
        
        result = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:30:00"))
        assert not result.passed
        assert result.verdict == GateVerdict.REJECT_DUPLICATE_BAR
    
    def test_max_positions_rejected(self):
        """Signal rejected when max positions reached."""
        gate = ExecutionGate(max_positions=2)
        gate.update_positions(2)
        
        result = gate.check(_make_executable("NIFTY 50"))
        
        assert not result.passed
        assert result.verdict == GateVerdict.REJECT_MAX_POSITIONS
    
    def test_daily_loss_limit_rejected(self):
        """Signal rejected when daily loss limit exceeded."""
        gate = ExecutionGate(daily_loss_limit=450.0)
        gate._current_date = date.today()  # Prevent auto-reset from clearing PnL
        gate.update_daily_pnl(-500.0)  # Lost more than 450
        
        result = gate.check(_make_executable("NIFTY 50"))
        
        assert not result.passed
        assert result.verdict == GateVerdict.REJECT_DAILY_LOSS


class TestExecutionGateReset:
    """Test daily reset behavior."""
    
    def test_reset_clears_traded_symbols(self):
        """Daily reset allows previously traded symbols to trade again."""
        gate = ExecutionGate(max_positions=3)
        
        gate.check(_make_executable("NIFTY 50"))
        gate.reset_daily()
        
        result = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-06 10:30:00"))
        assert result.passed
    
    def test_reset_clears_bar_timestamps(self):
        """Daily reset clears bar timestamp tracking."""
        gate = ExecutionGate(max_positions=3)
        
        gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:30:00"))
        gate.reset_daily()
        
        # Same bar_ts should now pass (new day)
        result = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:30:00"))
        assert result.passed
    
    def test_reset_clears_daily_pnl(self):
        """Daily reset clears daily PnL."""
        gate = ExecutionGate(daily_loss_limit=450.0)
        gate._current_date = date.today()  # Prevent auto-reset
        gate.update_daily_pnl(-500.0)
        
        result1 = gate.check(_make_executable("NIFTY 50"))
        assert not result1.passed
        
        gate.reset_daily()
        
        result2 = gate.check(_make_executable("NIFTY 50"))
        assert result2.passed


class TestExecutionGateUndoPass:
    """Test undo_pass for failed executions."""
    
    def test_undo_pass_allows_retry(self):
        """undo_pass removes symbol from traded set, allowing retry."""
        gate = ExecutionGate(max_positions=3)
        
        result1 = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:30:00"))
        assert result1.passed
        
        gate.undo_pass("NIFTY 50")
        
        result2 = gate.check(_make_executable("NIFTY 50", bar_ts="2026-01-05 10:45:00"))
        assert result2.passed
