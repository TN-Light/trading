"""ExecutionGate: Pre-execution filtering (dedup, risk, position limits)."""

from datetime import datetime, date
from typing import Dict, Optional, Set
from prometheus.pipeline.types import ExecutableSignal, GateResult, GateVerdict
from prometheus.utils.logger import logger


class ExecutionGate:
    """All pre-execution checks. Each check logs its decision."""
    
    def __init__(self, max_positions: int = 3, daily_loss_limit: float = 450.0):
        self._max_positions = max_positions
        self._daily_loss_limit = daily_loss_limit
        self._today_traded: Set[str] = set()
        self._last_bar_ts: Dict[str, str] = {}  # symbol -> last bar timestamp
        self._current_date: Optional[date] = None
        self._open_position_count: int = 0
        self._daily_pnl: float = 0.0
    
    def reset_daily(self):
        """Reset daily state (call at market open)."""
        self._today_traded.clear()
        self._last_bar_ts.clear()
        self._daily_pnl = 0.0
        self._current_date = date.today()
        logger.info("ExecutionGate: daily state reset")
    
    def update_positions(self, count: int):
        """Update current open position count."""
        self._open_position_count = count
    
    def update_daily_pnl(self, pnl: float):
        """Update today's realized PnL."""
        self._daily_pnl = pnl
    
    def check(self, signal: ExecutableSignal) -> GateResult:
        """Run all pre-execution checks.
        
        Returns GateResult with PASS or specific rejection reason.
        """
        # Auto-reset on new day
        today = date.today()
        if self._current_date != today:
            self.reset_daily()
        
        symbol = signal.symbol
        
        # 1. Symbol daily dedup
        if symbol in self._today_traded:
            reason = f"{symbol} already traded today"
            logger.info(f"ExecutionGate: REJECT — {reason}")
            return GateResult(
                verdict=GateVerdict.REJECT_DUPLICATE_SYMBOL,
                reason=reason,
            )
        
        # 2. Bar timestamp dedup
        bar_ts = signal.bar_timestamp
        if bar_ts and self._last_bar_ts.get(symbol) == bar_ts:
            reason = f"{symbol} same bar timestamp {bar_ts}"
            logger.info(f"ExecutionGate: REJECT — {reason}")
            return GateResult(
                verdict=GateVerdict.REJECT_DUPLICATE_BAR,
                reason=reason,
            )
        
        # 3. Max positions
        if self._open_position_count >= self._max_positions:
            reason = f"max positions reached ({self._open_position_count}/{self._max_positions})"
            logger.info(f"ExecutionGate: REJECT — {reason}")
            return GateResult(
                verdict=GateVerdict.REJECT_MAX_POSITIONS,
                reason=reason,
            )
        
        # 4. Daily loss limit
        if self._daily_pnl < -self._daily_loss_limit:
            reason = f"daily loss limit exceeded (PnL: Rs {self._daily_pnl:,.0f})"
            logger.info(f"ExecutionGate: REJECT — {reason}")
            return GateResult(
                verdict=GateVerdict.REJECT_DAILY_LOSS,
                reason=reason,
            )
        
        # PASSED — record the bar timestamp
        if bar_ts:
            self._last_bar_ts[symbol] = bar_ts
        self._today_traded.add(symbol)
        
        logger.info(f"ExecutionGate: PASS — {symbol} {signal.action}")
        return GateResult(verdict=GateVerdict.PASS)
    
    def undo_pass(self, symbol: str):
        """Undo a PASS (e.g., if execution fails after gate passed)."""
        self._today_traded.discard(symbol)
