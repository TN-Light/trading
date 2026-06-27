"""
Shared data types for the signal pipeline.

These typed containers replace loose dicts, making the pipeline
observable and testable at every step.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class DataStatus(Enum):
    """Status of a data fetch attempt."""
    OK = "ok"
    EMPTY = "empty"
    STALE = "stale"
    MISSING_COLUMNS = "missing_columns"
    FETCH_ERROR = "fetch_error"
    MARKET_CLOSED = "market_closed"


class GateVerdict(Enum):
    """Result of an execution gate check."""
    PASS = "pass"
    REJECT_DUPLICATE_SYMBOL = "reject_duplicate_symbol"
    REJECT_DUPLICATE_BAR = "reject_duplicate_bar"
    REJECT_MAX_POSITIONS = "reject_max_positions"
    REJECT_DAILY_LOSS = "reject_daily_loss"
    REJECT_STALE_SIGNAL = "reject_stale_signal"
    REJECT_MARKET_CLOSED = "reject_market_closed"
    REJECT_RISK_CHECK = "reject_risk_check"
    REJECT_VIX = "reject_vix"


@dataclass
class ScanData:
    """Validated market data ready for signal evaluation."""
    symbol: str
    primary: pd.DataFrame        # 15-minute bars
    hourly: pd.DataFrame         # 60-minute bars (for bias)
    daily: pd.DataFrame          # Daily bars (for regime)
    status: DataStatus = DataStatus.OK
    fetch_time: datetime = field(default_factory=datetime.now)
    last_bar_time: Optional[datetime] = None
    bar_count: int = 0
    staleness_seconds: float = 0.0
    error_message: str = ""

    def __post_init__(self):
        if not self.primary.empty and "timestamp" in self.primary.columns:
            self.bar_count = len(self.primary)
            try:
                self.last_bar_time = pd.to_datetime(self.primary["timestamp"].iloc[-1])
                self.staleness_seconds = (self.fetch_time - self.last_bar_time).total_seconds()
            except Exception:
                pass


@dataclass
class SignalResult:
    """Output from the signal evaluator."""
    raw_signal: Optional[Dict[str, Any]]
    symbol: str = ""
    direction: str = ""           # "bullish" or "bearish"
    strategy: str = ""
    confluence_score: float = 0.0
    bar_timestamp: str = ""
    regime: str = ""
    reasons: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_signal(self) -> bool:
        return self.raw_signal is not None and self.direction in ("bullish", "bearish")

    def __post_init__(self):
        if self.raw_signal:
            self.direction = self.raw_signal.get("direction", "")
            self.strategy = self.raw_signal.get("strategy", "")
            self.confluence_score = float(self.raw_signal.get("bull_score", 0) or
                                         self.raw_signal.get("bear_score", 0) or 0)
            self.bar_timestamp = str(self.raw_signal.get("bar_timestamp", ""))
            self.regime = self.raw_signal.get("regime", "")
            self.reasons = self.raw_signal.get("reasons", [])
            self.symbol = self.raw_signal.get("symbol", self.symbol)


@dataclass
class ExecutableSignal:
    """Signal converted to executable format."""
    symbol: str
    action: str                   # "BUY_CE" or "BUY_PE"
    direction: str                # "bullish" or "bearish"
    option_type: str              # "CE" or "PE"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    risk_reward: float = 0.0
    strike: float = 0.0
    expiry: str = ""
    lot_size: int = 0
    quantity: int = 0
    instrument: str = ""          # tradingsymbol
    confidence: float = 0.0
    bar_timestamp: str = ""
    trade_mode: str = "swing"
    regime: str = ""
    strategy: str = ""
    reasons: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    """Result of execution gate checks."""
    verdict: GateVerdict
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == GateVerdict.PASS


@dataclass
class SymbolScanResult:
    """Result of scanning a single symbol."""
    symbol: str
    data_status: DataStatus = DataStatus.OK
    data_error: str = ""
    signal: Optional[SignalResult] = None
    executable: Optional[ExecutableSignal] = None
    gate: Optional[GateResult] = None
    executed: bool = False
    execution_error: str = ""

    @property
    def had_signal(self) -> bool:
        return self.signal is not None and self.signal.has_signal

    @property
    def was_rejected(self) -> bool:
        return self.gate is not None and not self.gate.passed

    @property
    def rejection_reason(self) -> str:
        # Only report truly unusable data statuses as rejections
        if self.data_status not in (DataStatus.OK, DataStatus.STALE):
            return f"data:{self.data_status.value}"
        if self.signal and not self.signal.has_signal:
            return "no_signal"
        if self.gate and not self.gate.passed:
            return self.gate.reason
        if self.execution_error:
            return f"exec:{self.execution_error}"
        return ""


@dataclass
class ScanCycle:
    """Complete result of one scan cycle across all symbols."""
    timestamp: datetime = field(default_factory=datetime.now)
    results: List[SymbolScanResult] = field(default_factory=list)

    @property
    def total_symbols(self) -> int:
        return len(self.results)

    @property
    def signals_found(self) -> int:
        return sum(1 for r in self.results if r.had_signal)

    @property
    def signals_executed(self) -> int:
        return sum(1 for r in self.results if r.executed)

    @property
    def rejections(self) -> List[tuple]:
        """List of (symbol, reason) for rejected signals."""
        return [(r.symbol, r.rejection_reason) for r in self.results
                if r.rejection_reason and r.rejection_reason != "no_signal"]

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = [
            f"Scan {self.timestamp.strftime('%H:%M')}",
            f"{self.total_symbols} symbols",
            f"{self.signals_found} signals",
            f"{self.signals_executed} executed",
        ]
        rejects = self.rejections
        if rejects:
            parts.append(f"{len(rejects)} rejected")
        return " | ".join(parts)
