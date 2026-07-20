"""
Data structures for the paper trading subsystem.

These dataclasses are intentionally minimal and serializable (``to_dict`` /
``from_dict``) so they can be persisted to SQLite or CSV without coupling to
any other part of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class ExitReason(str, Enum):
    """Why a paper position was closed.

    Lifecycle-only reasons — these are *exit rule triggers*, NOT risk gates.
    Separating them cleanly lets us later add risk-driven exits (e.g.
    ``DAILY_LOSS_LIMIT``) without touching the strategy evaluation path.
    """

    STOP_LOSS = "stop_loss"
    TARGET = "target"
    TIME_STOP = "time_stop"             # max bars held
    SQUARE_OFF = "square_off"           # intraday force-close at session end
    END_OF_DAY = "end_of_day"           # swing mode end-of-day
    END_OF_SESSION = "end_OF_session"   # explicit session close (rare)
    REVERSE_SIGNAL = "reverse_signal"   # opposite-direction signal arrived
    MANUAL = "manual"

    def __str__(self) -> str:
        return self.value


class Direction(str, Enum):
    """Long/short direction of the trade."""

    LONG = "LONG"
    SHORT = "SHORT"

    @classmethod
    def from_signal_direction(cls, signal_direction: str) -> "Direction":
        """Map the legacy 'bullish'/'bearish' used throughout PROMETHEUS."""
        d = (signal_direction or "").lower().strip()
        if d == "bullish":
            return cls.LONG
        if d == "bearish":
            return cls.SHORT
        raise ValueError(f"Unknown signal direction: {signal_direction!r}")


@dataclass
class PaperTrade:
    """A single completed paper trade — entry + exit + all metadata.

    This is what gets persisted to SQLite/CSV. Open positions are tracked as
    ``Position`` instances and converted to ``PaperTrade`` only on close.
    """

    trade_id: str
    symbol: str                   # display name (e.g. "NIFTY 50")
    instrument: str              # API tradingsymbol (e.g. "NIFTY2672124150CE")
    underlying: str              # clean underlying (e.g. "NIFTY")
    direction: Direction
    quantity: int

    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason

    gross_pnl: float             # (exit-entry) * qty for LONG, opposite for SHORT
    costs: float                 # simulated fees
    net_pnl: float               # gross_pnl - costs
    return_pct: float            # net_pnl / (entry_price * qty) * 100
    holding_duration_seconds: int

    # Signal metadata for offline analysis
    strategy: str = ""
    signal_score: float = 0.0
    signal_confidence: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["exit_reason"] = self.exit_reason.value
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        return d


@dataclass
class Position:
    """An open paper-trade position (in-flight; not yet a PaperTrade)."""

    trade_id: str
    symbol: str
    instrument: str
    underlying: str
    direction: Direction
    quantity: int

    entry_price: float
    entry_time: datetime

    # Initial exit parameters — every Position must carry its own exit rules.
    # PositionTracker reads/updates these per bar.
    stop_loss: float
    target: float
    max_bars: int              # bars until time-stop; defaults to engine's
    bars_held: int = 0
    max_bars_allowed: Optional[int] = None

    # Trailing/structural state (optional — the engine may not enable it)
    breakeven_set: bool = False
    trailing_floor: float = 0.0       # SL never goes below this after breakeven
    high_water_mark: float = 0.0      # for trailing stop logic if enabled

    # Signal metadata
    strategy: str = ""
    signal_score: float = 0.0
    signal_confidence: float = 0.0
    trade_mode: str = "intraday"     # "intraday" or "swing"

    def unrealized_pnl(self, current_price: float) -> float:
        """Gross mark-to-market PnL at ``current_price`` (no costs).

        NB: Both LONG (bullish CE) and SHORT (bearish PE) positions here are
        LONG the option premium — we always BUY options. PnL is symmetric:
        positive when premium rallies, negative when it falls. The SHORT
        sign-flip removed 2026-07-18 (SHORT in this subsystem = underlying
        view, not position side).
        """
        return (current_price - self.entry_price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["entry_time"] = self.entry_time.isoformat()
        return d


@dataclass
class TradeStats:
    """Aggregate running performance metrics.

    Pure-computation view of the trade log; updated incrementally as trades
    close. ``max_drawdown`` here means peak-to-trough dollar drawdown of the
    cumulative realized PnL curve, not the equity curve (we don't track
    mark-to-market swings between trades — only realized closes).
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    open_positions: int = 0

    total_pnl: float = 0.0
    total_costs: float = 0.0
    gross_pnl: float = 0.0
    total_return_pct: float = 0.0

    avg_win_pnl: float = 0.0
    avg_loss_pnl: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win_pnl: float = 0.0
    largest_loss_pnl: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0          # E[PnL per trade] = wr*avg_win - (1-wr)*avg_loss

    cumulative_pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown_pnl: float = 0.0
    max_drawdown_pct: float = 0.0

    avg_holding_duration_seconds: float = 0.0

    # Per-exit-reason breakdown — useful for spotting systemic issues
    # (e.g. "60% of trades are square-off exits, signals are too weak")
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeSnapshot:
    """A single point-in-time snapshot of position + market state.

    Produced by the engine at every bar close; consumed by PositionTracker.
    Carrying the bar's OHLC lets the tracker simulate stop-loss hit-detection
    (intrabar high/low vs SL) without ambiguity about which bar closed first.
    """

    timestamp: datetime
    symbol: str
    instrument: str             # if known; otherwise empty
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bar_interval: str = "15minute"  # informational
