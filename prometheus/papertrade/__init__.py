"""
PROMETHEUS — Paper Trading Subsystem

A standalone paper trading engine whose sole purpose is to evaluate the
strategy itself — *not* the risk management. Every valid signal is executed
automatically, regardless of capital limits, exposure limits, or daily-loss
restrictions.

Architecture:

    SignalSource  →  PaperTradeEngine  →  PositionTracker
                          ↓                       ↑
                    FillSimulator ───────────────┘
                          ↓
                       Recorder  →  SQLite / CSV
                          ↓
                       Metrics  →  CLI / JSON

The components are deliberately pluggable so the *live* trading system can
later reuse FillSimulator / PositionTracker / Metrics with a real broker
underneath and risk overlays on top.

Design rules (per project requirements):

* Every valid signal → entry. No position sizing beyond "1 lot per signal"
  (override via a single CLI flag, not scattered risk checks).
* No daily loss limits, no exposure limits, no capital checks.
* Exit logic (SL/target/time-stop/square-off/EOD) lives in PositionTracker
  as *exit rules*, not risk gates — they describe *the trade's lifecycle*.
* The fill simulator never fills at 0.0 — that was the 2026-07-17 paper bug.
* Every trade is fully recorded with entry/exit price+time, direction, qty,
  P&L, return %, holding duration, exit reason — for offline analysis.

See ``prometheus/papertrade/engine.py`` for the top-level entry point.
"""

from prometheus.papertrade.types import (
    PaperTrade,
    Position,
    ExitReason,
    TradeStats,
    TradeSnapshot,
)
from prometheus.papertrade.engine import PaperTradeEngine

__all__ = [
    "PaperTrade",
    "Position",
    "ExitReason",
    "TradeStats",
    "TradeSnapshot",
    "PaperTradeEngine",
]
