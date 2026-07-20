"""
Prometheus Paper Executor -- live paper-trade signal capture.

Purpose
-------
This package wraps the existing ``prometheus.papertrade`` subsystem into a
*signal capture* adapter for live paper trading.

Goal: in paper-trading mode, execute EVERY valid signal end-to-end (open
paper position -> mark-to-market per bar -> exit on SL/target/time-stop/
square-off -> persist to CSV+SQLite -> expose live running stats) WITHOUT
applying any of the production risk gates. The production Kite live path
(``prometheus/execution/``) keeps all its safety brakes; this adapter does
not touch that code.

Architecture
------------
    ┌───────────────────┐    refined_signal (dict)
    │  signal pipeline  │ ─────────────────────────────┐
    │  (main.py scan    │                                │
    │   + fusion +     │                                │
    │   strategy)       │                                ▼
    └───────────────────┘              ┌───────────────────────────────┐
                                       │ LivePaperCapture              │
                                       │   on_signal(refined)          │  (this pkg)
                                       │   on_bar(symbol, ohlc)        │
                                       │   stats() / dump()            │
                                       └───────────────┬───────────────┘
                                                       │ wraps (no subclass)
                                                       ▼
                                       ┌───────────────────────────────┐
                                       │ prometheus.papertrade         │
                                       │  .PaperTradeEngine            │  (existing)
                                       │  .PositionTracker            │
                                       │  .FillSimulator              │
                                       │  .TradeRecorder              │
                                       │  .MetricsEngine              │
                                       └───────────────────────────────┘

Why an adapter, not edits to papertrade/
---------------------------------------
- ``papertrade/`` is the backtest/replay module (per CLAUDE.md Session 29).
  Its engine's caller drives the bar loop. For live mode we want the same
  exit logic and persistence but driven by live market ticks.
- Touching ``papertrade/`` risks un-doing the Session 30 cleanup ("Swing-15m
  is the locked execution path... keep any intraday code isolated").
- A thin adapter (this file) is ~250 lines and reuses the existing engine,
  types, recorder, metrics, exits, cost-model — zero duplication.

Production-vs-paper separation guarantee
-----------------------------------------
``LivePaperCapture`` is constructed only when:

  settings.system.mode == "paper"     AND
  settings.paper_capture.enabled == True

Otherwise ``self._paper_capture`` on the Prometheus instance is ``None`` and
``on_signal`` / ``on_bar`` are no-ops. Live Kite execution never reaches
this code path.

Future extension hook
----------------------
When the live Kite path needs the same exit logic with real risk gates, we
will subclass ``papertrade.PositionTracker`` with a
``LiveRiskAwarePositionTracker`` that consults ``settings.risk.*`` before
allowing an exit decision — keeping the strategy-evaluation exit logic
intact and adding risk as a *decorator*. This adapter is structured so that
swap is a one-line change in ``LivePaperCapture.__init__``.
"""
from prometheus.paper_executor.live_bridge import (
    LivePaperCapture,
    LivePriceFeed,
    get_paper_capture,
    is_paper_capture_enabled,
)

__all__ = [
    "LivePaperCapture",
    "LivePriceFeed",
    "get_paper_capture",
    "is_paper_capture_enabled",
]
