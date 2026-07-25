"""
PaperTradeEngine — top-level orchestrator for the paper trading subsystem.

Wires:
    SignalSource  →  position open
    PositionTracker (per-bar exit eval)
    FillSimulator (resolve entry + gap-fill exits)
    MetricsEngine (aggregate stats)
    TradeRecorder (SQLite/CSV persistence)

Lifecycle (one "session"):

    engine = PaperTradeEngine(...)
    while engine.is_active():
        # 1. Ask signal source for new signals
        for sig in engine.pending_signals():
            engine.process_new_signal(sig)

        # 2. Drive per-bar exit evaluation (caller passes the latest bar)
        for instrument, snapshot in engine.latest_bars().items():
            closed = engine.process_bar(snapshot, is_session_end=...)

        # 3. Check current metrics
        stats = engine.stats()

Engineering choices:

* The engine has NO notion of clock — the caller decides when to call
  ``process_bar`` and whether ``is_session_end`` is true. This lets us run
  the same code live (15:30 clock) or in replay mode (advancing bar by
  historical bar in seconds each).
* One-position-per-instrument is enforced: a new signal for an instrument
  with an open position is logged-but-ignored (with a count). The system
  has no notion of pyramiding. This is *not* a risk overlay; it's just
  sensible default — add an ``allow_pyramiding=True`` flag when needed.
* The 5-stage trailing stop logic lives in PositionTracker and is enabled
  by default via ``enable_trailing=True`` (matching the live system's
  behavior).
"""

from __future__ import annotations

import uuid
from datetime import datetime, time, date
from typing import Optional, List, Dict, Iterable, Any

from prometheus.papertrade.types import (
    PaperTrade, Position, Direction, ExitReason, TradeStats, TradeSnapshot,
)
from prometheus.papertrade.fill_simulator import FillSimulator, PriceFeed
from prometheus.papertrade.position_tracker import (
    PositionTracker, CostModel,
)
from prometheus.papertrade.metrics import MetricsEngine
from prometheus.papertrade.recorder import TradeRecorder
from prometheus.papertrade.signal_source import (
    SignalNotification, SignalSource, from_signal_dict,
)
from prometheus.utils.indian_market import get_lot_size, IST
from prometheus.utils.logger import logger
from prometheus.utils.symbol_format import resolve_underlying


class PaperTradeEngine:
    """Top-level paper-trading orchestrator.

    Args:
        feed: live/replay price feed (``PriceFeed`` protocol).
        signal_source: source of incoming signals (``SignalSource`` protocol).
        recorder: optional ``TradeRecorder``; pass ``None`` to skip
            persistence (only useful for tests).
        lot_size_override: when >0, use this lot size for all positions;
            else call ``get_lot_size(symbol)`` to compute per-symbol.
        enable_trailing: enables the 5-stage trailing-stop logic inside
            PositionTracker.
        default_max_bars_intraday: default ``max_bars_allowed`` for
            intraday-mode positions when signal doesn't supply one.
        default_max_bars_swing: same for swing-mode positions.
        max_concurrent_positions: cap on concurrently-open positions.
            Defaults high (200) so it never affects strategy evaluation
            by accident — paper mode shows *all* signals. Set to 1 if
            you'd rather enforce single-position-at-a-time discipline.
        allow_duplicate_instrument: when False (default), an existing open
            position in the same underlying/instrument suppresses new
            signals for that instrument. Set True if you don't care.
    """

    def __init__(
        self,
        feed: PriceFeed,
        signal_source: Optional[SignalSource] = None,
        recorder: Optional[TradeRecorder] = None,
        lot_size_override: int = 0,
        enable_trailing: bool = True,
        default_max_bars_intraday: int = 16,
        default_max_bars_swing: int = 96,
        max_concurrent_positions: int = 200,
        allow_duplicate_instrument: bool = False,
    ):
        self.feed = feed
        self.signal_source = signal_source
        self.recorder = recorder or TradeRecorder()
        self.lot_size_override = int(lot_size_override)
        self.max_concurrent_positions = int(max_concurrent_positions)
        self.allow_duplicate_instrument = bool(allow_duplicate_instrument)
        self.default_max_bars_intraday = int(default_max_bars_intraday)
        self.default_max_bars_swing = int(default_max_bars_swing)

        self.tracker = PositionTracker(
            fill_sim=FillSimulator(feed=feed),
            cost_model=CostModel(),
            enable_trailing=enable_trailing,
        )
        self.metrics = MetricsEngine()

        # Audit counters
        self.signals_seen = 0
        self.signals_skipped_duplicate = 0
        self.signals_skipped_full = 0
        self.signals_skipped_no_quote = 0
        self.signals_skipped_other = 0

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    def process_new_signal(self, signal: SignalNotification) -> Optional[str]:
        """Attempt to open a new position from a signal. Returns the trade_id
        on success, or ``None`` on skip/reject.
        """
        self.signals_seen += 1

        # 1. Capacity guard
        if len(self.tracker.open_positions) >= self.max_concurrent_positions:
            self.signals_skipped_full += 1
            logger.info(
                f"[PAPER] skip {signal.instrument}: max_concurrent_positions="
                f"{self.max_concurrent_positions} reached — paper-layer cap only"
            )
            return None

        # 2. Optional duplicate guard (per-instrument)
        if not self.allow_duplicate_instrument:
            for p in self.tracker.open_positions.values():
                if p.instrument == signal.instrument or p.underlying == signal.underlying:
                    self.signals_skipped_duplicate += 1
                    logger.info(
                        f"[PAPER] skip {signal.instrument}: existing position "
                        f"{p.trade_id} on same instrument/underlying"
                    )
                    return None

        # 3. Resolve entry fill
        # PROMETHEUS only ever BUYS options (CE for bullish, PE for bearish).
        # There is no option-selling path (documented in CLAUDE.md:
        # "NO naked option selling"). So both LONG-direction (bullish CE) and
        # SHORT-direction (bearish PE) translate to side="BUY" — they're
        # long the option premium in both cases. (Fix 2026-07-18: previously
        # SHORT was miswired to SELL, turning every bearish BUY_PE signal
        # into a naked PUT WRITE.)
        side = "BUY"
        fill = self.tracker.fill_sim.fill(
            instrument=signal.instrument,
            direction=signal.direction,
            price_hint=signal.entry_price_hint,
            side=side,
            theoretical_price=signal.entry_price_hint,
        )
        if fill.source == "rejected":
            self.signals_skipped_no_quote += 1
            logger.warning(
                f"[PAPER] reject signal {signal.instrument}: no fill price "
                f"available — fill simulator rejected"
            )
            return None

        # 4. Determine lot size / quantity
        if self.lot_size_override > 0:
            qty = self.lot_size_override
        else:
            try:
                qty = get_lot_size(signal.symbol)
            except Exception:
                qty = 1
            if qty <= 0:
                qty = 1

        # 5. Build Position and register
        from datetime import datetime as _dt, timezone as _tz
        trade_id = f"PAPER-{_dt.now(_tz.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

        # Resolve max bars per trade mode
        max_bars = signal.max_bars
        if max_bars is None:
            max_bars = (
                self.default_max_bars_intraday
                if signal.trade_mode == "intraday"
                else self.default_max_bars_swing
            )

        # Defensive: skip signals missing SL/target — they'd be useless paper trades
        if signal.stop_loss <= 0 or signal.target <= 0:
            logger.warning(
                f"[PAPER] skip {signal.instrument}: missing SL/target "
                f"(sl={signal.stop_loss} target={signal.target})"
            )
            self.signals_skipped_other += 1
            return None

        # Bug #1 (2026-07-22): the live path builds bar_timestamp as a tz-NAIVE
        # string (main.py:1834 str(.iloc[-1])) and _parse_bar_timestamp returns
        # a tz-naive datetime. Subtracting it from tz-aware datetime.now(IST)
        # raised TypeError and silently killed every LivePaperCapture signal.
        # Fix: normalize naive bar_timestamps to IST-aware before any arithmetic.
        _bar_ts = signal.bar_timestamp
        if _bar_ts is not None and _bar_ts.tzinfo is None:
            _bar_ts = IST.localize(_bar_ts)

        position = Position(
            trade_id=trade_id,
            symbol=signal.symbol,
            instrument=signal.instrument,
            underlying=signal.underlying or resolve_underlying(signal.symbol),
            direction=signal.direction,
            quantity=qty,
            entry_price=fill.fill_price,
            # Live entries use wall-clock time; replay/test paths use
            # the signal's bar_timestamp. We distinguish by bar age:
            # if the bar's timestamp is older than 1 day, the signal is
            # being driven off stale data (e.g. HistoricalDataBridge fed
            # yesterday's 15:15 bar today), and using it as the entry
            # time would produce negative holding_duration (the bug
            # observed 2026-07-21: -1095 min). In that case fall back to
            # datetime.now(). Test/replay bars (same-day or future-day
            # timestamps) are honored as before.
            #
            # Bug #1 (2026-07-22): the live path builds bar_timestamp
            # as a tz-NAIVE string (main.py:1834 str(.iloc[-1])), which
            # _parse_bar_timestamp parses back into a tz-naive datetime.
            # Subtracting it from tz-aware datetime.now(IST) raised
            # TypeError and silently killed every LivePaperCapture
            # signal. Fix: localize the naive bar_timestamp to IST
            # before the arithmetic (treats naive as local IST, which
            # is correct for this system — engine always stores IST).
            entry_time=(
                _bar_ts
                if _bar_ts is not None
                and (datetime.now(IST) - _bar_ts).total_seconds() < 86_400
                else datetime.now(IST)
            ),
            stop_loss=signal.stop_loss,
            target=signal.target,
            max_bars=max_bars,
            bars_held=0,
            strategy=signal.strategy,
            signal_score=signal.signal_score,
            signal_confidence=signal.signal_confidence,
            trade_mode=signal.trade_mode,
        )
        self.tracker.open_position(position)
        return trade_id

    # ------------------------------------------------------------------
    def process_bar(
        self,
        snapshot: TradeSnapshot,
        is_session_end: bool = False,
        is_square_off: bool = False,
    ) -> List[PaperTrade]:
        """Evaluate open positions against a new bar; close any that exit."""
        closed = self.tracker.on_bar(
            snapshot,
            is_session_end=is_session_end,
            is_square_off=is_square_off,
        )
        for trade in closed:
            self.metrics.record_close(trade)
            self.recorder.record_trade(trade)
        return closed

    # ------------------------------------------------------------------
    def gather_new_signals(self) -> List[SignalNotification]:
        """Ask the signal source for any new signals; convert to list."""
        if self.signal_source is None:
            return []
        return list(self.signal_source.next_batch())

    def stats(self) -> TradeStats:
        """Recompute and return aggregate metrics."""
        return self.metrics.snapshot(
            open_positions=len(self.tracker.open_positions),
        )

    def open_positions_count(self) -> int:
        return len(self.tracker.open_positions)

    def close(self) -> None:
        """Flush recorder, drop signal source hooks."""
        if self.signal_source is not None:
            try:
                self.signal_source.close()
            except Exception as e:
                logger.debug(f"signal_source.close failed: {e}")
        try:
            self.recorder.close()
        except Exception as e:
            logger.debug(f"recorder.close failed: {e}")

    # ------------------------------------------------------------------
    # Convenience — record a stats snapshot, useful for end-of-day CLI dump
    def log_stats_snapshot(self) -> None:
        stats = self.stats()
        self.recorder.record_stats_snapshot(stats)
        return stats
