"""
PositionTracker — owns all open paper-trade positions and decides when each
position should exit.

Exit rules implemented here are *lifecycle* rules (SL/target/time-stop/
square-off/end-of-day/reverse-signal) — NOT risk gates. They describe the
natural end of a trade. This is the only place where strategy-evaluation
exit logic should live; the production ``PositionMonitor`` can later borrow
this same module by injecting a different ``FillSimulator`` to use the real
broker.

Exit precedence on each bar close:

    1. Gap-then-SL or gap-then-target — bar's open already breaches; fill at
       the open.
    2. Intra-bar SL hit — fill at SL level (limit-fills on the wick).
    3. Intra-bar target hit — fill at target.
    4. Time-stop — ``bars_held >= max_bars_allowed`` → market close.
    5. Square-off (intraday only) — caller-driven (e.g. >= 15:15 IST) → close.
    6. End-of-day — swing mode, last bar of day → close.

Higher-priority rules fire first; if multiple rules fire the same bar, only
the topmost exits the position.

NOTE: We do NOT check intrabar stop *and* target in the same bar — that's a
simulation artifact we don't want. NSE exercised orders at first-touch only;
we honor the historical convention (stop-loss before target when both are
inside the same bar's range — conservative}});
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Callable

from prometheus.papertrade.types import (
    Position, Direction, ExitReason, PaperTrade, TradeSnapshot,
)
from prometheus.papertrade.fill_simulator import FillSimulator, FillResult
from prometheus.utils.logger import logger
from prometheus.utils.indian_market import IST


# Default exit timing (Indian market) — caller can override
DEFAULT_SQUARE_OFF_TIME = time(15, 15)
DEFAULT_SESSION_CLOSE_TIME = time(15, 30)


class CostModel:
    """Simple Zerodha-style options cost model.

    For paper-trade evaluation we apply a flat per-side cost on notional
    rather than dragging in the full ``ZerodhaCostModel`` — strategy
    evaluation doesn't benefit from sub-paisa precision, and the cost
    model's churn would couple us to the production engine.
    """

    def __init__(self, cost_per_side_bps: float = 1.0):
        # 1 bps = 0.01% per side; STT + brokerage + GST + stamp duty approx
        self.cost_per_side_bps = float(cost_per_side_bps)

    def cost_for_notional(self, notional: float) -> float:
        return notional * self.cost_per_side_bps / 10000.0


class PositionTracker:
    """Holds open positions; emits closed trades on exit events.

    Args:
        fill_sim: FillSimulator used to resolve exit fills when no concise
            price is implied (e.g. square-off / reverse signal / time-stop).
        cost_model: cost model applied to each leg (entry + exit).
        enable_trailing: when True, replicate the 5-stage trailing stop logic
            described in CLAUDE.md (breakeven at 0.4R, etc.). Defaults True.
            Set False for the simplest strategy-evaluation view (raw SL/target
            only).
        square_off_time: intraday mode force-close time (IST). Defaults 15:15.
        session_close_time: end of trading day (IST). Defaults 15:30.
    """

    def __init__(
        self,
        fill_sim: FillSimulator,
        cost_model: Optional[CostModel] = None,
        enable_trailing: bool = True,
        square_off_time: time = DEFAULT_SQUARE_OFF_TIME,
        session_close_time: time = DEFAULT_SESSION_CLOSE_TIME,
    ):
        self.fill_sim = fill_sim
        self.cost_model = cost_model or CostModel()
        self.enable_trailing = bool(enable_trailing)
        self.square_off_time = square_off_time
        self.session_close_time = session_close_time

        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[PaperTrade] = []

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------
    def open_position(self, position: Position) -> None:
        """Register a new position. Assumes ``position.entry_price`` is
        already filled (entry fill handled by the engine before calling
        this — engine responsibility, not the tracker's).
        """
        if position.trade_id in self.open_positions:
            logger.warning(
                f"PositionTracker: trade_id {position.trade_id} already open; "
                f"rejecting duplicate"
            )
            return
        self.open_positions[position.trade_id] = position
        logger.info(
            f"[PAPER-OPEN] {position.trade_id} {position.direction.value} "
            f"{position.quantity} {position.instrument} @ Rs {position.entry_price:.2f} "
            f"SL={position.stop_loss:.2f} TGT={position.target:.2f} "
            f"strategy={position.strategy} score={position.signal_score:.2f}"
        )

    def close_position(
        self,
        trade_id: str,
        timestamp: datetime,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> Optional[PaperTrade]:
        """Force-close by explicit price (e.g. live SL trigger). Records
        the PaperTrade, removes the open position, returns the trade.

        Returns ``None`` if the trade_id is unknown or already closed.
        """
        pos = self.open_positions.pop(trade_id, None)
        if pos is None:
            logger.warning(f"PositionTracker: cannot close unknown trade_id {trade_id}")
            return None

        # Need a nonzero exit fill — if caller passed 0, ask FillSimulator.
        # We bought the option (long-only system) → exit is a SELL regardless
        # of underlying direction. Both bullish-CE and bearish-PE positions
        # close by selling the contract; the PnL math below is therefore the
        # long-premium PnL: (exit - entry) * qty.
        if exit_price <= 0:
            side = "SELL"
            fill = self.fill_sim.fill(
                pos.instrument, pos.direction, price_hint=pos.stop_loss,
                side=side, theoretical_price=pos.entry_price,
            )
            if fill.source == "rejected":
                # Don't book a phantom loss; leave position open (caller
                # may try again next bar)
                logger.error(
                    f"PositionTracker: cannot resolve exit fill for {trade_id}; "
                    f"position left open for retry next bar."
                )
                self.open_positions[trade_id] = pos
                return None
            exit_price = fill.fill_price

        costs = self.cost_model.cost_for_notional(pos.entry_price * pos.quantity) \
              + self.cost_model.cost_for_notional(exit_price * pos.quantity)
        # Every option position in this subsystem is LONG THE PREMIUM (bought
        # CE for bullish, bought PE for bearish). PnL is symmetric in both
        # cases: profit when premium rallies, loss when premium falls. The
        # old code branched on `Direction.SHORT` to flip the math — wrong,
        # because SHORT here means "bearish underlying view" not "short the
        # option". (Fix 2026-07-18: previously bearish-PE trades had their
        # PnL sign inverted — winners booked as losses and vice versa.)
        gross = (exit_price - pos.entry_price) * pos.quantity
        net = gross - costs
        notional_in = pos.entry_price * pos.quantity
        ret_pct = (net / notional_in * 100.0) if notional_in > 0 else 0.0
        # Normalize tz-aware / tz-naive mismatch between entry and exit timestamps.
        # Live signals emit IST aware datetimes; historical/replay paths and some
        # test fixtures emit tz-naive datetimes. Subtraction only works if both
        # match. Strip tzinfo on the exit-side if entry is naive; vice versa.
        entry_ts = pos.entry_time
        exit_ts = timestamp
        if entry_ts.tzinfo is None and exit_ts.tzinfo is not None:
            exit_ts = exit_ts.replace(tzinfo=None)
        elif entry_ts.tzinfo is not None and exit_ts.tzinfo is None:
            exit_ts = exit_ts.replace(tzinfo=entry_ts.tzinfo)
        duration = int((exit_ts - entry_ts).total_seconds())

        trade = PaperTrade(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            instrument=pos.instrument,
            underlying=pos.underlying,
            direction=pos.direction,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_ts,
            exit_reason=exit_reason,
            gross_pnl=round(gross, 2),
            costs=round(costs, 2),
            net_pnl=round(net, 2),
            return_pct=round(ret_pct, 2),
            holding_duration_seconds=duration,
            strategy=pos.strategy,
            signal_score=pos.signal_score,
            signal_confidence=pos.signal_confidence,
            stop_loss=pos.stop_loss,
            target=pos.target,
        )
        self.closed_trades.append(trade)
        logger.info(
            f"[PAPER-CLOSE] {trade.trade_id} {trade.exit_reason.value} "
            f"exit={trade.exit_price:.2f} pnl=Rs {trade.net_pnl:+.2f} "
            f"({trade.return_pct:+.2f}%) held={duration}s"
        )
        return trade

    # ------------------------------------------------------------------
    # Per-bar exit evaluation
    # ------------------------------------------------------------------
    def on_bar(
        self,
        snapshot: TradeSnapshot,
        is_session_end: bool = False,
        is_square_off: bool = False,
    ) -> List[PaperTrade]:
        """Evaluate all open positions against ``snapshot`` for exit triggers.

        Args:
            snapshot: latest OHLC bar for one instrument.
            is_session_end: True if this is the last bar of the day/session.
            is_square_off: True if intraday square-off window has been
                reached for this bar. Caller decides this (engine enforces
                its own clock).
        Returns: list of PaperTrades closed this bar, in触发-order.

        Underlying vs. option bar — 2026-07-21 fix:

            The previous matcher was ``p.instrument == snapshot.instrument
            OR p.symbol == snapshot.symbol``, which meant ANY SENSEX option
            position matched ANY SENSEX underlying bar. When the caller fed
            the underlying's OHLC bar (e.g. SENSEX index level ~77,000),
            the tracker interpreted the index close as the option premium
            and fired a ``target`` exit at Rs 77,710.66, booking Rs 1.5M of
            fictional profit on a 20-lot SENSEX PUT position that had a
            real-world target of Rs 410.

            Now: positions only match a snapshot when their instrument
            strings are equal (strict match). If ``snapshot.instrument`` is
            empty (the "underlying bar, no specific option" case), the
            snapshot is used ONLY to advance ``bars_held`` — never to
            evaluate SL/target or trail the stop.
        """
        closed: List[PaperTrade] = []
        # Strict instrument-or-symbol match — but only positions on the same
        # instrument are *evaluated*. Same-symbol different-instrument bars
        # (the underlying-index bar) advance bars-held only.
        for tid, p in list(self.open_positions.items()):
            if p.instrument == snapshot.instrument and snapshot.instrument:
                # True match: same option contract. Full evaluation.
                p.bars_held += 1
                exit_price, exit_reason = self._evaluate_exit(
                    p, snapshot,
                    is_session_end=is_session_end,
                    is_square_off=is_square_off,
                )
                if exit_reason is None:
                    if self.enable_trailing:
                        self._maybe_advance_trailing_stop(p, snapshot.close)
                    continue
                trade = self.close_position(tid, snapshot.timestamp, exit_price, exit_reason)
                if trade is not None:
                    closed.append(trade)
                continue

            if p.symbol == snapshot.symbol and not snapshot.instrument:
                # Underlying bar (e.g. NIFTY 50 index bar) for an open option
                # position on the same symbol. Advance bars-held only — DO NOT
                # evaluate SL/target/trailing against the index price (that
                # was the 2026-07-21 Rs 1.5M phantom-profit bug).
                p.bars_held += 1
                # Force-evaluate session_end / square_off using the LTP feed
                # (not the snapshot's OHLC, which is the index level).
                if is_session_end or is_square_off:
                    exit_price, exit_reason = self._evaluate_exit_via_feed(
                        p, snapshot, is_session_end=is_session_end,
                        is_square_off=is_square_off,
                    )
                    if exit_reason is not None:
                        trade = self.close_position(tid, snapshot.timestamp, exit_price, exit_reason)
                        if trade is not None:
                            closed.append(trade)
        return closed

    def _evaluate_exit(
        self,
        pos: Position,
        snap: TradeSnapshot,
        is_session_end: bool,
        is_square_off: bool,
    ) -> Tuple[float, Optional[ExitReason]]:
        """Return (exit_price, exit_reason) or (0, None) if no exit fires.

        Checks intrusion into bar range for SL/target, then time-stop,
        then session-end / square-off.
        """
        # Direction-wise "good direction" and "bad direction" thresholds.
        # Both LONG (bullish CE) and SHORT (bearish PE) positions are LONG
        # the option PREMIUM — we bought the contract. So exit checks are
        # symmetric: SL is below entry (premium fell), target is above entry
        # (premium rallied). The old `else` branch applied inverted
        # conventions assuming SHORT = "short the asset", which was wrong
        # for our BUY_PE convention. (Fix 2026-07-18.)
        sl = pos.stop_loss
        tgt = pos.target

        # -- 1. Gap-open already breached the SL/target ---------------------
        # If the bar's OPEN price is beyond target or below SL, fill at OPEN.
        # (Conservative: a gap beyond target means the market gapped favorably
        # past target on open — fill at open. A gap below SL means we missed
        # the SL line — fill at the open, accepting the worse price.)
        if snap.open <= sl:
            return snap.open, ExitReason.STOP_LOSS
        if snap.open >= tgt:
            return snap.open, ExitReason.TARGET
        if snap.low <= sl:
            return sl, ExitReason.STOP_LOSS
        if snap.high >= tgt:
            return tgt, ExitReason.TARGET

        # -- 2. Time stop — order matters: SL/target already checked above -
        # don't exit on time if SL/target was hit; but we exited earlier in
        # that case so we don't reach here.
        max_bars = pos.max_bars_allowed or pos.max_bars
        if max_bars and pos.bars_held >= max_bars:
            return snap.close, ExitReason.TIME_STOP

        # -- 3. Square-off (intraday force close) ---------------------------
        if is_square_off and pos.trade_mode == "intraday":
            return snap.close, ExitReason.SQUARE_OFF

        # -- 4. End-of-day swing close --------------------------------------
        if is_session_end and pos.trade_mode == "swing":
            return snap.close, ExitReason.END_OF_DAY

        return 0.0, None

    def _evaluate_exit_via_feed(
        self,
        pos: Position,
        snap: TradeSnapshot,
        is_session_end: bool,
        is_square_off: bool,
    ) -> Tuple[float, Optional[ExitReason]]:
        """Evaluate SL/target/trailing using the live LTP feed instead of the
        snapshot's OHLC — used when ``on_bar`` received an *underlying* bar
        (e.g. NIFTY 50 index level) but the position is an *option* on that
        underlying. Without this guard, the tracker would interpret the
        index's close (~24000) as the option premium and immediately fire a
        fictional TARGET exit.

        Only called for ``is_session_end=True`` or ``is_square_off=True`` —
        i.e. we still want to fire EOD / square-off closes when we get the
        underlying bar, we just don't want to fabricate a SL/TARGET trigger
        from the index level.

        Returns ``(exit_price, exit_reason)`` or ``(0.0, None)``.
        """
        sl = pos.stop_loss
        tgt = pos.target
        # Look up the real option LTP. Failures are non-fatal — we'll fall
        # through to session_end / square_off below.
        try:
            ltp = self.fill_sim.feed.get_ltp(pos.instrument)
        except Exception:
            ltp = 0.0
        ltp = float(ltp or 0.0)

        if ltp > 0:
            # We have a real LTP for the option. Evaluate SL/target with it.
            if ltp <= sl:
                return sl, ExitReason.STOP_LOSS
            if ltp >= tgt:
                return ltp, ExitReason.TARGET
        # Otherwise no LTP — skip SL/target evaluation this bar (don't
        # fabricate an exit price from the underlying snapshot).

        # Square-off and end-of-day force-closes still fire (the LTP we
        # recovered — or fall back to ``fill_sim`` at fill time — supplies
        # the exit price).
        if is_square_off and pos.trade_mode == "intraday":
            # Fill price will be resolved by ``close_position`` via fill_sim
            # using the option LTP (caller-hint = entry_price if LTP missing).
            return max(ltp, 0.0), ExitReason.SQUARE_OFF
        if is_session_end and pos.trade_mode == "swing":
            return max(ltp, 0.0), ExitReason.END_OF_DAY
        return 0.0, None

    def _maybe_advance_trailing_stop(
        self, pos: Position, current_price: float,
    ) -> None:
        """5-stage trailing stop — see CLAUDE.md.

        Stage transitions:
            breakeven_set   at 0.4R   → SL = entry + costs (effectively entry)
            trail lock 20%  at 1.0R   → SL = entry + 0.2*risk_distance
            trail lock 50%  at 2.0R   → SL = entry + 0.5*risk_distance
            trail lock 70%  at 3.0R   → SL = entry + 0.7*risk_distance
            high-water mark beyond    → SL = max(70%-floor, hwm - buffer)

        We mirror this from ``prometheus/execution/position_monitor.py`` but
        keep it independent so the paper subsystem doesn't import execution
        internals.

        NB: Both LONG-direction (bullish CE) and SHORT-direction (bearish PE)
        positions in this subsystem are LONG the premium (we always BUY
        options). Trailing math is therefore identical for both — we trail
        UP on premium rallies in either case. The old `is_long` branching
        that flipped signs for SHORT was bogus (SHORT here = underlying view,
        not position side). Fixes 2026-07-18.

        Bug #5 (2026-07-22): this method's ``def`` line was accidentally
        dropped during an earlier edit. The body (with the docstring above)
        was left dangling as dead code immediately after the
        ``_evaluate_exit_via_feed`` method's closing ``return 0.0, None`` —
        unreachable because Python treats it as a string expression
        statement followed by ``if`` statements operating on out-of-scope
        locals. The caller in ``on_bar`` (``self._maybe_advance_trailing_stop``)
        therefore raised ``AttributeError`` on every bar whose position
        survived SL/target/time-stop evaluation. Added the ``def`` signature
        line back to restore the trailing-stop code path.
        """
        progress = (current_price - pos.entry_price) / max(
            (pos.entry_price - pos.stop_loss), 1e-9
        )
        if current_price > pos.high_water_mark:
            pos.high_water_mark = current_price

        risk_distance = abs(pos.entry_price - pos.stop_loss) or 1.0

        # Stage 1 — breakeven (at 0.4R)
        if not pos.breakeven_set and progress >= 0.4:
            new_sl = pos.entry_price
            # Only advance (never retreat)
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                pos.breakeven_set = True
                logger.debug(
                    f"[{pos.trade_id}] BREAKEVEN_SET: SL -> {new_sl:.2f} at progress={progress:.2f}R"
                )
        # Stage 2 — lock 20% at 1.0R
        elif pos.breakeven_set and progress >= 1.0 and pos.trailing_floor < 0.20:
            lock = 0.20
            new_sl = pos.entry_price + lock * risk_distance
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                pos.trailing_floor = lock
        # Stage 3 — lock 50% at 2.0R
        elif pos.breakeven_set and progress >= 2.0 and pos.trailing_floor < 0.50:
            lock = 0.50
            new_sl = pos.entry_price + lock * risk_distance
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                pos.trailing_floor = lock
        # Stage 4 — lock 70% at 3.0R
        elif pos.breakeven_set and progress >= 3.0 and pos.trailing_floor < 0.70:
            lock = 0.70
            new_sl = pos.entry_price + lock * risk_distance
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                pos.trailing_floor = lock
        # Stage 5 — high-water-trail (beyond 3.0R, never below 70% floor)
        # We don't have a single live "bar's trailing stop" here; we
        # approximate by tightening SL to max(current_sl, 70%-floor, hwm - small buffer)
        elif pos.breakeven_set and progress >= 3.5:
            hwm_floor = pos.entry_price + 0.70 * risk_distance
            trail_candidate = pos.high_water_mark - 0.05 * risk_distance  # 5% of R buffer
            new_sl = max(pos.stop_loss, hwm_floor, trail_candidate)
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
