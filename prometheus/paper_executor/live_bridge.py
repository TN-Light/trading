"""
LivePaperCapture -- thin adapter that drives ``papertrade.PaperTradeEngine``
from a live signal+bar stream, with no risk gates.

This is the only file the live ``main.py`` interacts with. Everything below
``LivePaperCapture`` is the existing ``prometheus.papertrade`` subsystem.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict

from prometheus.papertrade.engine import PaperTradeEngine
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.metrics import MetricsEngine
from prometheus.papertrade.recorder import TradeRecorder
from prometheus.papertrade.signal_source import from_signal_dict
from prometheus.papertrade.types import (
    PaperTrade, TradeStats, TradeSnapshot,
    Position, Direction,
)
from prometheus.utils.logger import logger
from prometheus.utils.indian_market import IST


# ---------------------------------------------------------------------
# Live price-feed shim -- bridges prometheus.data engine to PriceFeed
# ---------------------------------------------------------------------

class LivePriceFeed:
    """Implements the ``papertrade.fill_simulator.PriceFeed`` protocol by
    delegating to whatever LTP source the live Prometheus uses (Kite broker
    AngelOne fetcher, or the simulated PaperTrader). Both expose
    ``get_ltp(instrument)`` -- we forward to that.

    Falls back to direct Angel One live option quotes when the primary feed
    does not have the individual option leg in memory.
    """

    def __init__(self, ltp_source: Any, data_engine: Any = None):
        self._ltp = ltp_source
        self._data_engine = data_engine

    def get_ltp(self, instrument: str) -> float:
        try:
            v = self._ltp.get_ltp(instrument)
            if v is not None and float(v) > 0.0:
                return float(v)
        except Exception as e:
            logger.debug(f"LivePriceFeed.get_ltp({instrument}) failed on primary source: {e}")

        # Real option quote fallback via Angel One
        if self._data_engine and getattr(self._data_engine, "angelone_options", None):
            try:
                # Handle 2-leg credit spreads (e.g. NIFTY2690824000CE/NIFTY2690824150CE)
                if "/" in instrument:
                    legs = [l.strip() for l in instrument.split("/") if l.strip()]
                    if len(legs) == 2:
                        sym1, strike1, opt1 = self._parse_option_instrument(legs[0])
                        sym2, strike2, opt2 = self._parse_option_instrument(legs[1])
                        if sym1 and strike1 and opt1 and sym2 and strike2 and opt2:
                            q1 = self._data_engine.angelone_options.get_real_premium(sym1, strike1, opt1)
                            q2 = self._data_engine.angelone_options.get_real_premium(sym2, strike2, opt2)
                            if q1 and "ltp" in q1 and q2 and "ltp" in q2:
                                ltp1 = float(q1.get("ltp", 0.0) or 0.0)
                                ltp2 = float(q2.get("ltp", 0.0) or 0.0)
                                spread_val = max(0.05, ltp1 - ltp2)
                                return spread_val

                # Single-leg option quote
                sym, strike, opt_type = self._parse_option_instrument(instrument)
                if sym and strike and opt_type:
                    q = self._data_engine.angelone_options.get_real_premium(sym, strike, opt_type)
                    if q and "ltp" in q and q["ltp"] is not None:
                        return float(q["ltp"])
            except Exception as e:
                logger.debug(f"LivePriceFeed option fallback error for {instrument}: {e}")

        return 0.0

    def _parse_option_instrument(self, ts: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
        """Extract symbol, strike, and option_type from standard Indian option tradingsymbol."""
        ao = getattr(self._data_engine, "angelone_options", None) if self._data_engine else None
        if ao and hasattr(ao, "UNDERLYING_MAP") and hasattr(ao, "_parse_tradingsymbol"):
            for sym_key, underlying in ao.UNDERLYING_MAP.items():
                if ts.startswith(underlying):
                    parsed = ao._parse_tradingsymbol(ts, underlying)
                    if parsed and "strike" in parsed and "option_type" in parsed:
                        return sym_key, int(parsed["strike"]), parsed["option_type"]

        import re
        opt_type = "CE" if ts.endswith("CE") else ("PE" if ts.endswith("PE") else None)
        if not opt_type:
            return None, None, None
        
        sym_map = {
            "BANKNIFTY": "NIFTY BANK",
            "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
            "FINNIFTY": "NIFTY FIN SERVICE",
            "NIFTY": "NIFTY 50",
            "SENSEX": "SENSEX",
        }
        for prefix, sym in sym_map.items():
            if ts.startswith(prefix):
                m = re.search(r'(\d+)(?:CE|PE)$', ts)
                if m:
                    strike = int(m.group(1))
                    return sym, strike, opt_type
        return None, None, None

    def get_quote(self, instrument: str):
        # Bid/ask not always available; the FillSimulator falls back to LTP.
        try:
            if hasattr(self._ltp, "get_quote"):
                q = self._ltp.get_quote(instrument)
                if q is not None:
                    return q
        except Exception:
            pass
        ltp = self.get_ltp(instrument)
        if ltp > 0:
            # Synthesize a tightish quote for paper-mode simulation only.
            return (ltp, ltp * 0.999, ltp * 1.001)
        return None


# ---------------------------------------------------------------------
# LivePaperCapture -- owns one PaperTradeEngine + lifecycle plumbing
# ---------------------------------------------------------------------

@dataclass
class CaptureConfig:
    """All paper-capture knobs. Mirrors ``settings.paper_capture`` block."""
    enabled: bool = False
    csv_path: str = "reports/papertrade/live_ledger.csv"
    sqlite_path: str = "reports/papertrade/live_ledger.sqlite"
    max_concurrent_positions: int = 200    # effectively uncapped
    allow_duplicate_instrument: bool = True
    enable_trailing: bool = True
    default_max_bars_intraday: int = 16
    default_max_bars_swing: int = 96
    cost_per_side_bps: float = 1.0
    slippage_bps: int = 15

    @classmethod
    def from_settings(cls, settings: Optional[dict] = None) -> "CaptureConfig":
        if not settings:
            return cls()
        pcfg = settings.get("paper_capture", {}) or {}
        return cls(
            enabled=bool(pcfg.get("enabled", False)),
            csv_path=str(pcfg.get("csv_path", "reports/papertrade/live_ledger.csv")),
            sqlite_path=str(pcfg.get("sqlite_path", "reports/papertrade/live_ledger.sqlite")),
            max_concurrent_positions=int(pcfg.get("max_concurrent_positions", 200)),
            allow_duplicate_instrument=bool(pcfg.get("allow_duplicate_instrument", True)),
            enable_trailing=bool(pcfg.get("enable_trailing", True)),
            default_max_bars_intraday=int(pcfg.get("default_max_bars_intraday", 16)),
            default_max_bars_swing=int(pcfg.get("default_max_bars_swing", 96)),
            cost_per_side_bps=float(pcfg.get("cost_per_side_bps", 1.0)),
            slippage_bps=int(pcfg.get("slippage_bps", 15)),
        )


class LivePaperCapture:
    """Single-instance adapter used by Prometheus.

    Two entry points:

      ``on_signal(refined_signal: dict)`` -- called by main.py after the
      signal has passed quality filters. Opens a paper position via the
      engine.

      ``on_bar(symbol, ohlc)`` -- called by the bar polling loop with the
      most recent closed bar. Drives exit evaluation.

    Reads / writes ``reports/papertrade/live_ledger.csv`` and
    ``reports/papertrade/live_ledger.sqlite`` for trade history.
    """

    def __init__(self, config: CaptureConfig, ltp_source: Any, telegram: Any = None, data_engine: Any = None):
        self.config = config
        self.enabled = bool(config.enabled)
        # Optional telegram forwarder — passed in by Prometheus.init_paper_capture.
        # If None, alert helpers are silently skipped. Telegram is the LIVE
        # production bot instance; we only call its send_message / alert_*
        # methods (which are non-blocking on their own worker) and we never
        # mutate its state. Paper-capture activity must never interfere with
        # the live dispatch path's use of the same singleton.
        self._telegram = telegram
        self._data_engine = data_engine
        # Capture every closed trade so we can alert after each process_bar.
        # The wrapped process_bar (below) emits a callback per close.
        self._on_close_listeners = []

        # Construct the recorder (CSV+SQLite) directory-safe.
        Path(config.csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        self._feed = LivePriceFeed(ltp_source, data_engine=data_engine)
        self._recorder = TradeRecorder(
            sqlite_path=config.sqlite_path,
            csv_path=config.csv_path,
        )
        # Build the tracker with the requested cost model.
        # Bug C.2 (2026-07-25 audit): pass the recorder reference so
        # ``PositionTracker.open_position`` / ``close_position`` write to
        # the ``paper_open_positions`` SQLite table. That keeps open
        # position state durable across process restarts (the previous
        # failure mode was that open paper_capture positions were silently
        # abandoned on every restart — the trade simply vanished from the
        # in-memory dict and never got a recorded exit).
        tracker = PositionTracker(
            fill_sim=FillSimulator(
                feed=self._feed,
                slippage_bps=config.slippage_bps,
                use_bid_ask=True,
            ),
            cost_model=CostModel(cost_per_side_bps=config.cost_per_side_bps),
            enable_trailing=config.enable_trailing,
            recorder=self._recorder,
        )
        # PaperTradeEngine with high cap + allow_duplicate_instrument=True
        # so EVERY valid signal becomes a paper position (the user's stated
        # goal: evaluate the strategy itself, not risk management).
        self._engine = PaperTradeEngine(
            feed=self._feed,
            signal_source=None,
            recorder=self._recorder,
            enable_trailing=config.enable_trailing,
            default_max_bars_intraday=config.default_max_bars_intraday,
            default_max_bars_swing=config.default_max_bars_swing,
            max_concurrent_positions=config.max_concurrent_positions,
            allow_duplicate_instrument=config.allow_duplicate_instrument,
        )
        # Replace the engine's auto-constructed tracker with our own
        # configured one (PaperTradeEngine builds a default in __init__,
        # but we want our config-specific CostModel + slippage).
        self._engine.tracker = tracker
        # MetricsEngine is the metrics aggregator.
        self._metrics: MetricsEngine = self._engine.metrics

        # Bug C.2 (2026-07-25 audit): re-hydrate any open paper positions
        # left in the SQLite table from the previous run. Each row is a
        # ``Position`` snapshot captured when the position was first opened
        # (and kept in step on every bar via the tracker hooks above —
        # well, almost: today we only persist at OPEN and DELETE at CLOSE,
        # which is enough to survive a sudden crash but doesn't keep
        # ``bars_held`` / ``breakeven_set`` / ``high_water_mark`` in step
        # across a restart). The recovered positions lose any in-flight
        # trailing/progress state — they re-enter the exit logic at "fresh
        # entry" status the next bar, which is a conservative behavior:
        # the SL/target/time-stop all still apply; only the BE/trailing
        # progress resets. Acceptable for paper-mode integrity (the goal
        # is to NOT LOSE the position entirely).
        try:
            recovered = self._load_and_rehydrate_open_positions()
            if recovered:
                logger.info(
                    f"[PaperCapture] recovered {recovered} open position(s) "
                    f"from SQLite (Bug C.2 persistence)"
                )
        except Exception as e:
            logger.warning(f"[PaperCapture] open-position rehydrate failed: {e}")

        # Wrap ``self._engine.process_bar`` so we receive a callback every
        # time the engine closes a paper position (SL/target/trailing/time-stop/
        # square_off/end_of_data). Each closed PaperTrade is forwarded to the
        # telegram alert helpers and any other listeners (e.g. real-time
        # statistics aggregator). The original method's return value list is
        # preserved (we don't change its public contract).
        self._engine_process_bar_orig = self._engine.process_bar
        def _wrapped_process_bar(snapshot, is_session_end=False, is_square_off=False):
            closed = self._engine_process_bar_orig(snapshot, is_session_end=is_session_end, is_square_off=is_square_off)
            for trade in closed:
                self._on_trade_closed(trade)
            return closed
        self._engine.process_bar = _wrapped_process_bar

        logger.info(
            f"[PaperCapture] initialized "
            f"enabled={self.enabled} csv={config.csv_path} "
            f"sqlite={config.sqlite_path} "
            f"max_pos={config.max_concurrent_positions} "
            f"dup_ok={config.allow_duplicate_instrument} "
            f"trailing={config.enable_trailing} "
            f"telegram={'on' if self._telegram else 'off'}"
        )

    # -----------------------------------------------------------------
    # Public surface (called from main.py)
    # -----------------------------------------------------------------

    def on_signal(self, refined_signal: dict) -> Optional[str]:
        """Forward a refined signal to the engine. Returns trade_id on success
        or None on skip. Never raises.

        Robustness behaviors:
          - If ``strike`` / ``expiry`` / ``instrument`` are missing (live
            path couldn't price an option chain — common for ICICIBANK,
            TATAMOTORS, NIFTY MIDCAP, SENSEX when Angel One searchScrip
            returns no data for the symbol), we synthesize a placeholder
            instrument key and use ``entry_price_hint`` from the strategy
            (the `_price_options` ``source=BS`` fallback). FillSimulator
            fills at the hint price — we still get a tracked paper trade.
          - If, even after the synthetic-instrument fallback, the engine
            skips the signal (rare — only when entry_price_hint is also
            0), we log a ``WARNING`` so you can see exactly which signals
            disappear and why they couldn't be captured.
        """
        if not self.enabled:
            return None
        try:
            notif = self._build_signal_notification(refined_signal)
        except Exception as e:
            logger.warning(f"[PaperCapture] signal convert failed: {e}")
            return None

        # Anti-Overtrading Guard: Never open concurrent duplicate positions on the same symbol
        is_spread = "/" in (notif.instrument or "") or "SPREAD" in getattr(notif, "strategy", "").upper()
        for open_pos in self._engine.tracker.open_positions.values():
            if open_pos.symbol == notif.symbol:
                if is_spread and ("/" in (open_pos.instrument or "") or "SPREAD" in getattr(open_pos, "strategy", "").upper()):
                    logger.info(f"[PaperCapture] Skipping spread on {notif.symbol} — an active spread is already open ({open_pos.trade_id})")
                    return None
                if not is_spread and open_pos.direction == notif.direction:
                    logger.info(
                        f"[PaperCapture] Skipping duplicate {notif.direction.value} position on {notif.symbol} — "
                        f"active trade ({open_pos.trade_id} {open_pos.instrument}) is already open"
                    )
                    return None

        try:
            trade_id = self._engine.process_new_signal(notif)
            if trade_id:
                logger.info(
                    f"[PaperCapture] opened {notif.symbol} {notif.direction.value} "
                    f"{notif.instrument} @ hint={notif.entry_price_hint:.2f} "
                    f"id={trade_id}"
                )
                self._alert_position_opened(notif, trade_id)
            else:
                # The engine logs its own detailed skip reason.
                # Emit a WARNING so you have visibility into silent rejections.
                logger.warning(
                    f"[PaperCapture] SKIP {notif.symbol} {notif.direction.value} "
                    f"instrument={notif.instrument or '<empty>'} "
                    f"hint={notif.entry_price_hint:.2f} "
                    f"(seen={self._engine.signals_seen} "
                    f"skipped_full={self._engine.signals_skipped_full} "
                    f"skipped_dup={self._engine.signals_skipped_duplicate} "
                    f"no_quote={self._engine.signals_skipped_no_quote})"
                )
                # Send a telegram alert so silent rejections are visible.
                self._alert_signal_rejected(notif, refined_signal)
            return trade_id
        except Exception as e:
            logger.error(f"[PaperCapture] on_signal error: {e}")
            return None

    def _build_signal_notification(self, refined_signal: dict):
        """Convert a refined signal dict to a SignalNotification, with
        synthetic-instrument fallback for symbols where Angel One searchScrip
        returned no option chain (TATAMOTORS, ICICIBANK, NIFTY MIDCAP SELECT,
        SENSEX). Uses ``entry_price`` from the strategy (``source=BS`` fallback
        from ``_price_options``) as the fill hint.

        We attempt the standard ``from_signal_dict`` first; if the result has
        an empty instrument, we synthesize one and set ``entry_price_hint``
        from the strategy's premium estimate.
        """
        notif = from_signal_dict(refined_signal)

        # If live path provided no priced strike, attempt synthetic fallback
        # so we still track the signal end-to-end with the strategy's own
        # premium estimate (Black-Scholes theoretical price).
        if not notif.instrument and notif.entry_price_hint > 0:
            synth_id = (
                f"SYNTH_{(notif.symbol or 'UNK').replace(' ', '_')}_"
                f"{notif.direction.value}"
            )
            if notif.expiry:
                synth_id += f"_{notif.expiry.replace('-', '')}"
            if notif.strike > 0:
                synth_id += f"_{int(notif.strike)}{notif.option_type}"
            logger.info(
                f"[PaperCapture] synthetic-instrument fallback for "
                f"{notif.symbol} {notif.direction.value} "
                f"(no live strike; using hint=Rs {notif.entry_price_hint:.2f}): "
                f"id={synth_id}"
            )
            notif.instrument = synth_id

        return notif

    def _alert_signal_rejected(self, notif, original_refined: dict) -> None:
        """Send a telegram alert when a signal reached _dispatch_multi_account
        but the PaperCapture engine couldn't open a position. Gives you
        visibility into silent signal loss (e.g. no LTP, no priced strike).
        """
        if self._telegram is None:
            return
        try:
            if not getattr(self.config, "skip_alerts", True):
                return
            reason_parts = []
            if not notif.instrument:
                reason_parts.append("no instrument (Angel One searchScrip returned no data)")
            if notif.entry_price_hint <= 0:
                reason_parts.append("no entry_price hint (strategy premium estimate missing)")
            reason = "; ".join(reason_parts) if reason_parts else "FillSimulator rejected"
            try:
                self._telegram.send_message(
                    f"\u26a0\ufe0f <b>PAPER CAPTURE — signal skipped</b>\n"
                    f"{notif.symbol} {notif.direction.value}\n"
                    f"Reason: {reason}\n"
                    f"Hint price: Rs {notif.entry_price_hint:.2f}\n"
                    f"Strike: {notif.strike}  Expiry: {notif.expiry or 'none'}\n"
                    f"Score: {notif.signal_score:.2f}\n"
                    f"Strategy: {notif.strategy}"
                )
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[PaperCapture] _alert_signal_rejected failed: {e}")

    # -----------------------------------------------------------------
    # Telegram alert helpers (no-ops if no telegram instance wired)
    # -----------------------------------------------------------------

    def _alert_position_opened(self, notif, trade_id: str) -> None:
        # Internal capture logging — main signal alert already sent rank & execution details
        logger.info(
            f"[PaperCapture] Tracked position opened: {notif.symbol} {notif.instrument} "
            f"@{notif.entry_price_hint:.2f} (TradeID: {trade_id})"
        )

    def _alert_position_closed(self, trade) -> None:
        if self._telegram is None:
            return
        try:
            side = "BUY CE" if trade.direction.value == "LONG" else "BUY PE"
            trade_info = {
                "symbol": trade.symbol,
                "side": side,
                "quantity": trade.quantity,
                "price": trade.exit_price,
                "exit_price": trade.exit_price,
                "pnl": trade.net_pnl,
                "net_pnl": trade.net_pnl,
                "gross_pnl": trade.gross_pnl,
                "costs": {"total": trade.costs},
                "equity": 0,   # we don't track equity here — leave 0
            }
            reason = str(trade.exit_reason)
            try:
                if reason == "target":
                    self._telegram.alert_target_hit(trade_info)
                elif reason == "stop_loss":
                    self._telegram.alert_stop_loss_hit(trade_info)
                elif reason in ("stop_loss_premium_phase2",
                                "stop_loss_premium_phase3"):
                    # Trailing-stop lock: phase3 locks ≥70% of peak profit,
                    # phase2 locks ≥20%. Calling these "STOP LOSS HIT"
                    # mislabels a profitable exit as a loss (per backtest
                    # NIFTY 50: phase3 = 67%WR, +Rs 9,952; phase2 = 0%WR,
                    # small Rs -215). Route to a distinct alert so the
                    # operator sees what actually happened.
                    phase = "phase3" if reason.endswith("phase3") else "phase2"
                    self._telegram.alert_trailing_lock_hit(trade_info, phase=phase)
                else:
                    # time_stop / square_off / end_of_day / end_of_data /
                    # reverse_signal / manual — use generic close
                    self._telegram.alert_trade_closed(trade_info)
            except Exception:
                pass
            # Custom paper-capture summary line — gives full forensic detail
            try:
                pnl_emoji = "\U0001f4c8" if trade.net_pnl >= 0 else "\U0001f4c9"
                self._telegram.send_message(
                    f"{pnl_emoji} <b>PAPER CAPTURE closed</b>\n"
                    f"{trade.symbol} {side} {trade.instrument}\n"
                    f"Entry: Rs {trade.entry_price:.2f} → "
                    f"Exit: Rs {trade.exit_price:.2f}\n"
                    f"Reason: <b>{reason}</b>\n"
                    f"Gross: Rs {trade.gross_pnl:+,.2f} | "
                    f"Costs: Rs {trade.costs:.2f} | "
                    f"Net: Rs {trade.net_pnl:+,.2f} "
                    f"({trade.return_pct:+.2f}%)\n"
                    f"Hold: {trade.holding_duration_seconds // 60} min\n"
                    f"ID: <code>{trade.trade_id}</code>"
                )
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[PaperCapture] _alert_position_closed failed: {e}")

    def _on_trade_closed(self, trade) -> None:
        """Single chokepoint invoked by the wrapped process_bar on every
        closed PaperTrade. Updates in-memory listeners and pushes telegram.
        """
        self._alert_position_closed(trade)
        for listener in self._on_close_listeners:
            try:
                listener(trade)
            except Exception as e:
                logger.debug(f"[PaperCapture] on_close listener failed: {e}")

    def add_close_listener(self, callback) -> None:
        """Register an additional callback invoked after every close."""
        self._on_close_listeners.append(callback)

    def on_bar(self, symbol: str, bar: Dict[str, Any],
               is_session_end: bool = False, is_square_off: bool = False):
        """Feed a closed OHLC bar to the engine for exit evaluation.

        Args:
            symbol: display symbol ("NIFTY 50" etc).
            bar: dict with at least {timestamp, open, high, low, close, volume?}.
            is_session_end: True if this is the last bar of the trading day.
            is_square_off: True if intraday square-off window (>= 15:15 IST)
                has been reached. Caller decides this (prometheus.main drives
                the clock and computes the trigger). Propagated unmodified to
                ``PaperTradeEngine.process_bar`` so the tracker's square-off
                exit branch (otherwise dead code) fires for intraday positions.

        Bug #8 (2026-07-22): previously ``on_bar`` accepted only
        ``is_session_end`` — the live caller at ``main.py:520`` always
        passed ``is_session_end=False`` and never passed
        ``is_square_off=True`` even when the 15:15 IST square-off fired
        (the live-loop branch at main.py:4619 only invokes
        ``_square_off_intraday_positions`` against the legacy broker,
        NOT the paper engine). Result: intraday paper positions never
        force-closed at 15:15, leaving them open across session boundary
        and accumulating phantom P&L. Fix: extend the signature to
        accept ``is_square_off`` and forward it through to the engine,
        so upstream callers (``_paper_capture_feed_bars``, the
        ``_square_off_intraday_positions`` path, etc.) can drive
        square-off semantics for paper_capture just like they do for
        legacy broker positions.
        """
        if not self.enabled:
            return
        try:
            ts = bar.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            elif ts is None:
                ts = datetime.now(IST)
            snap = TradeSnapshot(
                timestamp=ts.replace(tzinfo=None) if ts.tzinfo else ts,
                symbol=symbol,
                instrument=bar.get("instrument", ""),
                open=float(bar.get("open", 0.0)),
                high=float(bar.get("high", 0.0)),
                low=float(bar.get("low", 0.0)),
                close=float(bar.get("close", 0.0)),
                volume=float(bar.get("volume", 0.0) or 0.0),
                bar_interval=str(bar.get("interval", "15minute")),
            )
            self._engine.process_bar(
                snap,
                is_session_end=is_session_end,
                is_square_off=is_square_off,
            )
        except Exception as e:
            logger.error(f"[PaperCapture] on_bar error for {symbol}: {e}")

    # -----------------------------------------------------------------
    # Query surface (called from CLI / Telegram)
    # -----------------------------------------------------------------

    def stats(self) -> Optional[TradeStats]:
        if not self.enabled:
            return None
        return self._engine.stats()

    def open_positions_view(self):
        if not self.enabled:
            return []
        return [p.to_dict() for p in self._engine.tracker.open_positions.values()]

    def _load_and_rehydrate_open_positions(self) -> int:
        """Bug C.2 (2026-07-25 audit): rehydrate open paper positions from
        the SQLite ``paper_open_positions`` table into the in-memory
        ``PositionTracker.open_positions`` dict.

        Returns the count of rows successfully re-hydrated. Skips rows
        whose schema doesn't match the current ``Position`` dataclass
        (logged at WARNING).

        Re-hydrated positions re-enter the exit logic on the next bar
        with their saved ``stop_loss`` / ``target`` / ``max_bars``
        intact — but their trailing-stop progress (``breakeven_set``,
        ``high_water_mark``, ``trailing_floor``) is whichever value we
        persisted at OPEN time (since the tracker doesn't currently
        update those on every bar). Conservative: positions lose any
        in-flight trailing progress across the restart, but they DO get
        evaluated against SL/target/time-stop on the next bar — that's
        the intended paper-integrity guarantee (no silent ghost-loss).
        """
        if not self.enabled:
            return 0
        rows = self._recorder.load_open_positions()
        if not rows:
            return 0
        count = 0
        for r in rows:
            try:
                from prometheus.papertrade.types import ExitReason  # noqa: F401
                # Parse entry_time (ISO string) back to tz-aware datetime.
                # The recorder stores whatever ``Position.to_dict`` emits
                # (which is ``self.entry_time.isoformat()``). If the entry
                # was IST-aware, the isoformat string preserves the tz.
                entry_time_str = r.get("entry_time") or ""
                entry_time = datetime.fromisoformat(entry_time_str)
                # ``Position.to_dict`` strips the tz from datetimes that
                # were stored tz-naive (Python's isoformat omits tzinfo if
                # it's None). Re-localize to IST to match the live path
                # (consistent with the bug #1 fix at engine.py:215-216).
                if entry_time.tzinfo is None:
                    entry_time = IST.localize(entry_time)
                pos = Position(
                    trade_id=r["trade_id"],
                    symbol=r.get("symbol", ""),
                    instrument=r.get("instrument", ""),
                    underlying=r.get("underlying", ""),
                    direction=Direction(r.get("direction", "LONG")),
                    quantity=int(r.get("quantity") or 0),
                    entry_price=float(r.get("entry_price") or 0),
                    entry_time=entry_time,
                    stop_loss=float(r.get("stop_loss") or 0),
                    target=float(r.get("target") or 0),
                    max_bars=int(r.get("max_bars") or 0),
                    bars_held=int(r.get("bars_held") or 0),
                    max_bars_allowed=None,
                    breakeven_set=bool(int(r.get("breakeven_set") or 0)),
                    trailing_floor=float(r.get("trailing_floor") or 0),
                    high_water_mark=float(r.get("high_water_mark") or 0),
                    strategy=r.get("strategy", "") or "",
                    signal_score=float(r.get("signal_score") or 0),
                    signal_confidence=float(r.get("signal_confidence") or 0),
                    trade_mode=r.get("trade_mode", "intraday") or "intraday",
                )
                # Inject into the tracker's in-memory dict ONLY if not
                # already present (defensive — should never happen, but
                # the engine re-runs ``open_position`` for duplicate IDs
                # and we don't want that to mask a re-hydrated row).
                if pos.trade_id in self._engine.tracker.open_positions:
                    logger.warning(
                        f"[PaperCapture] rehydrate skip existing "
                        f"{pos.trade_id}"
                    )
                    continue
                self._engine.tracker.open_positions[pos.trade_id] = pos
                count += 1
                logger.info(
                    f"[PaperCapture] rehydrated {pos.trade_id} "
                    f"{pos.direction.value} {pos.instrument} @ "
                    f"Rs {pos.entry_price:.2f} from SQLite"
                )
            except Exception as e:
                logger.warning(
                    f"[PaperCapture] rehydrate skip invalid row "
                    f"{r.get('trade_id')}: {e}"
                )
        return count

    def recent_trades(self, n: int = 20):
        """Return the N most-recently closed paper trades as dicts (newest last).

        Uses ``TradeRecorder.load_previously_closed_trades`` which reads the
        SQLite store; falls back to ``[]`` if recorder/SQLite unavailable.
        """
        if not self.enabled:
            return []
        try:
            trades = self._recorder.load_previously_closed_trades() or []
            tail = trades[-n:] if n > 0 else trades
            return [t.to_dict() for t in tail]
        except Exception as e:
            logger.debug(f"[PaperCapture] recent_trades failed: {e}")
            return []

    def close(self):
        """Flush state on shutdown lifecycle."""
        try:
            # Force the recorder to write any pending stats snapshot.
            snap = self.stats()
            if snap is not None:
                self._recorder.record_stats_snapshot(snap)
        except Exception:
            pass


# ---------------------------------------------------------------------
# Module-level convenience helpers (used by main.py)
# ---------------------------------------------------------------------

def is_paper_capture_enabled(settings: Optional[dict] = None) -> bool:
    """Decide, from current settings, whether to instantiate LivePaperCapture.

    Rule (per user requirement): paper-capture must run in paper mode only,
    opt-in via ``settings.paper_capture.enabled``. The production Kite path
    (mode != "paper") never constructs this object.
    """
    if not settings:
        return False
    if str(settings.get("system", {}).get("mode", "")).lower() != "paper":
        return False
    return bool((settings.get("paper_capture", {}) or {}).get("enabled", False))


def get_paper_capture(settings: Optional[dict] = None, ltp_source: Any = None, telegram: Any = None, data_engine: Any = None):
    """Factory called from main.py once at startup.

    Returns a LivePaperCapture if enabled, else None.

    Args:
        settings: full config dict (must include system.* and paper_capture.*).
        ltp_source: object exposing ``get_ltp(instrument) -> float`` for
            fill simulation (live broker, AngelOne fetcher, or test feed).
        telegram: optional telegram bot instance (Object exposing
            ``send_message``, ``alert_order_placed``, ``alert_target_hit``,
            ``alert_stop_loss_hit``, ``alert_trade_closed``). If None,
            telegram alerts are silently skipped.
        data_engine: optional DataEngine instance with AngelOneOptionChain for real option quotes.
    """
    if not is_paper_capture_enabled(settings):
        return None
    return LivePaperCapture(
        CaptureConfig.from_settings(settings),
        ltp_source,
        telegram=telegram,
        data_engine=data_engine,
    )
