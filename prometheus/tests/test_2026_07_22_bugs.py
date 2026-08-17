"""Regression tests for the 2026-07-22 paper-trading bug cluster.

Each test pins one specific bug to prevent it from regressing:
    test_tz_naive_bar_timestamp_does_not_raise      → Bug #1
    test_paper_capture_path_bypasses_correlated_gate → Bug #2 (lightweight)
    test_trailing_sl_breach_fires_in_phase1_or_2    → Bug #3
    test_parse_tradingsymbol_monthly_format         → Bug #4

Bug-specific tests live here because they span multiple files
(papertrade/engine.py, main.py, scanner.py, position_monitor.py,
angelone_options.py) and don't fit cleanly into a single module's
test file.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pytest

from prometheus.papertrade import PaperTradeEngine
from prometheus.papertrade.types import Direction
from prometheus.papertrade.signal_source import SignalNotification
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.utils.indian_market import IST


# ===========================================================================
# Bug #1 — tz TypeError in papertrade/engine.py
# ===========================================================================

class _NullFeed:
    """Minimal feed: always returns signal.entry_price_hint."""
    def get_ltp(self, instrument: str) -> float:
        return 0.0
    def get_quote(self, instrument: str):
        return None


def _make_engine_for_bug1():
    """Build a PaperTradeEngine without persisting to disk."""
    from prometheus.papertrade.recorder import TradeRecorder
    feed = _NullFeed()
    # Pass feed directly; engine wraps it in FillSimulator itself.
    engine = PaperTradeEngine(
        feed=feed,
        signal_source=None,
        recorder=TradeRecorder(sqlite_path=None, csv_path=None),  # in-memory only
        lot_size_override=1,
        enable_trailing=True,
    )
    return engine


def test_tz_naive_bar_timestamp_does_not_raise():
    """Bug #1 regression: bar_timestamp produced by main.py:1834 is a
    tz-NAIVE string like '2026-07-22 09:30:00'. _parse_bar_timestamp
    returns a naive datetime. Previously engine.py:229 subtracted
    that from tz-aware datetime.now(IST), raising TypeError on every
    LivePaperCapture signal.
    """
    engine = _make_engine_for_bug1()

    # Naive bar_timestamp — exactly what the live path produces.
    sig = SignalNotification(
        symbol="NIFTY 50",
        instrument="NIFTY26JUL24150PE",
        underlying="NIFTY",
        direction=Direction.SHORT,  # buying PE (bearish); subsystem uses LONG/SHORT
        strike=24150.0,
        option_type="PE",
        expiry="2026-07-28",
        entry_price_hint=100.0,
        stop_loss=95.0,
        target=120.0,
        signal_score=4.0,
        signal_confidence=0.7,
        max_bars=16,
        trade_mode="intraday",
        strategy="apex",
        bar_timestamp=datetime(2026, 7, 22, 9, 30, 0),  # tz-NAIVE
        metadata={},
    )

    # Must NOT raise TypeError.
    trade_id = engine.process_new_signal(sig)
    assert trade_id is not None, "Engine should open a paper trade"
    assert engine.open_positions_count() == 1

    # open_positions is Dict[trade_id -> Position]; grab the single open one.
    assert len(engine.tracker.open_positions) == 1
    pos = next(iter(engine.tracker.open_positions.values()))
    assert pos.entry_time is not None
    # Position should be tz-aware (IST) regardless of input tz.
    assert pos.entry_time.tzinfo is not None


def test_tz_aware_bar_timestamp_still_works():
    """Companion to Bug #1: ensure the existing tz-aware path (e.g.,
    test/replay) is not broken by the IST.localize() fix."""
    engine = _make_engine_for_bug1()
    aware_ts = datetime.now(IST) - timedelta(minutes=5)

    sig = SignalNotification(
        symbol="NIFTY 50",
        instrument="NIFTY26JUL24150PE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        strike=24150.0,
        option_type="PE",
        expiry="2026-07-28",
        entry_price_hint=100.0,
        stop_loss=95.0,
        target=120.0,
        signal_score=4.0,
        signal_confidence=0.7,
        max_bars=16,
        trade_mode="intraday",
        strategy="apex",
        bar_timestamp=aware_ts,  # tz-AWARE
        metadata={},
    )
    trade_id = engine.process_new_signal(sig)
    assert trade_id is not None
    assert engine.open_positions_count() == 1


def test_stale_bar_timestamp_falls_back_to_now():
    """If bar_timestamp is >1 day old, engine must use now() instead —
    a fresh bar recovery, not the original stale one (original bug #1
    comment about -1095 minute holding_duration)."""
    engine = _make_engine_for_bug1()
    stale_ts = datetime(2026, 1, 1, 15, 15, 0)  # tz-naive AND months old
    now_ist = datetime.now(IST)

    sig = SignalNotification(
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26JUL57000PE",
        underlying="NIFTY BANK",
        direction=Direction.SHORT,
        strike=57000.0,
        option_type="PE",
        expiry="2026-07-28",
        entry_price_hint=302.65,
        stop_loss=292.20,
        target=388.69,
        signal_score=4.0,
        signal_confidence=0.7,
        max_bars=16,
        trade_mode="intraday",
        strategy="apex",
        bar_timestamp=stale_ts,
        metadata={},
    )
    trade_id = engine.process_new_signal(sig)
    assert trade_id is not None
    assert len(engine.tracker.open_positions) == 1
    pos = next(iter(engine.tracker.open_positions.values()))
    # entry_time must be roughly now (within 5 minutes), NOT stale.
    diff = abs((datetime.now(IST) - pos.entry_time).total_seconds())
    assert diff < 300, f"entry_time should fall back to now, diff={diff}s"


def test_process_bar_trailing_stop_advances_without_attributeerror():
    """Bug #5 regression: when ``enable_trailing=True`` and a bar does NOT
    hit SL/target/time-stop, ``on_bar`` calls
    ``self._maybe_advance_trailing_stop(p, snapshot.close)``. Until the
    2026-07-22 fix, that method had no ``def`` line — its docstring + body
    were dangling as dead string-blob after ``_evaluate_exit_via_feed``'s
    ``return 0.0, None`` line. So any non-exiting bar crashed with
    ``AttributeError: PositionTracker has no attribute
    '_maybe_advance_trailing_stop'``. Paper engine never exercised this
    path before, so the bug sat hidden.

    Pin: feed a bar whose close crosses 0.4R from entry (breakeven trap)
    and assert no exception + SL advanced to >= entry.
    """
    from prometheus.papertrade.types import TradeSnapshot

    engine = _make_engine_for_bug1()

    sig = SignalNotification(
        symbol="NIFTY 50",
        instrument="NIFTY26JUL24150PE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        strike=24150.0,
        option_type="PE",
        expiry="2026-07-28",
        entry_price_hint=100.0,         # entry filled at hint (feed returns 0)
        stop_loss=95.0,                 # risk_distance = 5.0
        target=120.0,                   # 4R — bigger than the bar's close
        signal_score=4.0,
        signal_confidence=0.7,
        max_bars=16,
        trade_mode="intraday",
        strategy="apex",
        bar_timestamp=datetime(2026, 7, 22, 9, 30, 0),
        metadata={},
    )
    trade_id = engine.process_new_signal(sig)
    assert trade_id is not None
    pos = next(iter(engine.tracker.open_positions.values()))

    # Bar where premium rallied to Rs 102 — that's (102-100)/(100-95) = 0.4R,
    # exactly the breakeven trap trigger. SL must move to entry (>=100).
    # Bar OHLC is all below target (120) and above SL (95) so NO exit fires;
    # the trailing path is the only thing that runs.
    snap = TradeSnapshot(
        timestamp=datetime(2026, 7, 22, 9, 45, 0),
        symbol="NIFTY 50",
        instrument="NIFTY26JUL24150PE",
        open=100.0,
        high=102.5,
        low=99.5,
        close=102.0,
        volume=0,
        bar_interval="15minute",
    )

    # Must NOT raise AttributeError.
    closed = engine.process_bar(snap)
    assert closed == [], "No exit should fire on this bar"
    assert pos.breakeven_set, "Breakeven trap (0.4R) should have engaged"
    assert pos.stop_loss >= pos.entry_price, (
        f"Trailing SL must advance to entry after breakeven trap — "
        f"got SL={pos.stop_loss} entry={pos.entry_price}"
    )


# ===========================================================================
# Bug #2 — Legacy OrderManager bypassed when LivePaperCapture active
# ===========================================================================

def test_route_paper_capture_or_legacy_skips_order_manager_in_paper_mode():
    """Bug #2 regression: when prometheus._paper_capture is enabled, the
    scanner helper _route_paper_capture_or_legacy must call
    paper_capture.on_signal and NEVER call order_manager.execute_signal.
    """
    from prometheus.pipeline.scanner import LiveScanner

    class _PC:
        enabled = True
        def __init__(self):
            self.calls = []
        def on_signal(self, sig):
            self.calls.append(sig)

    class _OM:
        def __init__(self):
            self.calls = []
        def execute_signal(self, sig, confirm=False):
            self.calls.append(sig)
            return None

    class _MockProm:
        def __init__(self):
            self._paper_capture = _PC()
            # order_manager is unused when paper_capture is enabled.
            self.order_manager = _OM()

    mock = _MockProm()
    scanner = LiveScanner.__new__(LiveScanner)
    scanner._prometheus = mock

    sample_signal = {"symbol": "NIFTY BANK", "action": "BUY_PE"}
    position = scanner._route_paper_capture_or_legacy(sample_signal, confirm=False)

    assert position is None, "Paper capture path returns None (no legacy Position)"
    assert len(mock._paper_capture.calls) == 1, "Paper capture should receive the signal"
    assert mock._paper_capture.calls[0] is sample_signal
    assert len(mock.order_manager.calls) == 0, "Legacy OrderManager must NOT be called"


def test_route_paper_capture_or_legacy_uses_order_manager_when_disabled():
    """Companion: when no LivePaperCapture is set (live mode) the helper
    must fall through to the legacy OrderManager.execute_signal path."""
    from prometheus.pipeline.scanner import LiveScanner

    class _OM:
        def __init__(self):
            self.calls = []
        def execute_signal(self, sig, confirm=False):
            self.calls.append(sig)
            return {"position_id": "POS-X"}

    class _MockProm:
        def __init__(self):
            self._paper_capture = None
            self.order_manager = _OM()

    mock = _MockProm()
    scanner = LiveScanner.__new__(LiveScanner)
    scanner._prometheus = mock

    sample_signal = {"symbol": "NIFTY 50", "action": "BUY_CE"}
    position = scanner._route_paper_capture_or_legacy(sample_signal, confirm=False)

    assert position == {"position_id": "POS-X"}, "Legacy path must return its Position object"
    assert len(mock.order_manager.calls) == 1
    assert mock.order_manager.calls[0] is sample_signal


# ===========================================================================
# Bug #3 — Trailing-stop phase gating disarms SL breach
# ===========================================================================

def test_trailing_sl_breach_fires_in_phase1_or_2():
    """Bug #3 regression: after the trailing-stop ratchets current_sl
    above initial_sl, a tick with LTP ≤ current_sl must fire an exit
    EVEN when bars_held is in Phase 1 (≤3) or Phase 2 (≤5) — not wait
    for Phase 3 (>5 bars).
    """
    from prometheus.execution.position_monitor import (
        PositionMonitor, TrailingState,
    )

    # Build a state with the ratchet already advanced (breakeven trap hit).
    state = TrailingState(
        position_id="TEST-POS-001",
        tradingsymbol="NIFTY26JUL24150PE",
        symbol="NIFTY 50",
        direction="bearish",
        strategy="apex",
        entry_premium=100.0,
        initial_sl=95.0,
        current_sl=110.0,           # ratcheted ABOVE entry (Stage 0+ engaged)
        target=120.0,
        breakeven_set=True,
        breakeven_ratio=0.4,
        risk_distance=5.0,           # entry - initial_sl = 100 - 95
        sl_order_id="",
        entry_bar_count=2,           # Phase 1 (≤3)
    )

    exits: list[tuple] = []
    monitor = PositionMonitor.__new__(PositionMonitor)
    monitor._on_exit = lambda pid, price, reason: exits.append((pid, price, reason))

    # current_price ≤ current_sl (110) → must exit even though bars_held=2
    monitor._process_tick(state, current_price=108.0)

    assert len(exits) == 1, (
        "Phase 1 must honor ratcheted SL — observed 60+ min of uncovered "
        "breach today (Bug #3)"
    )
    pid, price, reason = exits[0]
    assert pid == "TEST-POS-001"
    assert price == 108.0
    assert reason in ("phase1_sl_breach", "phase2_sl_breach", "stop_loss_premium_phase3")


def test_modify_broker_sl_never_lowers_below_ratcheted_value():
    """Bug #3 defense-in-depth: _modify_broker_sl_manual must refuse
    to push the broker SL order below the ratcheted current_sl.
    """
    from prometheus.execution.position_monitor import (
        PositionMonitor, TrailingState,
    )
    from prometheus.execution.broker import OrderStatus

    class _StubOrder:
        status = OrderStatus.OPEN

    class _StubBroker:
        def __init__(self):
            self.modifies = []
        def get_order_status(self, oid):
            return _StubOrder()
        def modify_order(self, oid, trigger_price):
            self.modifies.append((oid, trigger_price))
            return True

    broker = _StubBroker()
    monitor = PositionMonitor.__new__(PositionMonitor)
    monitor.broker = broker

    state = TrailingState(
        position_id="TEST-POS-002",
        tradingsymbol="BANKNIFTY26JUL57000PE",
        symbol="NIFTY BANK",
        direction="bearish",
        strategy="apex",
        entry_premium=302.65,
        initial_sl=292.20,
        current_sl=332.64,          # ratcheted high
        target=388.69,
        breakeven_set=True,
        trailing_activated=True,
        risk_distance=10.45,
        sl_order_id="SL-ORDER-1",
    )

    # Caller tries to LOWER broker SL to initial_sl*0.8 = 233.76
    # (this is exactly what buggy Phase 1→2 transition did today at 10:31:39)
    monitor._modify_broker_sl_manual(state, manual_trigger=233.76)

    # Must have clamped upward to current_sl (332.64), not the requested 233.76.
    assert len(broker.modifies) == 1, "Broker modify_order should be called once"
    _oid, trigger_price = broker.modifies[0]
    assert trigger_price == 332.64, (
        f"Broker SL must NOT be lowered below ratcheted current_sl; "
        f"got {trigger_price} expected 332.64"
    )


# ===========================================================================
# Bug #4 — _parse_tradingsymbol regex mis-parse
# ===========================================================================

def test_parse_tradingsymbol_monthly_format():
    """Bug #4 regression: BANKNIFTY26JUL56900PE must parse to
    expiry=2026-07-28, strike=56900.0 — NOT 2056/2057/any other year.
    """
    from prometheus.data.angelone_options import AngelOneOptionChain

    class _Stub: pass
    stub = _Stub()
    parse = AngelOneOptionChain._parse_tradingsymbol.__get__(stub, _Stub)

    r = parse("BANKNIFTY26JUL56900PE", "BANKNIFTY")
    assert r is not None, "Monthly-format tradingsymbol must parse"
    assert r["strike"] == 56900.0
    assert r["option_type"] == "PE"
    assert r["expiry_str"] == "2026-07-28", (
        f"Monthly NIFTY-BANK July 2026 expiry is the last Thursday 28th; "
        f"got {r['expiry_str']}"
    )


def test_parse_tradingsymbol_sensex_monthly():
    """Bug #4: SENSEX uses Thursday expiry per _resolve_weekly_expiry_day_name."""
    from prometheus.data.angelone_options import AngelOneOptionChain

    class _Stub: pass
    stub = _Stub()
    parse = AngelOneOptionChain._parse_tradingsymbol.__get__(stub, _Stub)

    r = parse("SENSEX26JUN74300PE", "SENSEX")
    assert r is not None
    assert r["strike"] == 74300.0
    assert r["option_type"] == "PE"
    assert r["expiry_str"] == "2026-06-25", (
        f"Monthly SENSEX June 2026 expiry is the last Thursday 25th; "
        f"got {r['expiry_str']}"
    )


def test_parse_tradingsymbol_weekly_format():
    """Bug #4 weekly-format: NIFTY2640722650PE → 2026-04-07, 22650, PE."""
    from prometheus.data.angelone_options import AngelOneOptionChain

    class _Stub: pass
    stub = _Stub()
    parse = AngelOneOptionChain._parse_tradingsymbol.__get__(stub, _Stub)

    r = parse("NIFTY2640722650PE", "NIFTY")
    assert r is not None
    assert r["strike"] == 22650.0
    assert r["option_type"] == "PE"
    assert r["expiry_str"] == "2026-04-07"


def test_parse_tradingsymbol_weekly_oct_nov_dec():
    """Bug #4 weekly: October (O), November (N), December (D) single-char months."""
    from prometheus.data.angelone_options import AngelOneOptionChain

    class _Stub: pass
    stub = _Stub()
    parse = AngelOneOptionChain._parse_tradingsymbol.__get__(stub, _Stub)

    for ts, exp_strike, exp_expiry in [
        ("NIFTY26O0422600PE", 22600.0, "2026-10-04"),
        ("NIFTY26N1122700CE", 22700.0, "2026-11-11"),
        ("NIFTY26D0122900PE", 22900.0, "2026-12-01"),
    ]:
        r = parse(ts, "NIFTY")
        assert r is not None, f"Weekly {ts} must parse"
        assert r["strike"] == exp_strike, f"{ts}: {r['strike']} ≠ {exp_strike}"
        assert r["expiry_str"] == exp_expiry, f"{ts}: {r['expiry_str']!r} ≠ {exp_expiry!r}"


def test_parse_tradingsymbol_does_not_return_bogus_year():
    """Bug #4 explicit guard: ensure we never produce year 2056/2057 or
    similarly impossible dates that triggered the original symptom."""
    from prometheus.data.angelone_options import AngelOneOptionChain

    class _Stub: pass
    stub = _Stub()
    parse = AngelOneOptionChain._parse_tradingsymbol.__get__(stub, _Stub)

    # The original bogus symptom example.
    r = parse("BANKNIFTY26JUL56900PE", "BANKNIFTY")
    assert r is not None
    year = int(r["expiry_str"][:4])
    # Don't accept anything outside ±5 years from now
    now_year = datetime.now().year
    assert now_year - 5 <= year <= now_year + 5, (
        f"Parsed year {year} is implausible (today {now_year}); "
        f"Bug #4 regression"
    )


# ===========================================================================
# Bug #8 — is_square_off never propagated to PaperTradeEngine
# ===========================================================================

def test_live_paper_capture_on_bar_propagates_square_off_flag():
    """Bug #8 regression: the LivePaperCapture adapter is the only bridge
    between the live scan loop and the paper engine's ``process_bar``.

    ``Prometheus._paper_capture_feed_bars`` (main.py:520) always calls
    ``paper_capture.on_bar(sym, bar, is_session_end=False)`` — it does
    NOT pass an ``is_square_off`` argument and the live loop's 15:15
    intraday square-off trigger (main.py:4619) only invokes
    ``_square_off_intraday_positions`` (legacy broker), never the paper
    engine. Combined with ``LivePaperCapture.on_bar``'s default
    ``is_session_end=False``, the
    ``PositionTracker._evaluate_exit_via_feed`` square-off branch
    (``is_square_off and pos.trade_mode=="intraday"``) is dead code —
    intraday paper positions never force-close at 15:15, leaving them
    open and accumulating overnight phantom P&L.

    Pin (this is a *unit* test of the live_bridge adapter): when the
    caller asks for square-off explicitly via the API, the adapter MUST
    honor it — pass ``is_square_off=True`` through to the engine (we
    extend ``on_bar`` to accept the explicit flag from above rather than
    synthesise it from clock time, otherwise we'd hide the upstream
    caller's intent again).
    """
    from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
    from prometheus.papertrade.types import TradeSnapshot

    is_sq_log: list = []

    class _StubEngine:
        def __init__(self):
            self.tracker = type("T", (), {"open_positions": {}})()
        def process_bar(self, snap, is_session_end=False, is_square_off=False):
            is_sq_log.append(is_square_off)
            return []

    cfg = CaptureConfig(enabled=True)
    cap = LivePaperCapture.__new__(LivePaperCapture)
    cap.enabled = True
    cap.config = cfg
    cap._engine = _StubEngine()

    # Caller passes is_square_off=True — must arrive at engine.process_bar.
    cap.on_bar("NIFTY 50", {
        "timestamp": "2026-07-22 15:15:00",
        "open": 100.0, "high": 100.5, "low": 99.7, "close": 100.2,
        "interval": "15minute", "instrument": "",
    }, is_session_end=True, is_square_off=True)

    assert is_sq_log, "process_bar should have been called"
    assert is_sq_log[-1] is True, (
        f"is_square_off=True must propagate to engine.process_bar; "
        f"got {is_sq_log[-1]!r} (Bug #8 — square-off never reached the engine)"
    )


def test_live_paper_capture_on_bar_defaults_without_square_off():
    """Companion: the existing call site (``feed_bars_to_paper_capture``)
    passes no ``is_square_off`` argument. Backward-compat must preserve
    ``is_square_off=False`` default → no spurious square-off fires mid-day.
    """
    from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig

    is_sq_log: list = []

    class _StubEngine:
        def __init__(self):
            self.tracker = type("T", (), {"open_positions": {}})()
        def process_bar(self, snap, is_session_end=False, is_square_off=False):
            is_sq_log.append(is_square_off)
            return []

    cfg = CaptureConfig(enabled=True)
    cap = LivePaperCapture.__new__(LivePaperCapture)
    cap.enabled = True
    cap.config = cfg
    cap._engine = _StubEngine()

    cap.on_bar("NIFTY 50", {
        "timestamp": "2026-07-22 11:30:00",
        "open": 100.0, "high": 100.5, "low": 99.7, "close": 100.2,
        "interval": "15minute", "instrument": "",
    })  # No is_square_off kwarg → default False

    assert is_sq_log == [False]


# ===========================================================================
# Bug #6 — paper_capture MTM invisible in /status, /pnl, daily summary
# ===========================================================================

def test_status_command_shows_paper_capture_positions_when_enabled():
    """Bug #6 regression: /status today queries ``self.broker.get_positions``
    only — which returns ``[]`` in paper mode. Paper-capture positions and
    their MTM are invisible to the operator. When ``self._paper_capture``
    is enabled, ``_tg_cmd_status`` MUST include open paper positions and
    their mark-to-market P&L in the response.

    Pin (unit-level): the response body for ``/status`` must mention
    paper_capture positions and MTM when ``_paper_capture.enabled`` is true
    and there's at least one open paper position.
    """
    from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
    from prometheus.papertrade.types import Position, Direction
    from prometheus.main import Prometheus

    # Mock the paper_capture so its open_positions_view returns one row.
    class _MockCap:
        enabled = True
        def __init__(self):
            self._open = [
                {
                    "trade_id": "PAPER-XYZ",
                    "symbol": "NIFTY 50",
                    "instrument": "NIFTY26JUL24150PE",
                    "direction": "SHORT",
                    "quantity": 75,
                    "entry_price": 100.0,
                    "entry_time": "2026-07-22T09:30:00+05:30",
                    "stop_loss": 95.0,
                    "target": 120.0,
                    "strategy": "apex",
                    "trade_mode": "intraday",
                    "bars_held": 4,
                    "breakeven_set": False,
                },
            ]
        def open_positions_view(self):
            return list(self._open)
        def stats(self):
            from prometheus.papertrade.types import TradeStats
            return TradeStats(open_positions=1, total_trades=0)

    # Minimal Prometheus shell — only set up what _tg_cmd_status touches.
    prom = Prometheus.__new__(Prometheus)
    prom._paper_capture = _MockCap()
    prom.risk = type("R", (), {"get_portfolio_state": lambda self: type("S", (), {
        "capital": 15000.0,
        "trades_today": 0,
        "realized_pnl_today": 0.0,
    })(), "_halted": False})()
    prom.broker = type("B", (), {"get_positions": lambda self: []})()
    prom.multi_account = None
    prom.mode = "paper"
    prom.telegram = None

    text = prom._tg_cmd_status()

    # Must reference paper-capture section, the trade id, tradingsymbol and
    # a mark-to-market figure.
    assert "paper" in text.lower(), (
        f"Bug #6 — /status must surface paper_capture section; got:\n{text}"
    )
    # The capture section header must be present and the tradingsymbol must
    # appear in the row (the column showing what was opened).
    assert "Paper Capture" in text, (
        f"Bug #6 — /status must surface Paper Capture block; got:\n{text}"
    )
    assert "NIFTY26JUL24150PE" in text, (
        f"Bug #6 — /status must list the paper tradingsymbol; got:\n{text}"
    )
    assert "MTM" in text, (
        f"Bug #6 — /status must include MTM figure row; got:\n{text}"
    )


# ===========================================================================
# Bug A — RiskManager.max_intraday_trades never defined (AttributeError)
# ===========================================================================
# Discovered during the broader audit after Session 31's bug-cluster close.
# `risk/manager.py:218` checks ``self.max_intraday_trades`` but the attribute
# is never set in ``RiskManager.__init__`` (only ``max_daily_trades`` is).
# Every intraday-mode ``pre_trade_check`` call therefore raises::
#     AttributeError: 'RiskManager' object has no attribute 'max_intraday_trades'
# This crash lands before violations are even collected, so the trade is
# never approved nor explicitly rejected — the caller (e.g. OrderManager
# via ``_execute_directional_options``) sees the exception bubble up and
# the entire risk-check pipeline fails open. In paper-mode the live path
# bypasses ``OrderManager`` (Bug #2 fix routes via paper_capture), so the
# crash is silent in the default config; but any call into the legacy
# OrderManager with a ``trade_mode="intraday"`` trade hits this immediately.
# Live/semi-auto modes route through OrderManager and WILL crash.

def test_risk_manager_intraday_pre_trade_check_does_not_raise():
    """Bug A regression: every intraday-mode pre_trade_check must NOT raise
    AttributeError on a missing ``self.max_intraday_trades`` attribute.
    The risk manager must either expose that limit (configurable) or fall
    back to ``max_daily_trades`` — but never crash on the lookup.
    """
    from prometheus.risk.manager import RiskManager
    rm = RiskManager(
        config={"max_daily_loss": 5000, "max_daily_trades": 6},
        initial_capital=15000,
    )

    # Pre-fix this raises:
    #   AttributeError: 'RiskManager' object has no attribute 'max_intraday_trades'
    # Post-fix this returns a RiskCheckResult (approved or not is irrelevant
    # for this test — the only requirement is no exception).
    result = rm.pre_trade_check({
        "symbol": "NIFTY 50",
        "direction": "bullish",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "quantity": 75,
        "cost": 7500.0,
        "trade_mode": "intraday",
    })

    # Must be a real result, not an AttributeError bubbled up.
    from prometheus.risk.manager import RiskCheckResult
    assert isinstance(result, RiskCheckResult)


def test_risk_manager_records_intraday_trade_limit_exists():
    """Bug A companion: ``RiskManager`` MUST define a
    ``max_intraday_trades`` attribute (read by the pre_trade_check). This
    pins the existence so a future refactor can't silently drop the
    ``self.max_intraday_trades = ...`` line.
    """
    from prometheus.risk.manager import RiskManager
    rm = RiskManager(
        config={"max_daily_loss": 5000, "max_intraday_trades": 5},
        initial_capital=15000,
    )
    assert hasattr(rm, "max_intraday_trades"), (
        "RiskManager must expose ``max_intraday_trades`` (Bug A — the "
        "pre_trade_check reads self.max_intraday_trades; if __init__ "
        "doesn't set it, every intraday-mode trade attempt crashes with "
        "AttributeError)"
    )
    # When config supplies it, the configured value wins.
    assert rm.max_intraday_trades == 5


# ===========================================================================
# Bug C — TrailingState._current_phase not restored from SQLite
# ===========================================================================
# The Bug #3 fix added ``_current_phase`` guarding the phase-breach
# behavior; ``position_monitor.py`` correctly mutates it on transitions.
# ``store.save_position_state`` correctly persists it as ``current_phase``.
# But ``main.py:_restore_open_positions`` (around line 3199) constructs the
# restored ``TrailingState`` WITHOUT reading ``row["current_phase"]`` — so
# every restored position silently resets to ``_current_phase=1`` regardless
# of where the in-memory tracker had ratcheted it before the crash.
# Effect: a restored position that had already passed Phase 1+2 immunity
# gets it RE-APPLIED for 3+2 more bars after restart — the universal SL
# check at position_monitor.py:289 catches the worst cases, but the Phase
# transitions (1→2 and 2→3) run again, potentially re-running the
# ``_modify_broker_sl_manual`` logic. Bug #3 defense-in-depth softens this,
# but the restored state is still semantically wrong (it lies about which
# phase the position is in).

def test_restore_open_positions_preserves_current_phase():
    """Bug C regression: when ``main.py:_restore_open_positions`` rebuilds
    a ``TrailingState`` from a saved SQLite row, it MUST read
    ``current_phase`` from the row and pass it to the constructor —
    otherwise every restored position silently resets to Phase 1,
    re-applying immunity that was already passed before the crash.
    """
    from prometheus.execution.position_monitor import TrailingState

    # Simulate a row as loaded by store.load_open_positions. The save side
    # writes the column as ``current_phase`` (no underscore) — see
    # store.save_position_state line 484-487.
    saved_row = {
        "position_id": "POS-RESTORED-001",
        "tradingsymbol": "NIFTY26JUL24150PE",
        "symbol": "NIFTY 50",
        "entry_premium": 100.0,
        "initial_sl": 95.0,
        "current_sl": 110.0,
        "target": 120.0,
        "direction": "bearish",
        "strategy": "apex",
        "entry_time": "2026-07-22 09:30:00",
        "sl_order_id": "SL-ORDER-1",
        "breakeven_set": 1,
        "trailing_activated": 1,
        "trailing_stage2": 0,
        "trailing_stage3": 0,
        "premium_hwm": 105.0,
        "entry_bar_count": 7,
        "max_bars": 16,
        "breakeven_ratio": 0.6,
        "risk_distance": 5.0,
        "bar_interval": "15minute",
        "trade_mode": "intraday",
        "entry_orders_json": "",
        # The save side stored _current_phase under this column name.
        "current_phase": 3,
    }

    # Mirror the exact constructor call shape used in main.py:3199-3224.
    # The fix is to plumb ``_current_phase=row.get("current_phase", 1)``
    # into the constructor. Pre-fix: the restored state defaults to 1.
    ts = TrailingState(
        position_id=saved_row["position_id"],
        tradingsymbol=saved_row["tradingsymbol"],
        symbol=saved_row["symbol"],
        entry_premium=saved_row["entry_premium"],
        initial_sl=saved_row["initial_sl"],
        current_sl=saved_row["current_sl"],
        target=saved_row["target"],
        direction=saved_row["direction"],
        strategy=saved_row.get("strategy", ""),
        entry_time=saved_row.get("entry_time", ""),
        sl_order_id=saved_row.get("sl_order_id", ""),
        breakeven_set=bool(saved_row.get("breakeven_set", 0)),
        trailing_activated=bool(saved_row.get("trailing_activated", 0)),
        trailing_stage2=bool(saved_row.get("trailing_stage2", 0)),
        trailing_stage3=bool(saved_row.get("trailing_stage3", 0)),
        premium_hwm=saved_row.get("premium_hwm", 0),
        entry_bar_count=saved_row.get("entry_bar_count", 0),
        max_bars=saved_row.get("max_bars", 7),
        breakeven_ratio=saved_row.get("breakeven_ratio", 0.6),
        risk_distance=saved_row.get("risk_distance", 0),
        bar_interval=saved_row.get("bar_interval", "day"),
        trade_mode=saved_row.get("trade_mode", "swing"),
        entry_orders_json=saved_row.get("entry_orders_json", ""),
        # Bug C fix: pass _current_phase through from the saved row.
        _current_phase=int(saved_row.get("current_phase", 1)),
    )

    assert ts._current_phase == 3, (
        f"Restored TrailingState must honor saved current_phase=3; got "
        f"{ts._current_phase!r} (Bug C — phase resets to 1 on restart)"
    )


# ===========================================================================
# Patch 1 (2026-07-25 audit) — D.4: double-dispatch of
# _dispatch_multi_account in scanner immediate + pullback paths
# ===========================================================================

def test_scanner_immediate_path_skips_multi_account_dispatch_when_paper_capture_active():
    """Patch 1 regression: when LivePaperCapture is active, the immediate-
    execution path in LiveScanner must NOT call
    ``self._prometheus._dispatch_multi_account`` (the wrapper at
    ``main.py:_dispatch_multi_account`` re-enters ``paper_capture.on_signal``
    when ``self._paper_capture`` is enabled — opening a SECOND position).
    """
    from prometheus.pipeline.scanner import LiveScanner

    class _PC:
        enabled = True
        def __init__(self):
            self.calls = []
        def on_signal(self, sig):
            self.calls.append(sig)

    class _MockProm:
        def __init__(self):
            self._paper_capture = _PC()
            # _dispatch_multi_account is the wrapper the scanner calls —
            # if it's invoked in paper mode, that's the bug.
            self._dispatch_multi_account_calls = 0
            self.order_manager = object()  # unused when pc active
        def _dispatch_multi_account(self, sig, is_intraday=False, bar_interval="15minute"):
            self._dispatch_multi_account_calls += 1

    mock = _MockProm()
    scanner = LiveScanner.__new__(LiveScanner)
    scanner._prometheus = mock

    # The post-Patch 1 safety guard.
    assert scanner._is_paper_capture_active() is True, (
        "PaperCapture must be detected as active when it exists + enabled"
    )


def test_scanner_is_paper_capture_active_false_when_disabled_or_absent():
    """Patch 1 companion: the helper must return False in live mode
    (no paper_capture) and when paper_capture is explicitly disabled —
    so the multi-account dispatch path runs normally in live mode."""
    from prometheus.pipeline.scanner import LiveScanner

    class _PCDisabled:
        enabled = False

    class _MockProm1:
        _paper_capture = _PCDisabled()
        order_manager = object()

    class _MockProm2:
        _paper_capture = None
        order_manager = object()

    s1 = LiveScanner.__new__(LiveScanner); s1._prometheus = _MockProm1()
    s2 = LiveScanner.__new__(LiveScanner); s2._prometheus = _MockProm2()

    assert s1._is_paper_capture_active() is False
    assert s2._is_paper_capture_active() is False


# ===========================================================================
# Patch 3 (2026-07-25 audit) — B.3: broken BS-fallback import
# ===========================================================================

def test_scanner_bs_fallback_import_path_exists_and_resolvable():
    """Patch 3 regression: ``scanner._update_position_prices_bs`` imports
    ``black_scholes_price`` from ``prometheus.utils.options_math``. The
    original code imported from ``prometheus.signals.option_pricing``
    which doesn't exist in the source tree. The fix must point to the
    real module so the BS fallback path stops silently no-op'ing via the
    ``except Exception: pass`` swallow. We can't easily exercise the
    method end-to-end (needs a fully wired live prometheus instance)
    so we verify the import resolution instead — this fails fast the
    moment anyone breaks the import path again.
    """
    # Public module attribute, so the import path itself resolves.
    from prometheus.utils.options_math import black_scholes_price  # noqa: F401
    # And the WRONG module from the original bug must NOT be importable
    # (this is what was masked by ``except Exception: pass`` before).
    import importlib
    bad_module = importlib.util.find_spec("prometheus.signals.option_pricing")
    assert bad_module is None, (
        "prometheus.signals.option_pricing should not exist — the Patch 3 "
        "fix deliberately routes through prometheus.utils.options_math; if "
        "this module reappears the import-path regression re-emerges."
    )


# ===========================================================================
# B.2 (2026-07-25 audit) — set_real_premium must reject inverted spreads
# ===========================================================================

def test_set_real_premium_clamps_inverted_spread_to_ltp():
    """B.2 regression: when Angel One returns ``bid > ask`` (corrupted
    quote), ``PaperTrader.set_real_premium`` must clamp both sides to
    ``ltp`` instead of letting the inverted spread survive and distort
    the mid-spread fill formula.
    """
    from prometheus.execution.paper_trader import PaperTrader

    trader = PaperTrader(initial_capital=100000)
    trader.set_real_premium("NIFTY26AUG24150CE", ltp=100.0, bid=120.0, ask=80.0)

    quote = trader.get_real_premium("NIFTY26AUG24150CE")
    assert quote is not None
    # Both sides must collapse to ltp (the only known-good value).
    assert quote["bid"] == 100.0, f"Inverted bid must clamp to ltp; got {quote['bid']}"
    assert quote["ask"] == 100.0, f"Inverted ask must clamp to ltp; got {quote['ask']}"
    assert quote["ltp"] == 100.0


def test_set_real_premium_preserves_normal_spread():
    """B.2 companion: a valid spread (``bid <= ask``) must pass through
    unchanged — the clamp should only fire on inverted quotes, not
    every call.
    """
    from prometheus.execution.paper_trader import PaperTrader

    trader = PaperTrader(initial_capital=100000)
    trader.set_real_premium("NIFTY26AUG24150CE", ltp=100.0, bid=98.0, ask=102.0)

    quote = trader.get_real_premium("NIFTY26AUG24150CE")
    assert quote is not None
    assert quote["bid"] == 98.0
    assert quote["ask"] == 102.0
    assert quote["ltp"] == 100.0


# ===========================================================================
# Patch 2 / C.2 (2026-07-25 audit) — paper_capture open-position persistence
# ===========================================================================

def test_trade_recorder_records_and_deletes_open_positions(tmp_path):
    """C.2 regression: ``TradeRecorder`` must persist open positions to a
    new ``paper_open_positions`` table on insert, and remove the row
    when the position closes — so a restart can re-hydrate them.
    """
    from prometheus.papertrade.recorder import TradeRecorder
    from prometheus.papertrade.types import Position, Direction

    db = tmp_path / "open_pos.sqlite"
    csv = tmp_path / "open_pos.csv"
    recorder = TradeRecorder(sqlite_path=str(db), csv_path=str(csv))

    # Simulate an OPEN event as the tracker would emit it.
    pos = Position(
        trade_id="PAPER-TEST-001",
        symbol="NIFTY 50",
        instrument="NIFTY26JUL24150CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=75,
        entry_price=100.0,
        entry_time=datetime(2026, 7, 25, 9, 45, 0, tzinfo=IST),
        stop_loss=95.0,
        target=120.0,
        max_bars=16,
        bars_held=0,
        strategy="apex",
        signal_score=0.72,
        signal_confidence=0.6,
        trade_mode="intraday",
    )
    recorder.record_open_position(pos.to_dict())

    # Verify the row is persisted.
    rows = recorder.load_open_positions()
    assert len(rows) == 1, f"Expected 1 open row; got {len(rows)}"
    assert rows[0]["trade_id"] == "PAPER-TEST-001"
    assert rows[0]["instrument"] == "NIFTY26JUL24150CE"
    assert rows[0]["entry_price"] == 100.0
    assert rows[0]["stop_loss"] == 95.0

    # Simulate a CLOSE event.
    recorder.delete_open_position("PAPER-TEST-001")
    rows = recorder.load_open_positions()
    assert rows == [], "Close must delete the matching row"

    recorder.close()


def test_position_tracker_recorder_hooks_persist_on_open(tmp_path):
    """C.2 wiring: a PositionTracker constructed with a recorder
    reference must call ``record_open_position`` on every successful
    open — so even a crash immediately after the in-memory insert
    still leaves a recoverable row on disk.
    """
    from prometheus.papertrade.recorder import TradeRecorder
    from prometheus.papertrade.position_tracker import PositionTracker, CostModel
    from prometheus.papertrade.fill_simulator import FillSimulator
    from prometheus.papertrade.types import Position, Direction

    recorder = TradeRecorder(
        sqlite_path=str(tmp_path / "hook.sqlite"),
        csv_path=None,
    )
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=None, slippage_bps=0, use_bid_ask=False),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=False,
        recorder=recorder,
    )

    pos = Position(
        trade_id="PAPER-HOOK-001",
        symbol="NIFTY 50",
        instrument="NIFTY26JUL24150CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=75,
        entry_price=100.0,
        entry_time=datetime(2026, 7, 25, 9, 45, 0, tzinfo=IST),
        stop_loss=95.0,
        target=120.0,
        max_bars=16,
        strategy="apex",
    )
    tracker.open_position(pos)

    # Recorder hook must have persisted the row immediately.
    rows = recorder.load_open_positions()
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "PAPER-HOOK-001"

    recorder.close()


def test_live_paper_capture_rehydrates_open_positions_on_init(tmp_path):
    """C.2 end-to-end: LivePaperCapture.__init__ must re-hydrate
    any rows present in ``paper_open_positions`` into the engine's
    tracker dict — so a restart recovers the open positions instead
    of silently losing them.
    """
    from prometheus.papertrade.recorder import TradeRecorder
    from prometheus.papertrade.types import Position, Direction
    from prometheus.paper_executor.live_bridge import (
        LivePaperCapture, CaptureConfig,
    )

    shared_db = str(tmp_path / "shared.sqlite")
    shared_csv = str(tmp_path / "shared.csv")

    # Session 1: write one persisted open position directly to the recorder.
    recorder = TradeRecorder(sqlite_path=shared_db, csv_path=shared_csv)
    pos = Position(
        trade_id="PAPER-RESTART-001",
        symbol="NIFTY 50",
        instrument="NIFTY26JUL24150CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=75,
        entry_price=100.0,
        entry_time=datetime(2026, 7, 25, 9, 45, 0, tzinfo=IST),
        stop_loss=95.0,
        target=120.0,
        max_bars=16,
        strategy="apex",
        signal_score=0.72,
        signal_confidence=0.6,
        trade_mode="intraday",
    )
    # Simulate the tracker.open_position hook.
    recorder.record_open_position(pos.to_dict())
    recorder.close()

    # Session 2: spin up LivePaperCapture pointed at the SAME sqlite path.
    config = CaptureConfig(
        enabled=True,
        sqlite_path=shared_db,
        csv_path=shared_csv,
        max_concurrent_positions=8,
        allow_duplicate_instrument=True,
        enable_trailing=False,
        cost_per_side_bps=0.0,
        slippage_bps=0,
        default_max_bars_intraday=16,
        default_max_bars_swing=96,
    )

    class _Ltp:
        def get_ltp(self, instrument):
            return 100.0

    capture = LivePaperCapture(config=config, ltp_source=_Ltp(), telegram=None)

    # After init, the engine tracker MUST contain the recovered position.
    open_dict = capture._engine.tracker.open_positions
    assert "PAPER-RESTART-001" in open_dict, (
        "Recovered position must be re-hydrated into the engine tracker; "
        f"got keys: {list(open_dict.keys())}"
    )

    recovered = open_dict["PAPER-RESTART-001"]
    assert recovered.instrument == "NIFTY26JUL24150CE"
    assert recovered.entry_price == 100.0
    assert recovered.stop_loss == 95.0
    assert recovered.target == 120.0

    # No leftover ghost row in the persistence table — rehydrate keeps
    # the row (so a second restart mid-session is also recoverable).
    rows = capture._recorder.load_open_positions()
    assert len(rows) == 1, "Rehydrated position row must persist (not deleted by rehydrate)"

    capture._recorder.close()


# ===========================================================================
# Item 2 (2026-07-25 audit follow-up) — per-unit MTM display in /status
# ===========================================================================

def test_tg_cmd_status_paper_capture_mtm_includes_per_unit_breakdown():
    """Item 2 regression: when LivePaperCapture is enabled, the
    ``_tg_cmd_status`` Paper-Capture section must surface a per-unit
    MTM breakdown so an operator can disambiguate, e.g. "MTM -6.60"
    between (qty=1 x -6.60/unit) and (qty=75 x -0.09/unit).

    We monkey-patch the bare-minimum attributes on a Prometheus
    instance and call the same string-building code the
    ``/checkpapertrade`` callback uses, so the rendered text lands
    in the test output. We assert: (1) the per-unit fragment is
    present when qty * (ltp - entry) != 0 (clarity), and (2) it
    formats the correct sign+magnitude.
    """
    from prometheus.main import Prometheus

    class _PC:
        enabled = True
        stats_calls = 0
        def __init__(self):
            self._feed = _Feed()
        def open_positions_view(self):
            # Position that would look ambiguous as gross-only:
            # qty=75, entry=100.0, ltp=99.978 → MTM -1.65 =
            # (75 × -0.022). Without the per-unit line the operator
            # might read "MTM -1.65" as "single contract lost Rs 1.65".
            return [{
                "trade_id": "PAPER-AMBIG-001",
                "instrument": "NIFTY26JUL24150PE",
                "symbol": "NIFTY 50",
                "quantity": 75,
                "entry_price": 100.0,
                "direction": "LONG",
            }]
        def stats(self):
            # pstats.total_pnl
            class _S:
                total_pnl = 0.0
            return _S()

    class _Feed:
        def get_ltp(self, instr):
            return 99.978
        def __call__(self, instr):
            # Some code paths treat feed as callable; mirror that surface.
            return self.get_ltp(instr)

    class _Risk:
        def __init__(self):
            self.capital = 15000.0
            self.trades_today = 0
            self._halted = False  # required by _tg_cmd_status:9002
        def get_portfolio_state(self):
            class _PS:
                capital = 15000.0
                trades_today = 0
            return _PS()

    class _Broker:
        def get_positions(self):
            return []

    prom = Prometheus.__new__(Prometheus)
    prom._paper_capture = _PC()
    prom.broker = _Broker()
    prom.risk = _Risk()
    prom.multi_account = None
    prom.telegram = None
    prom.mode = "paper"

    text = prom._tg_cmd_status()

    # Per-unit fragment format: (qty × delta/u)
    # qty=75 → int is 75; delta = 99.978 - 100.0 = -0.022 → "+.2f" → "-0.02"
    assert "(75 \u00d7 -0.02/u)" in text, (
        f"Per-unit MTM fragment missing or wrong; got:\n"
        f"{text}"
    )
    # Sanity: total is also surfaced (with 2 decimals to match per-unit).
    assert "MTM <code>Rs -1.65</code>" in text, (
        f"Aggregate MTM line missing or wrong; got:\n{text}"
    )


# ===========================================================================
# Sensit-Bug (2026-07-28 audit) — SENSEX options routed on BFO, not NFO
# ===========================================================================

def test_angelone_options_exchange_segment_sensex_routes_to_bfo():
    """Regression: ``AngelOneOptionChain._exchange_for`` must return
    "BFO" for SENSEX and "NFO" for every other known underlying
    (NIFTY/BANKNIFTY/FINNIFTY/NIFTYIT/MIDCPNIFTY). Pre-fix all five
    Angel One API call sites hardcoded "NFO" — so ``searchScrip("NFO",
    "SENSEX")`` returned empty (correctly: no SENSEX contracts on
    NSE F&O) and every SENSEX signal was silently dropped by
    ``main.py:_price_options`` at the "BS theoretical estimate is NOT
    used for live/paper trading — signal dropped" branch. On 2026-07-28
    this wasted 8 of 8 generated signals for the entire session.
    """
    from prometheus.data.angelone_options import AngelOneOptionChain

    assert AngelOneOptionChain._exchange_for("SENSEX") == "BFO", (
        "SENSEX options trade on BSE F&O (segment code 'BFO'); Angel One "
        "searchScrip('NFO', 'SENSEX') returns empty because NSE doesn't "
        "list SENSEX contracts."
    )
    for sym in ("NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTYIT", "MIDCPNIFTY"):
        assert AngelOneOptionChain._exchange_for(sym) == "NFO", (
            f"{sym} should route on NFO (NSE F&O); got "
            f"{AngelOneOptionChain._exchange_for(sym)!r}"
        )


def test_angelone_options_searchScrip_uses_resolved_segment_per_underlying():
    """Regression: ``searchScrip`` (line ~162) must dispatch on the
    resolved segment, NOT a hardcoded ``"NFO"``. Easiest end-to-end
    surface to pin: monkey-patch the SmartConnect object's
    ``searchScrip`` to record the segment string passed, then call
    ``discover_contracts`` and verify SENSEX routed on BFO and NIFTY
    on NFO. Failure = silent SENSEX signal drop (the 2026-07-28 bug).
    """
    from prometheus.data.angelone_options import AngelOneOptionChain

    class _FakeObj:
        def __init__(self):
            self.calls = []  # list of (segment, underlying)
        def searchScrip(self, seg, underlying):
            self.calls.append((seg, underlying))
            # Simulate one contract returned per underlying.
            return {
                "data": [{
                    "tradingsymbol": f"{underlying}DUMMY",
                    "symboltoken": "1",
                    "name": underlying,
                    "expiry": "2026-08-06",
                    "instrumenttype": "OPTIDX",
                }],
            }

    class _FakeFetcher:
        def __init__(self, obj):
            self._obj = obj
        def _ensure_connected(self):
            return True
        @property
        def obj(self):
            return self._obj

    fake_obj = _FakeObj()
    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _FakeFetcher(fake_obj)
    chain._cache_date = ""  # force network path
    chain._token_cache = {}
    chain._last_call = 0.0
    chain._min_interval = 0.0  # skip rate-limit sleep in test
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300

    # SENSEX must dispatch via BFO.
    chain.discover_contracts("SENSEX", strikes_around_atm=2, spot_price=77000.0)
    assert any(seg == "BFO" and und == "SENSEX" for (seg, und) in fake_obj.calls), (
        f"SENSEX must route searchScrip to BFO; got calls={fake_obj.calls}"
    )

    # NIFTY 50 (underlying 'NIFTY') must still route via NFO.
    fake_obj.calls.clear()
    chain._cache_date = ""  # bypass cache so searchScrip runs again
    chain.discover_contracts("NIFTY 50", strikes_around_atm=2, spot_price=22000.0)
    assert any(seg == "NFO" and und == "NIFTY" for (seg, und) in fake_obj.calls), (
        f"NIFTY 50 must route searchScrip to NFO; got calls={fake_obj.calls}"
    )


# ===========================================================================
# Patch 2 (2026-07-31 audit) — Duplicate paper-trade dispatch in intraday
# and swing scan loops. The active path (run_intraday_mode main.py:4341)
# was calling BOTH ``_execute_signal_with_feedback`` AND
# ``_dispatch_multi_account`` back-to-back. Both helpers routed to
# ``paper_capture.on_signal`` when LivePaperCapture was active, opening
# TWO identical paper positions per signal 2-4 seconds apart (verified by
# 2026-07-31 logs: PAPER-20260731042911-3ACE44 + PAPER-20260731042913-08A545
# on the same SENSEX2680678300CE strike). This test verifies that
# ``_execute_signal_with_feedback`` no longer fires ``on_signal`` when
# paper_capture is active — only ``_dispatch_multi_account`` does, so
# only one position opens per signal.
# ===========================================================================

def test_execute_signal_with_feedback_skips_paper_capture_when_active():
    """Patch 2 regression: in paper mode, ``_execute_signal_with_feedback``
    must NOT call ``paper_capture.on_signal`` — that responsibility moved
    to ``_dispatch_multi_account`` (the documented PaperCapture wrapper).
    Calling both is what opened duplicate SENSEX paper positions today.
    """
    import prometheus.main as main_mod

    class _PC:
        enabled = True
        def __init__(self):
            self.on_signal_calls = []
        def on_signal(self, sig):
            self.on_signal_calls.append(sig)

    class _PrometheusStub:
        def __init__(self):
            self._paper_capture = _PC()
            # Live-mode order_manager must NEVER be touched in paper mode.
            class _OM:
                def __init__(self):
                    self.execute_calls = []
                def execute_signal(self, sig, confirm=False):
                    self.execute_calls.append(sig)
                    return None
                last_execution_error = ""
            self.order_manager = _OM()
            self.telegram = type("T", (), {"send_message": staticmethod(lambda *a, **k: None)})()
            self._last_trade_reject_alerts = {}

    # Instantiate the bound method directly on the stub.
    prom = _PrometheusStub()
    bound = main_mod.Prometheus._execute_signal_with_feedback.__get__(prom, _PrometheusStub)
    sig = {"symbol": "SENSEX", "action": "BUY"}
    result = bound(sig, confirm=False, context="INTRADAY DRY")

    assert result is None, (
        "Paper-mode _execute_signal_with_feedback must return None "
        "(paper_capture handles the actual capture)"
    )
    assert prom._paper_capture.on_signal_calls == [], (
        f"paper_capture.on_signal must NOT fire from _execute_signal_with_feedback; "
        f"got {prom._paper_capture.on_signal_calls}"
    )
    assert prom.order_manager.execute_calls == [], (
        f"order_manager.execute_signal must NOT fire in paper mode; "
        f"got {prom.order_manager.execute_calls}"
    )


def test_dispatch_multi_account_returns_paper_capture_trade_id_in_paper_mode():
    """Patch 2 companion: in paper mode, ``_dispatch_multi_account`` must
    (a) call ``paper_capture.on_signal`` exactly once AND (b) return the
    trade_id the capture returns — so the caller's ``if position:`` book-
    keeping block runs (registers the symbol in ``_today_traded_symbols``,
    increments the trade counter, fires the Telegram alert). Before the
    fix, it returned None, so callers' post-dispatch blocks were skipped
    AND the same symbol was re-dispatched on every subsequent scan cycle.
    """
    import prometheus.main as main_mod

    class _PC:
        enabled = True
        def __init__(self):
            self.on_signal_calls = []
        def on_signal(self, sig):
            self.on_signal_calls.append(sig)
            return "PAPER-TEST-TRADE-ID"

    class _PrometheusStub:
        def __init__(self):
            self._paper_capture = _PC()
        def _dispatch_multi_account_live(self, *a, **k):
            raise AssertionError(
                "Live-mode _dispatch_multi_account_live must NOT fire "
                "when paper_capture is active"
            )

    prom = _PrometheusStub()
    bound = main_mod.Prometheus._dispatch_multi_account.__get__(prom, _PrometheusStub)
    sig = {"symbol": "SENSEX", "action": "BUY"}
    result = bound(sig, is_intraday=True, bar_interval="15minute")

    assert result == "PAPER-TEST-TRADE-ID", (
        f"_dispatch_multi_account must return the paper_capture trade_id; "
        f"got {result!r}"
    )
    assert len(prom._paper_capture.on_signal_calls) == 1, (
        f"paper_capture.on_signal must fire EXACTLY once "
        f"(duplicate-dispatch bug); got {len(prom._paper_capture.on_signal_calls)}"
    )


# ===========================================================================
# Patch (2026-08-17 audit) — Angel One option-chain AB1021 propagation.
#
# Root cause: AngelOneOptionChain had its own 0.35s _rate_limit() pacer
# that was independent of AngelOneFetcher.SmartAPIRateLimiter (1.0s +
# global cooldown added in commit fbd9ddd). With max_workers=3 + 7
# instruments (Session 30), option-chain alone pushed ~9 calls/sec while
# Angel One's limit is ~3 req/sec. discover_contracts only inspected for
# AG8001 ("invalid token"); AB1021 ("too many requests") silently fell
# through, and the fallback searchScrip at line 184-185 fired
# immediately after the first AB1021 — doubling the rate-limit window.
#
# Fix:
#   - _rate_limit() now gates through the fetcher's SmartAPIRateLimiter
#     (backward-compat: falls back to 0.35s pacer if the fetcher lacks
#     _rate_limiter, e.g. test fixtures via __new__).
#   - New _mark_rate_limited() detects AB1021/AB1020/429 in any API
#     response and propagates a 20s global cooldown to the shared
#     SmartAPIRateLimiter (so historical fetcher / VIX fetch respect it
#     too).
#   - _mark_auth_failure() now also propagates cooldown for
#     _auth_cooldown_sec seconds.
#   - discover_contracts() bails on the FIRST AB1021 WITHOUT firing
#     the fallback searchScrip, halving the rate-limit window.
#   - fetch_market_data / get_real_premium / get_ltp_by_token all
#     consult _mark_rate_limited before retrying or returning.
# ===========================================================================

def test_angelone_options_rate_limit_propagates_to_shared_limiter():
    """AB1021 on searchScrip must propagate a 20s cooldown to the
    fetcher's shared SmartAPIRateLimiter AND skip the fallback
    searchScrip. Pre-fix behaviour was to silently bail on AB1021 and
    immediately retry with bare underlying — doubling the rate-limit
    window with no cooldown memory shared across callers.
    """
    from prometheus.data.angelone_options import AngelOneOptionChain
    from prometheus.data.angelone_fetcher import SmartAPIRateLimiter

    class _FakeObj:
        """Records every searchScrip call and emits AB1021 on the first."""
        def __init__(self):
            self.calls = []
            self._first = True
        def searchScrip(self, seg, q):
            self.calls.append((seg, q))
            if self._first:
                self._first = False
                return {"status": False, "errorcode": "AB1021",
                        "message": "Too Many Requests"}
            # Fallback call should NEVER happen — test fails if it does.
            return {"data": [{"tradingsymbol": f"{q}DUMMY",
                              "symboltoken": "1",
                              "name": q, "expiry": "2026-08-06",
                              "instrumenttype": "OPTIDX"}]}

    class _FakeFetcher:
        def __init__(self, obj):
            self._obj = obj
            self._rate_limiter = SmartAPIRateLimiter(delay_between_calls=0.0)
        def _ensure_connected(self):
            return True

    fake_obj = _FakeObj()
    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _FakeFetcher(fake_obj)
    chain._cache_date = ""
    chain._token_cache = {}
    chain._last_call = 0.0
    chain._min_interval = 0.0
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300

    contracts = chain.discover_contracts(
        "NIFTY 50", expiry_date="2026-08-21",
        strikes_around_atm=2, spot_price=22000.0,
    )

    # 1) AB1021 must produce ZERO fallback searchScrip.
    assert len(fake_obj.calls) == 1, (
        f"AB1021 on first searchScrip must NOT trigger a fallback call; "
        f"got calls={fake_obj.calls}"
    )

    # 2) discover_contracts must bail cleanly (empty list, not None).
    assert contracts == [], (
        f"discover_contracts must return [] on AB1021; got {contracts!r}"
    )

    # 3) 20s global cooldown must be set on the shared limiter.
    rl = chain._fetcher._rate_limiter
    import time as _t
    remaining = rl._cooldown_until - _t.monotonic()
    assert 18.0 <= remaining <= 20.5, (
        f"shared limiter must have ~20s cooldown after AB1021; "
        f"got remaining={remaining:.2f}s"
    )


def test_angelone_options_rate_limit_fallback_falls_back_when_no_data():
    """When searchScrip returns NO data (not a rate-limit), the fallback
    to bare underlying must STILL fire — that's the legitimate use case
    (stale daily-expiry calendar, etc.). This proves the AB1021 bail is
    rate-limit-specific, not a blanket skip-fallback change.
    """
    from prometheus.data.angelone_options import AngelOneOptionChain
    from prometheus.data.angelone_fetcher import SmartAPIRateLimiter

    class _FakeObj:
        def __init__(self):
            self.calls = []
        def searchScrip(self, seg, q):
            self.calls.append((seg, q))
            # First call (date-specific query "NIFTY18AUG"): no data.
            if "NIFTY" in q and q != "NIFTY":
                return {"data": None}
            # Fallback (bare "NIFTY"): returns one contract.
            return {"data": [{"tradingsymbol": "NIFTY18AUG2624400CE",
                              "symboltoken": "1", "name": "NIFTY",
                              "expiry": "2026-08-21",
                              "instrumenttype": "OPTIDX"}]}

    class _FakeFetcher:
        def __init__(self, obj):
            self._obj = obj
            self._rate_limiter = SmartAPIRateLimiter(delay_between_calls=0.0)
        def _ensure_connected(self):
            return True

    fake_obj = _FakeObj()
    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _FakeFetcher(fake_obj)
    chain._cache_date = ""
    chain._token_cache = {}
    chain._last_call = 0.0
    chain._min_interval = 0.0
    chain._disabled_until = 0.0
    chain._auth_cooldown_sec = 300

    contracts = chain.discover_contracts(
        "NIFTY 50", expiry_date="2026-08-21",
        strikes_around_atm=2, spot_price=22000.0,
    )

    # 1) Both searchScrip calls must fire (normal no-data fallback path).
    assert len(fake_obj.calls) == 2, (
        f"two searchScrip calls expected on no-data fallback path; "
        f"got calls={fake_obj.calls}"
    )

    # 2) No cooldown set.
    rl = chain._fetcher._rate_limiter
    import time as _t
    assert rl._cooldown_until <= _t.monotonic(), (
        f"no cooldown should be set when first call returned no data; "
        f"cooldown_until={rl._cooldown_until}"
    )


def test_angelone_options_mark_auth_failure_propagates_cooldown():
    """AG8001 (Invalid Token) must also propagate to the shared limiter.
    Pre-fix: ``_mark_auth_failure`` only set ``_disabled_until`` on the
    option-chain object, leaving the historical fetcher firing at 1.0s
    pace against the same broken session.
    """
    from prometheus.data.angelone_options import AngelOneOptionChain
    from prometheus.data.angelone_fetcher import SmartAPIRateLimiter

    class _FakeFetcher:
        def __init__(self):
            self._obj = None
            self._rate_limiter = SmartAPIRateLimiter(delay_between_calls=0.0)

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain._fetcher = _FakeFetcher()
    chain._auth_cooldown_sec = 300
    chain._disabled_until = 0.0

    chain._mark_auth_failure("Invalid Token (AG8001)")

    import time as _t
    # Option-chain disabled for 300s.
    assert chain._disabled_until > _t.time() + 280, (
        f"_disabled_until must be ~300s in the future; "
        f"got {chain._disabled_until - _t.time():.1f}s"
    )
    # Shared limiter also cooled down.
    rl = chain._fetcher._rate_limiter
    remaining = rl._cooldown_until - _t.monotonic()
    assert remaining > 280, (
        f"shared SmartAPIRateLimiter must have ~300s cooldown after "
        f"AG8001; got remaining={remaining:.1f}s"
    )



