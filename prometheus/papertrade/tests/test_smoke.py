"""
Smoke test for the paper trading subsystem.

Synthetic test with deterministic data:
* Mock feed supplies constant/linear walk prices.
* Signals are pre-seeded and fed in.
* Bars drive exits on schedule.
* Metrics at the end are validated.

Run:  python -m pytest prometheus/papertrade/tests/test_smoke.py -q
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pytest

from prometheus.papertrade import PaperTradeEngine
from prometheus.papertrade.types import (
    Direction, ExitReason, TradeSnapshot,
)
from prometheus.papertrade.signal_source import (
    SignalNotification, LiveReplaySource,
)
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.papertrade.recorder import TradeRecorder
from prometheus.utils.indian_market import IST


# ---------------------------------------------------------------------------
class MockFeed:
    """In-memory price feed — returns ``get_price(tradingsymbol, t)`` for
    each query, supporting a scripted price walk per instrument."""

    def __init__(self, prices: dict):
        # prices: {instrument: [(t, ltp, bid, ask), ...]}
        self.prices = prices
        self.cursor: dict = {}

    def _pick(self, instrument: str):
        rows = self.prices.get(instrument, [])
        if not rows:
            return None
        idx = self.cursor.get(instrument, 0)
        idx = min(idx, len(rows) - 1)
        self.cursor[instrument] = idx
        return rows[idx]

    def get_ltp(self, instrument: str) -> float:
        row = self._pick(instrument)
        return row[1] if row else 0.0

    def get_quote(self, instrument: str) -> Optional[tuple]:
        row = self._pick(instrument)
        if not row:
            return None
        _t, ltp, bid, ask = row
        if bid <= 0 or ask <= 0:
            return None
        return (ltp, bid, ask)


# ---------------------------------------------------------------------------
class MockSignalSource:
    """Yields pre-seeded signals in batches."""

    def __init__(self, signals):
        self.signals = list(signals)
        self._buffer = list(self.signals)
        self.signals = []  # consumed

    def next_batch(self):
        b = self._buffer
        self._buffer = []
        return b

    def close(self):
        pass


# ---------------------------------------------------------------------------
def make_signal(symbol, instrument, direction, entry, sl, target,
                strike, expiry="2026-07-17", trade_mode="intraday",
                strategy="test"):
    # Use "now - 5 min" so the entry timestamp is always recent (within the
    # 24h freshness guard added to PaperTradeEngine.process_new_signal on
    # 2026-07-21). Hard-coded historical dates (2026-07-17) would be rejected
    # as stale when the test runs later, breaking holding-duration math.
    _recent_bar_ts = datetime.now(IST) - timedelta(minutes=5)
    return SignalNotification(
        symbol=symbol,
        instrument=instrument,
        underlying=symbol.upper().replace(" ", "") if " " in symbol else symbol,
        direction=Direction.from_signal_direction(direction),
        strike=strike,
        option_type="CE" if direction == "bullish" else "PE",
        expiry=expiry,
        entry_price_hint=entry,
        stop_loss=sl,
        target=target,
        signal_score=0.6,
        signal_confidence=0.5,
        max_bars=16,
        trade_mode=trade_mode,
        strategy=strategy,
        # Bar timestamp 5 min before the exit bars below so durations are positive.
        bar_timestamp=_recent_bar_ts,
        metadata={},
    )


# ---------------------------------------------------------------------------
def build_engine(feed, signals, tmp_path):
    source = MockSignalSource(signals)
    recorder = TradeRecorder(
        sqlite_path=str(tmp_path / "test_paper.sqlite"),
        csv_path=str(tmp_path / "test_paper.csv"),
    )
    return PaperTradeEngine(
        feed=feed,
        signal_source=source,
        recorder=recorder,
        enable_trailing=False,    # keep test deterministic
        max_concurrent_positions=10,
        allow_duplicate_instrument=False,
    )


# ---------------------------------------------------------------------------
def test_target_hit_closes_position(tmp_path):
    """Long position: an uptrending candle triggers the target exit."""
    inst = "NIFTY2672124150CE"
    feed = MockFeed({
        inst: [
            # (time, ltp, bid, ask)
            ("2026-07-18T09:45:00", 175.00, 173.00, 177.00),
            ("2026-07-18T10:00:00", 175.00, 173.00, 177.00),
        ]
    })
    # Signal at 09:45 asking to BUY 174.96, SL 168.5, target 245 — too far.
    # But the bar will hit "target" only if the bar reaches it. Let's set
    # a target close enough to hit in 1 bar.
    sig = make_signal(
        symbol="NIFTY 50", instrument=inst, direction="bullish",
        entry=174.96, sl=168.50, target=185.00, strike=24150,
    )
    engine = build_engine(feed, [sig], tmp_path)

    # 1) Pull + open
    new_sigs = engine.gather_new_signals()
    assert len(new_sigs) == 1
    trade_id = engine.process_new_signal(new_sigs[0])
    assert trade_id is not None
    assert engine.open_positions_count() == 1

    # 2) Feed bar: opens 175, high 200, low 170, close 195 — target=185 inside high.
    bar = TradeSnapshot(
        timestamp=datetime(2026, 7, 18, 10, 0, 0, tzinfo=IST),
        symbol="NIFTY 50", instrument=inst,
        open=175.0, high=200.0, low=170.0, close=195.0,
        bar_interval="15minute",
    )
    closed = engine.process_bar(bar)
    assert len(closed) == 1, f"expected 1 close, got {len(closed)}"
    t = closed[0]
    assert t.exit_reason == ExitReason.TARGET
    assert t.exit_price == 185.00                # filled at target inside bar
    assert t.net_pnl > 0
    assert t.return_pct > 0
    assert engine.open_positions_count() == 0
    stats = engine.stats()
    assert stats.total_trades == 1
    assert stats.winning_trades == 1
    assert stats.win_rate == 100.0
    engine.close()


def test_stop_loss_hit_closes_position(tmp_path):
    inst = "NIFTY2672124150PE"
    feed = MockFeed({inst: [
        ("2026-07-17T11:46:00", 175.00, 173.00, 177.00),
        ("2026-07-17T12:00:00", 175.00, 173.00, 177.00),
    ]})
    sig = make_signal(
        symbol="NIFTY 50", instrument=inst, direction="bullish",
        entry=176.56, sl=175.00, target=240.00, strike=24150,
    )
    engine = build_engine(feed, [sig], tmp_path)
    new = engine.gather_new_signals()
    engine.process_new_signal(new[0])
    # Bar opens 178 -> low drops to 173 -> hit SL=175
    bar = TradeSnapshot(
        timestamp=datetime(2026, 7, 17, 12, 0, 0, tzinfo=IST),
        symbol="NIFTY 50", instrument=inst,
        open=178.0, high=179.0, low=173.0, close=174.0,
        bar_interval="15minute",
    )
    closed = engine.process_bar(bar)
    assert len(closed) == 1
    t = closed[0]
    assert t.exit_reason == ExitReason.STOP_LOSS
    assert t.exit_price == 175.00
    assert t.net_pnl < 0
    engine.close()


def test_time_stop(tmp_path):
    inst = "NIFTY2672124150CE"
    feed = MockFeed({inst: [("2026-07-17T11:46:00", 175.00, 173.00, 177.00)] * 50})
    sig = make_signal(
        symbol="NIFTY 50", instrument=inst, direction="bullish",
        entry=175.00, sl=170.00, target=185.00, strike=24150,
    )
    sig.max_bars = 3   # quick time stop for testing
    engine = build_engine(feed, [sig], tmp_path)
    engine.process_new_signal(engine.gather_new_signals()[0])
    base_ts = datetime(2026, 7, 17, 11, 46, 0, tzinfo=IST)
    for i in range(1, 4):  # bars 1..3
        # Advance time by 15 minutes per bar
        bar = TradeSnapshot(
            timestamp=base_ts + timedelta(minutes=15 * i),
            symbol="NIFTY 50", instrument=inst,
            open=175.0, high=176.0, low=174.0, close=175.2,    # neither SL nor target
            bar_interval="15minute",
        )
        closed = engine.process_bar(bar)
        if i < 3:
            assert closed == [], f"unexpected close on bar {i}"
        else:
            assert len(closed) == 1
            assert closed[0].exit_reason == ExitReason.TIME_STOP
            assert closed[0].exit_price == 175.2        # market close
    engine.close()


def test_square_off(tmp_path):
    inst = "ICICIBANK26JUL1400CE"
    feed = MockFeed({inst: [("2026-07-17T14:30:00", 50.00, 49.50, 50.50)]})
    sig = make_signal(
        symbol="ICICIBANK", instrument=inst, direction="bullish",
        entry=50.00, sl=45.00, target=60.00, strike=1400,
    )
    sig.trade_mode = "intraday"
    sig.max_bars = 50
    engine = build_engine(feed, [sig], tmp_path)
    engine.process_new_signal(engine.gather_new_signals()[0])
    bar = TradeSnapshot(
        timestamp=datetime(2026, 7, 17, 15, 15, 0, tzinfo=IST),
        symbol="ICICIBANK", instrument=inst,
        open=50.0, high=51.0, low=49.5, close=50.3,
        bar_interval="15minute",
    )
    closed = engine.process_bar(bar, is_square_off=True)
    assert len(closed) == 1
    assert closed[0].exit_reason == ExitReason.SQUARE_OFF
    engine.close()


def test_duplicate_instrument_skip(tmp_path):
    inst = "NIFTY2672124150CE"
    feed = MockFeed({inst: [("2026-07-17T11:46:00", 175.00, 173.00, 177.00)] * 50})
    s1 = make_signal(symbol="NIFTY 50", instrument=inst, direction="bullish",
                     entry=175, sl=170, target=185, strike=24150)
    s2 = make_signal(symbol="NIFTY 50", instrument=inst, direction="bullish",
                     entry=174, sl=170, target=185, strike=24150)

    # Use a source that yields signals one-at-a-time per next_batch
    class OneByOneSource:
        def __init__(self, signals):
            self._signals = list(signals)
            self._i = 0
        def next_batch(self):
            if self._i >= len(self._signals):
                return []
            s = self._signals[self._i]
            self._i += 1
            return [s]
        def close(self): pass

    recorder_path = tmp_path
    recorder = TradeRecorder(
        sqlite_path=str(recorder_path / "test_dup.sqlite"),
        csv_path=str(recorder_path / "test_dup.csv"),
    )
    from prometheus.papertrade import PaperTradeEngine
    source = OneByOneSource([s1, s2])
    engine = PaperTradeEngine(
        feed=feed, signal_source=source, recorder=recorder,
        enable_trailing=False, max_concurrent_positions=10,
        allow_duplicate_instrument=False,
    )

    # First round: should open s1
    seed_sigs = engine.gather_new_signals()
    assert len(seed_sigs) == 1
    tid = engine.process_new_signal(seed_sigs[0])
    assert tid is not None
    assert engine.open_positions_count() == 1

    # Second round: s2 — duplicate instrument/underlying → should skip
    seed_sigs = engine.gather_new_signals()
    assert len(seed_sigs) == 1
    tid2 = engine.process_new_signal(seed_sigs[0])
    assert tid2 is None
    assert engine.signals_skipped_duplicate == 1
    assert engine.open_positions_count() == 1
    engine.close()


def test_metrics_calculation(tmp_path):
    inst1 = "NIFTY2672124150CE"
    inst2 = "BANKNIFTY26JUL57800CE"   # different underlying so dup-guard doesn't skip
    feed = MockFeed({
        inst1: [("2026-07-17T11:46:00", 175.00, 173.00, 177.00)] * 100,
        inst2: [("2026-07-17T11:46:00", 200.00, 198.00, 202.00)] * 100,
    })
    sig_a = make_signal(symbol="NIFTY 50", instrument=inst1, direction="bullish",
                        entry=175, sl=170, target=180, strike=24150)
    sig_b = make_signal(symbol="NIFTY BANK", instrument=inst2, direction="bullish",
                        entry=200, sl=195, target=215, strike=57800)
    engine = build_engine(feed, [sig_a, sig_b], tmp_path)

    # Open both
    for s in engine.gather_new_signals():
        engine.process_new_signal(s)
    assert engine.open_positions_count() == 2

    # Bar 1: inst1 hits target (175 -> 180 = +5/lot), inst2 hits SL (200 -> 195)
    # Exit bars ~1 min after the signal's bar_timestamp (which is now-5min)
    # so holding_duration_seconds is positive and small.
    _exit_bar_ts = datetime.now(IST) - timedelta(minutes=4)
    bar1 = TradeSnapshot(
        timestamp=_exit_bar_ts,
        symbol="NIFTY 50", instrument=inst1,
        open=176, high=181, low=175, close=180.5, bar_interval="15minute",
    )
    bar2 = TradeSnapshot(
        timestamp=_exit_bar_ts,
        symbol="NIFTY BANK", instrument=inst2,
        open=199, high=200, low=194, close=195.5, bar_interval="15minute",
    )

    engine.process_bar(bar1)
    engine.process_bar(bar2)
    stats = engine.stats()
    assert stats.total_trades == 2
    assert stats.winning_trades == 1
    assert stats.losing_trades == 1
    assert stats.win_rate == 50.0
    # NIFTY win ~+5 * lot 75 = +375 - costs; BANKNIFTY loss -5 * lot 30 = -150 - costs
    # Total may be positive or negative depending on lot sizes; just check it's calculated
    assert stats.total_pnl != 0
    # exit-reason counts
    er = stats.exit_reason_counts
    assert er.get("target") == 1
    assert er.get("stop_loss") == 1
    # Holding duration should be positive (exit_time > entry_time)
    assert stats.avg_holding_duration_seconds > 0
    engine.close()
