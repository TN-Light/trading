"""
Signal source — a pluggable adapter that feeds signals into the paper engine.

Two operating modes today:

1. ``LiveReplaySource`` — hooks into the existing ``LiveScanner`` scan loop
   and re-emits every signal the scanner finds (swing AND intraday), without
   touching the production ``OrderManager``. Metered by the existing scanner
   schedule.
2. ``HistoricalReplaySource`` — replays bars through the existing backed
   signal generators (``SignalEvaluator`` / ``SignalConverter`` for swing;
   intraday APEX for intraday), one bar at a time. Used to fully re-run the
   strategy in deterministic fashion across historical data.

Both adapters translate the existing ``ExecutableSignal`` / ``TradeSetup``
dict into a ``SignalNotification``, the unified DTO ``PaperTradeEngine``
consumes. The DTO is intentionally a thin dataclass rather than reusing the
production signal dict — decoupling lets the production protocol evolve
without forcing paper-engine regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Iterator, Callable, Any, Protocol

from prometheus.papertrade.types import Direction
from prometheus.utils.logger import logger
from prometheus.utils.symbol_format import (
    api_tradingsymbol, resolve_underlying,
)


@dataclass
class SignalNotification:
    """Unified DTO — what PaperTradeEngine ingests from any source."""

    symbol: str                       # display symbol ("NIFTY 50", "ICICIBANK")
    instrument: str                   # API tradingsymbol ("NIFTY2672124150CE")
    underlying: str                   # "NIFTY"
    direction: Direction
    strike: float
    option_type: str                  # "CE" or "PE"
    expiry: str                       # ISO date "YYYY-MM-DD"

    # Initial SL/target/price — these are STRATEGY-LEVEL parameters, not
    # risk-manager overrides. The paper engine honors them because they
    # describe the strategy author's intended exit; the engine never
    # applies additional risk overlays.
    entry_price_hint: float = 0.0     # theoretical/live premium estimate
    stop_loss: float = 0.0
    target: float = 0.0
    signal_score: float = 0.0
    signal_confidence: float = 0.0

    # How many 15-minute-ish bars this position may live (strategy-level
    # time-stop, not a risk overlay). Defaults to None — engine uses its
    # configured default.
    max_bars: Optional[int] = None

    trade_mode: str = "intraday"      # "swing" or "intraday" — select exit logic

    strategy: str = ""                # for logging/audit

    bar_timestamp: Optional[datetime] = None  # when the signal was generated
    metadata: dict = None            # bag for any extra context


class SignalSource(Protocol):
    """Minimal interface the engine polls for new signals."""

    def next_batch(self) -> list: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers — convert existing types to SignalNotification
# ---------------------------------------------------------------------------
def from_executable_signal(executable, max_bars_default: int = 16) -> SignalNotification:
    """Convert a pipeline ``ExecutableSignal`` to ``SignalNotification``.

    The ``ExecutableSignal`` is what ``SignalConverter`` emits. Its
    ``instrument`` is already in proper API-format (Kite-style), so we
    accept it directly.
    """
    direction = Direction.from_signal_direction(executable.direction)

    instrument = executable.instrument
    underlying = ""
    expiry_str = executable.expiry or ""
    strike = float(executable.strike or 0)

    # Fallback: if instrument is missing or malformed, regenerate it.
    # This is the same defensive normalization we did in order_manager.py
    # for the legacy intraday path.
    needs_regen = (
        not instrument
        or " " in instrument
        or instrument != instrument.upper()
        or not (instrument.endswith("CE") or instrument.endswith("PE"))
    )
    if needs_regen and strike > 0 and expiry_str:
        instrument = api_tradingsymbol(executable.symbol, expiry_str, strike, executable.option_type) or instrument

    if instrument:
        underlying = resolve_underlying(executable.symbol)

    return SignalNotification(
        symbol=executable.symbol,
        instrument=instrument,
        underlying=underlying,
        direction=direction,
        strike=strike,
        option_type=executable.option_type,
        expiry=expiry_str,
        entry_price_hint=float(executable.entry_price or 0),
        stop_loss=float(executable.stop_loss or 0),
        target=float(executable.target or 0),
        signal_score=0.0,                                          # not exposed in ExecutableSignal
        signal_confidence=float(executable.confidence or 0),
        max_bars=max_bars_default,
        trade_mode="swing",                                         # SignalConverter path is swing
        strategy=executable.strategy or "",
        bar_timestamp=_parse_bar_timestamp(executable.bar_timestamp),
        metadata={"raw": executable.raw or {}},
    )


def from_trend_setup(setup, symbol: str, max_bars_default: int = 16) -> SignalNotification:
    """Convert a strategies/trend.py ``TradeSetup`` to ``SignalNotification``.

    The trend-strategy setup goes through the intraday APEX path. Its
    ``instrument`` may be a placeholder "NIFTY 50 24200 CE" (legacy) and its
    ``expiry`` is "WEEKLY" — we rebuild the API-format tradingsymbol here
    using the same helper used in order_manager. This is the 2026-07-17 fix
    in action: we never bank a placeholder as the broker-facing tradingsymbol.
    """
    direction = Direction.from_signal_direction(setup.signal_direction)
    option_type = "CE" if direction == Direction.LONG else "PE"

    strike = float(setup.strike or 0)
    underlying = resolve_underlying(symbol)

    # Resolve real expiry date from indian_market rather than trust setup.expiry
    expiry_str = ""
    try:
        from prometheus.utils.indian_market import get_expiry_date
        expiry_str = get_expiry_date(symbol).strftime("%Y-%m-%d")
    except Exception as e:
        logger.debug(f"from_trend_setup: expiry resolve failed for {symbol}: {e}")

    # Build proper API tradingsymbol, fall back to the legacy instrument if
    # formatter fails (so we at least log something).
    instrument = api_tradingsymbol(symbol, expiry_str, strike, option_type) or setup.instrument

    return SignalNotification(
        symbol=symbol,
        instrument=instrument,
        underlying=underlying,
        direction=direction,
        strike=strike,
        option_type=option_type,
        expiry=expiry_str,
        entry_price_hint=float(setup.entry_price or 0),
        stop_loss=float(setup.stop_loss or 0),
        target=float(setup.target or 0),
        signal_score=float(setup.signal_strength or 0),
        signal_confidence=0.0,
        max_bars=max_bars_default,
        trade_mode="intraday",
        strategy=setup.strategy or "trend",
        bar_timestamp=_parse_bar_timestamp(getattr(setup, "bar_timestamp", None) or None),
    )


def from_signal_dict(signal: dict, max_bars_default: int = 16) -> SignalNotification:
    """Generic converter for raw signal dicts in the existing OrderManager
    shape. Used by the live-replay source which subscribes to scanner events.
    """
    is_spread = signal.get("strategy_type") == "credit_spread" or "SPREAD" in str(signal.get("action", ""))
    dir_str = signal.get("direction", "")
    direction = Direction.from_signal_direction(dir_str)
    if is_spread and "CALL" in str(signal.get("spread_type", "")):
        direction = Direction.SHORT

    option_type = signal.get("option_type") or ("CE" if direction == Direction.LONG else "PE")
    symbol = signal.get("symbol", "")
    strike = float(signal.get("strike") or signal.get("short_strike") or 0)
    expiry_in = signal.get("expiry") or ""
    if hasattr(expiry_in, "strftime"):
        expiry_str = expiry_in.strftime("%Y-%m-%d")
    elif isinstance(expiry_in, str) and expiry_in != "WEEKLY":
        expiry_str = expiry_in[:10]
    else:
        expiry_str = ""

    underlying = resolve_underlying(symbol)
    instrument = signal.get("instrument", "") or signal.get("tradingsymbol", "") or ""
    if is_spread and not instrument:
        instrument = f"{underlying}_{int(strike)}{option_type}_SPREAD"
    else:
        needs_regen = (
            not is_spread and (
                not instrument
                or " " in instrument
                or instrument != instrument.upper()
                or not (instrument.endswith("CE") or instrument.endswith("PE"))
            )
        )
        if needs_regen and strike > 0 and expiry_str:
            instrument = api_tradingsymbol(symbol, expiry_str, strike, option_type) or instrument

    entry_hint = float(signal.get("entry_price") or signal.get("net_credit") or signal.get("entry_premium") or 0)
    stop_loss = float(signal.get("stop_loss") or signal.get("hard_sl_price") or 0)
    target = float(signal.get("target") or signal.get("target_decay_price") or 0)
    score_val = float(signal.get("signal_strength") or signal.get("signal_score") or (3.5 if is_spread else 0.0))

    return SignalNotification(
        symbol=symbol,
        instrument=instrument,
        underlying=underlying,
        direction=direction,
        strike=strike,
        option_type=option_type,
        expiry=expiry_str,
        entry_price_hint=entry_hint,
        stop_loss=stop_loss,
        target=target,
        signal_score=score_val,
        signal_confidence=float(signal.get("confidence") or (0.75 if is_spread else 0.0)),
        max_bars=int(signal.get("max_bars") or max_bars_default),
        trade_mode=signal.get("trade_mode", "intraday"),
        strategy=signal.get("strategy", "Hedged_Credit_Spread" if is_spread else ""),
        bar_timestamp=_parse_bar_timestamp(signal.get("bar_timestamp")),
        metadata={"raw": signal},
    )


def _parse_bar_timestamp(ts) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", ""))
    except Exception:
        # Try alternative format
        try:
            return datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Live-source: registers a callback with the running scanner
# ---------------------------------------------------------------------------
class LiveReplaySource:
    """Hooks an existing ``LiveScanner`` to emit signals to the paper engine.

    Doesn't drive the scan loop itself — the existing Prometheus run loop is
    the clock. The source subscribes via ``add_listener`` if available, or
    polls the scanner's last-batch queue otherwise.

    For now this is implemented as a thin wrapper: the actual integration
    happens in ``main.py`` which calls ``engine.notify_signal(signal_dict)``
    directly for every signal the scanner produces, regardless of risk.
    """

    def __init__(self, scanner: Any = None):
        self.scanner = scanner
        self._buffer: list = []
        # If the scanner supports listeners, register ours
        if scanner is not None:
            try:
                scanner.add_listener(self._on_signal)
            except AttributeError:
                pass

    # Called by the scanner when a new signal is produced
    def _on_signal(self, signal_dict: dict) -> None:
        try:
            self._buffer.append(from_signal_dict(signal_dict))
        except Exception as e:
            logger.error(f"LiveReplaySource: failed to convert signal: {e}")

    # ------------------------------------------------------------------
    # SignalSource protocol
    # ------------------------------------------------------------------
    def next_batch(self) -> list:
        if not self._buffer:
            return []
        batch = list(self._buffer)
        self._buffer.clear()
        return batch

    def close(self) -> None:
        # Try to uninstall the listener if supported
        if self.scanner is not None:
            try:
                self.scanner.remove_listener(self._on_signal)
            except AttributeError:
                pass


# ---------------------------------------------------------------------------
# Historical-source: walks bars through the existing signal generators
# ---------------------------------------------------------------------------
class HistoricalReplaySource:
    """Replays historical bars through the existing signal generators.

    Implementation notes:

    * For swing mode, uses ``SignalEvaluator`` + ``SignalConverter`` (the
      existing backtest-validated generator).
    * For intraday mode, uses the same `_make_signal_generator` factory the
      live intraday path uses (the APEX intraday engine).
    * We never invent a generator; we reuse the production pipeline.

    This is the same code path used by ``run_intraday_mode`` in main.py; the
    only difference is the engine-side consumer — instead of ``OrderManager``
    we feed the resulting signal to ``PaperTradeEngine``.
    """

    def __init__(
        self,
        symbols: list,
        days: int = 60,
        bar_interval: str = "15minute",
        trade_mode: str = "intraday",
        prometheus_instance: Any = None,
    ):
        self.symbols = list(symbols)
        self.days = int(days)
        self.bar_interval = bar_interval
        self.trade_mode = trade_mode
        # ``prometheus_instance`` is required for swing mode (the generator
        # factory reads regime_detector, risk.bracket_manager, and capital
        # profile helpers off of it). For intraday mode it's currently unused
        # but accepted for forward compatibility.
        self._prometheus = prometheus_instance
        self._initialized = False
        self._iterators: dict = {}
        self._buffer: list = []

    def _initialize(self) -> None:
        if self._initialized:
            return
        from datetime import datetime, timedelta
        from prometheus.data.engine import DataEngine
        from prometheus.utils.indian_market import IST

        end = datetime.now(IST).replace(hour=15, minute=29, second=0, microsecond=0)
        start = end - timedelta(days=self.days)
        self._data = DataEngine()

        # We need daily + hourly DataFrames per symbol to seed the
        # SignalEvaluator on its first evaluate() call (its initialize()
        # reads scan_data.daily for regime detection + scan_data.hourly for
        # intraday bias). We fetch them once per symbol; the per-bar ScanData
        # will reuse the same daily/hourly frames throughout the replay — the
        # generator has already baked regime/bias state in at init time, so
        # this matches the production semantics (scanner precomputes daily +
        # hourly once per scan, then pushes primary bars).
        for symbol in self.symbols:
            try:
                df_primary = self._data.fetch_historical(
                    symbol=symbol, days=self.days, interval=self.bar_interval,
                    force_refresh=False,
                )
                if df_primary is None or len(df_primary) < 30:
                    logger.warning(
                        f"HistoricalReplaySource: insufficient primary data for {symbol} "
                        f"({len(df_primary) if df_primary is not None else 0} bars) — skipping"
                    )
                    continue

                # Daily regime window — at least 120 days for stable regime
                # detection. The generator only needs this on its first call.
                df_daily = self._data.fetch_historical(
                    symbol=symbol, days=max(self.days, 120), interval="day",
                    force_refresh=False,
                )
                if df_daily is None:
                    df_daily = df_primary.iloc[0:0]

                # Hourly bias window — 30 days of 60-minute bars.
                if self.bar_interval == "day":
                    # Daily mode: hourly bias map is computed from the daily
                    # frame itself (see _compute_daily_bias).
                    df_hourly = df_daily
                else:
                    df_hourly = self._data.fetch_historical(
                        symbol=symbol, days=max(self.days, 30), interval="60minute",
                        force_refresh=False,
                    )
                    if df_hourly is None:
                        df_hourly = df_primary.iloc[0:0]

                logger.info(
                    f"HistoricalReplaySource: loaded {symbol} "
                    f"(primary={len(df_primary)} {self.bar_interval}, "
                    f"daily={len(df_daily)}, hourly={len(df_hourly)})"
                )
                self._iterators[symbol] = self._make_iterator(
                    symbol, df_primary, df_hourly, df_daily,
                )
            except Exception as e:
                logger.error(f"HistoricalReplaySource: load failed for {symbol}: {e}")

        self._initialized = True

    def _make_iterator(
        self,
        symbol: str,
        df_primary,
        df_hourly,
        df_daily,
    ) -> Iterator:
        """Yield one SignalNotification per bar where the strategy fires.

        Uses the same ``SignalEvaluator`` + ``SignalConverter`` pair the
        live scanner uses. The evaluator is constructed with the
        Prometheus instance (so it can access the regime detector, risk
        bracket manager, and capital-profile helpers), then fed a
        ``ScanData`` per bar whose ``primary`` slice grows one bar at a
        time. Daily + hourly frames are fixed (the generator bakes them
        in during its one-shot initialize()).
        """
        from prometheus.pipeline.signal_evaluator import SignalEvaluator
        from prometheus.pipeline.signal_converter import SignalConverter as Conv
        from prometheus.pipeline.types import ScanData, DataStatus

        if self._prometheus is None:
            raise RuntimeError(
                "HistoricalReplaySource: swing replay requires a "
                "prometheus_instance (passed by run_papertrade). None was "
                "provided."
            )

        evaluator = SignalEvaluator(
            self._prometheus, symbol, primary_interval=self.bar_interval,
        )
        converter = Conv()

        # Empty daily/hourly frames are acceptable — initialize() falls back
        # to scan_data.primary when daily/hourly are empty. We pass the real
        # ones if we succeeded in fetching them.
        empty_daily = df_daily.empty
        empty_hourly = df_hourly.empty

        for i in range(len(df_primary)):
            slice_df = df_primary.iloc[:i + 1]
            if len(slice_df) < 50:
                # SignalEvaluator itself enforces a 50-bar floor on primary
                # before invoking the underlying generator.
                continue
            try:
                scan = ScanData(
                    symbol=symbol,
                    primary=slice_df,
                    hourly=df_hourly if not empty_hourly else slice_df,
                    daily=df_daily if not empty_daily else slice_df,
                    status=DataStatus.OK,
                )
                result = evaluator.evaluate(scan)
                if result is None or not result.has_signal:
                    continue
                executable = converter.convert(result, symbol)
                if executable is None:
                    continue
                yield from_executable_signal(executable)
            except Exception as e:
                logger.debug(
                    f"HistoricalReplaySource: bar {i} for {symbol} threw {e}"
                )
                continue

    # ------------------------------------------------------------------
    # SignalSource protocol
    # ------------------------------------------------------------------
    def next_batch(self) -> list:
        self._initialize()
        batch = list(self._buffer)
        self._buffer.clear()
        # Pull one signal from each symbol's iterator, if available
        for symbol, it in list(self._iterators.items()):
            try:
                signal = next(it)
                if signal is not None:
                    batch.append(signal)
            except StopIteration:
                # Exhausted — remove so we don't poll it again
                self._iterators.pop(symbol, None)
            except Exception as e:
                logger.error(f"HistoricalReplaySource: iterator error for {symbol}: {e}")
                self._iterators.pop(symbol, None)
        return batch

    def close(self) -> None:
        # Nothing to release — DataEngine is shared by reference
        pass
