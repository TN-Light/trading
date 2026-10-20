"""
Post-Exit Signal Tracking & Attribution Engine.

Monitors closed paper and live trading contracts for up to 60 minutes after exit
(and at session end) to empirically determine:
  1. Premature Shakeout Rate: Did SL exits reverse and hit Target or reclaim Entry?
  2. MFE Truncation: How much additional upside was left on the table beyond Target?
  3. Drawdown Saved: Did Stop Loss exits prevent further catastrophic collapse?
  4. Kill-Switch Validation: Did 45m inactivity exits collapse or break out?

Runs asynchronously and lock-free off SharedPriceMicroCache or closed bars.
Zero broker API rate-limit overhead.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable

IST = timezone(timedelta(hours=5, minutes=30))
logger = logging.getLogger("prometheus.papertrade.post_exit_tracker")


@dataclass
class PostExitWatchItem:
    trade_id: str
    symbol: str
    instrument: str
    underlying: str
    direction: str
    tier: str
    entry_price: float
    exit_price: float
    exit_reason: str
    exit_time: datetime
    target: float = 0.0
    stop_loss: float = 0.0
    entry_spot: float = 0.0
    holding_duration_seconds: int = 0
    price_t5m: Optional[float] = None
    price_t15m: Optional[float] = None
    price_t30m: Optional[float] = None
    price_t60m: Optional[float] = None
    price_eod: Optional[float] = None
    post_exit_mfe: float = 0.0
    post_exit_mae: float = 0.0
    target_reclaimed: bool = False
    entry_reclaimed: bool = False
    is_premature_shakeout: bool = False
    alpha_left_pts: float = 0.0
    drawdown_saved_pts: float = 0.0
    status: str = "WATCHING"
    created_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["exit_time"] = self.exit_time.isoformat() if hasattr(self.exit_time, "isoformat") else str(self.exit_time)
        return d


class PostExitTracker:
    """
    Manages non-blocking post-exit observation of closed trades.
    """

    def __init__(
        self,
        recorder=None,
        max_watch_seconds: int = 3600,
        data_engine=None,
    ):
        self.recorder = recorder
        self.max_watch_seconds = max_watch_seconds
        self.data_engine = data_engine
        self._watches: Dict[str, PostExitWatchItem] = {}
        self._lock = threading.Lock()
        self._listeners: List[Callable[[PostExitWatchItem], None]] = []

    @property
    def active_watch_count(self) -> int:
        with self._lock:
            return len(self._watches)

    def add_listener(self, callback: Callable[[PostExitWatchItem], None]) -> None:
        """Register callback invoked on milestone updates or completion."""
        self._listeners.append(callback)

    def register_closed_trade(self, trade: Any) -> Optional[PostExitWatchItem]:
        """
        Register a closed PaperTrade or dictionary into the post-exit watchlist.
        """
        try:
            trade_id = getattr(trade, "trade_id", None) or (trade.get("trade_id") if isinstance(trade, dict) else "")
            if not trade_id:
                return None

            symbol = getattr(trade, "symbol", "") or (trade.get("symbol", "") if isinstance(trade, dict) else "")
            instrument = getattr(trade, "instrument", "") or (trade.get("instrument", "") if isinstance(trade, dict) else "")
            underlying = getattr(trade, "underlying", "") or (trade.get("underlying", "") if isinstance(trade, dict) else symbol)
            
            raw_dir = getattr(trade, "direction", "") or (trade.get("direction", "") if isinstance(trade, dict) else "LONG")
            direction_str = raw_dir.value if hasattr(raw_dir, "value") else str(raw_dir)

            tier = getattr(trade, "tier", "") or (trade.get("tier", "") if isinstance(trade, dict) else "")
            entry_price = float(getattr(trade, "entry_price", 0.0) or (trade.get("entry_price", 0.0) if isinstance(trade, dict) else 0.0))
            exit_price = float(getattr(trade, "exit_price", 0.0) or (trade.get("exit_price", 0.0) if isinstance(trade, dict) else 0.0))

            raw_reason = getattr(trade, "exit_reason", "") or (trade.get("exit_reason", "") if isinstance(trade, dict) else "unknown")
            exit_reason_str = raw_reason.value if hasattr(raw_reason, "value") else str(raw_reason)

            exit_time = getattr(trade, "exit_time", None) or (trade.get("exit_time") if isinstance(trade, dict) else None)
            if not isinstance(exit_time, datetime):
                exit_time = datetime.now(IST)

            target = float(getattr(trade, "target", 0.0) or (trade.get("target", 0.0) if isinstance(trade, dict) else 0.0))
            stop_loss = float(getattr(trade, "stop_loss", 0.0) or (trade.get("stop_loss", 0.0) if isinstance(trade, dict) else 0.0))
            entry_spot = float(getattr(trade, "entry_spot", 0.0) or (trade.get("entry_spot", 0.0) if isinstance(trade, dict) else 0.0))
            dur_sec = int(getattr(trade, "holding_duration_seconds", 0) or (trade.get("holding_duration_seconds", 0) if isinstance(trade, dict) else 0))

            watch_item = PostExitWatchItem(
                trade_id=trade_id,
                symbol=symbol,
                instrument=instrument,
                underlying=underlying,
                direction=direction_str,
                tier=tier,
                entry_price=entry_price,
                exit_price=exit_price,
                exit_reason=exit_reason_str,
                exit_time=exit_time,
                target=target,
                stop_loss=stop_loss,
                entry_spot=entry_spot,
                holding_duration_seconds=dur_sec,
                post_exit_mfe=exit_price,
                post_exit_mae=exit_price,
                status="WATCHING",
            )

            with self._lock:
                self._watches[trade_id] = watch_item

            if self.recorder and hasattr(self.recorder, "record_post_exit_observation"):
                self.recorder.record_post_exit_observation(watch_item.to_dict())

            logger.info(
                f"[PostExitTracker] registered {trade_id} ({instrument}) "
                f"exit_price={exit_price:.2f} reason={exit_reason_str} for 60m shadow tracking"
            )
            return watch_item
        except Exception as e:
            logger.error(f"[PostExitTracker] register_closed_trade failed: {e}")
            return None

    def on_tick(
        self,
        instrument: str,
        ltp: float,
        timestamp: Optional[datetime] = None,
    ) -> List[PostExitWatchItem]:
        """
        Process a new price tick for watched contracts.
        Updates MFE, MAE, milestones, and shakeout/preservation status.
        """
        if ltp <= 0 or not instrument:
            return []

        now_dt = timestamp or datetime.now(IST)
        now_mono = time.monotonic()
        updated_items: List[PostExitWatchItem] = []
        completed_ids: List[str] = []

        with self._lock:
            for tid, item in list(self._watches.items()):
                # Match instrument or composite leg
                if item.instrument and item.instrument != instrument and instrument not in item.instrument:
                    if (not item.symbol or item.symbol != instrument) and (not item.underlying or item.underlying != instrument):
                        continue

                # 1. Update Post-Exit MFE and MAE
                if ltp > item.post_exit_mfe:
                    item.post_exit_mfe = round(ltp, 2)
                if item.post_exit_mae <= 0 or ltp < item.post_exit_mae:
                    item.post_exit_mae = round(ltp, 2)

                # 2. Check Reclaim & Shakeout Conditions
                # Long premium position: target > entry_price (whether Call or Put buyer!)
                # Short premium / credit spread: target < entry_price (seller wants decay)
                is_long_premium = (item.target > item.entry_price) if item.target > 0 else ("SPREAD" not in item.instrument.upper() and "/" not in item.instrument)
                if is_long_premium:
                    if item.target > 0 and ltp >= item.target:
                        item.target_reclaimed = True
                    if ltp >= item.entry_price:
                        item.entry_reclaimed = True
                else:
                    # Bearish credit spread or short premium position
                    if item.target > 0 and ltp <= item.target:
                        item.target_reclaimed = True
                    if ltp <= item.entry_price:
                        item.entry_reclaimed = True

                # Premature Shakeout: Trade stopped out early, but price subsequently reached Target or Entry
                reason_u = item.exit_reason.upper()
                is_defensive_exit = (
                    "STOP" in reason_u
                    or "BAILOUT" in reason_u
                    or "INACTIVITY" in reason_u
                    or "KILL_SWITCH" in reason_u
                )
                if is_defensive_exit and (item.target_reclaimed or item.entry_reclaimed):
                    item.is_premature_shakeout = True

                if is_long_premium:
                    # Alpha left on table (points higher than exit)
                    item.alpha_left_pts = round(max(0.0, item.post_exit_mfe - item.exit_price), 2)
                    # Drawdown saved (points lower than exit, proving risk protection)
                    item.drawdown_saved_pts = round(max(0.0, item.exit_price - item.post_exit_mae), 2)
                else:
                    # For credit spread / short premium: lower premium is favorable
                    item.alpha_left_pts = round(max(0.0, item.exit_price - item.post_exit_mae), 2)
                    item.drawdown_saved_pts = round(max(0.0, item.post_exit_mfe - item.exit_price), 2)

                # 3. Milestone checkpoints based on elapsed wall-clock seconds
                t_now = now_dt
                t_exit = item.exit_time
                if t_now.tzinfo is None and t_exit.tzinfo is not None:
                    if hasattr(IST, "localize"):
                        try:
                            t_now = IST.localize(t_now)
                        except Exception:
                            t_now = t_now.replace(tzinfo=t_exit.tzinfo)
                    else:
                        t_now = t_now.replace(tzinfo=t_exit.tzinfo)
                elif t_now.tzinfo is not None and t_exit.tzinfo is None:
                    if hasattr(IST, "localize"):
                        try:
                            t_exit = IST.localize(t_exit)
                        except Exception:
                            t_exit = t_exit.replace(tzinfo=t_now.tzinfo)
                    else:
                        t_exit = t_exit.replace(tzinfo=t_now.tzinfo)

                elapsed_sec = (t_now - t_exit).total_seconds()
                if elapsed_sec < 0:
                    elapsed_sec = now_mono - item.created_at

                if elapsed_sec >= 300 and item.price_t5m is None:
                    item.price_t5m = round(ltp, 2)
                if elapsed_sec >= 900 and item.price_t15m is None:
                    item.price_t15m = round(ltp, 2)
                if elapsed_sec >= 1800 and item.price_t30m is None:
                    item.price_t30m = round(ltp, 2)
                if elapsed_sec >= 3600 and item.price_t60m is None:
                    item.price_t60m = round(ltp, 2)

                # 4. Check horizon completion (60 min = 3600s)
                if elapsed_sec >= self.max_watch_seconds:
                    item.status = "COMPLETED"
                    completed_ids.append(tid)

                updated_items.append(item)

                # Persist to SQLite on updates
                if self.recorder and hasattr(self.recorder, "record_post_exit_observation"):
                    try:
                        self.recorder.record_post_exit_observation(item.to_dict())
                    except Exception as rec_err:
                        logger.debug(f"[PostExitTracker] recording failed: {rec_err}")

            # Cleanly unregister completed watches
            for cid in completed_ids:
                logger.info(
                    f"[PostExitTracker] completed 60m observation for {cid}: "
                    f"MFE={self._watches[cid].post_exit_mfe:.2f} Shakeout={self._watches[cid].is_premature_shakeout}"
                )
                self._watches.pop(cid, None)

        for upd in updated_items:
            for listener in self._listeners:
                try:
                    listener(upd)
                except Exception:
                    pass

        return updated_items

    def on_bar(self, snapshot: Any) -> List[PostExitWatchItem]:
        """Convenience adapter to evaluate closed OHLC bar."""
        inst = getattr(snapshot, "instrument", "") or getattr(snapshot, "symbol", "")
        close_px = float(getattr(snapshot, "close", 0.0) or 0.0)
        ts = getattr(snapshot, "timestamp", None)
        return self.on_tick(inst, close_px, timestamp=ts)

    def poll_micro_cache(self) -> int:
        """
        Poll SharedPriceMicroCache lock-free for all active watch instruments.
        Zero broker network quota consumed.
        """
        try:
            from prometheus.data.price_micro_cache import get_shared_micro_cache
            cache = get_shared_micro_cache()
        except Exception:
            return 0

        updates = 0
        with self._lock:
            active_items = list(self._watches.values())

        for item in active_items:
            entry = cache.get(item.instrument) or cache.get_by_symbol(item.instrument)
            if entry and entry.price > 0 and entry.age() < 10.0:
                self.on_tick(item.instrument, entry.price)
                updates += 1
        return updates

    def on_session_end(self, quotes: Optional[Dict[str, float]] = None) -> None:
        """
        Finalize all active watches at session close (15:25–15:30 IST).
        Records price_eod and flushes status to COMPLETED.
        """
        with self._lock:
            for tid, item in list(self._watches.items()):
                if quotes and item.instrument in quotes:
                    eod_px = float(quotes[item.instrument])
                    item.price_eod = round(eod_px, 2)
                    if eod_px > item.post_exit_mfe:
                        item.post_exit_mfe = round(eod_px, 2)
                    if eod_px < item.post_exit_mae or item.post_exit_mae <= 0:
                        item.post_exit_mae = round(eod_px, 2)
                item.status = "COMPLETED"
                if self.recorder and hasattr(self.recorder, "record_post_exit_observation"):
                    try:
                        self.recorder.record_post_exit_observation(item.to_dict())
                    except Exception:
                        pass
            self._watches.clear()
            logger.info("[PostExitTracker] finalized all watches for session close")
