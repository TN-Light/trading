# ============================================================================
# PROMETHEUS — Execution: Live Position Monitor
# ============================================================================
"""
Background thread that polls option LTP and manages the 5-stage trailing
stop for all open positions.  Direct port of the backtest engine's trailing
logic (engine.py lines 939-989) to live/paper trading.

5-stage trailing stop:
  Stage 0 — Breakeven trap:  at 0.6R  → SL to entry + 0.10R
  Stage 1 — Lock 20%:        at 1.0R  → SL to entry + 0.20R
  Stage 2 — Lock 50%:        at 2.0R  → SL to entry + 0.50R
  Stage 3 — Lock 70% runner: at 3.0R  → SL to entry + 0.70R, init HWM
  Stage 4 — Dynamic trail:   ratchet 30% below HWM, floor at 0.70R
"""

import threading
import time
from typing import Dict, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, field, asdict

from prometheus.utils.logger import logger
from prometheus.utils.indian_market import is_market_open, is_trading_day


@dataclass
class TrailingState:
    """
    Mirrors the backtest engine's per-position trailing stop state.
    Persisted to SQLite for crash recovery.
    """
    position_id: str
    tradingsymbol: str
    symbol: str
    entry_premium: float
    initial_sl: float
    current_sl: float
    target: float
    direction: str           # "bullish" or "bearish"
    strategy: str = ""
    entry_time: str = ""
    sl_order_id: str = ""

    # 5-stage flags (exactly match backtest engine.py:939-989)
    breakeven_set: bool = False
    trailing_activated: bool = False    # Stage 1
    trailing_stage2: bool = False       # Stage 2
    trailing_stage3: bool = False       # Stage 3
    premium_hwm: float = 0.0           # High-water mark for Stage 4

    # Time stop
    entry_bar_count: int = 0
    max_bars: int = 7

    # Config
    breakeven_ratio: float = 0.6
    risk_distance: float = 0.0

    # Intraday support
    bar_interval: str = "day"        # "day", "15minute", "5minute"
    trade_mode: str = "swing"        # "swing" or "intraday"

    def __post_init__(self):
        if self.risk_distance == 0.0 and self.entry_premium > 0:
            self.risk_distance = (
                self.entry_premium - self.initial_sl
                if self.initial_sl > 0
                else self.entry_premium * 0.3
            )

    def current_stage(self) -> str:
        """Human-readable current trailing stage."""
        if self.trailing_stage3:
            return "RUNNER (70%+)"
        elif self.trailing_stage2:
            return "LOCK 50%"
        elif self.trailing_activated:
            return "LOCK 20%"
        elif self.breakeven_set:
            return "BREAKEVEN"
        return "INITIAL"

    def to_dict(self) -> dict:
        return asdict(self)


class PositionMonitor:
    """
    Background daemon thread that manages live position trailing stops.

    For each open position:
    1. Polls LTP via broker.get_ltp() every poll_interval seconds
    2. Runs the 5-stage trailing stop (ported from backtest engine)
    3. Modifies broker SL order when trailing advances
    4. Detects SL breach / target hit as safety net
    5. Increments time-stop bar count once per trading day
    6. Persists state via callbacks for crash recovery
    """

    def __init__(
        self,
        broker,
        poll_interval: int = 15,
        on_exit: Optional[Callable] = None,
        on_trailing_update: Optional[Callable] = None,
        on_state_changed: Optional[Callable] = None,
    ):
        self.broker = broker
        self.poll_interval = poll_interval
        self._positions: Dict[str, TrailingState] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks wired by main.py
        self._on_exit = on_exit
        self._on_trailing_update = on_trailing_update
        self._on_state_changed = on_state_changed

        self._last_bar_increment_date = ""

        # LTP failure tracking per position
        self._ltp_fail_counts: Dict[str, int] = {}
        self._ltp_alert_sent: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_position(self, state: TrailingState):
        """Register a new position for monitoring."""
        with self._lock:
            self._positions[state.position_id] = state
        if self._on_state_changed:
            self._on_state_changed(state)
        logger.info(
            f"PositionMonitor: tracking {state.position_id} "
            f"({state.tradingsymbol}) entry={state.entry_premium:.2f} "
            f"SL={state.current_sl:.2f} target={state.target:.2f}"
        )

    def remove_position(self, position_id: str):
        """Stop monitoring a position (after close)."""
        with self._lock:
            self._positions.pop(position_id, None)
        self._ltp_fail_counts.pop(position_id, None)
        self._ltp_alert_sent.pop(position_id, None)

    def restore_positions(self, states: List[TrailingState]):
        """Restore positions from SQLite persistence (crash recovery)."""
        with self._lock:
            for state in states:
                self._positions[state.position_id] = state
        if states:
            logger.info(
                f"PositionMonitor: restored {len(states)} position(s) from last session"
            )

    @property
    def active_count(self) -> int:
        return len(self._positions)

    def get_positions(self) -> Dict[str, TrailingState]:
        with self._lock:
            return self._positions.copy()

    def start(self):
        """Start the monitoring daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="position-monitor",
        )
        self._thread.start()
        logger.info(f"PositionMonitor started (poll every {self.poll_interval}s)")

    def stop(self):
        """Stop the monitoring thread."""
        self._running = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _monitor_loop(self):
        while self._running:
            try:
                # Skip polling on non-trading days to save API quota
                if not is_trading_day(datetime.now().date()):
                    time.sleep(60)
                    continue

                with self._lock:
                    positions = list(self._positions.items())

                if not positions:
                    time.sleep(self.poll_interval)
                    continue

                # Time stop: increment bar count once per trading day at 3:30 PM
                self._check_daily_bar_increment()
                # Intraday: increment bar count every N minutes
                self._check_intraday_bar_increment()

                for pid, state in positions:
                    try:
                        ltp = self.broker.get_ltp(
                            state.tradingsymbol, exchange="NFO"
                        )
                        if ltp <= 0:
                            # Track consecutive LTP failures per position
                            self._ltp_fail_counts[pid] = self._ltp_fail_counts.get(pid, 0) + 1
                            if self._ltp_fail_counts[pid] >= 20 and not self._ltp_alert_sent.get(pid, False):
                                logger.warning(
                                    f"PositionMonitor: LTP unavailable for {pid} "
                                    f"({self._ltp_fail_counts[pid]} consecutive failures) — "
                                    f"trailing stops NOT updating"
                                )
                                self._ltp_alert_sent[pid] = True
                            continue
                        # Reset failure counter on success
                        self._ltp_fail_counts[pid] = 0
                        self._ltp_alert_sent.pop(pid, None)
                        self._process_tick(state, ltp)
                    except Exception as e:
                        logger.error(f"PositionMonitor tick error {pid}: {e}")

                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"PositionMonitor loop error: {e}")
                time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Core trailing stop logic — direct port of engine.py:939-989
    # ------------------------------------------------------------------

    def _process_tick(self, state: TrailingState, current_price: float):
        """
        Run the 5-stage trailing stop on a single position.
        Uses real LTP (current_price) instead of backtest's modeled premium.
        """
        entry = state.entry_premium
        rd = state.risk_distance
        old_sl = state.current_sl
        stage_changed = False

        if rd <= 0:
            return

        # ── Safety net: SL breach (broker SL-M should catch, but double-check) ──
        # Both CE and PE are BOUGHT options — current_price is premium LTP.
        # Premium dropping below SL = loss, regardless of direction.
        if current_price <= state.current_sl:
            logger.warning(
                f"[MONITOR] SL breach: {state.position_id} "
                f"LTP={current_price:.2f} <= SL={state.current_sl:.2f}"
            )
            if self._on_exit:
                self._on_exit(state.position_id, current_price, "stop_loss")
            return

        # ── Target hit ──
        # Premium rising above target = profit, regardless of direction.
        if state.target > 0 and current_price >= state.target:
            logger.info(f"[MONITOR] Target hit: {state.position_id} LTP={current_price:.2f}")
            if self._on_exit:
                self._on_exit(state.position_id, current_price, "target")
            return

        # ── Time stop ──
        if state.max_bars > 0 and state.entry_bar_count >= state.max_bars:
            logger.info(
                f"[MONITOR] Time stop: {state.position_id} "
                f"after {state.entry_bar_count} bars"
            )
            if self._on_exit:
                self._on_exit(state.position_id, current_price, "time_stop")
            return

        # ── 5-STAGE TRAILING STOP (bullish — long options) ──
        # For bearish, we're buying PUTs so premium INCREASES when underlying drops.
        # The trailing logic is the same: premium goes up = profit.
        price_for_trail = current_price

        if not state.breakeven_set:
            # Stage 0: BREAKEVEN TRAP
            be_trigger = entry + rd * state.breakeven_ratio
            if price_for_trail >= be_trigger:
                new_sl = entry + rd * 0.10
                state.current_sl = new_sl
                state.breakeven_set = True
                stage_changed = True
                logger.info(
                    f"[TRAIL] {state.position_id} Stage 0 BREAKEVEN: "
                    f"SL {old_sl:.2f} -> {new_sl:.2f}"
                )

        elif not state.trailing_activated:
            # Stage 1: Lock 20% at 1.0R
            if price_for_trail >= entry + rd * 1.0:
                new_sl = entry + rd * 0.20
                state.current_sl = new_sl
                state.trailing_activated = True
                stage_changed = True
                logger.info(
                    f"[TRAIL] {state.position_id} Stage 1 LOCK 20%: "
                    f"SL {old_sl:.2f} -> {new_sl:.2f}"
                )

        elif not state.trailing_stage2:
            # Stage 2: Lock 50% at 2.0R
            if price_for_trail >= entry + rd * 2.0:
                new_sl = entry + rd * 0.50
                state.current_sl = new_sl
                state.trailing_stage2 = True
                stage_changed = True
                logger.info(
                    f"[TRAIL] {state.position_id} Stage 2 LOCK 50%: "
                    f"SL {old_sl:.2f} -> {new_sl:.2f}"
                )

        elif not state.trailing_stage3:
            # Stage 3: Lock 70%, begin runner
            if price_for_trail >= entry + rd * 3.0:
                new_sl = entry + rd * 0.70
                state.current_sl = new_sl
                state.trailing_stage3 = True
                state.premium_hwm = price_for_trail
                stage_changed = True
                logger.info(
                    f"[TRAIL] {state.position_id} Stage 3 RUNNER: "
                    f"SL {old_sl:.2f} -> {new_sl:.2f}, HWM={price_for_trail:.2f}"
                )

        else:
            # Stage 4: Dynamic trail with high-water mark
            if price_for_trail > state.premium_hwm:
                state.premium_hwm = price_for_trail
            floor_sl = entry + rd * 0.70
            trail_offset = (state.premium_hwm - entry) * 0.30
            dynamic_sl = state.premium_hwm - trail_offset
            new_sl = max(floor_sl, dynamic_sl)
            if new_sl > state.current_sl:
                state.current_sl = new_sl
                stage_changed = True
                logger.info(
                    f"[TRAIL] {state.position_id} Stage 4 DYNAMIC: "
                    f"SL {old_sl:.2f} -> {new_sl:.2f} "
                    f"(HWM={state.premium_hwm:.2f})"
                )

        # ── If SL changed, update broker order + persist ──
        if stage_changed and state.current_sl != old_sl:
            self._modify_broker_sl(state)
            if self._on_trailing_update:
                self._on_trailing_update(state, old_sl)
            if self._on_state_changed:
                self._on_state_changed(state)

    # ------------------------------------------------------------------
    # Broker SL modification
    # ------------------------------------------------------------------

    def _modify_broker_sl(self, state: TrailingState):
        """Modify the SL-M order on the broker to the new trigger price."""
        if not state.sl_order_id:
            logger.debug(f"No SL order ID for {state.position_id}, skip modify")
            return
        try:
            from prometheus.execution.broker import OrderStatus
            # Check if order is still pending/open
            order = self.broker.get_order_status(state.sl_order_id)
            if order.status in (OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                logger.info(
                    f"SL order {state.sl_order_id} already {order.status.value}, "
                    f"cannot modify"
                )
                return

            result = self.broker.modify_order(
                state.sl_order_id,
                trigger_price=round(state.current_sl, 2),
            )
            logger.info(
                f"Broker SL modified: {state.sl_order_id} -> "
                f"trigger={state.current_sl:.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to modify broker SL for {state.position_id}: {e}")

    # ------------------------------------------------------------------
    # Time stop — daily bar counter
    # ------------------------------------------------------------------

    def _check_daily_bar_increment(self):
        """Increment bar count once per trading day at 3:30 PM (swing only)."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        if today_str == self._last_bar_increment_date:
            return
        if not is_trading_day(now.date()):
            return
        if now.hour < 15 or (now.hour == 15 and now.minute < 30):
            return

        with self._lock:
            for state in self._positions.values():
                if state.bar_interval == "day":
                    state.entry_bar_count += 1
        self._last_bar_increment_date = today_str
        logger.info(
            f"PositionMonitor: daily bar count incremented "
            f"({len(self._positions)} positions)"
        )

    def _check_intraday_bar_increment(self):
        """Increment bar count for intraday positions based on elapsed time."""
        now = datetime.now()

        with self._lock:
            for state in self._positions.values():
                if state.bar_interval == "day":
                    continue

                interval_minutes = 5 if state.bar_interval == "5minute" else 15

                # Track last increment per-position using _last_bar_ts
                last_ts = getattr(state, "_last_bar_ts", None)
                if last_ts is None:
                    state._last_bar_ts = now
                    continue

                elapsed = (now - last_ts).total_seconds()
                if elapsed >= interval_minutes * 60:
                    bars_to_add = int(elapsed // (interval_minutes * 60))
                    state.entry_bar_count += bars_to_add
                    state._last_bar_ts = now
