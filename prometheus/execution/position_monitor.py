# ============================================================================
# PROMETHEUS — Execution: Live Position Monitor
# ============================================================================
"""
Background thread that polls option LTP and manages the 5-stage trailing
stop for all open positions.  Direct port of the backtest engine's trailing
logic (engine.py lines 939-989) to live/paper trading.

5-stage trailing stop:
  Stage 0 — Breakeven trap:  at 0.4R  → SL to entry + 0.10R
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
from prometheus.execution.broker import OrderStatus


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
    entry_orders_json: str = ""

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
    breakeven_ratio: float = 0.4
    risk_distance: float = 0.0

    # Intraday support
    bar_interval: str = "day"        # "day", "15minute", "5minute"
    trade_mode: str = "swing"        # "swing" or "intraday"

    # Adverse indicator exit
    adverse_exit_enabled: bool = True
    _last_adverse_check: float = 0.0  # timestamp of last check

    # 3-phase premium floor tracking (1=immunity, 2=buffered, 3=full)
    _current_phase: int = 1

    # Credit Spread support (Theta decay & inverted trailing)
    strategy_type: str = ""
    target_decay_price: float = 0.0
    breakeven_decay_price: float = 0.0
    hard_sl_price: float = 0.0

    def __post_init__(self):
        if self.risk_distance == 0.0 and self.entry_premium > 0:
            if getattr(self, "strategy_type", "") == "credit_spread":
                self.risk_distance = abs(self.initial_sl - self.entry_premium) if self.initial_sl > 0 else self.entry_premium * 0.5
            else:
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
        data_engine=None,
    ):
        self.broker = broker
        self.poll_interval = poll_interval
        self._data_engine = data_engine
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
                        import traceback
                        logger.error(f"PositionMonitor tick error {pid}: {e}\n{traceback.format_exc()}")

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
        old_sl = state.current_sl
        stage_changed = False

        # ── CREDIT SPREAD DECAY & STOP LOSS MONITORING ──
        if getattr(state, "strategy_type", "") == "credit_spread":
            net_credit = state.entry_premium
            target_decay = getattr(state, "target_decay_price", 0.0) or (net_credit * 0.30)
            breakeven_decay = getattr(state, "breakeven_decay_price", 0.0) or (net_credit * 0.50)
            hard_sl = getattr(state, "hard_sl_price", 0.0) or (net_credit * 1.50)

            # 1. Hard Stop Loss: spread jumped above 1.5x initial credit
            if current_price >= hard_sl:
                logger.warning(
                    f"[MONITOR] Credit Spread Hard SL: {state.position_id} "
                    f"Spread LTP={current_price:.2f} >= Hard SL={hard_sl:.2f} (Credit={net_credit:.2f})"
                )
                if self._on_exit:
                    self._on_exit(state.position_id, current_price, "stop_loss_credit_spread")
                return

            # 2. Target Decay: 70% of credit decayed (e.g. spread dropped to 30% of entry)
            if current_price <= target_decay and target_decay > 0:
                logger.info(
                    f"[MONITOR] Credit Spread Target Hit (70% Decay): {state.position_id} "
                    f"Spread LTP={current_price:.2f} <= Target={target_decay:.2f} (Credit={net_credit:.2f})"
                )
                if self._on_exit:
                    self._on_exit(state.position_id, current_price, "target_decay_credit_spread")
                return

            # 3. Breakeven Lock: 50% of credit decayed -> ratchet SL down to protect gain
            if current_price <= breakeven_decay and not state.breakeven_set:
                new_sl = net_credit * 0.85  # guaranteed 15% profit floor
                state.current_sl = new_sl
                state.breakeven_set = True
                logger.info(
                    f"[TRAIL] Credit Spread Breakeven Lock: {state.position_id} "
                    f"SL {old_sl:.2f} -> {new_sl:.2f} (Spread LTP={current_price:.2f})"
                )

            # 4. If current price crosses above ratcheted SL
            if state.breakeven_set and current_price >= state.current_sl:
                logger.info(
                    f"[MONITOR] Credit Spread Breakeven SL Hit: {state.position_id} "
                    f"Spread LTP={current_price:.2f} >= SL={state.current_sl:.2f}"
                )
                if self._on_exit:
                    self._on_exit(state.position_id, current_price, "breakeven_exit_credit_spread")
                return

            return

        rd = state.risk_distance
        if rd <= 0:
            return

        # ── 3-Phase Premium Floor (ported from engine.py L1422-1444) ──
        # Prevents premature exits from IV crush, spread widening, and
        # stop-loss hunts in the first few bars after entry.
        #   Phase 1 (≤3 bars): Total immunity — ignore premium noise
        #   Phase 2 (4-5 bars): Moderate buffer — SL at 80% of original
        #   Phase 3 (>5 bars): Full enforcement — trust options pricing
        #
        # Bug #3 (2026-07-22): The original implementation only consulted
        # `state.current_sl` (the ratcheted trailing-stop SL) in the
        # Phase 3 `else` branch. In Phase 1/2 it only checked the
        # catastrophic floor (entry×0.2) and `initial_sl×0.8`. As the
        # trailing-stop ratchet quickly advanced `state.current_sl` above
        # entry (Stage 0 BREAKEVEN → Stage 4 RUNNER) within minutes, any
        # subsequent drop below the ratcheted SL was SILENTLY IGNORED for
        # the ~5 bars of Phase 1+2 immunity — observed today as a 60+
        # minute uncovered breach on POS-20260722-0001 (57000PE).
        #
        # A compounding bug at line 296 (now fixed below) pushed the
        # broker SL order DOWN to `initial_sl×0.8` (often below entry)
        # on the Phase 1→2 transition, even when `state.current_sl` had
        # already ratcheted well above entry — disarming the broker-side
        # stop too.
        #
        # Fix: insert a UNIVERSAL SL check that ALWAYS honors the
        # ratcheted `state.current_sl` regardless of phase. Phase gating
        # now only adds catastrophic-floor protection on top, never
        # relaxes the ratcheted SL.
        bars_held = state.entry_bar_count
        if state.current_sl > 0 and state.current_sl > state.initial_sl and current_price <= state.current_sl:
            # The trailing-stop ratchet has advanced current_sl above the
            # initial SL (i.e., breakeven trap or higher has engaged).
            # Honor that ratcheted SL on every tick — do NOT let Phase 1/2
            # immunity silently disarm it.
            phase_label = (
                "phase1_sl_breach" if bars_held <= 3
                else "phase2_sl_breach" if bars_held <= 5
                else "stop_loss_premium_phase3"
            )
            logger.warning(
                f"[MONITOR] {phase_label}: {state.position_id} "
                f"LTP={current_price:.2f} <= current_sl={state.current_sl:.2f} "
                f"(initial_sl={state.initial_sl:.2f}, bars_held={bars_held})"
            )
            # Sync broker SL up to the ratcheted value before exiting
            # (in case the broker order was lagging — never lower it).
            self._modify_broker_sl_manual(state, state.current_sl)
            if self._on_exit:
                self._on_exit(state.position_id, current_price, phase_label)
            return

        if bars_held <= 3:
            # Phase 1: Immunity to IV crush / spread widening / stop hunts
            # But add a catastrophic circuit breaker — if premium drops > 80%,
            # something is genuinely wrong (not just noise).
            catastrophic_floor = entry * 0.20
            if current_price <= catastrophic_floor:
                logger.warning(
                    f"[MONITOR] Phase 1 CATASTROPHIC exit: {state.position_id} "
                    f"LTP={current_price:.2f} <= 20% of entry={entry:.2f}"
                )
                if self._on_exit:
                    self._on_exit(state.position_id, current_price, "catastrophic_phase1")
                return
        elif bars_held <= 5:
            # Phase 2: Allow spread to settle, use buffered SL
            buffered_sl = state.initial_sl * 0.8
            if current_price <= buffered_sl:
                logger.warning(
                    f"[MONITOR] Premium floor Phase 2 exit: {state.position_id} "
                    f"LTP={current_price:.2f} <= buffered SL={buffered_sl:.2f}"
                )
                if self._on_exit:
                    self._on_exit(state.position_id, current_price, "stop_loss_premium_phase2")
                return

            # Sync broker SL to Phase 2 buffered limit if we just transitioned
            # out of Phase 1. Bug #3 fix: NEVER lower the broker SL below the
            # ratcheted `state.current_sl`. Previously this path pushed the
            # broker SL order DOWN to `initial_sl×0.8` (e.g. 332.64 → 233.76
            # today), silently disarming the broker-side stop.
            if getattr(state, "_current_phase", 1) < 2:
                state._current_phase = 2
                # Only lower broker SL if the trailing ratchet never engaged
                # (current_sl still == initial_sl). If it ratcheted, keep it
                # at the ratcheted value — never give back profit-lock.
                broker_sl_target = (
                    state.current_sl
                    if state.current_sl > state.initial_sl
                    else buffered_sl
                )
                self._modify_broker_sl_manual(state, broker_sl_target)
        else:
            # Phase 3: Full enforcement — normal SL check
            if current_price <= state.current_sl:
                logger.warning(
                    f"[MONITOR] SL breach: {state.position_id} "
                    f"LTP={current_price:.2f} <= SL={state.current_sl:.2f}"
                )
                if self._on_exit:
                    self._on_exit(state.position_id, current_price, "stop_loss_premium_phase3")
                return

            # Sync broker SL to Phase 3 normal limit if we just transitioned out of Phase 2
            if getattr(state, "_current_phase", 1) < 3:
                state._current_phase = 3
                self._modify_broker_sl_manual(state, state.current_sl)

        # ── Target hit ──
        # Premium rising above target = profit, regardless of direction.
        if state.target > 0 and current_price >= state.target:
            logger.info(f"[MONITOR] Target hit: {state.position_id} LTP={current_price:.2f}")
            if self._on_exit:
                self._on_exit(state.position_id, current_price, "target")
            return

        # ── Adverse indicator exit ──
        try:
            from prometheus.config import get
            if get("intraday.adverse_exit.enabled", False) and state.trade_mode == "intraday":
                if self._check_adverse_indicator(state, current_price):
                    if self._on_exit:
                        self._on_exit(state.position_id, current_price, "adverse_reversal")
                    return
        except Exception as e:
            logger.debug(f"Adverse exit check error: {e}")

        # ── Time stop ──
        if state.max_bars > 0 and state.entry_bar_count >= state.max_bars:
            logger.info(
                f"[MONITOR] Time stop: {state.position_id} "
                f"after {state.entry_bar_count} bars"
            )
            if self._on_exit:
                self._on_exit(state.position_id, current_price, "time_stop")
            return

        # ── Stagnation Exit (cut stagnant options after 4 bars to prevent theta decay) ──
        if state.trade_mode == "intraday" and state.entry_bar_count >= 4 and not state.breakeven_set:
            if current_price < entry * 1.03:
                logger.info(
                    f"[MONITOR] Stagnation cut: {state.position_id} "
                    f"after {state.entry_bar_count} bars (LTP={current_price:.2f} <= Entry*1.03={entry*1.03:.2f})"
                )
                if self._on_exit:
                    self._on_exit(state.position_id, current_price, "stagnation_exit")
                return

        # ── 5-STAGE TRAILING STOP (bullish — long options) ──
        # For bearish, we're buying PUTs so premium INCREASES when underlying drops.
        # The trailing logic is the same: premium goes up = profit.
        price_for_trail = current_price

        if not state.breakeven_set:
            # Stage 0: BREAKEVEN TRAP — triggers at 0.4R OR at +10% premium gain
            be_trigger_rd = entry + rd * state.breakeven_ratio if rd > 0 else entry * 1.10
            be_trigger_pct = entry * 1.10
            be_trigger = min(be_trigger_rd, be_trigger_pct)
            if price_for_trail >= be_trigger:
                new_sl = entry + max(rd * 0.10, entry * 0.015)
                if new_sl > state.current_sl:
                    state.current_sl = new_sl
                    state.breakeven_set = True
                    stage_changed = True
                    logger.info(
                        f"[TRAIL] {state.position_id} Stage 0 BREAKEVEN: "
                        f"SL {old_sl:.2f} -> {new_sl:.2f} (LTP={price_for_trail:.2f}, Entry={entry:.2f})"
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

    def _check_adverse_indicator(self, state: TrailingState, current_price: float) -> bool:
        """Check if SuperTrend has flipped against the trade direction.
        
        Returns True if an adverse exit should be triggered.
        """
        if not state.adverse_exit_enabled:
            return False
        if not self._data_engine:
            return False
        
        # Don't check in first 3 bars (let the trade breathe)
        if state.entry_bar_count < 3:
            return False
        
        # Throttle: only check every 5 minutes
        import time as _time
        now = _time.time()
        check_interval = 300  # 5 minutes
        try:
            from prometheus.config import get
            check_interval = get("intraday.adverse_exit.check_interval_seconds", 300)
        except Exception:
            pass
        if now - state._last_adverse_check < check_interval:
            return False
        state._last_adverse_check = now
        
        # Fetch recent 15min data and compute SuperTrend
        try:
            from prometheus.signals.technical import calculate_supertrend
            symbol = state.symbol
            data = self._data_engine.fetch_historical(symbol, days=5, interval="15minute")
            if data is None or len(data) < 30:
                return False
            
            st_df = calculate_supertrend(data)
            if st_df.empty or "supertrend_direction" not in st_df.columns:
                return False
            
            st_direction = st_df["supertrend_direction"].iloc[-1]
            # st_direction: 1 = bullish, -1 = bearish
            
            trade_is_bullish = state.direction == "bullish"  # CE position
            st_is_bullish = (st_direction == 1)
            
            # Adverse = trade direction != SuperTrend direction
            if trade_is_bullish and not st_is_bullish:
                logger.warning(
                    f"[ADVERSE] SuperTrend BEARISH vs BULLISH position: {state.position_id} "
                    f"({state.tradingsymbol}) — triggering adverse exit"
                )
                return True
            elif not trade_is_bullish and st_is_bullish:
                logger.warning(
                    f"[ADVERSE] SuperTrend BULLISH vs BEARISH position: {state.position_id} "
                    f"({state.tradingsymbol}) — triggering adverse exit"
                )
                return True
            
            return False
        except Exception as e:
            logger.debug(f"Adverse indicator check failed for {state.position_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Broker SL modification
    # ------------------------------------------------------------------

    def _modify_broker_sl(self, state: TrailingState):
        """Modify the SL-M order on the broker to the new trigger price."""
        self._modify_broker_sl_manual(state, state.current_sl)

    def _modify_broker_sl_manual(self, state: TrailingState, manual_trigger: float):
        """Internal helper to modify broker SL to a specific trigger price.

        Bug #3 (2026-07-22) defense-in-depth: NEVER lower the broker SL-M
        order below the ratcheted ``state.current_sl`` — once the trailing
        stop has locked in profit (Stage 0+ engaged), giving back that
        protection by lowering the broker order is a logical error. The
        only legitimate transitions are upward (ratchet advances, or
        phase 3 sync). If a caller passes a trigger below current_sl,
        clamp upward to current_sl.
        """
        if not state.sl_order_id:
            logger.debug(f"No SL order ID for {state.position_id}, skip modify")
            return
        # Defensive: never lower broker SL below the ratcheted current_sl.
        # (Prevents the Phase 1→2 transition from pushing broker order
        # DOWN to `initial_sl×0.8` while `state.current_sl` has ratcheted
        # well above entry — observed today at 10:31:39.)
        if state.current_sl > 0 and manual_trigger < state.current_sl:
            logger.warning(
                f"[MONITOR] Refusing to LOWER broker SL below ratcheted value: "
                f"{state.position_id} requested={manual_trigger:.2f} "
                f"< current_sl={state.current_sl:.2f} — clamping up"
            )
            manual_trigger = state.current_sl
        try:
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
                trigger_price=round(manual_trigger, 2),
            )
            logger.info(
                f"Broker SL modified: {state.sl_order_id} -> "
                f"trigger={manual_trigger:.2f}"
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
                    try:
                        entry_dt = datetime.strptime(state.entry_time, "%Y-%m-%d %H:%M:%S")
                        elapsed_sec = (now - entry_dt).total_seconds()
                        if elapsed_sec > 0:
                            state.entry_bar_count = int(elapsed_sec // (interval_minutes * 60))
                            logger.info(f"[MONITOR] Initialized bar count for {state.position_id} to {state.entry_bar_count} based on entry time {state.entry_time}")
                    except Exception as e:
                        logger.error(f"[MONITOR] Error parsing entry_time for {state.position_id}: {e}")
                    state._last_bar_ts = now
                    continue

                elapsed = (now - last_ts).total_seconds()
                if elapsed >= interval_minutes * 60:
                    bars_to_add = int(elapsed // (interval_minutes * 60))
                    state.entry_bar_count += bars_to_add
                    state._last_bar_ts = now
