"""LiveScanner: Orchestrates scan cycles for paper/live trading.

Performance optimizations:
- Parallel data fetching via ThreadPoolExecutor (3 threads)
- Parallel signal evaluation (4 threads)  
- Resampled 15min→60min (eliminates 11 API calls)
- Inter-scan price refresh every 30s
- Async Telegram for scan summaries

Total: ~42s → ~8s per scan cycle (5x speedup)
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dtime, date
from typing import Dict, List, Optional, Tuple

from prometheus.pipeline.types import (
    ScanCycle, SymbolScanResult, DataStatus, ScanData, SignalResult,
    GateResult, GateVerdict,
)
from prometheus.pipeline.data_bridge import DataBridge
from prometheus.pipeline.signal_evaluator import SignalEvaluator
from prometheus.pipeline.signal_converter import SignalConverter
from prometheus.pipeline.execution_gate import ExecutionGate
from prometheus.pipeline.notifier import Notifier
from prometheus.utils.logger import logger


def is_market_open_now() -> bool:
    """Check if Indian market is currently open."""
    now = datetime.now()
    t = now.time()
    if t < dtime(9, 15) or t >= dtime(15, 30):
        return False
    if now.weekday() >= 5:
        return False
    return True


def is_trading_day_today() -> bool:
    """Check if today is a trading day."""
    try:
        from prometheus.utils.calendar import is_trading_day
        return is_trading_day(date.today())
    except ImportError:
        return datetime.now().weekday() < 5


class LiveScanner:
    """Main scan orchestrator for paper/live trading.
    
    Architecture:
      Phase 1 (parallel): Fetch data for all symbols via ThreadPoolExecutor
      Phase 2 (parallel): Evaluate signals for all symbols
      Phase 3 (sequential): OI confluence → gate check → live premium → execute
      Background: Price refresh thread (every 30s between scans)
      Background: OI cache thread (every 2 min for index symbols)
    """
    
    def __init__(
        self,
        prometheus_instance,
        symbols: List[str],
        scan_interval_seconds: int = 900,
        skip_first_minutes: int = 15,
        max_positions: int = 6,
        daily_loss_limit: float = 450.0,
    ):
        self._prometheus = prometheus_instance
        self._symbols = symbols
        self._scan_interval = max(900, scan_interval_seconds)
        self._skip_first_minutes = skip_first_minutes
        
        # Components
        self._bridge = DataBridge(prometheus_instance.data)
        self._evaluators: Dict[str, SignalEvaluator] = {}
        self._converter = SignalConverter()
        self._gate = ExecutionGate(
            max_positions=max_positions,
            daily_loss_limit=daily_loss_limit,
        )
        self._notifier = Notifier(prometheus_instance.telegram)
        
        # State
        self._running = False
        self._last_scan_time: Optional[datetime] = None
        self._today: Optional[date] = None
        
        # Pullback entry state (ported from engine.py entry_timing)
        # Instead of buying at market on breakout, queue a limit order
        # at signal_price - 0.3×ATR and wait for pullback fill.
        self._pending_signals: Dict[str, dict] = {}  # symbol -> pending signal
        self._pending_scans_waited: Dict[str, int] = {}  # symbol -> scan cycles waited
        from prometheus.config import get as cfg_get
        self._pullback_atr_fraction = float(
            cfg_get('intraday.v2.pullback_atr_fraction', 0.3)
        )
        self._pullback_max_wait_scans = int(
            cfg_get('intraday.v2.pullback_max_wait_scans', 2)
        )
        
        # Thread safety
        self.scan_lock = threading.Lock()
        
        # Price refresh config
        self._price_refresh_seconds = int(
            cfg_get('paper.price_refresh_seconds', 30)
        )
        self._price_refresh_thread: Optional[threading.Thread] = None
        self._price_refresh_stop = threading.Event()
        
        # VIX cache (refreshed each scan cycle)
        self._current_vix: float = 0.0
        
        # OI cache (background thread)
        self._oi_cache = None
        try:
            from prometheus.pipeline.oi_cache import OICache
            oi_analyzer = getattr(prometheus_instance, 'oi_analyzer', None)
            if oi_analyzer:
                self._oi_cache = OICache(
                    data_engine=prometheus_instance.data,
                    oi_analyzer=oi_analyzer,
                    symbols=symbols,
                    refresh_interval_seconds=120,
                )
                logger.info("LiveScanner: OI cache initialized")
            else:
                logger.info("LiveScanner: no oi_analyzer found, OI cache disabled")
        except Exception as e:
            logger.warning(f"LiveScanner: OI cache init failed: {e}")

    def _route_paper_capture_or_legacy(self, exec_dict: dict, confirm: bool = False):
        """Bug #2 (2026-07-22): Unified routing — when LivePaperCapture is
        active, paper-mode signals must NEVER enter the legacy
        OrderManager path (which gates through the 15K-capital RiskManager
        whose max_correlated_pct=50% and Duplicate-instrument checks fire
        on every signal after the first, silently dropping 11/14 signals
        today). The 6deda3f dual-path kill was incomplete because
        scanner.py retained two direct order_manager.execute_signal calls.

        Returns the position object (legacy path) or None (paper_capture
        path or rejection).
        """
        paper_capture = getattr(self._prometheus, "_paper_capture", None)
        if paper_capture is not None and getattr(paper_capture, "enabled", False):
            # Paper mode — route exclusively through LivePaperCapture;
            # the legacy 15K RiskManager is not consulted.
            paper_capture.on_signal(exec_dict)
            return None
        return self._prometheus.order_manager.execute_signal(
            exec_dict, confirm=confirm
        )

    def _is_paper_capture_active(self) -> bool:
        """Bug D.4 (2026-07-25 audit dispatch): paper-mode ingestion gate.

        ``True`` when LivePaperCapture is enabled — meaning every signal
        must be routed through ``paper_capture.on_signal`` exactly once,
        and the legacy ``_dispatch_multi_account`` / OrderManager paths
        must NOT be invoked at all (the wrapper at
        ``main.py:_dispatch_multi_account`` forwards to paper_capture
        when ``self._paper_capture`` is enabled, so calling it would
        re-feed the signal a second time).
        """
        pc = getattr(self._prometheus, "_paper_capture", None)
        return pc is not None and bool(getattr(pc, "enabled", False))

    def _ensure_evaluators(self):
        """Create evaluators for any new symbols."""
        for symbol in self._symbols:
            if symbol not in self._evaluators:
                self._evaluators[symbol] = SignalEvaluator(
                    self._prometheus, symbol, primary_interval='15minute'
                )
    
    def _refresh_evaluators_daily(self):
        """Refresh evaluators at start of each trading day."""
        today = date.today()
        if self._today != today:
            self._today = today
            for ev in self._evaluators.values():
                ev.refresh()
            self._gate.reset_daily()
            self._pending_signals.clear()
            self._pending_scans_waited.clear()
            logger.info(f"LiveScanner: daily refresh for {today}")
    
    # ------------------------------------------------------------------
    # VIX Gate
    # ------------------------------------------------------------------
    
    def _fetch_vix(self) -> float:
        """Fetch current India VIX (cached per scan cycle)."""
        try:
            data = getattr(self._prometheus, 'data', None)
            if data and hasattr(data, 'get_vix'):
                vix = data.get_vix()
                if vix and vix > 0:
                    self._current_vix = float(vix)
                    return self._current_vix
        except Exception as e:
            logger.debug(f"LiveScanner: VIX fetch failed: {e}")
        return self._current_vix
    
    def _check_vix_gate(self, signal) -> Optional[str]:
        """Check if VIX allows this trade.
        
        Returns rejection reason string, or None if OK.
        
        Rules (from institutional practice):
        - VIX > 28: options overpriced, theta bleed kills buyers
        - VIX > 35: crisis regime, halt all trading
        - VIX < 10: dead market, premiums too thin for meaningful RR
        """
        vix = self._current_vix
        if vix <= 0:
            return None  # No VIX data — allow trade
        
        if vix > 35:
            return f"VIX={vix:.1f} > 35 — crisis regime, all trades halted"
        
        if vix > 28 and signal.action in ('BUY_CE', 'BUY_PE'):
            return f"VIX={vix:.1f} > 28 — option buying blocked (theta bleed)"
        
        if vix < 10:
            return f"VIX={vix:.1f} < 10 — dead market, premiums too thin"
        
        return None  # OK
    
    # ------------------------------------------------------------------
    # Parallel Data Fetch + Signal Evaluation
    # ------------------------------------------------------------------
    
    def _fetch_symbol_data(self, symbol: str) -> Tuple[str, ScanData]:
        """Fetch data for one symbol (thread-safe, no shared mutable state)."""
        scan_data = self._bridge.fetch_scan_data(symbol)
        return (symbol, scan_data)
    
    def _evaluate_symbol(
        self, symbol: str, scan_data: ScanData
    ) -> Tuple[str, SignalResult]:
        """Evaluate signal for one symbol (thread-safe per evaluator)."""
        evaluator = self._evaluators[symbol]
        signal_result = evaluator.evaluate(scan_data)
        return (symbol, signal_result)
    
    # ------------------------------------------------------------------
    # Main Scan Cycle
    # ------------------------------------------------------------------
    
    def run_scan_cycle(self) -> ScanCycle:
        """Run one scan cycle across all symbols (thread-safe)."""
        if not self.scan_lock.acquire(blocking=False):
            logger.info("LiveScanner: scan skipped — another scan in progress")
            return ScanCycle(results=[])
        try:
            return self._run_scan_cycle_locked()
        finally:
            self.scan_lock.release()
    
    def _run_scan_cycle_locked(self) -> ScanCycle:
        """Internal scan cycle (must be called with scan_lock held)."""
        scan_start = time.monotonic()
        
        self._ensure_evaluators()
        self._refresh_evaluators_daily()
        
        # Fetch VIX once per scan cycle
        self._fetch_vix()
        if self._current_vix > 0:
            logger.info(f"LiveScanner: VIX={self._current_vix:.1f}")
        
        # Update position count for gate checks
        try:
            open_count = len([
                m for m in self._prometheus.order_manager.managed_positions.values()
                if m.status == 'open'
            ])
            self._gate.update_positions(open_count)
        except Exception:
            pass
        
        # ── Phase 1: Parallel data fetch (I/O bound → threading) ──
        scan_data_map: Dict[str, ScanData] = {}
        t0 = time.monotonic()
        
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix='fetch') as pool:
            futures = {
                pool.submit(self._fetch_symbol_data, sym): sym
                for sym in self._symbols
            }
            for future in as_completed(futures):
                try:
                    sym, data = future.result(timeout=30)
                    scan_data_map[sym] = data
                except Exception as e:
                    sym = futures[future]
                    logger.error(f"LiveScanner: fetch failed for {sym}: {e}")
                    scan_data_map[sym] = ScanData(
                        symbol=sym,
                        primary=__import__('pandas').DataFrame(),
                        hourly=__import__('pandas').DataFrame(),
                        daily=__import__('pandas').DataFrame(),
                        status=DataStatus.FETCH_ERROR,
                        fetch_time=datetime.now(),
                        error_message=str(e),
                    )
        
        fetch_elapsed = time.monotonic() - t0
        logger.info(f"LiveScanner: data fetch completed in {fetch_elapsed:.1f}s")
        
        # ── Phase 2: Parallel signal evaluation (numpy releases GIL) ──
        signal_map: Dict[str, SignalResult] = {}
        t1 = time.monotonic()
        
        # Only evaluate symbols with valid data
        eval_symbols = [
            sym for sym in self._symbols
            if scan_data_map.get(sym) and
               scan_data_map[sym].status in (DataStatus.OK, DataStatus.STALE)
        ]
        
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix='eval') as pool:
            futures = {
                pool.submit(self._evaluate_symbol, sym, scan_data_map[sym]): sym
                for sym in eval_symbols
            }
            for future in as_completed(futures):
                try:
                    sym, sig = future.result(timeout=10)
                    signal_map[sym] = sig
                except Exception as e:
                    sym = futures[future]
                    logger.error(f"LiveScanner: eval failed for {sym}: {e}")
                    signal_map[sym] = SignalResult(
                        raw_signal=None, symbol=sym,
                        diagnostics={'error': str(e)},
                    )
        
        eval_elapsed = time.monotonic() - t1
        logger.info(f"LiveScanner: signal eval completed in {eval_elapsed:.1f}s")
        
        # ── Phase 2.5: Try to fill pending pullback signals ──
        results: List[SymbolScanResult] = []
        self._try_fill_pending_signals(results)
        
        # ── Phase 3: Sequential gate → premium → execute → notify ──
        
        for symbol in self._symbols:
            result = SymbolScanResult(symbol=symbol)
            scan_data = scan_data_map.get(symbol)
            
            if not scan_data or scan_data.status not in (DataStatus.OK, DataStatus.STALE):
                result.data_status = scan_data.status if scan_data else DataStatus.FETCH_ERROR
                result.data_error = scan_data.error_message if scan_data else 'No data'
                results.append(result)
                continue
            
            result.data_status = scan_data.status
            
            # Get signal from Phase 2
            signal_result = signal_map.get(symbol)
            if not signal_result:
                results.append(result)
                continue
            
            result.signal = signal_result
            
            if not signal_result.has_signal:
                results.append(result)
                continue
            
            # Step 3: Convert to executable
            executable = self._converter.convert(signal_result, symbol)
            result.executable = executable
            
            if executable is None:
                results.append(result)
                continue
            
            # Step 3b: OI confluence — log only, do NOT modify confidence
            # Rationale: the ±15% boost/penalty was unvalidated. OI data is
            # logged and displayed in Telegram for manual review, but doesn't
            # affect the pass/fail gate until we have trade data proving
            # OI-confirmed signals actually outperform.
            oi_info = None
            if self._oi_cache:
                oi_info = self._oi_cache.get_oi_confluence(
                    symbol, executable.direction
                )
                if oi_info and not oi_info.get('stale'):
                    logger.info(
                        f"LiveScanner: {symbol} OI: {oi_info['summary']}"
                    )
                    # Attach OI data to raw dict for Telegram notification
                    if executable.raw:
                        executable.raw['oi_pcr'] = oi_info.get('pcr', 0)
                        executable.raw['oi_sentiment'] = oi_info.get('summary', '')
                        executable.raw['oi_agrees'] = oi_info.get('agrees', True)
            
            # Step 3c: Confidence gate (after OI boost/penalty)
            min_confidence = 0.35
            if executable.confidence < min_confidence:
                logger.info(
                    f"LiveScanner: {symbol} rejected — confidence "
                    f"{executable.confidence:.0%} < {min_confidence:.0%}"
                )
                results.append(result)
                continue
            
            # Step 3c: VIX gate
            vix_reject = self._check_vix_gate(executable)
            if vix_reject:
                logger.info(f"LiveScanner: {symbol} rejected — {vix_reject}")
                result.gate = GateResult(
                    verdict=GateVerdict.REJECT_VIX,
                    reason=vix_reject,
                )
                results.append(result)
                continue
            
            # Step 4: Gate check
            gate_result = self._gate.check(executable)
            result.gate = gate_result
            
            if not gate_result.passed:
                results.append(result)
                continue
            
            # Step 5: Fetch live premium from Angel One (BEFORE notify/execute)
            live_premium = self._fetch_live_premium(executable)
            if live_premium and live_premium.get('ltp', 0) > 0:
                real_ltp = live_premium['ltp']
                old_entry = executable.entry_price
                executable.entry_price = real_ltp
                if old_entry > 0:
                    ratio = real_ltp / old_entry
                    executable.stop_loss = round(executable.stop_loss * ratio, 2)
                    executable.target = round(executable.target * ratio, 2)
                if executable.raw:
                    executable.raw['entry_price'] = real_ltp
                    executable.raw['live_premium'] = real_ltp
                    executable.raw['is_live_premium'] = True
                    executable.raw['bs_estimate'] = old_entry
                    executable.raw['bid'] = live_premium.get('bid', 0)
                    executable.raw['ask'] = live_premium.get('ask', 0)
                logger.info(
                    f"LiveScanner: {symbol} live premium Rs {real_ltp:.2f} "
                    f"(BS estimate was {old_entry:.2f}, diff {((real_ltp/old_entry)-1)*100:+.1f}%)"
                )
            
            # Step 6: Execute
            try:
                self._notifier.notify_signal_alert(executable)
                exec_dict = {**executable.raw}
                exec_dict['action'] = executable.action
                exec_dict['instrument'] = executable.instrument or exec_dict.get('instrument', '')
                exec_dict['confidence'] = executable.confidence
                exec_dict['option_type'] = executable.option_type
                if executable.strike:
                    exec_dict['strike'] = executable.strike
                if executable.expiry:
                    exec_dict['option_expiry_date'] = executable.expiry
                if executable.lot_size:
                    exec_dict['lot_size'] = executable.lot_size
                if executable.quantity:
                    exec_dict['quantity'] = executable.quantity
                
                # Feed real premium to PaperTrader before execution
                self._feed_real_premium_to_broker(executable, live_premium)
                
                # ── Pullback Entry: Queue as pending instead of immediate ──
                # If ATR data is available, wait for a pullback before entering.
                # This avoids buying at the top of breakout candles.
                atr_at_signal = executable.raw.get('atr', 0) if executable.raw else 0
                pullback_fraction = executable.raw.get('entry_pullback_atr', self._pullback_atr_fraction) if executable.raw else self._pullback_atr_fraction
                spot_at_signal = executable.raw.get('spot_at_signal', 0) if executable.raw else 0
                
                if atr_at_signal > 0 and spot_at_signal > 0 and pullback_fraction > 0:
                    pullback_offset = atr_at_signal * pullback_fraction
                    if executable.direction == 'bullish':
                        limit_spot = spot_at_signal - pullback_offset
                    else:
                        limit_spot = spot_at_signal + pullback_offset
                    
                    self._pending_signals[symbol] = {
                        'executable': executable,
                        'exec_dict': exec_dict,
                        'live_premium': live_premium,
                        'limit_spot': limit_spot,
                        'signal_spot': spot_at_signal,
                        'atr': atr_at_signal,
                        'result': result,
                    }
                    self._pending_scans_waited[symbol] = 0
                    logger.info(
                        f"LiveScanner: {symbol} queued for pullback entry — "
                        f"limit_spot={limit_spot:.1f} "
                        f"(signal_spot={spot_at_signal:.1f}, "
                        f"pullback={pullback_offset:.1f})"
                    )
                    results.append(result)
                    continue
                
                # No ATR data → immediate execution (fallback)
                # Bug #2 (2026-07-22): route via paper_capture helper when active.
                position = self._route_paper_capture_or_legacy(
                    exec_dict, confirm=False
                )

                # Bug D.4 (2026-07-25 audit dispatch): when LivePaperCapture is
                # active, ``_route_paper_capture_or_legacy`` already opened a
                # paper position. Calling ``self._prometheus._dispatch_multi_account``
                # here would RE-ENTER ``paper_capture.on_signal`` via
                # ``main.py:_dispatch_multi_account`` (which also forwards to
                # paper_capture when ``self._paper_capture`` is enabled — it's
                # the *production paper-mode ingestion point*). That opened a
                # SECOND paper_capture position per signal — silently doubling
                # every paper-trade entry in production paper mode. Same bug
                # existed in the pullback path (next patch below).
                # Fix: skip the multi-account dispatch entirely when paper_capture
                # is active. The legacy ``_dispatch_multi_account_live`` path
                # is already disabled in paper mode (the wrapper returns early
                # at main.py:1009), so there's nothing left to dispatch — the
                # only role ``_dispatch_multi_account`` used to play was
                # re-feeding signals to paper_capture (the duplicate call).
                paper_capture_active = self._is_paper_capture_active()
                if not paper_capture_active and hasattr(self._prometheus, "_dispatch_multi_account"):
                    try:
                        self._prometheus._dispatch_multi_account(
                            exec_dict,
                            is_intraday=(executable.bar_interval != "day"),
                            bar_interval=executable.bar_interval
                        )
                    except Exception as mae:
                        logger.error(f"LiveScanner: multi-account dispatch failed: {mae}")

                if position:
                    result.executed = True
                    self._notifier.notify_execution_result(executable, position)

                    # Set up trailing stop
                    try:
                        from prometheus.config import get as cfg_get
                        intraday_cfg = cfg_get('intraday', {})
                        time_stop_bars = int(intraday_cfg.get('time_stop_bars_15min', 16))
                        ts = self._prometheus.order_manager.create_trailing_state(
                            position.position_id
                        )
                        if ts:
                            ts.trade_mode = 'swing'
                            ts.bar_interval = '15minute'
                            ts.max_bars = time_stop_bars
                            if hasattr(self._prometheus, 'position_monitor'):
                                self._prometheus.position_monitor.add_position(ts)
                    except Exception as e:
                        logger.warning(f"LiveScanner: trailing setup failed: {e}")

                    # Dispatch to multi-account traders (only when paper_capture
                    # is NOT active — see ``paper_capture_active`` comment above
                    # for the duplicate-ingestion rationale).
                    if not paper_capture_active:
                        self._dispatch_multi_account(exec_dict, executable)
                else:
                    error = getattr(
                        self._prometheus.order_manager, 'last_execution_error', ''
                    ) or 'Rejected by order manager'
                    result.execution_error = error
                    self._gate.undo_pass(symbol, executable.direction)
                    self._notifier.notify_execution_result(executable, None, error)
            except Exception as e:
                result.execution_error = str(e)
                self._gate.undo_pass(symbol, executable.direction)
                logger.error(f"LiveScanner: execution error for {symbol}: {e}")
            
            results.append(result)
        
        cycle = ScanCycle(results=results)
        self._notifier.notify_scan_result(cycle)
        self._last_scan_time = datetime.now()
        
        # Update option prices for open positions using Angel One live LTP
        self._update_position_prices()
        
        total_elapsed = time.monotonic() - scan_start
        logger.info(
            f"LiveScanner: {cycle.summary()} | "
            f"Total: {total_elapsed:.1f}s (fetch: {fetch_elapsed:.1f}s, "
            f"eval: {eval_elapsed:.1f}s)"
        )
        return cycle
    
    # ------------------------------------------------------------------
    # Pullback Entry: Pending Signal Fill Logic
    # ------------------------------------------------------------------
    
    def _try_fill_pending_signals(self, results: List[SymbolScanResult]):
        """Try to fill pending pullback signals on this scan cycle.
        
        For each pending signal:
        1. Fetch current spot price
        2. Check if spot has pulled back to limit_spot
        3. If yes: re-price premium via delta, execute
        4. If max wait exceeded: expire the pending signal
        """
        expired = []
        filled = []
        
        for symbol, pending in list(self._pending_signals.items()):
            self._pending_scans_waited[symbol] = self._pending_scans_waited.get(symbol, 0) + 1
            scans_waited = self._pending_scans_waited[symbol]
            
            executable = pending['executable']
            exec_dict = pending['exec_dict']
            limit_spot = pending['limit_spot']
            signal_spot = pending['signal_spot']
            
            # Get current spot
            current_spot = self._get_underlying_spot(symbol)
            if current_spot <= 0:
                logger.debug(f"LiveScanner: can't check pullback for {symbol}, no spot data")
                continue
            
            # Check if pullback condition is met
            is_filled = False
            if executable.direction == 'bullish' and current_spot <= limit_spot:
                is_filled = True
            elif executable.direction == 'bearish' and current_spot >= limit_spot:
                is_filled = True
            
            if is_filled:
                # Re-price premium using delta approximation (matches engine.py)
                delta = executable.raw.get('delta', 0.5) if executable.raw else 0.5
                if executable.direction == 'bearish':
                    delta = -abs(delta)
                old_premium = executable.entry_price
                spot_diff = current_spot - signal_spot
                new_premium = old_premium + delta * spot_diff
                new_premium = max(new_premium, old_premium * 0.5)
                new_premium = max(new_premium, 1.0)
                
                # Update executable with pullback-adjusted premium
                executable.entry_price = new_premium
                if executable.raw:
                    executable.raw['entry_price'] = new_premium
                    executable.raw['pullback_fill_spot'] = current_spot
                    executable.raw['entry_type'] = 'pullback_limit'
                exec_dict['entry_price'] = new_premium
                
                logger.info(
                    f"LiveScanner: {symbol} pullback FILLED — "
                    f"spot={current_spot:.1f} <= limit={limit_spot:.1f}, "
                    f"premium {old_premium:.2f} → {new_premium:.2f}"
                )
                
                # Execute the trade
                try:
                    live_premium = pending.get('live_premium')
                    self._feed_real_premium_to_broker(executable, live_premium)

                    # Bug #2 (2026-07-22): route via paper_capture helper when active.
                    position = self._route_paper_capture_or_legacy(
                        exec_dict, confirm=False
                    )

                    # Bug D.4 (2026-07-25 audit dispatch): same guard as the
                    # immediate-execution path above — when paper_capture is
                    # active, ``_route_paper_capture_or_legacy`` already opened
                    # a paper position via ``on_signal``. Calling
                    # ``self._prometheus._dispatch_multi_account`` here would
                    # re-enter ``paper_capture.on_signal`` (the wrapper at
                    # ``main.py:_dispatch_multi_account`` forwards to paper_capture
                    # when ``self._paper_capture`` is enabled). The earlier
                    # comment at line ~668 only suppressed the local
                    # ``self._dispatch_multi_account(exec_dict, executable)`` call,
                    # missing the upstream wrapper call below that ALSO hits
                    # paper_capture. Both must be suppressed.
                    paper_capture_active = self._is_paper_capture_active()
                    if not paper_capture_active and hasattr(self._prometheus, "_dispatch_multi_account"):
                        try:
                            self._prometheus._dispatch_multi_account(
                                exec_dict,
                                is_intraday=(executable.bar_interval != "day"),
                                bar_interval=executable.bar_interval
                            )
                        except Exception as mae:
                            logger.error(f"LiveScanner: multi-account dispatch failed: {mae}")

                    result = pending.get('result', SymbolScanResult(symbol=symbol))
                    if position:
                        result.executed = True
                        self._notifier.notify_execution_result(executable, position)

                        # Set up trailing stop
                        try:
                            from prometheus.config import get as cfg_get
                            intraday_cfg = cfg_get('intraday', {})
                            time_stop_bars = int(intraday_cfg.get('time_stop_bars_15min', 16))
                            ts = self._prometheus.order_manager.create_trailing_state(
                                position.position_id
                            )
                            if ts:
                                ts.trade_mode = 'swing'
                                ts.bar_interval = '15minute'
                                ts.max_bars = time_stop_bars
                                if hasattr(self._prometheus, 'position_monitor'):
                                    self._prometheus.position_monitor.add_position(ts)
                        except Exception as e:
                            logger.warning(f"LiveScanner: trailing setup failed: {e}")
                        # NOTE: multi-account dispatch already occurred above
                        # (self._prometheus._dispatch_multi_account) when
                        # paper_capture is NOT active. Do NOT re-dispatch here.
                    else:
                        self._gate.undo_pass(symbol, executable.direction)
                except Exception as e:
                    self._gate.undo_pass(symbol, executable.direction)
                    logger.error(f"LiveScanner: pullback execution error for {symbol}: {e}")
                
                filled.append(symbol)
            
            elif scans_waited >= self._pullback_max_wait_scans:
                # Signal expired without pullback
                logger.info(
                    f"LiveScanner: {symbol} pullback EXPIRED — "
                    f"waited {scans_waited} scans, "
                    f"spot={current_spot:.1f} never reached limit={limit_spot:.1f}"
                )
                self._gate.undo_pass(symbol, executable.direction)
                expired.append(symbol)
        
        # Clean up filled and expired signals
        for symbol in filled + expired:
            self._pending_signals.pop(symbol, None)
            self._pending_scans_waited.pop(symbol, None)
    
    # ------------------------------------------------------------------
    # Multi-Account Dispatch
    # ------------------------------------------------------------------
    
    def _dispatch_multi_account(self, exec_dict: dict, executable):
        """Send trade to all sub-accounts (capital-appropriate sizing)."""
        multi = getattr(self._prometheus, 'multi_account', None)
        if not multi:
            return
        
        try:
            multi.dispatch_signal(exec_dict)
            logger.info(
                f"LiveScanner: dispatched {executable.symbol} {executable.action} "
                f"to {len(multi.stacks)} sub-accounts"
            )
        except Exception as e:
            logger.warning(f"LiveScanner: multi-account dispatch failed: {e}")
    
    # ------------------------------------------------------------------
    # Live premium from Angel One
    # ------------------------------------------------------------------
    
    def _get_option_chain_client(self):
        """Get AngelOneOptionChain if available (returns None for mocks)."""
        try:
            from prometheus.data.angelone_options import AngelOneOptionChain
            data = getattr(self._prometheus, 'data', None)
            if data:
                client = getattr(data, 'angelone_options', None)
                if isinstance(client, AngelOneOptionChain):
                    return client
        except ImportError:
            pass
        return None
    
    def _fetch_live_premium(self, executable) -> dict:
        """Fetch live option premium from Angel One for a signal."""
        client = self._get_option_chain_client()
        if not client:
            return None
        
        try:
            symbol = executable.symbol
            strike = executable.strike
            opt_type = executable.option_type or "CE"
            expiry = executable.expiry
            spot = executable.raw.get('spot_at_signal', 0) if executable.raw else 0
            
            if not strike or strike <= 0:
                return None
            
            if hasattr(executable, 'instrument') and executable.instrument and hasattr(client, "_parse_tradingsymbol"):
                underlying = client.UNDERLYING_MAP.get(symbol, "NIFTY")
                parsed = client._parse_tradingsymbol(executable.instrument, underlying)
                if parsed and parsed.get("expiry_str"):
                    expiry = parsed["expiry_str"]

            if not expiry or expiry.upper() == "WEEKLY":
                expiry = None
            
            result = client.get_real_premium(
                symbol, strike, opt_type, expiry, spot_price=spot
            )
            return result
        except Exception as e:
            logger.debug(f"LiveScanner: live premium fetch failed: {e}")
            return None
    
    def _feed_real_premium_to_broker(self, executable, live_premium):
        """Feed live premium to PaperTrader for realistic fill prices."""
        if not live_premium or live_premium.get('ltp', 0) <= 0:
            return
        
        try:
            from prometheus.execution.paper_trader import PaperTrader
            broker = getattr(self._prometheus, 'broker', None)
            instrument = executable.instrument or ''
            
            if isinstance(broker, PaperTrader) and instrument:
                broker.set_real_premium(
                    instrument,
                    ltp=live_premium['ltp'],
                    bid=live_premium.get('bid', 0),
                    ask=live_premium.get('ask', 0),
                )
            
            multi = getattr(self._prometheus, 'multi_account', None)
            if multi and instrument:
                for stack in multi.stacks.values():
                    if hasattr(stack, 'trader'):
                        stack.trader.set_real_premium(
                            instrument,
                            ltp=live_premium['ltp'],
                            bid=live_premium.get('bid', 0),
                            ask=live_premium.get('ask', 0),
                        )
        except Exception as e:
            logger.debug(f"LiveScanner: feed premium to broker failed: {e}")
    
    # ------------------------------------------------------------------
    # Position Price Updates (Angel One live LTP + BS fallback)
    # ------------------------------------------------------------------
    
    def _update_position_prices(self):
        """Refresh option LTPs for open positions via Angel One."""
        try:
            p = self._prometheus
            if hasattr(p, '_refresh_open_paper_prices'):
                p._refresh_open_paper_prices()
                return
            
            pm = getattr(p, 'position_monitor', None)
            broker = getattr(p, 'broker', None)
            client = self._get_option_chain_client()
            
            if not pm or not broker or pm.active_count == 0:
                return
            
            if not client:
                self._update_position_prices_bs()
                return
            
            positions = pm.get_positions()
            price_updates = {}
            
            for pid, state in positions.items():
                try:
                    ltp = client.get_ltp_by_token(state.tradingsymbol)
                    if ltp and ltp > 0:
                        price_updates[state.tradingsymbol] = ltp
                except Exception as e:
                    logger.debug(f"LiveScanner: LTP fetch error for {pid}: {e}")
            
            if price_updates and hasattr(broker, 'update_prices'):
                broker.update_prices(price_updates)
                logger.info(
                    f"LiveScanner: updated {len(price_updates)} option prices "
                    f"via Angel One live LTP"
                )
        except Exception as e:
            logger.debug(f"LiveScanner: position price update failed: {e}")
    
    def _update_position_prices_bs(self):
        """Last-resort BS pricing when Angel One is unavailable."""
        # Bug B.3 (2026-07-25 audit): the import below referenced
        # ``prometheus.signals.option_pricing`` which doesn't exist in the
        # source tree (only ``prometheus.utils.options_math`` does). The
        # ``except Exception: pass`` swallow at the outer try-block (line
        # below) caught the ``ModuleNotFoundError`` every scan cycle and
        # silently no-op'd this whole method. Net effect: when Angel One
        # option-chain polling failed (auth cooldown, network outage, API
        # outage), the live scanner's BS-theoretical fallback path was
        # dead code — ``broker.update_prices`` never received fresh
        # estimates, ``PositionMonitor`` polled stale LTPs, and trailing
        # stops went into a "blind window" until Angel One recovered (or
        # were left at the last-known price indefinitely). Fix: import
        # from the correct path; also surface import/evaluate failures
        # with a warning instead of swallowing them so any future
        # dead-import regressions are noisy.
        try:
            pm = getattr(self._prometheus, 'position_monitor', None)
            broker = getattr(self._prometheus, 'broker', None)
            if not pm or not broker or pm.active_count == 0:
                return

            from prometheus.utils.options_math import black_scholes_price
            from prometheus.utils.indian_market import days_to_expiry
            
            positions = pm.get_positions()
            price_updates = {}
            
            for pid, state in positions.items():
                # Bug B.3 (continued): don't swallow per-position exceptions
                # either — log them so future parsing / BS-evaluation
                # regressions surface in the operator's logs.
                try:
                    spot = self._get_underlying_spot(state.symbol)
                    strike = self._parse_strike(state.tradingsymbol)
                    opt_type = "CE" if state.tradingsymbol.endswith("CE") else "PE"
                    if spot <= 0 or strike <= 0:
                        continue
                    dte = max(1, days_to_expiry(state.symbol))
                    T = float(dte) / 252.0
                    premium = black_scholes_price(spot, strike, T, 0.065, 0.15, opt_type)
                    if premium and premium > 0 and premium == premium:
                        price_updates[state.tradingsymbol] = float(premium)
                except Exception as inner:
                    logger.warning(
                        f"LiveScanner: BS-fallback update failed for "
                        f"pid={pid} sym={state.symbol} tm={state.tradingsymbol}: {inner}"
                    )
            
            if price_updates and hasattr(broker, 'update_prices'):
                broker.update_prices(price_updates)
                logger.info(
                    f"LiveScanner: updated {len(price_updates)} prices "
                    f"via BS fallback (Angel One unavailable)"
                )
        # Bug B.3 (continued): the original ``except: pass`` swallowed
        # every failure silently — including the ``ModuleNotFoundError``
        # that masked the broken import path for weeks. Promote to a
        # warning-level log so any future dead-import / API outage
        # regression is operator-visible.
        except Exception as e:
            logger.warning(f"LiveScanner: BS fallback pricing failed: {e}")
    
    def _get_underlying_spot(self, symbol: str) -> float:
        """Get latest spot price for a symbol."""
        try:
            if symbol in self._evaluators:
                evaluator = self._evaluators[symbol]
                if hasattr(evaluator, '_last_close') and evaluator._last_close:
                    return float(evaluator._last_close)
            scan_data = self._bridge.fetch_scan_data(symbol)
            if not scan_data.primary.empty:
                return float(scan_data.primary['close'].iloc[-1])
        except Exception:
            pass
        return 0.0
    
    def _parse_strike(self, tradingsymbol: str) -> float:
        """Parse strike price from tradingsymbol."""
        try:
            if tradingsymbol.endswith('CE') or tradingsymbol.endswith('PE'):
                body = tradingsymbol[:-2]
            else:
                return 0.0
            digits = ''
            for ch in reversed(body):
                if ch.isdigit():
                    digits = ch + digits
                else:
                    break
            return float(digits) if digits else 0.0
        except Exception:
            return 0.0
    
    # ------------------------------------------------------------------
    # Inter-Scan Price Refresh Thread
    # ------------------------------------------------------------------
    
    def _start_price_refresh_thread(self):
        """Start background thread that refreshes position prices every 30s."""
        if self._price_refresh_thread and self._price_refresh_thread.is_alive():
            return
        
        self._price_refresh_stop.clear()
        self._price_refresh_thread = threading.Thread(
            target=self._price_refresh_loop,
            name='price-refresh',
            daemon=True,
        )
        self._price_refresh_thread.start()
        logger.info(
            f"LiveScanner: price refresh thread started "
            f"(every {self._price_refresh_seconds}s)"
        )
    
    def _stop_price_refresh_thread(self):
        """Stop the background price refresh thread."""
        self._price_refresh_stop.set()
        if self._price_refresh_thread:
            self._price_refresh_thread.join(timeout=5)
    
    def _price_refresh_loop(self):
        """Background loop: refresh open position prices between scans."""
        while not self._price_refresh_stop.is_set():
            self._price_refresh_stop.wait(self._price_refresh_seconds)
            if self._price_refresh_stop.is_set():
                break
            
            # Don't refresh during a scan (scan_lock is held)
            if self.scan_lock.locked():
                continue
            
            try:
                if is_market_open_now():
                    self._update_position_prices()
            except Exception as e:
                logger.debug(f"LiveScanner: price refresh error: {e}")
    
    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    
    def run_loop(self):
        """Main loop: scan every 15 minutes during market hours."""
        self._running = True
        logger.info(
            f"LiveScanner: starting loop — {len(self._symbols)} symbols, "
            f"scan every {self._scan_interval}s, "
            f"price refresh every {self._price_refresh_seconds}s"
        )
        
        # Start background threads
        self._start_price_refresh_thread()
        if self._oi_cache:
            self._oi_cache.start()
            logger.info("LiveScanner: OI cache thread started")
        
        while self._running:
            try:
                now = datetime.now()
                
                if not is_trading_day_today():
                    logger.debug("LiveScanner: market holiday")
                    time.sleep(60)
                    continue
                
                current_time = now.time()
                
                # Pre-market
                if current_time < dtime(9, 15):
                    time.sleep(60)
                    continue
                
                # Skip opening noise
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                if (now - market_open).total_seconds() < self._skip_first_minutes * 60:
                    time.sleep(30)
                    continue
                
                # Market hours
                if is_market_open_now():
                    if (self._last_scan_time and 
                        (now - self._last_scan_time).total_seconds() < self._scan_interval):
                        time.sleep(10)
                        continue
                    
                    self.run_scan_cycle()
                    time.sleep(5)
                    continue
                
                # After hours
                time.sleep(60)
                
            except KeyboardInterrupt:
                self._running = False
                logger.info("LiveScanner: stopped by user")
                break
            except Exception as e:
                logger.error(f"LiveScanner: error in main loop: {e}")
                time.sleep(30)
        
        self._stop_price_refresh_thread()
        if self._oi_cache:
            self._oi_cache.stop()
    
    def stop(self):
        """Stop the scan loop, price refresh, and OI cache threads."""
        self._running = False
        self._stop_price_refresh_thread()
        if self._oi_cache:
            self._oi_cache.stop()
