"""LiveScanner: Orchestrates scan cycles for paper/live trading."""

import time
from datetime import datetime, time as dtime, date
from typing import Dict, List, Optional

from prometheus.pipeline.types import (
    ScanCycle, SymbolScanResult, DataStatus,
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
    # NSE: 9:15 - 15:30 IST on trading days
    if t < dtime(9, 15) or t >= dtime(15, 30):
        return False
    # Basic weekday check (Mon-Fri)
    if now.weekday() >= 5:
        return False
    return True


def is_trading_day_today() -> bool:
    """Check if today is a trading day."""
    try:
        from prometheus.utils.calendar import is_trading_day
        return is_trading_day(date.today())
    except ImportError:
        # Fallback: Mon-Fri
        return datetime.now().weekday() < 5


class LiveScanner:
    """Main scan orchestrator for paper/live trading.
    
    Creates persistent signal evaluators (one per symbol) and runs
    periodic scan cycles during market hours.
    """
    
    def __init__(
        self,
        prometheus_instance,
        symbols: List[str],
        scan_interval_seconds: int = 900,
        skip_first_minutes: int = 15,
        max_positions: int = 3,
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
        
        # Lock to prevent /scan command and auto-scan from running simultaneously
        import threading
        self.scan_lock = threading.Lock()
    
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
            logger.info(f"LiveScanner: daily refresh for {today}")
    
    def run_scan_cycle(self) -> ScanCycle:
        """Run one scan cycle across all symbols.
        
        This is the core method. It:
        1. Fetches data for each symbol
        2. Evaluates signals
        3. Converts to executable format
        4. Checks execution gates
        5. Executes if passed
        6. Notifies via Telegram
        
        Returns a ScanCycle with full diagnostic trail.
        Thread-safe: uses scan_lock to prevent concurrent scans.
        """
        if not self.scan_lock.acquire(blocking=False):
            logger.info("LiveScanner: scan skipped — another scan in progress")
            return ScanCycle(results=[])
        try:
            return self._run_scan_cycle_locked()
        finally:
            self.scan_lock.release()
    
    def _run_scan_cycle_locked(self) -> ScanCycle:
        """Internal scan cycle (must be called with scan_lock held)."""
        self._ensure_evaluators()
        self._refresh_evaluators_daily()
        
        # Update position count for gate checks
        try:
            open_count = len([
                m for m in self._prometheus.order_manager.managed_positions.values()
                if m.status == 'open'
            ])
            self._gate.update_positions(open_count)
        except Exception:
            pass
        
        results: List[SymbolScanResult] = []
        
        for symbol in self._symbols:
            result = SymbolScanResult(symbol=symbol)
            
            # Step 1: Fetch data
            scan_data = self._bridge.fetch_scan_data(symbol)
            result.data_status = scan_data.status
            result.data_error = scan_data.error_message
            
            if scan_data.status not in (DataStatus.OK, DataStatus.STALE):
                results.append(result)
                continue
            
            # Step 2: Evaluate signal
            evaluator = self._evaluators[symbol]
            signal_result = evaluator.evaluate(scan_data)
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
            
            # Step 3b: Confidence gate — reject weak signals
            min_confidence = 0.35
            if executable.confidence < min_confidence:
                logger.info(
                    f"LiveScanner: {symbol} rejected — confidence "
                    f"{executable.confidence:.0%} < {min_confidence:.0%}"
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
                # Override BS-estimated prices with real market prices
                old_entry = executable.entry_price
                executable.entry_price = real_ltp
                # Scale SL and target proportionally to maintain RR ratio
                if old_entry > 0:
                    ratio = real_ltp / old_entry
                    executable.stop_loss = round(executable.stop_loss * ratio, 2)
                    executable.target = round(executable.target * ratio, 2)
                # Update raw dict too
                if executable.raw:
                    executable.raw['entry_price'] = real_ltp
                    executable.raw['live_premium'] = real_ltp
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
                # Merge ExecutableSignal fields into raw dict for order_manager
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
                
                position = self._prometheus.order_manager.execute_signal(
                    exec_dict, confirm=False
                )
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
                else:
                    error = getattr(
                        self._prometheus.order_manager, 'last_execution_error', ''
                    ) or 'Rejected by order manager'
                    result.execution_error = error
                    self._gate.undo_pass(symbol)  # Don't block future retries
                    self._notifier.notify_execution_result(executable, None, error)
            except Exception as e:
                result.execution_error = str(e)
                self._gate.undo_pass(symbol)
                logger.error(f"LiveScanner: execution error for {symbol}: {e}")
            
            results.append(result)
        
        cycle = ScanCycle(results=results)
        self._notifier.notify_scan_result(cycle)
        self._last_scan_time = datetime.now()
        
        # Update option prices for open positions using Angel One live LTP
        self._update_position_prices()
        
        logger.info(f"LiveScanner: {cycle.summary()}")
        return cycle
    
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
        """Fetch live option premium from Angel One for a signal.
        
        Returns dict with {ltp, bid, ask, ...} or None if unavailable.
        This is called BEFORE notification so the entry price shown
        to the user is the REAL market price, not a BS estimate.
        """
        client = self._get_option_chain_client()
        if not client:
            return None
        
        try:
            symbol = executable.symbol  # e.g., "NIFTY 50"
            strike = executable.strike
            opt_type = executable.option_type or "CE"
            expiry = executable.expiry
            spot = executable.raw.get('spot_at_signal', 0) if executable.raw else 0
            
            if not strike or strike <= 0:
                return None
            
            # Pass None for expiry to get nearest
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
            
            # Also feed to multi-account traders
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
    
    def _update_position_prices(self):
        """Refresh option LTPs for open positions via Angel One.
        
        Falls back to the existing _refresh_open_paper_prices() on the
        Prometheus instance, which already handles Angel One LTP fetch
        for open paper positions. If that's unavailable, falls back to
        Black-Scholes as a last resort.
        """
        try:
            p = self._prometheus
            
            # Primary: use existing Angel One live refresh
            if hasattr(p, '_refresh_open_paper_prices'):
                p._refresh_open_paper_prices()
                return
            
            # Fallback: direct Angel One LTP fetch
            pm = getattr(p, 'position_monitor', None)
            broker = getattr(p, 'broker', None)
            client = self._get_option_chain_client()
            
            if not pm or not broker or pm.active_count == 0:
                return
            
            if not client:
                # Last resort: BS pricing (known to be ~27% off)
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
        try:
            pm = getattr(self._prometheus, 'position_monitor', None)
            broker = getattr(self._prometheus, 'broker', None)
            if not pm or not broker or pm.active_count == 0:
                return
            
            from prometheus.signals.option_pricing import black_scholes_price
            from prometheus.utils.indian_market import days_to_expiry
            
            positions = pm.get_positions()
            price_updates = {}
            
            for pid, state in positions.items():
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
                except Exception:
                    pass
            
            if price_updates and hasattr(broker, 'update_prices'):
                broker.update_prices(price_updates)
                logger.info(
                    f"LiveScanner: updated {len(price_updates)} prices "
                    f"via BS fallback (Angel One unavailable)"
                )
        except Exception:
            pass
    
    def _get_underlying_spot(self, symbol: str) -> float:
        """Get latest spot price for a symbol from cached scan data."""
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
        """Parse strike price from tradingsymbol (e.g. BANKNIFTY2662357600CE -> 57600)."""
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
    
    def run_loop(self):
        """Main loop: scan every 15 minutes during market hours.
        
        Handles pre-market, opening noise skip, market hours, and after hours.
        """
        self._running = True
        logger.info(
            f"LiveScanner: starting loop — {len(self._symbols)} symbols, "
            f"scan every {self._scan_interval}s"
        )
        
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
    
    def stop(self):
        """Stop the scan loop."""
        self._running = False
