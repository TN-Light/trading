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
            
            # Step 5: Execute
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
        
        # Update option prices for open positions using latest spot data
        self._update_position_prices()
        
        logger.info(f"LiveScanner: {cycle.summary()}")
        return cycle
    
    def _update_position_prices(self):
        """Re-price open option positions using Black-Scholes.
        
        In paper mode, we can't get real option LTP from the broker.
        Instead, we use the underlying spot price (which we CAN get from
        yfinance/AngelOne) to estimate the option premium via Black-Scholes.
        This feeds the position monitor's trailing stop system.
        """
        try:
            pm = getattr(self._prometheus, 'position_monitor', None)
            broker = getattr(self._prometheus, 'broker', None)
            if not pm or not broker or pm.active_count == 0:
                return
            
            from prometheus.signals.option_pricing import black_scholes_price
            from prometheus.utils.indian_market import (
                get_expiry_date, days_to_expiry
            )
            
            positions = pm.get_positions()
            price_updates = {}
            
            for pid, state in positions.items():
                try:
                    symbol = state.symbol  # e.g., "NIFTY BANK"
                    tsym = state.tradingsymbol  # e.g., "BANKNIFTY2662357600CE"
                    
                    # Get latest spot price for the underlying
                    spot = self._get_underlying_spot(symbol)
                    if spot <= 0:
                        continue
                    
                    # Parse strike and option type from tradingsymbol
                    strike = self._parse_strike(tsym)
                    opt_type = "CE" if tsym.endswith("CE") else "PE"
                    
                    if strike <= 0:
                        continue
                    
                    # Calculate time to expiry
                    dte = max(1, days_to_expiry(symbol))
                    T = float(dte) / 365.0
                    
                    # Implied volatility estimate (rough but functional)
                    sigma = 0.15  # conservative default
                    r = 0.065
                    
                    # Calculate theoretical premium
                    premium = black_scholes_price(spot, strike, T, r, sigma, opt_type)
                    if premium and premium > 0 and not (premium != premium):  # NaN check
                        price_updates[tsym] = float(premium)
                        
                except Exception as e:
                    logger.debug(f"LiveScanner: price update error for {pid}: {e}")
            
            if price_updates and hasattr(broker, 'update_prices'):
                broker.update_prices(price_updates)
                logger.info(
                    f"LiveScanner: updated {len(price_updates)} option prices "
                    f"via Black-Scholes"
                )
                
        except Exception as e:
            logger.debug(f"LiveScanner: position price update failed: {e}")
    
    def _get_underlying_spot(self, symbol: str) -> float:
        """Get latest spot price for a symbol from cached scan data."""
        try:
            if symbol in self._evaluators:
                evaluator = self._evaluators[symbol]
                # The evaluator's last scan data has the current price
                if hasattr(evaluator, '_last_close') and evaluator._last_close:
                    return float(evaluator._last_close)
            
            # Fallback: fetch fresh data
            scan_data = self._bridge.fetch_scan_data(symbol)
            if not scan_data.primary.empty:
                return float(scan_data.primary['close'].iloc[-1])
        except Exception:
            pass
        return 0.0
    
    def _parse_strike(self, tradingsymbol: str) -> float:
        """Parse strike price from Kite-format tradingsymbol.
        
        Examples:
            BANKNIFTY2662357600CE -> 57600
            NIFTY2662323800CE -> 23800
            SENSEX26JUN77000CE -> 77000
        """
        try:
            # Remove CE/PE suffix
            if tradingsymbol.endswith('CE') or tradingsymbol.endswith('PE'):
                body = tradingsymbol[:-2]
            else:
                return 0.0
            
            # Extract trailing digits as strike
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
