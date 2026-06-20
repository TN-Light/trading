"""SignalEvaluator: Persistent wrapper around the signal generator."""

from typing import Optional, Dict, Any
import pandas as pd
from prometheus.pipeline.types import ScanData, SignalResult
from prometheus.utils.logger import logger


class SignalEvaluator:
    """Wraps _make_signal_generator with persistent state.
    
    Unlike the old path which recreated the generator every scan,
    this keeps the generator alive across scans for regime state consistency.
    """
    
    def __init__(self, prometheus_instance, symbol: str, primary_interval: str = '15minute'):
        self._prometheus = prometheus_instance
        self._symbol = symbol
        self._interval = primary_interval
        self._generator = None
        self._initialized = False
        self._last_close = None  # Cache latest close for position pricing
    
    def initialize(self, scan_data: ScanData):
        """Create the signal generator using initial scan data.
        
        Called once on first scan. The generator persists across subsequent scans.
        """
        if self._initialized and self._generator is not None:
            return
        
        p = self._prometheus
        daily = scan_data.daily if not scan_data.daily.empty else scan_data.primary
        hourly = scan_data.hourly if not scan_data.hourly.empty else scan_data.primary
        
        regime_state = p.regime_detector.detect(daily) if len(daily) >= 50 else None
        
        if self._interval == 'day':
            hourly_bias_map = p._compute_daily_bias(daily)
        else:
            hourly_bias_map = p._compute_intraday_bias(hourly)
        
        capital = float(p.initial_capital)
        capital_tracker = {'capital': capital, 'peak': capital}
        
        from prometheus.config import get as cfg_get
        swing_cfg = cfg_get('swing', {})
        use_parrondo = bool(swing_cfg.get('parrondo', False))
        param_overrides = {'mr_min_score': 2.5} if self._interval != 'day' else None
        
        p.regime_detector.reset_cache()
        self._generator = p._make_signal_generator(
            regime_state=regime_state,
            hourly_bias_map=hourly_bias_map,
            capital=capital,
            primary_interval=self._interval,
            symbol=self._symbol,
            param_overrides=param_overrides,
            parrondo=use_parrondo,
            capital_tracker=capital_tracker,
        )
        self._initialized = True
        logger.info(f"SignalEvaluator: initialized for {self._symbol} ({self._interval})")
    
    def evaluate(self, scan_data: ScanData) -> SignalResult:
        """Evaluate the latest data and return a SignalResult.
        
        Always returns a SignalResult (never None). Check .has_signal for result.
        """
        symbol = self._symbol
        
        # Initialize on first call
        if not self._initialized:
            self.initialize(scan_data)
        
        if self._generator is None:
            logger.warning(f"SignalEvaluator: {symbol} — generator not initialized")
            return SignalResult(
                raw_signal=None,
                symbol=symbol,
                diagnostics={'error': 'generator_not_initialized'},
            )
        
        primary = scan_data.primary
        
        # Cache latest close for position price updates
        try:
            if not primary.empty:
                self._last_close = float(primary['close'].iloc[-1])
        except Exception:
            pass
        
        if len(primary) < 50:
            logger.info(f"SignalEvaluator: {symbol} — insufficient data ({len(primary)} bars < 50)")
            return SignalResult(
                raw_signal=None,
                symbol=symbol,
                diagnostics={'reason': 'insufficient_data', 'bars': len(primary)},
            )
        
        try:
            raw = self._generator(primary)
        except Exception as e:
            logger.error(f"SignalEvaluator: {symbol} — generator error: {e}")
            return SignalResult(
                raw_signal=None,
                symbol=symbol,
                diagnostics={'error': str(e)},
            )
        
        if raw:
            # Attach bar timestamp
            try:
                raw['bar_timestamp'] = str(primary['timestamp'].iloc[-1])
            except Exception:
                raw['bar_timestamp'] = ''
            raw['symbol'] = symbol
            
            result = SignalResult(raw_signal=raw, symbol=symbol)
            logger.info(
                f"SignalEvaluator: {symbol} — SIGNAL {result.direction} "
                f"(score={result.confluence_score:.1f}, regime={result.regime}, "
                f"reasons={result.reasons})"
            )
            return result
        
        logger.debug(f"SignalEvaluator: {symbol} — no signal (generator returned None)")
        return SignalResult(
            raw_signal=None,
            symbol=symbol,
            diagnostics={'reason': 'no_confluence'},
        )
    
    def refresh(self):
        """Force re-initialization on next evaluate() call."""
        self._generator = None
        self._initialized = False
        logger.info(f"SignalEvaluator: {self._symbol} — marked for refresh")
