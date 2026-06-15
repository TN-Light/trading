"""SignalConverter: Converts raw backtest signals to executable format."""

from typing import Optional, Dict, Any
from prometheus.pipeline.types import SignalResult, ExecutableSignal
from prometheus.utils.logger import logger

# Lazy imports to avoid circular deps
def _get_lot_size(symbol: str) -> int:
    try:
        from prometheus.utils.indian_market import get_lot_size
        return int(get_lot_size(symbol) or 0)
    except Exception:
        return 0

def _generate_tradingsymbol(underlying: str, expiry: str, strike: float, option_type: str) -> str:
    try:
        from prometheus.execution.kite_executor import generate_tradingsymbol
        return generate_tradingsymbol(underlying, expiry, strike, option_type)
    except Exception as e:
        logger.debug(f"SignalConverter: generate_tradingsymbol failed: {e}")
        return ''

def _load_max_score() -> float:
    """Compute the max possible confluence score from current signal weights.
    
    This is the sum of all individual weight values (the theoretical max
    when every indicator agrees on one direction).
    """
    try:
        import json
        from pathlib import Path
        weights_path = Path(__file__).parent.parent / "config" / "regression_weights.json"
        if weights_path.exists():
            with open(weights_path, "r") as f:
                wdata = json.load(f)
            nw = wdata.get("normalized_weights", {})
            max_baseline = 1.5
            weights = {k: v * max_baseline for k, v in nw.items()}
            return sum(weights.values())
    except Exception:
        pass
    # Fallback: sum of default weights
    return 8.5  # 1.5+1.5+1.0+1.0+1.5+0.5+0.5+0.5+0.5

# Cache the max score at module load
_MAX_SCORE = _load_max_score()


class SignalConverter:
    """Converts raw signal dicts to typed ExecutableSignal objects.
    
    Logs every rejection with a specific reason.
    """
    
    def convert(self, signal_result: SignalResult, symbol: str) -> Optional[ExecutableSignal]:
        """Convert a SignalResult to an ExecutableSignal.
        
        Returns None with logged reason if conversion fails.
        """
        if not signal_result or not signal_result.has_signal:
            logger.debug(f"SignalConverter: {symbol} — no signal to convert")
            return None
        
        raw = signal_result.raw_signal
        if not raw:
            return None
        
        # Reject expiry strategies
        strategy = str(raw.get('strategy', ''))
        if strategy.startswith('expiry_'):
            logger.info(f"SignalConverter: {symbol} — rejected: expiry strategy '{strategy}'")
            return None
        
        # Validate direction
        direction = raw.get('direction', '')
        if direction not in ('bullish', 'bearish'):
            logger.info(f"SignalConverter: {symbol} — rejected: invalid direction '{direction}'")
            return None
        
        action = 'BUY_CE' if direction == 'bullish' else 'BUY_PE'
        option_type = 'CE' if direction == 'bullish' else 'PE'
        
        entry = float(raw.get('entry_price', 0) or 0)
        sl = float(raw.get('stop_loss', 0) or 0)
        target = float(raw.get('target', 0) or 0)
        risk = abs(entry - sl)
        reward = abs(target - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0.0
        
        # Confidence: use the winning side's score, normalized by max possible
        bull = float(raw.get('bull_score', 0) or 0)
        bear = float(raw.get('bear_score', 0) or 0)
        score = bull if direction == 'bullish' else bear
        confidence = min(1.0, score / _MAX_SCORE) if score > 0 else 0.0
        
        strike = float(raw.get('strike', 0) or 0)
        expiry = str(raw.get('option_expiry_date', '') or raw.get('expiry', '') or '')
        
        lot_size = int(raw.get('lot_size') or _get_lot_size(symbol) or 0)
        quantity = int(raw.get('quantity', 0) or 0)
        
        # Generate Kite-compatible tradingsymbol
        instrument = ''
        if strike > 0 and expiry:
            sym_upper = symbol.upper()
            if 'SENSEX' in sym_upper:
                underlying = 'SENSEX'
            elif 'NIFTY IT' in sym_upper or 'NIFTYIT' in sym_upper:
                underlying = 'NIFTYIT'
            elif 'BANK' in sym_upper:
                underlying = 'BANKNIFTY'
            elif 'FIN' in sym_upper:
                underlying = 'FINNIFTY'
            else:
                underlying = 'NIFTY'
            instrument = _generate_tradingsymbol(underlying, expiry, strike, option_type)
        
        executable = ExecutableSignal(
            symbol=symbol,
            action=action,
            direction=direction,
            option_type=option_type,
            entry_price=entry,
            stop_loss=sl,
            target=target,
            risk_reward=rr,
            strike=strike,
            expiry=expiry,
            lot_size=lot_size,
            quantity=quantity,
            instrument=instrument,
            confidence=confidence,
            bar_timestamp=signal_result.bar_timestamp,
            trade_mode='swing',
            regime=signal_result.regime,
            strategy=signal_result.strategy,
            reasons=signal_result.reasons,
            raw=raw,
        )
        
        logger.info(
            f"SignalConverter: {symbol} {action} — "
            f"entry={entry:.2f} sl={sl:.2f} target={target:.2f} RR=1:{rr:.1f} "
            f"strike={strike} expiry={expiry} instrument={instrument} "
            f"confidence={confidence:.0%} (score={score:.2f}/{_MAX_SCORE:.1f})"
        )
        return executable
