# ============================================================================
# PROMETHEUS APEX — Loss Adjustment Protocol (LAP)
# ============================================================================
"""
Mathematically validates secondary trades after stop-outs. 
Distinguishes between revenge trading (blocked) and verified counter-signals (approved).
"""

from datetime import datetime, timedelta
from typing import Dict

class LapRecoveryProtocol:
    def __init__(self, cooldown_minutes: int = 120, min_reversal_score: int = 60, min_reentry_score: int = 85):
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.min_reversal_score = min_reversal_score
        self.min_reentry_score = min_reentry_score
        
        # Tracks last stop-out per symbol
        # { "NIFTY 50": {"time": dt, "direction": 1} }
        self.recorded_losses: Dict[str, dict] = {}
        
    def register_loss(self, symbol: str, direction: int, dt: datetime):
        """Record a stop-out block."""
        self.recorded_losses[symbol] = {
            "time": dt,
            "direction": direction
        }
        
    def clear_loss_state(self, symbol: str):
        if symbol in self.recorded_losses:
            del self.recorded_losses[symbol]
            
    def validate_signal(self, symbol: str, new_direction: int, dt: datetime, edge_score: int) -> tuple[bool, str]:
        """
        Calculates if a signal breaks the revenge trade heuristic or represents a valid regime pivot.
        """
        if symbol not in self.recorded_losses:
            return True, "STANDARD_ENTRY"
            
        last_loss = self.recorded_losses[symbol]
        time_since_loss = dt - last_loss["time"]
        
        if time_since_loss > self.cooldown:
            return True, "COOLDOWN_CLEARED"
            
        # Inside the cooldown envelope
        if new_direction == last_loss["direction"]:
            # Same direction as the loss. The QRD must be screaming EXPLOSIVE to allow this.
            if edge_score >= self.min_reentry_score:
                return True, "LAP_REENTRY_APPROVED"
            return False, "BLOCKED_SAME_DIR_REVENGE"
        else:
            # Opposite direction. The market shifted regimes, trapping our stop.
            # Only require STANDARD/CONVICTION tier to accept this pivot.
            if edge_score >= self.min_reversal_score:
                return True, "LAP_REVERSAL_APPROVED"
            return False, "LAP_REVERSAL_LOW_CONVICTION"


    def validate_hold_time(self, bars_held: int, max_allowed_candles: int = 4) -> bool:
        if bars_held >= max_allowed_candles:
            return False
        return True
