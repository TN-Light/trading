# ============================================================================
# PROMETHEUS APEX — Expiry Clock Alpha
# ============================================================================
"""
Translates the structural realities of Indian weekly options (forced hedging and rolling).
Specifically targets the high-probability 9:30 AM – 10:30 AM directional flows.
"""

from datetime import datetime, time
from typing import Dict, Any
from prometheus.utils.indian_market import is_weekly_expiry_day

class ExpiryClock:
    def __init__(self):
        # Institutional flow window
        self.squeeze_start = time(9, 30)
        self.squeeze_end = time(10, 30)
        self.afternoon_hedge = time(13, 30)
        
    def evaluate_window(self, dt: datetime, symbol: str) -> Dict[str, Any]:
        """
        Determines if the current timestamp falls in a structural alpha window.
        """
        is_expiry_day = bool(is_weekly_expiry_day(symbol, dt.date()))

        current_time = dt.time()
        is_morning_squeeze = is_expiry_day and (self.squeeze_start <= current_time <= self.squeeze_end)
        is_pm_unwind = is_expiry_day and (current_time >= self.afternoon_hedge)
        
        # Multipliers that feed into the AES Fusion Engine time_decay_edge
        alpha_mult = 1.0
        if is_morning_squeeze:
            alpha_mult = 1.6  # Massive boost for directional morning flows
        elif is_pm_unwind:
            alpha_mult = 0.5  # Suppress directional edges late on expiry (burns premium)
        elif is_expiry_day:
            alpha_mult = 1.2  # General expiry edge 
            
        return {
            "is_expiry_day": is_expiry_day,
            "is_morning_squeeze": is_morning_squeeze,
            "is_pm_unwind": is_pm_unwind,
            "alpha_multiplier": alpha_mult
        }

