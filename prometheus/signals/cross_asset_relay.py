# ============================================================================
# PROMETHEUS APEX — Cross-Asset Relay Filter
# ============================================================================
"""
Prevents taking 3 highly correlated positions simultaneously across instruments.
Ensures trade frequency scales intelligently.
"""

from typing import Dict

class CrossAssetRelay:
    def __init__(self, max_correlated_exposure: int = 2):
        self.max_correlated = max_correlated_exposure
        # Keeps track of current active positions: symbol -> direction (1 for Long, -1 for Short)
        self.active_positions: Dict[str, int] = {}
        
    def sync_portfolio(self, current_positions: list):
        """
        Synchronize state with OrderManager/Live execution.
        current_positions = [{"symbol": "NIFTY 50", "direction": 1}, ...]
        """
        self.active_positions.clear()
        for pos in current_positions:
            self.active_positions[pos["symbol"]] = pos["direction"]

    def can_take_signal(self, symbol: str, signal_direction: int, edge_score: int) -> tuple[bool, str]:
        """
        Filters correlated signals unless extreme conviction is present.
        """
        if symbol in self.active_positions:
            return False, "ALREADY_ACTIVE_IN_SYMBOL"
            
        correlated_count = 0
        for active_sym, active_dir in self.active_positions.items():
            if active_dir == signal_direction:
                correlated_count += 1
                
        if correlated_count >= self.max_correlated:
            # We are maxed out on this direction. 
            # Only bypass if the signal is EXPLOSIVE tier (85+ conviction).
            if edge_score >= 85:
                # Upgrades allocation through sheer statistical gravity edge
                return True, "BYPASSED_VIA_EXPLOSIVE_CONVICTION"
            else:
                return False, "CORRELATION_CAP_REACHED"
                
        return True, "SAFE_EXPOSURE"

