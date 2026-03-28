# ============================================================================
# PROMETHEUS APEX — Strike Gravity Mapping
# ============================================================================
"""
Translates options Open Interest into mechanical resistance/attraction physics.

If an intense concentration of OI sits directly in the path of a directional signal, 
delta-hedging inherently slows price momentum. We represent this as a "gravity penalty" 
to suppress the AES Conviction Score before allocating.
"""

from typing import Dict, List, Tuple
import numpy as np

class StrikeGravityMapper:
    def __init__(self, wall_threshold_mult: float = 1.5, scan_range_atr: float = 1.0):
        self.wall_threshold_mult = wall_threshold_mult
        self.scan_range_atr = scan_range_atr

    def calculate_gravity_penalty(
        self,
        current_price: float,
        target_price: float,
        atr: float,
        option_chain_oi: Dict[float, Dict[str, float]]  # { strike: {"CE_OI": 1M, "PE_OI": 500k} }
    ) -> float:
        """
        Evaluate if an OI wall stands between the entry and the target.
        Returns a penalty 0.0 (clear path) to 1.0 (blocked).
        
        Args:
            current_price: Execution price
            target_price: Planned take-profit price 
            atr: Volatility distance (determines scan band)
            option_chain_oi: Strike dictionary with Call/Put Open Interest
        """
        if not option_chain_oi:
            return 0.0

        direction = 1 if target_price > current_price else -1
        
        # Determine relevant scan range
        distance_to_target = abs(target_price - current_price)
        max_scan_distance = min(distance_to_target, atr * self.scan_range_atr)
        
        relevant_strikes = []
        ce_oi_total = 0
        pe_oi_total = 0
        
        # 1. Filter out strikes outside our path
        for strike, oi_data in option_chain_oi.items():
            if direction == 1 and current_price < strike <= (current_price + max_scan_distance):
                relevant_strikes.append((strike, oi_data.get("CE_OI", 0), "CE"))
            elif direction == -1 and current_price > strike >= (current_price - max_scan_distance):
                relevant_strikes.append((strike, oi_data.get("PE_OI", 0), "PE"))
                
            ce_oi_total += oi_data.get("CE_OI", 0)
            pe_oi_total += oi_data.get("PE_OI", 0)
            
        if not relevant_strikes or len(option_chain_oi) == 0:
            return 0.0

        # Baseline average OI across the entire chain to find "walls"
        avg_ce_oi = ce_oi_total / len(option_chain_oi) if len(option_chain_oi) > 0 else 1
        avg_pe_oi = pe_oi_total / len(option_chain_oi) if len(option_chain_oi) > 0 else 1
        
        max_penalty = 0.0
        
        # 2. Look for the defining wall in our vector path
        for strike, oi, type_ in relevant_strikes:
            avg_oi_ref = avg_ce_oi if type_ == "CE" else avg_pe_oi
            if avg_oi_ref == 0: continue
            
            relative_size = oi / avg_oi_ref
            
            # If the OI at this strike is X times the average, it is a wall
            if relative_size > self.wall_threshold_mult:
                # The penalty scales with the size of the wall
                base_penalty = (relative_size - self.wall_threshold_mult) * 0.2
                
                # The closer the wall is to the current price, the more immediate the gravity
                distance_ratio = abs(strike - current_price) / max_scan_distance
                proximity_weight = 1.0 - (distance_ratio * 0.5) # Closer = heavier penalty
                
                wall_penalty = min(1.0, base_penalty * proximity_weight)
                
                # Take the worst penalty in our path
                max_penalty = max(max_penalty, wall_penalty)

        # 3. Cap max penalty at 1.0 
        return float(min(1.0, max_penalty))

