# ============================================================================
# PROMETHEUS APEX — Adaptive Edge Score (AES) Fusion Engine
# ============================================================================
"""
AES solves the quality-frequency trade-off by modeling Conviction as a 0-100 spectrum.
It routes QRD (Quantum Regime Detection), Technical Signal features, and Gravity Maps 
into a unified allocation score.

Tiers:
- <40:   Signal rejection
- 40-54: Micro-tier (fractional sizing)
- 55-69: Standard-tier (baseline sizing)
- 70-84: Conviction-tier (full sizing)
- 85+:   Explosive-tier (Kelly-max allocation)
"""

from typing import Dict, Any, Tuple
from .qrd_estimator import QuantumRegime
import math

class AesFusionEngine:
    def __init__(self):
        # Weights for the core pillars of Conviction
        self.weights = {
            "regime_alignment": 0.30,   # Does direction match QRD probability?
            "signal_confluence": 0.25,  # Raw count of indicator agreement
            "volatility_support": 0.15, # Are options mispriced or is vol expanding favorably?
            "gravity_clearance": 0.15,  # Is the path clear of OI walls?
            "time_decay_edge": 0.10,    # E.g., Expiry clock alpha
            "macro_flow": 0.05          # Broad market / sector alignment
        }

    def calculate_edge_score(
        self,
        signal_direction: float,    # -1.0 to 1.0 (from technical indicators)
        qrd_state: QuantumRegime,
        signal_features: Dict[str, Any],
        gravity_penalty: float = 0.0,  # 0.0 = clear path, 1.0 = total block
        is_expiry_thursday: bool = False,
        is_opening_session: bool = False
    ) -> Tuple[int, Dict[str, float]]:
        """
        Produce a deterministic 0-100 score and its decomposition vector.
        """
        components = {}
        
        # 1. Regime Alignment (0-30 points)
        # Match signal (-1/1) with QRD directional bias. Maximize points if signal is
        # perfectly aligned with dominant regime probability.
        dir_alignment = max(0.0, signal_direction * qrd_state.directional_bias)
        
        if qrd_state.dominant_regime == "TREND":
            alignment_score = dir_alignment * qrd_state.trend_prob
        elif qrd_state.dominant_regime == "REVERSION":
            # For reversion, alignment is inversely proportional to standard trend
            alignment_score = dir_alignment * qrd_state.reversion_prob
        else:
            alignment_score = dir_alignment * 0.5 # Volatile penalty
            
        components["regime_alignment"] = min(1.0, alignment_score * 1.5) * (self.weights["regime_alignment"] * 100)
        
        # 2. Signal Confluence (0-25 points)
        # Pulls from traditional PROMETHEUS indicators but treated as a continuum
        conf_score = signal_features.get("confluence_score", 0)
        # normalized: assume 5 is a max confluence
        norm_conf = min(1.0, conf_score / 5.0) 
        components["signal_confluence"] = norm_conf * (self.weights["signal_confluence"] * 100)
        
        # 3. Volatility Support (0-15 points)
        # High vol probability supports breakout/explosive signals.
        vol_support = qrd_state.volatile_prob if qrd_state.dominant_regime == "TREND" else qrd_state.factors.get("vol_score", 0.0)
        components["volatility_support"] = min(1.0, vol_support * 1.2) * (self.weights["volatility_support"] * 100)
        
        # 4. Gravity Clearance (0-15 points)
        # Gravity maps directly penalize the conviction.
        clearance = 1.0 - gravity_penalty
        components["gravity_clearance"] = clearance * (self.weights["gravity_clearance"] * 100)
        
        # 5. Time Decay / Expiry Clock Edge (0-10 points)
        time_edge = 0.0
        if is_expiry_thursday and is_opening_session:
            time_edge = 1.0  # Max score during the 9:30-10:30 AM squeeze
        elif is_expiry_thursday:
            time_edge = 0.5  # Elevated but not maximum
        else:
            time_edge = 0.2  # Base holding edge
        components["time_decay_edge"] = time_edge * (self.weights["time_decay_edge"] * 100)
        
        # 6. Macro Flow (0-5 points)
        # Baseline placeholder if cross-index relay isn't provided
        macro_edge = signal_features.get("macro_alignment", 0.5) 
        components["macro_flow"] = macro_edge * (self.weights["macro_flow"] * 100)

        # Total Aggregation
        total_score = sum(components.values())
        
        # Non-linear boost for perfect alignments (Convex Kelly Curve mapping)
        if total_score > 75:
            # Squeeze higher scores towards 100 faster
            boost = (total_score - 75) * 0.2
            total_score = min(100.0, total_score + boost)
            
        final_score = int(math.floor(total_score))
        
        return final_score, components

    def get_sizing_tier(self, edge_score: int) -> Tuple[str, float]:
        """Maps an edge score to a tier and base allocation multiplier."""
        if edge_score < 30:
            return "REJECT", 0.0
        elif 30 <= edge_score <= 45:
            return "MICRO", 0.25      # 25% sizing
        elif 46 <= edge_score <= 60:
            return "STANDARD", 0.50   # 50% sizing
        elif 61 <= edge_score <= 75:
            return "CONVICTION", 1.0  # 100% sizing
        else:
            return "EXPLOSIVE", 1.5   # 150% sizing (Kelly Max)


    def get_bounded_sizing(self, edge_score: int, available_capital: float, lot_premium: float) -> Tuple[str, int]:
        tier, allocation_multiplier = self.get_sizing_tier(edge_score)
        if tier == "REJECT":
            return "REJECT", 0
        max_affordable_lots = int(available_capital // lot_premium)
        if max_affordable_lots < 1:
            return "REJECT_INSUFFICIENT_FUNDS", 0
        if available_capital <= 25000:
            return tier, 1
        calculated_lots = int(max_affordable_lots * allocation_multiplier)
        final_lots = max(1, min(calculated_lots, max_affordable_lots))
        return tier, final_lots
