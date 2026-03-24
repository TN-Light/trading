# ============================================================================
# PROMETHEUS — Signal Engine: Signal Fusion Model
# ============================================================================
"""
Combines all signal sources into unified BUY/SELL/HOLD decisions.
Uses weighted confluence — no single indicator drives decisions alone.

Signal sources and their institutional trust weights:
  Volume Profile / VWAP:     0.85  (institutional benchmark)
  OI Analysis:               0.80  (unique to Indian F&O)
  Liquidity Sweeps:          0.70  (institutional order flow)
  Regime Detector:           0.75  (strategy selector)
  FVG / Imbalance:           0.60  (structural)
  RSI Divergence:            0.40  (secondary confluence only)
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prometheus.signals.technical import TechnicalSignal
from prometheus.signals.oi_analyzer import OISignal
from prometheus.signals.regime_detector import RegimeState, MarketRegime


@dataclass
class FusedSignal:
    """A unified signal after fusion of all signal sources."""
    timestamp: str
    symbol: str
    action: str              # "BUY_CE", "BUY_PE", "SELL_CE", "SELL_PE", "HOLD"
    direction: str           # "bullish", "bearish", "neutral"
    confidence: float        # 0 to 1 — overall confidence
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    risk_reward: float = 0.0
    regime: str = ""
    strategy: str = ""       # which strategy module should handle this
    reasoning: str = ""
    contributing_signals: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "action": self.action,
            "direction": self.direction,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "risk_reward": self.risk_reward,
            "regime": self.regime,
            "strategy": self.strategy,
            "reasoning": self.reasoning,
            "contributing_signals": self.contributing_signals,
        }


class SignalFusionEngine:
    """
    Combines multiple signal sources into actionable trading decisions.

    The fusion model uses weighted scoring:
    1. Each signal source has a weight reflecting institutional reliability
    2. Signals must pass minimum confluence threshold
    3. Regime context modulates which signals are emphasized
    4. Risk/reward is computed before any signal is emitted
    """

    # Signal source weights — higher = more trusted
    SIGNAL_WEIGHTS = {
        "volume_profile": 0.85,
        "vwap": 0.80,
        "oi_analysis": 0.80,
        "liquidity_sweep": 0.70,
        "regime": 0.75,
        "ema": 0.75,
        "fvg_imbalance": 0.60,
        "fibonacci_ote": 0.55,
        "supertrend": 0.50,
        "rsi_divergence": 0.40,
        "ai_sentiment": 0.65,
    }

    def __init__(self, min_confluence_score: float = 3.0, min_rr: float = 2.0, min_confluence: Optional[float] = None):
        """
        Args:
            min_confluence_score: Minimum confluence on 0-10 scale to emit a signal
            min_rr: Minimum risk:reward ratio required
            min_confluence: (deprecated 0-1 scale). If provided, converted to 0-10.
        """
        if min_confluence is not None:
            # backward-compatible: assume 0-1 scale if <=1, else already 0-10
            self.min_confluence_score = min_confluence * 10 if min_confluence <= 1 else min_confluence
        else:
            self.min_confluence_score = min_confluence_score
        self.min_rr = min_rr

    def fuse(
        self,
        symbol: str,
        spot_price: float,
        technical_signals: List[TechnicalSignal],
        oi_signals: List[OISignal],
        regime: RegimeState,
        ai_sentiment: Optional[Dict] = None,
        min_rr: Optional[float] = None
    ) -> Optional[FusedSignal]:
        """
        Run the fusion algorithm.

        Returns a FusedSignal if confluence is sufficient, None otherwise.
        """
        # Collect all directional votes with weights
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        contributing = []

        # Process technical signals
        for sig in technical_signals:
            weight = self.SIGNAL_WEIGHTS.get(sig.name, 0.5)
            weighted_strength = sig.strength * weight

            if sig.direction == "bullish":
                bullish_score += weighted_strength
            elif sig.direction == "bearish":
                bearish_score += weighted_strength
            total_weight += weight

            contributing.append(f"{sig.name}:{sig.direction}({sig.strength:.2f})")

        # Process OI signals
        for sig in oi_signals:
            weight = self.SIGNAL_WEIGHTS.get("oi_analysis", 0.80)
            weighted_strength = sig.strength * weight

            if sig.direction == "bullish":
                bullish_score += weighted_strength
            elif sig.direction == "bearish":
                bearish_score += weighted_strength
            total_weight += weight

            contributing.append(f"OI:{sig.signal_type}:{sig.direction}({sig.strength:.2f})")

        # Process regime signal
        regime_weight = self.SIGNAL_WEIGHTS["regime"]
        if regime.trend_strength > 0.2:
            bullish_score += abs(regime.trend_strength) * regime_weight * regime.confidence
        elif regime.trend_strength < -0.2:
            bearish_score += abs(regime.trend_strength) * regime_weight * regime.confidence
        total_weight += regime_weight
        contributing.append(f"regime:{regime.regime.value}({regime.confidence:.2f})")

        # Process AI sentiment if available
        if ai_sentiment:
            ai_weight = self.SIGNAL_WEIGHTS.get("ai_sentiment", 0.65)
            ai_score = ai_sentiment.get("score", 0)
            ai_conf = ai_sentiment.get("confidence", 0)

            if ai_score > 0:
                bullish_score += ai_score * ai_weight * ai_conf
            elif ai_score < 0:
                bearish_score += abs(ai_score) * ai_weight * ai_conf
            total_weight += ai_weight
            contributing.append(f"AI:{ai_sentiment.get('direction', 'neutral')}({ai_conf:.2f})")

        # Calculate confluence score
        if total_weight == 0:
            return None

        bullish_pct = bullish_score / total_weight
        bearish_pct = bearish_score / total_weight
        confluence_score = max(bullish_pct, bearish_pct) * 10  # normalize to 0-10 scale

        # Determine direction
        if bullish_pct > bearish_pct and confluence_score >= self.min_confluence_score:
            direction = "bullish"
            confidence = bullish_pct
            action = "BUY_CE"
        elif bearish_pct > bullish_pct and confluence_score >= self.min_confluence_score:
            direction = "bearish"
            confidence = bearish_pct
            action = "BUY_PE"
        else:
            # No confluence — HOLD
            return FusedSignal(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol=symbol,
                action="HOLD",
                direction="neutral",
                confidence=max(bullish_pct, bearish_pct),
                regime=regime.regime.value,
                strategy=(regime.recommended_strategies[0] if regime.recommended_strategies else "trend"),
                reasoning=f"Insufficient confluence score {confluence_score:.2f} (threshold {self.min_confluence_score:.1f})",
                contributing_signals=contributing,
            )

        # Find best entry, SL, target from contributing signals
        entry, sl, target = self._calculate_levels(
            direction, spot_price, technical_signals, regime
        )

        # Calculate risk:reward
        target_rr = min_rr if min_rr is not None else self.min_rr
        risk = abs(entry - sl) if sl > 0 else entry * 0.02  # default 2% risk
        reward = abs(target - entry) if target > 0 else risk * target_rr
        rr = reward / risk if risk > 0 else 0

        # Reject if R:R is too low
        if rr < target_rr:
            return FusedSignal(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol=symbol,
                action="HOLD",
                direction=direction,
                confidence=confidence,
                entry_price=entry,
                stop_loss=sl,
                target=target,
                risk_reward=rr,
                regime=regime.regime.value,
                strategy=(regime.recommended_strategies[0] if regime.recommended_strategies else "trend"),
                reasoning=f"R:R too low ({rr:.1f}x, minimum {target_rr}x required)",
                contributing_signals=contributing,
            )

        # Build reasoning
        reasoning = self._build_reasoning(
            direction, confidence, regime, contributing, rr
        )

        # Select strategy module based on regime
        strategy = regime.recommended_strategies[0] if regime.recommended_strategies else "trend"

        return FusedSignal(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol=symbol,
            action=action,
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target=round(target, 2),
            risk_reward=round(rr, 2),
            regime=regime.regime.value,
            strategy=strategy,
            reasoning=reasoning,
            contributing_signals=contributing,
        )

    def _calculate_levels(
        self,
        direction: str,
        spot: float,
        tech_signals: List[TechnicalSignal],
        regime: RegimeState
    ) -> tuple:
        """Calculate entry, stop-loss, and target levels."""
        entry = spot

        # Collect SL/target suggestions from signals
        sl_suggestions = [s.stop_loss for s in tech_signals if s.stop_loss > 0]
        target_suggestions = [s.target for s in tech_signals if s.target > 0]

        if direction == "bullish":
            # SL below recent swing low or ATR-based
            sl = min(sl_suggestions) if sl_suggestions else spot * 0.985  # default 1.5% below
            target = max(target_suggestions) if target_suggestions else spot * 1.03  # default 3% above
        else:
            # SL above recent swing high
            sl = max(sl_suggestions) if sl_suggestions else spot * 1.015
            target = min(target_suggestions) if target_suggestions else spot * 0.97

        return entry, sl, target

    def _build_reasoning(
        self,
        direction: str,
        confidence: float,
        regime: RegimeState,
        contributing: list,
        rr: float
    ) -> str:
        """Build plain-language reasoning for the signal."""
        parts = [
            f"{direction.upper()} signal with {confidence:.0%} confidence.",
            f"Market regime: {regime.regime.value}.",
            f"Risk:Reward = 1:{rr:.1f}.",
            f"Contributing signals: {', '.join(contributing[:5])}."
        ]
        return " ".join(parts)
