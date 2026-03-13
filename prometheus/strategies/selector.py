# ============================================================================
# PROMETHEUS — Strategy: Regime-Based Strategy Selector
# ============================================================================
"""
AI-driven strategy selector that picks the right strategy module
based on current market regime.

Maps AMD cycle + volatility regime to strategy activation:
  ACCUMULATION (low vol, range) → Expiry theta harvesting via debit spreads
  MARKUP (trending up)          → Trend module: buy CE
  DISTRIBUTION (topping)        → Trend module: buy PE / debit put spreads
  MARKDOWN (trending down)      → Trend module: buy PE
  VOLATILE (high vol)           → Volatility module: straddles or hedge
  PRE-EVENT                     → Volatility module: buy straddles/strangles
"""

from typing import Dict, List, Optional, Tuple
from datetime import date, datetime

from prometheus.signals.regime_detector import RegimeDetector, MarketRegime, RegimeState
from prometheus.utils.logger import logger


# Strategy activation matrix
REGIME_STRATEGY_MAP = {
    MarketRegime.ACCUMULATION: {
        "primary": "expiry",
        "secondary": "mean_reversion",
        "options_bias": "neutral",
        "description": "Low volatility range — harvest theta with defined-risk spreads",
    },
    MarketRegime.MARKUP: {
        "primary": "trend",
        "secondary": "volatility",
        "options_bias": "buy_ce",
        "description": "Bullish trend — buy calls on pullbacks to VWAP/support",
    },
    MarketRegime.DISTRIBUTION: {
        "primary": "trend",
        "secondary": "volatility",
        "options_bias": "buy_pe",
        "description": "Topping pattern — buy puts or bear put spreads",
    },
    MarketRegime.MARKDOWN: {
        "primary": "trend",
        "secondary": "volatility",
        "options_bias": "buy_pe",
        "description": "Bearish trend — buy puts on rallies to resistance",
    },
    MarketRegime.VOLATILE: {
        "primary": "volatility",
        "secondary": "trend",
        "options_bias": "straddle",
        "description": "High volatility — trade breakouts or straddles",
    },
    MarketRegime.UNKNOWN: {
        "primary": "none",
        "secondary": "none",
        "options_bias": "hold",
        "description": "Unclear regime — sit on hands, preserve capital",
    },
}


class StrategySelector:
    """
    Selects the optimal strategy based on market regime and conditions.

    Decision hierarchy:
    1. Check for upcoming events → override to volatility module
    2. Detect market regime (AMD + volatility)
    3. Map regime to primary strategy
    4. Validate capital availability for selected strategy
    5. Return strategy recommendation with confidence
    """

    def __init__(self, config: Dict, capital: float = 200000):
        self.config = config
        self.capital = capital
        self.regime_detector = RegimeDetector()
        self.active_modules = config.get("active_modules", ["trend", "volatility", "expiry"])

    def select(
        self,
        regime: RegimeState,
        vix: Optional[float] = None,
        upcoming_events: Optional[List[Dict]] = None,
        iv_percentile: Optional[float] = None,
    ) -> Dict:
        """
        Select the best strategy for current conditions.

        Returns:
            {
                "strategy": str,           # strategy module name
                "options_bias": str,       # buy_ce, buy_pe, straddle, neutral, hold
                "confidence": float,       # 0-1
                "reasoning": str,
                "regime": str,
                "capital_suitable": bool,
            }
        """
        # Event override: upcoming events within 2 days → volatility module
        if upcoming_events:
            today = date.today()
            for event in upcoming_events:
                event_date = event.get("date")
                if isinstance(event_date, str):
                    try:
                        event_date = datetime.strptime(event_date, "%Y-%m-%d").date()
                    except ValueError:
                        continue

                if event_date and 0 < (event_date - today).days <= 2:
                    if "volatility" in self.active_modules:
                        return {
                            "strategy": "volatility",
                            "options_bias": "straddle",
                            "confidence": 0.80,
                            "reasoning": (
                                f"Event override: {event.get('type', 'unknown')} in "
                                f"{(event_date - today).days} days. "
                                f"Activating volatility module for pre-event positioning."
                            ),
                            "regime": regime.regime.value,
                            "capital_suitable": True,
                        }

        # VIX extreme override
        if vix is not None:
            if vix > 25 and "volatility" in self.active_modules:
                return {
                    "strategy": "volatility",
                    "options_bias": "buy_ce",  # VIX spike = fear = often bottom
                    "confidence": 0.70,
                    "reasoning": (
                        f"VIX spike ({vix:.1f} > 25). Fear elevated. "
                        f"Contrarian opportunity. Buy calls or wait for VIX to peak."
                    ),
                    "regime": regime.regime.value,
                    "capital_suitable": True,
                }
            elif vix < 11 and "volatility" in self.active_modules:
                return {
                    "strategy": "volatility",
                    "options_bias": "straddle",
                    "confidence": 0.65,
                    "reasoning": (
                        f"VIX compression ({vix:.1f} < 11). Vol is cheap. "
                        f"Buy strangles for potential expansion."
                    ),
                    "regime": regime.regime.value,
                    "capital_suitable": True,
                }

        # Standard regime-based selection
        regime_config = REGIME_STRATEGY_MAP.get(
            regime.regime,
            REGIME_STRATEGY_MAP[MarketRegime.UNKNOWN]
        )

        primary = regime_config["primary"]
        secondary = regime_config["secondary"]

        # Use primary if available, else fallback to secondary
        if primary in self.active_modules:
            selected = primary
        elif secondary in self.active_modules:
            selected = secondary
        else:
            selected = "none"

        # Capital suitability check
        capital_ok = self._check_capital_suitability(selected)

        # If selected strategy not capital-suitable, try alternatives
        if not capital_ok and selected != "none":
            for module in self.active_modules:
                if self._check_capital_suitability(module):
                    selected = module
                    capital_ok = True
                    break

        confidence = regime.confidence
        if not capital_ok:
            confidence *= 0.5  # reduce confidence if capital constrained

        return {
            "strategy": selected,
            "options_bias": regime_config["options_bias"],
            "confidence": round(confidence, 3),
            "reasoning": regime_config["description"],
            "regime": regime.regime.value,
            "capital_suitable": capital_ok,
        }

    def _check_capital_suitability(self, strategy: str) -> bool:
        """Check if capital is sufficient for the strategy."""
        # Under 2L limits
        capital_requirements = {
            "trend": 5000,          # need ~5K for 1 lot ATM option
            "volatility": 10000,    # need ~10K for straddle
            "expiry": 15000,        # need ~15K for debit spread
            "mean_reversion": 100000,  # need 1L+ for iron condors
        }
        required = capital_requirements.get(strategy, 0)
        return self.capital >= required

    def get_strategy_explanation(self, selection: Dict) -> str:
        """Generate plain-language explanation of the strategy selection."""
        lines = [
            f"Market Regime: {selection['regime'].upper()}",
            f"Selected Strategy: {selection['strategy'].upper()}",
            f"Options Bias: {selection['options_bias'].replace('_', ' ').upper()}",
            f"Confidence: {selection['confidence']:.0%}",
            f"Reasoning: {selection['reasoning']}",
        ]
        if not selection["capital_suitable"]:
            lines.append("WARNING: Capital may be insufficient for optimal sizing")
        return "\n".join(lines)
