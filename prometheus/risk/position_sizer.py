import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger("prometheus")


@dataclass
class CapitalBracket:
    name: str
    max_capital: float
    max_loss_per_trade: float
    min_rr: float
    base_target: float
    sl_atr_mult: float
    min_margin: float


class CapitalBracketManager:
    """
    Manages the dynamic assignment of risk parameters and sizing rules based on
    the current portfolio equity. It maps current capital to one of 5 distinct
    brackets (e.g. 15K, 30K, 50K, 1L, 2L+).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.brackets = self._load_brackets(config)
        # Default fallback bracket if config is missing
        self.default_bracket = CapitalBracket(
            name="Fallback",
            max_capital=float('inf'),
            max_loss_per_trade=500,
            min_rr=2.0,
            base_target=2.5,
            sl_atr_mult=1.2,
            min_margin=0
        )

    def _load_brackets(self, config: Dict[str, Any]) -> list[CapitalBracket]:
        """Loads and sorts brackets from configuration."""
        brackets_config = config.get("brackets", {})
        brackets_list = []
        
        for key, value in brackets_config.items():
            bracket = CapitalBracket(
                name=value.get("name", "Unknown"),
                max_capital=float(value.get("max_capital", 0)),
                max_loss_per_trade=float(value.get("max_loss_per_trade", 0)),
                min_rr=float(value.get("min_rr", 2.0)),
                base_target=float(value.get("base_target", 2.5)),
                sl_atr_mult=float(value.get("sl_atr_mult", 1.2)),
                min_margin=float(value.get("min_margin", 0)),
            )
            brackets_list.append(bracket)
            
        # Sort by max_capital ascending to allow simple iteration for assignment
        brackets_list.sort(key=lambda b: b.max_capital)
        return brackets_list

    def get_bracket(self, current_capital: float) -> CapitalBracket:
        """Determines the appropriate bracket for the current capital amount."""
        if not self.brackets:
            return self.default_bracket
            
        for bracket in self.brackets:
            if current_capital <= bracket.max_capital:
                return bracket
                
        # If capital exceeds all max_capitals, return the highest tier
        return self.brackets[-1]

    def get_min_rr(self, current_capital: float) -> float:
        return self.get_bracket(current_capital).min_rr

    def get_base_target(self, current_capital: float) -> float:
        return self.get_bracket(current_capital).base_target

    def get_sl_atr_mult(self, current_capital: float) -> float:
        return self.get_bracket(current_capital).sl_atr_mult

    def get_max_loss_per_trade(self, current_capital: float) -> float:
        return self.get_bracket(current_capital).max_loss_per_trade
