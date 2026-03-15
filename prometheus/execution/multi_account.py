# ============================================================================
# PROMETHEUS — Multi-Account Paper Trading
# ============================================================================
"""
Manages N independent paper trading accounts simultaneously.
Each account has its own PaperTrader, OrderManager, RiskManager,
equity curve, and position tracking.

All accounts receive the SAME signals but independently:
- Size positions based on their own capital
- Track their own equity curve
- Apply their own risk limits
- Can reject trades their risk manager blocks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prometheus.execution.paper_trader import PaperTrader
from prometheus.execution.order_manager import OrderManager
from prometheus.risk.manager import RiskManager
from prometheus.utils.logger import logger


@dataclass
class AccountConfig:
    """Configuration for a single simulated account."""
    label: str                  # "micro_15k", "starter_50k", etc.
    initial_capital: float      # 15000, 50000, 100000, 200000
    risk_overrides: Dict = field(default_factory=dict)


class AccountStack:
    """One complete execution stack for a single account."""

    def __init__(self, config: AccountConfig, risk_config: Dict):
        self.config = config
        self.trader = PaperTrader(config.initial_capital)

        # Merge risk overrides into config
        merged_risk = dict(risk_config)
        merged_risk.update(config.risk_overrides)
        self.risk = RiskManager(merged_risk, config.initial_capital)
        self.order_manager = OrderManager(self.trader, self.risk, "paper")
        self.equity_curve: List[float] = [config.initial_capital]

    def get_summary(self) -> Dict:
        """Return current account state for dashboard display."""
        margins = self.trader.get_margins()
        equity = margins.get("equity", self.config.initial_capital)
        return {
            "label": self.config.label,
            "initial": self.config.initial_capital,
            "equity": equity,
            "realized_pnl": margins.get("realized_pnl", 0),
            "unrealized_pnl": margins.get("total_pnl", 0),
            "total_costs": margins.get("total_costs", 0),
            "trades": len(self.trader.trade_history),
            "open_positions": len(self.trader.get_positions()),
            "win_rate": self._compute_win_rate(),
            "pnl": equity - self.config.initial_capital,
            "return_pct": (equity - self.config.initial_capital) / self.config.initial_capital * 100,
        }

    def _compute_win_rate(self) -> float:
        if not self.trader.trade_history:
            return 0.0
        wins = sum(1 for t in self.trader.trade_history if t.get("net_pnl", 0) > 0)
        return wins / len(self.trader.trade_history) * 100

    def record_equity(self):
        """Snapshot current equity for equity curve tracking."""
        margins = self.trader.get_margins()
        self.equity_curve.append(margins.get("equity", self.config.initial_capital))


class MultiAccountPaperTrader:
    """
    Wrapper managing N independent PaperTrader stacks.

    Each account receives the SAME signals but independently sizes,
    risks, and tracks positions.
    """

    DEFAULT_ACCOUNTS = [
        AccountConfig("micro_15k", 15000),
        AccountConfig("starter_50k", 50000),
        AccountConfig("growth_100k", 100000),
        AccountConfig("full_200k", 200000),
    ]

    def __init__(self, accounts: List[AccountConfig] = None, risk_config: Dict = None):
        if accounts is None:
            accounts = self.DEFAULT_ACCOUNTS
        if risk_config is None:
            risk_config = {}

        self.stacks: Dict[str, AccountStack] = {}
        for acc in accounts:
            self.stacks[acc.label] = AccountStack(acc, risk_config)

        labels = ", ".join(f"{a.label}=Rs{a.initial_capital:,.0f}" for a in accounts)
        logger.info(f"MultiAccount initialized: {len(self.stacks)} accounts ({labels})")

    def dispatch_signal(self, signal: Dict, confirm: bool = False) -> Dict[str, bool]:
        """
        Send the same signal to all accounts. Each independently decides to trade or not.
        Returns {label: True/False} indicating whether each account opened a position.
        """
        results = {}
        for label, stack in self.stacks.items():
            try:
                position = stack.order_manager.execute_signal(signal, confirm=confirm)
                results[label] = position is not None
                if position:
                    logger.info(f"[{label}] Position opened: {position.position_id}")
            except Exception as e:
                logger.error(f"[{label}] Signal dispatch error: {e}")
                results[label] = False
        return results

    def close_all(self, reason: str = "session_end") -> Dict[str, float]:
        """Close all positions across all accounts."""
        pnls = {}
        for label, stack in self.stacks.items():
            try:
                pnl = stack.order_manager.close_all_positions(reason)
                pnls[label] = pnl
            except Exception as e:
                logger.error(f"[{label}] close_all error: {e}")
                pnls[label] = 0.0
        return pnls

    def update_all_prices(self, prices: Dict[str, float]):
        """Feed price updates to all accounts' PaperTraders."""
        for stack in self.stacks.values():
            stack.trader.update_prices(prices)

    def record_all_equity(self):
        """Record equity snapshot for all accounts."""
        for stack in self.stacks.values():
            stack.record_equity()

    def get_summary_table(self) -> List[Dict]:
        """Get all account summaries for dashboard display."""
        return [stack.get_summary() for label, stack in self.stacks.items()]

    def get_stack(self, label: str) -> Optional[AccountStack]:
        """Get a specific account stack by label."""
        return self.stacks.get(label)
