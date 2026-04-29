# ============================================================================
# PROMETHEUS — Risk Management Core
# ============================================================================
"""
NON-NEGOTIABLE risk management layer.

Every rule here is a HARD LIMIT — the system CANNOT bypass these.
Capital preservation is sacred. These guards protect against:
  - Blowing up account (max daily/weekly loss)
  - Revenge trading (consecutive loss pause)
  - Over-concentration (position limits)
  - Black swans (max drawdown halt)
  - Over-leverage (margin utilization limits)

Design principle: Risk rules are checked BEFORE every order.
If ANY rule is violated, the order is REJECTED — no exceptions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum

from prometheus.utils.logger import logger, log_risk
from prometheus.risk.position_sizer import CapitalBracketManager


class RiskViolation(Enum):
    """Types of risk violations."""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MAX_POSITIONS = "max_open_positions"
    POSITION_SIZE = "position_size_exceeded"
    CORRELATION = "correlated_exposure"
    CONSECUTIVE_LOSSES = "consecutive_loss_pause"
    DRAWDOWN_HALT = "drawdown_halt"
    MARGIN_LIMIT = "margin_utilization"
    TRADING_HOURS = "outside_trading_hours"
    COOL_OFF = "cool_off_period"
    MAX_DAILY_TRADES = "max_daily_trades"


@dataclass
class RiskCheckResult:
    """Result of a risk pre-trade check."""
    approved: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_quantity: int = 0
    reason: str = ""


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk calculations."""
    capital: float
    deployed: float
    cash: float
    unrealized_pnl: float
    realized_pnl_today: float
    realized_pnl_week: float
    open_positions: List[Dict]
    trades_today: int
    consecutive_losses: int
    peak_capital: float
    current_drawdown_pct: float


class RiskManager:
    """
    Central risk management engine.

    All risk limits are hardcoded from config and CANNOT be overridden
    at runtime. The only way to change limits is by editing the config
    file and restarting the system.
    """

    def __init__(self, config: Dict, initial_capital: float = 200000):
        # HARD LIMITS — loaded once, never changed
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Initialize the bracket manager
        from prometheus.config import get_capital_config
        self.bracket_manager = CapitalBracketManager(get_capital_config())

        # Daily limits
        self.max_daily_loss = config.get("max_daily_loss", 5000)
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 3.0)
        self.max_daily_trades = config.get("max_daily_trades", 6)

        # Weekly limits
        self.max_weekly_loss = config.get("max_weekly_loss", 10000)
        self.max_weekly_loss_pct = config.get("max_weekly_loss_pct", 6.0)

        # Position limits
        self.max_open_positions = config.get("max_open_positions", 3)
        self.max_single_position_pct = config.get("max_single_position_pct", 30.0)
        self.max_correlated_pct = config.get("max_correlated_pct", 50.0)

        # Stop loss
        self.max_stop_loss_pct = config.get("max_stop_loss_pct", 50.0)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)

        # Margin
        self.margin_utilization_max = config.get("margin_utilization_max", 70.0)

        # Circuit breakers
        self.consecutive_losses_pause = config.get("consecutive_losses_pause", 3)
        self.pause_duration_minutes = config.get("pause_duration_minutes", 60)
        self.drawdown_halt_pct = config.get("drawdown_halt_pct", 10.0)

        # Intraday limits
        self.max_intraday_trades = config.get("max_intraday_trades", 4)
        self._intraday_trades_today = 0

        # State tracking
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._trades_today = 0
        self._consecutive_losses = 0
        self._last_loss_time: Optional[datetime] = None
        self._open_positions: List[Dict] = []
        self._halted = False
        self._halt_reason = ""
        self._today = date.today()

        logger.info(
            f"Risk Manager initialized: "
            f"Max daily loss=Rs {self.max_daily_loss}, "
            f"Max positions={self.max_open_positions}, "
            f"Drawdown halt={self.drawdown_halt_pct}%"
        )

    def pre_trade_check(
        self,
        trade: Dict,
        current_time: Optional[datetime] = None
    ) -> RiskCheckResult:
        """
        MANDATORY pre-trade risk check. Called before EVERY order.

        Returns RiskCheckResult — trade proceeds ONLY if approved=True.
        """
        if current_time is None:
            current_time = datetime.now()

        # Reset daily counters if new day
        self._check_day_reset(current_time.date())

        violations = []
        warnings = []

        # 1. System halt check
        if self._halted:
            violations.append(f"SYSTEM HALTED: {self._halt_reason}")
            log_risk("BLOCKED", {"trade": trade, "reason": self._halt_reason})
            return RiskCheckResult(
                approved=False, violations=violations,
                reason=f"System halted: {self._halt_reason}"
            )

        # 2. Trading hours check
        from prometheus.utils.indian_market import is_market_open
        if not is_market_open(current_time):
            violations.append("Market is closed. No trading outside 9:15-15:30 IST.")

        # 3. Daily loss limit (only triggers on LOSSES, not profits)
        # Check dynamic max loss limit depending on the current capital bracket
        bracket = self.bracket_manager.get_bracket(self.current_capital)
        # Use stricter of configured limit and bracket-derived cap
        dynamic_max_loss = min(self.max_daily_loss, bracket.max_loss_per_trade * 2)

        if self._daily_pnl <= -dynamic_max_loss:
            violations.append(
                f"Daily loss limit reached: Rs {self._daily_pnl:.0f} "
                f"(limit: Rs -{dynamic_max_loss:.0f} for {bracket.name})"
            )

        daily_loss_pct = (-self._daily_pnl / self.current_capital * 100) if self._daily_pnl < 0 else 0
        if daily_loss_pct >= self.max_daily_loss_pct:
            violations.append(
                f"Daily loss % limit: {daily_loss_pct:.1f}% "
                f"(limit: {self.max_daily_loss_pct}%)"
            )

        # 4. Weekly loss limit (only triggers on LOSSES)
        if self._weekly_pnl <= -self.max_weekly_loss:
            violations.append(
                f"Weekly loss limit reached: Rs {self._weekly_pnl:.0f} "
                f"(limit: Rs {self.max_weekly_loss})"
            )

        # 5. Max daily trades
        if self._trades_today >= self.max_daily_trades:
            violations.append(
                f"Max daily trades reached: {self._trades_today} "
                f"(limit: {self.max_daily_trades})"
            )

        # 5b. Intraday trade limit (separate counter)
        trade_mode = trade.get("trade_mode", "swing")
        if trade_mode == "intraday" and self._intraday_trades_today >= self.max_intraday_trades:
            violations.append(
                f"Max intraday trades reached: {self._intraday_trades_today} "
                f"(limit: {self.max_intraday_trades})"
            )

        # 6. Max open positions
        if len(self._open_positions) >= self.max_open_positions:
            violations.append(
                f"Max open positions: {len(self._open_positions)} "
                f"(limit: {self.max_open_positions})"
            )

        # 7. Position size check
        trade_cost = trade.get("cost", trade.get("entry_price", 0) * trade.get("quantity", 0))
        max_position_value = self.current_capital * self.max_single_position_pct / 100
        if trade_cost > max_position_value:
            violations.append(
                f"Position too large: Rs {trade_cost:.0f} "
                f"(limit: Rs {max_position_value:.0f}, {self.max_single_position_pct}% of capital)"
            )

        # 8. Consecutive losses pause
        if self._consecutive_losses >= self.consecutive_losses_pause:
            if self._last_loss_time:
                pause_end = self._last_loss_time + timedelta(minutes=self.pause_duration_minutes)
                if current_time < pause_end:
                    remaining = (pause_end - current_time).total_seconds() / 60
                    violations.append(
                        f"Cool-off period: {self._consecutive_losses} consecutive losses. "
                        f"Wait {remaining:.0f} more minutes."
                    )
                else:
                    self._consecutive_losses = 0  # Reset after cool-off

        # 9. Drawdown halt
        if self.current_capital < self.peak_capital:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
            if drawdown >= self.drawdown_halt_pct:
                self._halted = True
                self._halt_reason = f"Max drawdown {drawdown:.1f}% exceeded {self.drawdown_halt_pct}%"
                violations.append(self._halt_reason)

        # 10. Correlated exposure check
        correlation_pct = self._check_correlation(trade)
        if correlation_pct > self.max_correlated_pct:
            violations.append(
                f"Correlated exposure too high: {correlation_pct:.0f}% "
                f"(limit: {self.max_correlated_pct}%)"
            )

        # 10b. Block duplicate instrument positions outright
        if any(p.get("symbol") == trade.get("symbol") for p in self._open_positions):
            violations.append("Duplicate instrument position not allowed (symbol already open)")

        # 11. Regulatory 2026 Edge Thresholds
        # Reject trades mathematically doomed by the Union Budget April 2026 STT realities.
        entry_price = trade.get("entry_price", 0)
        target_price = trade.get("target_price", 0)
        if entry_price and target_price:
            expected_pts = abs(target_price - entry_price)
            # If the trade is an options structure, the underlying needs to move ~6 pts just to clear the 2.12 pt option BEP + slippage.
            # If futures, needs >15 pts index move to clear 14.83pt BEP.
            instrument_type = trade.get("instrument_type", "options")
            if instrument_type == "options" and expected_pts < 6.0:
                violations.append(f"Expected index move ({expected_pts:.2f} pts) yields < 3 option pts. Triggers negative EV under SEBI 2026 rules.")
            elif instrument_type == "futures" and expected_pts < 15.0:
                violations.append(f"Expected index move ({expected_pts:.2f} pts) < SEBI 2026 Futures BEP (14.83 pts).")

        # Warnings (non-blocking)
        if trade_cost > max_position_value * 0.8:
            warnings.append(f"Position size near limit ({trade_cost/max_position_value*100:.0f}%)")

        if self._trades_today >= self.max_daily_trades - 1:
            warnings.append("This is the last allowed trade today")

        if self._consecutive_losses >= 2:
            warnings.append(f"{self._consecutive_losses} consecutive losses — be cautious")

        approved = len(violations) == 0

        if not approved:
            log_risk("TRADE_BLOCKED", {
                "trade": trade.get("instrument", "unknown"),
                "violations": violations
            })

        return RiskCheckResult(
            approved=approved,
            violations=violations,
            warnings=warnings,
            adjusted_quantity=trade.get("quantity", 0),
            reason=violations[0] if violations else "Approved"
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        lot_size: int,
        risk_per_trade_pct: float = 2.0
    ) -> Dict:
        """
        Calculate position size using risk-based sizing.

        Logic: Risk X% of capital per trade.
        Position size = (Capital * risk%) / (Entry - StopLoss) per unit
        Then round down to nearest lot.
        """
        risk_amount = self.current_capital * risk_per_trade_pct / 100
        # Enforce bracket max loss per trade as a hard ceiling
        bracket = self.bracket_manager.get_bracket(self.current_capital)
        risk_amount = min(risk_amount, bracket.max_loss_per_trade)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            return {"lots": 0, "quantity": 0, "risk_amount": 0, "error": "Invalid SL"}

        max_units = risk_amount / risk_per_unit
        lots = int(max_units / lot_size)  # ALWAYS round down
        if lots < 1:
            return {
                "lots": 0,
                "quantity": 0,
                "risk_amount": 0,
                "error": "Risk budget insufficient for 1 lot"
            }

        # Cap by max position size
        max_cost = self.current_capital * self.max_single_position_pct / 100
        position_cost = entry_price * lot_size * lots
        if position_cost > max_cost:
            lots = int(max_cost / (entry_price * lot_size))
            if lots < 1:
                return {
                    "lots": 0,
                    "quantity": 0,
                    "risk_amount": 0,
                    "error": "Position cost would breach max_single_position_pct"
                }

        actual_risk = risk_per_unit * lot_size * lots

        return {
            "lots": lots,
            "quantity": lots * lot_size,
            "risk_amount": round(actual_risk, 2),
            "position_cost": round(entry_price * lot_size * lots, 2),
            "risk_pct_of_capital": round(actual_risk / self.current_capital * 100, 2),
        }

    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str = "bullish"
    ) -> float:
        """
        ATR-based dynamic stop loss.

        For options buying: SL = 40-50% of premium (maximum)
        For index reference: SL = entry ± (ATR × multiplier)
        """
        atr_sl_distance = atr * self.atr_multiplier

        if direction == "bullish":
            sl = entry_price - atr_sl_distance
        else:
            sl = entry_price + atr_sl_distance

        # Cap stop loss at max allowed
        max_loss_pct = self.max_stop_loss_pct / 100
        if direction == "bullish":
            sl = max(sl, entry_price * (1 - max_loss_pct))
        else:
            sl = min(sl, entry_price * (1 + max_loss_pct))

        return round(sl, 2)

    def record_trade_result(self, pnl: float, trade: Optional[Dict] = None):
        """
        Record a closed trade's P&L and update all risk state.
        Called after every trade exit.
        """
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self.current_capital += pnl

        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
            self._last_loss_time = datetime.now()
            if self._consecutive_losses >= self.consecutive_losses_pause:
                log_risk("CONSECUTIVE_LOSSES", {
                    "count": self._consecutive_losses,
                    "pause_minutes": self.pause_duration_minutes
                })
        else:
            self._consecutive_losses = 0

        # Remove from open positions
        if trade and trade.get("instrument"):
            self._open_positions = [
                p for p in self._open_positions
                if p.get("instrument") != trade.get("instrument")
            ]

        logger.info(
            f"Trade result: PnL=Rs {pnl:.0f} | "
            f"Daily=Rs {self._daily_pnl:.0f} | "
            f"Capital=Rs {self.current_capital:.0f} | "
            f"Consecutive losses={self._consecutive_losses}"
        )

    def record_trade_entry(self, trade: Dict):
        """Record a new trade entry."""
        self._trades_today += 1
        if trade.get("trade_mode") == "intraday":
            self._intraday_trades_today += 1
        self._open_positions.append(trade)

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio risk state."""
        deployed = sum(
            p.get("cost", p.get("entry_price", 0) * p.get("quantity", 0))
            for p in self._open_positions
        )
        unrealized = sum(p.get("unrealized_pnl", 0) for p in self._open_positions)
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100

        return PortfolioState(
            capital=self.current_capital,
            deployed=deployed,
            cash=self.current_capital - deployed,
            unrealized_pnl=unrealized,
            realized_pnl_today=self._daily_pnl,
            realized_pnl_week=self._weekly_pnl,
            open_positions=self._open_positions,
            trades_today=self._trades_today,
            consecutive_losses=self._consecutive_losses,
            peak_capital=self.peak_capital,
            current_drawdown_pct=round(drawdown, 2),
        )

    def scenario_analysis(
        self,
        spot_change_pct: float,
        positions: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Answer: "What happens to the portfolio if NIFTY drops/rises X%?"

        Estimates portfolio impact using delta approximation.
        """
        if positions is None:
            positions = self._open_positions

        total_impact = 0
        position_impacts = []

        for pos in positions:
            entry = pos.get("entry_price", 0)
            qty = pos.get("quantity", 0)
            direction = pos.get("direction", "bullish")
            delta = pos.get("delta", 0.5)

            # Option P&L ≈ delta × spot_change × quantity
            spot_change = entry * spot_change_pct / 100
            option_change = delta * spot_change

            if direction == "bearish":
                option_change = -option_change

            position_pnl = option_change * qty
            total_impact += position_pnl

            position_impacts.append({
                "instrument": pos.get("instrument", "unknown"),
                "pnl_estimate": round(position_pnl, 2),
            })

        new_capital = self.current_capital + total_impact
        new_drawdown = (self.peak_capital - new_capital) / self.peak_capital * 100

        return {
            "scenario": f"NIFTY {'+' if spot_change_pct > 0 else ''}{spot_change_pct}%",
            "total_impact": round(total_impact, 2),
            "new_capital": round(new_capital, 2),
            "new_drawdown_pct": round(max(new_drawdown, 0), 2),
            "would_trigger_halt": new_drawdown >= self.drawdown_halt_pct,
            "position_impacts": position_impacts,
        }

    def _check_correlation(self, trade: Dict) -> float:
        """Check correlated exposure — same direction on same/related instruments."""
        if not self._open_positions:
            return 0.0

        trade_direction = trade.get("direction", "neutral")
        trade_symbol = trade.get("symbol", "")
        trade_cost = trade.get("cost", trade.get("entry_price", 0) * trade.get("quantity", 0))

        correlated_exposure = trade_cost

        for pos in self._open_positions:
            pos_direction = pos.get("direction", "neutral")
            pos_symbol = pos.get("symbol", "")
            pos_cost = pos.get("cost", pos.get("entry_price", 0) * pos.get("quantity", 0))

            # Same direction on same or related instruments
            if pos_direction == trade_direction:
                if pos_symbol == trade_symbol:
                    correlated_exposure += pos_cost

                # NIFTY and BANKNIFTY are correlated
                related = {
                    "NIFTY 50": ["NIFTY BANK", "NIFTY FIN SERVICE"],
                    "NIFTY BANK": ["NIFTY 50", "NIFTY FIN SERVICE"],
                    "NIFTY FIN SERVICE": ["NIFTY BANK", "NIFTY 50"],
                }
                if trade_symbol in related.get(pos_symbol, []):
                    correlated_exposure += pos_cost * 0.7  # 70% correlation

        return correlated_exposure / self.current_capital * 100

    def _check_day_reset(self, today: date):
        """Reset daily counters on new trading day."""
        if today != self._today:
            if today.weekday() == 0:  # Monday — reset weekly too
                self._weekly_pnl = 0.0

            self._daily_pnl = 0.0
            self._trades_today = 0
            self._intraday_trades_today = 0
            self._today = today

            # Auto-reset halt if drawdown has recovered below threshold
            if self._halted and self.current_capital > 0 and self.peak_capital > 0:
                current_dd = (self.peak_capital - self.current_capital) / self.peak_capital * 100
                if current_dd < self.drawdown_halt_pct:
                    self._halted = False
                    self._halt_reason = ""
                    logger.info(
                        f"System halt auto-reset: drawdown recovered to {current_dd:.1f}% "
                        f"(threshold: {self.drawdown_halt_pct}%)"
                    )

    def force_close_all(self) -> List[Dict]:
        """Emergency: mark all positions for immediate closure."""
        log_risk("FORCE_CLOSE_ALL", {"positions": len(self._open_positions)})
        positions_to_close = self._open_positions.copy()
        return positions_to_close

    def reset_halt(self):
        """Manually reset system halt (requires explicit user action)."""
        self._halted = False
        self._halt_reason = ""
        logger.warning("System halt has been manually reset")
