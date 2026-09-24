"""
PROMETHEUS — Strategy Drift & Continuous Evolution Supervisor
============================================================
Compliant with BIS Paper 111 ('Algorithmic trading and market structure')
and SEC Market Access Rule 15c3-5 algorithmic risk governance.

Monitors rolling performance metrics (last N closed trades, default 15-20)
to detect regime shifts, parameter decay, and statistical edge degradation:
- Rolling Win Rate
- Rolling Gross Wins / Gross Losses
- Rolling Profit Factor (PF) = Gross Wins / max(Gross Losses, 1.0)
- Mathematical Expectancy E = (WR * Avg Win) - ((1 - WR) * Avg Loss)

Dynamic Risk Adaptation (Meta-Learning Circuit):
- HEALTHY (PF >= 1.30 and E > 0): Multiplier = 1.0 (Full sizing)
- DEGRADED (1.0 <= PF < 1.30 or E near 0): Multiplier = 0.5 (Derate lots by 50% to conserve capital)
- DECAYING (PF < 1.0 or E < 0 with >= 5 trades): Multiplier = 0.0 (Quarantine strategy to Paper Capture / Block Live)
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from prometheus.utils.logger import logger


class StrategyDriftSupervisor:
    """
    Automated Continuous Meta-Adaptation & Strategy Drift Supervisor.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        rolling_window: int = 15,
        min_trades_for_eval: int = 5,
        healthy_pf_threshold: float = 1.30,
        degraded_pf_threshold: float = 1.00,
        baseline_win_rate: float = 0.55,
        baseline_profit_factor: float = 2.00,
        max_deviation_pct: float = 0.25,
    ):
        self.db_path = db_path
        self.rolling_window = rolling_window
        self.min_trades_for_eval = min_trades_for_eval
        self.healthy_pf_threshold = healthy_pf_threshold
        self.degraded_pf_threshold = degraded_pf_threshold
        self.baseline_win_rate = baseline_win_rate
        self.baseline_profit_factor = baseline_profit_factor
        self.max_deviation_pct = max_deviation_pct
        self._in_memory_trades: List[Dict[str, Any]] = []


    def record_trade(self, pnl: float, strategy: str = "", trade_id: str = "") -> None:
        """Record a completed trade outcome for online rolling adaptation."""
        self._in_memory_trades.append({
            "pnl": float(pnl),
            "strategy": strategy,
            "trade_id": trade_id,
        })

    def get_recent_pnls(self, strategy: str = "") -> List[float]:
        """
        Retrieve recent trade PnLs, prioritizing in-memory live stream
        with fallback to persistent SQLite ledger if db_path is provided and no in-memory trades exist.
        """
        # If in-memory trades have been recorded, use them directly
        if self._in_memory_trades:
            if strategy:
                return [t["pnl"] for t in self._in_memory_trades if t.get("strategy") == strategy][-self.rolling_window:]
            return [t["pnl"] for t in self._in_memory_trades][-self.rolling_window:]

        # Query persistent SQLite ledger only if db_path is explicitly set and exists
        if self.db_path:
            p = Path(self.db_path)
            if p.exists():
                try:
                    conn = sqlite3.connect(str(p))
                    cursor = conn.cursor()
                    if strategy:
                        cursor.execute(
                            "SELECT net_pnl FROM paper_trades WHERE strategy = ? ORDER BY rowid DESC LIMIT ?",
                            (strategy, self.rolling_window),
                        )
                    else:
                        cursor.execute(
                            "SELECT net_pnl FROM paper_trades ORDER BY rowid DESC LIMIT ?",
                            (self.rolling_window,),
                        )
                    rows = cursor.fetchall()
                    conn.close()
                    if rows:
                        return [float(r[0]) for r in reversed(rows)]
                except Exception as e:
                    logger.warning(f"StrategyDriftSupervisor: Failed to query ledger {e}")

        return []



    def evaluate_regime_health(self, strategy: str = "") -> Dict[str, Any]:
        """
        Perform quantitative health evaluation of the strategy in current market regime.
        Returns health status, sizing multiplier, Profit Factor, Win Rate, and Expectancy.
        """
        pnls = self.get_recent_pnls(strategy)
        total_trades = len(pnls)

        if total_trades < self.min_trades_for_eval:
            return {
                "status": "INSUFFICIENT_DATA",
                "multiplier": 1.0,
                "trades_count": total_trades,
                "profit_factor": 1.0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "message": f"Insufficient trade history ({total_trades}/{self.min_trades_for_eval}); operating standard risk.",
            }

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 99.0 if gross_profit > 0 else 1.0

        # Statistical Process Control (SPC) Deviations against Backtest Baseline
        wr_deviation_pct = ((self.baseline_win_rate - win_rate) / self.baseline_win_rate * 100.0) if self.baseline_win_rate > 0 else 0.0
        pf_deviation_pct = ((self.baseline_profit_factor - profit_factor) / self.baseline_profit_factor * 100.0) if self.baseline_profit_factor > 0 else 0.0

        if profit_factor < self.degraded_pf_threshold or expectancy <= 0:
            status = "DECAYING"
            multiplier = 0.0
            msg = f"Strategy edge in decay (PF={profit_factor:.2f} < 1.0, Exp=Rs {expectancy:.1f}); quarantined to paper mode."
        elif profit_factor < self.healthy_pf_threshold or wr_deviation_pct > (self.max_deviation_pct * 100.0) or pf_deviation_pct > (self.max_deviation_pct * 100.0):
            status = "DEGRADED"
            multiplier = 0.5
            reasons = []
            if wr_deviation_pct > (self.max_deviation_pct * 100.0):
                reasons.append(f"WR ({win_rate*100:.1f}%) drifted {wr_deviation_pct:.0f}% below baseline ({self.baseline_win_rate*100:.0f}%)")
            if pf_deviation_pct > (self.max_deviation_pct * 100.0):
                reasons.append(f"PF ({profit_factor:.2f}) drifted {pf_deviation_pct:.0f}% below baseline ({self.baseline_profit_factor:.1f})")
            if profit_factor < self.healthy_pf_threshold:
                reasons.append(f"PF ({profit_factor:.2f}) < healthy threshold ({self.healthy_pf_threshold:.2f})")
            reason_str = "; ".join(reasons)
            msg = f"Statistical Process Control: Edge degraded ({reason_str}); derating risk 50%."
        else:
            status = "HEALTHY"
            multiplier = 1.0
            msg = f"Strategy edge healthy (PF={profit_factor:.2f}, Exp=Rs {expectancy:.1f}); full allocation."

        return {
            "status": status,
            "multiplier": multiplier,
            "trades_count": total_trades,
            "profit_factor": round(profit_factor, 2),
            "win_rate": round(win_rate * 100, 1),
            "expectancy": round(expectancy, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "baseline_win_rate": self.baseline_win_rate,
            "baseline_profit_factor": self.baseline_profit_factor,
            "wr_deviation_pct": round(wr_deviation_pct, 1),
            "pf_deviation_pct": round(pf_deviation_pct, 1),
            "message": msg,
        }


    def get_drift_multiplier(self, strategy: str = "") -> float:
        """Return the dynamic risk multiplier (1.0, 0.5, or 0.0) based on regime health."""
        health = self.evaluate_regime_health(strategy)
        return float(health.get("multiplier", 1.0))

    def notify_if_drift(self, telegram_bot: Any = None, strategy: str = "") -> bool:
        """
        Check if strategy has drifted and fire Telegram alert if degraded or decaying.
        """
        health = self.evaluate_regime_health(strategy)
        status = health.get("status")
        if status in ("DEGRADED", "DECAYING") and telegram_bot:
            try:
                if hasattr(telegram_bot, "alert_strategy_drift_warning"):
                    telegram_bot.alert_strategy_drift_warning(health)
                    return True
            except Exception as e:
                logger.warning(f"Failed to dispatch drift alert to Telegram: {e}")
        return False

