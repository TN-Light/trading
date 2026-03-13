# ============================================================================
# PROMETHEUS — Backtesting Engine
# ============================================================================
"""
Walk-forward backtesting with realistic cost modeling for Indian F&O.

Key principles:
  - NO look-ahead bias (signals only use data available at that time)
  - Walk-forward: train on window, test on next, slide forward
  - Realistic costs: Zerodha brokerage, STT, GST, stamp duty, slippage
  - Options-aware: handles decay, IV changes, not just price direction
  - Monte Carlo simulation for robustness testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from prometheus.utils.logger import logger


@dataclass
class BacktestTrade:
    """A single trade in backtesting."""
    entry_time: str
    exit_time: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    costs: float
    net_pnl: float
    strategy: str
    hold_duration_minutes: int = 0
    exit_reason: str = ""

    # Signal features (for regression training)
    signal_liqsweep: bool = False
    signal_fvg: bool = False
    signal_vp: bool = False
    signal_ote: bool = False
    signal_rsi_div: bool = False
    signal_vol_surge: bool = False
    signal_vol_confirm: bool = False
    signal_vwap: bool = False
    signal_bias: bool = False

    # Signal context (for feature extraction)
    bull_score: float = 0.0
    bear_score: float = 0.0
    atr_at_entry: float = 0.0
    regime_at_entry: str = ""
    option_expiry_date: str = ""  # For DTE-aware theta


@dataclass
class BacktestResult:
    """Complete backtest results."""
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_pnl: float
    avg_hold_duration_min: float
    total_costs: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    monthly_returns: Dict[str, float]
    trades: List[Dict]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"\n{'='*60}\n"
            f"BACKTEST RESULTS: {self.strategy}\n"
            f"{'='*60}\n"
            f"Period: {self.start_date} to {self.end_date}\n"
            f"Initial Capital: Rs {self.initial_capital:,.0f}\n"
            f"Final Capital: Rs {self.final_capital:,.0f}\n"
            f"Total Return: {self.total_return_pct:.1f}%\n"
            f"Annualized Return: {self.annualized_return_pct:.1f}%\n"
            f"{'─'*60}\n"
            f"Total Trades: {self.total_trades}\n"
            f"Win Rate: {self.win_rate:.1f}%\n"
            f"Avg Win: Rs {self.avg_win:,.0f}\n"
            f"Avg Loss: Rs {self.avg_loss:,.0f}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"{'─'*60}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Sortino Ratio: {self.sortino_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown_pct:.1f}%\n"
            f"Max DD Duration: {self.max_drawdown_duration_days} days\n"
            f"{'─'*60}\n"
            f"Total Costs (brokerage, STT, etc.): Rs {self.total_costs:,.0f}\n"
            f"Avg Trade PnL: Rs {self.avg_trade_pnl:,.0f}\n"
            f"{'='*60}\n"
        )


class ZerodhaCostModel:
    """
    Realistic cost model for Zerodha (most popular Indian broker).

    Costs per trade:
    - Brokerage: Rs 20 per order (or 0.03% whichever is lower) for F&O
    - STT: 0.0625% on sell side for options
    - Transaction charges: 0.053% (NSE)
    - GST: 18% on (brokerage + transaction charges)
    - SEBI charges: Rs 10 per crore
    - Stamp duty: 0.003% on buy side
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.brokerage_per_order = cfg.get("options_brokerage", 20)
        self.stt_options_sell_pct = cfg.get("stt_options_sell", 0.0625) / 100
        self.stt_futures_pct = cfg.get("stt_futures", 0.0125) / 100
        self.transaction_charges_pct = cfg.get("transaction_charges", 0.053) / 100
        self.gst_pct = cfg.get("gst", 18.0) / 100
        self.sebi_charges_pct = cfg.get("sebi_charges", 0.0001) / 100
        self.stamp_duty_pct = cfg.get("stamp_duty", 0.003) / 100
        self.slippage_pct = cfg.get("slippage_pct", 0.05) / 100

    def calculate_costs(
        self,
        buy_value: float,
        sell_value: float,
        instrument_type: str = "options"
    ) -> Dict:
        """Calculate all trading costs for a round-trip trade."""
        # Brokerage: Rs 20 per order × 2 (buy + sell)
        brokerage = self.brokerage_per_order * 2

        # STT (Securities Transaction Tax) — only on sell side
        if instrument_type == "options":
            stt = sell_value * self.stt_options_sell_pct
        else:
            stt = sell_value * self.stt_futures_pct

        # Transaction charges — on both sides
        transaction = (buy_value + sell_value) * self.transaction_charges_pct

        # GST on brokerage + transaction charges
        gst = (brokerage + transaction) * self.gst_pct

        # SEBI charges
        sebi = (buy_value + sell_value) * self.sebi_charges_pct

        # Stamp duty — only on buy side
        stamp = buy_value * self.stamp_duty_pct

        # Slippage (estimated)
        slippage = (buy_value + sell_value) * self.slippage_pct

        total = brokerage + stt + transaction + gst + sebi + stamp + slippage

        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "transaction_charges": round(transaction, 2),
            "gst": round(gst, 2),
            "sebi_charges": round(sebi, 2),
            "stamp_duty": round(stamp, 2),
            "slippage": round(slippage, 2),
            "total": round(total, 2),
        }


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Process:
    1. Iterate through historical data bar by bar
    2. At each bar, run the signal engine (with only past data — no future)
    3. If signal generated, simulate entry with slippage
    4. Track position, check exit conditions (SL/target/time)
    5. Calculate costs on exit
    6. Compute comprehensive performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 200000,
        cost_config: Optional[Dict] = None
    ):
        self.initial_capital = initial_capital
        self.cost_model = ZerodhaCostModel(cost_config)
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []

    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        strategy_name: str = "default",
        warmup_bars: int = 50
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV DataFrame with timestamp, open, high, low, close, volume
            signal_generator: Function that takes (data_so_far) and returns signal dict or None
                Signal dict: {direction, entry_price, stop_loss, target, quantity}
            strategy_name: Name for the backtest
            warmup_bars: Number of bars to skip for indicator warmup
        """
        capital = self.initial_capital
        peak_capital = capital
        self.trades = []
        self.equity_curve = [capital]

        position = None  # Current open position
        daily_pnl = 0.0
        daily_trades = 0
        last_date = None

        logger.info(f"Starting backtest: {strategy_name} on {len(data)} bars")

        for i in range(warmup_bars, len(data)):
            current_bar = data.iloc[i]
            current_time = str(current_bar.get("timestamp", i))
            current_date = str(current_bar.get("timestamp", ""))[:10]

            # Reset daily counters
            if current_date != last_date:
                daily_pnl = 0.0
                daily_trades = 0
                last_date = current_date

            # Check daily loss limit
            if daily_pnl < -self.initial_capital * 0.03:
                if position:
                    # Force close on daily loss limit
                    if position.get("instrument_type") == "options":
                        force_exit_price = position.get("current_premium", position["entry_price"])
                    else:
                        force_exit_price = current_bar["close"]
                    capital, trade = self._close_position(
                        position, force_exit_price, current_time, capital, "daily_loss_limit"
                    )
                    self.trades.append(trade)
                    daily_pnl += trade.net_pnl
                    position = None
                self.equity_curve.append(capital)
                continue

            # If we have a position, check exit conditions
            if position:
                exit_triggered, exit_price, exit_reason = self._check_exit(
                    position, current_bar
                )
                if exit_triggered:
                    capital, trade = self._close_position(
                        position, exit_price, current_time, capital, exit_reason
                    )
                    self.trades.append(trade)
                    daily_pnl += trade.net_pnl
                    daily_trades += 1
                    position = None

            # If no position, check for new signal
            if position is None and daily_trades < 6:
                # Only pass data up to current bar (no look-ahead)
                data_so_far = data.iloc[:i + 1]
                signal = signal_generator(data_so_far)

                if signal:
                    position = self._open_position(signal, current_bar, current_time)

            # Update equity curve
            unrealized = 0
            if position:
                if position.get("instrument_type") == "options":
                    # Options: unrealized based on current premium vs entry premium
                    unrealized = (position["current_premium"] - position["entry_price"]) * position["quantity"]
                else:
                    if position["direction"] == "bullish":
                        unrealized = (current_bar["close"] - position["entry_price"]) * position["quantity"]
                    else:
                        unrealized = (position["entry_price"] - current_bar["close"]) * position["quantity"]

            self.equity_curve.append(capital + unrealized)

            # Track peak
            if capital > peak_capital:
                peak_capital = capital

        # Close any remaining position at last bar
        if position:
            if position.get("instrument_type") == "options":
                exit_price = position.get("current_premium", position["entry_price"])
            else:
                exit_price = data.iloc[-1]["close"]
            capital, trade = self._close_position(
                position, exit_price,
                str(data.iloc[-1].get("timestamp", len(data))),
                capital, "end_of_data"
            )
            self.trades.append(trade)

        # Calculate metrics
        return self._calculate_metrics(strategy_name, data, capital)

    def _open_position(self, signal: Dict, bar: pd.Series, timestamp: str) -> Dict:
        """Simulate opening a position with slippage."""
        is_options = signal.get("instrument_type") == "options"

        if is_options:
            # For options: entry_price is the premium, track it separately
            premium_entry = signal.get("entry_price", bar["close"] * 0.012)
            slippage = premium_entry * self.cost_model.slippage_pct
            if signal.get("direction") == "bullish":
                premium_entry += slippage
            else:
                premium_entry -= slippage

            return {
                "entry_time": timestamp,
                "symbol": signal.get("symbol", "NIFTY"),
                "direction": signal.get("direction", "bullish"),
                "entry_price": premium_entry,
                "stop_loss": signal.get("stop_loss", 0),
                "target": signal.get("target", 0),
                "quantity": signal.get("quantity", 1),
                "strategy": signal.get("strategy", "default"),
                "instrument_type": "options",
                "delta": signal.get("delta", 0.5),
                "current_premium": premium_entry,
                "prev_close": bar["close"],
                "max_bars": signal.get("max_bars", 0),
                "bar_interval": signal.get("bar_interval", "day"),
                "breakeven_ratio": signal.get("breakeven_ratio", 0.4),
            }
        else:
            entry_price = signal.get("entry_price", bar["close"])
            slippage = entry_price * self.cost_model.slippage_pct
            if signal.get("direction") == "bullish":
                entry_price += slippage
            else:
                entry_price -= slippage

            return {
                "entry_time": timestamp,
                "symbol": signal.get("symbol", "NIFTY"),
                "direction": signal.get("direction", "bullish"),
                "entry_price": entry_price,
                "stop_loss": signal.get("stop_loss", 0),
                "target": signal.get("target", 0),
                "quantity": signal.get("quantity", 1),
                "strategy": signal.get("strategy", "default"),
                "instrument_type": "futures",
            }

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        timestamp: str,
        capital: float,
        reason: str
    ) -> Tuple[float, BacktestTrade]:
        """Simulate closing a position with costs."""
        quantity = position["quantity"]
        entry = position["entry_price"]

        if position.get("instrument_type") == "options":
            # Options: P&L is based on premium, not index price
            # exit_price here is already the premium level (set by _check_exit or current_premium)
            premium_entry = entry
            premium_exit = exit_price

            # Apply slippage on the premium
            slippage = premium_exit * self.cost_model.slippage_pct
            if position["direction"] == "bullish":
                premium_exit -= slippage  # Sell lower
            else:
                premium_exit -= slippage  # Sell lower (liquidating put)
            premium_exit = max(premium_exit, 0)

            gross_pnl = (premium_exit - premium_entry) * quantity

            # Costs based on premium values (options trade value)
            buy_value = premium_entry * quantity
            sell_value = premium_exit * quantity
            costs = self.cost_model.calculate_costs(buy_value, sell_value, "options")
            total_cost = costs["total"]

            net_pnl = gross_pnl - total_cost
            capital += net_pnl

            trade = BacktestTrade(
                entry_time=position["entry_time"],
                exit_time=timestamp,
                symbol=position["symbol"],
                direction=position["direction"],
                entry_price=round(premium_entry, 2),
                exit_price=round(premium_exit, 2),
                quantity=quantity,
                gross_pnl=round(gross_pnl, 2),
                costs=round(total_cost, 2),
                net_pnl=round(net_pnl, 2),
                strategy=position["strategy"],
                exit_reason=reason,
            )
            return capital, trade

        # Futures/equity: original logic
        # Apply slippage on exit
        slippage = exit_price * self.cost_model.slippage_pct
        if position["direction"] == "bullish":
            exit_price -= slippage  # Sell lower
        else:
            exit_price += slippage  # Buy higher

        # Gross P&L
        if position["direction"] == "bullish":
            gross_pnl = (exit_price - entry) * quantity
        else:
            gross_pnl = (entry - exit_price) * quantity

        # Costs
        buy_value = entry * quantity
        sell_value = exit_price * quantity
        costs = self.cost_model.calculate_costs(buy_value, sell_value)
        total_cost = costs["total"]

        net_pnl = gross_pnl - total_cost
        capital += net_pnl

        trade = BacktestTrade(
            entry_time=position["entry_time"],
            exit_time=timestamp,
            symbol=position["symbol"],
            direction=position["direction"],
            entry_price=entry,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=round(gross_pnl, 2),
            costs=round(total_cost, 2),
            net_pnl=round(net_pnl, 2),
            strategy=position["strategy"],
            exit_reason=reason,
        )

        return capital, trade

    def _check_exit(
        self,
        position: Dict,
        bar: pd.Series
    ) -> Tuple[bool, float, str]:
        """Check if exit conditions are met."""
        sl = position.get("stop_loss", 0)
        target = position.get("target", 0)

        if position.get("instrument_type") == "options":
            # Delta-adjusted premium tracking for options
            delta = position.get("delta", 0.5)
            prev_close = position.get("prev_close", bar["close"])
            current_premium = position.get("current_premium", position["entry_price"])
            entry_premium = position["entry_price"]

            # Track bars held
            bars_held = position.get("bars_held", 0) + 1
            position["bars_held"] = bars_held

            # TIME STOP: exit after max_bars if set (prevents theta bleed)
            max_bars = position.get("max_bars", 0)
            if max_bars > 0 and bars_held >= max_bars:
                position["current_premium"] = current_premium
                position["prev_close"] = bar["close"]
                return True, current_premium, "time_stop"

            # Theta decay: scale based on bar interval
            # For 15min bars: ~0.15% per bar (96 bars/day, daily theta ~15% spread across bars)
            # For daily bars: 0.8%-2.5% per bar (original)
            bar_interval = position.get("bar_interval", "day")
            if bar_interval == "15minute":
                # 15min bars: theta is tiny per bar but accumulates
                # ATM weekly: ~2-3% daily theta / 26 trading bars per day ≈ 0.1% per bar
                if bars_held <= 10:
                    theta_pct = 0.001  # 0.1% per 15min bar
                else:
                    theta_pct = 0.002  # Accelerates if held too long
            else:
                # Daily bars: original progressive decay
                if bars_held <= 3:
                    theta_pct = 0.008
                elif bars_held <= 6:
                    theta_pct = 0.015
                else:
                    theta_pct = 0.025
            theta_decay = current_premium * theta_pct

            # Estimate premium range using delta × index range
            index_move_high = bar["high"] - prev_close
            index_move_low = bar["low"] - prev_close
            index_move_close = bar["close"] - prev_close

            if position["direction"] == "bullish":
                # Call option: premium moves WITH index, minus theta
                premium_high = current_premium + delta * index_move_high - theta_decay
                premium_low = current_premium + delta * index_move_low - theta_decay
                premium_close = current_premium + delta * index_move_close - theta_decay
            else:
                # Put option: premium moves AGAINST index, minus theta
                premium_high = current_premium - delta * index_move_low - theta_decay
                premium_low = current_premium - delta * index_move_high - theta_decay
                premium_close = current_premium - delta * index_move_close - theta_decay

            # Clamp premiums — can't go below zero
            premium_high = max(premium_high, 0)
            premium_low = max(premium_low, 0)
            premium_close = max(premium_close, 0)

            # Check SL/target against premium, not index
            # SL hit: premium drops to SL level
            if sl > 0 and premium_low <= sl:
                # Update position premium to SL level for P&L calculation
                position["current_premium"] = sl
                position["prev_close"] = bar["close"]
                return True, sl, "stop_loss"

            # Target hit: premium rises to target level
            if target > 0 and premium_high >= target:
                position["current_premium"] = target
                position["prev_close"] = bar["close"]
                return True, target, "target"

            # TRAILING STOP with BREAKEVEN TRAP + PROFIT RUNNER
            # 4-stage system: breakeven → trail → lock → runner
            # Breakeven trap caps losses; runner stages let big winners fly
            risk_distance = entry_premium - sl if sl > 0 else entry_premium * 0.3

            if not position.get("breakeven_set", False):
                # STAGE 0 — BREAKEVEN TRAP: at configurable R:R, move SL to entry + costs
                # Converts full SL losses into near-zero losses (the KEY mechanism)
                be_ratio = position.get("breakeven_ratio", 0.4)
                breakeven_trigger = entry_premium + risk_distance * be_ratio
                if premium_high >= breakeven_trigger:
                    new_sl = entry_premium + risk_distance * 0.10
                    position["stop_loss"] = new_sl
                    position["breakeven_set"] = True
            elif not position.get("trailing_activated", False):
                # STAGE 1: at 1.0:1 R:R, lock 20% profit
                trail_trigger = entry_premium + risk_distance * 1.0
                if premium_high >= trail_trigger:
                    new_sl = entry_premium + risk_distance * 0.20
                    position["stop_loss"] = new_sl
                    position["trailing_activated"] = True
            elif not position.get("trailing_stage2", False):
                # STAGE 2: at 2.0:1 R:R, lock 50% profit (big move confirmed)
                trail_trigger_2 = entry_premium + risk_distance * 2.0
                if premium_high >= trail_trigger_2:
                    new_sl = entry_premium + risk_distance * 0.50
                    position["stop_loss"] = new_sl
                    position["trailing_stage2"] = True
            elif not position.get("trailing_stage3", False):
                # STAGE 3 — RUNNER: at 3.0:1 R:R, lock 70% (let it fly to target)
                trail_trigger_3 = entry_premium + risk_distance * 3.0
                if premium_high >= trail_trigger_3:
                    new_sl = entry_premium + risk_distance * 0.70
                    position["stop_loss"] = new_sl
                    position["trailing_stage3"] = True

            # Update running premium for next bar
            position["current_premium"] = premium_close
            position["prev_close"] = bar["close"]

            return False, 0, ""

        # Futures/equity: original logic — compare bar OHLC against SL/target
        if position["direction"] == "bullish":
            # Stop loss hit
            if sl > 0 and bar["low"] <= sl:
                return True, sl, "stop_loss"
            # Target hit
            if target > 0 and bar["high"] >= target:
                return True, target, "target"
        else:
            # Stop loss hit
            if sl > 0 and bar["high"] >= sl:
                return True, sl, "stop_loss"
            # Target hit
            if target > 0 and bar["low"] <= target:
                return True, target, "target"

        return False, 0, ""

    def _calculate_metrics(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        final_capital: float
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        initial = self.initial_capital

        if not self.trades:
            return BacktestResult(
                strategy=strategy_name,
                start_date=str(data["timestamp"].iloc[0])[:10] if "timestamp" in data.columns else "N/A",
                end_date=str(data["timestamp"].iloc[-1])[:10] if "timestamp" in data.columns else "N/A",
                initial_capital=initial,
                final_capital=final_capital,
                total_return_pct=0, annualized_return_pct=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                max_drawdown_pct=0, max_drawdown_duration_days=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                avg_trade_pnl=0, avg_hold_duration_min=0, total_costs=0,
                equity_curve=self.equity_curve, drawdown_curve=[],
                monthly_returns={}, trades=[]
            )

        # Basic metrics
        total_return = (final_capital - initial) / initial * 100
        days = max((pd.to_datetime(data["timestamp"].iloc[-1]) -
                     pd.to_datetime(data["timestamp"].iloc[0])).days, 1) if "timestamp" in data.columns else 365
        annual_return = total_return * 365 / days

        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
        total_wins = sum(t.net_pnl for t in wins)
        total_losses = abs(sum(t.net_pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        total_costs = sum(t.costs for t in self.trades)

        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = drawdown.max()

        # Max drawdown duration
        dd_duration = 0
        max_dd_duration = 0
        for i in range(1, len(equity)):
            if equity[i] < peak[i]:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Sharpe & Sortino
        if len(self.trades) > 1:
            daily_returns = np.diff(equity) / equity[:-1]
            daily_returns = daily_returns[daily_returns != 0]

            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe = np.mean(daily_returns) / daily_returns.std() * np.sqrt(252)
                downside_returns = daily_returns[daily_returns < 0]
                downside_std = downside_returns.std() if len(downside_returns) > 0 else daily_returns.std()
                sortino = np.mean(daily_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            else:
                sharpe = 0
                sortino = 0
        else:
            sharpe = 0
            sortino = 0

        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        return BacktestResult(
            strategy=strategy_name,
            start_date=str(data["timestamp"].iloc[0])[:10] if "timestamp" in data.columns else "N/A",
            end_date=str(data["timestamp"].iloc[-1])[:10] if "timestamp" in data.columns else "N/A",
            initial_capital=initial,
            final_capital=round(final_capital, 2),
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annual_return, 2),
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(win_rate, 1),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            avg_trade_pnl=round(np.mean([t.net_pnl for t in self.trades]), 2),
            avg_hold_duration_min=round(np.mean([t.hold_duration_minutes for t in self.trades]), 1),
            total_costs=round(total_costs, 2),
            equity_curve=self.equity_curve,
            drawdown_curve=drawdown.tolist(),
            monthly_returns={},
            trades=[{
                "entry": t.entry_time, "exit": t.exit_time,
                "direction": t.direction, "pnl": t.net_pnl,
                "entry_price": t.entry_price, "exit_price": t.exit_price,
            } for t in self.trades]
        )

    def monte_carlo_simulation(
        self,
        result: BacktestResult,
        num_simulations: int = 1000,
        num_trades: int = 100
    ) -> Dict:
        """
        Monte Carlo simulation — reshuffle trade sequence to test robustness.

        If a strategy is robust, its metrics should be stable across
        different orderings of the same trades.
        """
        if not result.trades:
            return {"error": "No trades to simulate"}

        trade_pnls = [t["pnl"] for t in result.trades]

        final_capitals = []
        max_drawdowns = []

        for _ in range(num_simulations):
            # Random sample of trades (with replacement)
            sampled = np.random.choice(trade_pnls, size=min(num_trades, len(trade_pnls)), replace=True)
            equity = [self.initial_capital]

            for pnl in sampled:
                equity.append(equity[-1] + pnl)

            equity_arr = np.array(equity)
            peak = np.maximum.accumulate(equity_arr)
            dd = (peak - equity_arr) / peak * 100

            final_capitals.append(equity[-1])
            max_drawdowns.append(dd.max())

        final_arr = np.array(final_capitals)
        dd_arr = np.array(max_drawdowns)

        return {
            "simulations": num_simulations,
            "median_final_capital": round(np.median(final_arr), 2),
            "mean_final_capital": round(np.mean(final_arr), 2),
            "p5_final_capital": round(np.percentile(final_arr, 5), 2),
            "p95_final_capital": round(np.percentile(final_arr, 95), 2),
            "p5_worst_capital": round(np.percentile(final_arr, 5), 2),
            "prob_profit": round(np.mean(final_arr > self.initial_capital) * 100, 1),
            "prob_20pct_drawdown": round(np.mean(dd_arr > 20) * 100, 1),
            "median_max_drawdown": round(np.median(dd_arr), 2),
            "p95_max_drawdown": round(np.percentile(dd_arr, 95), 2),
        }
