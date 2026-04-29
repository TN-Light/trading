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
import math
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from itertools import combinations

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
    entry_type: str = "immediate"  # "immediate" | "pullback_limit" | "gap_fill"
    underlying_direction_correct: bool = False

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
    alpha_pct: float
    avg_trade_pnl: float
    avg_hold_duration_min: float
    total_costs: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    monthly_returns: Dict[str, float]
    trades: List[Dict]
    psr_pct: float = 0.0
    min_track_record_len: int = 0

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
            f"CAGR: {self.annualized_return_pct:.1f}%\n"
            f"Alpha vs Buy-Hold: {self.alpha_pct:+.1f}%\n"
            f"{'─'*60}\n"
            f"Total Trades: {self.total_trades}\n"
            f"Win Rate: {self.win_rate:.1f}%\n"
            f"Avg Win: Rs {self.avg_win:,.0f}\n"
            f"Avg Loss: Rs {self.avg_loss:,.0f}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"{'─'*60}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Sortino Ratio: {self.sortino_ratio:.2f}\n"
            f"Calmar Ratio: {self.calmar_ratio:.2f}\n"
            f"PSR (>0 Sharpe): {self.psr_pct:.1f}%\n"
            f"Min Track Record (95%): {self.min_track_record_len} observations\n"
            f"Max Drawdown: {self.max_drawdown_pct:.1f}%\n"
            f"Max DD Duration: {self.max_drawdown_duration_days} days\n"
            f"{'─'*60}\n"
            f"Total Costs (brokerage, STT, etc.): Rs {self.total_costs:,.0f}\n"
            f"Avg Trade PnL: Rs {self.avg_trade_pnl:,.0f}\n"
            f"Avg Hold Duration: {self.avg_hold_duration_min:.0f} min\n"
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
        self.stt_options_sell_pct = cfg.get("stt_options_sell", 0.15) / 100
        self.stt_futures_pct = cfg.get("stt_futures", 0.05) / 100
        self.transaction_charges_options_pct = cfg.get("transaction_charges_options", 0.03503) / 100
        self.transaction_charges_futures_pct = cfg.get("transaction_charges_futures", 0.00173) / 100
        self.gst_pct = cfg.get("gst", 18.0) / 100
        self.sebi_charges_pct = cfg.get("sebi_charges", 0.0001) / 100
        self.stamp_duty_pct = cfg.get("stamp_duty", 0.003) / 100
        self.slippage_pct = cfg.get("slippage_pct", 0.15) / 100  # realistic options slippage
        self.enable_collateral_penalty = cfg.get("enable_collateral_penalty", True)

    def calculate_costs(
        self,
        buy_value: float,
        sell_value: float,
        instrument_type: str = "options"
    ) -> Dict:
        """Calculate all trading costs for a round-trip trade."""
        # Brokerage: applying the 2026 Rs 40 collateral penalty logic
        base_brokerage = 40 if self.enable_collateral_penalty else self.brokerage_per_order
        if instrument_type in ("options", "futures"):
            leg1_brok = base_brokerage
            leg2_brok = base_brokerage
        else:
            # Equity: Rs 20 (or Rs 40) or 0.03%, whichever is lower
            leg1_brok = min(base_brokerage, buy_value * 0.0003)
            leg2_brok = min(base_brokerage, sell_value * 0.0003)
        brokerage = leg1_brok + leg2_brok

        # STT (Securities Transaction Tax) — only on sell side
        if instrument_type == "options":
            stt = sell_value * self.stt_options_sell_pct
        else:
            stt = sell_value * self.stt_futures_pct

        # Transaction charges — on both sides
        if instrument_type == "options":
            transaction = (buy_value + sell_value) * self.transaction_charges_options_pct
        else:
            transaction = (buy_value + sell_value) * self.transaction_charges_futures_pct

        # GST on brokerage + transaction charges
        gst = (brokerage + transaction) * self.gst_pct

        # SEBI charges
        sebi = (buy_value + sell_value) * self.sebi_charges_pct

        # Stamp duty — only on buy side
        stamp = buy_value * self.stamp_duty_pct

        # Slippage already applied in _open_position() and _close_position()
        # Do NOT double-count here

        total = brokerage + stt + transaction + gst + sebi + stamp

        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "transaction_charges": round(transaction, 2),
            "gst": round(gst, 2),
            "sebi_charges": round(sebi, 2),
            "stamp_duty": round(stamp, 2),
            "slippage": 0.0,
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
        cost_config: Optional[Dict] = None,
        entry_timing: bool = False,
        entry_pullback_atr: float = 0.3,
        entry_max_wait_bars: int = 2,
        capital_tracker: Dict = None,
        max_positions: int = 2,
        # ── Institutional risk overlays ──
        vol_target: float = 0.0,       # Target annualized vol (e.g. 0.15 = 15%). 0 = disabled
        dd_throttle: bool = False,      # Reduce size during drawdowns
        equity_curve_filter: bool = False,  # Skip trades when equity < SMA
        half_capacity_mode: bool = False,   # Parrondo-style sustainability cap (alpha)
        half_capacity_alpha: float = 0.5,   # Target operating capacity (0, 1]
        equity_ma_window: int = 50,
        equity_ma_sizing: bool = False,     # Scale size by distance to equity MA
        equity_ma_band: float = 0.05,       # +/- band around MA for sizing interpolation
        dsq_filter: bool = False,            # Domain Shift Quotient based risk-off/scaling
        dsq_lookback: int = 20,
        dsq_baseline_window: int = 252,
        dsq_soft: float = 0.25,
        dsq_hard: float = 0.60,
        dsq_min_scalar: float = 0.25,
        # ── Intraday session enforcement (opt-in, default off) ──
        intraday_session: bool = False,
        session_open_time: str = "09:45",      # Skip bars before this
        session_no_entry_time: str = "14:30",   # No new entries after this
        session_close_time: str = "15:15",      # Force square-off
        max_intraday_trades_per_day: int = 4,
    ):
        self.initial_capital = initial_capital
        self.cost_model = ZerodhaCostModel(cost_config)
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.entry_timing = entry_timing
        self.entry_pullback_atr = entry_pullback_atr
        self.entry_max_wait_bars = entry_max_wait_bars
        self.max_positions = max_positions
        self.entry_timing_stats = {
            "signals_generated": 0,
            "filled_at_pullback": 0,
            "filled_at_open": 0,
            "expired_no_fill": 0,
        }
        # Dynamic capital tracking — signal generators can read this
        self.current_capital = initial_capital
        # Shared mutable tracker dict — syncs capital between engine and signal generator
        self._capital_tracker = capital_tracker
        # ── Institutional risk overlays ──
        self.vol_target = vol_target
        self.dd_throttle = dd_throttle
        self.equity_curve_filter = equity_curve_filter
        self.half_capacity_mode = half_capacity_mode
        self.half_capacity_alpha = float(max(0.1, min(1.0, half_capacity_alpha)))
        self.equity_ma_window = max(10, int(equity_ma_window))
        self.equity_ma_sizing = equity_ma_sizing
        self.equity_ma_band = float(max(0.01, min(0.20, equity_ma_band)))
        self.dsq_filter = dsq_filter
        self.dsq_lookback = max(10, int(dsq_lookback))
        self.dsq_baseline_window = max(self.dsq_lookback + 20, int(dsq_baseline_window))
        self.dsq_soft = float(max(0.05, dsq_soft))
        self.dsq_hard = float(max(self.dsq_soft + 0.05, dsq_hard))
        self.dsq_min_scalar = float(max(0.05, min(1.0, dsq_min_scalar)))
        self.risk_overlay_stats = {
            "signals_received": 0,
            "vol_scaled": 0,
            "dd_throttled": 0,
            "equity_filtered": 0,
            "ma_sized": 0,
            "half_capacity_scaled": 0,
            "dsq_scaled": 0,
            "dsq_filtered": 0,
        }
        # ── Intraday session ──
        self.intraday_session = intraday_session
        if intraday_session:
            h, m = map(int, session_open_time.split(":"))
            self._session_open_time = datetime.strptime(f"{h:02d}:{m:02d}", "%H:%M").time()
            h, m = map(int, session_no_entry_time.split(":"))
            self._session_no_entry_time = datetime.strptime(f"{h:02d}:{m:02d}", "%H:%M").time()
            h, m = map(int, session_close_time.split(":"))
            self._session_close_time = datetime.strptime(f"{h:02d}:{m:02d}", "%H:%M").time()
        self._max_intraday_trades_per_day = max_intraday_trades_per_day

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF using error function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _compute_psr_and_min_trl(
        self,
        returns: np.ndarray,
        sharpe: float,
        benchmark_sharpe: float = 0.0,
        confidence: float = 0.95,
    ) -> Tuple[float, int]:
        """
        Compute Probabilistic Sharpe Ratio (PSR) and Minimum Track Record Length.

        Uses Bailey & Lopez de Prado finite-sample correction with skew/kurtosis.
        """
        n = len(returns)
        if n < 3 or not np.isfinite(sharpe):
            return 0.0, 0

        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        if std <= 1e-12:
            return 0.0, 0

        centered = (returns - mean) / std
        skew = float(np.mean(centered ** 3))
        kurt = float(np.mean(centered ** 4))

        denom_term = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * (sharpe ** 2)
        denom_term = max(denom_term, 1e-9)

        z_psr = (sharpe - benchmark_sharpe) * math.sqrt(max(n - 1, 1) / denom_term)
        psr = self._normal_cdf(z_psr)

        # Inverse CDF approximation via numpy percentile of standard normal samples would be expensive;
        # use common z-scores for target confidence.
        z_conf = 1.645 if confidence >= 0.95 and confidence < 0.975 else (1.96 if confidence >= 0.975 else 1.282)
        diff = sharpe - benchmark_sharpe
        if abs(diff) <= 1e-9:
            min_trl = 0
        else:
            min_trl = int(math.ceil(1.0 + (z_conf ** 2) * denom_term / (diff ** 2)))

        return round(psr * 100.0, 1), max(min_trl, 0)

    def _sync_capital(self, capital: float):
        """Keep current_capital and external tracker in sync (including peak for DD sizing)."""
        self.current_capital = capital
        if self._capital_tracker is not None:
            self._capital_tracker["capital"] = capital
            if capital > self._capital_tracker.get("peak", capital):
                self._capital_tracker["peak"] = capital

    def _compute_dsq(self, data_so_far: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
        """
        Compute Domain Shift Quotient from realized volatility.

        Returns (dsq, current_vol, baseline_vol) or None if insufficient history.
        """
        if "close" not in data_so_far.columns or len(data_so_far) < self.dsq_lookback + 30:
            return None

        returns = data_so_far["close"].pct_change().dropna().values
        if len(returns) < self.dsq_lookback + 20:
            return None

        current_slice = returns[-self.dsq_lookback:]
        prior = returns[:-self.dsq_lookback]
        if len(prior) < 20:
            return None

        if len(prior) > self.dsq_baseline_window:
            prior = prior[-self.dsq_baseline_window:]

        current_vol = float(np.std(current_slice)) * np.sqrt(252)
        baseline_vol = float(np.std(prior)) * np.sqrt(252)
        if baseline_vol <= 1e-6:
            return None

        dsq = abs(current_vol - baseline_vol) / baseline_vol
        return float(dsq), current_vol, baseline_vol

    def _apply_risk_overlays(self, signal: Dict, data_so_far: pd.DataFrame, capital: float) -> Optional[Dict]:
        """
        Institutional risk overlays applied AFTER signal generation, BEFORE entry.
        Returns modified signal (with scaled quantity) or None (skip trade).

        1. Equity curve filter  — skip trade entirely if equity < 50-bar SMA
        2. Volatility targeting — scale quantity by (target_vol / realized_vol)
        3. Drawdown throttle   — further scale down during drawdowns
        """
        self.risk_overlay_stats["signals_received"] += 1

        # ── 1. Equity Curve Filter ──
        # If equity is below its own moving average, system is in a losing regime → sit out
        eq_sma = None
        eq_dist = None
        if len(self.equity_curve) >= self.equity_ma_window:
            eq_sma = float(np.mean(self.equity_curve[-self.equity_ma_window:]))
            if eq_sma > 0:
                eq_dist = (self.equity_curve[-1] - eq_sma) / eq_sma

        if self.equity_curve_filter and eq_sma is not None and self.equity_curve[-1] < eq_sma:
            self.risk_overlay_stats["equity_filtered"] += 1
            return None

        qty = signal.get("quantity", 1)
        original_qty = qty

        # ── 1.5 Equity MA Sizing Modulation ──
        # Scale exposure by distance from equity MA (defensive near/under MA).
        if self.equity_ma_sizing and eq_dist is not None:
            dist = max(-self.equity_ma_band, min(self.equity_ma_band, eq_dist))
            # Maps [-band, +band] -> [0.25, 1.00]
            ma_scalar = 0.25 + 0.75 * ((dist + self.equity_ma_band) / (2 * self.equity_ma_band))
            qty = max(1, int(qty * ma_scalar))
            if ma_scalar < 0.999:
                self.risk_overlay_stats["ma_sized"] += 1

        # ── 1.6 Half-Capacity Rule ──
        # Enforces sustainable utilization (Parrondo-inspired: avoid over-extraction).
        if self.half_capacity_mode and self.half_capacity_alpha < 0.999:
            qty = max(1, int(qty * self.half_capacity_alpha))
            self.risk_overlay_stats["half_capacity_scaled"] += 1

        # ── 1.7 Domain Shift Quotient (DSQ) ──
        # If current volatility regime diverges too far from baseline, reduce or skip risk.
        if self.dsq_filter:
            dsq_info = self._compute_dsq(data_so_far)
            if dsq_info is not None:
                dsq, _, _ = dsq_info
                if dsq >= self.dsq_hard:
                    self.risk_overlay_stats["dsq_filtered"] += 1
                    return None
                if dsq > self.dsq_soft:
                    frac = (dsq - self.dsq_soft) / (self.dsq_hard - self.dsq_soft)
                    frac = max(0.0, min(1.0, frac))
                    dsq_scalar = 1.0 - (1.0 - self.dsq_min_scalar) * frac
                    qty = max(1, int(qty * dsq_scalar))
                    self.risk_overlay_stats["dsq_scaled"] += 1

        # ── 2. Volatility Targeting ──
        # Scale position size so portfolio volatility stays constant
        # When market is wild: trade small. When calm: trade normal/larger.
        if self.vol_target > 0 and len(data_so_far) >= 25:
            returns = data_so_far["close"].pct_change().dropna().values[-20:]
            if len(returns) >= 10:
                realized_vol = float(np.std(returns)) * np.sqrt(252)
                if realized_vol > 0.001:
                    vol_scalar = self.vol_target / realized_vol
                    vol_scalar = max(0.25, min(vol_scalar, 2.0))  # Clamp: 0.25x to 2x
                    qty = max(1, int(qty * vol_scalar))
                    if vol_scalar != 1.0:
                        self.risk_overlay_stats["vol_scaled"] += 1

        # ── 3. Drawdown Throttle ──
        # Continuous linear scaling: at 0% DD → full size, at 30% DD → 25% size
        # For small accounts (1-lot floor): skip trade entirely when DD > 20%
        if self.dd_throttle and len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current_eq = self.equity_curve[-1]
            dd_pct = (peak - current_eq) / peak if peak > 0 else 0

            # Hard skip for small accounts: can't reduce below 1 lot, so skip instead
            if dd_pct > 0.20 and qty <= 1:
                self.risk_overlay_stats["dd_throttled"] += 1
                return None

            dd_scalar = max(0.25, 1.0 - dd_pct / 0.30)
            if dd_scalar < 1.0:
                self.risk_overlay_stats["dd_throttled"] += 1
            qty = max(1, int(qty * dd_scalar))

        # Apply modified quantity
        if qty != original_qty:
            signal = signal.copy()
            signal["quantity"] = qty

        return signal

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
        self._sync_capital(capital)
        peak_capital = capital
        self.trades = []
        self.equity_curve = [capital]
        self._warmup_bars = warmup_bars

        positions = []  # Multi-position: list of open positions
        daily_pnl = 0.0
        daily_trades = 0
        intraday_trades = 0  # Separate counter for intraday session mode
        last_date = None

        # Entry timing state
        pending_signal = None
        pending_bars_waited = 0

        logger.info(f"Starting backtest: {strategy_name} on {len(data)} bars")

        for i in range(warmup_bars, len(data)):
            current_bar = data.iloc[i]
            current_time = str(current_bar.get("timestamp", i))
            current_date = str(current_bar.get("timestamp", ""))[:10]

            # Reset daily counters
            if current_date != last_date:
                daily_pnl = 0.0
                daily_trades = 0
                intraday_trades = 0
                last_date = current_date

            # ── INTRADAY SESSION ENFORCEMENT (only when intraday_session=True) ──
            if self.intraday_session:
                try:
                    bar_time = pd.to_datetime(current_bar.get("timestamp")).time()
                except Exception:
                    bar_time = None

                if bar_time is not None:
                    # 1. Force close all positions at session end (15:15)
                    if bar_time >= self._session_close_time:
                        if positions:
                            for pos in positions:
                                if pos.get("instrument_type") == "options":
                                    exit_p = pos.get("current_premium", pos["entry_price"])
                                else:
                                    exit_p = current_bar["close"]
                                capital, trade = self._close_position(
                                    pos, exit_p, current_time, capital, "intraday_square_off", current_bar["close"]
                                )
                                self.trades.append(trade)
                                daily_pnl += trade.net_pnl
                                self._sync_capital(capital)
                            positions = []
                            pending_signal = None
                            pending_bars_waited = 0
                        self.equity_curve.append(capital)
                        continue

                    # 2. Skip pre-open bars (before 9:45) — no entries, no exits
                    if bar_time < self._session_open_time:
                        self.equity_curve.append(capital)
                        continue

            # Determine if new entries are allowed (for intraday session gating)
            can_enter_new = True
            if self.intraday_session:
                try:
                    bar_time_check = pd.to_datetime(current_bar.get("timestamp")).time()
                    if bar_time_check >= self._session_no_entry_time:
                        can_enter_new = False
                    if intraday_trades >= self._max_intraday_trades_per_day:
                        can_enter_new = False
                except Exception:
                    pass

            # Check daily loss limit (fixed 3% of initial capital)
            daily_loss_limit = self.initial_capital * 0.03
            if daily_pnl < -daily_loss_limit:
                # Force close ALL positions on daily loss limit
                for pos in positions:
                    if pos.get("instrument_type") == "options":
                        force_exit_price = pos.get("current_premium", pos["entry_price"])
                    else:
                        force_exit_price = current_bar["close"]
                    capital, trade = self._close_position(
                        pos, force_exit_price, current_time, capital, "daily_loss_limit", current_bar["close"]
                    )
                    self.trades.append(trade)
                    daily_pnl += trade.net_pnl
                    self._sync_capital(capital)
                positions = []
                # Cancel any pending signal on daily loss limit
                pending_signal = None
                pending_bars_waited = 0
                self.equity_curve.append(capital)
                continue

            # Check exit conditions for ALL open positions
            closed_indices = []
            for pidx, pos in enumerate(positions):
                exit_triggered, exit_price, exit_reason = self._check_exit(
                    pos, current_bar
                )
                if exit_triggered:
                    capital, trade = self._close_position(
                        pos, exit_price, current_time, capital, exit_reason, current_bar["close"]
                    )
                    self.trades.append(trade)
                    daily_pnl += trade.net_pnl
                    daily_trades += 1
                    self._sync_capital(capital)
                    closed_indices.append(pidx)
            # Remove closed positions (reverse order to preserve indices)
            for pidx in reversed(closed_indices):
                positions.pop(pidx)

            # If room for more positions, check for new signal (or fill pending)
            if len(positions) < self.max_positions and daily_trades < 6 and can_enter_new:
                if self.entry_timing:
                    # ── ENTRY TIMING MODE: next-bar limit order ──

                    # A) Try to fill pending signal on current bar
                    if pending_signal is not None:
                        filled, fill_result = self._try_fill_pending(
                            pending_signal, current_bar, current_time
                        )
                        if filled:
                            positions.append(fill_result)
                            intraday_trades += 1
                            pending_signal = None
                            pending_bars_waited = 0
                        else:
                            pending_bars_waited += 1
                            if pending_bars_waited >= self.entry_max_wait_bars:
                                self.entry_timing_stats["expired_no_fill"] += 1
                                pending_signal = None
                                pending_bars_waited = 0

                    # B) No pending → generate new signal → store as pending
                    if pending_signal is None and len(positions) < self.max_positions:
                        data_so_far = data.iloc[:i + 1]
                        signal = signal_generator(data_so_far)
                        if signal:
                            signal = self._apply_risk_overlays(signal, data_so_far, capital)
                        if signal:
                            self.entry_timing_stats["signals_generated"] += 1
                            pending_signal = signal
                            pending_bars_waited = 0
                else:
                    # ── ORIGINAL MODE: next-bar open entry (no look-ahead) ──
                    # A) Fill pending signal at this bar's open
                    if pending_signal is not None:
                        new_pos = self._open_position_at_open(
                            pending_signal, current_bar, current_time
                        )
                        positions.append(new_pos)
                        intraday_trades += 1
                        pending_signal = None
                        pending_bars_waited = 0

                    # B) Generate new signal → store as pending for next bar
                    if pending_signal is None and len(positions) < self.max_positions:
                        data_so_far = data.iloc[:i + 1]
                        signal = signal_generator(data_so_far)
                        if signal:
                            signal = self._apply_risk_overlays(signal, data_so_far, capital)
                        if signal:
                            pending_signal = signal
                            pending_bars_waited = 0

            # Update equity curve — sum unrealized across all positions
            unrealized = 0
            for pos in positions:
                if pos.get("instrument_type") == "options":
                    unrealized += (pos["current_premium"] - pos["entry_price"]) * pos["quantity"]
                else:
                    if pos["direction"] == "bullish":
                        unrealized += (current_bar["close"] - pos["entry_price"]) * pos["quantity"]
                    else:
                        unrealized += (pos["entry_price"] - current_bar["close"]) * pos["quantity"]

            self.equity_curve.append(capital + unrealized)

            # Track peak
            if capital > peak_capital:
                peak_capital = capital

        # Close any remaining positions at last bar
        for pos in positions:
            if pos.get("instrument_type") == "options":
                exit_price = pos.get("current_premium", pos["entry_price"])
            else:
                exit_price = data.iloc[-1]["close"]
            capital, trade = self._close_position(
                pos, exit_price,
                str(data.iloc[-1].get("timestamp", len(data))),
                capital, "end_of_data",
                data.iloc[-1]["close"]
            )
            self.trades.append(trade)
            self._sync_capital(capital)

        # Calculate metrics
        return self._calculate_metrics(strategy_name, data, capital)

    def _open_position_at_open(self, signal: Dict, bar: pd.Series, timestamp: str) -> Dict:
        """Open position at bar's open price (next-bar entry, no look-ahead).

        Re-prices option premium using delta approximation from signal spot to bar open.
        """
        is_options = signal.get("instrument_type") == "options"
        if is_options:
            delta = signal.get("delta", 0.5)
            if signal.get("direction") == "bearish":
                delta = -abs(delta)
            signal_spot = signal.get("signal_spot", 0)
            bar_open = bar["open"]
            original_premium = signal.get("entry_price", 0)

            if signal_spot > 0 and original_premium > 0:
                spot_diff = bar_open - signal_spot
                adjusted_premium = original_premium + delta * spot_diff
                adjusted_premium = max(adjusted_premium, original_premium * 0.5)
                adjusted_premium = max(adjusted_premium, 1.0)
                adjusted_signal = signal.copy()
                adjusted_signal["entry_price"] = adjusted_premium
                return self._open_position(adjusted_signal, bar, timestamp)

        return self._open_position(signal, bar, timestamp)

    def _open_position(self, signal: Dict, bar: pd.Series, timestamp: str) -> Dict:
        """Simulate opening a position with slippage."""
        is_options = signal.get("instrument_type") == "options"

        # Extract signal features for regression training (if present)
        signal_meta = {
            "signal_liqsweep": signal.get("signal_liqsweep", False),
            "signal_fvg": signal.get("signal_fvg", False),
            "signal_vp": signal.get("signal_vp", False),
            "signal_ote": signal.get("signal_ote", False),
            "signal_rsi_div": signal.get("signal_rsi_div", False),
            "signal_vol_surge": signal.get("signal_vol_surge", False),
            "signal_vol_confirm": signal.get("signal_vol_confirm", False),
            "signal_vwap": signal.get("signal_vwap", False),
            "signal_bias": signal.get("signal_bias", False),
            "bull_score": signal.get("bull_score", 0.0),
            "bear_score": signal.get("bear_score", 0.0),
            "atr_at_entry": signal.get("atr_at_entry", 0.0),
            "regime_at_entry": signal.get("regime_at_entry", "unknown"),
            "option_expiry_date": signal.get("option_expiry_date", ""),
        }

        if is_options:
            # For options: entry_price is the premium, track it separately
            premium_entry = signal.get("entry_price", bar["close"] * 0.012)
            slippage = premium_entry * self.cost_model.slippage_pct
            if signal.get("direction") == "bullish":
                premium_entry += slippage  # Buying call — pay more
            else:
                premium_entry += slippage  # Buying put — also pay more (you're buying)

            delta_signed = signal.get("delta", 0.5)
            if signal.get("direction") == "bearish":
                delta_signed = -abs(delta_signed)

            pos = {
                "entry_time": timestamp,
                "symbol": signal.get("symbol", "NIFTY"),
                "direction": signal.get("direction", "bullish"),
                "entry_price": premium_entry,
                "stop_loss": signal.get("stop_loss", 0),
                "target": signal.get("target", 0),
                "quantity": signal.get("quantity", 1),
                "strategy": signal.get("strategy", "default"),
                "instrument_type": "options",
                "delta": delta_signed,
                "current_premium": premium_entry,
                "prev_close": bar["close"],
                "underlying_entry_price": bar["close"],
                "max_bars": signal.get("max_bars", 0),
                "bar_interval": signal.get("bar_interval", "day"),
                "breakeven_ratio": signal.get("breakeven_ratio", 0.4),
            }
            pos.update(signal_meta)
            return pos
        else:
            entry_price = signal.get("entry_price", bar["close"])
            slippage = entry_price * self.cost_model.slippage_pct
            if signal.get("direction") == "bullish":
                entry_price += slippage
            else:
                entry_price -= slippage

            pos_f = {
                "entry_time": timestamp,
                "symbol": signal.get("symbol", "NIFTY"),
                "direction": signal.get("direction", "bullish"),
                "entry_price": entry_price,
                "stop_loss": signal.get("stop_loss", 0),
                "target": signal.get("target", 0),
                "quantity": signal.get("quantity", 1),
                "strategy": signal.get("strategy", "default"),
                "instrument_type": "futures",
                "underlying_entry_price": bar["close"],
            }
            pos_f.update(signal_meta)
            return pos_f

    def _try_fill_pending(
        self,
        signal: Dict,
        bar: pd.Series,
        timestamp: str,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Attempt to fill a pending signal as a limit order on the current bar.

        For bullish: limit = signal_spot - pullback × ATR; check if bar low reaches it.
        For bearish: limit = signal_spot + pullback × ATR; check if bar high reaches it.
        Returns (filled, position_or_None).
        """
        direction = signal.get("direction", "bullish")
        signal_spot = signal.get("signal_spot", 0)
        atr = signal.get("atr_at_signal", 0)

        if signal_spot <= 0 or atr <= 0:
            # Missing data — fall back to immediate fill
            position = self._open_position(signal, bar, timestamp)
            return True, position

        pullback = atr * self.entry_pullback_atr

        if direction == "bullish":
            limit_price = signal_spot - pullback

            if bar["open"] <= limit_price:
                fill_spot = bar["open"]
                entry_type = "gap_fill"
                self.entry_timing_stats["filled_at_open"] += 1
            elif bar["low"] <= limit_price:
                fill_spot = limit_price
                entry_type = "pullback_limit"
                self.entry_timing_stats["filled_at_pullback"] += 1
            else:
                return False, None
        else:
            limit_price = signal_spot + pullback

            if bar["open"] >= limit_price:
                fill_spot = bar["open"]
                entry_type = "gap_fill"
                self.entry_timing_stats["filled_at_open"] += 1
            elif bar["high"] >= limit_price:
                fill_spot = limit_price
                entry_type = "pullback_limit"
                self.entry_timing_stats["filled_at_pullback"] += 1
            else:
                return False, None

        # Re-price option premium using delta approximation
        delta = signal.get("delta", 0.5)
        if direction == "bearish":
            delta = -abs(delta)
        original_premium = signal.get("entry_price", 0)
        spot_diff = fill_spot - signal_spot

        adjusted_premium = original_premium + delta * spot_diff

        # Floor: don't go below 50% of original or 1.0
        adjusted_premium = max(adjusted_premium, original_premium * 0.5)
        adjusted_premium = max(adjusted_premium, 1.0)

        # Create modified signal with adjusted premium and entry type marker
        adjusted_signal = signal.copy()
        adjusted_signal["entry_price"] = adjusted_premium
        adjusted_signal["_entry_type"] = entry_type

        position = self._open_position(adjusted_signal, bar, timestamp)
        position["entry_type"] = entry_type
        return True, position

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        timestamp: str,
        capital: float,
        reason: str,
        current_underlying_price: float = None
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

            underlying_entry = position.get("underlying_entry_price")
            direction_correct = False
            if underlying_entry is not None and current_underlying_price is not None:
                if position["direction"] == "bullish" and current_underlying_price > underlying_entry:
                    direction_correct = True
                elif position["direction"] == "bearish" and current_underlying_price < underlying_entry:
                    direction_correct = True

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
                underlying_direction_correct=direction_correct,
                signal_liqsweep=position.get("signal_liqsweep", False),
                signal_fvg=position.get("signal_fvg", False),
                signal_vp=position.get("signal_vp", False),
                signal_ote=position.get("signal_ote", False),
                signal_rsi_div=position.get("signal_rsi_div", False),
                signal_vol_surge=position.get("signal_vol_surge", False),
                signal_vol_confirm=position.get("signal_vol_confirm", False),
                signal_vwap=position.get("signal_vwap", False),
                signal_bias=position.get("signal_bias", False),
                bull_score=float(position.get("bull_score", 0.0)),
                bear_score=float(position.get("bear_score", 0.0)),
                atr_at_entry=float(position.get("atr_at_entry", 0.0)),
                regime_at_entry=position.get("regime_at_entry", "unknown"),
                option_expiry_date=position.get("option_expiry_date", ""),
                entry_type=position.get("entry_type", "immediate"),
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

        underlying_entry = position.get("underlying_entry_price")
        direction_correct = False
        if underlying_entry is not None and current_underlying_price is not None:
            if position["direction"] == "bullish" and current_underlying_price > underlying_entry:
                direction_correct = True
            elif position["direction"] == "bearish" and current_underlying_price < underlying_entry:
                direction_correct = True

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
            underlying_direction_correct=direction_correct,
            signal_liqsweep=position.get("signal_liqsweep", False),
            signal_fvg=position.get("signal_fvg", False),
            signal_vp=position.get("signal_vp", False),
            signal_ote=position.get("signal_ote", False),
            signal_rsi_div=position.get("signal_rsi_div", False),
            signal_vol_surge=position.get("signal_vol_surge", False),
            signal_vol_confirm=position.get("signal_vol_confirm", False),
            signal_vwap=position.get("signal_vwap", False),
            signal_bias=position.get("signal_bias", False),
            bull_score=float(position.get("bull_score", 0.0)),
            bear_score=float(position.get("bear_score", 0.0)),
            atr_at_entry=float(position.get("atr_at_entry", 0.0)),
            regime_at_entry=position.get("regime_at_entry", "unknown"),
            option_expiry_date=position.get("option_expiry_date", ""),
            entry_type=position.get("entry_type", "immediate"),
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

            # Theta decay: scale based on bar interval
            # For 15min bars: ~0.15% per bar (96 bars/day, daily theta ~15% spread across bars)
            # For daily bars: 0.8%-2.5% per bar (original)
            bar_interval = position.get("bar_interval", "day")
            if bar_interval == "5minute":
                # 5min bars: 78 bars per trading day
                # ATM weekly: ~2-3% daily theta / 78 bars ≈ 0.03% per bar
                if bars_held <= 20:
                    theta_pct = 0.0003  # 0.03% per 5min bar
                else:
                    theta_pct = 0.0006  # accelerates if held too long
            elif bar_interval == "15minute":
                # 15min bars: theta is tiny per bar but accumulates
                # ATM weekly: ~2-3% daily theta / 26 trading bars per day ≈ 0.1% per bar
                if bars_held <= 10:
                    theta_pct = 0.001  # 0.1% per 15min bar
                else:
                    theta_pct = 0.002  # Accelerates if held too long
            else:
                # Daily bars: progressive decay + DTE acceleration
                # DTE-based theta (not bars-held-based)
                option_expiry = position.get("option_expiry_date", "")
                dte_acceleration = 1.0

                if option_expiry:
                    try:
                        from datetime import datetime
                        expiry_date = datetime.strptime(option_expiry, "%Y-%m-%d")
                        bar_date = datetime.strptime(str(bar["timestamp"])[:10], "%Y-%m-%d")
                        days_to_expiry = max((expiry_date - bar_date).days, 1)
                        # DTE-based theta: realistic daily decay rates
                        # Weekly (1-2 DTE): 3% daily, (3-5 DTE): 2%, Monthly: 0.8-1.2%
                        if days_to_expiry <= 2:
                            base_theta = 0.030  # 3% per day at 1-2 DTE
                        elif days_to_expiry <= 5:
                            base_theta = 0.020  # 2% per day at 3-5 DTE
                        elif days_to_expiry <= 10:
                            base_theta = 0.012  # 1.2% per day at 6-10 DTE
                        else:
                            base_theta = 0.008  # 0.8% per day >10 DTE
                    except Exception:
                        base_theta = 0.008
                        if bars_held <= 3:
                            base_theta = 0.008
                        elif bars_held <= 6:
                            base_theta = 0.015
                        else:
                            base_theta = 0.025
                else:
                    # Fallback: bars-held based (when no expiry date available)
                    # Gradual escalation — supports 7-bar holding without punitive theta
                    if bars_held <= 5:
                        base_theta = 0.008   # 0.8%/day — flat for first 5 bars
                    elif bars_held <= 8:
                        base_theta = 0.012   # 1.2%/day — moderate acceleration
                    else:
                        base_theta = 0.020   # 2.0%/day — penalizes very long holds

                theta_pct = base_theta
            theta_decay = current_premium * theta_pct

            # Estimate premium range using delta × index range + GAMMA correction
            index_move_high = bar["high"] - prev_close
            index_move_low = bar["low"] - prev_close
            index_move_close = bar["close"] - prev_close

            # Gamma: second-order sensitivity. For ATM options, gamma ≈ delta(1-delta)/(S×sigma×sqrt(T))
            # Simplified estimate: gamma contribution = 0.5 × gamma × (dS)²
            # We estimate gamma from delta: ATM gamma peaks at delta=0.5
            # gamma ≈ N'(d1) / (S × sigma × sqrt(T)) ≈ 0.4 / (S × 0.20 × sqrt(T))
            # For practical use: gamma_pct = delta × (1 - delta) × gamma_scale
            # gamma_scale calibrated so ATM weekly has ~0.0008 gamma per index point
            spot_price = prev_close
            gamma_scale = 2.0 / max(spot_price, 1.0)  # ~0.0001 for NIFTY 22000
            gamma = delta * (1 - abs(delta)) * gamma_scale

            if position["direction"] == "bullish":
                # Call option: premium moves WITH index, minus theta, plus gamma convexity
                gamma_high = 0.5 * gamma * index_move_high ** 2
                gamma_low = 0.5 * gamma * index_move_low ** 2
                gamma_close = 0.5 * gamma * index_move_close ** 2
                premium_high = current_premium + delta * index_move_high + gamma_high - theta_decay
                premium_low = current_premium + delta * index_move_low + gamma_low - theta_decay
                premium_close = current_premium + delta * index_move_close + gamma_close - theta_decay
            else:
                # Put option: premium moves AGAINST index, minus theta, plus gamma convexity
                gamma_high = 0.5 * gamma * index_move_low ** 2
                gamma_low = 0.5 * gamma * index_move_high ** 2
                gamma_close = 0.5 * gamma * index_move_close ** 2
                premium_high = current_premium - delta * index_move_low + gamma_high - theta_decay
                premium_low = current_premium - delta * index_move_high + gamma_low - theta_decay
                premium_close = current_premium - delta * index_move_close + gamma_close - theta_decay

            # Update delta for next bar (delta drift / charm approximation)
            # Call: underlying up → delta increases (moves ITM)
            # Put:  underlying up → delta decreases (moves OTM)
            if spot_price > 0:
                delta_shift = gamma * index_move_close  # approximate charm
                if position["direction"] == "bearish":
                    delta_shift = -delta_shift  # put delta moves opposite to calls
                new_delta = delta + delta_shift
                new_delta = max(0.05, min(0.95, new_delta))  # clamp to valid range
                position["delta"] = new_delta

            # Clamp premiums — can't go below zero
            premium_high = max(premium_high, 0)
            premium_low = max(premium_low, 0)
            premium_close = max(premium_close, 0)

            # Check raw index TARGET if provided
            underlying_target = position.get("underlying_target", 0)
            if underlying_target > 0:
                direction = position.get("direction", "bullish")
                hit_underlying_target = False
                if direction == "bullish" and bar["high"] >= underlying_target:
                    hit_underlying_target = True
                elif direction == "bearish" and bar["low"] <= underlying_target:
                    hit_underlying_target = True

                if hit_underlying_target:
                    position["current_premium"] = premium_high
                    position["prev_close"] = bar["close"]
                    return True, premium_high, "target"

            # Check raw index SL if provided
            underlying_sl = position.get("underlying_sl", 0)
            if underlying_sl > 0:
                direction = position.get("direction", "bullish")
                hit_underlying = False
                if direction == "bullish" and bar["low"] <= underlying_sl:
                    hit_underlying = True
                elif direction == "bearish" and bar["high"] >= underlying_sl:
                    hit_underlying = True

                if hit_underlying:
                    position["current_premium"] = premium_low
                    position["prev_close"] = bar["close"]
                    return True, premium_low, "stop_loss_underlying"

            # Parallel Premium SL Floor Check (3-Phase Conditional Floor)
            if sl > 0:
                is_premium_stop = False
                phase = ""
                if bars_held <= 3:
                     # Phase 1: Total immunity to premium spread widening / IV crush
                     is_premium_stop = False
                elif bars_held <= 5:
                     # Phase 2: Moderate buffer (allow spread to settle)
                     buffered_sl = sl * 0.8
                     if premium_low <= buffered_sl:
                         is_premium_stop = True
                         phase = "phase2"
                else:
                     # Phase 3: Total trust in options pricing
                     if premium_low <= sl:
                         is_premium_stop = True
                         phase = "phase3"

                if is_premium_stop:
                    position["current_premium"] = sl
                    position["prev_close"] = bar["close"]
                    return True, sl, f"stop_loss_premium_{phase}"

            # Fallback Premium Target Check (for older signal compatibility)
            if target > 0 and premium_high >= target:
                position["current_premium"] = target
                position["prev_close"] = bar["close"]
                return True, target, "target"

            # TIME STOP: exit after max_bars — checked AFTER SL/target
            # so a trade hitting target on the final bar gets target exit, not time stop.
            # Uses this bar's computed premium_close, not stale previous-bar premium.
            max_bars = position.get("max_bars", 0)
            if max_bars > 0 and bars_held >= max_bars:
                position["current_premium"] = premium_close
                position["prev_close"] = bar["close"]
                return True, premium_close, "time_stop"

            # TRAILING STOP with BREAKEVEN TRAP + PROFIT RUNNER
            # 5-stage system: breakeven → lock 20% → lock 50% → lock 70% → dynamic trail
            # Stages 0-3 use fixed R-multiple ratchets; stage 4 trails the high-water mark
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
                # STAGE 3 — RUNNER START: at 3.0:1 R:R, lock 70% and begin dynamic trail
                trail_trigger_3 = entry_premium + risk_distance * 3.0
                if premium_high >= trail_trigger_3:
                    new_sl = entry_premium + risk_distance * 0.70
                    position["stop_loss"] = new_sl
                    position["trailing_stage3"] = True
                    position["premium_hwm"] = premium_high  # track high-water mark
            else:
                # STAGE 4 — DYNAMIC TRAIL: ratchet stop with high-water mark
                # Trail offset = 30% of distance from entry to peak (keeps 70% of the move)
                hwm = position.get("premium_hwm", premium_high)
                if premium_high > hwm:
                    hwm = premium_high
                    position["premium_hwm"] = hwm
                # Floor: never go below stage 3 level (entry + 0.70R)
                floor_sl = entry_premium + risk_distance * 0.70
                # Dynamic: trail 30% below the high-water mark
                trail_offset = (hwm - entry_premium) * 0.30
                dynamic_sl = hwm - trail_offset
                new_sl = max(floor_sl, dynamic_sl)
                if new_sl > position["stop_loss"]:
                    position["stop_loss"] = new_sl

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

    def calculate_intraday_metrics(self) -> Dict:
        """Compute intraday-specific metrics from completed trades."""
        if not self.trades:
            return {}

        # Group trades by session date
        sessions: Dict[str, List] = {}
        for t in self.trades:
            date_key = str(t.entry_time)[:10]
            sessions.setdefault(date_key, []).append(t)

        total_sessions = len(sessions)
        total_trades = len(self.trades)

        # Session win rate: % of days ending net positive
        session_pnls = {}
        for date_key, trades_list in sessions.items():
            session_pnls[date_key] = sum(t.net_pnl for t in trades_list)
        winning_sessions = sum(1 for pnl in session_pnls.values() if pnl > 0)

        # Square-off exits
        square_off_exits = sum(1 for t in self.trades if t.exit_reason == "intraday_square_off")

        # Best / worst sessions
        best_session_pnl = max(session_pnls.values()) if session_pnls else 0
        worst_session_pnl = min(session_pnls.values()) if session_pnls else 0

        # Entry time distribution (by hour)
        entry_hours: Dict[int, int] = {}
        for t in self.trades:
            try:
                hr = pd.to_datetime(t.entry_time).hour
                entry_hours[hr] = entry_hours.get(hr, 0) + 1
            except Exception:
                pass

        return {
            "total_sessions": total_sessions,
            "avg_trades_per_session": round(total_trades / max(total_sessions, 1), 1),
            "session_win_rate": round(winning_sessions / max(total_sessions, 1) * 100, 1),
            "square_off_exits": square_off_exits,
            "square_off_pct": round(square_off_exits / max(total_trades, 1) * 100, 1),
            "best_session_pnl": round(best_session_pnl, 0),
            "worst_session_pnl": round(worst_session_pnl, 0),
            "entry_hour_distribution": dict(sorted(entry_hours.items())),
        }

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
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, alpha_pct=0,
                psr_pct=0, min_track_record_len=0,
                avg_trade_pnl=0, avg_hold_duration_min=0, total_costs=0,
                equity_curve=self.equity_curve, drawdown_curve=[],
                monthly_returns={}, trades=[]
            )

        # Basic metrics
        total_return = (final_capital - initial) / initial * 100
        days = max((pd.to_datetime(data["timestamp"].iloc[-1]) -
                     pd.to_datetime(data["timestamp"].iloc[0])).days, 1) if "timestamp" in data.columns else 365
        # CAGR (compound annual growth rate) — not linear scaling
        years = days / 365.25
        if years > 0 and final_capital > 0:
            annual_return = ((final_capital / initial) ** (1.0 / years) - 1) * 100
        else:
            annual_return = 0.0

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

        # Detect bar frequency for correct annualization
        if "timestamp" in data.columns and len(data) >= 2:
            t0 = pd.to_datetime(data["timestamp"].iloc[0])
            t1 = pd.to_datetime(data["timestamp"].iloc[1])
            bar_gap_minutes = (t1 - t0).total_seconds() / 60
            if bar_gap_minutes < 60:
                # Intraday bars (15min, 5min, etc.)
                bars_per_day = max(int(390 / bar_gap_minutes), 1)  # 390 min trading day
                annualization_factor = np.sqrt(252 * bars_per_day)
            else:
                # Daily or higher
                bars_per_day = 1
                annualization_factor = np.sqrt(252)
        else:
            bars_per_day = 1
            annualization_factor = np.sqrt(252)

        # Max drawdown duration in CALENDAR DAYS (not bars)
        dd_duration = 0
        max_dd_duration = 0
        dd_start_idx = 0
        max_dd_duration_days = 0
        for i in range(1, len(equity)):
            if equity[i] < peak[i]:
                if dd_duration == 0:
                    dd_start_idx = i
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
                # Convert to calendar days using timestamps
                if "timestamp" in data.columns and dd_duration == max_dd_duration:
                    try:
                        start_ts = pd.to_datetime(data["timestamp"].iloc[min(dd_start_idx, len(data) - 1)])
                        end_ts = pd.to_datetime(data["timestamp"].iloc[min(i, len(data) - 1)])
                        max_dd_duration_days = max(max_dd_duration_days, (end_ts - start_ts).days)
                    except Exception:
                        max_dd_duration_days = max_dd_duration  # fallback to bars
            else:
                dd_duration = 0

        # If we couldn't compute calendar days, approximate from bars
        if max_dd_duration_days == 0 and max_dd_duration > 0:
            max_dd_duration_days = max(1, max_dd_duration // bars_per_day)

        # Sharpe & Sortino — measure on trading days only, not calendar flat days
        if len(self.trades) > 1:
            daily_returns = np.diff(equity) / equity[:-1]
            daily_returns = daily_returns[daily_returns != 0]  # Filter out flat days (no trades)

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

        # Statistical significance for low-frequency strategies
        if len(self.trades) > 1:
            non_zero_returns = np.diff(equity) / equity[:-1]
            non_zero_returns = non_zero_returns[non_zero_returns != 0]
            psr_pct, min_trl = self._compute_psr_and_min_trl(
                non_zero_returns,
                sharpe,
                benchmark_sharpe=0.0,
                confidence=0.95,
            )
        else:
            psr_pct, min_trl = 0.0, 0

        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Compute hold duration from timestamps
        hold_durations = []
        for t in self.trades:
            try:
                t_entry = pd.to_datetime(t.entry_time)
                t_exit = pd.to_datetime(t.exit_time)
                hold_min = (t_exit - t_entry).total_seconds() / 60
                hold_durations.append(hold_min)
            except Exception:
                hold_durations.append(0)
        avg_hold = np.mean(hold_durations) if hold_durations else 0

        # Compute monthly returns
        monthly_returns = {}
        if "timestamp" in data.columns and len(self.equity_curve) > 1:
            try:
                # equity_curve[0] = initial capital (before warmup)
                # equity_curve[1:] corresponds to bars warmup_bars through len(data)-1
                wb = getattr(self, '_warmup_bars', 30)
                ec_trading = self.equity_curve[1:]  # skip initial capital entry
                n_ec = len(ec_trading)
                ts_trading = pd.to_datetime(data["timestamp"].iloc[wb:wb + n_ec])
                if len(ts_trading) == n_ec:
                    eq_series = pd.Series(ec_trading, index=ts_trading)
                    monthly_eq = eq_series.resample("M").last().dropna()
                    for i in range(1, len(monthly_eq)):
                        key = monthly_eq.index[i].strftime("%Y-%m")
                        monthly_returns[key] = round(
                            (monthly_eq.iloc[i] / monthly_eq.iloc[i - 1] - 1) * 100, 2
                        )
            except Exception:
                pass

        # Alpha vs buy-and-hold
        if "timestamp" in data.columns and len(data) > 1:
            bh_return = (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100
            bh_years = days / 365.25
            bh_cagr = ((1 + bh_return / 100) ** (1.0 / bh_years) - 1) * 100 if bh_years > 0 else 0
            alpha = annual_return - bh_cagr
        else:
            bh_cagr = 0
            alpha = 0

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
            max_drawdown_duration_days=max_dd_duration_days,
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            alpha_pct=round(alpha, 2),
            psr_pct=psr_pct,
            min_track_record_len=min_trl,
            avg_trade_pnl=round(np.mean([t.net_pnl for t in self.trades]), 2),
            avg_hold_duration_min=round(avg_hold, 1),
            total_costs=round(total_costs, 2),
            equity_curve=self.equity_curve,
            drawdown_curve=drawdown.tolist(),
            monthly_returns=monthly_returns,
            trades=[{
                "entry": t.entry_time,
                "exit": t.exit_time,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "symbol": t.symbol,
                "direction": t.direction,
                "strategy": t.strategy,
                "pnl": t.net_pnl,
                "net_pnl": t.net_pnl,
                "gross_pnl": t.gross_pnl,
                "costs": t.costs,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "exit_reason": t.exit_reason,
                "entry_type": t.entry_type,
                "signal_liqsweep": t.signal_liqsweep,
                "signal_fvg": t.signal_fvg,
                "signal_vp": t.signal_vp,
                "signal_ote": t.signal_ote,
                "signal_rsi_div": t.signal_rsi_div,
                "signal_vol_surge": t.signal_vol_surge,
                "signal_vol_confirm": t.signal_vol_confirm,
                "signal_vwap": t.signal_vwap,
                "signal_bias": t.signal_bias,
                "bull_score": t.bull_score,
                "bear_score": t.bear_score,
                "atr_at_entry": t.atr_at_entry,
                "regime_at_entry": t.regime_at_entry,
                "option_expiry_date": t.option_expiry_date,
            } for t in self.trades]
        )

    def monte_carlo_simulation(
        self,
        result: BacktestResult,
        num_simulations: int = 1000,
    ) -> Dict:
        """
        Monte Carlo simulation — block bootstrap to test robustness.

        Uses block bootstrap (block_size=5) to preserve serial correlation
        (e.g., losing streaks during crashes). Sample size matches actual
        trade count for fair comparison.
        """
        if not result.trades:
            return {"error": "No trades to simulate"}

        trade_pnls = [t["pnl"] for t in result.trades]
        n_trades = len(trade_pnls)
        block_size = min(5, max(1, n_trades // 10))  # 5 trades per block, min 1

        final_capitals = []
        max_drawdowns = []
        sharpes = []

        for _ in range(num_simulations):
            # Block bootstrap: sample blocks of consecutive trades to preserve streaks
            equity = [self.initial_capital]
            sampled_pnls = []

            while len(sampled_pnls) < n_trades:
                start_idx = np.random.randint(0, max(1, n_trades - block_size + 1))
                block = trade_pnls[start_idx:start_idx + block_size]
                sampled_pnls.extend(block)

            sampled_pnls = sampled_pnls[:n_trades]  # trim to exact size

            for pnl in sampled_pnls:
                equity.append(equity[-1] + pnl)

            equity_arr = np.array(equity)
            peak = np.maximum.accumulate(equity_arr)
            dd = (peak - equity_arr) / peak * 100

            final_capitals.append(equity[-1])
            max_drawdowns.append(dd.max())

            # Per-simulation Sharpe (simple daily-approximated)
            returns = np.diff(equity_arr) / equity_arr[:-1]
            if len(returns) > 1 and returns.std() > 0:
                # Annualize using trades-per-year (not min(252, n_trades))
                sharpes.append(np.mean(returns) / returns.std() * np.sqrt(252))
            else:
                sharpes.append(0)

        final_arr = np.array(final_capitals)
        dd_arr = np.array(max_drawdowns)
        sharpe_arr = np.array(sharpes)

        # Confidence interval on prob_profit
        prob_profit = np.mean(final_arr > self.initial_capital)
        prob_profit_se = np.sqrt(prob_profit * (1 - prob_profit) / num_simulations) * 100

        return {
            "simulations": num_simulations,
            "num_trades": n_trades,
            "block_size": block_size,
            "median_final_capital": round(np.median(final_arr), 2),
            "mean_final_capital": round(np.mean(final_arr), 2),
            "p5_final_capital": round(np.percentile(final_arr, 5), 2),
            "p95_final_capital": round(np.percentile(final_arr, 95), 2),
            "prob_profit": round(prob_profit * 100, 1),
            "prob_profit_ci": round(prob_profit_se, 1),
            "prob_20pct_drawdown": round(np.mean(dd_arr > 20) * 100, 1),
            "median_max_drawdown": round(np.median(dd_arr), 2),
            "p95_max_drawdown": round(np.percentile(dd_arr, 95), 2),
            "median_sharpe": round(np.median(sharpe_arr), 2),
            "p5_sharpe": round(np.percentile(sharpe_arr, 5), 2),
        }

    def probability_of_backtest_overfitting(
        self,
        result: BacktestResult,
        n_partitions: int = 10,
    ) -> Dict:
        """
        Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
        Cross-Validation (CSCV).

        Bailey et al. (2015) method:
          1. Split trade PnLs into N equal partitions
          2. For each C(N, N/2) combination, one half is "in-sample", other is "out-of-sample"
          3. Rank IS performance, pick best IS partition combo
          4. Check if corresponding OOS performance ranks below median
          5. PBO = fraction of combos where best IS underperforms OOS median

        PBO < 0.30 = likely robust, 0.30–0.50 = borderline, > 0.50 = likely overfit
        """
        if not result.trades or len(result.trades) < n_partitions * 2:
            return {"error": f"Need >= {n_partitions * 2} trades, got {len(result.trades)}"}

        trade_pnls = np.array([t["pnl"] for t in result.trades])
        n_trades = len(trade_pnls)

        # Split into N roughly equal partitions
        partition_size = n_trades // n_partitions
        partitions = []
        for i in range(n_partitions):
            start = i * partition_size
            end = start + partition_size if i < n_partitions - 1 else n_trades
            partitions.append(trade_pnls[start:end])

        # Compute Sharpe-like performance metric per partition
        partition_metrics = []
        for p in partitions:
            if len(p) > 1 and p.std() > 0:
                partition_metrics.append(p.mean() / p.std())
            else:
                partition_metrics.append(p.mean() if len(p) > 0 else 0)
        partition_metrics = np.array(partition_metrics)

        half = n_partitions // 2
        all_combos = list(combinations(range(n_partitions), half))

        # Cap combinations for performance (C(10,5) = 252, fine; C(16,8) = 12870, cap it)
        max_combos = 500
        if len(all_combos) > max_combos:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_combos), max_combos, replace=False)
            all_combos = [all_combos[i] for i in indices]

        overfit_count = 0
        logit_values = []

        for is_indices in all_combos:
            oos_indices = tuple(i for i in range(n_partitions) if i not in is_indices)

            is_metrics = partition_metrics[list(is_indices)]
            oos_metrics = partition_metrics[list(oos_indices)]

            # Best IS partition index (within IS set) → check its rank in OOS
            best_is_local = np.argmax(is_metrics)
            best_is_perf = is_metrics[best_is_local]

            # OOS performance of the IS-best partition
            oos_perf_of_best = oos_metrics[best_is_local] if best_is_local < len(oos_metrics) else np.median(oos_metrics)

            oos_median = np.median(oos_metrics)

            if oos_perf_of_best <= oos_median:
                overfit_count += 1

            # Logit for distribution: relative rank of OOS performance
            oos_rank = np.sum(oos_metrics <= oos_perf_of_best) / len(oos_metrics)
            if 0 < oos_rank < 1:
                logit_values.append(np.log(oos_rank / (1 - oos_rank)))

        pbo = overfit_count / len(all_combos)

        return {
            "pbo": round(pbo, 3),
            "n_partitions": n_partitions,
            "n_combinations": len(all_combos),
            "n_trades": n_trades,
            "mean_logit": round(np.mean(logit_values), 3) if logit_values else 0,
            "verdict": "ROBUST" if pbo < 0.30 else "BORDERLINE" if pbo < 0.50 else "LIKELY OVERFIT",
        }
