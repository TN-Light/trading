# ============================================================================
# PROMETHEUS — Master Orchestrator
# ============================================================================
"""
Entry point for the PROMETHEUS trading system.

Modes:
  - backtest   : Run backtests on historical data
  - paper      : Paper trading with live signals (no real money)
  - signal     : Signal-only mode (shows signals, user trades manually)
  - semi_auto  : Places orders with user confirmation
  - full_auto  : Autonomous execution within risk limits

Usage:
  python main.py backtest              # Run backtests
  python main.py paper                 # Paper trading
  python main.py signal                # Signal mode
  python main.py signal --symbol "NIFTY 50"
  python main.py scan                  # One-time scan for opportunities
  python main.py setup                 # First time setup wizard
"""

import sys
import os
import time
import signal as sig
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

import pandas as pd
import numpy as np

# ── PROMETHEUS Modules ───────────────────────────────────────────────────────
from prometheus.config import load_config, get, get_credential, get_risk_limits, get_capital_config
from prometheus.utils.logger import logger, setup_logging
from prometheus.utils.indian_market import (
    is_market_open, is_trading_day, days_to_expiry,
    get_lot_size, get_atm_strike, IST
)
from prometheus.utils.options_math import iv_percentile, iv_rank, black_scholes_price, calculate_greeks, OptionType

from prometheus.data.engine import DataEngine
from prometheus.data.store import DataStore

from prometheus.signals.technical import (
    calculate_vwap, calculate_volume_profile, detect_liquidity_sweeps,
    detect_fair_value_gaps, fibonacci_ote_levels, calculate_rsi,
    detect_rsi_divergence, calculate_atr, calculate_supertrend,
    TechnicalSignal
)
from prometheus.signals.oi_analyzer import OIAnalyzer
from prometheus.signals.regime_detector import RegimeDetector, RegimeState, MarketRegime
from prometheus.signals.fusion import SignalFusionEngine, FusedSignal

from prometheus.strategies.trend import TrendStrategy
from prometheus.strategies.volatility import VolatilityStrategy
from prometheus.strategies.expiry import ExpiryStrategy
from prometheus.strategies.selector import StrategySelector

from prometheus.risk.manager import RiskManager

from prometheus.backtest.engine import BacktestEngine, ZerodhaCostModel

from prometheus.execution.broker import BrokerBase, Order, OrderSide, OrderType, ProductType
from prometheus.execution.paper_trader import PaperTrader
from prometheus.execution.order_manager import OrderManager

from prometheus.interface.cli_dashboard import CLIDashboard
from prometheus.interface.telegram_bot import TelegramBot


# ═════════════════════════════════════════════════════════════════════════════
# PROMETHEUS CORE
# ═════════════════════════════════════════════════════════════════════════════
class Prometheus:
    """
    Master orchestrator — wires all subsystems together.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Load config
        cfg_path = config_path or str(PROJECT_ROOT / "config" / "settings.yaml")
        self.config = load_config(cfg_path)
        setup_logging(get("logging.level", "INFO"))

        logger.info("=" * 60)
        logger.info("  PROMETHEUS Trading System v1.0")
        logger.info("=" * 60)

        # Capital
        cap_cfg = get_capital_config()
        self.initial_capital = cap_cfg.get("initial", 200000)
        self.capital = self.initial_capital

        # Data Engine
        self.data = DataEngine(
            kite_api_key=get("broker.api_key", ""),
            kite_access_token=get("broker.access_token", ""),
        )
        self.store = DataStore()

        # Signal Engine
        sig_cfg = get("signals", {})
        self.fusion = SignalFusionEngine(
            min_confluence=sig_cfg.get("min_confluence", 0.65),
            min_rr=2.0,
        )
        self.oi_analyzer = OIAnalyzer()
        self.regime_detector = RegimeDetector()

        # Strategies
        strat_cfg = get("strategies", {})
        self.trend = TrendStrategy(strat_cfg.get("trend", {}), self.capital)
        self.volatility = VolatilityStrategy(strat_cfg.get("volatility", {}), self.capital)
        self.expiry = ExpiryStrategy(strat_cfg.get("expiry", {}), self.capital)
        self.selector = StrategySelector(strat_cfg, self.capital)

        # Risk Manager
        self.risk = RiskManager(get_risk_limits(), self.initial_capital)

        # Execution
        mode = get("system.mode", "paper")
        if mode in ("paper", "backtest", "signal"):
            self.broker = PaperTrader(self.initial_capital)
        else:
            from prometheus.execution.kite_executor import KiteExecutor
            self.broker = KiteExecutor(
                api_key=get("broker.api_key", ""),
                api_secret=get("broker.api_secret", ""),
                access_token=get("broker.access_token", ""),
            )
        self.order_manager = OrderManager(self.broker, self.risk, mode)

        # Interface
        self.dashboard = CLIDashboard()
        tg_cfg = get("interface.telegram", {})
        self.telegram = TelegramBot(
            bot_token=tg_cfg.get("bot_token", ""),
            chat_id=tg_cfg.get("chat_id", ""),
        )

        # AI (lazy load)
        self._ai = None

        # State
        self.mode = mode
        self.running = False
        self.symbols = get("market.indices", ["NIFTY 50", "NIFTY BANK"])

        logger.info(f"Mode: {self.mode} | Capital: Rs {self.capital:,.0f} | Symbols: {self.symbols}")

    @property
    def ai(self):
        """Lazy-load AI engine."""
        if self._ai is None:
            try:
                from prometheus.intelligence.llm_analyzer import IntelligenceEngine
                ai_cfg = get("ai", {})
                # Merge API keys from credentials into AI config
                groq_key = get_credential("groq.api_key")
                if groq_key:
                    ai_cfg.setdefault("groq", {})["api_key"] = groq_key
                gemini_key = get_credential("gemini.api_key")
                if gemini_key:
                    ai_cfg.setdefault("gemini", {})["api_key"] = gemini_key
                self._ai = IntelligenceEngine(ai_cfg)
                logger.info("AI Intelligence Engine loaded")
            except Exception as e:
                logger.warning(f"AI Engine not available: {e}")
                self._ai = None
        return self._ai

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS PIPELINE
    # ─────────────────────────────────────────────────────────────────────
    def analyze(self, symbol: str) -> Optional[FusedSignal]:
        """
        Run the full analysis pipeline on a single symbol.

        Pipeline:
        1. Fetch data (OHLCV + options chain)
        2. Run technical analysis
        3. Run OI analysis
        4. Detect regime
        5. Run AI analysis (if available)
        6. Fuse all signals
        7. Return actionable signal
        """
        logger.info(f"Analyzing {symbol}...")

        # 1. Fetch data
        data_daily = self.data.fetch_historical(symbol, days=120, interval="day")
        data_hourly = self.data.fetch_historical(symbol, days=30, interval="60minute")
        data_15m = self.data.fetch_historical(symbol, days=10, interval="15minute")

        if data_daily.empty:
            logger.warning(f"No data available for {symbol}")
            return None

        spot = data_daily["close"].iloc[-1] if not data_daily.empty else 0

        # 2. Technical signals
        tech_signals = []

        # VWAP
        if not data_15m.empty:
            vwap_df = calculate_vwap(data_15m)
            if not vwap_df.empty:
                last_close = data_15m["close"].iloc[-1]
                last_vwap = vwap_df["vwap"].iloc[-1]
                vwap_dir = "bullish" if last_close > last_vwap else "bearish"
                vwap_dist = abs(last_close - last_vwap) / last_vwap
                tech_signals.append(TechnicalSignal(
                    name="vwap",
                    direction=vwap_dir,
                    strength=min(vwap_dist * 20, 0.9),  # normalise
                    timeframe="15minute",
                ))

        # Supertrend
        if not data_15m.empty and len(data_15m) >= 20:
            st_df = calculate_supertrend(data_15m)
            if not st_df.empty and "supertrend_direction" in st_df.columns:
                st_dir = st_df["supertrend_direction"].iloc[-1]
                tech_signals.append(TechnicalSignal(
                    name="supertrend",
                    direction="bullish" if st_dir == 1 else "bearish",
                    strength=0.65,
                    timeframe="15minute",
                ))

        # RSI Divergence
        if not data_hourly.empty and len(data_hourly) > 44:
            div_result = detect_rsi_divergence(data_hourly, rsi_period=14)
            if div_result:
                tech_signals.append(TechnicalSignal(
                    name="rsi_divergence",
                    direction=div_result["direction"],
                    strength=div_result.get("strength", 0.5),
                    timeframe="60minute",
                ))

        # FVG / Imbalances
        if not data_15m.empty and len(data_15m) > 3:
            fvgs = detect_fair_value_gaps(data_15m)
            for f in fvgs[-3:]:
                direction = "bullish" if f.get("type") == "bullish_fvg" else "bearish"
                tech_signals.append(TechnicalSignal(
                    name="fvg_imbalance",
                    direction=direction,
                    strength=min(f.get("gap_pct", 0.1) / 0.5, 0.8),
                    timeframe="15minute",
                ))

        # Liquidity Sweeps
        if not data_15m.empty and len(data_15m) > 10:
            sweeps = detect_liquidity_sweeps(data_15m)
            for s in sweeps[-3:]:
                direction = "bullish" if s.get("type") == "bullish_sweep" else "bearish"
                tech_signals.append(TechnicalSignal(
                    name="liquidity_sweep",
                    direction=direction,
                    strength=s.get("strength", 0.5),
                    timeframe="15minute",
                ))

        # ATR for stop loss
        atr_value = 0.0
        if not data_daily.empty and len(data_daily) >= 15:
            atr_series = calculate_atr(data_daily)
            if not atr_series.empty:
                atr_value = float(atr_series.iloc[-1])

        # 3. OI Analysis
        oi_signals = []
        options_chain = self.data.fetch_options_chain(symbol)
        if not options_chain.empty:
            oi_signals = self.oi_analyzer.analyze(options_chain, spot)

        # 4. Regime Detection
        regime = self.regime_detector.detect(data_daily)

        # 5. AI Sentiment (if available)
        ai_sentiment = None
        if self.ai:
            try:
                change_pct = (
                    ((data_daily["close"].iloc[-1] / data_daily["close"].iloc[-2]) - 1) * 100
                    if len(data_daily) > 1 else 0
                )
                analysis = self.ai.analyze_market_context({
                    "symbol": symbol,
                    "spot_price": spot,
                    "change_pct": change_pct,
                    "regime": regime.regime.value,
                    "trend_strength": regime.trend_strength,
                    "vix": self.data.get_vix(),
                })
                if analysis:
                    ai_sentiment = analysis
            except Exception as e:
                logger.debug(f"AI analysis unavailable: {e}")

        # 6. Fuse signals
        signal = self.fusion.fuse(
            symbol=symbol,
            spot_price=spot,
            technical_signals=tech_signals,
            oi_signals=oi_signals,
            regime=regime,
            ai_sentiment=ai_sentiment,
        )

        # Attach ATR to signal for risk management
        if signal and atr_value > 0:
            if signal.stop_loss == 0 or signal.stop_loss == spot * 0.985:
                sl = self.risk.calculate_dynamic_stop_loss(spot, atr_value, signal.direction)
                signal.stop_loss = sl
                risk = abs(spot - sl)
                if risk > 0 and signal.target > 0:
                    signal.risk_reward = round(abs(signal.target - spot) / risk, 2)

        return signal

    def scan_all(self) -> list:
        """Scan all configured symbols and return signals."""
        signals = []
        for symbol in self.symbols:
            signal = self.analyze(symbol)
            if signal:
                signals.append(signal)
        return signals

    def refine_with_strategy(self, signal: FusedSignal) -> Dict:
        """
        Route a fused signal through the appropriate strategy module
        for strike selection, premium estimation, and position sizing.

        Returns an enriched signal dict ready for execution.
        """
        symbol = signal.symbol
        base_dict = signal.to_dict()

        if signal.action == "HOLD":
            return base_dict

        # Determine which strategy to use
        try:
            regime_enum = MarketRegime(signal.regime.lower()) if signal.regime else MarketRegime.UNKNOWN
        except ValueError:
            regime_enum = MarketRegime.UNKNOWN
        selection = self.selector.select(
            RegimeState(
                regime=regime_enum,
                confidence=signal.confidence,
                volatility_regime="normal",
                trend_strength=0,
                mean_reversion_score=0.5,
                recommended_strategies=[signal.strategy] if signal.strategy else [],
            ),
            vix=self.data.get_vix(),
        )
        strategy_name = selection.get("strategy", signal.strategy or "trend")

        # Fetch data needed by strategy modules
        data_hourly = self.data.fetch_historical(symbol, days=30, interval="60minute")
        data_15m = self.data.fetch_historical(symbol, days=10, interval="15minute")
        data_daily = self.data.fetch_historical(symbol, days=120, interval="day")
        options_chain = self.data.fetch_options_chain(symbol)
        spot = signal.entry_price

        try:
            if strategy_name == "trend" and not data_hourly.empty and not data_15m.empty:
                regime_state = self.regime_detector.detect(data_daily) if not data_daily.empty else None
                if regime_state:
                    oi_signals = []
                    oi_metrics = {}
                    if not options_chain.empty:
                        oi_signals = self.oi_analyzer.analyze(options_chain, spot)
                        oi_metrics = {"pcr": 1.0}  # fallback

                    setup = self.trend.generate_setup(
                        symbol=symbol,
                        spot_price=spot,
                        df_hourly=data_hourly,
                        df_15min=data_15m,
                        regime=regime_state,
                        oi_signals=oi_signals,
                        oi_metrics=oi_metrics,
                        options_chain=options_chain if not options_chain.empty else None,
                    )
                    if setup:
                        enriched = setup.to_dict()
                        enriched["confidence"] = signal.confidence
                        enriched["reasoning"] = signal.reasoning
                        enriched["contributing_signals"] = signal.contributing_signals
                        # Map action for order_manager
                        if setup.option_type == "CE":
                            enriched["action"] = "BUY_CE"
                        elif setup.option_type == "PE":
                            enriched["action"] = "BUY_PE"
                        enriched["direction"] = signal.direction
                        logger.info(f"Trend strategy refined: {setup.instrument} {setup.lots}L @ {setup.entry_price}")
                        return enriched

            elif strategy_name == "expiry" and not data_15m.empty:
                setup = self.expiry.check_expiry_opportunity(
                    symbol=symbol,
                    spot_price=spot,
                    data=data_15m,
                    options_chain=options_chain if not options_chain.empty else None,
                )
                if setup:
                    enriched = setup.to_dict()
                    enriched["action"] = base_dict["action"]
                    enriched["direction"] = signal.direction
                    enriched["confidence"] = signal.confidence
                    enriched["entry_price"] = setup.total_cost
                    logger.info(f"Expiry strategy refined: {setup.strategy_type}")
                    return enriched

        except Exception as e:
            logger.debug(f"Strategy refinement failed ({strategy_name}): {e}")

        # Fallback: return base fused signal dict (still valid for execution)
        return base_dict

    # ─────────────────────────────────────────────────────────────────────
    # MODE: SIGNAL (Show signals to user)
    # ─────────────────────────────────────────────────────────────────────
    def run_signal_mode(self, interval_seconds: int = 300):
        """
        Signal mode — continuously analyze and display signals.
        Does NOT place orders. User trades manually.
        """
        self.running = True
        self.dashboard.show_header()
        self.telegram.alert_system_start()
        logger.info("Starting SIGNAL mode...")

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.dashboard.show_status_line("Market holiday. Waiting...")
                    time.sleep(60)
                    continue

                if not is_market_open(now):
                    self.dashboard.show_status_line(
                        "Market closed. Next session: 9:15 AM IST"
                    )
                    time.sleep(60)
                    continue

                # Run analysis on all symbols
                for symbol in self.symbols:
                    signal = self.analyze(symbol)
                    if signal:
                        self.dashboard.show_signal(signal.to_dict())

                        if signal.action != "HOLD":
                            self.telegram.alert_new_signal(signal.to_dict())

                        # Show regime
                        data = self.data.fetch_historical(symbol, days=60, interval="day")
                        if not data.empty:
                            regime = self.regime_detector.detect(data)
                            selection = self.selector.select(regime)
                            self.dashboard.show_regime({
                                "regime": regime.regime.value,
                                "confidence": regime.confidence,
                                "volatility_state": regime.volatility_regime,
                                "trend_strength": regime.trend_strength,
                                "recommended_strategy": selection["strategy"],
                            })

                # Show risk status
                portfolio_state = self.risk.get_portfolio_state()
                self.dashboard.show_risk_status({
                    "daily_pnl": portfolio_state.realized_pnl_today,
                    "daily_limit": self.risk.max_daily_loss,
                    "weekly_pnl": portfolio_state.realized_pnl_week,
                    "weekly_limit": self.risk.max_weekly_loss,
                    "consecutive_losses": portfolio_state.consecutive_losses,
                    "system_halted": self.risk._halted,
                })

                self.dashboard.show_status_line(
                    f"Next scan in {interval_seconds}s. Press Ctrl+C to stop."
                )
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                self.running = False
                logger.info("Signal mode stopped by user")
                break
            except Exception as e:
                logger.error(f"Signal mode error: {e}")
                self.telegram.alert_system_error(str(e))
                time.sleep(30)

    # ─────────────────────────────────────────────────────────────────────
    # MODE: PAPER TRADING
    # ─────────────────────────────────────────────────────────────────────
    def run_paper_mode(self, interval_seconds: int = 300):
        """
        Paper trading — simulates real trading with virtual money.
        Places paper orders, tracks P&L, tests the full pipeline.
        """
        self.running = True
        self.dashboard.show_header()
        self.telegram.alert_system_start()
        logger.info("Starting PAPER TRADING mode...")

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.dashboard.show_status_line("Market holiday.")
                    time.sleep(60)
                    continue

                if not is_market_open(now):
                    # End of day summary
                    self._send_daily_summary()
                    self.dashboard.show_status_line("Market closed.")
                    time.sleep(60)
                    continue

                # Analyze and execute
                for symbol in self.symbols:
                    signal = self.analyze(symbol)
                    if signal and signal.action != "HOLD":
                        self.dashboard.show_signal(signal.to_dict())
                        self.telegram.alert_new_signal(signal.to_dict())

                        # Refine through strategy module for strike/premium/sizing
                        refined = self.refine_with_strategy(signal)

                        # Execute in paper mode
                        position = self.order_manager.execute_signal(
                            refined, confirm=False
                        )

                        if position:
                            self.dashboard.show_status_line(
                                f"Paper trade opened: {position.position_id}"
                            )

                # Update portfolio display
                portfolio_value = self.broker.get_portfolio_value() if hasattr(self.broker, "get_portfolio_value") else self.capital
                self.dashboard.show_portfolio_summary({
                    "initial_capital": self.initial_capital,
                    "equity": portfolio_value,
                    "daily_pnl": self.risk._daily_pnl,
                    "open_positions": len(self.broker.get_positions()),
                    "margin_used_pct": 0,
                })

                # Show positions
                positions = self.broker.get_positions()
                if positions:
                    pos_data = [{
                        "symbol": p.tradingsymbol,
                        "quantity": p.quantity,
                        "entry": p.average_price,
                        "ltp": p.last_price,
                        "pnl": p.unrealized_pnl,
                        "sl": 0,
                        "target": 0,
                        "strategy": "paper",
                    } for p in positions]
                    self.dashboard.show_positions(pos_data)

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                self.running = False
                # Close all paper positions
                total_pnl = self.order_manager.close_all_positions("session_end")
                logger.info(f"Paper trading stopped. Session P&L: Rs {total_pnl:+,.0f}")
                break
            except Exception as e:
                logger.error(f"Paper mode error: {e}")
                time.sleep(30)

    # ─────────────────────────────────────────────────────────────────────
    # BACKTEST HELPERS (shared by backtest, walkforward, sensitivity)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_daily_bias(self, data_daily: pd.DataFrame) -> dict:
        """Compute hourly bias map from daily data only (5-day structure analysis)."""
        bias_map = {}
        if len(data_daily) >= 10:
            for i in range(5, len(data_daily)):
                date_key = str(data_daily["timestamp"].iloc[i])[:10]
                recent_d = data_daily.iloc[max(0, i-5):i+1]
                d_highs = recent_d["high"].values
                d_lows = recent_d["low"].values
                d_hh = sum(1 for j in range(1, len(d_highs)) if d_highs[j] > d_highs[j-1])
                d_hl = sum(1 for j in range(1, len(d_lows)) if d_lows[j] > d_lows[j-1])
                d_lh = sum(1 for j in range(1, len(d_highs)) if d_highs[j] < d_highs[j-1])
                d_ll = sum(1 for j in range(1, len(d_lows)) if d_lows[j] < d_lows[j-1])
                d_bull = d_hh + d_hl
                d_bear = d_lh + d_ll
                if d_bull >= 3 and d_bull > d_bear + 1:
                    bias_map[date_key] = "bullish"
                elif d_bear >= 3 and d_bear > d_bull + 1:
                    bias_map[date_key] = "bearish"
                else:
                    bias_map[date_key] = "neutral"
        return bias_map

    def _make_signal_generator(
        self,
        regime_state,
        hourly_bias_map: dict,
        capital: float,
        primary_interval: str,
        symbol: str,
        param_overrides: dict = None,
        parrondo: bool = False,
    ):
        """
        Factory that returns a pro_signal_generator closure.

        When parrondo=True, enables per-bar regime detection and routes to
        trend-following or mean-reversion logic based on current regime
        (Parrondo's Paradox: alternating strategies beats either alone).

        param_overrides keys (all optional):
            - confluence_trending: float (default 2.5)
            - confluence_sideways: float (default 3.5)
            - target_atr_mult: float (default 3.0 for <50K)
            - time_stop_bars: int (overrides capital-adaptive default)
            - kelly_wr: float (default 0.35)
            - breakeven_ratio: float (default 0.4, passed through to engine)
            - recheck_bars: int (default 5 daily / 10 15min, Parrondo only)
            - mr_sl_atr: float (default 0.8, mean-reversion SL in ATR)
            - mr_target_atr: float (default 1.5, mean-reversion target in ATR)
            - mr_min_rr: float (default 1.5, mean-reversion minimum R:R)
            - mr_time_stop: int (default 3 daily / 8 15min)
            - mr_kelly_wr: float (default 0.50)
            - mr_breakeven_ratio: float (default 0.3)
            - mr_vwap_deviation: float (default 1.0, min ATR from VWAP)
            - mr_rsi_oversold: int (default 30)
            - mr_rsi_overbought: int (default 70)
        """
        overrides = param_overrides or {}

        # Capture regime detector for Parrondo per-bar detection
        regime_detector = self.regime_detector if parrondo else None
        if parrondo and regime_detector:
            regime_detector.reset_cache()
            recheck_interval = overrides.get(
                "recheck_bars", 5 if primary_interval == "day" else 10
            )

        # Parrondo regime transition tracking
        _prev_regime = [None]  # mutable container for nonlocal in closure
        _regime_transitions = []

        # ================================================================
        # SHARED HELPERS (used by both trend and mean-reversion paths)
        # ================================================================

        def _compute_indicators(recent_window, current, prev_bar, atr):
            """Compute all 8 professional indicators. Returns a dict of results."""
            # 1. LIQUIDITY SWEEPS
            sweeps = detect_liquidity_sweeps(recent_window, lookback=20, threshold_pct=0.001)
            recent_sweep = None
            sweep_direction = None
            if sweeps:
                recent_sweep = sweeps[-1]
                sweep_direction = "bullish" if recent_sweep["type"] == "bullish_sweep" else "bearish"

            # 2. FAIR VALUE GAPS
            fvgs = detect_fair_value_gaps(recent_window, min_gap_pct=0.0005)
            active_fvg = None
            fvg_direction = None
            for fvg in reversed(fvgs):
                if fvg["filled"]:
                    continue
                fvg_margin = atr * 0.3
                if fvg["type"] == "bullish_fvg" and (fvg["bottom"] - fvg_margin) <= current <= (fvg["top"] + fvg_margin):
                    active_fvg = fvg
                    fvg_direction = "bullish"
                    break
                elif fvg["type"] == "bearish_fvg" and (fvg["bottom"] - fvg_margin) <= current <= (fvg["top"] + fvg_margin):
                    active_fvg = fvg
                    fvg_direction = "bearish"
                    break

            # 3. VOLUME PROFILE
            vp = calculate_volume_profile(recent_window, lookback=20)
            poc = vp.get("poc", 0) if vp else 0
            va_high = vp.get("value_area_high", 0) if vp else 0
            va_low = vp.get("value_area_low", 0) if vp else 0
            vp_direction = None
            if poc > 0 and va_low > 0 and va_high > 0:
                tolerance = current * 0.004
                if abs(current - va_low) <= tolerance:
                    vp_direction = "bullish"
                elif abs(current - va_high) <= tolerance:
                    vp_direction = "bearish"
                elif abs(current - poc) <= tolerance:
                    if prev_bar["close"] < poc and current >= poc:
                        vp_direction = "bullish"
                    elif prev_bar["close"] > poc and current <= poc:
                        vp_direction = "bearish"

            # 4. FIBONACCI OTE
            ote_direction = None
            if len(recent_window) >= 20:
                recent_20 = recent_window.tail(20)
                swing_high = recent_20["high"].max()
                swing_low = recent_20["low"].min()

                if swing_high > swing_low:
                    bull_fib = fibonacci_ote_levels(swing_high, swing_low, "bullish")
                    bear_fib = fibonacci_ote_levels(swing_high, swing_low, "bearish")

                    if bull_fib["ote_zone_bottom"] <= current <= bull_fib["ote_zone_top"]:
                        ote_direction = "bullish"
                    elif bear_fib["ote_zone_bottom"] <= current <= bear_fib["ote_zone_top"]:
                        ote_direction = "bearish"

            # 5. RSI DIVERGENCE
            divergence = detect_rsi_divergence(recent_window) if len(recent_window) >= 50 else None
            div_direction = None
            if divergence:
                div_direction = divergence.get("direction")

            # 6. VOLUME SURGE
            vol_avg = recent_window["volume"].rolling(20).mean().iloc[-1] if len(recent_window) >= 20 else 0
            current_vol = recent_window["volume"].iloc[-1]
            volume_surge = current_vol > vol_avg * 1.3 if vol_avg > 0 else False

            # 7. VWAP
            vwap_df = calculate_vwap(recent_window.copy())
            vwap_val = vwap_df["vwap"].iloc[-1] if "vwap" in vwap_df.columns else current
            vwap_direction = "bullish" if current > vwap_val else "bearish"

            # 8. VOLUME ACCUMULATION/DISTRIBUTION
            vol_confirm_dir = None
            if len(recent_window) >= 5:
                recent_5 = recent_window.tail(5)
                price_change = recent_5["close"].iloc[-1] - recent_5["close"].iloc[0]
                vol_trend = recent_5["volume"].iloc[-1] > recent_5["volume"].iloc[0]
                if price_change > 0 and vol_trend:
                    vol_confirm_dir = "bullish"
                elif price_change < 0 and vol_trend:
                    vol_confirm_dir = "bearish"

            return {
                "recent_sweep": recent_sweep, "sweep_direction": sweep_direction,
                "fvg_direction": fvg_direction,
                "vp_direction": vp_direction, "poc": poc, "va_high": va_high, "va_low": va_low,
                "ote_direction": ote_direction,
                "div_direction": div_direction,
                "volume_surge": volume_surge,
                "vwap_val": vwap_val, "vwap_direction": vwap_direction,
                "vol_confirm_dir": vol_confirm_dir,
            }

        def _price_options(current, direction, atr, data_so_far):
            """Black-Scholes pricing. Returns (premium, delta, lot_size, strike, sigma) or None."""
            lot_size = get_lot_size(symbol)
            strike = get_atm_strike(current, symbol)

            bar_date = pd.Timestamp(data_so_far["timestamp"].iloc[-1])
            try:
                dte = days_to_expiry(symbol, from_date=bar_date.date())
            except Exception:
                dte = 5
            dte = max(dte, 1)
            T = dte / 365.0

            daily_vol = atr / current
            if primary_interval == "day":
                sigma = daily_vol * np.sqrt(252)
            else:
                sigma = daily_vol * np.sqrt(26 * 252)
            sigma = max(sigma, 0.10)
            sigma = min(sigma, 0.60)

            r = 0.07
            opt_type = OptionType.CALL if direction == "bullish" else OptionType.PUT

            premium = black_scholes_price(current, strike, T, r, sigma, opt_type)
            premium = max(premium, current * 0.003)

            greeks = calculate_greeks(current, strike, T, r, sigma, opt_type)
            delta = abs(greeks.get("delta", 0.5))
            delta = max(delta, 0.20)

            return premium, delta, lot_size, strike, sigma

        def _size_position(premium, premium_sl, lot_size):
            """Position sizing. Returns total_quantity or None if invalid."""
            if capital < 30000:
                risk_pct = 0.04
            elif capital < 75000:
                risk_pct = 0.03
            else:
                risk_pct = 0.02

            risk_per_trade = capital * risk_pct
            loss_per_lot = (premium - premium_sl) * lot_size
            if loss_per_lot <= 0:
                return None

            lots = max(1, int(risk_per_trade / loss_per_lot))
            premium_per_lot = premium * lot_size
            max_deploy = 0.45 if capital < 50000 else (0.35 if capital < 100000 else 0.25)
            max_lots = max(1, int((capital * max_deploy) / premium_per_lot)) if premium_per_lot > 0 else 1
            lots = min(lots, max_lots)
            return lots * lot_size

        def _build_signal(direction, premium, premium_sl, premium_target,
                          total_quantity, reasons, time_stop_bars, be_ratio, strategy_prefix="pro"):
            """Build the final signal dict."""
            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": premium,
                "stop_loss": premium_sl,
                "target": premium_target,
                "quantity": total_quantity,
                "strategy": f"{strategy_prefix}_{'+'.join(reasons)}",
                "instrument_type": "options",
                "delta": 0,  # placeholder, overwritten below
                "max_bars": time_stop_bars,
                "bar_interval": primary_interval,
                "breakeven_ratio": be_ratio,
            }

        # ================================================================
        # TREND SIGNAL GENERATION (existing logic)
        # ================================================================

        def _generate_trend_signal(data_so_far, recent_window, current, prev_bar,
                                   atr, bias, min_confluence, indicators):
            """Trend-following signal: directional confluence → BS pricing → sizing."""
            ind = indicators

            # --- Confluence scoring ---
            bull_score = 0
            bear_score = 0
            bull_reasons = []
            bear_reasons = []

            if ind["sweep_direction"] == "bullish":
                bull_score += 1.5; bull_reasons.append("LiqSweep")
            elif ind["sweep_direction"] == "bearish":
                bear_score += 1.5; bear_reasons.append("LiqSweep")

            if ind["fvg_direction"] == "bullish":
                bull_score += 1.5; bull_reasons.append("FVG")
            elif ind["fvg_direction"] == "bearish":
                bear_score += 1.5; bear_reasons.append("FVG")

            if ind["vp_direction"] == "bullish":
                bull_score += 1.0; bull_reasons.append("VP")
            elif ind["vp_direction"] == "bearish":
                bear_score += 1.0; bear_reasons.append("VP")

            if ind["ote_direction"] == "bullish":
                bull_score += 1.0; bull_reasons.append("OTE")
            elif ind["ote_direction"] == "bearish":
                bear_score += 1.0; bear_reasons.append("OTE")

            if ind["div_direction"] == "bullish":
                bull_score += 1.5; bull_reasons.append("RSI_Div")
            elif ind["div_direction"] == "bearish":
                bear_score += 1.5; bear_reasons.append("RSI_Div")

            if ind["volume_surge"]:
                if bull_score > bear_score:
                    bull_score += 0.5; bull_reasons.append("VolSurge")
                elif bear_score > bull_score:
                    bear_score += 0.5; bear_reasons.append("VolSurge")

            if ind["vol_confirm_dir"] == "bullish":
                bull_score += 0.5; bull_reasons.append("Acc")
            elif ind["vol_confirm_dir"] == "bearish":
                bear_score += 0.5; bear_reasons.append("Dist")

            if ind["vwap_direction"] == "bullish":
                bull_score += 0.5; bull_reasons.append("VWAP")
            else:
                bear_score += 0.5; bear_reasons.append("VWAP")

            if bias == "bullish":
                bull_score += 0.5
            elif bias == "bearish":
                bear_score += 0.5

            # --- Directional decision ---
            net_bull = bull_score - bear_score
            net_bear = bear_score - bull_score

            if bull_score >= min_confluence and net_bull >= 1.5:
                direction = "bullish"
                score = bull_score
                reasons = bull_reasons
            elif bear_score >= min_confluence and net_bear >= 1.5:
                direction = "bearish"
                score = bear_score
                reasons = bear_reasons
            else:
                return None

            # Bias filter
            if bias == "bullish" and direction == "bearish":
                return None
            if bias == "bearish" and direction == "bullish":
                return None

            # --- Options pricing ---
            pricing = _price_options(current, direction, atr, data_so_far)
            if pricing is None:
                return None
            premium, delta, lot_size, strike, sigma = pricing

            # --- SL & TARGET ---
            sl_atr_mult = 1.0 if capital < 50000 else (1.2 if capital < 100000 else 1.5)

            if ind["recent_sweep"] and ind["sweep_direction"] == direction:
                sl_level = ind["recent_sweep"]["level"]
                if direction == "bullish":
                    sl_index_move = current - sl_level + atr * 0.3
                else:
                    sl_index_move = sl_level - current + atr * 0.3
            else:
                sl_index_move = atr * sl_atr_mult

            sl_premium_drop = delta * sl_index_move
            premium_sl = max(premium - sl_premium_drop, premium * 0.35)

            risk_check = premium - premium_sl
            if risk_check <= 0:
                return None

            # Target ATR multiplier
            o_target_atr = overrides.get("target_atr_mult", None)
            if o_target_atr is not None:
                base_target = o_target_atr
            elif capital < 50000:
                base_target = 3.0
            elif capital < 100000:
                base_target = 2.5
            else:
                base_target = 2.0

            if score >= 5.0:
                target_multiplier = base_target + 1.0
            elif score >= 3.5:
                target_multiplier = base_target + 0.5
            else:
                target_multiplier = base_target

            target_index_move = atr * target_multiplier
            premium_target = premium + delta * target_index_move

            if capital < 50000:
                min_rr = 2.5
            elif capital < 100000:
                min_rr = 2.0
            else:
                min_rr = 1.5

            reward = premium_target - premium
            if risk_check > 0 and reward / risk_check < min_rr:
                premium_target = premium + risk_check * min_rr

            # Kelly gate
            kelly_wr = overrides.get("kelly_wr", 0.35)
            final_reward = premium_target - premium
            final_risk = premium - premium_sl
            ev = kelly_wr * final_reward - (1 - kelly_wr) * final_risk
            if ev <= 0:
                return None

            # --- Position sizing ---
            total_quantity = _size_position(premium, premium_sl, lot_size)
            if total_quantity is None:
                return None

            # --- Time stop ---
            o_time_stop = overrides.get("time_stop_bars", None)
            if o_time_stop is not None:
                time_stop_bars = o_time_stop
            elif primary_interval == "day":
                time_stop_bars = 5 if capital < 50000 else (4 if capital < 100000 else 3)
            else:
                time_stop_bars = 16 if capital < 50000 else (12 if capital < 100000 else 10)

            be_ratio = overrides.get("breakeven_ratio", 0.4)

            sig = _build_signal(direction, premium, premium_sl, premium_target,
                                total_quantity, reasons, time_stop_bars, be_ratio, "pro")
            sig["delta"] = delta
            return sig

        # ================================================================
        # MEAN-REVERSION SIGNAL GENERATION (Parrondo — new)
        # ================================================================

        def _generate_mean_reversion_signal(data_so_far, recent_window, current, prev_bar,
                                            atr, bias, bar_regime, indicators):
            """
            Mean-reversion signal for sideways/accumulation/distribution regimes.

            Fades overextended moves back toward VWAP/POC.
            Entry requires: VWAP deviation + RSI extreme + VP boundary + no active trend.
            """
            ind = indicators

            # --- Mean-reversion parameters (overridable) ---
            mr_vwap_dev = overrides.get("mr_vwap_deviation", 1.0)
            mr_rsi_os = overrides.get("mr_rsi_oversold", 30)
            mr_rsi_ob = overrides.get("mr_rsi_overbought", 70)

            vwap_val = ind["vwap_val"]
            poc = ind["poc"]
            va_high = ind["va_high"]
            va_low = ind["va_low"]

            # Need valid VWAP and VP data
            if vwap_val <= 0 or poc <= 0 or va_high <= 0 or va_low <= 0:
                return None

            # Check regime confirms no strong trend
            if bar_regime and abs(bar_regime.trend_strength) > 0.3:
                return None

            # --- RSI for mean-reversion ---
            rsi_series = calculate_rsi(recent_window["close"], period=14)
            if rsi_series.empty:
                return None
            current_rsi = rsi_series.iloc[-1]
            if pd.isna(current_rsi):
                return None

            # --- Determine mean-reversion direction ---
            vwap_distance = (current - vwap_val) / atr  # in ATR units
            direction = None
            reasons = []

            # Score-based mean-reversion (need 2+ conditions, not all 3)
            mr_bull_score = 0
            mr_bear_score = 0
            mr_bull_reasons = []
            mr_bear_reasons = []

            # VWAP deviation (strongest MR signal)
            if vwap_distance >= mr_vwap_dev:
                mr_bear_score += 1.5
                mr_bear_reasons.append("MR_VWAP")
            elif vwap_distance <= -mr_vwap_dev:
                mr_bull_score += 1.5
                mr_bull_reasons.append("MR_VWAP")

            # RSI extremes
            if current_rsi >= mr_rsi_ob:
                mr_bear_score += 1.5
                mr_bear_reasons.append("MR_RSI_OB")
            elif current_rsi <= mr_rsi_os:
                mr_bull_score += 1.5
                mr_bull_reasons.append("MR_RSI_OS")

            # Volume Profile boundary (price at/beyond VA edge, fading back to POC)
            if current >= va_high:
                mr_bear_score += 1.0
                mr_bear_reasons.append("MR_VA_H")
            elif current <= va_low:
                mr_bull_score += 1.0
                mr_bull_reasons.append("MR_VA_L")

            # RSI divergence confirmation
            if ind["div_direction"] == "bearish":
                mr_bear_score += 1.0
                mr_bear_reasons.append("RSI_Div")
            elif ind["div_direction"] == "bullish":
                mr_bull_score += 1.0
                mr_bull_reasons.append("RSI_Div")

            # Need score >= 2.5 (at least 2 conditions) and net edge >= 1.0
            mr_min_score = 2.5
            if mr_bear_score >= mr_min_score and mr_bear_score - mr_bull_score >= 1.0:
                direction = "bearish"
                reasons = mr_bear_reasons
            elif mr_bull_score >= mr_min_score and mr_bull_score - mr_bear_score >= 1.0:
                direction = "bullish"
                reasons = mr_bull_reasons

            if direction is None:
                return None

            # Bias filter (relaxed for MR — allow counter-bias with extra confirmation)
            if bias == "bullish" and direction == "bearish":
                return None
            if bias == "bearish" and direction == "bullish":
                return None

            # --- Options pricing ---
            pricing = _price_options(current, direction, atr, data_so_far)
            if pricing is None:
                return None
            premium, delta, lot_size, strike, sigma = pricing

            # --- SL & TARGET (tighter for mean-reversion) ---
            mr_sl = overrides.get("mr_sl_atr", 0.8)
            mr_target = overrides.get("mr_target_atr", 1.5)
            mr_min_rr = overrides.get("mr_min_rr", 1.5)

            sl_index_move = atr * mr_sl
            sl_premium_drop = delta * sl_index_move
            premium_sl = max(premium - sl_premium_drop, premium * 0.35)

            risk_check = premium - premium_sl
            if risk_check <= 0:
                return None

            # Target: aim for VWAP/POC reversion
            target_index_move = atr * mr_target
            premium_target = premium + delta * target_index_move

            # Enforce minimum R:R
            reward = premium_target - premium
            if risk_check > 0 and reward / risk_check < mr_min_rr:
                premium_target = premium + risk_check * mr_min_rr

            # Kelly gate (higher WR assumption for mean-reversion)
            mr_kelly = overrides.get("mr_kelly_wr", 0.50)
            final_reward = premium_target - premium
            final_risk = premium - premium_sl
            ev = mr_kelly * final_reward - (1 - mr_kelly) * final_risk
            if ev <= 0:
                return None

            # --- Position sizing ---
            total_quantity = _size_position(premium, premium_sl, lot_size)
            if total_quantity is None:
                return None

            # --- Time stop (shorter for mean-reversion) ---
            mr_ts = overrides.get("mr_time_stop", None)
            if mr_ts is not None:
                time_stop_bars = mr_ts
            elif primary_interval == "day":
                time_stop_bars = 3
            else:
                time_stop_bars = 8

            mr_be = overrides.get("mr_breakeven_ratio", 0.3)

            sig = _build_signal(direction, premium, premium_sl, premium_target,
                                total_quantity, reasons, time_stop_bars, mr_be, "mr")
            sig["delta"] = delta
            return sig

        # ================================================================
        # MAIN SIGNAL GENERATOR CLOSURE
        # ================================================================

        def pro_signal_generator(data_so_far: pd.DataFrame) -> Optional[Dict]:
            if len(data_so_far) < 50:
                return None

            recent_window = data_so_far.tail(100) if len(data_so_far) > 100 else data_so_far

            close = recent_window["close"]
            current = close.iloc[-1]
            current_bar = recent_window.iloc[-1]
            prev_bar = recent_window.iloc[-2] if len(recent_window) >= 2 else current_bar

            current_date = str(current_bar.get("timestamp", ""))[:10]
            bias = hourly_bias_map.get(current_date, "neutral")

            # --- ATR ---
            atr_series = calculate_atr(recent_window, period=14)
            atr = float(atr_series.iloc[-1]) if not atr_series.empty and len(atr_series) >= 14 else current * 0.005

            # --- INDICATORS (computed once, shared by both paths) ---
            indicators = _compute_indicators(recent_window, current, prev_bar, atr)

            # ================================================================
            # REGIME ROUTING
            # ================================================================
            if parrondo and regime_detector:
                # PER-BAR REGIME DETECTION (Parrondo mode)
                bar_regime = regime_detector.detect_fast(data_so_far, recheck_every=recheck_interval)
                regime_value = bar_regime.regime.value

                # Track transitions
                if _prev_regime[0] is not None and bar_regime.regime != _prev_regime[0]:
                    _regime_transitions.append({
                        "bar": len(data_so_far),
                        "from": _prev_regime[0].value,
                        "to": regime_value,
                        "confidence": bar_regime.confidence,
                    })
                _prev_regime[0] = bar_regime.regime

                # ROUTE based on detected regime
                if regime_value == "volatile":
                    # Capital preservation — skip trading in volatile regime
                    return None
                elif regime_value in ("accumulation", "distribution"):
                    # MEAN-REVERSION mode
                    mr_signal = _generate_mean_reversion_signal(
                        data_so_far, recent_window, current, prev_bar,
                        atr, bias, bar_regime, indicators
                    )
                    if mr_signal:
                        return mr_signal
                    # Fallback: try trend with higher confluence if MR doesn't fire
                    conf_sideways = overrides.get("confluence_sideways", 3.5)
                    return _generate_trend_signal(
                        data_so_far, recent_window, current, prev_bar,
                        atr, bias, conf_sideways, indicators
                    )
                else:
                    # TREND mode (markup, markdown, unknown)
                    conf_trending = overrides.get("confluence_trending", 2.5)
                    return _generate_trend_signal(
                        data_so_far, recent_window, current, prev_bar,
                        atr, bias, conf_trending, indicators
                    )
            else:
                # STATIC REGIME (original behavior — backward compatible)
                conf_trending = overrides.get("confluence_trending", 2.5)
                conf_sideways = overrides.get("confluence_sideways", 3.5)
                if regime_state:
                    regime = regime_state.regime.value
                    if regime in ("accumulation", "distribution", "unknown", "volatile"):
                        min_confluence = conf_sideways
                    else:
                        min_confluence = conf_trending
                else:
                    min_confluence = conf_trending

                return _generate_trend_signal(
                    data_so_far, recent_window, current, prev_bar,
                    atr, bias, min_confluence, indicators
                )

        # Attach transition log to the closure for post-backtest analysis
        pro_signal_generator.regime_transitions = _regime_transitions

        return pro_signal_generator

    def _run_backtest_on_slice(
        self,
        data_slice: pd.DataFrame,
        symbol: str,
        strategy_name: str = "default",
        param_overrides: dict = None,
        verbose: bool = True,
        parrondo: bool = False,
    ):
        """
        Run a backtest on an arbitrary date-sliced DataFrame.
        Handles regime detection, bias, and signal generator independently (no leakage).
        Returns (BacktestResult, BacktestEngine).
        """
        # Regime detection on this slice only (static — used when parrondo=False)
        regime_state = None
        if len(data_slice) >= 50:
            regime_state = self.regime_detector.detect(data_slice)
            if verbose:
                logger.info(f"  Regime: {regime_state.regime.value} | Confidence: {regime_state.confidence:.2f}")
                if parrondo:
                    logger.info("  Parrondo regime-switching: ENABLED (per-bar detection)")

        # Compute bias from daily data only (consistent for all periods)
        hourly_bias_map = self._compute_daily_bias(data_slice)
        if verbose:
            logger.info(f"  Bias map: {len(hourly_bias_map)} sessions")

        # Create signal generator
        capital = self.initial_capital
        signal_gen = self._make_signal_generator(
            regime_state=regime_state,
            hourly_bias_map=hourly_bias_map,
            capital=capital,
            primary_interval="day",
            symbol=symbol,
            param_overrides=param_overrides,
            parrondo=parrondo,
        )

        # Run backtest
        cost_cfg = get("backtest.costs", {})
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_config=cost_cfg,
        )

        result = engine.run(
            data=data_slice,
            signal_generator=signal_gen,
            strategy_name=strategy_name,
            warmup_bars=30,
        )

        if verbose:
            print(result.summary())

            # Monte Carlo
            if result.total_trades > 10:
                mc = engine.monte_carlo_simulation(result, num_simulations=1000)
                print(f"  MC: P(profit)={mc.get('prob_profit', 0):.1f}%, "
                      f"Median={mc.get('median_final_capital', 0):,.0f}, "
                      f"5th pct={mc.get('p5_final_capital', 0):,.0f}")

            # Exit reason analysis
            if engine.trades:
                reason_stats = {}
                for t in engine.trades:
                    r = getattr(t, 'exit_reason', 'unknown')
                    if r not in reason_stats:
                        reason_stats[r] = {"count": 0, "wins": 0, "total_pnl": 0}
                    reason_stats[r]["count"] += 1
                    if t.net_pnl > 0:
                        reason_stats[r]["wins"] += 1
                    reason_stats[r]["total_pnl"] += t.net_pnl
                print("  Exit Reasons:")
                for r, s in sorted(reason_stats.items(), key=lambda x: -x[1]["count"]):
                    wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                    print(f"    {r}: {s['count']} trades, {wr:.0f}% WR, PnL Rs {s['total_pnl']:+,.0f}")

            # Parrondo regime transition analysis
            if parrondo and hasattr(signal_gen, 'regime_transitions'):
                transitions = signal_gen.regime_transitions
                if transitions:
                    print(f"\n  Regime Transitions: {len(transitions)} switches")
                    # Count transitions by type
                    from_to_counts = {}
                    for t in transitions:
                        key = f"{t['from']} -> {t['to']}"
                        from_to_counts[key] = from_to_counts.get(key, 0) + 1
                    for key, count in sorted(from_to_counts.items(), key=lambda x: -x[1]):
                        print(f"    {key}: {count}x")

                    # Count strategy mode distribution
                    mr_trades = sum(1 for t in engine.trades if getattr(t, 'strategy', '').startswith('mr_'))
                    trend_trades = sum(1 for t in engine.trades if getattr(t, 'strategy', '').startswith('pro_'))
                    print(f"  Strategy Split: {trend_trades} trend, {mr_trades} mean-reversion")

        return result, engine

    # ─────────────────────────────────────────────────────────────────────
    # MODE: BACKTEST
    # ─────────────────────────────────────────────────────────────────────
    def run_backtest(
        self,
        symbol: str = "NIFTY 50",
        days: int = 365,
        strategy: str = "trend",
        parrondo: bool = False,
    ):
        """Run backtest on historical data using professional signal stack."""
        self.dashboard.show_header()
        logger.info(f"Starting backtest: {strategy} on {symbol} ({days} days)" +
                     (" [PARRONDO]" if parrondo else ""))

        # ================================================================
        # MULTI-TIMEFRAME DATA FETCH
        # ================================================================
        # yfinance limits: 15min = ~60 days, hourly = ~730 days, daily = 20+ years
        # For long backtests (>60 days): use daily bars as primary
        # For short backtests (<=59 days): use 15min bars as primary

        use_daily_primary = days > 59
        if use_daily_primary:
            logger.info(f"Long backtest ({days} days) — using DAILY bars as primary timeframe")
            data_primary = self.data.fetch_historical(symbol, days=days, interval="day")
            # For hourly bias — yfinance gives ~730 days max
            hourly_days = min(days, 729)
            data_hourly = self.data.fetch_historical(symbol, days=hourly_days, interval="60minute")
            data_daily = data_primary  # same data
            primary_interval = "day"
        else:
            logger.info(f"Short backtest ({days} days) — using 15min bars as primary timeframe")
            data_primary = self.data.fetch_historical(symbol, days=days, interval="15minute")
            data_hourly = self.data.fetch_historical(symbol, days=days, interval="60minute")
            data_daily = self.data.fetch_historical(symbol, days=max(days, 120), interval="day")
            primary_interval = "15minute"

            if data_primary.empty:
                logger.error("No 15min data available. Ensure yfinance is installed: pip install yfinance")
                logger.info("Falling back to daily bars...")
                data_primary = data_daily
                data_hourly = data_daily
                primary_interval = "day"
                use_daily_primary = True

        logger.info(f"Loaded: {len(data_primary)} x {primary_interval}, {len(data_hourly)} x hourly, {len(data_daily)} x daily bars")

        # ================================================================
        # REGIME DETECTION (from daily data — stable, not noisy)
        # ================================================================
        regime_state = None
        if len(data_daily) >= 50:
            regime_state = self.regime_detector.detect(data_daily)
            logger.info(f"Regime: {regime_state.regime.value} | Confidence: {regime_state.confidence:.2f} | "
                       f"Trend: {regime_state.trend_strength:+.2f} | Hurst: {regime_state.mean_reversion_score:.2f}")

        # ================================================================
        # HOURLY BIAS COMPUTATION (pre-compute for the whole dataset)
        # ================================================================
        # Determine market structure bias from hourly data:
        # Higher highs + higher lows = bullish bias
        # Lower highs + lower lows = bearish bias
        # For daily backtests >730 days: also compute bias from daily data as fallback
        hourly_bias_map = {}  # date_str -> "bullish" / "bearish" / "neutral"

        # Compute from hourly data (covers last ~730 days)
        if not data_hourly.empty and len(data_hourly) >= 10:
            hourly_vwap = calculate_vwap(data_hourly.copy())
            for i in range(10, len(data_hourly)):
                chunk = data_hourly.iloc[max(0, i-20):i+1]
                date_key = str(chunk["timestamp"].iloc[-1])[:10]

                # Structure: compare last 5 hourly candles
                recent_h = chunk.tail(5)
                highs = recent_h["high"].values
                lows = recent_h["low"].values

                # Higher highs and higher lows count
                hh_count = sum(1 for j in range(1, len(highs)) if highs[j] > highs[j-1])
                hl_count = sum(1 for j in range(1, len(lows)) if lows[j] > lows[j-1])
                lh_count = sum(1 for j in range(1, len(highs)) if highs[j] < highs[j-1])
                ll_count = sum(1 for j in range(1, len(lows)) if lows[j] < lows[j-1])

                # VWAP position
                close_val = recent_h["close"].iloc[-1]
                vwap_val = hourly_vwap["vwap"].iloc[i] if "vwap" in hourly_vwap.columns and i < len(hourly_vwap) else close_val

                bull_points = hh_count + hl_count + (1 if close_val > vwap_val else 0)
                bear_points = lh_count + ll_count + (1 if close_val < vwap_val else 0)

                if bull_points >= 4 and bull_points > bear_points + 1:
                    hourly_bias_map[date_key] = "bullish"
                elif bear_points >= 4 and bear_points > bull_points + 1:
                    hourly_bias_map[date_key] = "bearish"
                else:
                    hourly_bias_map[date_key] = "neutral"

        logger.info(f"Hourly bias computed for {len(hourly_bias_map)} sessions")

        # For long backtests: fill gaps with bias from daily data (5-day structure)
        if use_daily_primary and len(data_daily) >= 10:
            for i in range(5, len(data_daily)):
                date_key = str(data_daily["timestamp"].iloc[i])[:10]
                if date_key in hourly_bias_map:
                    continue  # hourly data already covers this date
                recent_d = data_daily.iloc[max(0, i-5):i+1]
                d_highs = recent_d["high"].values
                d_lows = recent_d["low"].values
                d_hh = sum(1 for j in range(1, len(d_highs)) if d_highs[j] > d_highs[j-1])
                d_hl = sum(1 for j in range(1, len(d_lows)) if d_lows[j] > d_lows[j-1])
                d_lh = sum(1 for j in range(1, len(d_highs)) if d_highs[j] < d_highs[j-1])
                d_ll = sum(1 for j in range(1, len(d_lows)) if d_lows[j] < d_lows[j-1])
                d_bull = d_hh + d_hl
                d_bear = d_lh + d_ll
                if d_bull >= 3 and d_bull > d_bear + 1:
                    hourly_bias_map[date_key] = "bullish"
                elif d_bear >= 3 and d_bear > d_bull + 1:
                    hourly_bias_map[date_key] = "bearish"
                else:
                    hourly_bias_map[date_key] = "neutral"
            logger.info(f"Bias map expanded to {len(hourly_bias_map)} sessions (daily fallback)")

        # ================================================================
        # PROFESSIONAL SIGNAL GENERATOR (adapts to primary timeframe)
        # ================================================================

        # When Parrondo mode is enabled, use the factory-generated signal
        # generator which supports per-bar regime detection and mean-reversion.
        # Otherwise, use the original inline signal generator (backward compatible).
        _use_parrondo_gen = parrondo
        if _use_parrondo_gen:
            self.regime_detector.reset_cache()
            pro_signal_generator = self._make_signal_generator(
                regime_state=regime_state,
                hourly_bias_map=hourly_bias_map,
                capital=self.initial_capital,
                primary_interval=primary_interval,
                symbol=symbol,
                parrondo=True,
            )

        def _inline_signal_generator(data_so_far: pd.DataFrame) -> Optional[Dict]:
            """
            Professional signal generator v4 — Smart Exits approach.

            PHILOSOPHY: Liberal entries + aggressive exit management = higher win rate.
            v1 had 56 trades/13.8% return but 44% WR. v3 over-filtered to 23 trades/4.5%.
            v4: keep v1's liberal entries, add trailing stop + adaptive R:R = more wins.

            ENTRY: min 2.5 confluence (trending) / 3.5 (sideways)
            EXIT: Trailing stop at 0.7:1 R:R, adaptive R:R (1.5-3.0 based on score)
            """
            if len(data_so_far) < 50:
                return None

            # PERFORMANCE: Limit indicator window to last 100 bars
            # All indicators use 14-50 bar lookbacks — no need for full history
            recent_window = data_so_far.tail(100) if len(data_so_far) > 100 else data_so_far

            close = recent_window["close"]
            current = close.iloc[-1]
            current_bar = recent_window.iloc[-1]
            prev_bar = recent_window.iloc[-2] if len(recent_window) >= 2 else current_bar

            # --- CONFIRMATION (soft — adds to score, not hard filter) ---
            bar_bullish = current_bar["close"] > current_bar["open"]
            bar_bearish = current_bar["close"] < current_bar["open"]
            prev_bar_bullish = prev_bar["close"] > prev_bar["open"] if len(data_so_far) >= 2 else False
            prev_bar_bearish = prev_bar["close"] < prev_bar["open"] if len(data_so_far) >= 2 else False

            # --- Get current date for hourly bias lookup ---
            current_date = str(current_bar.get("timestamp", ""))[:10]
            bias = hourly_bias_map.get(current_date, "neutral")

            # --- REGIME FILTER ---
            if regime_state:
                regime = regime_state.regime.value
                if regime in ("accumulation", "distribution", "unknown"):
                    regime_min_confluence = 3.5  # Moderate in sideways
                elif regime == "volatile":
                    regime_min_confluence = 3.5
                else:
                    regime_min_confluence = 2.5  # MARKUP/MARKDOWN — trend gives edge
            else:
                regime_min_confluence = 2.5

            # --- ATR ---
            atr_series = calculate_atr(recent_window, period=14)
            atr = float(atr_series.iloc[-1]) if not atr_series.empty and len(atr_series) >= 14 else current * 0.005

            # ================================================================
            # PROFESSIONAL INDICATORS
            # ================================================================

            # 1. LIQUIDITY SWEEPS
            sweeps = detect_liquidity_sweeps(recent_window, lookback=20, threshold_pct=0.001)
            recent_sweep = None
            sweep_direction = None
            if sweeps:
                recent_sweep = sweeps[-1]
                sweep_direction = "bullish" if recent_sweep["type"] == "bullish_sweep" else "bearish"

            # 2. FAIR VALUE GAPS
            fvgs = detect_fair_value_gaps(recent_window, min_gap_pct=0.0005)
            active_fvg = None
            fvg_direction = None
            for fvg in reversed(fvgs):
                if fvg["filled"]:
                    continue
                # Price within or near FVG (within half-ATR proximity)
                fvg_margin = atr * 0.3
                if fvg["type"] == "bullish_fvg" and (fvg["bottom"] - fvg_margin) <= current <= (fvg["top"] + fvg_margin):
                    active_fvg = fvg
                    fvg_direction = "bullish"
                    break
                elif fvg["type"] == "bearish_fvg" and (fvg["bottom"] - fvg_margin) <= current <= (fvg["top"] + fvg_margin):
                    active_fvg = fvg
                    fvg_direction = "bearish"
                    break

            # 3. VOLUME PROFILE (wider zones — within 0.4% of VA edge)
            vp = calculate_volume_profile(recent_window, lookback=20)
            poc = vp.get("poc", 0) if vp else 0
            va_high = vp.get("value_area_high", 0) if vp else 0
            va_low = vp.get("value_area_low", 0) if vp else 0
            vp_direction = None
            if poc > 0 and va_low > 0 and va_high > 0:
                tolerance = current * 0.004  # 0.4% zone
                if abs(current - va_low) <= tolerance:
                    vp_direction = "bullish"
                elif abs(current - va_high) <= tolerance:
                    vp_direction = "bearish"
                elif abs(current - poc) <= tolerance:
                    if prev_bar["close"] < poc and current >= poc:
                        vp_direction = "bullish"
                    elif prev_bar["close"] > poc and current <= poc:
                        vp_direction = "bearish"

            # 4. FIBONACCI OTE
            ote_direction = None
            if len(recent_window) >= 20:
                recent_20 = recent_window.tail(20)
                swing_high = recent_20["high"].max()
                swing_low = recent_20["low"].min()

                if swing_high > swing_low:
                    bull_fib = fibonacci_ote_levels(swing_high, swing_low, "bullish")
                    bear_fib = fibonacci_ote_levels(swing_high, swing_low, "bearish")

                    if bull_fib["ote_zone_bottom"] <= current <= bull_fib["ote_zone_top"]:
                        ote_direction = "bullish"
                    elif bear_fib["ote_zone_bottom"] <= current <= bear_fib["ote_zone_top"]:
                        ote_direction = "bearish"

            # 5. RSI DIVERGENCE
            divergence = detect_rsi_divergence(recent_window) if len(recent_window) >= 50 else None
            div_direction = None
            if divergence:
                div_direction = divergence.get("direction")

            # 6. VOLUME SURGE
            vol_avg = recent_window["volume"].rolling(20).mean().iloc[-1] if len(recent_window) >= 20 else 0
            current_vol = recent_window["volume"].iloc[-1]
            volume_surge = current_vol > vol_avg * 1.3 if vol_avg > 0 else False

            # 7. VWAP POSITION
            vwap_df = calculate_vwap(recent_window.copy())
            vwap_val = vwap_df["vwap"].iloc[-1] if "vwap" in vwap_df.columns else current
            vwap_direction = "bullish" if current > vwap_val else "bearish"

            # 8. VOLUME ACCUMULATION/DISTRIBUTION
            vol_confirm_dir = None
            if len(recent_window) >= 5:
                recent_5 = recent_window.tail(5)
                price_change = recent_5["close"].iloc[-1] - recent_5["close"].iloc[0]
                vol_trend = recent_5["volume"].iloc[-1] > recent_5["volume"].iloc[0]
                if price_change > 0 and vol_trend:
                    vol_confirm_dir = "bullish"
                elif price_change < 0 and vol_trend:
                    vol_confirm_dir = "bearish"

            # ================================================================
            # CONFLUENCE SCORING
            # ================================================================
            bull_score = 0
            bear_score = 0
            bull_reasons = []
            bear_reasons = []

            # Liquidity Sweep — 1.5 weight
            if sweep_direction == "bullish":
                bull_score += 1.5
                bull_reasons.append("LiqSweep")
            elif sweep_direction == "bearish":
                bear_score += 1.5
                bear_reasons.append("LiqSweep")

            # FVG — 1.5 weight
            if fvg_direction == "bullish":
                bull_score += 1.5
                bull_reasons.append("FVG")
            elif fvg_direction == "bearish":
                bear_score += 1.5
                bear_reasons.append("FVG")

            # Volume Profile — 1.0 weight
            if vp_direction == "bullish":
                bull_score += 1.0
                bull_reasons.append("VP")
            elif vp_direction == "bearish":
                bear_score += 1.0
                bear_reasons.append("VP")

            # Fibonacci OTE — 1.0 weight
            if ote_direction == "bullish":
                bull_score += 1.0
                bull_reasons.append("OTE")
            elif ote_direction == "bearish":
                bear_score += 1.0
                bear_reasons.append("OTE")

            # RSI Divergence — 1.5 weight
            if div_direction == "bullish":
                bull_score += 1.5
                bull_reasons.append("RSI_Div")
            elif div_direction == "bearish":
                bear_score += 1.5
                bear_reasons.append("RSI_Div")

            # Volume surge — 0.5 bonus to leading side
            if volume_surge:
                if bull_score > bear_score:
                    bull_score += 0.5
                    bull_reasons.append("VolSurge")
                elif bear_score > bull_score:
                    bear_score += 0.5
                    bear_reasons.append("VolSurge")

            # Volume Acc/Dist — 0.5 boost
            if vol_confirm_dir == "bullish":
                bull_score += 0.5
                bull_reasons.append("Acc")
            elif vol_confirm_dir == "bearish":
                bear_score += 0.5
                bear_reasons.append("Dist")

            # VWAP position — 0.5 bonus
            if vwap_direction == "bullish":
                bull_score += 0.5
                bull_reasons.append("VWAP")
            else:
                bear_score += 0.5
                bear_reasons.append("VWAP")

            # (Confirmation candle is now a hard filter, not scoring)

            # Hourly bias agreement — 0.5 bonus (NOT hard filter)
            if bias == "bullish":
                bull_score += 0.5
            elif bias == "bearish":
                bear_score += 0.5

            # ================================================================
            # DIRECTIONAL DECISION
            # ================================================================
            min_confluence = regime_min_confluence

            # Net score must have clear edge (directional clarity)
            net_bull = bull_score - bear_score
            net_bear = bear_score - bull_score

            if bull_score >= min_confluence and net_bull >= 1.5:
                direction = "bullish"
                score = bull_score
                reasons = bull_reasons
            elif bear_score >= min_confluence and net_bear >= 1.5:
                direction = "bearish"
                score = bear_score
                reasons = bear_reasons
            else:
                return None

            # ONLY hard filter: hourly bias must NOT actively disagree
            if bias == "bullish" and direction == "bearish":
                return None
            if bias == "bearish" and direction == "bullish":
                return None

            # (No confirmation candle filter — it kills trades without improving WR)

            # ================================================================
            # OPTIONS PRICING via Black-Scholes
            # ================================================================
            lot_size = get_lot_size(symbol)
            strike = get_atm_strike(current, symbol)

            # Historically-aware DTE: use bar timestamp, not today's date
            bar_date = pd.Timestamp(data_so_far["timestamp"].iloc[-1])
            try:
                dte = days_to_expiry(symbol, from_date=bar_date.date())
            except Exception:
                dte = 5
            dte = max(dte, 1)
            T = dte / 365.0

            # IV from ATR — annualization depends on bar timeframe
            daily_vol = atr / current
            if primary_interval == "day":
                # Daily bars: ATR is already daily-scale → sqrt(252)
                sigma = daily_vol * np.sqrt(252)
            else:
                # 15min bars: ATR is intra-bar scale → sqrt(26 bars/day × 252 days)
                sigma = daily_vol * np.sqrt(26 * 252)
            sigma = max(sigma, 0.10)
            sigma = min(sigma, 0.60)

            r = 0.07
            opt_type = OptionType.CALL if direction == "bullish" else OptionType.PUT

            premium = black_scholes_price(current, strike, T, r, sigma, opt_type)
            premium = max(premium, current * 0.003)

            greeks = calculate_greeks(current, strike, T, r, sigma, opt_type)
            delta = abs(greeks.get("delta", 0.5))
            delta = max(delta, 0.20)

            # ================================================================
            # SL & TARGET — Maximum Extraction (breakeven trap = free optionality)
            # ================================================================
            # KEY INSIGHT: breakeven trap caps losses near zero, so wider targets
            # are FREE upside. Push targets aggressively.
            #
            # SL: tight (1.0 ATR for <50K, 1.2 for <100K, 1.5 for >100K)
            sl_atr_mult = 1.0 if capital < 50000 else (1.2 if capital < 100000 else 1.5)

            if recent_sweep and sweep_direction == direction:
                sl_level = recent_sweep["level"]
                if direction == "bullish":
                    sl_index_move = current - sl_level + atr * 0.3
                else:
                    sl_index_move = sl_level - current + atr * 0.3
            else:
                sl_index_move = atr * sl_atr_mult

            sl_premium_drop = delta * sl_index_move
            premium_sl = max(premium - sl_premium_drop, premium * 0.35)

            # KELLY FILTER: Skip low-edge trades
            # Kelly f* = (b*p - q) / b where b=win/loss ratio, p=win rate
            # Using conservative estimates: p=0.35 (with breakeven trap), b=3.5
            # Full Kelly = 10%, Half Kelly = 5%, Quarter Kelly = 2.5%
            # If this trade's R:R is below 2:1, the Kelly edge is marginal → skip
            risk_check = premium - premium_sl
            if risk_check <= 0:
                return None

            # Targets: WIDE. Breakeven trap makes this free.
            # Small capital (<50K): 3.0-4.0 ATR (need Rs 2K+ per winner)
            # Medium capital (50K-1L): 2.5-3.5 ATR
            # Large capital (>1L): 2.0-3.0 ATR
            if capital < 50000:
                base_target = 3.0
            elif capital < 100000:
                base_target = 2.5
            else:
                base_target = 2.0

            if score >= 5.0:
                target_multiplier = base_target + 1.0  # High conviction: max extraction
            elif score >= 3.5:
                target_multiplier = base_target + 0.5  # Medium conviction
            else:
                target_multiplier = base_target          # Still wide

            target_index_move = atr * target_multiplier
            premium_target = premium + delta * target_index_move

            # Enforce minimum R:R — 2.5:1 for <50K, 2.0:1 for <100K, 1.5:1 larger
            if capital < 50000:
                min_rr = 2.5
            elif capital < 100000:
                min_rr = 2.0
            else:
                min_rr = 1.5

            reward = premium_target - premium
            if risk_check > 0 and reward / risk_check < min_rr:
                premium_target = premium + risk_check * min_rr

            # KELLY GATE: Skip if expected value is negative
            # Conservative: assume 35% WR (breakeven trap adjusted)
            # EV = p * reward - q * risk > 0
            final_reward = premium_target - premium
            final_risk = premium - premium_sl
            ev = 0.35 * final_reward - 0.65 * final_risk
            if ev <= 0:
                return None

            # ================================================================
            # POSITION SIZING — Aggressive for small capital (breakeven protects)
            # ================================================================
            # Risk per trade: 4% for <30K, 3% for 30K-75K, 2% for >75K
            # Breakeven trap means actual realized losses are ~Rs 64 avg, not full SL
            if capital < 30000:
                risk_pct = 0.04
            elif capital < 75000:
                risk_pct = 0.03
            else:
                risk_pct = 0.02

            risk_per_trade = capital * risk_pct
            loss_per_lot = (premium - premium_sl) * lot_size
            if loss_per_lot <= 0:
                return None

            lots = max(1, int(risk_per_trade / loss_per_lot))
            premium_per_lot = premium * lot_size
            max_deploy = 0.45 if capital < 50000 else (0.35 if capital < 100000 else 0.25)
            max_lots = max(1, int((capital * max_deploy) / premium_per_lot)) if premium_per_lot > 0 else 1
            lots = min(lots, max_lots)
            total_quantity = lots * lot_size

            # Time stop — timeframe-dependent
            if primary_interval == "day":
                # Daily bars: 3-5 trading days
                if capital < 50000:
                    time_stop_bars = 5
                elif capital < 100000:
                    time_stop_bars = 4
                else:
                    time_stop_bars = 3
            else:
                # 15min bars: 10-16 intraday bars
                if capital < 50000:
                    time_stop_bars = 16   # 4 hours
                elif capital < 100000:
                    time_stop_bars = 12
                else:
                    time_stop_bars = 10

            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": premium,
                "stop_loss": premium_sl,
                "target": premium_target,
                "quantity": total_quantity,
                "strategy": f"pro_{'+'.join(reasons)}",
                "instrument_type": "options",
                "delta": delta,
                "max_bars": time_stop_bars,
                "bar_interval": primary_interval,
            }

        # Select signal generator: Parrondo (factory) or original (inline)
        if not _use_parrondo_gen:
            pro_signal_generator = _inline_signal_generator

        # Track capital across bars for dynamic sizing
        capital = self.initial_capital

        # Run backtest on primary timeframe data
        cost_cfg = get("backtest.costs", {})
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_config=cost_cfg,
        )

        logger.info(f"Running backtest on {len(data_primary)} bars of {primary_interval} data..."
                     + (" [PARRONDO]" if parrondo else ""))

        result = engine.run(
            data=data_primary,
            signal_generator=pro_signal_generator,
            strategy_name=f"pro_{strategy}_{symbol.replace(' ', '_')}",
            warmup_bars=30,
        )

        # Display results
        self.dashboard.show_backtest_results(result)
        print(result.summary())

        # Show regime context
        if regime_state:
            print(f"\n--- Regime Context ---")
            print(f"  {regime_state.details}")
            print(f"  Recommended strategies: {regime_state.recommended_strategies}")

        # Show hourly bias distribution
        if hourly_bias_map:
            bias_counts = {}
            for b in hourly_bias_map.values():
                bias_counts[b] = bias_counts.get(b, 0) + 1
            print(f"\n--- Hourly Bias Distribution ---")
            for b, c in sorted(bias_counts.items()):
                print(f"  {b}: {c} sessions ({c/len(hourly_bias_map)*100:.0f}%)")

        # Monte Carlo robustness
        if result.total_trades > 10:
            mc = engine.monte_carlo_simulation(result, num_simulations=1000)
            print("\n--- Monte Carlo Simulation (1000 runs) ---")
            print(f"  Probability of profit: {mc.get('prob_profit', 0):.1f}%")
            print(f"  Median final capital: Rs {mc.get('median_final_capital', 0):,.0f}")
            print(f"  5th percentile (worst case): Rs {mc.get('p5_final_capital', 0):,.0f}")
            print(f"  95th percentile (best case): Rs {mc.get('p95_final_capital', 0):,.0f}")
            print(f"  Median max drawdown: {mc.get('median_max_drawdown', 0):.1f}%")
            print(f"  Prob of >20% drawdown: {mc.get('prob_20pct_drawdown', 0):.1f}%")

        # Exit reason analysis
        if engine.trades:
            reason_stats = {}
            for t in engine.trades:
                r = getattr(t, 'exit_reason', 'unknown')
                if r not in reason_stats:
                    reason_stats[r] = {"count": 0, "wins": 0, "total_pnl": 0}
                reason_stats[r]["count"] += 1
                if t.net_pnl > 0:
                    reason_stats[r]["wins"] += 1
                reason_stats[r]["total_pnl"] += t.net_pnl
            print("\n--- Exit Reason Analysis ---")
            for r, s in sorted(reason_stats.items(), key=lambda x: -x[1]["count"]):
                wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                print(f"  {r}: {s['count']} trades, {wr:.0f}% WR, PnL Rs {s['total_pnl']:+,.0f}")

        # Parrondo regime transition analysis
        if parrondo and hasattr(pro_signal_generator, 'regime_transitions'):
            transitions = pro_signal_generator.regime_transitions
            if transitions:
                print(f"\n--- Parrondo Regime Transitions: {len(transitions)} switches ---")
                from_to_counts = {}
                for t in transitions:
                    key = f"{t['from']} -> {t['to']}"
                    from_to_counts[key] = from_to_counts.get(key, 0) + 1
                for key, count in sorted(from_to_counts.items(), key=lambda x: -x[1]):
                    print(f"  {key}: {count}x")
                mr_trades = sum(1 for t in engine.trades if getattr(t, 'strategy', '').startswith('mr_'))
                trend_trades = sum(1 for t in engine.trades if getattr(t, 'strategy', '').startswith('pro_'))
                print(f"  Strategy Split: {trend_trades} trend, {mr_trades} mean-reversion")

        # Scenario analysis
        print("\n--- Scenario Analysis ---")
        for pct in [-5, -3, -1, 1, 3, 5]:
            scenario = self.risk.scenario_analysis(pct)
            halt_warning = " [HALT]" if scenario["would_trigger_halt"] else ""
            print(f"  NIFTY {pct:+d}%: Impact Rs {scenario['total_impact']:+,.0f} "
                  f"→ Capital Rs {scenario['new_capital']:,.0f}{halt_warning}")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # MODE: WALK-FORWARD VALIDATION
    # ─────────────────────────────────────────────────────────────────────
    def run_walkforward(
        self,
        symbol: str = "NIFTY 50",
        train_end: str = "2020-12-31",
        test_start: str = "2021-01-01",
        strategy: str = "trend",
        parrondo: bool = False,
    ):
        """
        Walk-forward validation: train on historical, test on unseen data.
        Proves whether the system is overfit or genuinely robust.
        """
        self.dashboard.show_header()
        print("\n" + "=" * 70)
        print("  WALK-FORWARD VALIDATION")
        print(f"  Symbol: {symbol}")
        print(f"  Train: earliest...{train_end}  |  Test: {test_start}...present")
        print("=" * 70)

        # Fetch maximum available daily data
        logger.info(f"Fetching maximum daily data for {symbol}...")
        data_all = self.data.fetch_historical(symbol, days=6750, interval="day", force_refresh=True)

        if data_all.empty or len(data_all) < 100:
            logger.error(f"Insufficient data for {symbol}: {len(data_all)} bars")
            return None

        # Split by date
        train_end_ts = pd.Timestamp(train_end)
        test_start_ts = pd.Timestamp(test_start)

        data_train = data_all[data_all["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
        data_test = data_all[data_all["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

        if len(data_train) < 100:
            logger.error(f"Insufficient training data: {len(data_train)} bars (need 100+)")
            return None
        if len(data_test) < 50:
            logger.error(f"Insufficient test data: {len(data_test)} bars (need 50+)")
            return None

        train_start_date = data_train["timestamp"].iloc[0]
        train_end_date = data_train["timestamp"].iloc[-1]
        test_start_date = data_test["timestamp"].iloc[0]
        test_end_date = data_test["timestamp"].iloc[-1]

        print(f"\n  Train: {str(train_start_date)[:10]} to {str(train_end_date)[:10]} ({len(data_train)} bars)")
        print(f"  Test:  {str(test_start_date)[:10]} to {str(test_end_date)[:10]} ({len(data_test)} bars)")

        # ── IN-SAMPLE ──
        print("\n" + "=" * 70)
        print("  IN-SAMPLE (Training Period)")
        print("=" * 70)
        result_train, engine_train = self._run_backtest_on_slice(
            data_train, symbol, f"WF_train_{symbol.replace(' ', '_')}",
            parrondo=parrondo,
        )

        # ── OUT-OF-SAMPLE ──
        print("\n" + "=" * 70)
        print("  OUT-OF-SAMPLE (Test Period — UNSEEN DATA)")
        print("=" * 70)
        result_test, engine_test = self._run_backtest_on_slice(
            data_test, symbol, f"WF_test_{symbol.replace(' ', '_')}",
            parrondo=parrondo,
        )

        # ── COMPARISON ──
        self._display_walkforward_comparison(result_train, result_test, symbol)

        return result_train, result_test

    def _display_walkforward_comparison(
        self,
        result_train,
        result_test,
        symbol: str,
    ):
        """Compare in-sample vs out-of-sample with pass/fail criteria."""
        print("\n" + "=" * 70)
        print("  WALK-FORWARD COMPARISON")
        print("=" * 70)

        metrics = [
            ("Total Return %", result_train.total_return_pct, result_test.total_return_pct),
            ("Annualized Return %", result_train.annualized_return_pct, result_test.annualized_return_pct),
            ("Total Trades", result_train.total_trades, result_test.total_trades),
            ("Win Rate %", result_train.win_rate * 100 if result_train.win_rate <= 1 else result_train.win_rate,
                           result_test.win_rate * 100 if result_test.win_rate <= 1 else result_test.win_rate),
            ("Profit Factor", result_train.profit_factor, result_test.profit_factor),
            ("Sharpe Ratio", result_train.sharpe_ratio, result_test.sharpe_ratio),
            ("Max Drawdown %", result_train.max_drawdown_pct, result_test.max_drawdown_pct),
        ]

        print(f"\n  {'Metric':<22} {'In-Sample':>12} {'Out-of-Sample':>15} {'Change':>12}")
        print("  " + "-" * 65)
        for name, train_val, test_val in metrics:
            if train_val != 0:
                change = (test_val - train_val) / abs(train_val) * 100
                chg_str = f"{change:+.1f}%"
            else:
                chg_str = "N/A"
            print(f"  {name:<22} {train_val:>12.2f} {test_val:>15.2f} {chg_str:>12}")

        # Pass/fail
        print("\n  " + "-" * 65)
        print("  VALIDATION CRITERIA")
        print("  " + "-" * 65)

        checks = []

        # 1. OOS PF > 1.0
        pf_pass = result_test.profit_factor > 1.0
        checks.append(("OOS Profit Factor > 1.0", pf_pass,
                        f"PF = {result_test.profit_factor:.2f}"))

        # 2. OOS Sharpe > 0.5
        sharpe_pass = result_test.sharpe_ratio > 0.5
        checks.append(("OOS Sharpe > 0.5", sharpe_pass,
                        f"Sharpe = {result_test.sharpe_ratio:.2f}"))

        # 3. OOS PF within 30% of IS PF
        if result_train.profit_factor > 0:
            pf_ratio = result_test.profit_factor / result_train.profit_factor
            pf_close = pf_ratio >= 0.70
        else:
            pf_close = False
            pf_ratio = 0
        checks.append(("OOS PF within 30% of IS", pf_close,
                        f"Ratio = {pf_ratio:.2f} (need >= 0.70)"))

        # 4. OOS positive return
        return_pass = result_test.total_return_pct > 0
        checks.append(("OOS profitable", return_pass,
                        f"Return = {result_test.total_return_pct:.1f}%"))

        # 5. Trade frequency stable
        if result_train.total_trades > 0:
            try:
                train_days = max(1, (pd.Timestamp(result_train.end_date) - pd.Timestamp(result_train.start_date)).days)
                test_days = max(1, (pd.Timestamp(result_test.end_date) - pd.Timestamp(result_test.start_date)).days)
                train_rate = result_train.total_trades / train_days
                test_rate = result_test.total_trades / test_days
                trade_rate_ratio = test_rate / train_rate if train_rate > 0 else 0
            except Exception:
                trade_rate_ratio = 1.0
            trade_pass = trade_rate_ratio >= 0.50
        else:
            trade_pass = False
            trade_rate_ratio = 0
        checks.append(("Trade frequency stable (>50%)", trade_pass,
                        f"Rate ratio = {trade_rate_ratio:.2f}"))

        for name, passed, detail in checks:
            marker = "[PASS]" if passed else "[FAIL]"
            print(f"  {marker} {name}: {detail}")

        pass_count = sum(1 for _, p, _ in checks if p)
        total_checks = len(checks)

        print("\n  " + "=" * 65)
        if pass_count == total_checks:
            print(f"  VERDICT: ALL {total_checks}/{total_checks} PASSED — System is VALIDATED")
            print(f"  The strategy generalizes to unseen data. NOT overfit.")
        elif pass_count >= 3:
            print(f"  VERDICT: {pass_count}/{total_checks} PASSED — PARTIALLY VALIDATED")
            print(f"  Strategy shows promise but review failed criteria.")
        else:
            print(f"  VERDICT: {pass_count}/{total_checks} PASSED — MAY BE OVERFIT")
            print(f"  Significant degradation on unseen data.")
        print("  " + "=" * 65)

    # ─────────────────────────────────────────────────────────────────────
    # MODE: PARAMETER SENSITIVITY SWEEP
    # ─────────────────────────────────────────────────────────────────────
    def run_sensitivity(
        self,
        symbol: str = "NIFTY 50",
        days: int = 5475,
        strategy: str = "trend",
    ):
        """
        Parameter sensitivity sweep: tweak each key param and compare.
        Tests whether the system is robust to ±20-30% parameter changes.
        """
        self.dashboard.show_header()
        print("\n" + "=" * 70)
        print("  PARAMETER SENSITIVITY SWEEP")
        print(f"  Symbol: {symbol} | Period: {days} days")
        print("=" * 70)

        # Fetch data once
        logger.info(f"Fetching {days} days of daily data for {symbol}...")
        data_all = self.data.fetch_historical(symbol, days=days, interval="day", force_refresh=True)

        if data_all.empty or len(data_all) < 100:
            logger.error(f"Insufficient data: {len(data_all)} bars")
            return None

        print(f"\n  Data: {len(data_all)} bars ({str(data_all['timestamp'].iloc[0])[:10]} to {str(data_all['timestamp'].iloc[-1])[:10]})")

        # Define param grid
        param_grid = [
            ("confluence_trending", [2.0, 2.5, 3.0]),
            ("target_atr_mult",     [2.5, 3.0, 3.5]),
            ("time_stop_bars",      [3, 5, 7]),
            ("kelly_wr",            [0.30, 0.35, 0.40]),
            ("breakeven_ratio",     [0.3, 0.4, 0.5]),
        ]

        # Baseline run (all defaults)
        print("\n  --- Baseline (default params) ---")
        baseline, _ = self._run_backtest_on_slice(
            data_all, symbol, "baseline", param_overrides=None, verbose=False
        )
        print(f"  Baseline: PF={baseline.profit_factor:.2f}, Sharpe={baseline.sharpe_ratio:.2f}, "
              f"WR={baseline.win_rate*100 if baseline.win_rate <= 1 else baseline.win_rate:.0f}%, "
              f"DD={baseline.max_drawdown_pct:.1f}%, Trades={baseline.total_trades}")

        # Sweep
        results = []
        for param_name, test_values in param_grid:
            for val in test_values:
                overrides = {param_name: val}
                label = f"{param_name}={val}"
                logger.info(f"  Testing: {label}")
                result, _ = self._run_backtest_on_slice(
                    data_all, symbol, label, param_overrides=overrides, verbose=False
                )
                results.append((param_name, val, result))

        # Display results
        self._display_sensitivity_results(baseline, results, param_grid)

    def _display_sensitivity_results(self, baseline, results, param_grid):
        """Display parameter sensitivity sweep results."""
        print("\n" + "=" * 70)
        print("  SENSITIVITY RESULTS")
        print("=" * 70)

        defaults = {
            "confluence_trending": 2.5,
            "target_atr_mult": 3.0,
            "time_stop_bars": 5,
            "kelly_wr": 0.35,
            "breakeven_ratio": 0.4,
        }

        min_pf = baseline.profit_factor
        all_above_1_2 = True
        any_below_1_0 = False

        for param_name, test_values in param_grid:
            print(f"\n  --- {param_name} (default: {defaults.get(param_name, '?')}) ---")
            print(f"  {'Value':>8} {'PF':>8} {'Sharpe':>8} {'WR%':>6} {'DD%':>8} {'Trades':>7} {'vs Base':>10}")
            print("  " + "-" * 60)

            for pname, val, result in results:
                if pname != param_name:
                    continue
                is_default = (val == defaults.get(param_name))
                pf_change = (result.profit_factor - baseline.profit_factor) / baseline.profit_factor * 100 if baseline.profit_factor > 0 else 0
                marker = " <-- default" if is_default else ""
                wr = result.win_rate * 100 if result.win_rate <= 1 else result.win_rate
                print(f"  {val:>8} {result.profit_factor:>8.2f} {result.sharpe_ratio:>8.2f} "
                      f"{wr:>5.0f}% {result.max_drawdown_pct:>7.1f}% {result.total_trades:>7} "
                      f"{pf_change:>+9.1f}%{marker}")

                min_pf = min(min_pf, result.profit_factor)
                if result.profit_factor < 1.2:
                    all_above_1_2 = False
                if result.profit_factor < 1.0:
                    any_below_1_0 = True

        # Robustness verdict
        print("\n" + "=" * 70)
        print("  ROBUSTNESS VERDICT")
        print("=" * 70)
        print(f"  Baseline PF: {baseline.profit_factor:.2f}")
        print(f"  Min PF across all variations: {min_pf:.2f}")

        if all_above_1_2:
            print(f"  Status: ROBUST — PF stays >= 1.2 across all parameter variations")
            print(f"  The system is NOT sensitive to parameter tuning.")
        elif not any_below_1_0:
            print(f"  Status: MODERATELY ROBUST — PF stays >= 1.0 but dips below 1.2")
            print(f"  Some parameter sensitivity exists but system remains profitable.")
        else:
            print(f"  Status: FRAGILE — PF drops below 1.0 on some variations")
            print(f"  System may be overfit to specific parameter values.")
        print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # MODE: ONE-TIME SCAN
    # ─────────────────────────────────────────────────────────────────────
    def run_scan(self):
        """Run a single scan and display results."""
        self.dashboard.show_header()
        print(f"\nScanning {len(self.symbols)} symbols at {datetime.now().strftime('%H:%M:%S')}...\n")

        for symbol in self.symbols:
            print(f"{'─' * 50}")
            signal = self.analyze(symbol)

            if signal:
                self.dashboard.show_signal(signal.to_dict())

                if signal.action != "HOLD":
                    self.telegram.alert_new_signal(signal.to_dict())

                # Show regime
                data = self.data.fetch_historical(symbol, days=60, interval="day")
                if not data.empty:
                    regime = self.regime_detector.detect(data)
                    selection = self.selector.select(regime)
                    self.dashboard.show_regime({
                        "regime": regime.regime.value,
                        "confidence": regime.confidence,
                        "volatility_state": regime.volatility_regime,
                        "trend_strength": regime.trend_strength,
                        "recommended_strategy": selection["strategy"],
                    })
            else:
                print(f"  {symbol}: No data or analysis failed")

            print()

        # VIX
        vix = self.data.get_vix()
        if vix:
            print(f"India VIX: {vix:.2f}")
            if vix > 20:
                print("  ⚠ High fear — consider hedging or reducing position sizes")
            elif vix < 12:
                print("  ⚠ Low vol — options are cheap, good for buying straddles")

    # ─────────────────────────────────────────────────────────────────────
    # SETUP WIZARD
    # ─────────────────────────────────────────────────────────────────────
    def run_setup(self):
        """First-time setup wizard."""
        print("=" * 60)
        print("  PROMETHEUS — First Time Setup")
        print("=" * 60)

        print("\nStep 1: Install dependencies")
        print("  Run: pip install -r requirements.txt")

        print("\nStep 2: Install Ollama (for free local AI)")
        print("  Download from: https://ollama.ai")
        print("  Then run: ollama pull llama3.2:3b")

        print("\nStep 3: Configure Zerodha Kite Connect")
        print("  1. Log in to https://developers.kite.trade")
        print("  2. Create an app (Rs 2000/month)")
        print("  3. Copy API key and secret to config/credentials.yaml")

        print("\nStep 4: Configure Telegram Bot")
        print("  1. Message @BotFather on Telegram")
        print("  2. Send /newbot, follow instructions")
        print("  3. Copy bot token to config/credentials.yaml")
        print("  4. Message your bot, then visit:")
        print("     https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
        print("  5. Copy chat_id to config/credentials.yaml")

        print("\nStep 5: Create credentials file")
        creds_path = PROJECT_ROOT / "config" / "credentials.yaml"
        print(f"  File: {creds_path}")

        if not creds_path.exists():
            creds_content = (
                "# PROMETHEUS Credentials (DO NOT COMMIT TO GIT)\n\n"
                "zerodha:\n"
                "  api_key: \"\"\n"
                "  api_secret: \"\"\n"
                "  access_token: \"\"  # refreshed daily via login flow\n\n"
                "telegram:\n"
                "  bot_token: \"\"\n"
                "  chat_id: \"\"\n\n"
                "gemini:\n"
                "  api_key: \"\"  # optional, only if using Gemini fallback\n"
            )
            creds_path.write_text(creds_content)
            print("  Created credentials.yaml template")
        else:
            print("  credentials.yaml already exists")

        # Create .gitignore
        gitignore_path = PROJECT_ROOT.parent / ".gitignore"
        if not gitignore_path.exists():
            gitignore_content = (
                "# Credentials\n"
                "credentials.yaml\n"
                "*.env\n\n"
                "# Data\n"
                "data/*.db\n"
                "data/cache/\n"
                "data/models/\n\n"
                "# Logs\n"
                "logs/\n\n"
                "# Python\n"
                "__pycache__/\n"
                "*.pyc\n"
                ".venv/\n"
                "venv/\n"
            )
            gitignore_path.write_text(gitignore_content)
            print("\n  Created .gitignore")

        print("\nStep 6: Verify setup")
        print("  Run: python main.py scan")
        print("  This will test data fetching and signal generation")

        print("\nStep 7: Run backtest")
        print("  Run: python main.py backtest")
        print("  This validates strategies on historical data")

        print("\nStep 8: Start paper trading")
        print("  Run: python main.py paper")
        print("  Simulates real trading with virtual money")

        print("\n" + "=" * 60)
        print("  Setup complete! Start with: python main.py scan")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _send_daily_summary(self):
        """Send end-of-day summary."""
        state = self.risk.get_portfolio_state()
        self.telegram.alert_daily_summary({
            "daily_pnl": state.realized_pnl_today,
            "total_trades": state.trades_today,
            "winning_trades": 0,  # would need trade-level tracking
            "equity": state.capital,
        })

    def stop(self):
        """Stop the system gracefully."""
        self.running = False
        logger.info("PROMETHEUS shutting down...")


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="PROMETHEUS — Indian F&O Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py scan                    One-time market scan\n"
            "  python main.py backtest                Run backtests\n"
            "  python main.py backtest --days 730     Backtest last 2 years\n"
            "  python main.py walkforward             Walk-forward validation\n"
            "  python main.py sensitivity             Parameter sensitivity sweep\n"
            "  python main.py paper                   Paper trading (live sim)\n"
            "  python main.py signal                  Signal-only mode\n"
            "  python main.py setup                   First-time setup wizard\n"
        ),
    )

    parser.add_argument(
        "mode",
        choices=["scan", "backtest", "paper", "signal", "setup", "walkforward", "sensitivity"],
        help="Operating mode",
    )
    parser.add_argument(
        "--symbol", "-s",
        default=None,
        help='Symbol to analyze (default: all configured). Example: "NIFTY 50"',
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Number of days for backtest (default: 365)",
    )
    parser.add_argument(
        "--strategy",
        default="trend",
        choices=["trend", "volatility", "expiry"],
        help="Strategy to backtest (default: trend)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=300,
        help="Scan interval in seconds for signal/paper mode (default: 300)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to custom config file",
    )
    parser.add_argument(
        "--train-end",
        default="2020-12-31",
        help="End date for training period in walkforward mode (default: 2020-12-31)",
    )
    parser.add_argument(
        "--test-start",
        default="2021-01-01",
        help="Start date for test period in walkforward mode (default: 2021-01-01)",
    )
    parser.add_argument(
        "--parrondo",
        action="store_true",
        default=False,
        help="Enable Parrondo regime-switching: per-bar regime detection with mean-reversion in sideways markets",
    )

    args = parser.parse_args()

    # Initialize system
    prometheus = Prometheus(config_path=args.config)

    # Override symbols if specified
    if args.symbol:
        prometheus.symbols = [args.symbol]

    # Handle graceful shutdown
    def handle_shutdown(signum, frame):
        prometheus.stop()

    sig.signal(sig.SIGINT, handle_shutdown)

    # Dispatch to mode
    if args.mode == "setup":
        prometheus.run_setup()

    elif args.mode == "scan":
        prometheus.run_scan()

    elif args.mode == "backtest":
        prometheus.run_backtest(
            symbol=args.symbol or "NIFTY 50",
            days=args.days,
            strategy=args.strategy,
            parrondo=args.parrondo,
        )

    elif args.mode == "signal":
        prometheus.run_signal_mode(interval_seconds=args.interval)

    elif args.mode == "paper":
        prometheus.run_paper_mode(interval_seconds=args.interval)

    elif args.mode == "walkforward":
        prometheus.run_walkforward(
            symbol=args.symbol or "NIFTY 50",
            train_end=args.train_end,
            test_start=args.test_start,
            strategy=args.strategy,
            parrondo=args.parrondo,
        )

    elif args.mode == "sensitivity":
        prometheus.run_sensitivity(
            symbol=args.symbol or "NIFTY 50",
            days=args.days,
            strategy=args.strategy,
        )


if __name__ == "__main__":
    main()
