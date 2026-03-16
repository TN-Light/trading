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
from datetime import datetime, date, time as dtime, timedelta
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
    get_lot_size, get_atm_strike, get_expiry_date, get_strike_interval, IST
)
from prometheus.utils.options_math import iv_percentile, iv_rank, black_scholes_price, calculate_greeks, OptionType, get_implied_vol_at_strike

from prometheus.data.engine import DataEngine
from prometheus.data.store import DataStore

from prometheus.signals.technical import (
    calculate_vwap, calculate_session_vwap, calculate_volume_profile,
    detect_liquidity_sweeps, detect_fair_value_gaps, fibonacci_ote_levels,
    calculate_rsi, detect_rsi_divergence, calculate_atr, calculate_supertrend,
    calculate_ema, TechnicalSignal
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
        if "initial" not in cap_cfg:
            logger.warning(
                "capital.initial not found in settings.yaml — using default Rs 2,00,000. "
                "Set capital.initial in config/settings.yaml to your actual capital."
            )
        self.capital = self.initial_capital

        # Data Engine
        self.data = DataEngine(
            kite_api_key=get("broker.api_key", ""),
            kite_access_token=get("broker.access_token", ""),
        )
        self.store = DataStore()
        # Prune old data on startup to prevent unbounded DB growth
        try:
            self.store.prune_old_data()
        except Exception as e:
            logger.debug(f"DB prune on startup: {e}")

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
        if mode in ("paper", "backtest", "signal", "dry_run"):
            self.broker = PaperTrader(self.initial_capital)
        elif mode in ("semi_auto", "full_auto"):
            from prometheus.execution.kite_executor import KiteExecutor
            self.broker = KiteExecutor(
                api_key=get_credential("broker.api_key") or get("broker.api_key", ""),
                api_secret=get_credential("broker.api_secret") or get("broker.api_secret", ""),
                access_token=get_credential("broker.access_token") or get("broker.access_token", ""),
            )
            if not self.broker.connect():
                logger.error("KiteExecutor connection failed! Falling back to PaperTrader.")
                self.broker = PaperTrader(self.initial_capital)
                mode = "dry_run"
        else:
            self.broker = PaperTrader(self.initial_capital)
        self.order_manager = OrderManager(self.broker, self.risk, mode)

        # Position monitor (initialized on demand by semi_auto/full_auto/dry_run)
        self.position_monitor = None

        # Interface
        self.dashboard = CLIDashboard()
        tg_cfg = get("interface.telegram", {})
        self.telegram = TelegramBot(
            bot_token=get_credential("telegram.bot_token") or tg_cfg.get("bot_token", ""),
            chat_id=get_credential("telegram.chat_id") or tg_cfg.get("chat_id", ""),
            proxy=tg_cfg.get("proxy", ""),
            api_base_url=tg_cfg.get("api_base_url", ""),
        )

        # AI (lazy load)
        self._ai = None

        # State
        self.mode = mode
        self.running = False
        self.symbols = get("market.indices", ["NIFTY 50", "NIFTY BANK"])

        # Multi-account paper trading (initialized on demand by --multi-account)
        self.multi_account = None

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
    # MULTI-ACCOUNT & DATA COLLECTION
    # ─────────────────────────────────────────────────────────────────────

    def init_multi_account(self):
        """Initialize multi-account paper trading from config."""
        from prometheus.execution.multi_account import (
            MultiAccountPaperTrader, AccountConfig,
        )
        ma_cfg = get("multi_account", {})
        accounts = []
        for acc in ma_cfg.get("accounts", []):
            accounts.append(AccountConfig(
                label=acc["label"],
                initial_capital=acc["capital"],
                risk_overrides=acc.get("risk_overrides", {}),
            ))
        if not accounts:
            accounts = MultiAccountPaperTrader.DEFAULT_ACCOUNTS
        self.multi_account = MultiAccountPaperTrader(
            accounts, get_risk_limits(),
        )

    def run_data_collection(
        self,
        symbol: str = None,
        interval: str = "5minute",
        days: int = 180,
    ):
        """Collect and store historical data from Angel One into SQLite."""
        symbols = [symbol] if symbol else self.symbols

        if not self.data.angelone:
            logger.error(
                "Angel One not configured. "
                "Add credentials to config/credentials.yaml"
            )
            return

        self.dashboard.show_header()
        logger.info(f"Data collection: {interval}, {days} days, {len(symbols)} symbols")

        for sym in symbols:
            logger.info(f"Collecting {sym} {interval} data ({days} days)...")
            try:
                df = self.data.angelone.fetch_historical(
                    sym, days=days, interval=interval,
                )
                if df.empty:
                    logger.warning(f"  No data returned for {sym}")
                    continue
                self.data.store.save_ohlcv(df, sym, interval)
                logger.info(f"  Stored {len(df)} candles for {sym}")
            except Exception as e:
                logger.error(f"  Error collecting {sym}: {e}")

        # Summary
        logger.info("=" * 60)
        logger.info("  DATA COLLECTION SUMMARY")
        logger.info("=" * 60)
        for sym in symbols:
            for intv in [interval]:
                cached = self.data.store.get_ohlcv(sym, intv)
                if not cached.empty:
                    ts_col = "timestamp" if "timestamp" in cached.columns else cached.columns[0]
                    first = cached[ts_col].iloc[0]
                    last = cached[ts_col].iloc[-1]
                    logger.info(f"  {sym} {intv}: {len(cached)} candles ({first} to {last})")
                else:
                    logger.info(f"  {sym} {intv}: no data")

    def _dispatch_multi_account(self, refined_signal, is_intraday: bool = False,
                               bar_interval: str = "15minute"):
        """If multi-account is initialized, dispatch signal to all accounts."""
        if self.multi_account is not None:
            # Sync primary broker's price feed to all multi-account traders
            if isinstance(self.broker, PaperTrader) and self.broker._price_feed:
                self.multi_account.update_all_prices(self.broker._price_feed)
            results = self.multi_account.dispatch_signal(refined_signal, confirm=False)
            opened = sum(1 for v in results.values() if v)
            if opened:
                logger.info(f"Multi-account: {opened}/{len(results)} accounts opened positions")
            # Register multi-account positions with PositionMonitor for trailing stops
            if self.position_monitor:
                for label, did_open in results.items():
                    if did_open:
                        stack = self.multi_account.get_stack(label)
                        if stack:
                            # Find the most recent position opened by this stack
                            managed = list(stack.order_manager.managed_positions.values())
                            if managed:
                                last = managed[-1]
                                ts = stack.order_manager.create_trailing_state(
                                    last.position_id
                                )
                                if ts:
                                    # Tag with account label for exit routing
                                    ts._multi_account_label = label
                                    # Copy intraday tags from caller
                                    if is_intraday:
                                        ts.trade_mode = "intraday"
                                        ts.bar_interval = bar_interval
                                        intraday_cfg = get("intraday", {})
                                        bi_key = ts.bar_interval.replace("minute", "min")
                                        ts.max_bars = intraday_cfg.get(
                                            f"time_stop_bars_{bi_key}",
                                            intraday_cfg.get("time_stop_bars_15min", 20)
                                        )
                                        ts.breakeven_ratio = intraday_cfg.get(
                                            "breakeven_ratio", 0.5
                                        )
                                    self.position_monitor.add_position(ts)

    def _feed_real_premium(self, refined_signal: Dict):
        """
        If Angel One option chain is available, fetch real LTP/bid/ask
        for the instrument and feed to PaperTrader for realistic fills.
        """
        if not getattr(self.data, "angelone_options", None):
            return
        instrument = refined_signal.get("instrument") or refined_signal.get("tradingsymbol")
        if not instrument:
            return
        try:
            symbol = refined_signal.get("symbol", "NIFTY 50")
            strike = refined_signal.get("strike", 0)
            opt_type = refined_signal.get("option_type", "CE")
            expiry = refined_signal.get("expiry", "")
            if not strike:
                return
            # "WEEKLY" or empty = nearest expiry, pass None to skip expiry filter
            if not expiry or expiry.upper() == "WEEKLY":
                expiry = None
            result = self.data.angelone_options.get_real_premium(
                symbol, strike, opt_type, expiry,
            )
            if result and result.get("ltp", 0) > 0:
                if isinstance(self.broker, PaperTrader):
                    self.broker.set_real_premium(
                        instrument,
                        ltp=result["ltp"],
                        bid=result.get("bid", 0),
                        ask=result.get("ask", 0),
                    )
                # Also feed to multi-account traders
                if self.multi_account:
                    for stack in self.multi_account.stacks.values():
                        stack.trader.set_real_premium(
                            instrument,
                            ltp=result["ltp"],
                            bid=result.get("bid", 0),
                            ask=result.get("ask", 0),
                        )
                logger.info(
                    f"Real premium for {instrument}: LTP={result['ltp']:.2f}, "
                    f"Bid={result.get('bid', 0):.2f}, Ask={result.get('ask', 0):.2f}"
                )
        except Exception as e:
            logger.debug(f"Real premium fetch failed for {instrument}: {e}")

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
        try:
            return self._analyze_impl(symbol)
        except Exception as e:
            logger.error(f"analyze({symbol}) failed: {e}")
            return None

    def _analyze_impl(self, symbol: str) -> Optional[FusedSignal]:
        """Internal implementation — separated for per-symbol error isolation."""
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

    # ─────────────────────────────────────────────────────────────────────
    # INTRADAY ANALYSIS PIPELINE
    # ─────────────────────────────────────────────────────────────────────
    def analyze_intraday(self, symbol: str, bar_interval: str = "5minute") -> Optional[FusedSignal]:
        """
        Intraday analysis pipeline — uses 5min/15min bars as primary.

        Key differences from analyze():
        1. Primary data = 5min or 15min bars (not daily)
        2. Bias from one-level-up timeframe
        3. Session-anchored VWAP (resets at market open)
        4. Tags signal as intraday with MIS product
        """
        try:
            return self._analyze_intraday_impl(symbol, bar_interval)
        except Exception as e:
            logger.error(f"analyze_intraday({symbol}) failed: {e}")
            return None

    def _analyze_intraday_impl(self, symbol: str, bar_interval: str = "5minute") -> Optional[FusedSignal]:
        """Internal implementation — separated for per-symbol error isolation."""
        logger.info(f"Intraday analyzing {symbol} ({bar_interval})...")

        # 1. Fetch data
        data_primary = self.data.fetch_intraday(symbol, interval=bar_interval, days=5)
        if bar_interval == "5minute":
            data_bias = self.data.fetch_historical(symbol, days=10, interval="15minute")
        else:
            data_bias = self.data.fetch_historical(symbol, days=30, interval="60minute")
        data_daily = self.data.fetch_historical(symbol, days=120, interval="day")

        if data_primary.empty:
            logger.warning(f"No intraday data for {symbol}")
            return None

        spot = data_primary["close"].iloc[-1]

        # 2. Technical signals
        tech_signals = []

        # ATR from primary for intraday SL (compute early — needed by EMA margin)
        atr_value = 0.0
        if len(data_primary) >= 15:
            atr_series = calculate_atr(data_primary)
            if not atr_series.empty:
                atr_value = float(atr_series.iloc[-1])

        # Session VWAP (key for intraday)
        vwap_df = calculate_session_vwap(data_primary)
        if not vwap_df.empty and "vwap" in vwap_df.columns:
            last_vwap = vwap_df["vwap"].iloc[-1]
            if last_vwap > 0:
                vwap_dir = "bullish" if spot > last_vwap else "bearish"
                vwap_dist = abs(spot - last_vwap) / last_vwap
                tech_signals.append(TechnicalSignal(
                    name="vwap",
                    direction=vwap_dir,
                    strength=min(vwap_dist * 20, 0.9),
                    timeframe=bar_interval,
                ))

        # Supertrend on primary
        if len(data_primary) >= 20:
            st_df = calculate_supertrend(data_primary)
            if not st_df.empty and "supertrend_direction" in st_df.columns:
                st_dir = st_df["supertrend_direction"].iloc[-1]
                tech_signals.append(TechnicalSignal(
                    name="supertrend",
                    direction="bullish" if st_dir == 1 else "bearish",
                    strength=0.65,
                    timeframe=bar_interval,
                ))

        # EMA 9/21 alignment (Session 23 addition)
        if len(data_primary) >= 25:
            ema9 = data_primary["close"].ewm(span=9, adjust=False).mean()
            ema21 = data_primary["close"].ewm(span=21, adjust=False).mean()
            if not ema9.empty and not ema21.empty:
                ema9_now = ema9.iloc[-1]
                ema21_now = ema21.iloc[-1]
                margin = atr_value * 0.1 if atr_value > 0 else 0
                if ema9_now > ema21_now + margin:
                    tech_signals.append(TechnicalSignal(
                        name="ema",
                        direction="bullish",
                        strength=0.65,
                        timeframe=bar_interval,
                    ))
                elif ema9_now < ema21_now - margin:
                    tech_signals.append(TechnicalSignal(
                        name="ema",
                        direction="bearish",
                        strength=0.65,
                        timeframe=bar_interval,
                    ))

        # RSI Divergence on bias TF
        if not data_bias.empty and len(data_bias) > 44:
            div_result = detect_rsi_divergence(data_bias, rsi_period=14)
            if div_result:
                tech_signals.append(TechnicalSignal(
                    name="rsi_divergence",
                    direction=div_result["direction"],
                    strength=div_result.get("strength", 0.5),
                    timeframe="bias",
                ))

        # FVG on primary
        if len(data_primary) > 3:
            fvgs = detect_fair_value_gaps(data_primary)
            for f in fvgs[-3:]:
                direction = "bullish" if f.get("type") == "bullish_fvg" else "bearish"
                tech_signals.append(TechnicalSignal(
                    name="fvg_imbalance",
                    direction=direction,
                    strength=min(f.get("gap_pct", 0.1) / 0.5, 0.8),
                    timeframe=bar_interval,
                ))

        # Liquidity Sweeps on primary
        if len(data_primary) > 10:
            sweeps = detect_liquidity_sweeps(data_primary)
            for s in sweeps[-3:]:
                direction = "bullish" if s.get("type") == "bullish_sweep" else "bearish"
                tech_signals.append(TechnicalSignal(
                    name="liquidity_sweep",
                    direction=direction,
                    strength=s.get("strength", 0.5),
                    timeframe=bar_interval,
                ))

        # 3. Regime from daily (for strategy selection)
        regime = self.regime_detector.detect(data_daily) if not data_daily.empty else None

        # 4. Fuse with Session 23 intraday weight overrides
        # Swap in a NEW dict with intraday-specific weights (avoids mutating class dict)
        saved_weights = self.fusion.SIGNAL_WEIGHTS
        intraday_weights = dict(saved_weights)
        intraday_weights["vwap"] = 1.0         # Session VWAP is primary
        intraday_weights["supertrend"] = 1.0   # Supertrend boosted
        intraday_weights["ema"] = 0.75          # EMA 9/21 alignment
        self.fusion.SIGNAL_WEIGHTS = intraday_weights

        try:
            signal = self.fusion.fuse(
                symbol=symbol,
                spot_price=spot,
                technical_signals=tech_signals,
                oi_signals=[],
                regime=regime,
                ai_sentiment=None,
            )
        finally:
            # Restore original weights (so swing analyze() is unaffected)
            self.fusion.SIGNAL_WEIGHTS = saved_weights

        # Attach ATR-based SL
        if signal and atr_value > 0:
            if signal.stop_loss == 0 or signal.stop_loss == spot * 0.985:
                sl = self.risk.calculate_dynamic_stop_loss(spot, atr_value, signal.direction)
                signal.stop_loss = sl
                risk = abs(spot - sl)
                if risk > 0 and signal.target > 0:
                    signal.risk_reward = round(abs(signal.target - spot) / risk, 2)

        return signal

    def _select_intraday_interval(self) -> str:
        """Auto-select 5min vs 15min based on VIX."""
        intraday_cfg = get("intraday", {})
        configured = intraday_cfg.get("bar_interval", "auto")
        if configured != "auto":
            return configured

        vix = self.data.get_vix()
        threshold = intraday_cfg.get("vix_threshold_5min", 18.0)
        return "5minute" if vix > threshold else "15minute"

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
                        self._feed_real_premium(enriched)
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
                    self._feed_real_premium(enriched)
                    return enriched

        except Exception as e:
            logger.debug(f"Strategy refinement failed ({strategy_name}): {e}")

        # Feed real premium if Angel One is available (for PaperTrader uses)
        result = base_dict
        self._feed_real_premium(result)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # MODE: SIGNAL (Show signals to user)
    # ─────────────────────────────────────────────────────────────────────
    def run_signal_mode(self, interval_seconds: int = 300):
        """
        Signal mode — smart daily-bar signal scanner.

        Since the system trades daily bars (next-bar entry), the primary signal
        scan runs after market close (3:35 PM IST) when the daily candle is complete.
        During market hours, runs periodic regime checks every `interval_seconds`.
        Telegram commands (/scan, /status, etc.) work anytime.
        """
        self.running = True
        self.dashboard.show_header()
        self._setup_telegram_commands()
        self.telegram.alert_system_start()
        logger.info("Starting SIGNAL mode...")

        _did_post_close_scan = False  # Track if today's post-close scan ran

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.dashboard.show_status_line("Market holiday. Waiting...")
                    time.sleep(60)
                    continue

                current_time = now.time()

                # Reset post-close flag at market open
                if current_time < dtime(9, 15):
                    _did_post_close_scan = False
                    self.dashboard.show_status_line(
                        "Pre-market. Scan at 3:35 PM after daily candle closes."
                    )
                    time.sleep(60)
                    continue

                # ── PRIMARY SCAN: After market close (3:35-4:00 PM) ──
                # Daily candle is complete → signals are valid for next-day entry
                if dtime(15, 35) <= current_time <= dtime(16, 0) and not _did_post_close_scan:
                    logger.info("Daily candle closed — running primary signal scan...")
                    self.telegram.send_message(
                        "\U0001f50d <b>POST-CLOSE SCAN</b>\n"
                        "Daily candle complete. Scanning for tomorrow's signals..."
                    )

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

                    _did_post_close_scan = True
                    logger.info("Post-close scan complete.")
                    self.telegram.send_message(
                        "\u2705 <b>SCAN COMPLETE</b>\n"
                        "All signals for tomorrow sent above (if any).\n"
                        "Next scan: tomorrow 3:35 PM after daily candle."
                    )
                    time.sleep(60)
                    continue

                # ── DURING MARKET HOURS: lightweight status ──
                if is_market_open(now):
                    self.dashboard.show_status_line(
                        f"Market open. Daily scan at 3:35 PM. "
                        f"Use /scan in Telegram for on-demand scan."
                    )
                    time.sleep(interval_seconds)
                    continue

                # ── OUTSIDE HOURS ──
                self.dashboard.show_status_line(
                    "Market closed. Next scan: trading day 3:35 PM IST."
                )
                time.sleep(60)

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

        Uses smart 3:35 PM scan timing (daily candle must be complete for
        valid daily-bar signals).  When a signal fires, it auto-executes
        paper trades through the full pipeline: signal → strategy refinement
        → risk check → paper order placement → P&L tracking.

        Telegram commands (/scan, /status, etc.) work anytime.
        """
        self.running = True
        self.dashboard.show_header()
        self._setup_telegram_commands()
        self.telegram.alert_system_start()
        logger.info("Starting PAPER TRADING mode...")

        _did_midday_scan = False           # Track if today's mid-day scan ran
        _did_post_close_scan = False      # Track if today's post-close scan ran
        _did_send_daily_summary = False   # Avoid spamming daily summary
        _today_traded_symbols = set()     # Dedup: one trade per symbol per day

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.dashboard.show_status_line("Market holiday. Waiting...")
                    time.sleep(60)
                    continue

                current_time = now.time()

                # Reset flags at pre-market
                if current_time < dtime(9, 15):
                    _did_midday_scan = False
                    _did_post_close_scan = False
                    _did_send_daily_summary = False
                    _today_traded_symbols.clear()
                    self.dashboard.show_status_line(
                        "Pre-market. Scans at 1:00 PM (early) & 3:35 PM (primary)."
                    )
                    time.sleep(60)
                    continue

                # ── MID-DAY EARLY SCAN: 1:00 PM ──
                # Daily candle ~75% formed. Only fire for VERY strong signals.
                # Higher confidence threshold filters out noise from incomplete candle.
                if dtime(13, 0) <= current_time <= dtime(13, 15) and not _did_midday_scan:
                    logger.info("Mid-day scan — looking for high-confluence signals...")
                    _mid_found = 0

                    for symbol in self.symbols:
                        signal = self.analyze(symbol)
                        if signal and signal.action != "HOLD":
                            # Stricter filter: confidence >= 0.80 AND 4+ contributing signals
                            n_signals = len(signal.contributing_signals) if signal.contributing_signals else 0
                            if signal.confidence >= 0.80 and n_signals >= 4:
                                _mid_found += 1
                                self.dashboard.show_signal(signal.to_dict())
                                self.telegram.send_message(
                                    "\U0001f525 <b>EARLY SIGNAL (1 PM scan)</b>\n"
                                    f"High-confluence ({signal.confidence:.0%}, "
                                    f"{n_signals} indicators) on <b>{symbol}</b>"
                                )
                                self.telegram.alert_new_signal(signal.to_dict())

                                if symbol not in _today_traded_symbols:
                                    refined = self.refine_with_strategy(signal)
                                    position = self.order_manager.execute_signal(
                                        refined, confirm=False
                                    )
                                    self._dispatch_multi_account(refined)
                                    if position:
                                        _today_traded_symbols.add(symbol)
                                        self.dashboard.show_status_line(
                                            f"Paper trade opened (early): {position.position_id}"
                                        )

                    _did_midday_scan = True
                    if _mid_found == 0:
                        logger.info("Mid-day scan: no high-confluence signals. Will check again at 3:35 PM.")
                    time.sleep(60)
                    continue

                # ── PRIMARY SCAN: After market close (3:35-4:00 PM) ──
                # Daily candle is complete → signals are valid for next-day entry
                if dtime(15, 35) <= current_time <= dtime(16, 0) and not _did_post_close_scan:
                    logger.info("Daily candle closed — running paper trade scan...")
                    self.telegram.send_message(
                        "\U0001f50d <b>POST-CLOSE SCAN (Paper)</b>\n"
                        "Daily candle complete. Scanning & auto-executing paper trades..."
                    )

                    for symbol in self.symbols:
                        signal = self.analyze(symbol)
                        if signal:
                            self.dashboard.show_signal(signal.to_dict())

                            if signal.action != "HOLD" and symbol not in _today_traded_symbols:
                                self.telegram.alert_new_signal(signal.to_dict())

                                # Refine through strategy module for strike/premium/sizing
                                refined = self.refine_with_strategy(signal)

                                # Execute in paper mode
                                position = self.order_manager.execute_signal(
                                    refined, confirm=False
                                )
                                self._dispatch_multi_account(refined)

                                if position:
                                    _today_traded_symbols.add(symbol)
                                    self.dashboard.show_status_line(
                                        f"Paper trade opened: {position.position_id}"
                                    )

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

                    # Show portfolio
                    portfolio_value = self.broker.get_portfolio_value() if hasattr(self.broker, "get_portfolio_value") else self.capital
                    self.dashboard.show_portfolio_summary({
                        "initial_capital": self.initial_capital,
                        "equity": portfolio_value,
                        "daily_pnl": self.risk._daily_pnl,
                        "open_positions": len(self.broker.get_positions()),
                        "margin_used_pct": 0,
                    })

                    _did_post_close_scan = True
                    logger.info("Post-close paper scan complete.")
                    self.telegram.send_message(
                        "\u2705 <b>PAPER SCAN COMPLETE</b>\n"
                        "All paper trades executed above (if any).\n"
                        "Next scan: tomorrow 1:00 PM (early) & 3:35 PM (primary)."
                    )
                    time.sleep(60)
                    continue

                # ── DURING MARKET HOURS: lightweight status ──
                if is_market_open(now):
                    # Show open positions if any
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

                    self.dashboard.show_status_line(
                        f"Market open. Scans: 1 PM (early) & 3:35 PM (primary). "
                        f"Open positions: {len(positions)}. "
                        f"/scan in Telegram for on-demand."
                    )
                    time.sleep(interval_seconds)
                    continue

                # ── AFTER HOURS: send daily summary once ──
                if not _did_send_daily_summary and current_time >= dtime(16, 0):
                    self._send_daily_summary()
                    _did_send_daily_summary = True

                self.dashboard.show_status_line(
                    "Market closed. Next: trading day 1:00 PM & 3:35 PM IST."
                )
                time.sleep(60)

            except KeyboardInterrupt:
                self.running = False
                # Close all paper positions
                total_pnl = self.order_manager.close_all_positions("session_end")
                if self.multi_account:
                    self.multi_account.close_all("session_end")
                self._persist_equity_state()
                self.stop()
                logger.info(f"Paper trading stopped. Session P&L: Rs {total_pnl:+,.0f}")
                break
            except Exception as e:
                logger.error(f"Paper mode error: {e}")
                self.telegram.alert_system_error(str(e))
                time.sleep(30)

    # ─────────────────────────────────────────────────────────────────────
    # POSITION MONITOR HELPERS (shared by semi_auto, full_auto, dry_run)
    # ─────────────────────────────────────────────────────────────────────

    def _start_position_monitor(self):
        """Initialize and start the PositionMonitor thread."""
        from prometheus.execution.position_monitor import PositionMonitor
        pm_cfg = get("position_monitor", {})
        self.position_monitor = PositionMonitor(
            broker=self.broker,
            poll_interval=pm_cfg.get("poll_interval_seconds", 15),
            on_exit=self._handle_position_exit,
            on_trailing_update=self._handle_trailing_update,
            on_state_changed=self._handle_state_persist,
        )
        self.position_monitor.start()

    def _handle_position_exit(self, position_id: str, exit_price: float, reason: str):
        """Callback when PositionMonitor triggers an exit."""
        # Check if this is a multi-account position
        ma_label = None
        if self.position_monitor:
            positions = self.position_monitor.get_positions()
            state = positions.get(position_id)
            if state:
                ma_label = getattr(state, "_multi_account_label", None)

        pnl = None
        if ma_label and self.multi_account:
            # Route exit to the correct sub-account stack
            stack = self.multi_account.get_stack(ma_label)
            if stack:
                pnl = stack.order_manager.close_position(position_id, reason)
                logger.info(f"[{ma_label}] Position {position_id} closed ({reason}): P&L Rs {pnl if pnl is not None else 0:+,.0f}")
            else:
                logger.warning(f"Multi-account stack '{ma_label}' not found for {position_id}")
        else:
            # Primary account exit
            pnl = self.order_manager.close_position(position_id, reason)

        if self.position_monitor:
            self.position_monitor.remove_position(position_id)
        self.store.close_position_state(position_id, pnl if pnl is not None else 0.0)

        # Persist equity state for crash recovery
        self._persist_equity_state()

        reason_labels = {
            "stop_loss": "SL Hit",
            "target": "Target Hit",
            "time_stop": "Time Stop",
            "intraday_square_off": "Intraday Square-Off (3:15 PM)",
        }
        label = reason_labels.get(reason, reason)
        pnl_text = f"Rs {pnl:+,.0f}" if pnl is not None else "unknown"
        self.telegram.send_message(
            f"\U0001f6a8 <b>POSITION CLOSED — {label}</b>\n\n"
            f"ID: <code>{position_id}</code>\n"
            f"Exit price: {exit_price:.2f}\n"
            f"P&L: {pnl_text}"
        )

    def _handle_trailing_update(self, state, old_sl: float):
        """Callback when trailing stop advances a stage."""
        self.telegram.send_message(
            f"\U0001f4c8 <b>TRAILING STOP UPDATE</b>\n\n"
            f"<code>{state.tradingsymbol}</code>\n"
            f"Stage: <b>{state.current_stage()}</b>\n"
            f"SL: {old_sl:.2f} \u2192 {state.current_sl:.2f}"
        )

    def _handle_state_persist(self, state):
        """Callback to persist trailing state to SQLite."""
        self.store.save_position_state(state.to_dict())

    def _handle_loop_error(self, mode_label: str, error: Exception,
                           consecutive_errors: int) -> tuple:
        """Handle errors in trading loops with backoff and circuit breaker.

        Returns (should_halt: bool, sleep_seconds: float).
        """
        backoff = min(30 * (2 ** min(consecutive_errors - 1, 4)), 300)
        logger.error(f"{mode_label} error ({consecutive_errors} consecutive): {error}")
        self.telegram.alert_system_error(str(error))
        if consecutive_errors >= 20:
            logger.critical(
                f"{mode_label}: {consecutive_errors} consecutive errors — "
                f"halting system. Manual restart needed."
            )
            self.telegram.send_message(
                f"\U0001f6d1 <b>SYSTEM HALTED</b>\n\n"
                f"{consecutive_errors} consecutive errors.\n"
                f"Last: {str(error)[:200]}\n\n"
                f"Manual restart required."
            )
            return True, 0
        return False, backoff

    def _persist_equity_state(self):
        """Save current equity/capital to SQLite for crash recovery."""
        try:
            if isinstance(self.broker, PaperTrader):
                margins = self.broker.get_margins()
                self.store.save_equity_snapshot(
                    "primary",
                    equity=margins.get("equity", self.initial_capital),
                    peak=self.risk.peak_capital,
                    total_costs=self.broker.total_costs,
                    realized_pnl=margins.get("realized_pnl", 0),
                )
            # Multi-account stacks
            if self.multi_account:
                for label, stack in self.multi_account.stacks.items():
                    m = stack.trader.get_margins()
                    self.store.save_equity_snapshot(
                        label,
                        equity=m.get("equity", stack.config.initial_capital),
                        peak=stack.risk.peak_capital,
                        total_costs=stack.trader.total_costs,
                        realized_pnl=m.get("realized_pnl", 0),
                    )
        except Exception as e:
            logger.debug(f"Equity persist error: {e}")

    def _restore_equity_state(self):
        """Restore equity/capital from SQLite after crash/restart."""
        try:
            snap = self.store.load_equity_snapshot("primary")
            if snap and isinstance(self.broker, PaperTrader):
                saved_equity = snap.get("equity", 0)
                saved_peak = snap.get("peak", 0)
                if saved_equity > 0:
                    # Adjust PaperTrader cash to reflect cumulative P&L
                    pnl_since_start = saved_equity - self.initial_capital
                    self.broker.available_cash = self.initial_capital + pnl_since_start
                    self.broker.total_costs = snap.get("total_costs", 0)
                    # Sync risk manager capital
                    self.risk.current_capital = saved_equity
                    if saved_peak > 0:
                        self.risk.peak_capital = saved_peak
                    logger.info(
                        f"Equity restored: Rs {saved_equity:,.0f} "
                        f"(peak: Rs {saved_peak:,.0f}, "
                        f"P&L since start: Rs {pnl_since_start:+,.0f})"
                    )
            # Multi-account stacks
            if self.multi_account:
                for label, stack in self.multi_account.stacks.items():
                    ms = self.store.load_equity_snapshot(label)
                    if ms and ms.get("equity", 0) > 0:
                        pnl = ms["equity"] - stack.config.initial_capital
                        stack.trader.available_cash = stack.config.initial_capital + pnl
                        stack.trader.total_costs = ms.get("total_costs", 0)
                        stack.risk.current_capital = ms["equity"]
                        if ms.get("peak", 0) > 0:
                            stack.risk.peak_capital = ms["peak"]
                        logger.info(f"[{label}] Equity restored: Rs {ms['equity']:,.0f}")
        except Exception as e:
            logger.debug(f"Equity restore error: {e}")

    def _restore_persisted_positions(self):
        """Restore open positions from SQLite after crash/restart."""
        saved = self.store.load_open_positions()
        if not saved or not self.position_monitor:
            return
        from prometheus.execution.position_monitor import TrailingState
        states = []
        for row in saved:
            states.append(TrailingState(
                position_id=row["position_id"],
                tradingsymbol=row["tradingsymbol"],
                symbol=row["symbol"],
                entry_premium=row["entry_premium"],
                initial_sl=row["initial_sl"],
                current_sl=row["current_sl"],
                target=row["target"],
                direction=row["direction"],
                strategy=row.get("strategy", ""),
                entry_time=row.get("entry_time", ""),
                sl_order_id=row.get("sl_order_id", ""),
                breakeven_set=bool(row.get("breakeven_set", 0)),
                trailing_activated=bool(row.get("trailing_activated", 0)),
                trailing_stage2=bool(row.get("trailing_stage2", 0)),
                trailing_stage3=bool(row.get("trailing_stage3", 0)),
                premium_hwm=row.get("premium_hwm", 0),
                entry_bar_count=row.get("entry_bar_count", 0),
                max_bars=row.get("max_bars", 7),
                breakeven_ratio=row.get("breakeven_ratio", 0.6),
                risk_distance=row.get("risk_distance", 0),
                bar_interval=row.get("bar_interval", "day"),
                trade_mode=row.get("trade_mode", "swing"),
            ))
        self.position_monitor.restore_positions(states)
        if states:
            self.telegram.send_message(
                f"\U0001f504 Restored {len(states)} open position(s) from last session."
            )

    def _register_position_with_monitor(self, position):
        """After a trade executes, register it with the PositionMonitor."""
        if not self.position_monitor or not position:
            return
        state = self.order_manager.create_trailing_state(position.position_id)
        if state:
            self.position_monitor.add_position(state)
            self._handle_state_persist(state)

    # ─────────────────────────────────────────────────────────────────────
    # FULL AUTO MODE
    # ─────────────────────────────────────────────────────────────────────

    def run_full_auto_mode(self, interval_seconds: int = 300):
        """
        Full auto: scan → execute → monitor → trail → exit.

        Same scan timing as paper mode (1 PM early + 3:35 PM primary).
        PositionMonitor runs continuously for 5-stage trailing stop.
        State persisted to SQLite for crash recovery.
        """
        self.running = True
        self.dashboard.show_header()
        self._setup_telegram_commands()
        self.telegram.alert_system_start()

        # Start position monitor + restore any persisted positions
        self._start_position_monitor()
        self._restore_equity_state()
        self._restore_persisted_positions()

        is_dry_run = isinstance(self.broker, PaperTrader)
        mode_label = "DRY RUN" if is_dry_run else "FULL AUTO"
        logger.info(f"Starting {mode_label} mode...")
        self.telegram.send_message(
            f"\U0001f916 <b>{mode_label} MODE STARTED</b>\n"
            f"Capital: Rs {self.capital:,.0f}\n"
            f"Trailing: 5-stage | Monitor: every 15s\n"
            f"Scans: 1:00 PM (early) + 3:35 PM (primary)"
        )

        _did_midday_scan = False
        _did_post_close_scan = False
        _did_send_daily_summary = False
        _today_traded_symbols = set()

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.dashboard.show_status_line(
                        f"{mode_label}: Market holiday. Waiting..."
                    )
                    time.sleep(60)
                    continue

                current_time = now.time()

                # Reset flags at pre-market
                if current_time < dtime(9, 15):
                    _did_midday_scan = False
                    _did_post_close_scan = False
                    _did_send_daily_summary = False
                    _today_traded_symbols.clear()
                    self.dashboard.show_status_line(
                        f"{mode_label}: Pre-market. Scans at 1:00 PM & 3:35 PM."
                    )
                    time.sleep(60)
                    continue

                # ── MID-DAY EARLY SCAN: 1:00 PM ──
                if dtime(13, 0) <= current_time <= dtime(13, 15) and not _did_midday_scan:
                    logger.info(f"{mode_label}: Mid-day scan...")
                    for symbol in self.symbols:
                        signal = self.analyze(symbol)
                        if signal and signal.action != "HOLD":
                            n_signals = len(signal.contributing_signals) if signal.contributing_signals else 0
                            if signal.confidence >= 0.80 and n_signals >= 4:
                                if symbol not in _today_traded_symbols:
                                    self.telegram.alert_new_signal(signal.to_dict())
                                    refined = self.refine_with_strategy(signal)
                                    position = self.order_manager.execute_signal(
                                        refined, confirm=False
                                    )
                                    self._dispatch_multi_account(refined)
                                    if position:
                                        _today_traded_symbols.add(symbol)
                                        self._register_position_with_monitor(position)
                    _did_midday_scan = True
                    time.sleep(60)
                    continue

                # ── PRIMARY SCAN: 3:35-4:00 PM ──
                if dtime(15, 35) <= current_time <= dtime(16, 0) and not _did_post_close_scan:
                    logger.info(f"{mode_label}: Post-close scan...")
                    self.telegram.send_message(
                        f"\U0001f50d <b>POST-CLOSE SCAN ({mode_label})</b>\n"
                        "Daily candle complete. Scanning..."
                    )
                    for symbol in self.symbols:
                        signal = self.analyze(symbol)
                        if signal and signal.action != "HOLD" and symbol not in _today_traded_symbols:
                            self.telegram.alert_new_signal(signal.to_dict())
                            refined = self.refine_with_strategy(signal)
                            position = self.order_manager.execute_signal(
                                refined, confirm=False
                            )
                            self._dispatch_multi_account(refined)
                            if position:
                                _today_traded_symbols.add(symbol)
                                self._register_position_with_monitor(position)

                    _did_post_close_scan = True
                    n_pos = self.position_monitor.active_count if self.position_monitor else 0
                    self.telegram.send_message(
                        f"\u2705 <b>SCAN COMPLETE</b>\n"
                        f"Active positions: {n_pos}\n"
                        f"Next scan: tomorrow 1:00 PM & 3:35 PM."
                    )
                    time.sleep(60)
                    continue

                # ── DURING MARKET HOURS: show status ──
                if is_market_open(now):
                    n_pos = self.position_monitor.active_count if self.position_monitor else 0
                    self.dashboard.show_status_line(
                        f"{mode_label}: Market open. "
                        f"Monitoring {n_pos} position(s). "
                        f"Scans: 1 PM & 3:35 PM."
                    )
                    time.sleep(interval_seconds)
                    continue

                # ── AFTER HOURS ──
                if not _did_send_daily_summary and current_time >= dtime(16, 0):
                    self._send_daily_summary()
                    _did_send_daily_summary = True

                self.dashboard.show_status_line(
                    f"{mode_label}: Market closed. Next: trading day 1 PM & 3:35 PM."
                )
                time.sleep(60)

            except KeyboardInterrupt:
                self.running = False
                if self.position_monitor:
                    self.position_monitor.stop()
                total_pnl = self.order_manager.close_all_positions("session_end")
                if self.multi_account:
                    self.multi_account.close_all("session_end")
                self._persist_equity_state()
                self.stop()
                logger.info(f"{mode_label} stopped. Session P&L: Rs {total_pnl:+,.0f}")
                break
            except Exception as e:
                logger.error(f"{mode_label} error: {e}")
                self.telegram.alert_system_error(str(e))
                time.sleep(30)

    # ─────────────────────────────────────────────────────────────────────
    # SEMI AUTO MODE
    # ─────────────────────────────────────────────────────────────────────

    def run_semi_auto_mode(self, interval_seconds: int = 300):
        """
        Semi-auto: scan → signal → Telegram /confirm → execute.

        Same scan timing as paper mode. User must /confirm each trade.
        PositionMonitor runs for trailing stop management after entry.
        """
        self.running = True
        self.dashboard.show_header()
        self._setup_telegram_commands()
        self.telegram.alert_system_start()

        self._start_position_monitor()
        self._restore_equity_state()
        self._restore_persisted_positions()

        pm_cfg = get("position_monitor", {})
        confirm_timeout = pm_cfg.get("confirmation_timeout", 1800)

        logger.info("Starting SEMI-AUTO mode...")
        self.telegram.send_message(
            "\U0001f91d <b>SEMI-AUTO MODE STARTED</b>\n"
            f"Capital: Rs {self.capital:,.0f}\n"
            "You will be asked to /confirm or /reject each trade.\n"
            f"Timeout: {confirm_timeout // 60} minutes per signal."
        )

        _did_midday_scan = False
        _did_post_close_scan = False
        _did_send_daily_summary = False
        _today_traded_symbols = set()

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.dashboard.show_status_line("SEMI-AUTO: Market holiday. Waiting...")
                    time.sleep(60)
                    continue

                current_time = now.time()

                if current_time < dtime(9, 15):
                    _did_midday_scan = False
                    _did_post_close_scan = False
                    _did_send_daily_summary = False
                    _today_traded_symbols.clear()
                    self.dashboard.show_status_line(
                        "SEMI-AUTO: Pre-market. Scans at 1:00 PM & 3:35 PM."
                    )
                    time.sleep(60)
                    continue

                # ── MID-DAY SCAN: 1:00 PM (stricter) ──
                if dtime(13, 0) <= current_time <= dtime(13, 15) and not _did_midday_scan:
                    logger.info("SEMI-AUTO: Mid-day scan...")
                    for symbol in self.symbols:
                        signal = self.analyze(symbol)
                        if signal and signal.action != "HOLD":
                            n_signals = len(signal.contributing_signals) if signal.contributing_signals else 0
                            if signal.confidence >= 0.80 and n_signals >= 4 and symbol not in _today_traded_symbols:
                                refined = self.refine_with_strategy(signal)
                                confirmed = self.telegram.request_confirmation(
                                    signal.to_dict(), timeout=confirm_timeout
                                )
                                if confirmed:
                                    position = self.order_manager.execute_signal(
                                        refined, confirm=False
                                    )
                                    if position:
                                        _today_traded_symbols.add(symbol)
                                        self._register_position_with_monitor(position)
                                        self.telegram.send_message(
                                            f"\u2705 Trade executed: {position.position_id}"
                                        )
                    _did_midday_scan = True
                    time.sleep(60)
                    continue

                # ── PRIMARY SCAN: 3:35-4:00 PM ──
                if dtime(15, 35) <= current_time <= dtime(16, 0) and not _did_post_close_scan:
                    logger.info("SEMI-AUTO: Post-close scan...")
                    for symbol in self.symbols:
                        signal = self.analyze(symbol)
                        if signal and signal.action != "HOLD" and symbol not in _today_traded_symbols:
                            refined = self.refine_with_strategy(signal)
                            confirmed = self.telegram.request_confirmation(
                                signal.to_dict(), timeout=confirm_timeout
                            )
                            if confirmed:
                                position = self.order_manager.execute_signal(
                                    refined, confirm=False
                                )
                                if position:
                                    _today_traded_symbols.add(symbol)
                                    self._register_position_with_monitor(position)
                                    self.telegram.send_message(
                                        f"\u2705 Trade executed: {position.position_id}"
                                    )
                    _did_post_close_scan = True
                    time.sleep(60)
                    continue

                # ── MARKET HOURS ──
                if is_market_open(now):
                    n_pos = self.position_monitor.active_count if self.position_monitor else 0
                    self.dashboard.show_status_line(
                        f"SEMI-AUTO: Monitoring {n_pos} position(s). "
                        f"Scans: 1 PM & 3:35 PM."
                    )
                    time.sleep(interval_seconds)
                    continue

                # ── AFTER HOURS ──
                if not _did_send_daily_summary and current_time >= dtime(16, 0):
                    self._send_daily_summary()
                    _did_send_daily_summary = True

                self.dashboard.show_status_line(
                    "SEMI-AUTO: Market closed. Next: trading day 1 PM & 3:35 PM."
                )
                time.sleep(60)

            except KeyboardInterrupt:
                self.running = False
                if self.position_monitor:
                    self.position_monitor.stop()
                total_pnl = self.order_manager.close_all_positions("session_end")
                if self.multi_account:
                    self.multi_account.close_all("session_end")
                self._persist_equity_state()
                self.stop()
                logger.info(f"Semi-auto stopped. Session P&L: Rs {total_pnl:+,.0f}")
                break
            except Exception as e:
                logger.error(f"Semi-auto error: {e}")
                self.telegram.alert_system_error(str(e))
                time.sleep(30)

    # ─────────────────────────────────────────────────────────────────────
    # MODE: INTRADAY TRADING
    # ─────────────────────────────────────────────────────────────────────

    def run_intraday_mode(self, interval_seconds: int = 180):
        """
        Intraday trading — continuous scanning during market hours.

        Timing:
        - 9:15-9:45: Skip (opening noise)
        - 9:45-14:30: Scan aligned to bar interval (5min→300s, 15min→900s)
        - 14:30-15:15: Monitor only, no new entries
        - 15:15: Force close all intraday positions

        Works with paper, dry_run, semi_auto, full_auto broker types.
        """
        self.running = True
        self.dashboard.show_header()
        self._setup_telegram_commands()
        self.telegram.alert_system_start()

        # Start position monitor with faster intraday polling
        intraday_cfg = get("intraday", {})
        from prometheus.execution.position_monitor import PositionMonitor
        poll_s = intraday_cfg.get("trailing_poll_seconds", 10)
        self.position_monitor = PositionMonitor(
            broker=self.broker,
            poll_interval=poll_s,
            on_exit=self._handle_position_exit,
            on_trailing_update=self._handle_trailing_update,
            on_state_changed=self._handle_state_persist,
        )
        self.position_monitor.start()
        self._restore_equity_state()
        self._restore_persisted_positions()

        is_dry_run = isinstance(self.broker, PaperTrader)
        mode_label = "INTRADAY DRY" if is_dry_run else "INTRADAY LIVE"

        logger.info(f"Starting {mode_label} mode...")
        self.telegram.send_message(
            f"\U0001f4c8 <b>{mode_label} MODE STARTED</b>\n"
            f"Capital: Rs {self.capital:,.0f}\n"
            f"Scan aligned to bar interval | Square-off at 3:15 PM"
        )

        _today_traded_symbols = set()
        _intraday_trades_today = 0
        _did_square_off = False
        _last_scan_time = None
        _did_send_daily_summary = False

        max_trades = intraday_cfg.get("max_daily_trades", 4)
        skip_minutes = intraday_cfg.get("skip_first_minutes", 30)
        last_entry_str = intraday_cfg.get("last_entry_time", "14:30")
        square_off_str = intraday_cfg.get("square_off_time", "15:15")
        last_entry_h, last_entry_m = map(int, last_entry_str.split(":"))
        square_off_h, square_off_m = map(int, square_off_str.split(":"))
        last_entry_time = dtime(last_entry_h, last_entry_m)
        square_off_time = dtime(square_off_h, square_off_m)
        intraday_instruments = intraday_cfg.get("instruments", self.symbols)
        be_ratio = intraday_cfg.get("breakeven_ratio", 0.5)

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.dashboard.show_status_line(f"{mode_label}: Holiday. Waiting...")
                    time.sleep(60)
                    continue

                current_time = now.time()

                # Pre-market reset
                if current_time < dtime(9, 15):
                    _today_traded_symbols.clear()
                    _intraday_trades_today = 0
                    _did_square_off = False
                    _last_scan_time = None
                    _did_send_daily_summary = False
                    self.dashboard.show_status_line(
                        f"{mode_label}: Pre-market. Scan starts at 9:45 AM."
                    )
                    time.sleep(60)
                    continue

                # Skip opening noise (9:15-9:45)
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                if (now - market_open).total_seconds() < skip_minutes * 60:
                    self.dashboard.show_status_line(
                        f"{mode_label}: Skipping opening noise. Scan at 9:45 AM."
                    )
                    time.sleep(30)
                    continue

                # MANDATORY SQUARE-OFF at 3:15 PM
                if current_time >= square_off_time and not _did_square_off:
                    self._square_off_intraday_positions()
                    _did_square_off = True
                    time.sleep(60)
                    continue

                # After square-off — send daily summary after hours, otherwise wait
                if _did_square_off:
                    if current_time >= dtime(16, 0) and not _did_send_daily_summary:
                        self._send_daily_summary()
                        _did_send_daily_summary = True
                    self.dashboard.show_status_line(
                        f"{mode_label}: Squared off. Market closing soon."
                    )
                    time.sleep(60)
                    continue

                # No new entries after 2:30 PM — monitor only
                if current_time >= last_entry_time:
                    n_pos = self.position_monitor.active_count if self.position_monitor else 0
                    self.dashboard.show_status_line(
                        f"{mode_label}: No new entries. Monitoring {n_pos} position(s). "
                        f"Square-off at {square_off_str}."
                    )
                    time.sleep(30)
                    continue

                # ── INTRADAY SCAN ──
                bar_interval = self._select_intraday_interval()
                # Auto-match scan interval to bar interval (5min→300s, 15min→900s)
                scan_interval = 300 if bar_interval == "5minute" else 900

                # Rate limit scans — wait for candle close
                if _last_scan_time and (now - _last_scan_time).total_seconds() < scan_interval:
                    time.sleep(10)
                    continue

                # Max trades check
                if _intraday_trades_today >= max_trades:
                    self.dashboard.show_status_line(
                        f"{mode_label}: Max trades ({max_trades}) reached. Monitoring."
                    )
                    time.sleep(60)
                    continue

                logger.info(f"{mode_label}: Scanning ({bar_interval}, next in {scan_interval}s)...")

                for symbol in intraday_instruments:
                    if symbol in _today_traded_symbols:
                        continue

                    signal = self.analyze_intraday(symbol, bar_interval)
                    if signal and signal.action != "HOLD":
                        self.telegram.alert_new_signal(signal.to_dict())
                        refined = self.refine_with_strategy(signal)

                        position = self.order_manager.execute_signal(
                            refined, confirm=False
                        )
                        self._dispatch_multi_account(
                            refined, is_intraday=True,
                            bar_interval=bar_interval
                        )
                        if position:
                            _today_traded_symbols.add(symbol)
                            _intraday_trades_today += 1
                            # Register with monitor (intraday-tagged)
                            ts = self.order_manager.create_trailing_state(
                                position.position_id
                            )
                            if ts:
                                ts.trade_mode = "intraday"
                                ts.bar_interval = bar_interval
                                ts.breakeven_ratio = be_ratio
                                intraday_ts_cfg = intraday_cfg.get(
                                    f"time_stop_bars_{bar_interval.replace('minute', 'min')}",
                                    intraday_cfg.get("time_stop_bars_15min", 20)
                                )
                                ts.max_bars = intraday_ts_cfg
                                self.position_monitor.add_position(ts)
                                self._handle_state_persist(ts)
                            self.telegram.send_message(
                                f"\u2705 Intraday trade: {position.position_id} "
                                f"({bar_interval})"
                            )

                _last_scan_time = now

                # ── AFTER HOURS: daily summary + slow sleep ──
                if current_time >= dtime(16, 0):
                    if not _did_send_daily_summary:
                        self._send_daily_summary()
                        _did_send_daily_summary = True
                    time.sleep(60)
                    continue

                n_pos = self.position_monitor.active_count if self.position_monitor else 0
                self.dashboard.show_status_line(
                    f"{mode_label}: {n_pos} position(s) | "
                    f"Trades: {_intraday_trades_today}/{max_trades} | "
                    f"Bar: {bar_interval}"
                )
                time.sleep(10)

            except KeyboardInterrupt:
                self.running = False
                # Stop monitor FIRST to prevent race with square-off
                if self.position_monitor:
                    self.position_monitor.stop()
                self._square_off_intraday_positions()
                if self.multi_account:
                    self.multi_account.close_all("session_end")
                self._persist_equity_state()
                self.stop()
                logger.info(f"{mode_label} stopped by user.")
                break
            except Exception as e:
                logger.error(f"{mode_label} error: {e}")
                self.telegram.alert_system_error(str(e))
                time.sleep(30)

    def _square_off_intraday_positions(self):
        """Force close all intraday positions at 3:15 PM."""
        if not self.position_monitor:
            return

        positions = self.position_monitor.get_positions()
        intraday_pids = [
            pid for pid, state in positions.items()
            if getattr(state, "trade_mode", "swing") == "intraday"
        ]

        if not intraday_pids:
            logger.info("Square-off: no intraday positions to close.")
            return

        logger.info(f"Square-off: closing {len(intraday_pids)} intraday position(s)")
        self.telegram.send_message(
            f"\u23f0 <b>INTRADAY SQUARE-OFF (3:15 PM)</b>\n"
            f"Closing {len(intraday_pids)} position(s)..."
        )

        for pid in intraday_pids:
            try:
                state = positions[pid]
                ltp = self.broker.get_ltp(state.tradingsymbol, exchange="NFO")
                if ltp <= 0:
                    ltp = state.current_sl  # fallback
                self._handle_position_exit(pid, ltp, "intraday_square_off")
            except Exception as e:
                logger.error(f"Square-off error for {pid}: {e}")

    def run_combined_mode(self, interval_seconds: int = 180):
        """
        Combined swing + intraday in single event loop.

        Timing:
        - 09:45-13:00: Intraday scan every N seconds
        - 13:00: Swing scan #1 (strict — requires higher confidence)
        - 13:00-14:30: Intraday scan continues
        - 14:30-15:15: Monitor only (no intraday entries), swing positions untouched
        - 15:15: Intraday square-off (swing positions kept)
        - 15:35: Swing scan #2 (primary — daily candle nearly complete)
        """
        self.running = True
        self.dashboard.show_header()
        self._setup_telegram_commands()
        self.telegram.alert_system_start()

        # Start single PositionMonitor for both modes
        from prometheus.execution.position_monitor import PositionMonitor
        self.position_monitor = PositionMonitor(
            broker=self.broker,
            poll_interval=10,  # faster for intraday
            on_exit=self._handle_position_exit,
            on_trailing_update=self._handle_trailing_update,
            on_state_changed=self._handle_state_persist,
        )
        self.position_monitor.start()
        self._restore_equity_state()
        self._restore_persisted_positions()

        is_dry_run = isinstance(self.broker, PaperTrader)
        mode_label = "COMBINED DRY" if is_dry_run else "COMBINED LIVE"

        logger.info(f"Starting {mode_label} mode (swing + intraday)...")
        self.telegram.send_message(
            f"\U0001f680 <b>{mode_label} MODE STARTED</b>\n"
            f"Capital: Rs {self.capital:,.0f}\n"
            f"Swing: 1:00 PM + 3:35 PM scans\n"
            f"Intraday: 9:45-14:30 scan | 15:15 square-off"
        )

        # Intraday state
        intraday_cfg = get("intraday", {})
        _intra_traded_symbols = set()
        _intra_trades_today = 0
        _did_square_off = False
        _last_intra_scan = None

        intra_max_trades = intraday_cfg.get("max_daily_trades", 4)
        skip_minutes = intraday_cfg.get("skip_first_minutes", 30)
        last_entry_str = intraday_cfg.get("last_entry_time", "14:30")
        square_off_str = intraday_cfg.get("square_off_time", "15:15")
        last_entry_h, last_entry_m = map(int, last_entry_str.split(":"))
        square_off_h, square_off_m = map(int, square_off_str.split(":"))
        last_entry_time = dtime(last_entry_h, last_entry_m)
        square_off_time = dtime(square_off_h, square_off_m)
        intraday_instruments = intraday_cfg.get("instruments", self.symbols)
        be_ratio = intraday_cfg.get("breakeven_ratio", 0.5)

        # Swing state
        _did_1pm_scan = False
        _did_335pm_scan = False
        _did_send_daily_summary = False
        _swing_traded_symbols = set()  # dedup: one swing trade per symbol per day
        _consecutive_errors = 0

        while self.running:
            try:
                now = datetime.now()

                if not is_trading_day(now.date()):
                    self.telegram.reconnect()
                    self.dashboard.show_status_line(f"{mode_label}: Holiday. Waiting...")
                    time.sleep(60)
                    continue

                current_time = now.time()

                # Pre-market reset
                if current_time < dtime(9, 15):
                    _intra_traded_symbols.clear()
                    _intra_trades_today = 0
                    _did_square_off = False
                    _last_intra_scan = None
                    _did_1pm_scan = False
                    _did_335pm_scan = False
                    _did_send_daily_summary = False
                    _swing_traded_symbols.clear()
                    # Retry Telegram if it failed at startup
                    self.telegram.reconnect()
                    self.dashboard.show_status_line(
                        f"{mode_label}: Pre-market. Waiting for 9:45 AM."
                    )
                    time.sleep(60)
                    _consecutive_errors = 0  # Reset AFTER successful pre-market iteration
                    continue

                # Skip opening noise (9:15-9:45)
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                if (now - market_open).total_seconds() < skip_minutes * 60:
                    self.dashboard.show_status_line(
                        f"{mode_label}: Skipping opening noise."
                    )
                    time.sleep(30)
                    continue

                # ── INTRADAY SQUARE-OFF at 15:15 ──
                if current_time >= square_off_time and not _did_square_off:
                    self._square_off_intraday_positions()
                    _did_square_off = True

                # ── SWING SCAN #2 at 3:35 PM (but not after 4 PM — stale data guard) ──
                if dtime(15, 35) <= current_time < dtime(16, 0) and not _did_335pm_scan:
                    logger.info(f"{mode_label}: Swing scan #2 (3:35 PM)")
                    for symbol in self.symbols:
                        if symbol in _swing_traded_symbols:
                            continue
                        signal = self.analyze(symbol)
                        if signal and signal.action != "HOLD":
                            self.telegram.alert_new_signal(signal.to_dict())
                            refined = self.refine_with_strategy(signal)
                            position = self.order_manager.execute_signal(
                                refined, confirm=False
                            )
                            self._dispatch_multi_account(refined)
                            if position:
                                _swing_traded_symbols.add(symbol)
                                ts = self.order_manager.create_trailing_state(
                                    position.position_id
                                )
                                if ts:
                                    ts.trade_mode = "swing"
                                    ts.bar_interval = "day"
                                    self.position_monitor.add_position(ts)
                                    self._handle_state_persist(ts)
                    _did_335pm_scan = True
                    time.sleep(60)
                    continue

                # ── SWING SCAN #1 at 1:00 PM (strict) ──
                if dtime(13, 0) <= current_time < dtime(13, 5) and not _did_1pm_scan:
                    logger.info(f"{mode_label}: Swing scan #1 (1:00 PM strict)")
                    for symbol in self.symbols:
                        signal = self.analyze(symbol)
                        if signal and signal.action != "HOLD" and signal.confidence >= 0.80:
                            n_signals = len(signal.contributing_signals) if signal.contributing_signals else 0
                            if n_signals >= 4 and symbol not in _swing_traded_symbols:
                                self.telegram.alert_new_signal(signal.to_dict())
                                refined = self.refine_with_strategy(signal)
                                position = self.order_manager.execute_signal(
                                    refined, confirm=False
                                )
                                self._dispatch_multi_account(refined)
                                if position:
                                    _swing_traded_symbols.add(symbol)
                                    ts = self.order_manager.create_trailing_state(
                                        position.position_id
                                    )
                                    if ts:
                                        ts.trade_mode = "swing"
                                        ts.bar_interval = "day"
                                        self.position_monitor.add_position(ts)
                                        self._handle_state_persist(ts)
                    _did_1pm_scan = True

                # ── INTRADAY SCAN (9:45-14:30) ──
                if current_time < last_entry_time and not _did_square_off:
                    if _intra_trades_today < intra_max_trades:
                        bar_interval = self._select_intraday_interval()
                        # Auto-match scan interval to bar interval
                        intra_scan_interval = 300 if bar_interval == "5minute" else 900
                        if _last_intra_scan is None or (now - _last_intra_scan).total_seconds() >= intra_scan_interval:
                            logger.info(f"{mode_label}: Intraday scan ({bar_interval}, next in {intra_scan_interval}s)...")

                            for isym in intraday_instruments:
                                if isym in _intra_traded_symbols:
                                    continue
                                signal = self.analyze_intraday(isym, bar_interval)
                                if signal and signal.action != "HOLD":
                                    self.telegram.alert_new_signal(signal.to_dict())
                                    refined = self.refine_with_strategy(signal)
                                    position = self.order_manager.execute_signal(
                                        refined, confirm=False
                                    )
                                    self._dispatch_multi_account(
                                        refined, is_intraday=True,
                                        bar_interval=bar_interval
                                    )
                                    if position:
                                        _intra_traded_symbols.add(isym)
                                        _intra_trades_today += 1
                                        ts = self.order_manager.create_trailing_state(
                                            position.position_id
                                        )
                                        if ts:
                                            ts.trade_mode = "intraday"
                                            ts.bar_interval = bar_interval
                                            ts.breakeven_ratio = be_ratio
                                            intraday_ts_cfg = intraday_cfg.get(
                                                f"time_stop_bars_{bar_interval.replace('minute', 'min')}",
                                                intraday_cfg.get("time_stop_bars_15min", 20)
                                            )
                                            ts.max_bars = intraday_ts_cfg
                                            self.position_monitor.add_position(ts)
                                            self._handle_state_persist(ts)

                            _last_intra_scan = now

                # ── AFTER HOURS: daily summary + slow sleep ──
                if current_time >= dtime(16, 0):
                    if not _did_send_daily_summary:
                        self._send_daily_summary()
                        _did_send_daily_summary = True
                    time.sleep(60)  # slow poll after hours
                    continue

                # Status
                n_pos = self.position_monitor.active_count if self.position_monitor else 0
                self.dashboard.show_status_line(
                    f"{mode_label}: {n_pos} pos | "
                    f"Intra: {_intra_trades_today}/{intra_max_trades} | "
                    f"Swing scans: {'1PM ' if _did_1pm_scan else ''}{'3:35PM' if _did_335pm_scan else ''}"
                )
                time.sleep(10)
                _consecutive_errors = 0  # Reset on successful iteration

            except KeyboardInterrupt:
                self.running = False
                # Stop monitor FIRST to prevent race with square-off
                if self.position_monitor:
                    self.position_monitor.stop()
                self._square_off_intraday_positions()
                total_pnl = self.order_manager.close_all_positions("session_end")
                if self.multi_account:
                    self.multi_account.close_all("session_end")
                self._persist_equity_state()
                self.stop()
                logger.info(f"{mode_label} stopped. Session P&L: Rs {total_pnl:+,.0f}")
                break
            except Exception as e:
                _consecutive_errors += 1
                should_halt, backoff = self._handle_loop_error(
                    mode_label, e, _consecutive_errors
                )
                if should_halt:
                    self.running = False
                    break
                time.sleep(backoff)

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
        capital_tracker: dict = None,
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

        # Capture regime detector — always use per-bar detection (no look-ahead)
        regime_detector = self.regime_detector
        regime_detector.reset_cache()
        recheck_interval = overrides.get(
            "recheck_bars", 5 if primary_interval == "day" else 10
        )

        # Parrondo regime transition tracking
        _prev_regime = [None]  # mutable container for nonlocal in closure
        _regime_transitions = []

        # ================================================================
        # LOAD LEARNED SIGNAL WEIGHTS (from regression training)
        # Falls back to hardcoded baseline if no weights file exists.
        # ================================================================
        _learned_weights = None
        try:
            import json as _json
            _weights_path = Path(__file__).parent / "config" / "regression_weights.json"
            if _weights_path.exists():
                with open(_weights_path, "r") as _f:
                    _wdata = _json.load(_f)
                _nw = _wdata.get("normalized_weights", {})
                # Scale normalized (0-1) weights to confluence range (0.5-1.5)
                _max_baseline = 1.5
                _learned_weights = {k: v * _max_baseline for k, v in _nw.items()}
                logger.info(f"Loaded learned signal weights from {_weights_path}")
        except Exception:
            pass  # Silently fall back to hardcoded

        # Default hardcoded weights (used if no learned weights)
        _default_weights = {
            "signal_liqsweep": 1.5,
            "signal_fvg": 1.5,
            "signal_vp": 1.0,
            "signal_ote": 1.0,
            "signal_rsi_div": 1.5,
            "signal_vol_surge": 0.5,
            "signal_vol_confirm": 0.5,
            "signal_vwap": 0.5,
            "signal_bias": 0.5,
            "signal_supertrend": 0.0,  # disabled by default (swing unchanged)
            "signal_ema": 0.0,          # disabled by default (swing unchanged)
        }
        _w = _learned_weights if _learned_weights else _default_weights

        # For intraday: ignore learned regression weights (R² negative = worse than random)
        if overrides.get("use_default_weights", False):
            _w = dict(_default_weights)

        # Apply per-indicator weight overrides (e.g., intraday boosts)
        _weight_overrides = overrides.get("weight_overrides", None)
        if _weight_overrides:
            _w = dict(_w)  # defensive copy
            _w.update(_weight_overrides)

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

            # 7. VWAP — session-anchored for intraday, cumulative for daily
            if primary_interval in ("5minute", "15minute"):
                vwap_df = calculate_session_vwap(recent_window.copy())
            else:
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

            # 9. SUPERTREND (trend filter — live parity, gated by weight)
            supertrend_direction = None
            if len(recent_window) >= 20:
                st_df = calculate_supertrend(recent_window)
                if not st_df.empty and "supertrend_direction" in st_df.columns:
                    st_dir = st_df["supertrend_direction"].iloc[-1]
                    supertrend_direction = "bullish" if st_dir == 1 else "bearish"

            # 10. EMA 9/21 TREND ALIGNMENT (momentum confirmation, gated by weight)
            ema_direction = None
            if len(recent_window) >= 21:
                ema9 = calculate_ema(recent_window["close"], 9)
                ema21 = calculate_ema(recent_window["close"], 21)
                if not ema9.empty and not ema21.empty:
                    ema9_now = float(ema9.iloc[-1])
                    ema21_now = float(ema21.iloc[-1])
                    margin = current * 0.001  # 0.1% noise filter
                    if ema9_now > ema21_now + margin:
                        ema_direction = "bullish"
                    elif ema9_now < ema21_now - margin:
                        ema_direction = "bearish"

            return {
                "recent_sweep": recent_sweep, "sweep_direction": sweep_direction,
                "fvg_direction": fvg_direction,
                "vp_direction": vp_direction, "poc": poc, "va_high": va_high, "va_low": va_low,
                "ote_direction": ote_direction,
                "div_direction": div_direction,
                "volume_surge": volume_surge,
                "vwap_val": vwap_val, "vwap_direction": vwap_direction,
                "vol_confirm_dir": vol_confirm_dir,
                "supertrend_direction": supertrend_direction,
                "ema_direction": ema_direction,
            }

        def _price_options(current, direction, atr, data_so_far):
            """Black-Scholes pricing with IV skew. Returns (premium, delta, lot_size, strike, sigma, expiry_str) or None."""
            lot_size = get_lot_size(symbol)
            interval = get_strike_interval(symbol)
            atm_strike = get_atm_strike(current, symbol)

            # OTM for small accounts: lower premium, better capital efficiency
            if capital < 50000:
                if direction == "bullish":
                    strike = atm_strike + interval  # 1-strike OTM CE
                else:
                    strike = atm_strike - interval  # 1-strike OTM PE
            else:
                strike = atm_strike

            bar_date = pd.Timestamp(data_so_far["timestamp"].iloc[-1])
            try:
                dte = days_to_expiry(symbol, from_date=bar_date.date())
                expiry_date = get_expiry_date(symbol, from_date=bar_date.date())
                expiry_date_str = expiry_date.strftime("%Y-%m-%d")
            except Exception:
                dte = 5
                expiry_date_str = ""
            dte = max(dte, 1)
            T = dte / 365.0

            daily_vol = atr / current
            if primary_interval == "day":
                atm_sigma = daily_vol * np.sqrt(252)
            elif primary_interval == "5minute":
                atm_sigma = daily_vol * np.sqrt(78 * 252)  # 78 five-min bars/day
            else:
                atm_sigma = daily_vol * np.sqrt(26 * 252)
            atm_sigma = max(atm_sigma, 0.10)
            atm_sigma = min(atm_sigma, 0.60)

            # Apply IV skew adjustment for strike distance from spot
            sigma = get_implied_vol_at_strike(current, strike, atm_sigma)

            r = 0.07
            opt_type = OptionType.CALL if direction == "bullish" else OptionType.PUT

            premium = black_scholes_price(current, strike, T, r, sigma, opt_type)
            premium = max(premium, current * 0.003)

            greeks = calculate_greeks(current, strike, T, r, sigma, opt_type)
            delta = abs(greeks.get("delta", 0.5))
            delta = max(delta, 0.20)

            return premium, delta, lot_size, strike, sigma, expiry_date_str

        def _size_position(premium, premium_sl, lot_size):
            """Position sizing with drawdown-adjusted risk.
            Uses initial capital for bracket selection, but scales risk DOWN
            proportionally to current drawdown. This is structural protection —
            no tunable parameters that can overfit.
            """
            cap = capital  # initial capital for bracket selection
            if cap < 30000:
                risk_pct = 0.04
            elif cap < 75000:
                risk_pct = 0.03
            else:
                risk_pct = 0.02

            # DD throttle removed from signal generator — engine handles it in
            # _apply_risk_overlays() to avoid double-compounding the DD scalar.

            risk_per_trade = cap * risk_pct
            loss_per_lot = (premium - premium_sl) * lot_size
            if loss_per_lot <= 0:
                return None

            lots = max(1, int(risk_per_trade / loss_per_lot))
            premium_per_lot = premium * lot_size
            max_deploy = 0.45 if cap < 50000 else (0.35 if cap < 100000 else 0.25)
            max_lots = max(1, int((cap * max_deploy) / premium_per_lot)) if premium_per_lot > 0 else 1
            lots = min(lots, max_lots)
            return lots * lot_size

        def _build_signal(direction, premium, premium_sl, premium_target,
                          total_quantity, reasons, time_stop_bars, be_ratio, strategy_prefix="pro",
                          signal_features=None, signal_spot=0.0, atr_at_signal=0.0,
                          option_expiry_date=""):
            """Build the final signal dict."""
            sig = {
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
                "signal_spot": signal_spot,
                "atr_at_signal": atr_at_signal,
                "option_expiry_date": "",  # DTE theta disabled — use bars-held fallback
            }
            # Attach signal features for regression training
            if signal_features:
                sig.update(signal_features)
            return sig

        # ================================================================
        # TREND SIGNAL GENERATION (existing logic)
        # ================================================================

        def _generate_trend_signal(data_so_far, recent_window, current, prev_bar,
                                   atr, bias, min_confluence, indicators, regime_name="unknown"):
            """Trend-following signal: directional confluence → BS pricing → sizing."""
            ind = indicators

            # --- Confluence scoring (using learned or default weights) ---
            bull_score = 0
            bear_score = 0
            bull_reasons = []
            bear_reasons = []

            if ind["sweep_direction"] == "bullish":
                bull_score += _w["signal_liqsweep"]; bull_reasons.append("LiqSweep")
            elif ind["sweep_direction"] == "bearish":
                bear_score += _w["signal_liqsweep"]; bear_reasons.append("LiqSweep")

            if ind["fvg_direction"] == "bullish":
                bull_score += _w["signal_fvg"]; bull_reasons.append("FVG")
            elif ind["fvg_direction"] == "bearish":
                bear_score += _w["signal_fvg"]; bear_reasons.append("FVG")

            if ind["vp_direction"] == "bullish":
                bull_score += _w["signal_vp"]; bull_reasons.append("VP")
            elif ind["vp_direction"] == "bearish":
                bear_score += _w["signal_vp"]; bear_reasons.append("VP")

            if ind["ote_direction"] == "bullish":
                bull_score += _w["signal_ote"]; bull_reasons.append("OTE")
            elif ind["ote_direction"] == "bearish":
                bear_score += _w["signal_ote"]; bear_reasons.append("OTE")

            if ind["div_direction"] == "bullish":
                bull_score += _w["signal_rsi_div"]; bull_reasons.append("RSI_Div")
            elif ind["div_direction"] == "bearish":
                bear_score += _w["signal_rsi_div"]; bear_reasons.append("RSI_Div")

            if ind["volume_surge"]:
                if bull_score > bear_score:
                    bull_score += _w["signal_vol_surge"]; bull_reasons.append("VolSurge")
                elif bear_score > bull_score:
                    bear_score += _w["signal_vol_surge"]; bear_reasons.append("VolSurge")

            if ind["vol_confirm_dir"] == "bullish":
                bull_score += _w["signal_vol_confirm"]; bull_reasons.append("Acc")
            elif ind["vol_confirm_dir"] == "bearish":
                bear_score += _w["signal_vol_confirm"]; bear_reasons.append("Dist")

            if ind["vwap_direction"] == "bullish":
                bull_score += _w["signal_vwap"]; bull_reasons.append("VWAP")
            else:
                bear_score += _w["signal_vwap"]; bear_reasons.append("VWAP")

            # SUPERTREND confluence (weight-gated: 0.0 for swing, >0 for intraday)
            st_w = _w.get("signal_supertrend", 0)
            if st_w > 0:
                if ind.get("supertrend_direction") == "bullish":
                    bull_score += st_w; bull_reasons.append("ST")
                elif ind.get("supertrend_direction") == "bearish":
                    bear_score += st_w; bear_reasons.append("ST")

            # EMA 9/21 alignment (weight-gated: 0.0 for swing, >0 for intraday)
            ema_w = _w.get("signal_ema", 0)
            if ema_w > 0:
                if ind.get("ema_direction") == "bullish":
                    bull_score += ema_w; bull_reasons.append("EMA")
                elif ind.get("ema_direction") == "bearish":
                    bear_score += ema_w; bear_reasons.append("EMA")

            if bias == "bullish":
                bull_score += _w["signal_bias"]
            elif bias == "bearish":
                bear_score += _w["signal_bias"]

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
            premium, delta, lot_size, strike, sigma, expiry_str = pricing

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

            # For small accounts (<30K): cap absolute loss per trade
            # Max loss = Rs 150 per trade → SL = premium - (max_loss / lot_size)
            if capital < 30000:
                max_loss_per_trade = 150  # Rs 150 hard cap
                max_premium_drop = max_loss_per_trade / lot_size
                sl_premium_drop = min(sl_premium_drop, max_premium_drop)
                # No premium floor — let tight SL work
                premium_sl = premium - sl_premium_drop
                premium_sl = max(premium_sl, 0.50)  # absolute minimum Rs 0.50
            elif capital < 50000:
                max_loss_per_trade = 300
                max_premium_drop = max_loss_per_trade / lot_size
                sl_premium_drop = min(sl_premium_drop, max_premium_drop)
                premium_sl = premium - sl_premium_drop
                premium_sl = max(premium_sl, premium * 0.25)
            else:
                premium_sl = max(premium - sl_premium_drop, premium * 0.35)

            risk_check = premium - premium_sl
            if risk_check <= 0:
                return None

            # Target ATR multiplier
            o_target_atr = overrides.get("target_atr_mult", None)
            if o_target_atr is not None:
                base_target = o_target_atr
            elif capital < 30000:
                base_target = 5.0   # Very high target — let trailing stop manage exit
            elif capital < 50000:
                base_target = 4.0
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

            # Min R:R — higher for small accounts (tight SL = need high reward)
            if capital < 30000:
                min_rr = 3.5  # Rs 150 loss → need Rs 525+ target
            elif capital < 50000:
                min_rr = 2.5
            elif capital < 100000:
                min_rr = 2.0
            else:
                min_rr = 1.5

            reward = premium_target - premium
            if risk_check > 0 and reward / risk_check < min_rr:
                premium_target = premium + risk_check * min_rr

            # Kelly gate — adaptive win rate based on confluence score
            # Higher confluence → more confirming signals → higher expected WR
            # Base 30% at min confluence, scales to 45% at high confluence
            kelly_base = overrides.get("kelly_wr", 0.30)
            kelly_wr = min(0.50, kelly_base + score * 0.02)  # +2% per confluence point
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
                time_stop_bars = 7 if capital < 50000 else (6 if capital < 100000 else 5)
            elif primary_interval == "5minute":
                time_stop_bars = 36 if capital < 50000 else (30 if capital < 100000 else 24)
            else:
                time_stop_bars = 16 if capital < 50000 else (12 if capital < 100000 else 10)

            be_ratio = overrides.get("breakeven_ratio", 0.6)

            # --- Signal features for regression training ---
            signal_features = {
                "signal_liqsweep": ind["sweep_direction"] == direction,
                "signal_fvg": ind["fvg_direction"] == direction,
                "signal_vp": ind["vp_direction"] == direction,
                "signal_ote": ind["ote_direction"] == direction,
                "signal_rsi_div": ind["div_direction"] == direction,
                "signal_vol_surge": bool(ind["volume_surge"]),
                "signal_vol_confirm": ind["vol_confirm_dir"] == direction,
                "signal_vwap": ind["vwap_direction"] == direction,
                "signal_bias": bias == direction,
                "signal_supertrend": ind.get("supertrend_direction") == direction,
                "signal_ema": ind.get("ema_direction") == direction,
                "bull_score": float(bull_score),
                "bear_score": float(bear_score),
                "atr_at_entry": float(atr),
                "regime_at_entry": regime_name,
            }

            sig = _build_signal(direction, premium, premium_sl, premium_target,
                                total_quantity, reasons, time_stop_bars, be_ratio, "pro",
                                signal_features=signal_features,
                                signal_spot=current, atr_at_signal=atr,
                                option_expiry_date=expiry_str)
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
            mr_min_score = overrides.get("mr_min_score", 2.5)
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
            premium, delta, lot_size, strike, sigma, expiry_str = pricing

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
            elif primary_interval == "5minute":
                time_stop_bars = 24  # ~2 hours
            else:
                time_stop_bars = 8

            mr_be = overrides.get("mr_breakeven_ratio", 0.4)

            # Build signal features for proper trade attribution
            mr_signal_features = {
                "bull_score": float(mr_bull_score),
                "bear_score": float(mr_bear_score),
                "atr_at_entry": float(atr),
                "regime_at_entry": bar_regime.regime.value if bar_regime else "unknown",
                "signal_liqsweep": False,
                "signal_fvg": False,
                "signal_vp": ind["vp_direction"] == direction,
                "signal_ote": False,
                "signal_rsi_div": ind["div_direction"] == direction,
                "signal_vol_surge": False,
                "signal_vol_confirm": False,
                "signal_vwap": "MR_VWAP" in reasons,
                "signal_bias": bias == direction,
            }

            sig = _build_signal(direction, premium, premium_sl, premium_target,
                                total_quantity, reasons, time_stop_bars, mr_be, "mr",
                                signal_features=mr_signal_features,
                                signal_spot=current, atr_at_signal=atr,
                                option_expiry_date=expiry_str)
            sig["delta"] = delta
            return sig

        def _generate_expiry_spread_signal(data_so_far, recent_window, current, prev_bar, atr,
                                           regime_name="unknown"):
            """
            Expiry debit spread signal — fires on DTE 0-2 days.

            Uses EMA 8/21 crossover for direction.
            Generates defined-risk debit spread: Buy ATM + Sell 1-strike OTM.
            Max cost: 15% of capital.
            """
            lot_size = get_lot_size(symbol)
            bar_date = pd.Timestamp(data_so_far["timestamp"].iloc[-1])
            try:
                dte = days_to_expiry(symbol, from_date=bar_date.date())
            except Exception:
                return None

            if dte > 2:
                return None

            # EMA 8/21 crossover for direction
            close_series = recent_window["close"]
            if len(close_series) < 21:
                return None
            ema8 = calculate_ema(close_series, 8)
            ema21 = calculate_ema(close_series, 21)
            if ema8.empty or ema21.empty:
                return None

            ema8_now = float(ema8.iloc[-1])
            ema21_now = float(ema21.iloc[-1])

            if ema8_now > ema21_now * 1.001:
                direction = "bullish"
            elif ema8_now < ema21_now * 0.999:
                direction = "bearish"
            else:
                return None

            # Price the spread: Buy ATM, Sell 1-strike OTM
            interval = get_strike_interval(symbol)
            atm_strike = get_atm_strike(current, symbol)

            if direction == "bullish":
                buy_strike = atm_strike  # Buy ATM CE
                sell_strike = atm_strike + interval  # Sell OTM CE
            else:
                buy_strike = atm_strike  # Buy ATM PE
                sell_strike = atm_strike - interval  # Sell OTM PE

            T = max(dte, 1) / 365.0
            daily_vol = atr / current
            if primary_interval == "day":
                atm_sigma = daily_vol * np.sqrt(252)
            elif primary_interval == "5minute":
                atm_sigma = daily_vol * np.sqrt(78 * 252)  # 78 five-min bars/day
            else:
                atm_sigma = daily_vol * np.sqrt(26 * 252)
            atm_sigma = max(atm_sigma, 0.10)
            atm_sigma = min(atm_sigma, 0.60)

            r = 0.07
            opt_type = OptionType.CALL if direction == "bullish" else OptionType.PUT

            buy_premium = black_scholes_price(current, buy_strike, T, r, atm_sigma, opt_type)
            sell_premium = black_scholes_price(current, sell_strike, T, r, atm_sigma, opt_type)
            buy_premium = max(buy_premium, current * 0.002)
            sell_premium = max(sell_premium, 0)

            # Net debit = what we pay
            net_debit = buy_premium - sell_premium
            if net_debit <= 0:
                return None

            # Max profit = strike width - net debit (for debit spread)
            max_profit = interval - net_debit
            if max_profit <= 0:
                return None

            # R:R must be at least 1.5:1
            rr = max_profit / net_debit
            if rr < 1.5:
                return None

            # Max cost check: 15% of capital
            spread_cost = net_debit * lot_size
            if spread_cost > capital * 0.15:
                return None

            # Position sizing: 1 lot for spreads (defined risk)
            total_quantity = lot_size

            # SL: lose max 80% of debit (don't hold to total loss)
            premium_sl = net_debit * 0.20  # exit when premium drops to 20% of debit
            # Target: 80% of max profit
            premium_target = net_debit + max_profit * 0.80

            greeks = calculate_greeks(current, buy_strike, T, r, atm_sigma, opt_type)
            delta = abs(greeks.get("delta", 0.5))

            # Time stop: exit 1 bar before expiry if on daily, or 10 bars on 15min
            time_stop = 1 if primary_interval == "day" else 10

            try:
                expiry_date = get_expiry_date(symbol, from_date=bar_date.date())
                expiry_str = expiry_date.strftime("%Y-%m-%d")
            except Exception:
                expiry_str = ""

            expiry_signal_features = {
                "bull_score": 0.0,
                "bear_score": 0.0,
                "atr_at_entry": float(atr),
                "regime_at_entry": regime_name,
                "signal_liqsweep": False,
                "signal_fvg": False,
                "signal_vp": False,
                "signal_ote": False,
                "signal_rsi_div": False,
                "signal_vol_surge": False,
                "signal_vol_confirm": False,
                "signal_vwap": False,
                "signal_bias": False,
            }

            sig = _build_signal(
                direction, net_debit, premium_sl, premium_target,
                total_quantity, [f"Spread_DTE{dte}"], time_stop, 0.3,
                strategy_prefix="expiry",
                signal_features=expiry_signal_features,
                signal_spot=current,
                atr_at_signal=atr,
                option_expiry_date=expiry_str)
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
            # REGIME ROUTING (per-bar detection — no look-ahead bias)
            # ================================================================
            bar_regime = regime_detector.detect_fast(data_so_far, recheck_every=recheck_interval)
            regime_value = bar_regime.regime.value

            if parrondo:
                # PARRONDO MODE: full regime routing + MR + volatile skip
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
                    # Fallback: trend with higher confluence
                    conf_sideways = overrides.get("confluence_sideways", 3.5)
                    return _generate_trend_signal(
                        data_so_far, recent_window, current, prev_bar,
                        atr, bias, conf_sideways, indicators, regime_name=regime_value
                    )
                else:
                    # TREND mode (markup, markdown, unknown)
                    conf_trending = overrides.get("confluence_trending", 3.0)
                    trend_sig = _generate_trend_signal(
                        data_so_far, recent_window, current, prev_bar,
                        atr, bias, conf_trending, indicators, regime_name=regime_value
                    )
                    if trend_sig:
                        return trend_sig
                    return None
            else:
                # NON-PARRONDO: per-bar regime for confluence threshold only
                conf_trending = overrides.get("confluence_trending", 3.0)
                conf_sideways = overrides.get("confluence_sideways", 3.5)
                if regime_value == "volatile":
                    # Capital preservation — skip volatile regime
                    return None
                elif regime_value in ("accumulation", "distribution", "unknown"):
                    min_confluence = conf_sideways
                else:
                    min_confluence = conf_trending

                trend_sig = _generate_trend_signal(
                    data_so_far, recent_window, current, prev_bar,
                    atr, bias, min_confluence, indicators, regime_name=regime_value
                )
                if trend_sig:
                    return trend_sig
                return None

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
        entry_timing: bool = False,
        entry_pullback_atr: float = 0.3,
        entry_max_wait_bars: int = 2,
        vol_target: float = 0.0,
        dd_throttle: bool = True,
        equity_curve_filter: bool = False,
    ):
        """
        Run a backtest on an arbitrary date-sliced DataFrame.
        Handles regime detection, bias, and signal generator independently (no leakage).
        Returns (BacktestResult, BacktestEngine).
        """
        # Extract regime_overrides if present in param_overrides (for tuning sweeps)
        overrides = param_overrides or {}
        regime_overrides = overrides.get("regime_overrides", None)
        mr_min_score = overrides.get("mr_min_score", 2.5)

        # Create a tuned regime detector if overrides provided
        if regime_overrides:
            tuned_detector = RegimeDetector(**regime_overrides)
            regime_state = tuned_detector.detect(data_slice) if len(data_slice) >= 50 else None
        else:
            tuned_detector = None
            regime_state = self.regime_detector.detect(data_slice) if len(data_slice) >= 50 else None
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

        # Mutable container for dynamic capital tracking between engine and signal generator
        capital_tracker = {"capital": capital, "peak": capital}

        # If regime_overrides provided, force parrondo=True for per-bar detection
        use_parrondo = parrondo or bool(regime_overrides)

        # Create clean param_overrides (without regime_overrides) for signal generator
        clean_param_overrides = {k: v for k, v in overrides.items() if k not in ["regime_overrides"]}

        # Set tuned detector if available
        if tuned_detector:
            original_detector = self.regime_detector
            self.regime_detector = tuned_detector

        signal_gen = self._make_signal_generator(
            regime_state=regime_state,
            hourly_bias_map=hourly_bias_map,
            capital=capital,
            primary_interval="day",
            symbol=symbol,
            param_overrides=clean_param_overrides,
            parrondo=use_parrondo,
            capital_tracker=capital_tracker,
        )

        # Restore original detector if we swapped it
        if tuned_detector:
            self.regime_detector = original_detector

        # Run backtest
        cost_cfg = get("backtest.costs", {})
        # Cap positions by capital: <30K can only afford 1 position at a time
        max_pos = 1 if self.initial_capital < 30000 else 2
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_config=cost_cfg,
            entry_timing=entry_timing,
            entry_pullback_atr=entry_pullback_atr,
            entry_max_wait_bars=entry_max_wait_bars,
            capital_tracker=capital_tracker,
            max_positions=max_pos,
            vol_target=vol_target,
            dd_throttle=dd_throttle,
            equity_curve_filter=equity_curve_filter,
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
    # MODE: INTRADAY BACKTEST (completely separate from swing backtest)
    # ─────────────────────────────────────────────────────────────────────
    def _run_intraday_backtest_on_slice(
        self,
        data_slice: pd.DataFrame,
        data_daily: pd.DataFrame,
        symbol: str,
        bar_interval: str = "15minute",
        strategy_name: str = "intraday",
        parrondo: bool = False,
        dd_throttle: bool = True,
        param_overrides: dict = None,
        verbose: bool = True,
    ):
        """
        Run intraday backtest on a slice of intraday data.
        Uses separate daily data for regime detection (intraday data too short).
        Returns (BacktestResult, BacktestEngine).
        """
        overrides = param_overrides or {}

        # Regime detection from daily data (intraday bars too short for regime)
        regime_state = self.regime_detector.detect(data_daily) if len(data_daily) >= 50 else None
        if regime_state and verbose:
            logger.info(f"  Regime: {regime_state.regime.value} | Confidence: {regime_state.confidence:.2f}")

        # Compute bias from daily data
        hourly_bias_map = self._compute_daily_bias(data_daily)

        # Capital tracker
        capital = self.initial_capital
        capital_tracker = {"capital": capital, "peak": capital}

        # Intraday-specific overrides
        intraday_cfg = get("intraday", {})
        intraday_overrides = {
            "breakeven_ratio": intraday_cfg.get("breakeven_ratio", 0.5),
            # Force default weights + activate intraday-specific indicators
            "use_default_weights": True,
            "weight_overrides": {
                "signal_supertrend": 1.0,   # Activate Supertrend for intraday
                "signal_ema": 0.75,          # EMA 9/21 trend alignment
                "signal_vwap": 1.0,          # Boost session VWAP (key intraday indicator)
            },
        }
        intraday_overrides.update(overrides)

        # Create signal generator with intraday interval
        signal_gen = self._make_signal_generator(
            regime_state=regime_state,
            hourly_bias_map=hourly_bias_map,
            capital=capital,
            primary_interval=bar_interval,
            symbol=symbol,
            param_overrides=intraday_overrides,
            parrondo=parrondo,
            capital_tracker=capital_tracker,
        )

        # Create engine with intraday session enforcement
        cost_cfg = get("backtest.costs", {})
        max_pos = 1 if self.initial_capital < 30000 else 2
        max_intra_trades = intraday_cfg.get("max_daily_trades", 4)

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_config=cost_cfg,
            capital_tracker=capital_tracker,
            max_positions=max_pos,
            dd_throttle=dd_throttle,
            intraday_session=True,
            session_open_time="09:45",
            session_no_entry_time=intraday_cfg.get("last_entry_time", "14:30"),
            session_close_time=intraday_cfg.get("square_off_time", "15:15"),
            max_intraday_trades_per_day=max_intra_trades,
        )

        warmup = 20 if bar_interval == "5minute" else 10
        result = engine.run(
            data=data_slice,
            signal_generator=signal_gen,
            strategy_name=strategy_name,
            warmup_bars=warmup,
        )

        if verbose:
            print(result.summary())

            # Monte Carlo
            if result.total_trades > 5:
                mc = engine.monte_carlo_simulation(result, num_simulations=1000)
                print(f"\n--- Monte Carlo Simulation (1000 runs) ---")
                print(f"  Probability of profit: {mc.get('prob_profit', 0):.1f}%")
                print(f"  Median final capital: Rs {mc.get('median_final_capital', 0):,.0f}")
                print(f"  5th percentile: Rs {mc.get('p5_final_capital', 0):,.0f}")
                print(f"  95th percentile: Rs {mc.get('p95_final_capital', 0):,.0f}")

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
                print(f"\n--- Exit Reason Analysis ---")
                for r, s in sorted(reason_stats.items(), key=lambda x: -x[1]["count"]):
                    wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                    print(f"  {r}: {s['count']} trades, {wr:.0f}% WR, PnL Rs {s['total_pnl']:+,.0f}")

            # Intraday-specific metrics
            intra_metrics = engine.calculate_intraday_metrics()
            if intra_metrics:
                print(f"\n--- Intraday Session Metrics ---")
                print(f"  Total sessions: {intra_metrics['total_sessions']}")
                print(f"  Avg trades/session: {intra_metrics['avg_trades_per_session']}")
                print(f"  Session win rate: {intra_metrics['session_win_rate']:.1f}%")
                print(f"  Square-off exits: {intra_metrics['square_off_exits']} ({intra_metrics['square_off_pct']:.1f}%)")
                print(f"  Best session P&L: Rs {intra_metrics['best_session_pnl']:+,.0f}")
                print(f"  Worst session P&L: Rs {intra_metrics['worst_session_pnl']:+,.0f}")
                if intra_metrics.get("entry_hour_distribution"):
                    print(f"  Entry hour distribution:")
                    for hr, cnt in intra_metrics["entry_hour_distribution"].items():
                        print(f"    {hr:02d}:00 - {cnt} trades")

        return result, engine

    def run_intraday_backtest(
        self,
        symbol: str = "NIFTY 50",
        days: int = 59,
        bar_interval: str = "auto",
        parrondo: bool = False,
        dd_throttle: bool = True,
        param_overrides: Optional[Dict] = None,
    ):
        """Run intraday backtest — completely separate from swing backtest."""
        self.dashboard.show_header()

        # Auto-select interval
        if bar_interval == "auto":
            bar_interval = self._select_intraday_interval()
        logger.info(f"Starting INTRADAY backtest: {symbol} ({days} days, {bar_interval})" +
                     (" [PARRONDO]" if parrondo else ""))

        # Cap days to yfinance limit for intraday (unless Angel One is available)
        if days > 59 and not self.data.angelone:
            logger.warning(f"yfinance limits intraday data to ~60 days. Capping {days} -> 59 days.")
            days = 59

        # Fetch intraday data
        data_intraday = self.data.fetch_historical(symbol, days=days, interval=bar_interval)
        if data_intraday.empty:
            logger.error(f"No {bar_interval} data available for {symbol}. Check yfinance.")
            return

        # Fetch daily data separately for regime detection
        data_daily = self.data.fetch_historical(symbol, days=max(days, 120), interval="day")

        logger.info(f"Loaded: {len(data_intraday)} x {bar_interval}, {len(data_daily)} x daily bars")

        print(f"\n{'='*70}")
        print(f" INTRADAY BACKTEST: {symbol} | {bar_interval} | {days} days")
        print(f" Capital: Rs {self.initial_capital:,.0f} | Parrondo: {'ON' if parrondo else 'OFF'}")
        print(f" Session: 09:45-14:30 entries | 15:15 square-off")
        print(f"{'='*70}")

        result, engine = self._run_intraday_backtest_on_slice(
            data_slice=data_intraday,
            data_daily=data_daily,
            symbol=symbol,
            bar_interval=bar_interval,
            strategy_name=f"intraday_{bar_interval}",
            parrondo=parrondo,
            dd_throttle=dd_throttle,
            param_overrides=param_overrides,
        )

        return result, engine

    def run_intraday_walkforward(
        self,
        symbol: str = "NIFTY 50",
        bar_interval: str = "auto",
        train_pct: float = 0.65,
        parrondo: bool = False,
        dd_throttle: bool = True,
    ):
        """Intraday walk-forward validation (percentage-split, ~60 day limit)."""
        self.dashboard.show_header()

        if bar_interval == "auto":
            bar_interval = self._select_intraday_interval()
        logger.info(f"Starting INTRADAY walk-forward: {symbol} ({bar_interval})")

        # Fetch max available intraday data
        data_intraday = self.data.fetch_historical(symbol, days=59, interval=bar_interval)
        data_daily = self.data.fetch_historical(symbol, days=180, interval="day")

        if len(data_intraday) < 100:
            logger.error(f"Insufficient intraday data ({len(data_intraday)} bars). Need at least 100.")
            return

        # Split by percentage
        split_idx = int(len(data_intraday) * train_pct)
        data_train = data_intraday.iloc[:split_idx].copy().reset_index(drop=True)
        data_test = data_intraday.iloc[split_idx:].copy().reset_index(drop=True)

        train_start = str(data_train["timestamp"].iloc[0])[:10]
        train_end = str(data_train["timestamp"].iloc[-1])[:10]
        test_start = str(data_test["timestamp"].iloc[0])[:10]
        test_end = str(data_test["timestamp"].iloc[-1])[:10]

        logger.info(f"Train: {train_start} to {train_end} ({len(data_train)} bars)")
        logger.info(f"Test:  {test_start} to {test_end} ({len(data_test)} bars)")

        print(f"\n{'='*70}")
        print(f" INTRADAY WALK-FORWARD: {symbol} | {bar_interval}")
        print(f" Train: {train_start} to {train_end} ({len(data_train)} bars)")
        print(f" Test:  {test_start} to {test_end} ({len(data_test)} bars)")
        print(f" WARNING: Limited data (~60 days via yfinance)")
        print(f"{'='*70}")

        # In-sample
        print(f"\n--- IN-SAMPLE (Train: {train_start} to {train_end}) ---")
        result_is, engine_is = self._run_intraday_backtest_on_slice(
            data_slice=data_train,
            data_daily=data_daily,
            symbol=symbol,
            bar_interval=bar_interval,
            strategy_name="intraday_IS",
            parrondo=parrondo,
            dd_throttle=dd_throttle,
        )

        # Out-of-sample
        print(f"\n--- OUT-OF-SAMPLE (Test: {test_start} to {test_end}) ---")
        result_oos, engine_oos = self._run_intraday_backtest_on_slice(
            data_slice=data_test,
            data_daily=data_daily,
            symbol=symbol,
            bar_interval=bar_interval,
            strategy_name="intraday_OOS",
            parrondo=parrondo,
            dd_throttle=dd_throttle,
        )

        # Comparison
        print(f"\n{'='*70}")
        print(f" INTRADAY WALK-FORWARD COMPARISON")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'IS':>15} {'OOS':>15} {'Degradation':>15}")
        print(f"{'-'*70}")

        def safe_pf(r):
            return getattr(r, 'profit_factor', 0) or 0
        def safe_sharpe(r):
            return getattr(r, 'sharpe_ratio', 0) or 0
        def safe_dd(r):
            return getattr(r, 'max_drawdown_pct', 0) or 0

        is_pf = safe_pf(result_is)
        oos_pf = safe_pf(result_oos)
        pf_deg = ((oos_pf - is_pf) / is_pf * 100) if is_pf > 0 else 0

        print(f"{'Profit Factor':<25} {is_pf:>15.2f} {oos_pf:>15.2f} {pf_deg:>14.1f}%")
        print(f"{'Sharpe Ratio':<25} {safe_sharpe(result_is):>15.2f} {safe_sharpe(result_oos):>15.2f}")
        print(f"{'Max Drawdown':<25} {safe_dd(result_is):>14.1f}% {safe_dd(result_oos):>14.1f}%")
        print(f"{'Trades':<25} {result_is.total_trades:>15} {result_oos.total_trades:>15}")
        print(f"{'Final Capital':<25} {'Rs {:,.0f}'.format(result_is.final_capital):>15} {'Rs {:,.0f}'.format(result_oos.final_capital):>15}")

        # Validation criteria
        print(f"\n--- Validation Criteria ---")
        checks = [
            ("OOS PF > 1.0", oos_pf > 1.0),
            ("PF degradation < 50%", pf_deg > -50),
            ("OOS trades >= 5", result_oos.total_trades >= 5),
            ("OOS profitable", result_oos.final_capital > self.initial_capital),
        ]
        pass_count = 0
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            if passed:
                pass_count += 1
            print(f"  {status}: {name}")
        print(f"\n  Result: {pass_count}/{len(checks)} criteria passed")

    # ─────────────────────────────────────────────────────────────────────
    # MODE: BACKTEST (swing — LOCKED, DO NOT MODIFY)
    # ─────────────────────────────────────────────────────────────────────
    def run_backtest(
        self,
        symbol: str = "NIFTY 50",
        days: int = 365,
        strategy: str = "trend",
        parrondo: bool = False,
        regime_overrides: Optional[Dict] = None,
        mr_min_score: float = 2.5,
        entry_timing: bool = False,
        entry_pullback_atr: float = 0.3,
        entry_max_wait_bars: int = 2,
        vol_target: float = 0.0,
        dd_throttle: bool = True,
        equity_curve_filter: bool = False,
        param_overrides: Optional[Dict] = None,
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
        # CREATE TUNED REGIME DETECTOR (if regime_overrides provided for sensitivity sweep)
        # ================================================================
        if regime_overrides:
            self.regime_detector = RegimeDetector(**regime_overrides)
            logger.info(f"Parrondo regime overrides applied: {regime_overrides}")

        # ================================================================
        # REGIME DETECTION (display only — trading uses per-bar detect_fast)
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
        # Always use factory signal generator (unified path — no look-ahead bias)
        self.regime_detector.reset_cache()
        param_overrides_dict = {"mr_min_score": mr_min_score}
        if regime_overrides:
            if "recheck_bars" in regime_overrides:
                param_overrides_dict["recheck_bars"] = regime_overrides["recheck_bars"]
        if param_overrides:
            param_overrides_dict.update(param_overrides)

        # Track capital across bars for dynamic sizing
        capital = self.initial_capital
        capital_tracker = {"capital": capital, "peak": capital}

        pro_signal_generator = self._make_signal_generator(
            regime_state=regime_state,
            hourly_bias_map=hourly_bias_map,
            capital=self.initial_capital,
            primary_interval=primary_interval,
            symbol=symbol,
            param_overrides=param_overrides_dict,
            parrondo=parrondo,
            capital_tracker=capital_tracker,
        )

        # Run backtest on primary timeframe data
        cost_cfg = get("backtest.costs", {})
        max_pos = 1 if self.initial_capital < 30000 else 2
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_config=cost_cfg,
            entry_timing=entry_timing,
            entry_pullback_atr=entry_pullback_atr,
            entry_max_wait_bars=entry_max_wait_bars,
            capital_tracker=capital_tracker,
            max_positions=max_pos,
            vol_target=vol_target,
            dd_throttle=dd_throttle,
            equity_curve_filter=equity_curve_filter,
        )

        logger.info(f"Running backtest on {len(data_primary)} bars of {primary_interval} data..."
                     + (" [PARRONDO]" if parrondo else "")
                     + (" [RISK-OVERLAYS]" if (vol_target or dd_throttle or equity_curve_filter) else ""))

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

        # Risk overlay stats
        ros = engine.risk_overlay_stats
        if ros["signals_received"] > 0 and (vol_target or dd_throttle or equity_curve_filter):
            print("\n--- Risk Overlay Stats ---")
            print(f"  Signals received: {ros['signals_received']}")
            if equity_curve_filter:
                print(f"  Equity-filtered (skipped): {ros['equity_filtered']} "
                      f"({ros['equity_filtered']/ros['signals_received']*100:.0f}%)")
            if vol_target:
                print(f"  Vol-scaled: {ros['vol_scaled']} (target={vol_target*100:.0f}%)")
            if dd_throttle:
                print(f"  DD-throttled: {ros['dd_throttled']}")

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

        # Entry timing analysis
        if entry_timing and hasattr(engine, 'entry_timing_stats'):
            stats = engine.entry_timing_stats
            total_signals = stats.get("signals_generated", 0)
            filled_pullback = stats.get("filled_at_pullback", 0)
            filled_open = stats.get("filled_at_open", 0)
            expired = stats.get("expired_no_fill", 0)
            total_filled = filled_pullback + filled_open
            fill_rate = total_filled / total_signals * 100 if total_signals > 0 else 0
            print(f"\n--- Entry Timing Analysis ---")
            print(f"  Signals generated: {total_signals}")
            print(f"  Filled at pullback limit: {filled_pullback}")
            print(f"  Filled at gap open: {filled_open}")
            print(f"  Expired (no fill): {expired}")
            print(f"  Fill rate: {fill_rate:.1f}%")
            print(f"  Pullback ATR fraction: {entry_pullback_atr}")
            print(f"  Max wait bars: {entry_max_wait_bars}")

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
        entry_timing: bool = False,
        entry_pullback_atr: float = 0.3,
        entry_max_wait_bars: int = 2,
        vol_target: float = 0.0,
        dd_throttle: bool = True,
        equity_curve_filter: bool = False,
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
            entry_timing=entry_timing,
            entry_pullback_atr=entry_pullback_atr,
            entry_max_wait_bars=entry_max_wait_bars,
            vol_target=vol_target,
            dd_throttle=dd_throttle,
            equity_curve_filter=equity_curve_filter,
        )

        # ── OUT-OF-SAMPLE ──
        print("\n" + "=" * 70)
        print("  OUT-OF-SAMPLE (Test Period — UNSEEN DATA)")
        print("=" * 70)
        result_test, engine_test = self._run_backtest_on_slice(
            data_test, symbol, f"WF_test_{symbol.replace(' ', '_')}",
            parrondo=parrondo,
            entry_timing=entry_timing,
            entry_pullback_atr=entry_pullback_atr,
            entry_max_wait_bars=entry_max_wait_bars,
            vol_target=vol_target,
            dd_throttle=dd_throttle,
            equity_curve_filter=equity_curve_filter,
        )

        # ── COMPARISON ──
        self._display_walkforward_comparison(result_train, result_test, symbol,
                                             engine_train=engine_train,
                                             engine_test=engine_test)

        return result_train, result_test

    def _display_walkforward_comparison(
        self,
        result_train,
        result_test,
        symbol: str,
        engine_train=None,
        engine_test=None,
    ):
        """Compare in-sample vs out-of-sample with pass/fail criteria."""
        print("\n" + "=" * 70)
        print("  WALK-FORWARD COMPARISON")
        print("=" * 70)

        metrics = [
            ("Total Return %", result_train.total_return_pct, result_test.total_return_pct),
            ("Annualized Return %", result_train.annualized_return_pct, result_test.annualized_return_pct),
            ("Alpha vs Buy-Hold %", result_train.alpha_pct, result_test.alpha_pct),
            ("Total Trades", result_train.total_trades, result_test.total_trades),
            ("Win Rate %", result_train.win_rate * 100 if result_train.win_rate <= 1 else result_train.win_rate,
                           result_test.win_rate * 100 if result_test.win_rate <= 1 else result_test.win_rate),
            ("Profit Factor", result_train.profit_factor, result_test.profit_factor),
            ("Sharpe Ratio", result_train.sharpe_ratio, result_test.sharpe_ratio),
            ("Calmar Ratio", result_train.calmar_ratio, result_test.calmar_ratio),
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

        # ── Monte Carlo on OOS ──
        mc_oos = None
        if engine_test and result_test.total_trades > 10:
            mc_oos = engine_test.monte_carlo_simulation(result_test, num_simulations=1000)
            print("\n  " + "-" * 65)
            print("  MONTE CARLO ON OUT-OF-SAMPLE (1000 sims, block bootstrap)")
            print("  " + "-" * 65)
            print(f"  Probability of profit:  {mc_oos.get('prob_profit', 0):.1f}% "
                  f"(±{mc_oos.get('prob_profit_ci', 0):.1f}%)")
            print(f"  Median final capital:   Rs {mc_oos.get('median_final_capital', 0):,.0f}")
            print(f"  5th pctl (worst case):  Rs {mc_oos.get('p5_final_capital', 0):,.0f}")
            print(f"  95th pctl (best case):  Rs {mc_oos.get('p95_final_capital', 0):,.0f}")
            print(f"  Median Sharpe:          {mc_oos.get('median_sharpe', 0):.2f}")
            print(f"  5th pctl Sharpe:        {mc_oos.get('p5_sharpe', 0):.2f}")
            print(f"  Median max drawdown:    {mc_oos.get('median_max_drawdown', 0):.1f}%")
            print(f"  Prob of >20% drawdown:  {mc_oos.get('prob_20pct_drawdown', 0):.1f}%")

        # ── PBO (Probability of Backtest Overfitting) ──
        pbo_result = None
        if engine_train and result_train.total_trades >= 20:
            pbo_result = engine_train.probability_of_backtest_overfitting(
                result_train, n_partitions=10
            )
            if "error" not in pbo_result:
                print("\n  " + "-" * 65)
                print("  PBO — Probability of Backtest Overfitting (CSCV)")
                print("  " + "-" * 65)
                print(f"  PBO score:       {pbo_result['pbo']:.3f}  "
                      f"(<0.30 robust, 0.30-0.50 borderline, >0.50 overfit)")
                print(f"  Partitions:      {pbo_result['n_partitions']} "
                      f"({pbo_result['n_combinations']} combinations)")
                print(f"  Mean logit:      {pbo_result['mean_logit']:.3f}")
                print(f"  Verdict:         {pbo_result['verdict']}")

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

        # 6. OOS Alpha positive
        alpha_pass = result_test.alpha_pct > 0
        checks.append(("OOS Alpha > 0 (beats buy-hold)", alpha_pass,
                        f"Alpha = {result_test.alpha_pct:+.1f}%"))

        # 7. OOS Calmar > 0.5
        calmar_pass = result_test.calmar_ratio > 0.5
        checks.append(("OOS Calmar > 0.5", calmar_pass,
                        f"Calmar = {result_test.calmar_ratio:.2f}"))

        # 8. MC OOS P(profit) > 90%
        if mc_oos and "prob_profit" in mc_oos:
            mc_pass = mc_oos["prob_profit"] > 90
            checks.append(("MC OOS P(profit) > 90%", mc_pass,
                            f"P(profit) = {mc_oos['prob_profit']:.1f}%"))

        # 9. PBO < 0.50
        if pbo_result and "pbo" in pbo_result:
            pbo_pass = pbo_result["pbo"] < 0.50
            checks.append(("PBO < 0.50 (not overfit)", pbo_pass,
                            f"PBO = {pbo_result['pbo']:.3f}"))

        for name, passed, detail in checks:
            marker = "[PASS]" if passed else "[FAIL]"
            print(f"  {marker} {name}: {detail}")

        pass_count = sum(1 for _, p, _ in checks if p)
        total_checks = len(checks)

        print("\n  " + "=" * 65)
        if pass_count == total_checks:
            print(f"  VERDICT: ALL {total_checks}/{total_checks} PASSED — System is VALIDATED")
            print(f"  The strategy generalizes to unseen data. NOT overfit.")
        elif pass_count >= total_checks - 2:
            print(f"  VERDICT: {pass_count}/{total_checks} PASSED — PARTIALLY VALIDATED")
            print(f"  Strategy shows promise but review failed criteria.")
        else:
            print(f"  VERDICT: {pass_count}/{total_checks} PASSED — MAY BE OVERFIT")
            print(f"  Significant degradation on unseen data.")
        print("  " + "=" * 65)

    # ─────────────────────────────────────────────────────────────────────
    # MODE: PARRONDO REGIME TUNING — 243-COMBO SENSITIVITY SWEEP
    # ─────────────────────────────────────────────────────────────────────
    def run_parrondo_tuning(
        self,
        symbol: str = "BANKNIFTY",
        days: int = 5475,
    ):
        """
        Parrondo regime-switching parameter sensitivity sweep (243 combinations).

        Tests all 5 tunable regime detection parameters:
        - trend_strength_strong: {0.35, 0.4, 0.45} (MARKUP/MARKDOWN threshold)
        - trend_strength_sideways: {0.25, 0.3, 0.35} (sideways classification)
        - vol_expanding_mult: {1.15, 1.2, 1.25} (volatility expansion detector)
        - hurst_accumulation: {0.40, 0.45, 0.50} (mean-reversion gate)
        - mr_min_score: {2.0, 2.5, 3.0} (confluence minimum for MR trades)

        Filters for: WR >= 52% AND DD <= 55% AND PF >= 1.4
        """
        self.dashboard.show_header()
        print("\n" + "=" * 80)
        print("  PARRONDO REGIME TUNING — 243-COMBO SENSITIVITY SWEEP")
        print(f"  Symbol: {symbol} | Period: {days} days (15yr baseline)")
        print("=" * 80)

        # Define parameter ranges (3 x 3 x 3 x 3 x 3 = 243 combos)
        param_grid = {
            "trend_strength_strong": [0.35, 0.40, 0.45],
            "trend_strength_sideways": [0.25, 0.30, 0.35],
            "vol_expanding_mult": [1.15, 1.20, 1.25],
            "hurst_accumulation": [0.40, 0.45, 0.50],
            "mr_min_score": [2.0, 2.5, 3.0],
        }

        # Fetch data once (avoid repeated network calls)
        logger.info(f"Fetching {days} days of daily data for {symbol}...")
        data_all = self.data.fetch_historical(symbol, days=days, interval="day", force_refresh=True)

        if data_all.empty or len(data_all) < 100:
            logger.error(f"Insufficient data: {len(data_all)} bars")
            return None

        print(f"\n  Data: {len(data_all)} bars ({str(data_all['timestamp'].iloc[0])[:10]} to {str(data_all['timestamp'].iloc[-1])[:10]})")

        # Run baseline (defaults)
        print("\n  --- Baseline (default params) ---")
        baseline, _ = self._run_backtest_on_slice(
            data_all, symbol, "baseline", param_overrides=None, verbose=False
        )
        print(f"  Baseline: PF={baseline.profit_factor:.2f}, Sharpe={baseline.sharpe_ratio:.2f}, "
              f"WR={baseline.win_rate*100 if baseline.win_rate <= 1 else baseline.win_rate:.0f}%, "
              f"DD={baseline.max_drawdown_pct:.1f}%, Trades={baseline.total_trades}")

        # Generate all 243 combinations
        import itertools
        combos = list(itertools.product(
            param_grid["trend_strength_strong"],
            param_grid["trend_strength_sideways"],
            param_grid["vol_expanding_mult"],
            param_grid["hurst_accumulation"],
            param_grid["mr_min_score"],
        ))

        print(f"\n  Running {len(combos)} parameter combinations...")
        print(f"  (Progress will update every 24 combos)")
        print("  ─" * 40)

        results = []
        for idx, (strong, sideways, vol_exp, hurst, mr_score) in enumerate(combos):
            regime_override = {
                "trend_strength_strong": strong,
                "trend_strength_sideways": sideways,
                "vol_expanding_mult": vol_exp,
                "hurst_accumulation": hurst,
            }

            # Build label for tracking
            label = (f"TS{strong:.2f}_SS{sideways:.2f}_VE{vol_exp:.2f}_"
                    f"HA{hurst:.2f}_MR{mr_score:.1f}")

            try:
                # Run backtest with these specific overrides
                result, trades_df = self._run_backtest_on_slice(
                    data_all,
                    symbol,
                    label,
                    param_overrides={
                        "regime_overrides": regime_override,
                        "mr_min_score": mr_score,
                    },
                    verbose=False,
                )

                results.append({
                    "combo": label,
                    "trend_strength_strong": strong,
                    "trend_strength_sideways": sideways,
                    "vol_expanding_mult": vol_exp,
                    "hurst_accumulation": hurst,
                    "mr_min_score": mr_score,
                    "pf": result.profit_factor,
                    "sharpe": result.sharpe_ratio,
                    "wr": result.win_rate,
                    "dd": result.max_drawdown_pct,
                    "return": result.total_return_pct,
                    "trades": result.total_trades,
                })

            except Exception as e:
                logger.warning(f"Combo {label} failed: {e}")
                results.append({
                    "combo": label,
                    "trend_strength_strong": strong,
                    "trend_strength_sideways": sideways,
                    "vol_expanding_mult": vol_exp,
                    "hurst_accumulation": hurst,
                    "mr_min_score": mr_score,
                    "pf": 0,
                    "sharpe": 0,
                    "wr": 0,
                    "dd": 100,
                    "return": 0,
                    "trades": 0,
                })

            # Progress update every 24 combos
            if (idx + 1) % 24 == 0:
                print(f"  {idx + 1}/{len(combos)} combos tested...")

        # Filter for good combos: WR >= 52% AND DD <= 55% AND PF >= 1.4
        filtered = [
            r for r in results
            if r['wr'] >= 0.52 and r['dd'] <= 55.0 and r['pf'] >= 1.4
        ]

        print(f"\n  ═" * 40)
        print(f"  RESULTS: {len(filtered)} combos passed filters (WR>=52%, DD<=55%, PF>=1.4)")
        print(f"  ═" * 40)

        # Sort by Sharpe (best risk-adjusted returns first)
        filtered.sort(key=lambda x: -x['sharpe'])

        # Print top 20
        print(f"\n  TOP 20 COMBOS (sorted by Sharpe):\n")
        print(f"{'Rank':<6} {'Combo':<40} {'PF':<6} {'Sharpe':<7} {'WR':<6} {'DD':<7} {'MR_Min':<7}")
        print("  ─" * 70)

        for rank, r in enumerate(filtered[:20], 1):
            print(f"  {rank:<5} {r['combo'][:39]:<40} {r['pf']:>5.2f} {r['sharpe']:>6.2f} "
                  f"{r['wr']*100:>5.0f}% {r['dd']:>6.1f}% {r['mr_min_score']:>6.1f}")

        # Save full results to CSV
        import csv
        csv_file = f"parrondo_sweep_{symbol}_{days}days.csv"
        try:
            import os
            csv_path = os.path.join(os.path.dirname(__file__), "..", csv_file)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  Full results saved to: {csv_file}")
        except Exception as e:
            logger.warning(f"Could not save CSV: {e}")

        # Print best combo summary
        if filtered:
            best = filtered[0]
            print(f"\n  ✓ BEST COMBO (Sharpe-optimized):")
            print(f"    trend_strength_strong:    {best['trend_strength_strong']:.2f}")
            print(f"    trend_strength_sideways:  {best['trend_strength_sideways']:.2f}")
            print(f"    vol_expanding_mult:       {best['vol_expanding_mult']:.2f}")
            print(f"    hurst_accumulation:       {best['hurst_accumulation']:.2f}")
            print(f"    mr_min_score:             {best['mr_min_score']:.1f}")
            print(f"    PF={best['pf']:.2f}, Sharpe={best['sharpe']:.2f}, "
                  f"WR={best['wr']*100:.0f}%, DD={best['dd']:.1f}%")
        else:
            print(f"\n  ✗ No combos met the filter criteria. Relax filters or increase search range.")

        print("  " + "=" * 80)
        return filtered

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
            ("confluence_trending", [2.5, 3.0, 3.5]),
            ("target_atr_mult",     [2.5, 3.0, 3.5]),
            ("time_stop_bars",      [5, 7, 9]),
            ("kelly_wr",            [0.30, 0.35, 0.40]),
            ("breakeven_ratio",     [0.4, 0.6, 0.8]),
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
            "confluence_trending": 3.0,
            "target_atr_mult": 3.0,
            "time_stop_bars": 7,
            "kelly_wr": 0.35,
            "breakeven_ratio": 0.6,
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
    # MODE: INTRADAY DAILY-PROXY WALK-FORWARD
    # ─────────────────────────────────────────────────────────────────────
    def run_intraday_daily_walkforward(
        self,
        symbol: str = "NIFTY 50",
        train_end: str = "2020-12-31",
        test_start: str = "2021-01-01",
        dd_throttle: bool = True,
    ):
        """
        Walk-forward validation using daily bars with intraday signal weights.
        PROXY: Validates whether intraday signal combination (ST+EMA+VWAP)
        predicts direction over 15yr of statistically significant data.
        """
        self.dashboard.show_header()
        print("\n" + "=" * 70)
        print("  INTRADAY DAILY-PROXY WALK-FORWARD VALIDATION")
        print(f"  Symbol: {symbol}")
        print(f"  Train: earliest...{train_end}  |  Test: {test_start}...present")
        print("  NOTE: Daily bars with intraday weights (ST+EMA+VWAP) — PROXY only")
        print("=" * 70)

        intraday_daily_overrides = {
            "use_default_weights": True,
            "weight_overrides": {
                "signal_supertrend": 1.0,
                "signal_ema": 0.75,
                "signal_vwap": 1.0,
            },
            "time_stop_bars": 3,
            "breakeven_ratio": 0.5,
        }

        logger.info(f"Fetching maximum daily data for {symbol}...")
        data_all = self.data.fetch_historical(
            symbol, days=6750, interval="day", force_refresh=True
        )

        if data_all.empty or len(data_all) < 100:
            logger.error(f"Insufficient data for {symbol}: {len(data_all)} bars")
            return None

        train_end_ts = pd.Timestamp(train_end)
        test_start_ts = pd.Timestamp(test_start)

        data_train = data_all[data_all["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
        data_test = data_all[data_all["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

        if len(data_train) < 100:
            logger.error(f"Insufficient training data: {len(data_train)} bars")
            return None
        if len(data_test) < 50:
            logger.error(f"Insufficient test data: {len(data_test)} bars")
            return None

        print(f"\n  Train: {str(data_train['timestamp'].iloc[0])[:10]} to "
              f"{str(data_train['timestamp'].iloc[-1])[:10]} ({len(data_train)} bars)")
        print(f"  Test:  {str(data_test['timestamp'].iloc[0])[:10]} to "
              f"{str(data_test['timestamp'].iloc[-1])[:10]} ({len(data_test)} bars)")

        print("\n" + "=" * 70)
        print("  IN-SAMPLE — Intraday Weights on Daily Bars")
        print("=" * 70)
        result_train, engine_train = self._run_backtest_on_slice(
            data_train, symbol,
            f"IntradayProxy_IS_{symbol.replace(' ', '_')}",
            param_overrides=intraday_daily_overrides,
            parrondo=True,
            dd_throttle=dd_throttle,
        )

        print("\n" + "=" * 70)
        print("  OUT-OF-SAMPLE — Intraday Weights on Daily Bars")
        print("=" * 70)
        result_test, engine_test = self._run_backtest_on_slice(
            data_test, symbol,
            f"IntradayProxy_OOS_{symbol.replace(' ', '_')}",
            param_overrides=intraday_daily_overrides,
            parrondo=True,
            dd_throttle=dd_throttle,
        )

        self._display_walkforward_comparison(
            result_train, result_test, symbol,
            engine_train=engine_train,
            engine_test=engine_test,
        )

        print("\n  " + "-" * 65)
        print("  PROXY CAVEAT")
        print("  " + "-" * 65)
        print("  This is a DAILY-BAR proxy for intraday signal validation.")
        print("  Supertrend, EMA 9/21, and VWAP are timeframe-agnostic indicators.")
        print("  If they predict direction on daily bars over 15yr, the directional")
        print("  signal transfers to intraday — but execution dynamics differ.")
        print("  Actual intraday WR/PF may differ due to: noise, spread, session gaps.")
        print("  " + "-" * 65)

        return result_train, result_test

    # ─────────────────────────────────────────────────────────────────────
    # MODE: INTRADAY DAILY-PROXY SENSITIVITY SWEEP
    # ─────────────────────────────────────────────────────────────────────
    def run_intraday_daily_sensitivity(
        self,
        symbol: str = "NIFTY 50",
        days: int = 5475,
    ):
        """
        Parameter sensitivity sweep using daily bars with intraday signal weights.
        Tests intraday-relevant parameters on 15yr daily data.
        """
        self.dashboard.show_header()
        print("\n" + "=" * 70)
        print("  INTRADAY DAILY-PROXY SENSITIVITY SWEEP")
        print(f"  Symbol: {symbol} | Period: {days} days")
        print("  NOTE: Daily bars with intraday weights — PROXY for intraday params")
        print("=" * 70)

        base_overrides = {
            "use_default_weights": True,
            "weight_overrides": {
                "signal_supertrend": 1.0,
                "signal_ema": 0.75,
                "signal_vwap": 1.0,
            },
            "time_stop_bars": 3,
            "breakeven_ratio": 0.5,
        }

        logger.info(f"Fetching {days} days of daily data for {symbol}...")
        data_all = self.data.fetch_historical(
            symbol, days=days, interval="day", force_refresh=True
        )

        if data_all.empty or len(data_all) < 100:
            logger.error(f"Insufficient data: {len(data_all)} bars")
            return None

        print(f"\n  Data: {len(data_all)} bars "
              f"({str(data_all['timestamp'].iloc[0])[:10]} to "
              f"{str(data_all['timestamp'].iloc[-1])[:10]})")

        param_grid = [
            ("signal_supertrend",   [0.5, 0.75, 1.0, 1.25]),
            ("signal_ema",          [0.5, 0.75, 1.0]),
            ("signal_vwap",         [0.75, 1.0, 1.25]),
            ("confluence_trending", [2.5, 3.0, 3.5]),
            ("time_stop_bars",      [3, 4, 5]),
            ("breakeven_ratio",     [0.4, 0.5, 0.6]),
        ]

        print("\n  --- Baseline (intraday default params) ---")
        baseline, _ = self._run_backtest_on_slice(
            data_all, symbol, "intraday_baseline",
            param_overrides=base_overrides,
            verbose=False,
            parrondo=True,
        )
        wr_b = baseline.win_rate * 100 if baseline.win_rate <= 1 else baseline.win_rate
        print(f"  Baseline: PF={baseline.profit_factor:.2f}, "
              f"Sharpe={baseline.sharpe_ratio:.2f}, "
              f"WR={wr_b:.0f}%, "
              f"DD={baseline.max_drawdown_pct:.1f}%, Trades={baseline.total_trades}")

        results = []
        for param_name, test_values in param_grid:
            for val in test_values:
                sweep_overrides = dict(base_overrides)
                if param_name.startswith("signal_"):
                    sweep_overrides["weight_overrides"] = dict(base_overrides["weight_overrides"])
                    sweep_overrides["weight_overrides"][param_name] = val
                else:
                    sweep_overrides[param_name] = val

                label = f"{param_name}={val}"
                logger.info(f"  Testing: {label}")
                result, _ = self._run_backtest_on_slice(
                    data_all, symbol, label,
                    param_overrides=sweep_overrides,
                    verbose=False,
                    parrondo=True,
                )
                results.append((param_name, val, result))

        self._display_intraday_daily_sensitivity_results(baseline, results, param_grid)

    def _display_intraday_daily_sensitivity_results(self, baseline, results, param_grid):
        """Display intraday daily-proxy sensitivity sweep results."""
        print("\n" + "=" * 70)
        print("  INTRADAY DAILY-PROXY SENSITIVITY RESULTS")
        print("=" * 70)

        defaults = {
            "signal_supertrend": 1.0,
            "signal_ema": 0.75,
            "signal_vwap": 1.0,
            "confluence_trending": 3.0,
            "time_stop_bars": 3,
            "breakeven_ratio": 0.5,
        }

        min_pf = baseline.profit_factor
        best_wr = baseline.win_rate * 100 if baseline.win_rate <= 1 else baseline.win_rate
        best_wr_label = "baseline"
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
                pf_change = ((result.profit_factor - baseline.profit_factor) /
                             baseline.profit_factor * 100
                             if baseline.profit_factor > 0 else 0)
                marker = " <-- default" if is_default else ""
                wr = result.win_rate * 100 if result.win_rate <= 1 else result.win_rate
                print(f"  {val:>8} {result.profit_factor:>8.2f} {result.sharpe_ratio:>8.2f} "
                      f"{wr:>5.0f}% {result.max_drawdown_pct:>7.1f}% {result.total_trades:>7} "
                      f"{pf_change:>+9.1f}%{marker}")

                min_pf = min(min_pf, result.profit_factor)
                if wr > best_wr:
                    best_wr = wr
                    best_wr_label = f"{param_name}={val}"
                if result.profit_factor < 1.2:
                    all_above_1_2 = False
                if result.profit_factor < 1.0:
                    any_below_1_0 = True

        print("\n" + "=" * 70)
        print("  INTRADAY DAILY-PROXY ROBUSTNESS VERDICT")
        print("=" * 70)
        print(f"  Baseline PF: {baseline.profit_factor:.2f}")
        print(f"  Min PF across all variations: {min_pf:.2f}")
        wr_base = baseline.win_rate * 100 if baseline.win_rate <= 1 else baseline.win_rate
        print(f"  Baseline WR: {wr_base:.0f}% | Best WR: {best_wr:.0f}% ({best_wr_label})")

        if all_above_1_2:
            print(f"  Status: ROBUST — PF stays >= 1.2 across all parameter variations")
        elif not any_below_1_0:
            print(f"  Status: MODERATELY ROBUST — PF stays >= 1.0 but dips below 1.2")
        else:
            print(f"  Status: FRAGILE — PF drops below 1.0 on some variations")
        print("  The system is NOT sensitive to parameter tuning." if all_above_1_2 else "")
        print("=" * 70)

        print("\n  PROXY CAVEAT: These results use daily bars as proxy for intraday.")
        print("  Actual intraday execution may differ due to noise and microstructure.")

    # ─────────────────────────────────────────────────────────────────────
    # MODE: ONE-TIME SCAN
    # ─────────────────────────────────────────────────────────────────────
    def run_scan(self):
        """Run a multi-index scan, rank by regime-adjusted confidence."""
        self.dashboard.show_header()
        print(f"\nScanning {len(self.symbols)} symbols at {datetime.now().strftime('%H:%M:%S')}...\n")

        # Regime quality multipliers from backtest WR data:
        # markup 62% WR, markdown 58%, accum/distrib moderate, unknown 26%
        REGIME_MULTIPLIER = {
            "markup":       1.00,
            "markdown":     0.95,
            "accumulation": 0.75,
            "distribution": 0.70,
            "volatile":     0.55,
            "unknown":      0.40,
        }

        # Non-executable symbols (signal-only, can't trade via Kite)
        SIGNAL_ONLY = {"SENSEX", "NIFTY MIDCAP SELECT", "NIFTY NEXT 50"}

        scan_results = []

        for symbol in self.symbols:
            print(f"  Analyzing {symbol}...", end="", flush=True)
            signal = self.analyze(symbol)

            if not signal:
                print(" no data")
                continue

            # Get regime for this symbol
            data = self.data.fetch_historical(symbol, days=60, interval="day")
            regime_str = "unknown"
            regime_conf = 0.0
            if not data.empty:
                regime = self.regime_detector.detect(data)
                regime_str = regime.regime.value
                regime_conf = regime.confidence

            # Count contributing signals
            sig_count = len(signal.contributing_signals) if signal.contributing_signals else 0

            # Raw confluence from fusion engine
            raw_confidence = signal.confidence

            # Regime-adjusted confidence
            regime_mult = REGIME_MULTIPLIER.get(regime_str, 0.40)
            adj_confidence = raw_confidence * regime_mult

            executable = symbol not in SIGNAL_ONLY

            scan_results.append({
                "symbol": symbol,
                "action": signal.action,
                "direction": signal.direction,
                "raw_confidence": raw_confidence,
                "adj_confidence": adj_confidence,
                "regime": regime_str,
                "regime_confidence": regime_conf,
                "signal_count": sig_count,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "target": signal.target,
                "risk_reward": signal.risk_reward,
                "reasoning": signal.reasoning,
                "executable": executable,
            })

            print(f" {signal.action} ({adj_confidence:.0%})")

        # Display ranked table (CLI)
        self.dashboard.show_scanner_table(scan_results)

        # Send scanner summary to Telegram (replaces individual signal alerts)
        self.telegram.alert_scanner_summary(scan_results)

        # Show detailed cards for top actionable signals
        actionable = [r for r in scan_results
                      if r["action"] != "HOLD" and r["adj_confidence"] >= 0.50]
        actionable.sort(key=lambda x: x["adj_confidence"], reverse=True)

        if actionable:
            print(f"\n{'─' * 60}")
            print(f"  TOP SIGNALS (Adj Confidence >= 50%)")
            print(f"{'─' * 60}")
            for r in actionable[:5]:
                self.dashboard.show_signal({
                    "action": r["action"],
                    "symbol": r["symbol"],
                    "confidence": r["adj_confidence"],
                    "entry_price": r["entry_price"],
                    "stop_loss": r["stop_loss"],
                    "target": r["target"],
                    "risk_reward": r["risk_reward"],
                    "regime": r["regime"],
                    "reasoning": r["reasoning"],
                })
                # Warning for weak regimes
                regime = r["regime"]
                if regime == "unknown":
                    print("    [!!] UNKNOWN regime — 26% historical WR. Consider skipping.")
                elif regime == "volatile":
                    print("    [!] VOLATILE regime — lower conviction. Tighter sizing recommended.")
                elif not r["executable"]:
                    print(f"    [i] {r['symbol']} — signal only (not executable via Kite)")
                print()

        # VIX
        vix = self.data.get_vix()
        if vix:
            print(f"\nIndia VIX: {vix:.2f}")
            if vix > 20:
                print("  HIGH fear — consider hedging or reducing position sizes")
            elif vix < 12:
                print("  LOW vol — options are cheap, good for buying")

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
        # Compute winning trades from PaperTrader history
        winning = 0
        if isinstance(self.broker, PaperTrader):
            winning = sum(1 for t in self.broker.trade_history if t.get("net_pnl", 0) > 0)
        self.telegram.alert_daily_summary({
            "daily_pnl": state.realized_pnl_today,
            "total_trades": state.trades_today,
            "winning_trades": winning,
            "equity": state.capital,
        })

        # Multi-account summary
        if self.multi_account is not None:
            self.multi_account.record_all_equity()
            summaries = self.multi_account.get_summary_table()
            self.dashboard.show_multi_account_summary(summaries)
            # Also send to Telegram
            lines = ["\U0001f4ca <b>MULTI-ACCOUNT DAILY SUMMARY</b>"]
            for s in summaries:
                emoji = "\u2705" if s["pnl"] >= 0 else "\u274c"
                lines.append(
                    f"{emoji} <b>{s['label']}</b>: Rs {s['equity']:,.0f} "
                    f"({s['return_pct']:+.1f}%) | {s['trades']} trades | WR {s['win_rate']:.0f}%"
                )
            self.telegram.send_message("\n".join(lines))

        # Persist equity state at end of day (crash recovery)
        self._persist_equity_state()

    # ─────────────────────────────────────────────────────────────────────
    # TELEGRAM COMMAND HANDLERS
    # ─────────────────────────────────────────────────────────────────────
    def _setup_telegram_commands(self):
        """Register Telegram bot command handlers and start listening."""
        tg = self.telegram

        tg.register_command("help", self._tg_cmd_help)
        tg.register_command("scan", self._tg_cmd_scan)
        tg.register_command("status", self._tg_cmd_status)
        tg.register_command("pnl", self._tg_cmd_pnl)
        tg.register_command("regime", self._tg_cmd_regime)
        tg.register_command("start", self._tg_cmd_help)  # Telegram auto-sends /start
        tg.register_command("confirm", lambda a: tg.handle_confirm())
        tg.register_command("reject", lambda a: tg.handle_reject())
        tg.register_command("positions", self._tg_cmd_positions)
        tg.register_command("set_price", self._tg_cmd_set_price)

        tg.start_listening()

    def _tg_cmd_help(self, args: str = "") -> str:
        """Handle /help command."""
        ma_info = ""
        if self.multi_account is not None:
            labels = [s.config.label for s in self.multi_account.stacks.values()]
            ma_info = f"\U0001f4b3 Accounts: {', '.join(labels)}\n\n"
        return (
            f"\U0001f4d6 <b>PROMETHEUS</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"{ma_info}"
            f"/scan          Swing + Intraday scan\n"
            f"/status        System status\n"
            f"/pnl            Today's P&amp;L\n"
            f"/positions   Open positions\n"
            f"/regime       Market regime\n"
            f"/help           This message\n\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"<i>Signals sent automatically during market hours</i>"
        )

    def _tg_cmd_scan(self, args: str = "") -> str:
        """Handle /scan command — run multi-index scanner with swing + intraday."""
        self.telegram.send_message("\U0001f50e Scanning all indices (swing + intraday)... please wait.")

        # Regime multipliers (same as run_scan)
        REGIME_MULTIPLIER = {
            "markup": 1.00, "markdown": 0.95, "accumulation": 0.75,
            "distribution": 0.70, "volatile": 0.55, "unknown": 0.40,
        }
        SIGNAL_ONLY = {"SENSEX", "NIFTY MIDCAP SELECT", "NIFTY NEXT 50"}

        scan_results = []
        for symbol in self.symbols:
            try:
                signal = self.analyze(symbol)
                if not signal:
                    continue

                data = self.data.fetch_historical(symbol, days=90, interval="day")
                regime_str = "unknown"
                if not data.empty:
                    regime = self.regime_detector.detect(data)
                    regime_str = regime.regime.value

                raw_confidence = signal.confidence
                regime_mult = REGIME_MULTIPLIER.get(regime_str, 0.40)
                adj_confidence = raw_confidence * regime_mult

                sig_count = len(signal.contributing_signals) if signal.contributing_signals else 0

                scan_results.append({
                    "symbol": symbol,
                    "action": signal.action,
                    "raw_confidence": raw_confidence,
                    "adj_confidence": adj_confidence,
                    "regime": regime_str,
                    "signal_count": sig_count,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "target": signal.target,
                    "risk_reward": signal.risk_reward,
                    "reasoning": signal.reasoning,
                    "executable": symbol not in SIGNAL_ONLY,
                    "timeframe": "swing",
                })
            except Exception as e:
                logger.debug(f"Scan failed for {symbol}: {e}")

        # Intraday scan (during market hours only)
        from datetime import time as dtime
        now = datetime.now()
        intraday_cfg = self.config.get("intraday", {})
        intraday_instruments = intraday_cfg.get("instruments", self.symbols[:2])
        mkt_open = dtime(9, 15)
        mkt_close = dtime(15, 15)

        if mkt_open <= now.time() <= mkt_close and is_trading_day(now.date()):
            bar_interval = self._select_intraday_interval()
            for symbol in intraday_instruments:
                try:
                    signal = self.analyze_intraday(symbol, bar_interval)
                    if not signal:
                        continue

                    data = self.data.fetch_historical(symbol, days=5, interval=bar_interval)
                    regime_str = "unknown"
                    if not data.empty:
                        data_daily = self.data.fetch_historical(symbol, days=90, interval="day")
                        if not data_daily.empty:
                            regime = self.regime_detector.detect(data_daily)
                            regime_str = regime.regime.value

                    raw_confidence = signal.confidence
                    regime_mult = REGIME_MULTIPLIER.get(regime_str, 0.40)
                    adj_confidence = raw_confidence * regime_mult

                    sig_count = len(signal.contributing_signals) if signal.contributing_signals else 0

                    scan_results.append({
                        "symbol": f"{symbol} (intraday {bar_interval})",
                        "action": signal.action,
                        "raw_confidence": raw_confidence,
                        "adj_confidence": adj_confidence,
                        "regime": regime_str,
                        "signal_count": sig_count,
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "target": signal.target,
                        "risk_reward": getattr(signal, "risk_reward", 0),
                        "reasoning": getattr(signal, "reasoning", ""),
                        "executable": symbol not in SIGNAL_ONLY,
                        "timeframe": "intraday",
                    })
                except Exception as e:
                    logger.debug(f"Intraday scan failed for {symbol}: {e}")

        # Send formatted summary (handles empty case too)
        self.telegram.alert_scanner_summary(scan_results)
        return None  # already sent via alert_scanner_summary

    def _tg_cmd_status(self, args: str = "") -> str:
        """Handle /status command — shows all accounts if multi-account is active."""
        mode_str = self.mode.upper()

        # Multi-account summary
        if self.multi_account is not None:
            text = (
                f"\U0001f4ca <b>STATUS  \u2014  MULTI-ACCOUNT</b>\n"
                f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
                f"Mode: <b>{mode_str}</b>\n\n"
            )

            for label, stack in self.multi_account.stacks.items():
                s = stack.get_summary()
                halted = "\U0001f6d1" if stack.risk._halted else "\u2705"
                pnl_emoji = "\U0001f7e2" if s["pnl"] >= 0 else "\U0001f534"
                text += (
                    f"{halted} <b>{s['label']}</b>  (Rs {s['initial']:,.0f})\n"
                    f"    Equity <code>Rs {s['equity']:,.0f}</code>\n"
                    f"    {pnl_emoji} P&amp;L <code>Rs {s['pnl']:+,.0f}</code>  ({s['return_pct']:+.1f}%)\n"
                    f"    Trades <code>{s['trades']}</code>  \u2502  Open <code>{s['open_positions']}</code>  \u2502  WR <code>{s['win_rate']:.0f}%</code>\n\n"
                )
            return text

        # Single account fallback
        state = self.risk.get_portfolio_state()
        positions = self.broker.get_positions() if hasattr(self.broker, "get_positions") else []
        n_pos = len(positions)
        halted = "\U0001f6d1 HALTED" if self.risk._halted else "\u2705 ACTIVE"

        text = (
            f"\U0001f4ca <b>STATUS</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"Mode:       <b>{mode_str}</b>  {halted}\n"
            f"Capital:    <code>Rs {state.capital:,.0f}</code>\n"
            f"Positions:  <code>{n_pos}</code>\n"
        )

        if positions:
            text += f"\n<b>Open Positions</b>\n"
            for p in positions:
                pnl = getattr(p, "unrealized_pnl", 0)
                pnl_emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
                text += (
                    f"  {pnl_emoji} {p.tradingsymbol}  "
                    f"<code>{p.quantity}x</code>  "
                    f"<code>Rs {pnl:+,.0f}</code>\n"
                )

        return text

    def _tg_cmd_pnl(self, args: str = "") -> str:
        """Handle /pnl command — shows all accounts if multi-account is active."""
        # Multi-account P&L
        if self.multi_account is not None:
            text = (
                f"\U0001f4b0 <b>TODAY'S P&amp;L</b>\n"
                f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            )
            total_pnl = 0
            for label, stack in self.multi_account.stacks.items():
                s = stack.get_summary()
                risk_state = stack.risk.get_portfolio_state()
                pnl = risk_state.realized_pnl_today
                total_pnl += pnl
                emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
                text += (
                    f"{emoji} <b>{s['label']}</b>\n"
                    f"    Realized  <code>Rs {pnl:+,.0f}</code>\n"
                    f"    Trades <code>{risk_state.trades_today}</code>  \u2502  "
                    f"Equity <code>Rs {s['equity']:,.0f}</code>\n\n"
                )
            summary_emoji = "\U0001f7e2" if total_pnl >= 0 else "\U0001f534"
            text += (
                f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                f"{summary_emoji} <b>TOTAL:  <code>Rs {total_pnl:+,.0f}</code></b>"
            )
            return text

        # Single account fallback
        state = self.risk.get_portfolio_state()
        pnl = state.realized_pnl_today
        emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
        used_pct = abs(pnl) / self.risk.max_daily_loss * 100 if self.risk.max_daily_loss > 0 else 0

        return (
            f"\U0001f4b0 <b>TODAY'S P&amp;L</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"{emoji} Realized:  <code>Rs {pnl:+,.0f}</code>\n"
            f"Trades:     <code>{state.trades_today}</code>\n"
            f"Capital:    <code>Rs {state.capital:,.0f}</code>\n\n"
            f"Daily limit: <code>Rs {self.risk.max_daily_loss:,.0f}</code>\n"
            f"Used:        <code>{used_pct:.0f}%</code>"
        )

    def _tg_cmd_regime(self, args: str = "") -> str:
        """Handle /regime command — show current regime for all indices."""
        lines = [
            "\U0001f30d <b>MARKET REGIME</b>",
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            "",
        ]

        for symbol in self.symbols:
            try:
                data = self.data.fetch_historical(symbol, days=90, interval="day")
                if data.empty:
                    lines.append(f"\u26aa {symbol}:  <i>no data</i>")
                    continue

                regime = self.regime_detector.detect(data)
                r_val = regime.regime.value
                r_conf = regime.confidence
                trend = regime.trend_strength

                from prometheus.interface.telegram_bot import REGIME_QUALITY
                quality, wr = REGIME_QUALITY.get(r_val, ("???", ""))

                if trend > 0.2:
                    arrow = "\u2b06\ufe0f"
                elif trend < -0.2:
                    arrow = "\u2b07\ufe0f"
                else:
                    arrow = "\u27a1\ufe0f"

                lines.append(
                    f"{arrow} <b>{symbol}</b>\n"
                    f"    {r_val.upper()} <code>{r_conf:.0%}</code>  \u2502  {quality} ({wr})\n"
                    f"    Trend: <code>{trend:+.2f}</code>"
                )
            except Exception as e:
                lines.append(f"\u26aa {symbol}:  <i>error</i>")

        return "\n".join(lines)

    def _tg_cmd_positions(self, args: str = "") -> str:
        """Handle /positions — show open positions with trailing stop state."""
        if not self.position_monitor or self.position_monitor.active_count == 0:
            return (
                "\U0001f4ca <b>POSITIONS</b>\n"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
                "No open positions."
            )

        lines = [
            "\U0001f4ca <b>OPEN POSITIONS</b>",
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            "",
        ]
        for pid, state in self.position_monitor.get_positions().items():
            pnl_pct = ((self.broker.get_ltp(state.tradingsymbol) - state.entry_premium)
                       / state.entry_premium * 100) if state.entry_premium > 0 else 0
            pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
            acct_label = getattr(state, "_multi_account_label", "primary")
            lines.append(
                f"{pnl_emoji} <b>{state.tradingsymbol}</b>\n"
                f"    <i>{acct_label}</i>\n"
                f"    Entry <code>{state.entry_premium:.2f}</code>  \u2502  SL <code>{state.current_sl:.2f}</code>\n"
                f"    Stage <code>{state.current_stage()}</code>  \u2502  "
                f"Bars <code>{state.entry_bar_count}/{state.max_bars}</code>\n"
                f"    P&amp;L <code>{pnl_pct:+.1f}%</code>\n"
            )
        return "\n".join(lines)

    def _tg_cmd_set_price(self, args: str = "") -> str:
        """Handle /set_price — push synthetic LTP for dry-run testing.

        Usage: /set_price NIFTY25D2622500CE 250
        """
        if not isinstance(self.broker, PaperTrader):
            return "set_price only works in dry_run/paper mode."
        parts = args.strip().split()
        if len(parts) < 2:
            return "Usage: /set_price <tradingsymbol> <price>"
        tsym = parts[0]
        try:
            price = float(parts[1])
        except ValueError:
            return f"Invalid price: {parts[1]}"
        self.broker.update_prices({tsym: price})
        return f"Price set: {tsym} = {price:.2f}"

    def stop(self):
        """Stop the system gracefully."""
        self.running = False
        self.telegram.stop_listening()
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
            "  python main.py paper --intraday        Intraday paper trading\n"
            "  python main.py dry_run --intraday      Intraday dry run\n"
            "  python main.py signal                  Signal-only mode\n"
            "  python main.py setup                   First-time setup wizard\n"
        ),
    )

    parser.add_argument(
        "mode",
        choices=["scan", "backtest", "paper", "signal", "setup", "walkforward", "sensitivity", "parrondo_tuning", "semi_auto", "full_auto", "dry_run", "collect"],
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
        choices=["trend", "volatility"],
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
    parser.add_argument(
        "--entry-timing",
        action="store_true",
        default=False,
        help="Enable next-bar limit order entry timing: waits for pullback before entering",
    )
    parser.add_argument(
        "--entry-pullback-atr",
        type=float,
        default=0.3,
        help="Fraction of ATR for pullback limit order (default: 0.3)",
    )
    parser.add_argument(
        "--entry-max-wait",
        type=int,
        default=2,
        help="Max bars to wait for pullback fill (default: 2)",
    )
    # ── Institutional risk overlays ──
    parser.add_argument(
        "--vol-target",
        type=float,
        default=0.0,
        help="Volatility targeting: annualized vol target (e.g., 0.15 for 15%%). 0=disabled (default: 0)",
    )
    parser.add_argument(
        "--dd-throttle",
        action="store_true",
        default=True,
        help="Drawdown throttle: skip trades when DD>20%% (1-lot accounts), scale down larger accounts",
    )
    parser.add_argument(
        "--equity-filter",
        action="store_true",
        default=False,
        help="Equity curve filter: skip trades when equity < 50-bar SMA",
    )
    parser.add_argument(
        "--risk-overlays",
        action="store_true",
        default=False,
        help="Enable ALL institutional risk overlays (vol-target=0.15, dd-throttle, equity-filter)",
    )
    parser.add_argument(
        "--high-quality",
        action="store_true",
        default=False,
        help="Raise signal quality thresholds: confluence 3.0/4.0, net edge 2.0, Kelly WR 0.35",
    )
    parser.add_argument(
        "--intraday",
        action="store_true",
        default=False,
        help="Enable intraday mode: continuous scanning, auto bar interval, 3:15 PM square-off",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        default=False,
        help="Run both swing and intraday simultaneously in one process",
    )
    parser.add_argument(
        "--intraday-daily",
        action="store_true",
        default=False,
        help="Use daily bars as proxy for intraday signal validation (15yr data with intraday weights)",
    )
    parser.add_argument(
        "--multi-account",
        action="store_true",
        default=False,
        help="Run paper trading with multiple simulated capital levels simultaneously",
    )
    parser.add_argument(
        "--collect-interval",
        default="5minute",
        choices=["5minute", "15minute", "60minute", "day"],
        help="Candle interval for data collection (default: 5minute)",
    )

    args = parser.parse_args()

    # ── Resolve --risk-overlays convenience flag ──
    if args.risk_overlays:
        if args.vol_target == 0.0:
            args.vol_target = 0.15
        # Note: equity filter NOT enabled by default — too aggressive for small capital

    # DD throttle always on — structural protection for small accounts
    args.dd_throttle = True

    # Initialize system
    prometheus = Prometheus(config_path=args.config)

    # Override mode from CLI (settings.yaml is default, CLI takes precedence)
    if args.mode in ("semi_auto", "full_auto", "dry_run"):
        cli_mode = args.mode
        if cli_mode == "dry_run":
            # Force PaperTrader for dry run
            if not isinstance(prometheus.broker, PaperTrader):
                prometheus.broker = PaperTrader(prometheus.initial_capital)
                prometheus.order_manager = OrderManager(prometheus.broker, prometheus.risk, "dry_run")
        elif cli_mode in ("semi_auto", "full_auto"):
            # If settings.yaml says paper but CLI says live mode, switch broker
            if isinstance(prometheus.broker, PaperTrader):
                try:
                    from prometheus.execution.kite_executor import KiteExecutor
                    kite = KiteExecutor(
                        api_key=get_credential("broker.api_key") or get("broker.api_key", ""),
                        api_secret=get_credential("broker.api_secret") or get("broker.api_secret", ""),
                        access_token=get_credential("broker.access_token") or get("broker.access_token", ""),
                    )
                    if kite.connect():
                        prometheus.broker = kite
                        prometheus.order_manager = OrderManager(prometheus.broker, prometheus.risk, cli_mode)
                        logger.info(f"Switched to KiteExecutor for {cli_mode} mode")
                    else:
                        logger.warning("Kite connect failed. Running in dry_run mode (PaperTrader).")
                        args.mode = "dry_run"
                except Exception as e:
                    logger.warning(f"KiteExecutor unavailable ({e}). Running in dry_run mode.")
                    args.mode = "dry_run"
        prometheus.mode = args.mode

    # Override symbols if specified
    if args.symbol:
        prometheus.symbols = [args.symbol]

    # Handle graceful shutdown
    # SIGINT: Python's default handler raises KeyboardInterrupt (caught by loop except blocks)
    # SIGTERM: Convert to KeyboardInterrupt for same cleanup path
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    try:
        sig.signal(sig.SIGTERM, _sigterm_handler)
    except (OSError, ValueError):
        pass  # SIGTERM not available on Windows in some contexts

    # Dispatch to mode
    if args.mode == "setup":
        prometheus.run_setup()

    elif args.mode == "scan":
        prometheus.run_scan()

    elif args.mode == "backtest":
        if args.intraday:
            # ── INTRADAY BACKTEST (separate from swing) ──
            prometheus.run_intraday_backtest(
                symbol=args.symbol or "NIFTY 50",
                days=min(args.days, 59),
                parrondo=args.parrondo,
                dd_throttle=args.dd_throttle,
            )
        else:
            # ── SWING BACKTEST (LOCKED — do not modify) ──
            hq_overrides = {}
            if args.high_quality:
                hq_overrides = {
                    "confluence_trending": 3.0,
                    "confluence_sideways": 4.5,
                    "kelly_wr": 0.35,
                }

            prometheus.run_backtest(
                symbol=args.symbol or "NIFTY 50",
                days=args.days,
                strategy=args.strategy,
                parrondo=args.parrondo,
                entry_timing=args.entry_timing,
                entry_pullback_atr=args.entry_pullback_atr,
                entry_max_wait_bars=args.entry_max_wait,
                vol_target=args.vol_target,
                dd_throttle=args.dd_throttle,
                equity_curve_filter=args.equity_filter,
                param_overrides=hq_overrides if hq_overrides else None,
            )

    elif args.mode == "signal":
        prometheus.run_signal_mode(interval_seconds=args.interval)

    elif args.mode == "paper":
        if getattr(args, "multi_account", False):
            prometheus.init_multi_account()
        if args.combined:
            prometheus.run_combined_mode(interval_seconds=args.interval)
        elif args.intraday:
            prometheus.run_intraday_mode(interval_seconds=args.interval)
        else:
            prometheus.run_paper_mode(interval_seconds=args.interval)

    elif args.mode == "semi_auto":
        if getattr(args, "multi_account", False):
            prometheus.init_multi_account()
        prometheus.run_semi_auto_mode(interval_seconds=args.interval)

    elif args.mode == "full_auto":
        if getattr(args, "multi_account", False):
            prometheus.init_multi_account()
        if args.combined:
            prometheus.run_combined_mode(interval_seconds=args.interval)
        elif args.intraday:
            prometheus.run_intraday_mode(interval_seconds=args.interval)
        else:
            prometheus.run_full_auto_mode(interval_seconds=args.interval)

    elif args.mode == "dry_run":
        # Same as full_auto but forces PaperTrader (already done above)
        if getattr(args, "multi_account", False):
            prometheus.init_multi_account()
        if args.combined:
            prometheus.run_combined_mode(interval_seconds=args.interval)
        elif args.intraday:
            prometheus.run_intraday_mode(interval_seconds=args.interval)
        else:
            prometheus.run_full_auto_mode(interval_seconds=args.interval)

    elif args.mode == "walkforward":
        if args.intraday_daily:
            # ── DAILY-PROXY INTRADAY WALK-FORWARD (15yr validation) ──
            prometheus.run_intraday_daily_walkforward(
                symbol=args.symbol or "NIFTY 50",
                train_end=args.train_end,
                test_start=args.test_start,
                dd_throttle=args.dd_throttle,
            )
        elif args.intraday:
            # ── INTRADAY WALK-FORWARD (separate from swing) ──
            prometheus.run_intraday_walkforward(
                symbol=args.symbol or "NIFTY 50",
                parrondo=args.parrondo,
                dd_throttle=args.dd_throttle,
            )
        else:
            # ── SWING WALK-FORWARD (LOCKED — do not modify) ──
            prometheus.run_walkforward(
                symbol=args.symbol or "NIFTY 50",
                train_end=args.train_end,
                test_start=args.test_start,
                strategy=args.strategy,
                parrondo=args.parrondo,
                entry_timing=args.entry_timing,
                entry_pullback_atr=args.entry_pullback_atr,
                entry_max_wait_bars=args.entry_max_wait,
                vol_target=args.vol_target,
                dd_throttle=args.dd_throttle,
                equity_curve_filter=args.equity_filter,
            )

    elif args.mode == "sensitivity":
        if args.intraday_daily:
            # ── DAILY-PROXY INTRADAY SENSITIVITY (15yr sweep) ──
            prometheus.run_intraday_daily_sensitivity(
                symbol=args.symbol or "NIFTY 50",
                days=args.days,
            )
        else:
            prometheus.run_sensitivity(
                symbol=args.symbol or "NIFTY 50",
                days=args.days,
                strategy=args.strategy,
            )

    elif args.mode == "parrondo_tuning":
        prometheus.run_parrondo_tuning(
            symbol=args.symbol or "BANKNIFTY",
            days=args.days,
        )

    elif args.mode == "collect":
        prometheus.run_data_collection(
            symbol=args.symbol,
            interval=args.collect_interval,
            days=args.days,
        )


if __name__ == "__main__":
    main()
