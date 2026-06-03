**Project Status — PROMETHEUS (snapshot: 2026-06-02)**

**Overview**
- **Purpose:** Indian F&O options-buying trading system with paper/live parity, backtest/walk-forward validation, and strict risk guardrails.
- **Languages / libs:** Python; pandas, numpy, yfinance, kiteconnect (optional), sqlite, requests.

**Architecture (high level)**
- **Data Layer:** fetch/historic store (multi-source: Kite, AngelOne, yfinance, CSV override). See [prometheus/data/engine.py](prometheus/data/engine.py#L1) and [prometheus/data/store.py](prometheus/data/store.py#L1).
- **Signals:** Technical + OI + Regime detectors fused into actionable signals. Key files: [prometheus/signals/technical.py](prometheus/signals/technical.py#L1), [prometheus/signals/oi_analyzer.py](prometheus/signals/oi_analyzer.py#L1), [prometheus/signals/regime_detector.py](prometheus/signals/regime_detector.py#L1), [prometheus/signals/fusion.py](prometheus/signals/fusion.py#L1).
- **Strategies:** Regime-aware strategy modules (trend, expiry, selector). Key files: [prometheus/strategies/trend.py](prometheus/strategies/trend.py#L1), [prometheus/strategies/expiry.py](prometheus/strategies/expiry.py#L1), [prometheus/strategies/selector.py](prometheus/strategies/selector.py#L1).
- **Risk & Sizing:** Non-bypassable hard limits, bracket-based sizing, dynamic stop logic. See [prometheus/risk/manager.py](prometheus/risk/manager.py#L1).
- **Execution:** Order manager orchestrates strategy → broker, with paper simulator and Kite executor implementations. Files: [prometheus/execution/order_manager.py](prometheus/execution/order_manager.py#L1), [prometheus/execution/paper_trader.py](prometheus/execution/paper_trader.py#L1), [prometheus/execution/kite_executor.py](prometheus/execution/kite_executor.py#L1), [prometheus/execution/position_monitor.py](prometheus/execution/position_monitor.py#L1).
- **Interface:** Telegram two-way bot for alerts and semi-auto confirmations: [prometheus/interface/telegram_bot.py](prometheus/interface/telegram_bot.py#L1).
- **Backtest / Engine:** Walk-forward, Monte Carlo, PBO logic lives under [prometheus/backtest/engine.py](prometheus/backtest/engine.py#L1) (not fully enumerated here).

**Core file summaries (concise)**
- **prometheus/main.py**: CLI orchestrator — builds signal generator, routes to backtest/paper/live, wires all sub-systems (read earlier). [prometheus/main.py](prometheus/main.py#L1)
- **prometheus/data/engine.py**: Multi-provider fetcher, IST normalisation, CSV override for heavy intraday. [prometheus/data/engine.py](prometheus/data/engine.py#L1)
- **prometheus/data/store.py**: SQLite persistence (ohlcv, options_chain, trades, managed_positions). [prometheus/data/store.py](prometheus/data/store.py#L1)
- **prometheus/signals/technical.py**: VWAP, volume profile, FVG, liquidity sweeps, Supertrend, RSI divergence, ATR, EMA. [prometheus/signals/technical.py](prometheus/signals/technical.py#L1)
- **prometheus/signals/oi_analyzer.py**: PCR, Max Pain, OI support/resistance, OI velocity, IV skew, sentiment aggregation. [prometheus/signals/oi_analyzer.py](prometheus/signals/oi_analyzer.py#L1)
- **prometheus/signals/regime_detector.py**: AMD/Parrondo-style regime classification (volatility, ADX-like trend, Hurst), `detect_fast()` caching for per-bar calls. [prometheus/signals/regime_detector.py](prometheus/signals/regime_detector.py#L1)
- **prometheus/signals/fusion.py**: Weighted confluence engine producing `FusedSignal` (0–10 score), R:R gating, entry/SL/target compute. [prometheus/signals/fusion.py](prometheus/signals/fusion.py#L1)
- **prometheus/strategies/trend.py**: Multi-timeframe trend strategy; strict R:R, ATR & premium-aware sizing, 1-strike OTM for low capital. [prometheus/strategies/trend.py](prometheus/strategies/trend.py#L1)
- **prometheus/strategies/expiry.py**: Expiry-day logic (debit spreads, momentum breakout, scalp). Prefers debit spreads over naked when capital constrained. [prometheus/strategies/expiry.py](prometheus/strategies/expiry.py#L1)
- **prometheus/strategies/selector.py**: Regime → strategy mapping with event/VIX overrides and capital suitability checks. [prometheus/strategies/selector.py](prometheus/strategies/selector.py#L1)
- **prometheus/risk/manager.py**: Hard risk limits (daily/weekly loss, max positions, drawdown halt), risk-based position sizing, dynamic ATR stop calculation, record/restore hooks. [prometheus/risk/manager.py](prometheus/risk/manager.py#L1)
- **prometheus/execution/order_manager.py**: Full trade lifecycle, pre-trade risk gating, builds multi-leg orders, integrates with broker API, records to DataStore. [prometheus/execution/order_manager.py](prometheus/execution/order_manager.py#L1)
- **prometheus/execution/paper_trader.py**: Realistic paper fills, Zerodha cost model, real-premium feed support (Angel One), SL/trigger simulation. [prometheus/execution/paper_trader.py](prometheus/execution/paper_trader.py#L1)
- **prometheus/execution/kite_executor.py**: Zerodha/Kite Connect bindings, order placement, LTP/margins fetch, tradingsymbol generator. [prometheus/execution/kite_executor.py](prometheus/execution/kite_executor.py#L1)
- **prometheus/execution/position_monitor.py**: Background daemon implementing 5-stage trailing stops and time-stops, with SL modifications to broker and persistence hooks. [prometheus/execution/position_monitor.py](prometheus/execution/position_monitor.py#L1)
- **prometheus/execution/lap_recovery.py**: Loss Adjustment Protocol (blocks revenge trades, cooldowns, re-entry rules). [prometheus/execution/lap_recovery.py](prometheus/execution/lap_recovery.py#L1)
- **prometheus/interface/telegram_bot.py**: Robust two-way Telegram integration (proxy / SNI fallbacks), semi-auto confirmation flow, formatted alerts and scanner summaries. [prometheus/interface/telegram_bot.py](prometheus/interface/telegram_bot.py#L1)

**Operational/Design Notes (observations)**
- Swing-15m path is the locked execution path — do not mix with separate intraday executor unless intentionally enabled.
- RiskManager enforces hard stops BEFORE any order; orders are rejected when limits are hit (non-bypassable behavior confirmed).
- PaperTrader supports feeding real bid/ask for realistic fills and uses a Zerodha-like cost model — good parity for backtests.
- PositionMonitor implements the 5-stage trailing (breakeven → 20% → 50% → 70% → dynamic HWM trail) consistent with CLAUDE.md.
- OI analysis and RegimeDetector enable Parrondo-style routing in signal factory.

**Gaps / Next verification steps**
- Confirm `prometheus/main.py` wiring for live vs paper start flags and preflight; perform a smoke run in a safe environment if desired.
- Validate Black-Scholes/greeks correctness across `prometheus/utils/options_math.py` (spot-check gamma/delta sign handling referenced in change-log).
- Run unit/integration tests (prometheus/tests) — existing repo notes show prior tests passing after fixes; I can run `pytest` if you want.

**Next actions (I will proceed unless you instruct otherwise)**
- Finish reading any remaining core modules not covered (additional utils, backtest engine internals). 
- Produce a final detailed `Project_Status.md` (this file) — done. If you want a more detailed per-function inventory or include more files, I can expand.

---
Status: Partial analysis completed; I read the main data, signals, strategies, risk and execution layers and created this summary. Ready to run tests or expand file-level detail on request.
