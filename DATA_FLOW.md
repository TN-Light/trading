**PROMETHEUS — Architecture & Data Flow

This document maps the core components, data paths, and responsibilities inside the repository. Use it as a quick reference to understand where data comes from, how signals are computed, and how trades flow from generation to execution and reporting.

**High-Level Components**
- **Data Sources:** live broker (Kite), AngelOne fetcher, yfinance, and CSV dataset files. (See `prometheus/data/engine.py`.)
- **Data Engine:** `prometheus/data/engine.py` — orchestrates fetching, caching, IST normalization, OHLCV/option chain assembly, and persistence to `prometheus/data/store.py` (SQLite).
- **Storage / Cache:** local SQLite (`data/prometheus.db`) plus `data/cache` and CSV fallbacks.
- **Signals:** technical indicators, OI analysis, FVG, liquidity sweeps, RSI divergence computed in `prometheus/signals/*`. Signals run on configured timeframes (primary/secondary/execution) from `prometheus/config/settings.yaml`.
- **Fusion:** `prometheus/signals/fusion.py` — weight-based fusion engine. Outputs `FusedSignal` containing direction, scores, entry features (atr, delta, premium estimate), and meta flags.
- **Regime Detection & Selector:** `prometheus/signals/regime_detector.py` + `prometheus/strategies/selector.py` route signals into `trend`, `volatility`, or `expiry` strategies depending on regime and config.
- **Strategies:** `prometheus/strategies/*` produce concrete `TradeSetup` dicts with: symbol, instrument_type (options/futures), entry_price (premium or underlying), stop_loss, target, quantity, bar_interval, option_expiry_date, delta, breakeven_ratio, max_bars.
- **Backtest Engine:** `prometheus/backtest/engine.py` — simulates lifecycle: checkpointing, pending-signal entry timing, open/close position logic, PnL accounting, cost model, DSQ, volatility-targeting, drawdown throttle, and metrics.
- **Risk Manager:** `prometheus/risk/manager.py` — enforces pre-trade checks, sizing, hard limits, circuit-breakers, and paper/live overrides (driven by `config/settings.yaml`).
- **Order Manager & Executors:** `prometheus/execution/order_manager.py`, `kite_executor.py`, `paper_trader.py` — handle order build, submission, fills, slippage and broker abstraction.
- **Position Monitor:** `prometheus/execution/position_monitor.py` — implements the 5-stage trailing (breakeven trap → lock 20% → lock 50% → lock 70% → dynamic trail) and time-stop enforcement.
- **Interface & Reporting:** CLI (`prometheus/interface/cli_dashboard.py`) and Telegram (`prometheus/interface/telegram_bot.py`) plus logging channels in `prometheus/utils/logger.py` and artifacts under `reports/`.

**Primary Data Flow (per-bar):**
1. DataEngine appends new bar(s) → normalized OHLCV and options chain.
2. Signal layer consumes historical window → emits indicator flags and scores.
3. Fusion engine merges signals → produces `FusedSignal` if confluence >= `signals.min_confluence`.
4. Regime Detector + StrategySelector chooses strategy → strategy builds `TradeSetup` with entry features (premium estimate, delta, ATR, expiry).
5. Risk overlays (in BacktestEngine and RiskManager) apply: equity MA sizing, DSQ filter, volatility targeting, DD throttle, pilot guardrails (intraday), and capital bracket rules from `config/settings.yaml`.
6. If allowed, the signal enters pending state (entry-timing) or is executed next-bar open:
   - Entry timing mode: tries to fill as a limit (pullback ATR) using `_try_fill_pending()`;
   - Next-bar open mode: `_open_position_at_open()` adjusts premium using delta × spot_diff and opens position.
7. Open positions tracked with option-specific state: `entry_price` (premium), signed `delta`, `current_premium`, `prev_close`, `underlying_entry_price`, `bars_held`, `premium_hwm`.
8. Each bar, `_check_exit()` computes premium evolution using: delta×dS + 0.5×gamma×dS^2 − theta decay, with conservative gamma/theta approximations; it checks underlying SL/target, premium SL (3-phase floor), target, time-stop, and applies trailing logic.
9. `_close_position()` applies cost model (commission, STT, GST, slippage, stamp duty) and records `BacktestTrade`.
10. OrderManager / Executor are used in live/paper modes to place real orders and track fills; `paper_trader` simulates realistic fills using same cost model.
11. Metrics & reports computed by BacktestEngine (equity curve, PSR, PBO-ready exports) and stored in `reports/`.

**Key Design Notes / Constraints**
- Options P&L is premium-based; entries/exits use estimated premium adjusted for spot moves via delta and a simple gamma term.
- Theta decay modeled heuristically per timeframe and DTE; daily vs intraday bar logic differs.
- Breakeven Trap: system sets a new SL at configurable R:R (breakeven_ratio) to convert big SLs into near-zero loss and enable runners.
- 5-stage trailing logic is central; dynamic trail stage tracks high-water mark and ratchets SL while preserving a floor.
- Risk overlays include Domain Shift Quotient (DSQ), volatility targeting, and drawdown throttle — these can skip or scale trades.
- Checkpointing: backtests save checkpoints for resuming runs (`_save_run_checkpoint`).
- Strike/lot rules, expiry rules, and market calendar are encoded in `prometheus/utils/indian_market.py` (including the 2025 expiry cutover and holiday lists).
- Config-driven: most behavior driven by `prometheus/config/settings.yaml` (timeframes, weights, cost model selection, capital brackets, intraday profiles, paper overrides).

**Pointers to implementation files**
- Data Engine: [prometheus/data/engine.py](prometheus/data/engine.py)
- Signals + Fusion: [prometheus/signals](prometheus/signals)
- Strategies: [prometheus/strategies](prometheus/strategies)
- Backtest Engine: [prometheus/backtest/engine.py](prometheus/backtest/engine.py)
- Risk Manager: [prometheus/risk/manager.py](prometheus/risk/manager.py)
- Order/execution: [prometheus/execution/order_manager.py](prometheus/execution/order_manager.py), [prometheus/execution/kite_executor.py](prometheus/execution/kite_executor.py), [prometheus/execution/paper_trader.py](prometheus/execution/paper_trader.py)
- Option math & market helpers: [prometheus/utils/options_math.py](prometheus/utils/options_math.py), [prometheus/utils/indian_market.py](prometheus/utils/indian_market.py)
- Config file: [prometheus/config/settings.yaml](prometheus/config/settings.yaml)

**Next recommended verification steps**
- Target `prometheus/utils/options_math.py` and all places that estimate gamma/delta for parity with finance expectations. (High priority.)
- Run static analysis (`ruff`, `mypy`) to flag API mismatches between Strategy outputs and Backtest input expectations.
- Execute a reproducible backtest run with a small window to validate entry/exit math (example: `python prometheus/main.py backtest --days 120 --symbol "NIFTY 50"`).

----
Generated on 2026-06-02 by the code review assistant.
