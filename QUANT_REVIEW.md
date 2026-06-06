# Quantitative & Architectural Codebase Review

## 1. Executive Summary
*High-level summary of the codebase quality, major red flags, and overall assessment of the trading systems.*

## 2. Backtesting Integrity & Infrastructure
*Analysis of the backtesting engines (e.g., `prometheus/backtest/engine.py`, `nexus_system.py`, `alpha_intraday_engine.py`).*
- **Lookahead Bias:** Are future data points leaking into current state calculations?
- **Execution Assumptions:** Are fills realistic? Are slippage and transaction costs accurately modeled?
- **Data Integrity:** Survivorship bias, split/dividend handling, data frequency alignment.

## 3. Alpha Generation & Signals
*Analysis of `prometheus/signals/` and alpha generation modules.*
- **Overfitting Risk:** Hardcoded parameters, excessive parameter tuning, curve-fitting.
- **Robustness:** Does the alpha generation logic hold up across different market regimes?
- **Logical Flaws:** Errors in mathematical calculations, indicator formulations, or cross-asset correlations.

## 4. Strategy Implementation
*Analysis of `prometheus/strategies/` (trend, volatility, expiry, etc.).*
- **State Management:** How are trade lifecycles managed?
- **Regime Filtering:** Are strategies correctly identifying and adapting to market conditions?
- **Entry/Exit Logic:** Sanity check on the conditions that trigger trades.

## 5. Risk Management & Sizing
*Analysis of `prometheus/risk/`.*
- **Drawdown Controls:** Effectiveness of loss elimination and portfolio scaling.
- **Position Sizing:** Are constraints respected? Are sizing algorithms statistically sound?

## 6. Execution Logistics
*Analysis of `prometheus/execution/`.*
- **Broker Integration:** Resilience to API failures, rate limiting, and disconnects.
- **Order Management:** Latency considerations, partial fills, and order state synchronization.

## 7. Operational & Utility Code
*Analysis of data pipelines, analysis tools, and general utilities.*
- **Data Fetching & Storage:** Latency, caching, consistency.
- **Research/Analysis Scripts:** Validity of metrics calculated in loss analysis, microstructure tools, etc.

## 8. Conclusion & Actionable Recommendations
*Steps for refactoring, fixing bugs, and improving the robustness of the system.*

### Backtesting Engines Review Findings

#### Lookahead Bias & Data Integrity
* `alpha_intraday_engine.py`: Uses `(df['high'] - df['close'].shift()).abs()` for True Range calculations, which correctly references previous close. However, the event loop logic evaluates entries checking current conditions against next bar's open (`open_arr[i+1]`) after evaluating `uptrend`/`downtrend` rules at index `i`. While seemingly avoiding lookahead by entering on `i+1`, it evaluates `rsi_arr[i]` which incorporates data up to the close of `i`. There isn't obvious egregious lookahead bias in the core intraday engines.
* `prometheus/backtest/engine.py`: Explicitly implements `_open_position_at_open` for "next-bar entry (no lookahead)". Wait, in `run()`, when evaluating signals:
  `data_so_far = data.iloc[:i + 1]` -> `signal = signal_generator(data_so_far)`.
  If it enters on the *same* bar `i`, it would be lookahead bias. The code checks `pending_signal`: if not None, enters `_open_position_at_open(pending_signal, current_bar)`. Then generates a new signal to be pending for the *next* bar. This is correct and avoids lookahead bias!
* `nexus_system.py`: The `calc_sne` function computes SNE with a sliding window, `window = ret_arr[i-14:i]`. This only uses past returns up to `i-1`. In the intra-day loop, it checks `day_df_5m`, filters `morning_points < 930`, and correctly updates trailing stops based on past bars.

#### Execution Assumptions & Realism
* `prometheus/backtest/engine.py` implements a sophisticated `CostModel` including slippage, STT, brokerage, and GST.
* Options execution model dynamically prices premium using `delta` approximations and applies theta decay (e.g. `theta_decay = current_premium * theta_pct`). Also models Gamma `gamma = abs(delta) * (1 - abs(delta)) * gamma_scale`. This is a surprisingly good approximation for an idealized backtest environment, avoiding the common pitfall of assuming exact BSM pricing without microstructure effects.
* Slippage: Explicitly applied based on premium size `slippage = premium_exit * self.cost_model.slippage_pct`.
* Intraday Engines (`alpha_intraday_engine`): Uses fixed assumptions (fixed point SL, simplistic lot multiplier for drawdown). It's more of an exploratory engine compared to Prometheus.

#### Potential Flaws
* `alpha_intraday_engine.py`: Assumes fill exactness. `sl_px = entry_price - sl_atr * atr_arr[i]`. `elif low_arr[i] <= sl_px: trade_pnl = sl_px - entry_price`. A low below SL implies a fill exactly AT the SL. In reality, gapping past SL can cause severe slippage.
* `nexus_system.py`: Trailing SL relies heavily on simulated option pricing: `sim_premium = active_trade['premium_entry'] + ((row_5m['close'] - active_trade['nifty_entry_px']) * dm * DELTA_PROXY)`. If `sim_premium_low <= active_trade['stop_px']`, exit at `stop_px`. Similar flaw: guarantees stop fill exactly at limit price in a fast market.

### Signal Generation & Intelligence Logic Findings

#### Alpha Generation Mechanisms
* **`fusion.py` (SignalFusionEngine)**: This engine operates on a weighted sum model to build a confluence score across technicals (VWAP, Volume Profile, Liquidity Sweeps, FVGs), OI signals, regime, and AI sentiment. The design correctly translates a multi-dimensional analysis into a single decision vector. However, weights are completely hardcoded (e.g., `volume_profile: 0.85`, `ai_sentiment: 0.65`). This suggests parameter tuning that might be overfitted to a specific historical regime.
* **`apex_generator.py`**: A complex 5-component technical stack used for 0-100 edge score evaluation. Includes advanced concepts like "Gamma Ambush" (restricting trading to 10:45-11:15 on Thursday expiries) and "Compression Coil" / EMA21 Retest gates. The use of specific time windows and strict coil ratios (e.g., `< 0.35`) is logically sound for avoiding volatility expansion traps, but the exact parameter values risk curve-fitting.
* **Technical Implementations (`technical.py`)**:
  - `calculate_session_vwap()` correctly resets the VWAP metric at the start of each daily session, which is vital to avoid cross-day lookahead and matches actual institutional execution benchmark logic.
  - `detect_fair_value_gaps()` and `detect_liquidity_sweeps()` accurately map structural concepts from price-action trading into programmable rules.
  - `calculate_atr()` logic correctly factors in gap risk by using previous close (`high_close`, `low_close`).

#### Overfitting Risk & Structural Observations
* **Over-Parameterization**: The logic relies on multiple hardcoded thresholds: e.g., `min_rr = 2.0`, `bull_score >= 3 and bear_score < 2`, `coil_len = 6`. While these rules combine to create a selective filter (which improves backtest metrics), they may be highly brittle to out-of-sample data unless frequently retrained or adapted via the intelligence layer.
* **Regression Module**: `signal_regression.py` contains logic to train linear/ridge regression on signal features to predict PnL. This indicates an attempt to dynamically adjust signal weights instead of purely relying on hardcoded heuristics. If integrated properly, this offsets the overfitting risk mentioned above.

#### Logical Flaws
* No egregious mathematical errors were found in standard indicator formulations (RSI, ATR, VWAP, EMA, Supertrend).

### Strategy Implementations Findings

#### Expiry Strategy (`prometheus/strategies/expiry.py`)
* Logic heavily filters trades based on Days to Expiry (`dte <= 2`).
* Distinct modes depending on conditions: Momentum Breakout (first 30 minutes opening range break) and Debit Spread (near expiry capital efficiency).
* Premium estimation falls back to an exact Black-Scholes formula using a fixed assumed IV (`sigma = 0.15`). Since IV fluctuates massively on expiry days (IV crush), fixing it at 15% across all states is a significant flaw and can lead to heavily distorted backtest returns or PnL targets for paper trades.

#### Trend Strategy (`prometheus/strategies/trend.py`)
* Implements a classic multi-timeframe alignment check (Hourly for bias, 15m for entry zone).
* Strike selection algorithm checks available capital: if capital < 50,000, it buys 1-strike OTM instead of ATM.
* Again, options premium estimation falls back to Black-Scholes if chain data isn't provided.
* Incorporates OI (Open Interest) walls as a veto mechanism: `distance_to_resistance > 0.005` (requires at least 0.5% room to the nearest major resistance before taking a bullish trade). This is a solid, realistic market microstructure check.
* The directional bias calculation re-uses the exact same hardcoded weights found in `fusion.py` (e.g., VWAP 0.80, volume profile 0.85).

#### Conclusion on Strategy Logic
Strategies are logically sound from a qualitative perspective (incorporating macro timeframe alignment, OI walls, and time-based entry constraints). The primary weakness lies in the execution price estimation fallback (using constant implied volatility in Black-Scholes), which could severely misprice options in backtest environments missing complete historical option chain ticks.

### Risk Management & Execution Layers Findings

#### Risk Management (`prometheus/risk/`)
* **`loss_elimination_engine.py`**: This is an extremely sophisticated pre-trade risk layer. It implements multiple layers:
  1. `PreTradeKillSwitch`: Hard blocks trades scoring > 70 on pattern-match against known loss heatmaps.
  2. `AdaptiveStopLoss`: Dynamically adapts SL strategy based on the recent dominant "loss archetype" (e.g. Stop Hunt vs Chop Grind).
  3. `TemporalBlackoutManager`: Implements strict no-trade windows (e.g., first and last 15 minutes of the session).
  4. `RegimeGate`: Hard blocks strategies incompatible with the current regime (e.g., trend following blocked in "accumulation").
  5. `CircuitBreaker`: Reduces position sizes or enforces paper trading upon consecutive losses.
* This represents institutional-level risk controls. The risk engine is designed to prevent behavioral mistakes and algorithmic "tilt".

#### Execution Layer (`prometheus/execution/`)
* **`kite_executor.py`**: Implements the actual live connection using the `kiteconnect` library. It includes rate-limiting safety (`_min_order_interval = 0.5`). Proper handling of basic order operations (place, modify, cancel).
* **`order_manager.py`**: Handles the orchestration between generating a signal and placing it with the broker.
* **Flaws/Concerns**:
  - `kite_executor.py` lacks robust retry mechanisms or network exception handling. If `self.kite.place_order` throws a `requests.exceptions.ConnectionError`, the order just fails. In a fast-moving market, this could leave the algorithm blind to dropped packets or API 502s without a fallback queue.
  - Latency handling: `paper_trader.py` exists, but there's no explicit logic to track live partial fills or manage exact state synchronization if Kite sends a webhook update late.


### Data Handling, Analysis, and Utilities Findings

#### Data Engineering (`prometheus/data/`)
* **`engine.py` (DataEngine)**: Handles data fetching with graceful fallbacks. Primary fetch is Zerodha Kite Connect, fallback is Yahoo Finance (`YFinanceFallback`), and options chain data uses direct NSE endpoints (`NSEDirectFeed`). This hybrid approach is highly robust for maintaining live operation during API failures.
* Data integrity looks solid. `YFinanceFallback` accurately maps ticker names, and NSE scraping handles the cookie-initialization requirement of the NSE website gracefully.

#### Research & Analysis Scripts (`prometheus/analysis/`)
* **`loss_dna_tagger.py`**: A massive analytical script that tags trades with over 40 attributes at entry (e.g., regime, sequence of wins/losses, cost pct impact, time holding, distance from EMAs). This acts as the raw data generation layer for pattern mining.
* **`pattern_miner.py`**:
  * Implements `LossArchetypeClassifier` which clusters losses into archetypes like "Stop Hunt", "False Signal", "Chop Grind", and "Overextension".
  * Integrates an advanced `HiddenPatternDetector` utilizing XGBoost and SHAP (Shapley Additive exPlanations) values to identify non-linear relationships that cause losses.
  * This is an incredibly powerful research pipeline. However, relying on XGBoost without strict train/test/val splits *on out-of-sample data* could lead to the kill switches (in `loss_elimination_engine.py`) over-filtering the strategy in a way that curves fits the backtest. If the system "learned" these loss rules from the entire dataset, the reported backtest performance will be heavily inflated by lookahead bias *in the ruleset itself*.


### Overall Conclusion & Actionable Recommendations

**General Assessment:**
This is an institutional-grade, highly sophisticated F&O algorithmic trading system. The architecture is modular, clearly separating Alpha Generation, Risk Elimination, Execution, and Data Analysis. It leverages advanced quantitative methods (GMM regime clustering, options pricing proxies for futures backtesting, institutional flow mapping via VWAP/FVGs).

**Key Vulnerabilities:**
1. **Execution Model Execution Bias:** The backtesting trailing stops assume exact limit fills during rapid price movements (found in `nexus_system.py` and `alpha_intraday_engine.py`). This underestimates slippage during volatile events (e.g., stop hunts or macro data drops).
2. **Hardcoded IV in Strategy Estimates:** The `expiry.py` and `trend.py` strategies fallback to a hardcoded 15% Implied Volatility when pricing options via Black-Scholes. This creates severe distortions during high IV environments (like earnings or elections) or IV crush on expiry days.
3. **Meta-Lookahead Bias:** The `loss_elimination_engine.py` is deeply intertwined with `pattern_miner.py` (which uses XGBoost to find loss profiles). If the loss heatmap constraints were designed by analyzing the *entire* historical dataset, any backtest utilizing these kill-switches is curve-fitted. The rules must only be derived from a strict in-sample training window and applied strictly to out-of-sample forward walks.
4. **Parameter Overfitting:** `fusion.py` relies on statically weighted confluence scoring (e.g., VWAP=0.80).

**Recommendations for Future Work:**
1. **Dynamic Volatility Surface Integration:** Replace the static `sigma=0.15` in `_estimate_premium` with an expanding window IV proxy (e.g., using India VIX or historical 20-day realized volatility).
2. **Strict Walk-Forward Analysis:** Ensure that the XGBoost loss-pattern rules are generated in a walk-forward manner (e.g., train on 2018-2020, apply rules to 2021).
3. **Execution Delay Simulation:** Introduce randomized or latency-induced slippage (e.g., fill at 1 ATR worse than stop-loss) in the backtesting engine to simulate true fast-market conditions.
4. **Broker Resiliency Layer:** Enhance `kite_executor.py` with retry logic, exponential backoffs, and heartbeat websocket monitoring to gracefully handle dropped connections.

*End of Review.*

### Telegram Bot & Interface Review
* **Signal Flow:**
  - Signals flow through `prometheus/main.py` directly to the Telegram bot via `self.telegram.alert_new_signal(signal)`.
  - There are multiple entry points for signals depending on the mode (`run_scan`, `run_full_auto_mode`, `run_combined_mode`).
  - A key discrepancy observed: If backtesting produces many trades but the Telegram bot shows 0, it is typically because the live/paper mode relies on the `LossEliminationEngine` (the pre-trade kill switch) and `MultiAccount` routing constraints, which actively filter and reject "weak" signals before they are transmitted. The backtest engine handles execution completely separately (`prometheus/backtest/engine.py`) and does not always hook into the live Telegram alert flow.
  - Furthermore, `is_expiry_thursday` and other time-based filters aggressively drop signals in live trading that might pass in a less restrictive backtest loop.

### Paper Trading & Execution Realism (`prometheus/execution/paper_trader.py`)
* The `PaperTrader` attempts to be highly realistic. It explicitly implements:
  - **Spread Modeling:** Uses real-time bid/ask spreads from Angel One when placing market orders (`buy at ask`, `sell at bid`), avoiding the illusion of "mid-price" fills. If Angel One data is missing, it falls back to a 0.15% fixed slippage model.
  - **Zerodha Cost Model:** It correctly subtracts STT, GST, and brokerage fees in real-time, just like the backtester.
* **Flaws in Paper Trading:**
  - `LIMIT` and `SL` orders assume absolute perfect limit fills (`current <= order.price -> fill exactly at order.price`). In reality, Limit orders may not be filled due to queue position, and Stop-Loss Market (SL-M) orders suffer gap slippage. The code uses `current` as the SL-M fill price, which is slightly better than using the trigger price exactly, but still lacks gap/queue simulation.

### Signal Generation Discrepancies (Backtest vs. Live Telegram Bot)
There is a structural divergence between how signals are treated in backtesting vs. live paper/Telegram modes:
1. **Backtesting (`prometheus/backtest/engine.py`)**: Continuously iterates through historical bars and triggers trades based directly on `signal_generator` logic. If `alpha_intraday_engine.py` or the `nexus_system.py` logic finds a pattern, it enters immediately.
2. **Live/Paper Mode (`prometheus/main.py`)**: Live loops run via `run_scan`, `run_paper_mode`, or `run_full_auto_mode`. Signals generated here are routed through multiple constraint filters:
   - `Pre_Trade_Risk_Manager` or `LossEliminationEngine`: Hard-blocks signals if current loss sequences are high or if the ML model tags the signal as "Stop Hunt" / "Regime Mismatch".
   - `MultiAccount` Constraints: Signals are filtered via `_route_candidates_for_capital()`. If there isn't a liquid option strike available that respects the max risk (e.g., 2% of capital), the signal is silently dropped.
   - **Time Filters**: Live scans are triggered explicitly (e.g. `_run_paper_mode_swing_15m` runs a while loop checking specific intervals, and filters `is_expiry_thursday` or specific `catch_up_post_close` logic).

**Conclusion:**
If you see numerous trades in backtest but 0 signals on Telegram, it is because the live trading constraints (Risk Management kill-switches, capital allocation logic, and strict time-of-day execution windows) are aggressively pruning "weak" signals that the backtest naive loop simply executes.
