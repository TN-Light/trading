# PROMETHEUS Trading System

## Architecture
- Indian F&O trading system, 37 files, ~9000 lines Python
- Capital: 15K–2L INR (adaptive) — NO naked option selling, focus on buying + debit spreads
- Broker: Zerodha Kite Connect
- AI: 100% free multi-provider stack (Groq 70B → Gemini 2.0 Flash → Ollama + FinBERT + Sentence-Transformers)

## Project Structure
```
prometheus/
  config/       — settings.yaml, credentials.yaml
  data/         — DataEngine (Kite, yfinance, NSE direct), SQLite store (limit=50000)
  signals/      — Technical (VWAP, Volume Profile, Supertrend, RSI, FVG, Liquidity Sweeps, Shannon Entropy, SHI)
                  OI Analyzer, Regime Detector (AMD/Wyckoff + Entropy Stop-Hunt), Signal Fusion Engine
  intelligence/ — Multi-provider LLM (Groq→Gemini→Ollama), FinBERT sentiment, Embedding pattern matcher
  strategies/   — Trend (options buying), Volatility (event straddles), Expiry (debit spreads), Selector
  risk/         — Hard limits, position sizing, circuit breakers, scenario analysis
  backtest/     — Walk-forward engine, Zerodha cost model, Monte Carlo simulation, 4-stage trailing stop
  execution/    — Broker abstraction, KiteExecutor, PaperTrader, OrderManager
  interface/    — Rich CLI dashboard, Telegram bot
  utils/        — Indian market rules, Black-Scholes/Greeks, Logger
  main.py       — Entry point: scan | backtest | paper | signal | setup
```

## Key Design Decisions
- Signal fusion uses weighted scoring (institutional tools weighted higher than retail indicators)
- Risk manager has NON-BYPASSABLE hard limits (daily loss, weekly loss, drawdown halt)
- All strategies must have minimum 2.5:1 risk:reward ratio (<50K capital)
- Paper trading mode simulates real broker behavior including slippage and costs
- **Breakeven trap** is the core edge — at 0.4:1 R:R, moves SL to entry+costs
- 4-stage trailing: breakeven (0.4:1) → lock 20% (1.0:1) → lock 50% (2.0:1) → runner 70% (3.0:1)
- Capital-adaptive parameters (SL width, targets, risk%, deployment) scale with capital brackets
- Kelly Criterion EV filter — confluence-adaptive win rate (30-50% based on signal score)
- Multi-timeframe: auto-selects daily bars (>59 days) or 15min bars (<=59 days)
- Historically-aware DTE: computes days-to-expiry from bar timestamp, not current date
- **Parrondo regime-switching**: per-bar regime detection routes to trend-following (markup/markdown) or mean-reversion (accumulation/distribution), volatile regimes try expiry spreads only
- **Entropy stop-hunt regime**: REVERTED Session 15 — overfit in walk-forward (PBO 0.687). Code removed.
- **Gamma-aware premium model**: `dP = delta×dS + 0.5×gamma×dS² - theta×dt` with dynamic delta updates
- **Direction-aware delta drift**: put delta decreases on rallies, call delta increases (fixed Session 13)
- **Dynamic capital sizing**: position size scales with current equity, not just initial capital
- **Drawdown-adjusted risk**: DD throttle single-layer (engine only) — skips trades when DD>20% (1-lot accounts), scales down larger accounts. Continuous linear formula, no stepped thresholds.
- **5-stage trailing stop**: breakeven→20%→50%→70%→dynamic trail (high-water mark with 70% floor)
- **Unified signal path**: single factory generator for both Parrondo and baseline (no code duplication)
- **Causal regime detection**: per-bar `detect_fast()` in all modes (no look-ahead bias)
- **Reporting**: CAGR (compound), Alpha vs buy-and-hold, Calmar ratio, correct Sharpe annualization
- **Next-bar entry**: signals on bar i enter on bar i+1 open (no same-bar look-ahead bias)
- **Multi-position support**: up to 2 concurrent positions (1 for <30K capital, 2 for >=30K)
- **Expiry debit spreads**: DTE ≤ 2 triggers EMA 8/21 crossover debit spread (fires in all regimes)
- **OTM strike selection**: accounts <50K use 1-strike OTM for capital efficiency
- **Realistic slippage**: 0.15% default (up from 0.05%) — conservative real-world options slippage
- **Time stop after SL/target**: time stop checks AFTER premium computation and SL/target (fixed Session 17 bug)
- **Softened theta escalation**: bars-held theta 0.8%/day for first 5 bars, gradual ramp (Session 17)
- **Extended time stop**: 7 bars daily (<50K), 6 bars (50-100K), 5 bars (100K+) — was 5/4/3
- **Higher confluence filter**: min trending score 3.0 (was 2.5) — requires 3+ confirming indicators
- **Delayed breakeven trap**: BE trigger at 0.6R (was 0.4R) — more breathing room before SL moves to entry
- **Intraday Supertrend + EMA**: backtest path now computes Supertrend + EMA 9/21 (was live-only), weight-gated so swing=0.0 (Session 23)
- **Intraday weight override**: bypasses broken regression weights (R²=-0.225) for intraday, boosts session VWAP to 1.0 (Session 23)
- **Intraday time stop**: 5min 60→36 bars, 15min 22→16 bars — cuts losers before square-off (Session 23)

## Backtest Results (Session 23 — Comprehensive Validation, March 2026)

**Session 23 changes**: Added Supertrend + EMA 9/21 to backtest indicators (weight-gated, swing=0), intraday weight override mechanism, reduced intraday time stops. ZERO swing impact (verified: identical metrics before/after changes).

**NOTE**: Numbers differ from Session 17 due to yfinance retroactive data adjustments (dividends, splits). Verified NOT a code regression — old code produces same numbers as new code on current data.

All "CAGR" figures below are true compound annual growth rate. All Sharpe values are correctly annualized. DD throttle always active. Capital: 15K INR.

### Walk-Forward Validation — 3-Index Cross-Validation (Parrondo, 15K capital)

#### NIFTY 50
| Split | PF | Sharpe | Calmar | Max DD | Alpha | MC P(profit) | Trades |
|-------|-----|--------|--------|--------|-------|-------------|--------|
| IS (2007-2020) | 1.41 | 0.76 | 0.07 | 98.5% | -1.0% | 82.4% | 686 |
| OOS (2021-2026) | **2.35** | **2.98** | **1.99** | **23.1%** | **+35.8%** | **98.8%** | **157** |
| PBO | **0.282 (ROBUST)** | | | | | | |
| Verdict | **9/9 ALL PASSED** | | | | | | |

#### NIFTY BANK
| Split | PF | Sharpe | Calmar | Max DD | Alpha | MC P(profit) | Trades |
|-------|-----|--------|--------|--------|-------|-------------|--------|
| IS (2007-2020) | 0.99 | -0.52 | -0.00 | 128.8% | -11.6% | 44.1% | 561 |
| OOS (2021-2026) | **2.48** | **2.01** | **0.51** | **68.9%** | **+24.4%** | **97.6%** | **131** |
| PBO | **0.063 (VERY ROBUST)** | | | | | | |
| Verdict | **9/9 ALL PASSED** | | | | | | |

#### SENSEX
| Split | PF | Sharpe | Calmar | Max DD | Alpha | MC P(profit) | Trades |
|-------|-----|--------|--------|--------|-------|-------------|--------|
| IS (2007-2020) | 2.97 | 1.30 | 0.22 | 89.2% | +11.8% | — | 493 |
| OOS (2021-2026) | **3.04** | **2.87** | **1.65** | **27.8%** | **+37.0%** | **99.7%** | **119** |
| PBO | **0.000 (PERFECT)** | | | | | | |
| Verdict | **9/9 ALL PASSED** | | | | | | |

#### Cross-Index OOS Summary
| Metric | NIFTY 50 | NIFTY BANK | SENSEX |
|--------|----------|------------|--------|
| PF | 2.35 | 2.48 | **3.04** |
| Sharpe | **2.98** | 2.01 | 2.87 |
| WR | 44.6% | **52.7%** | 49.6% |
| Alpha | +35.8% | +24.4% | **+37.0%** |
| PBO | 0.282 | **0.063** | **0.000** |
| Verdict | **9/9 PASS** | **9/9 PASS** | **9/9 PASS** |

### Sensitivity Analysis (NIFTY 50, 15 parameter variations, ~15yr)
| Parameter | Default | Range Tested | Min PF | Max PF | Stable? |
|-----------|---------|-------------|--------|--------|---------|
| confluence_trending | 3.0 | 2.5-3.5 | 1.63 | 2.63 | YES |
| target_atr_mult | 3.0 | 2.5-3.5 | 1.91 | 2.27 | YES |
| time_stop_bars | 7 | 5-9 | 1.69 | 2.61 | YES |
| kelly_wr | 0.35 | 0.30-0.40 | 2.61 | 2.61 | YES (identical) |
| breakeven_ratio | 0.6 | 0.4-0.8 | 2.29 | 2.61 | YES |
| **Verdict** | | | **Min PF 1.63** | | **ROBUST** |

### Intraday Backtest Results (Session 23, 59 days, 5min, Parrondo)
| Metric | NIFTY 50 | SENSEX | NIFTY BANK |
|--------|----------|--------|------------|
| Trades | **20** | 17 | 18 |
| WR | **40.0%** | 35.3% | 16.7% |
| PF | **3.11** | 1.82 | 0.58 |
| Sharpe | **4.27** | 1.64 | -0.91 |
| Max DD | 19.6% | 15.8% | 38.9% |
| Final Cap | **Rs 26,233** | Rs 19,733 | Rs 11,213 |
| MC P(profit) | **91.8%** | 81.0% | 27.1% |
| Square-off exits | 0 | 0 | 0 |

**Intraday caveats**: yfinance caps at ~60 days for 5min data. With 8-20 trades, confidence interval on WR is ±30%. BANKNIFTY intraday is unprofitable — system doesn't generalize across all indices for intraday. NIFTY 50 + SENSEX are profitable but sample is too small for confidence.

### Overfitting Status (Session 23)
- **Swing: NOT OVERFIT** across all 3 indices (PBO: 0.282, 0.063, 0.000)
- All 3 indices pass **9/9 walk-forward criteria**
- OOS consistently beats IS on key metrics (Sharpe, Calmar, Alpha)
- Sensitivity analysis: PF stays >= 1.63 across ALL parameter variations (ROBUST)
- kelly_wr parameter has ZERO effect on results (0.30-0.40 → identical PF 2.61)
- **Intraday: Cannot determine** — 59 days / 8-20 trades is statistically insufficient
- Intraday improvements are structural (indicator parity, well-known indicators, fixing dysfunctional time stop)

## Running
```bash
python prometheus/main.py setup          # First time
python prometheus/main.py scan           # Market scan
python prometheus/main.py backtest       # Backtest (default: 59 days, 15min bars)
python prometheus/main.py backtest --days 1825  # 5-year backtest (daily bars)
python prometheus/main.py backtest --days 1825 --parrondo  # 5yr with Parrondo regime-switching
python prometheus/main.py backtest --days 5475 --parrondo  # 15yr with Parrondo
python prometheus/main.py backtest --days 1825 --parrondo --entry-timing  # Parrondo + entry timing
python prometheus/main.py backtest --days 1825 --entry-timing --entry-pullback-atr 0.2  # Custom pullback
python prometheus/main.py walkforward --parrondo  # Walk-forward with Parrondo
python prometheus/main.py walkforward --parrondo --entry-timing  # Walk-forward with entry timing
python prometheus/main.py backtest --days 6750  # MAX ~18.5yr backtest (daily bars, back to 2007)
python prometheus/main.py paper          # Paper trade
python prometheus/main.py signal         # Signals only
```

## Dependencies
Core: pandas, numpy, scipy, PyYAML, loguru, rich, requests, yfinance
AI (optional): ollama, transformers, torch, sentence-transformers
Broker (optional): kiteconnect
Alerts (optional): python-telegram-bot
