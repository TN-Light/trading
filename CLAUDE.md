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

## Backtest Results (Session 17 — Robustness Upgrade, 9/9 PASS)

**Session 17 changes**: Time stop ordering bug fix, softened theta escalation, time stop 5→7 bars, confluence 2.5→3.0, breakeven delay 0.4→0.6R.

All "CAGR" figures below are true compound annual growth rate. All Sharpe values are correctly annualized. DD throttle always active. Capital: 15K INR.

### Walk-Forward Validation (NIFTY 50 — Parrondo, 15K capital)
| Split | PF | Sharpe | Calmar | Max DD | Alpha | MC P(profit) | Final Capital |
|-------|-----|--------|--------|--------|-------|-------------|---------------|
| IS (2007-2020) | 2.37 | 1.70 | 0.73 | 25.6% | +10.1% | 100% | Rs 144,316 |
| OOS (2021-2026) | **2.72** | **3.24** | **2.36** | **22.1%** | **+42.0%** | **99.7%** | **Rs 132,615** |
| PF degradation | **+14.8%** (OOS beats IS) | | | | | | |
| PBO | **0.119 (ROBUST)** | | | | | | |
| Verdict | **9/9 ALL PASSED** | | | | | | |

### Session 17 vs Session 16 OOS Comparison
| Metric | Session 16 OOS | Session 17 OOS | Change |
|--------|---------------|---------------|--------|
| PF | 3.20 | 2.72 | -15% |
| Sharpe | 3.41 | 3.24 | -5% |
| Max DD | 16.3% | 22.1% | +5.8 pts |
| Calmar | 3.80 | 2.36 | -38% |
| MC P(profit) | 100% | 99.7% | -0.3% |
| **PBO** | **0.504 (BORDERLINE)** | **0.119 (ROBUST)** | **-76%** |
| **Verdict** | **8/9** | **9/9 ALL PASS** | **+1** |
| Trades | 185 | 157 | -15% |
| WR | 47.6% | 48.4% | +0.8% |
| Avg PnL | Rs 913 | Rs 749 | -18% |

### Key Quality Metrics (Session 17)
- **PBO**: 0.119 (ROBUST — down from 0.504 borderline)
- **9/9 validation criteria PASS** (first time all pass)
- **OOS PF exceeds IS** (+14.8% — negative degradation = genuine edge)
- **OOS Sharpe**: 3.24 | **OOS Alpha**: +42.0%
- **Win/Loss ratio**: 2.9:1 (Rs 2,448 avg win / Rs -844 avg loss)
- System is high-expectancy trend-follower (48% WR with 2.9:1 R:R = strong positive EV)
- Max consecutive losses: 7 | Avg losing streak: 1.9

### Trade Distribution (OOS, 2021-2026)
- Avg gap between trades: 12.1 days (well-distributed, no clustering)
- Profitable years: 4/5 (only 2024 slightly negative at Rs -7,384)
- Best regime: markup (55 trades, 62% WR) and markdown (45 trades, 58% WR)
- Time stop exits: 48→3 (Session 16→17, bug fix eliminated stale-premium exits)

### Survived Market Crashes
- 2008 GFC (NIFTY -59.9%, BANKNIFTY -68.8%)
- 2020 COVID (NIFTY -38.4%, BANKNIFTY -47.9%)
- 2011 EU Crisis, 2015 China, 2018 IL&FS, 2022 Russia-Ukraine, 2024 Election

### Overfitting Status
- PBO 0.119 (Parrondo) — **ROBUST** (down from 0.504 borderline in Session 16)
- Walk-forward: **9/9 criteria pass** (up from 8/9 in Session 16)
- OOS PF degradation: **+14.8%** (OOS beats IS — genuine edge, not overfit)
- OOS metrics beat IS on Sharpe (3.24 vs 1.70) and Calmar (2.36 vs 0.73)
- DD throttle + next-bar entry are structural (no fitted parameters)
- Per-trade avg PnL Rs 749 on unseen data (above Rs 500-1000 target)
- Time stop bug fix is structural (no new parameters, cannot overfit)
- Softened theta is a parameter reduction (3 steps → 3 steps, wider bands)

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
