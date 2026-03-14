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

## Backtest Results (Session 16 — Honesty Fixes + Multi-Position + Expiry Spreads)

**Session 16 fixes**: Slippage 0.05%→0.15%, next-bar entry (no same-bar look-ahead), removed double DD throttle, multi-position support, expiry debit spreads, OTM strike selection for <50K, MC Sharpe annualization fix.

All "CAGR" figures below are true compound annual growth rate. All Sharpe values are correctly annualized. DD throttle always active. Capital: 15K INR.

### NIFTY 50 — Parrondo 5yr (15K capital)
| Period | Return | CAGR | Trades | WR | PF | Sharpe | Calmar | Max DD | Alpha | Avg PnL |
|--------|--------|------|--------|-----|-----|--------|--------|--------|-------|---------|
| 5yr (2021-2026) | 1064% | 63.5% | 186 | 47.3% | 3.10 | 3.06 | 3.28 | 19.3% | +53.5% | Rs 858 |

### Walk-Forward Validation (NIFTY 50 — Parrondo, 15K capital)
| Split | PF | Sharpe | Calmar | Max DD | Alpha | MC P(profit) | Final Capital |
|-------|-----|--------|--------|--------|-------|-------------|---------------|
| IS (2007-2020) | 3.25 | 1.71 | 0.74 | 26.0% | +10.8% | 100% | Rs 156,749 |
| OOS (2021-2026) | **3.20** | **3.41** | **3.80** | **16.3%** | **+51.9%** | **100%** | **Rs 183,821** |
| PF degradation | -1.5% | | | | | | |
| PBO | **0.504** (BORDERLINE) | | | | | | |
| Verdict | **8/9 PASSED** | | | | | | |

### Session 16 vs Session 15 OOS Improvement
| Metric | Session 15 OOS | Session 16 OOS | Change |
|--------|---------------|---------------|--------|
| PF | 1.29 | **3.20** | +148% |
| Sharpe | 1.26 | **3.41** | +171% |
| Max DD | 34.4% | **16.3%** | -18 pts |
| Calmar | 0.47 (FAIL) | **3.80** (PASS) | +709% |
| MC P(profit) | 81.9% (FAIL) | **100%** (PASS) | +18 pts |
| PBO | 0.417 | 0.504 | +0.087 |
| Verdict | 7/9 | **8/9** | +1 check |

### Key Quality Metrics (Session 16)
- **OOS Sharpe**: 3.41
- **OOS Alpha**: +51.9% excess return over buy-and-hold on unseen data
- **OOS Profit Factor**: 3.20
- **OOS Max DD**: 16.3%
- **PBO**: 0.504 (BORDERLINE — just above 0.50 threshold)
- **PF degradation**: -1.5% (IS→OOS) — almost zero degradation
- **Per-trade avg PnL**: Rs 913 (OOS), Rs 858 (backtest)

### Survived Market Crashes
- 2008 GFC (NIFTY -59.9%, BANKNIFTY -68.8%)
- 2020 COVID (NIFTY -38.4%, BANKNIFTY -47.9%)
- 2011 EU Crisis, 2015 China, 2018 IL&FS, 2022 Russia-Ukraine, 2024 Election

### Overfitting Status
- PBO 0.504 (Parrondo) — BORDERLINE, just above 0.50 threshold
- Walk-forward: 8/9 criteria pass (up from 7/9 in Session 15)
- OOS PF degradation: -1.5% (IS→OOS) — near-zero degradation
- OOS metrics beat IS on Sharpe (3.41 vs 1.71) and Calmar (3.80 vs 0.74)
- DD throttle + next-bar entry are structural (no fitted parameters)
- Per-trade avg PnL Rs 913 on unseen data (above Rs 500-1000 target)

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
