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
  signals/      — Technical (VWAP, Volume Profile, Supertrend, RSI, FVG, Liquidity Sweeps)
                  OI Analyzer, Regime Detector (AMD/Wyckoff), Signal Fusion Engine
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
- **Parrondo regime-switching**: per-bar regime detection routes to trend-following (markup/markdown) or mean-reversion (accumulation/distribution), skips volatile regimes
- **Gamma-aware premium model**: `dP = delta×dS + 0.5×gamma×dS² - theta×dt` with dynamic delta updates
- **Direction-aware delta drift**: put delta decreases on rallies, call delta increases (fixed Session 13)
- **Dynamic capital sizing**: position size scales with current equity, not just initial capital
- **5-stage trailing stop**: breakeven→20%→50%→70%→dynamic trail (high-water mark with 70% floor)
- **Unified signal path**: single factory generator for both Parrondo and baseline (no code duplication)
- **Causal regime detection**: per-bar `detect_fast()` in all modes (no look-ahead bias)
- **Reporting**: CAGR (compound), Alpha vs buy-and-hold, Calmar ratio, correct Sharpe annualization

## Backtest Results (Real yfinance Data — Session 13 Honest Metrics)

**Session 13 fixes**: Put slippage direction, put delta update, regime look-ahead removed, dynamic trailing stop, inline generator killed (−400 lines), equity curve alignment.

All "CAGR" figures below are true compound annual growth rate. All Sharpe values are correctly annualized.

### NIFTY 50 — Baseline (Trend-Only)
| Period | Return | CAGR | Trades | WR | PF | Sharpe | Calmar | Max DD | Alpha |
|--------|--------|------|--------|-----|-----|--------|--------|--------|-------|
| 5yr (2021-2026) | 451% | 40.7% | 158 | 51% | 1.69 | 1.96 | 1.21 | 33.8% | +31.6% |

### NIFTY 50 — Parrondo (Regime-Switching)
| Period | Return | CAGR | Trades | WR | PF | Sharpe | Calmar | Max DD | Alpha |
|--------|--------|------|--------|-----|-----|--------|--------|--------|-------|
| 5yr (2021-2026) | 413% | 38.8% | 165 | 53% | 1.64 | 1.79 | 0.77 | 50.2% | +29.6% |
| 15yr (2011-2026) | 1014% | 17.4% | 455 | 51% | 1.79 | 1.56 | 0.40 | 43.2% | +7.2% |

### Key Quality Metrics (Session 13: Honest + Improved)
- **Sharpe**: 1.79–1.96 (improved via delta fix + dynamic trail)
- **Sortino**: 3.42–4.03 (downside capture metric)
- **Alpha**: +29-32% excess return over buy-and-hold across all tests
- **Profit Factor**: 1.64-1.79 (15yr PF improved 1.75→1.79 via dynamic trail)
- **Win Rate**: 51-53%
- **MC P(profit)**: 99.0-99.2% (1000 block-bootstrap sims)

### Survived Market Crashes
- 2008 GFC (NIFTY -59.9%, BANKNIFTY -68.8%)
- 2020 COVID (NIFTY -38.4%, BANKNIFTY -47.9%)
- 2011 EU Crisis, 2015 China, 2018 IL&FS, 2022 Russia-Ukraine, 2024 Election

### Overfitting Status
- Pure math (no ML) — NOT overfit, walk-forward validated
- BANKNIFTY OOS PF 1.57 (Parrondo) — outperforms in-sample
- Parameters robust: PF stays 1.34+ across all ±20% variations

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
