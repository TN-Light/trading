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
- Kelly Criterion EV filter skips negative expected value trades
- Multi-timeframe: auto-selects daily bars (>59 days) or 15min bars (<=59 days)
- Historically-aware DTE: computes days-to-expiry from bar timestamp, not current date

## Backtest Results (Real yfinance Data — Stress Tested)

### NIFTY 50 (Primary — Best Risk-Adjusted)
| Period | Return | Annual | Trades | WR | PF | Sharpe | Max DD |
|--------|--------|--------|--------|-----|-----|--------|--------|
| 5yr (2021-2026) | 226% | 45% | 212 | 47% | 1.23 | 1.09 | 58.5% |
| 10yr (2016-2026) | **800%** | 80% | 448 | 49% | **1.51** | **1.52** | 35.7% |
| 15yr (2011-2026) | **1,107%** | 74% | 655 | 48% | **1.55** | **1.59** | **24.9%** |
| 18.5yr (2007-2026) | **1,183%** | 64% | 774 | 47% | **1.52** | 1.27 | 30.8% |

### BANKNIFTY
| Period | Return | Annual | Trades | WR | PF | Sharpe | Max DD |
|--------|--------|--------|--------|-----|-----|--------|--------|
| 5yr (2021-2026) | 553% | 111% | 206 | 49% | 1.37 | 1.35 | 52.9% |
| 10yr (2016-2026) | 774% | 77% | 416 | 48% | 1.27 | 1.09 | 64.3% |
| 15yr (2011-2026) | 1,202% | 80% | 646 | 49% | 1.34 | 1.22 | 40.7% |
| 18.5yr (2007-2026) | 1,252% | 68% | 769 | 48% | 1.32 | 1.00 | 42.3% |

### Survived Market Crashes
- 2008 GFC (NIFTY -59.9%, BANKNIFTY -68.8%)
- 2020 COVID (NIFTY -38.4%, BANKNIFTY -47.9%)
- 2011 EU Crisis, 2015 China, 2018 IL&FS, 2022 Russia-Ukraine, 2024 Election

### Overfitting Status
- Pure math (no ML) — NOT overfit, but ~10 hand-tuned params need walk-forward validation
- Walk-Forward Validation planned: train 2007-2020, test 2021-2026 (next session)

## Running
```bash
python prometheus/main.py setup          # First time
python prometheus/main.py scan           # Market scan
python prometheus/main.py backtest       # Backtest (default: 59 days, 15min bars)
python prometheus/main.py backtest --days 1825  # 5-year backtest (daily bars)
python prometheus/main.py backtest --days 3650  # 10-year backtest (daily bars)
python prometheus/main.py backtest --days 5475  # 15-year backtest (daily bars)
python prometheus/main.py backtest --days 6750  # MAX ~18.5yr backtest (daily bars, back to 2007)
python prometheus/main.py paper          # Paper trade
python prometheus/main.py signal         # Signals only
```

## Dependencies
Core: pandas, numpy, scipy, PyYAML, loguru, rich, requests, yfinance
AI (optional): ollama, transformers, torch, sentence-transformers
Broker (optional): kiteconnect
Alerts (optional): python-telegram-bot
