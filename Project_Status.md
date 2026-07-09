# Project Status — PROMETHEUS (snapshot: 2026-07-07)

## Overview
- **Purpose**: Indian F&O options-buying system (NO naked option selling)
- **Capital**: ₹15K–2L INR (adaptive bracket sizing)
- **Broker**: AngelOne (primary data feed), Zerodha Kite Connect (execution)
- **Mode**: Paper (current), targeting Live
- **Instruments**: NIFTY 50, NIFTY BANK, SENSEX
- **Languages/libs**: Python; pandas, numpy, scipy, yfinance, kiteconnect, sqlite, requests, loguru, rich

---

## Architecture — Two Signal Engines

### 1. APEX Intraday Engine (primary live path)
- **Entry**: `signals/apex_generator.py` → `signals/aes_fusion.py`
- **Technical stack**: VWAP, Session VWAP, EMA 9/21 cross, SuperTrend, RSI Divergence
- **Scoring**: AES 6-factor edge score (0–100):
  - regime_alignment: 30%, signal_confluence: 25%, volatility_support: 15%
## Architecture — Signal Engine (July 2026)

### Apex Hunter (live path — the only active signal engine)
- **Entry**: `_make_signal_generator()` in `main.py` → `signals/fusion.py` + `signals/technical.py`
- **Technical stack**: VWAP, Volume Profile, FVG, Liquidity Sweeps, OTE, RSI Divergence, SuperTrend, EMA (ICT/SMC concepts)
- **Scoring**: Weighted confluence sum (`fusion.py`)
- **Sizing**: Risk-based (`_size_position()`)
- **Intraday tuning**: V2 config overrides applied per scan (confluence thresholds, RR, time stops, dead zones, VWAP/EMA/SuperTrend alignment gates)
- **Bar interval**: Auto-selected via VIX (≥18 → 5min, <18 → 15min)
- **Config**: `intraday.use_backtest_generator: true`
- **Validated**: 818 trades / 11 years, 18.7% max drawdown (swing backtest)

> **Note**: A separate `apex_generator.py` was built and tested but deleted on 2026-07-09 after -58% on Bank Nifty in 30 days. All references removed. The term "Apex Hunter" now refers exclusively to `_make_signal_generator()`.

---

## Core Files (current, verified July 2026)

### Signals
| File | Purpose |
|------|---------|
| `prometheus/signals/technical.py` | VWAP, Volume Profile, FVG, Liquidity Sweeps, OTE, RSI, ATR, SuperTrend, EMA |
| `prometheus/signals/fusion.py` | Weighted confluence engine — primary signal path (Apex Hunter) |
| `prometheus/signals/aes_fusion.py` | AES 6-factor edge scoring (used by fusion.py) |
| `prometheus/signals/regime_detector.py` | AMD/Parrondo regime classification |
| `prometheus/signals/oi_analyzer.py` | PCR, Max Pain, OI support/resistance, IV skew |
| `prometheus/signals/qrd_estimator.py` | Quantum Regime Detection |
| `prometheus/signals/strike_gravity.py` | Strike-level gravity/penalty mapping |
| `prometheus/signals/expiry_clock.py` | Expiry session timing evaluation |

### Strategies
| File | Purpose | Status |
|------|---------|--------|
| `prometheus/strategies/trend.py` | Multi-timeframe trend strategy | LIVE (imported by main.py) |
| `prometheus/strategies/volatility.py` | Volatility strategy | LIVE (imported by main.py) |
| `prometheus/strategies/expiry.py` | Expiry-day strategy | LIVE (imported by main.py) |
| `prometheus/strategies/selector.py` | Regime → strategy routing | LIVE (imported by main.py) |

### Risk
| File | Purpose |
|------|---------|
| `prometheus/risk/manager.py` | Non-bypassable pre-trade gate (13 checks) |
| `prometheus/risk/position_sizer.py` | Capital-bracket sizing |
| `prometheus/risk/portfolio_scaler.py` | Portfolio-level scaling |

### Execution
| File | Purpose |
|------|---------|
| `prometheus/execution/order_manager.py` | Full trade lifecycle management |
| `prometheus/execution/paper_trader.py` | Paper fills with Zerodha cost model + Angel One bid/ask |
| `prometheus/execution/kite_executor.py` | Zerodha Kite Connect bindings |
| `prometheus/execution/position_monitor.py` | 5-stage trailing stop + time stops |

### Other
| File | Purpose |
|------|---------|
| `prometheus/intelligence/llm_analyzer.py` | LLM analysis (lazy-loaded; AI currently disabled) |
| `prometheus/interface/telegram_bot.py` | Two-way Telegram bot with proxy/SNI fallbacks |
| `prometheus/backtest/engine.py` | Walk-forward, Monte Carlo, PBO validation |
| `prometheus/main.py` | CLI orchestrator (~8,900 lines) |
| `smoke_test_quick.py` | Pre-commit 4-check smoke test |

---

## Deleted Files (cleaned up June–July 2026)
- `signals/cross_asset_relay.py` — dead code, zero imports
- `risk/loss_elimination_engine.py` — dead code, zero imports
- `intelligence/signal_regression.py` — dead code, zero imports
- `signals/apex_generator.py` — deleted 2026-07-09; -58% on Bank Nifty; replaced by Apex Hunter

---

## Stop-Loss Architecture (Two Separate Systems)

### 3-Phase Premium SL Floor (`engine.py` lines 1422–1444)
| Phase | Bars Held | Behavior |
|-------|-----------|----------|
| 1 | ≤3 | SL immune (absorb IV crush noise) |
| 2 | 4–5 | SL at 80% buffer |
| 3 | >5 | Full SL enforcement |

### 5-Stage Trailing Ratchet (`engine.py` lines 1461–1511)
| Stage | Trigger | Action |
|-------|---------|--------|
| 0 | Premium hits 0.4R above entry | Move SL to entry + 0.10R (breakeven trap) |
| 1 | Premium hits 1.0R | Lock 20% profit |
| 2 | Premium hits 2.0R | Lock 50% profit |
| 3 | Premium hits 3.0R | Lock 70% + init high-water mark |
| 4 | Ongoing | Dynamic trail: 30% below HWM, floor at 0.70R |

---

## Risk Manager — 13 Pre-Trade Checks (non-bypassable)
1. System halt check
2. Trading hours validation
3. Daily loss limit (dynamic bracket)
4. Daily loss % limit
5. Weekly loss limit
6. Daily trade count
7. Intraday trade limit
8. Max open positions
9. Position size validation
10. Consecutive losses pause
11. Drawdown halt
12. Correlated exposure check
13. Duplicate instrument check

---

## Key Configuration (settings.yaml, current)
| Setting | Value |
|---------|-------|
| `system.mode` | `paper` |
| `intraday.enabled` | `true` |
| `intraday.instruments` | NIFTY 50, NIFTY BANK, SENSEX |
| `intraday.use_backtest_generator` | `true` |
| `swing.use_backtest_generator` | `true` |
| `ai.gemini.enabled` | `false` (cleanly disabled) |
| `ai.groq.enabled` | `false` (cleanly disabled) |
| Entry window | 09:30–14:30 |
| Dead zone | 11:30–13:30 |
| Square-off | 15:15 |
| Max daily loss | Rs 15,000 |
| Max positions | 6 |
| Drawdown halt | 50% |
| `intraday.v2.max_daily_trades` | 5 |

### Paper Mode Risk Overrides (currently active)
Paper mode uses relaxed limits (`100%` thresholds, `150%` position size) for uninterrupted statistics collection. **Base risk limits must be verified before live deployment.**

---

## Known Bugs Fixed (July 2026)
| Bug | Impact | Fixed |
|-----|--------|-------|
| `--interval 300` misused as bar interval → `300minute` candles | Zero trades in backtest | ✅ 2026-07-09 |
| `--apex` flag silently ignored (apex param never wired) | Backtest used wrong engine | ✅ 2026-07-09 (then flag removed) |
| Duplicate process on reboot (3 startup mechanisms) | 2 live processes running simultaneously | ✅ 2026-07-09 |
| Telegram credentials empty | No signals delivered | ✅ 2026-07-09 |

---

## Pre-Commit Checklist
Run before every `git push`:
```powershell
python smoke_test_quick.py
```
Catches: ImportErrors, CLI crashes, 300minute regression, zero-trade regression.

---

## Related Documentation
| File | Type | Notes |
|------|------|-------|
| `CLAUDE.md` | Session log + architecture | Updated July 2026 |
| `DATA_FLOW.md` | Architecture & data flow | Updated July 2026 |
| `Upgrade.md` | Design whitepaper | Implementation status annotated |
| `prometheus_audit_v2.md` | Historical audit | ⚠️ Contains errors — see header |
| `counter_audit.md` | Counter-audit | ✅ Accurate, with status updates |
| `QUANT_REVIEW.md` | Historical review | ⚠️ References deleted modules |
