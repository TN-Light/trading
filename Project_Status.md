# Project Status — PROMETHEUS 2.0 (snapshot: August 2026)

## Overview
- **Purpose**: Indian F&O Barbell Dual-Regime System (Intraday Momentum Buying + Hedged Credit Spreads)
- **Capital**: ₹15K–2L INR (adaptive bracket sizing)
- **Broker**: AngelOne (primary data feed + options chain), Zerodha Kite Connect (execution)
- **Mode**: Paper (current), targeting Live
- **Instruments**: NIFTY 50, NIFTY BANK, SENSEX, NIFTY MIDCAP SELECT
- **Languages/libs**: Python; pandas, numpy, scipy, yfinance, kiteconnect, sqlite, requests, loguru, rich

---

## Architecture — Barbell Dual-Regime Engine (August 2026)

### 1. Momentum Option Buyer (Trending Regimes)
- **Entry**: `signals/price_action_momentum.py` → `_get_intraday_signal_for_execution()` in `main.py`
- **Technical Stack**: Opening Range Breakout (ORB: 09:15–09:45 box), VWAP Position & Slope, SuperTrend (10, 3), Consolidation Squeeze Breakouts
- **Execution**: BUY ATM/ITM Option (CE / PE) with high gamma acceleration
- **Target**: Quick Target 1 (+18% to +22% premium gain), Target 2 (+35% to +50% runner)
- **Protection**: Fast Breakeven Lock at +10% gain, Stagnation Exit after 4 bars (60 min) flat, SuperTrend Adverse Cut

### 2. Hedged Credit Spread Seller (Sideways / Range Regimes)
- **Entry**: `strategies/credit_spread.py` → `_get_intraday_signal_for_execution()` in `main.py`
- **Strategy**: Bull Put Spreads (Sell 1-OTM PE + Buy 3-OTM PE Hedge) & Bear Call Spreads (Sell 1-OTM CE + Buy 3-OTM CE Hedge)
- **Defined Risk**: Hedge leg executed first, securing SEBI margin relief (~₹35,000–₹45,000 per lot)
- **Decay Trailing**: Inverted Trailing Stop (50% decay locks breakeven, 70% decay triggers take-profit exit, 1.5x credit hard SL)

### 3. Session Timing & Noise Gate
- **09:15–09:50 AM**: Range Formation Gate (Observes ORB high/low, avoids 37.9% WR open chop)
- **09:50–11:45 AM**: Morning Momentum Window
- **11:45–13:00 PM**: Lunchtime Deadzone (No new directional entries)
- **13:00–14:30 PM**: Afternoon Continuation Window
- **15:15 PM**: Mandatory Intraday Square-Off

---

## Core Files (August 2026)

### Signals & Strategies
| File | Purpose | Status |
|------|---------|--------|
| `prometheus/signals/price_action_momentum.py` | ORB, VWAP, SuperTrend momentum breakout scanner | ACTIVE (Primary Trend) |
| `prometheus/strategies/credit_spread.py` | Hedged Bull Put / Bear Call credit spreads | ACTIVE (Primary Sideways) |
| `prometheus/signals/technical.py` | VWAP, SuperTrend, EMA, ATR, Volume Profile | ACTIVE |
| `prometheus/signals/regime_detector.py` | AMD/Parrondo regime classification | ACTIVE |
| `prometheus/strategies/selector.py` | Barbell Dual-Regime strategy routing | ACTIVE |
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
