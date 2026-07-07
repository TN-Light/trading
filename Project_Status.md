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
  - gravity_clearance: 15%, time_decay_edge: 10%, macro_flow: 5%
- **Gates**: Confluence (≥3/5), Compression Coil / EMA21 Retest, AES threshold, TVI (theta vulnerability), Gamma Ambush (Thursdays 10:45–11:15)
- **Sizing**: Edge-tier bounded (`get_bounded_sizing()`)
- **Supporting modules**: `qrd_estimator.py` (QRD regime), `strike_gravity.py`, `expiry_clock.py`
- **Config**: `intraday.enabled: true`, `intraday.use_backtest_generator: true`

### 2. Swing/Pro Engine (research/backtest)
- **Entry**: `signals/fusion.py` + `signals/technical.py`
- **Technical stack**: VWAP, Volume Profile, FVG, Liquidity Sweeps, OTE, RSI Divergence, SuperTrend (ICT/SMC concepts)
- **Scoring**: Weighted confluence sum (hardcoded weights in fusion.py)
- **Sizing**: Risk-based (`_size_position()`)
- **Config**: `swing.use_backtest_generator: true`

> **Important**: Performance numbers from swing backtests (61.3% WR, 12.55 PF from `prometheus_audit_v2.md`) apply to the swing engine only, NOT to APEX. APEX performance is in `reports/apex_yearly_*.md`.

---

## Core Files (current, verified July 2026)

### Signals
| File | Purpose |
|------|---------|
| `prometheus/signals/technical.py` | VWAP, Volume Profile, FVG, Liquidity Sweeps, OTE, RSI, ATR, SuperTrend, EMA |
| `prometheus/signals/apex_generator.py` | APEX 5-component intraday signal generator |
| `prometheus/signals/aes_fusion.py` | AES 6-factor edge scoring (0–100) |
| `prometheus/signals/fusion.py` | Weighted confluence engine for swing/pro signals |
| `prometheus/signals/regime_detector.py` | AMD/Parrondo regime classification |
| `prometheus/signals/oi_analyzer.py` | PCR, Max Pain, OI support/resistance, IV skew |
| `prometheus/signals/qrd_estimator.py` | Quantum Regime Detection for APEX |
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
| `prometheus/main.py` | CLI orchestrator (~9000 lines) |

---

## Deleted Files (cleaned up June–July 2026)
- `signals/cross_asset_relay.py` — dead code, zero imports
- `risk/loss_elimination_engine.py` — dead code, zero imports
- `intelligence/signal_regression.py` — dead code, zero imports

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
| Entry window | 09:30–14:15 |
| Dead zone | 12:00–13:00 |
| `intraday.v2.max_daily_trades` | 5 |
| `intraday.v2.time_stop_bars` | 16 (APEX reads from config) |

### Paper Mode Risk Overrides (currently active)
Paper mode uses relaxed limits (`100%` thresholds, `150%` position size) for uninterrupted statistics collection. **Base risk limits must be verified before live deployment.**

---

## Operational Notes
- **Swing-15m** is the locked execution path for paper/live
- The legacy `analyze_intraday()` path is being consolidated (`use_backtest_generator=True` unconditionally)
- Walk-forward optimizer (`run_apex_optimizer.py`) validates 4 hyperparameters with **two-sided OOS check** (rejects both degradation AND suspiciously good OOS)
- Backtest CAGR numbers should NOT be used for forward planning (strategy design hill-climbing caveat)

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
