# PROMETHEUS 2.0 — Architecture & Data Flow (August 2026)

## High-Level Components

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                              │
│  AngelOne (live LTP/bid-ask/Greeks) · Kite Connect · yfinance    │
│  NSE Direct (options chain)         · CSV overrides              │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │ DataEngine │  (data/engine.py + data/store.py)
                    │  IST norm  │  SQLite persistence, Angel One extended intraday
                    └─────┬─────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
 ┌──────────────────────┐    ┌──────────────────────┐
 │  REGIME A: TREND     │    │  REGIME B: SIDEWAYS  │
 │  (35% Breakout Days) │    │  (65% Range Days)    │
 ├──────────────────────┤    ├──────────────────────┤
 │ PriceActionMomentum  │    │ CreditSpreadStrategy │
 │ ORB + VWAP + SuperTr │    │ Bull Put / Bear Call │
 ├──────────────────────┤    ├──────────────────────┤
 │ BUY ATM/ITM Option   │    │ SELL Defined Spread  │
 └──────────┬───────────┘    └──────────┬───────────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                    ┌─────▼─────┐
                    │   Risk     │  (risk/manager.py — 13 checks)
                    │   Manager  │  NON-BYPASSABLE, contract-level deduplication
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ Execution  │  OrderManager → PaperTrader / KiteExecutor
                    └─────┬─────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
   ┌────────▼────────┐         ┌────────▼────────┐
   │ Position Monitor │         │  Telegram Bot    │
   │ 5-stage trailing │         │  CLI Dashboard   │
   │ Inverted decay   │         │  Real-time Marks │
   └─────────────────┘         └─────────────────┘
```

---

## Signal Path 1: APEX Intraday (Live/Paper)

**Files**: `apex_generator.py` → `aes_fusion.py`, supported by `qrd_estimator.py`, `strike_gravity.py`, `expiry_clock.py`

### Per-Bar Flow:
1. **DataEngine** appends new 15-min bar → normalized OHLCV
2. **Technical indicators** computed: VWAP, Session VWAP, EMA 9/21, SuperTrend, RSI
3. **Confluence scoring** — 5 components scored as bull/bear:

| # | Component | Bull Condition | Bear Condition |
|---|-----------|---------------|----------------|
| 1 | VWAP | close > vwap | close < vwap |
| 2 | Session VWAP | close > session_vwap | close < session_vwap |
| 3 | EMA 9/21 | ema9 > ema21 | ema9 < ema21 |
| 4 | SuperTrend | direction == 1 | direction == -1 |
| 5 | RSI Div / EMA cross | bullish div or bull cross ≤3 bars | bearish div or bear cross ≤3 bars |

4. **Confluence gate**: bull_score ≥ 3 AND bear_score < 2 (or inverse for PE)
5. **Compression gate**: Coil ratio < 0.35 OR EMA21 retest within 3 bars
6. **QRD regime**: `qrd_estimator.py` classifies regime state
7. **AES scoring**: 6-factor weighted edge score (0–100):
   - regime_alignment (30%), signal_confluence (25%), volatility_support (15%)
   - gravity_clearance (15%), time_decay_edge (10%), macro_flow (5%)
   - Non-linear boost for scores > 75: `+(score - 75) * 0.2`, capped at 100
8. **AES threshold gate**: score must pass minimum edge threshold
9. **TVI (Theta Vulnerability Index)**: Rejects if theta decay alone can fire the SL within `max_bars`
10. **Option pricing**: Black-Scholes computes premium, delta, strike selection
    - Live mode: BS estimate overridden by Angel One LTP
11. **Dual-trigger SL**: ATR-based underlying SL + premium decay SL (40% catastrophic floor)
12. **Signal dict emitted** with: action, strike, entry_price, stop_loss, target, delta, max_bars, edge_score, aes_factors, regime, strategy

### Special Gates:
- **Gamma Ambush** (Thursdays 10:45–11:15): OTM 1–3 strikes, premium ₹15–50, VWAP directional gate
- **Entry window**: 09:30–14:15 (configurable)
- **Dead zone**: 12:00–13:00 (configurable)
- **Time stop**: max_bars from config (`intraday.v2.time_stop_bars: 16`)

---

## Signal Path 2: Swing/Pro (Research/Backtest)

**Files**: `technical.py` → `oi_analyzer.py` → `regime_detector.py` → `fusion.py` → `strategies/selector.py` → `strategies/trend.py|volatility.py|expiry.py`

### Per-Bar Flow:
1. **DataEngine** provides historical OHLCV window
2. **Technical signals** computed: VWAP, Volume Profile, FVG, Liquidity Sweeps, OTE, RSI Divergence, SuperTrend, EMA (ICT/SMC methodology)
3. **OI analysis**: PCR, Max Pain, OI support/resistance, IV skew, sentiment
4. **Regime detection**: AMD/Parrondo classification (markup, markdown, accumulation, distribution, volatile)
5. **Fusion scoring**: Weighted confluence sum across technicals, OI, regime, AI sentiment
   - Hardcoded weights (e.g., volume_profile: 0.85, VWAP: 0.80)
   - Outputs `FusedSignal` (0–10 score) with direction, R:R gate, entry/SL/target
6. **Strategy selection**: Regime → strategy routing via `selector.py`
   - Markup/markdown → `trend.py` (multi-timeframe alignment, OI walls as veto)
   - Accumulation/distribution → mean-reversion
   - Volatile → skip
7. **Trade setup** built with: symbol, instrument_type, entry_price, stop_loss, target, quantity, delta, max_bars, option_expiry_date

---

## Shared Downstream: Risk → Execution → Monitoring

### Risk Manager (`risk/manager.py`)
- 13 pre-trade checks, **non-bypassable** (no bypass path exists in code)
- Checks: system halt, trading hours, daily/weekly loss, trade counts, max positions, sizing, consecutive losses, drawdown halt, correlated exposure, duplicate instrument
- Paper mode uses relaxed overrides for statistics collection

### Execution Layer
- **OrderManager** (`execution/order_manager.py`): Trade lifecycle, pre-trade risk gating, multi-leg order build
- **PaperTrader** (`execution/paper_trader.py`): Realistic fills using Angel One bid/ask, Zerodha cost model (STT, GST, brokerage, slippage), SL/trigger simulation
- **KiteExecutor** (`execution/kite_executor.py`): Zerodha Kite Connect order placement, LTP/margins fetch, rate limiting (0.5s min interval)

### Position Monitor (`execution/position_monitor.py`)
**3-Phase Premium SL Floor** (controls when SL is enforced):
- Phase 1 (≤3 bars): SL immune — absorb IV crush noise
- Phase 2 (4–5 bars): SL at 80% buffer
- Phase 3 (>5 bars): Full SL enforcement

**5-Stage Trailing Ratchet** (profit protection):
- Stage 0: Premium hits 0.4R → breakeven trap (SL to entry + 0.10R)
- Stage 1: Premium hits 1.0R → lock 20% profit
- Stage 2: Premium hits 2.0R → lock 50% profit
- Stage 3: Premium hits 3.0R → lock 70% + init HWM
- Stage 4: Ongoing → dynamic trail: 30% below HWM, floor at 0.70R

**Time stop**: Configurable per timeframe. Currently 16 bars × 15min = 4 hours.

### Cost Model (backtest + paper)
- Commission (₹20/order), STT, GST (18%), stamp duty
- Slippage: 0.15% default (configurable)
- Premium evolution: `dP = delta×dS + 0.5×gamma×dS² - theta×dt`
- Gamma approximation: `gamma = abs(delta) * (1 - abs(delta)) * gamma_scale`

---

## Interface & Reporting
- **CLI Dashboard** (`interface/cli_dashboard.py`): Real-time signal count, positions, PnL
- **Telegram Bot** (`interface/telegram_bot.py`): Two-way alerts, semi-auto confirmation, proxy/SNI fallbacks
- **Reports**: Walk-forward results, yearly breakdowns, trade CSVs under `reports/`

---

## Key File References

| Layer | Files |
|-------|-------|
| Data | `data/engine.py`, `data/store.py` |
| APEX signals | `signals/apex_generator.py`, `signals/aes_fusion.py`, `signals/qrd_estimator.py`, `signals/strike_gravity.py`, `signals/expiry_clock.py` |
| Swing signals | `signals/technical.py`, `signals/fusion.py`, `signals/oi_analyzer.py`, `signals/regime_detector.py` |
| Strategies | `strategies/trend.py`, `strategies/volatility.py`, `strategies/expiry.py`, `strategies/selector.py` |
| Risk | `risk/manager.py`, `risk/position_sizer.py`, `risk/portfolio_scaler.py` |
| Execution | `execution/order_manager.py`, `execution/paper_trader.py`, `execution/kite_executor.py`, `execution/position_monitor.py` |
| Config | `config/settings.yaml`, `config/credentials.yaml` |
| Utils | `utils/indian_market.py`, `utils/options_math.py`, `utils/logger.py` |
| Backtest | `backtest/engine.py` |

### Deleted (dead code, removed June–July 2026):
- ~~`signals/cross_asset_relay.py`~~
- ~~`risk/loss_elimination_engine.py`~~
- ~~`intelligence/signal_regression.py`~~

---

## Config-Driven Behavior
Most behavior is driven by `prometheus/config/settings.yaml`. Key sections:
- `intraday.*` — APEX pipeline settings (enabled, instruments, entry/dead zone times, max trades)
- `intraday.v2.*` — V2 profile settings (time_stop_bars, max_daily_trades, tier profiles)
- `swing.*` — Swing pipeline settings
- `risk.*` — Hard risk limits (daily/weekly loss, max positions, consecutive losses)
- `signals.*` — Signal thresholds (min_confluence, min_rr)
- `paper.*` — Paper mode risk overrides (relaxed for stats collection)
- `ai.*` — AI features (currently all disabled)
