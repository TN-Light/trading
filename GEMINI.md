# Quantitative Engineering & System Integrity Guidelines

## 1. Zero-Hallucination & Empirical Verification Mandate
- **No Blind Agreement**: Never agree to or implement discretionary changes to stop losses, trailing ladders, or targets without rigorous mathematical justification and empirical testing against historical ledger trades (`reports/papertrade/live_ledger.sqlite`).
- **Telemetry Verification**: When a metric reads zero or abnormal (e.g. `commitment_ratio: 0.00`), inspect the underlying broker API payload (`AngelOne`, `Kite`) before drawing conclusions about market participant behavior. Never assume market flow from missing dictionary keys.
- **Traceable Attribution**: Every price level, target, stop loss, and P&L figure cited must originate directly from verified log files (`logs/prometheus.log`) or the SQLite ledger.

## 2. Microstructure & Trailing Stop Invariants
- **Noise Envelope Preservation**: Bank Nifty and Sensex options carry a persistent 3–8 pt bid-ask spread and 10–15 pt 1-minute candle noise. Never move stop losses inside this noise envelope for high-conviction trend setups (Tier S / Tier B), as this truncates the right-tail runner distribution.
- **Tier-Differentiated Policy**:
  - **Tier C (Scalps / Chop / Counter-Trend)**: Aggressive micro-locking at $\ge 12.0$ pts gain ($\text{SL} = \text{Entry} + \text{Cost Buffer}$) to harvest chop alpha and protect capital against mean reversion.
  - **Tier S / Tier B (Trend Runners)**: Strictly preserve wide breathing room ($\ge 18.0$ pt Bank Nifty / $\ge 20.0$ pt Sensex breakeven threshold) to capture multi-ATR runners.
- **Single-Lot Rule**: Sizing is strictly capped at 1 lot (`max_lots_per_trade: 1`). Because partial profit-booking is impossible on 1 lot, runner preservation is mathematically essential to offset inevitable stop-outs.

## 3. Regime-Aware Target Calibration
- **Macro Alignment**: If higher-timeframe (1H) trend is `NEUTRAL` or conflicting with a 15-minute breakout, or if Dealer GEX is strongly positive (`LONG_GAMMA`), target gain must be compressed to scalp boundaries (Bank Nifty: 20–30 pts, Sensex: 22–32 pts, Nifty: 8–14 pts) and retest expansion bypassed.
- **Uncompressed Runners**: Full multi-ATR targets are reserved exclusively for macro-aligned Tier S and Tier B setups.

## 4. Execution Telemetry & Ledger Standards
- **Wall-Clock Duration**: Trade entry and exit timestamps must use `datetime.now(IST)` so that `holding_duration_seconds = max(1, int((exit_time - entry_time).total_seconds()))` reliably reflects true elapsed time rather than 0 seconds.
- **Re-Entry Lockouts**: Intraday symbol lockouts must prevent re-trading closed contracts on the same day.
