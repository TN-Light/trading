# Quantitative Engineering & System Integrity Guidelines

## 1. Zero-Hallucination & Empirical Verification Mandate
- **Zero-Tolerance for Convincing-Sounding Speculation (Reproducible Proof Standard)**: Plausible or well-written explanations without executed code verification are strictly forbidden. The operator has zero tolerance for unverified analysis masquerading as proof. Every diagnostic statement, P&L claim, or strategy review MUST include the exact, runnable Python/SQLite command and raw database output that proves it. If an analysis cannot be verified via raw database output, it must be explicitly declared as unverified.
- **No Blind Agreement**: Never agree to or implement discretionary changes to stop losses, trailing ladders, or targets without rigorous mathematical justification and empirical testing against historical ledger trades (`reports/papertrade/live_ledger.sqlite`).
- **Telemetry Verification**: When a metric reads zero or abnormal (e.g. `commitment_ratio: 0.00`), inspect the underlying broker API payload (`AngelOne`, `Kite`) before drawing conclusions about market participant behavior. Never assume market flow from missing dictionary keys.
- **Traceable Attribution**: Every price level, target, stop loss, and P&L figure cited must originate directly from verified log files (`logs/prometheus.log`) or the SQLite ledger.
- **Mandatory Direct DB Querying (Anti-Stale Citation)**: Never cite historical statistics, win rates, or P&L figures from older Markdown reports or text summaries. All statistical metrics MUST be queried live from `reports/papertrade/live_ledger.sqlite` across the full active dataset (`SELECT COUNT(*)`).
- **Net-Over-Gross Standard**: All trade analyses, win rates, and expectancy evaluations must be calculated on `net_pnl` after statutory transaction fees, never on `gross_pnl`.
- **Full-Population Accounting**: When evaluating subsets (e.g. DTE buckets or strategy performance), query the entire dataset and group by bucket (`GROUP BY`), verifying that subset counts sum to the total population. Cherry-picking winning subsets is strictly prohibited.
- **No Score-Only Live Promotion**: Never promote a setup to live broker execution (Tier S / Tier B) purely based on a high confluence score (e.g. Score >= 8.0). Confluence score measures 15-minute point-in-time momentum alignment, not multi-day risk or impulse exhaustion; in `live_ledger.sqlite`, Score >= 8.0 setups carry a 63.3% net loss rate (31 losses out of 49 trades, Net P&L: -₹8,762.69). Live eligibility strictly requires structural multi-dimensional confirmation (1H macro alignment, valid DTE, and non-climax entry).
- **Asymmetric GEX Gate Invariant**: In positive Dealer GEX regimes ($\text{Net GEX} > 0$ / Long Gamma), Call Buying (`LONG`) is strictly prohibited from live broker execution (Tier S / Tier B) and demoted to Tier C Paper Only. When dealers are long call gamma, delta hedging forces them to dump index futures into every upside breakout, resulting in institutional supply absorption and an empirical 12.5% net win rate (1W / 7L, -₹5,881.83 net across 8 historical trades). Conversely, Put Buying (`SHORT`) fading range ceilings in positive GEX is permitted for Tier B (Score >= 8.0) because dealer inventory unwinding and mean-reversion rejections produce positive expectancy (5W / 3L, 62.5% net win rate, +₹3,135.84 net across 8 historical trades).

## 2. Microstructure & Trailing Stop Invariants
- **Noise Envelope Preservation**: Bank Nifty and Sensex options carry a persistent 3–8 pt bid-ask spread and 10–15 pt 1-minute candle noise. Never move stop losses inside this noise envelope for high-conviction trend setups (Tier S / Tier B), as this truncates the right-tail runner distribution.
- **Tier-Differentiated Policy**:
  - **Tier C (Scalps / Chop / Counter-Trend)**: Aggressive micro-locking at $\ge 12.0$ pts gain ($\text{SL} = \text{Entry} + \text{Cost Buffer}$) to harvest chop alpha and protect capital against mean reversion.
  - **Tier S / Tier B (Trend Runners)**: Strictly preserve wide breathing room ($\ge 18.0$ pt Bank Nifty / $\ge 20.0$ pt Sensex breakeven threshold) to capture multi-ATR runners.
- **Single-Lot Rule**: Sizing is strictly capped at 1 lot (`max_lots_per_trade: 1`). Because partial profit-booking is impossible on 1 lot, runner preservation is mathematically essential to offset inevitable stop-outs.
- **Single-Lot Statutory Cost Floor**: A breakeven trailing stop must never use an arbitrary flat buffer (e.g. 1.5 pts). It must equal or exceed the maximum observed round-trip statutory fee hurdle plus buffer (Sensex >= 4.50 pts, Bank Nifty >= 5.20 pts, Nifty >= 1.40 pts) so that a breakeven exit never realizes a negative net rupee return.

## 3. Regime-Aware Target Calibration
- **Macro Alignment**: If higher-timeframe (1H) trend is `NEUTRAL` or conflicting with a 15-minute breakout, or if Dealer GEX is strongly positive (`LONG_GAMMA`), target gain must be compressed to scalp boundaries (Bank Nifty: 20–30 pts, Sensex: 22–32 pts, Nifty: 8–14 pts) and retest expansion bypassed.
- **Uncompressed Runners**: Full multi-ATR targets are reserved exclusively for macro-aligned Tier S and Tier B setups.
- **DTE-Gated Credit Spread Policy**: Across 23 credit spreads in `live_ledger.sqlite`, net P&L is -₹7,705.78 (Gross: -₹4,857.50, Costs: ₹2,848.28, 8W / 15L, 34.8% net win rate). Monthly (8–14 DTE) contracts carry negligible intraday theta (< 0.15 pts/hr) while requiring ₹60k–₹70k margin per lot. Monthly credit spreads are strictly confined to Tier C (Paper Only / Statistical Tracking) and barred from live broker execution until forward testing across >= 20 trades demonstrates >= 65% win rate.

## 4. Execution Telemetry & Ledger Standards
- **Wall-Clock Duration**: Trade entry and exit timestamps must use `datetime.now(IST)` so that `holding_duration_seconds = max(1, int((exit_time - entry_time).total_seconds()))` reliably reflects true elapsed time rather than 0 seconds.
- **Directional Martingale Elimination**: Intraday symbol lockouts must not merely lock the specific option token; once a position hits a stop-loss, the engine must lock out the entire `(symbol, direction)` pair (e.g. SENSEX BEAR_CALL_SPREAD) for the remainder of the session to eliminate rolling-strike martingale ladders (which cost -₹3,667.27 on Oct 15 across 4 trades, and -₹18,941.45 across all multi-stop sessions).
- **Inactivity Kill Invariant**: The inactivity kill-switch must never liquidate an active trend runner based solely on candle counts without verifying wall-clock time (>= 40 minutes) and active technical breakdown (loss of SuperTrend or EMA 9x21). Liquidations under the legacy 3-bar rule destroyed -₹10,156.11 across 14 trades (averaging -₹725.44 per trade). Defined-risk credit spreads are permanently exempt from inactivity kill switches.

## 5. Algorithmic Integrity & Anti-Hallucination Governance
- **Root-Cause Analysis of Past Analytical Failures**:
  1. *Sub-Population Sampling Bias*: Quoting intermediate subsets (e.g. 47 trades from an interim report or 18 trades from a partial script) instead of querying the full SQLite population ($N=74$), leading to distorted win rates.
  2. *Confirmation Bias & Selective Reporting*: Highlighting winning subsets (e.g. "0-1 DTE spreads: 4/4 wins") while omitting the full multi-week reality (11 trades, 4W/7L, -₹3,513.34).
  3. *Gross vs. Net Obfuscation*: Counting trades as "wins" or "non-losing" based on gross points while ignoring statutory brokerage fees (₹80–₹125/trade) that flipped them into net rupee losses.
  4. *Synthetic Script Overrides*: Embedding hardcoded trade outcomes in simulation or replay scripts (e.g. `2BDB15` in `replay_trailing_stop_models.py`) rather than computing true deterministic exits.
- **The 4-Stage Pre-Commit Verification Gate**:
  - Before writing or citing ANY performance claim, diagnostic metric, or ledger analysis:
    1. **Direct DB Execution**: Execute an explicit `sqlite3` query against `reports/papertrade/live_ledger.sqlite`.
    2. **Conservation of Population**: Assert that $\sum \text{subsets} = N_{\text{total}}$ and $\text{Wins} + \text{Losses} = N_{\text{total}}$.
    3. **Net Rupee Primacy**: Compute all metrics on `net_pnl`. Report gross metrics strictly alongside statutory transaction costs (`costs`).
    4. **Automated Test Certification**: Ensure all changes are verified by automated tests in `prometheus/tests/`.
- **Zero-Manipulation Invariant**: Hardcoding trade outcomes, synthetic price fills, or manual parameter overrides in replay, backtesting, or tracking scripts is strictly prohibited and treated as a critical system defect.
