# 📖 PROMETHEUS — Permanent Engineering & Trading Logbook

> **Operational Principle:** This file is an **append-only audit journal**. Never delete or overwrite past entries. Every day, add new sections below with timestamps, trade results, observed market dynamics, features implemented, and parameter changes.

---

## 📅 Entry 1: Tuesday, August 25, 2026

### 1. Market Regime & Macro Context
* **India VIX:** 11.19 - 11.45 (Ultra-low volatility regime).
* **Expiry Day:** NIFTY 50 & NIFTY MIDCAP SELECT weekly expiry (0DTE).
* **Price Dynamics:** Morning sharp bearish breakdown (10:15) -> violent V-reversal short covering (11:45) -> flat horizontal consolidation (12:00 - 14:20) -> explosive afternoon gamma breakout (14:30 - 14:55).

---

### 2. Trade Execution Summary
* **Base Capital:** Rs 15,000
* **Total Trades Executed:** 4
* **Gross Profit:** +Rs 335.00
* **Gross Loss:** -Rs 1,250.00
* **Net Realized P&L:** **🔴 -Rs 915.00 (-6.10%)**

#### Detailed Trade Logs:
1. **NIFTY 24150 PE (Trade 1):** Entry @ Rs 32.65 -> Reached Rs 42.00 (+28.6% peak gain) -> Trailing stop locked at Rs 37.10 -> Closed with **+Rs 335.00 profit**.
2. **NIFTY 24150 PE (Trade 2 - Re-entry):** Entry @ Rs 35.75 -> Morning V-reversal hit -20% hard SL -> Closed with **-Rs 465.00 loss**.
3. **MIDCAP 14875 PE (Trade 3):** Entry @ Rs 31.00 -> Hit -20% hard SL -> Closed with **-Rs 651.00 loss**.
4. **SENSEX 77800 PE (Trade 4 - Scale test):** Closed with **-Rs 134.00 loss**.

---

### 3. Key Quantitative Learnings
1. **Low VIX Breakdown Fragility:** Breakouts during VIX < 12 reverse rapidly because market lacks institutional trend momentum.
2. **Averaging Down Hazard:** Taking a 2nd signal on the same strike after a loss increases risk concentration without edge.
3. **15-Min Lag on 0DTE Expiry:** Zero-day option gamma moves double in 5-8 minutes; waiting for a 15-minute candle close causes late entries at the top of spikes.
4. **The 2:30 PM Hard Cutoff Conflict:** The default 14:30 entry cutoff shut off scanning right before the explosive 2:30-3:00 PM expiry power hour rally.

---

### 4. Upgrades Implemented & Deployed (All Active in Production)

* **Upgrade 1: Higher Momentum Pyramiding & Loss-Blocking Gate (commit 79fe31b)**
  * *Rule:* Never re-enter a losing strike; require Edge Score >= 5.0 and positive P&L to scale into winners. Resets daily.
* **Upgrade 2: 7-Character DDMONYY Token Parser Fix (commit b94f7d0)**
  * *Rule:* Fixed Angel One token parsing (MIDCPNIFTY25AUG2614875PE -> MIDCPNIFTY 14875 PE) with single-tap Kite copy box.
* **Upgrade 3: Actionable Real-Time Stop-Loss Trigger Updates (commit 77637e3)**
  * *Rule:* Dispatches live Kite trigger values (Old SL -> New SL) to Telegram on every trailing milestone.
* **Upgrade 4: Automated Persistent Daily CSV Ledger & Monthly Dashboard (commit e619276)**
  * *Rule:* Auto-updates daily_performance_ledger.csv and compiles monthly_performance_tracker.md at market close without resetting past history.
* **Upgrade 5: Option A - Adverse VWAP Structural Fast-Exit (commit 2cce630)**
  * *Rule:* Evaluates every 60s; if index spot crosses back over VWAP, exits immediately to cap losses at -5% to -8% instead of full -20%.
* **Upgrade 6: Option B - Low-VIX Regime Adaptive Mode (commit 70463bd)**
  * *Rule:* When VIX < 12, automatically switches to +22% scalp targets, moves Breakeven at +8% gain, and locks +8% at +14% gain.
* **Upgrade 7: Option C - Expiry Power Hour Extension & 3-Min Fast Scans (commit 2c8ae88)**
  * *Rule:* On weekly expiry sessions (Tuesdays/Thursdays), extends entry cutoff to **15:05 (3:05 PM)** and scans every **180s (3 minutes)** between 13:30 and 15:05.

---
*(Next trading day entry will be appended below)*
## 📅 Entry 2: Wednesday, August 26, 2026 (Morning Session)

### 1. Market Context & Alerts
* **Opening Range High:** NIFTY 50 morning high established at 24,363.20.
* **Barbell / Credit Spread Strategy Fix:** User received a sideways Bear Call Spread signal. Discovered that multi-account paper trade candidate router was omitting credit spreads, and Telegram copy strings lacked explicit weekly expiry dates.
* **Upgrades Deployed:**
  * **Credit Spread Paper Execution:** Multi-account candidate builder now supports 2-leg credit spreads so Barbell trades open and track P&L seamlessly in paper trading.
  * **Kite 1-Tap Copy Strings with Weekly Expiry:** Formats explicit weekly dates (NIFTY 28 AUG 24200 PE and NIFTY 28 AUG 24050 PE) so searching on Zerodha Kite always pulls up the exact matching ₹37 / ₹12 weekly contracts without monthly expiry mismatch.
  * **Kite Basket Order Instructions:** Clear Telegram execution guidance added: Buy hedge leg first (margin discount) -> Sell main leg.
  * **PaperCapture Credit Spread Execution Fix:** Mapped top-level `entry_price` = `net_credit`, `strike` = `short_strike`, `stop_loss` = `hard_sl_price`, `target` = `target_decay_price` on `Hedged_Credit_Spread` signals, completely eliminating the `PAPER CAPTURE — signal skipped (no entry_price hint)` error.
  * **Live Market Option Chain Connected to Credit Spreads:** Replaced the offline fallback heuristic formula (`37.5` / `12.0`) with real-time Angel One SmartAPI live option chain quotes (`get_real_premium`), guaranteeing Telegram alerts and paper execution reflect the exact live market premiums (e.g. ₹67 & ₹138).
  * **SENSEX Friday Expiry Date Fix:** Corrected SENSEX weekly expiry day from Thursday to **Friday (`2026-08-28`)** to match BSE contract listings, ensuring Kite copy-paste searches always find the live contract.
  * **Strict Policy — Zero Mathematical Fallbacks:** Permanently banned all offline mathematical formulas (ATR/strike-width approximations and Black-Scholes estimates) across the entire engine. If live market option LTPs cannot be fetched from Angel One API, the signal is strictly discarded (`return None`) with an explicit warning log.
  * **Option Buying Execution Fix & Dual-Regime Prioritization:** Fixed an indentation bug in `main.py` that trapped Option Buying signals inside an `else:` block, completely restoring live `BUY_CE` / `BUY_PE` momentum breakout alerts. Configured concurrent dual-regime evaluation: directional breakouts (momentum score >= 3.5) take top priority #1, while sideways range markets trigger Hedged Credit Spreads (priority #2).
  * **Global Cross-Symbol Leaderboard & Shadow Paper Trading:** Deployed batch candle scanning across all 4 indices. Candidate signals are aggregated and sorted by edge conviction: Rank #1 executes on the primary trading account, while Rank #2 and Rank #3 are automatically paper-traded in the shadow engine (`paper_capture.on_signal`) with live P&L and trailing stops so no signal is dropped or lost.
  * **System Error Fix (`execution_signal` UnboundLocalError):** Initialized `execution_signal = None` and `cs_sig = None` at the top of `_get_intraday_signal_for_execution`, completely resolving the runtime error when evaluating Credit Spreads without momentum breakouts.

### 2. End-of-Day Trade Performance & Forensic Breakdown
* **Total Trades Recorded:** 12 trades
* **Total Realized Net P&L:** **+₹4,691.88** (Overall Profitable Day)
* **Overall Win Rate:** 66.7% (8 Wins / 4 Losses)

#### Strategy Breakdown:
| Strategy | Trades | Win Rate | Net P&L | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Hedged Credit Spreads (Selling)** | 8 | **100.0% (8W / 0L)** | **+₹11,199.25** | Perfect capture of Theta decay in sideways consolidation (NIFTY & SENSEX). |
| **PriceAction Momentum (Buying)** | 4 | **0.0% (0W / 4L)** | **-₹6,507.37** | False breakouts chopped out due to extreme low volatility (VIX = 10.65). |

#### Critical Engineering Learnings:
1. **The Barbell Architecture Validated:** Hedged Credit Spreads saved the day and generated +₹11,199 in pure profits while pure option buying struggled in low-volatility chop.
2. **Low-VIX Regime Gate:** When VIX < 11.0, breakout velocity is suppressed. Option buying requires higher momentum threshold (Score $\ge$ 4.5) to avoid false expansion entries.
3. **Monthly Expiry Nominal Risk Guard:** Large monthly contracts (e.g. BankNifty @ ₹924 LTP) must have strict capital allocation caps to prevent single-trade oversize risk.

---
*(Next trading day entry will be appended below)*
## 📅 Entry 3: Thursday, August 27, 2026 (Morning Session)

### 1. Market Context & System Upgrades
* **Telegram Clutter Elimination & Unified Signal Ranking:**
  * Embedded `🥇 RANK #1 SIGNAL (PRIMARY EXECUTION)` and `🥈 RANK #2 SIGNAL (SHADOW PAPER TRADED)` directly into the primary signal alert header.
  * Eliminated redundant follow-up messages (`RANK #1 TRADE EXECUTED`, `SHADOW SIGNAL OBSERVED`, and raw `PAPER CAPTURE opened`), reducing Telegram message volume from 6+ messages to exactly 1 clean, actionable alert per candidate.
* **Kite Monthly vs Weekly Search String Resolution:**
  * Updated `human_search_name` in `symbol_format.py` so that monthly contracts omit the day (e.g. `SENSEX AUG 77400 PE` / `NIFTY AUG 24250 PE`), matching Zerodha Kite's search index 100%.
  * Weekly contracts retain the explicit day (e.g. `SENSEX 21 AUG 77400 PE` / `NIFTY 28 AUG 24250 PE`).
* **Live Pricing Transparency:**
  * Renamed all Telegram label references from `Entry hint` to `Live Entry LTP (Angel One)` to ensure complete user clarity that execution prices are 100% real live market quotes from Angel One SmartAPI with zero fallback formulas.

### 2. End-of-Day Trade Performance & Forensic Breakdown
* **Total Trades Recorded:** 6 trades
* **Total Realized Net P&L:** **-₹4,458.05**
* **Overall Win Rate:** 0.0% (0 Wins / 6 Losses)

#### Trade Log:
| Trade ID | Symbol | Contract | Strategy | Entry $\rightarrow$ Exit | Net P&L | Return % | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PAPER-FEB6D7` | **NIFTY 50** | `24200 PE` | PriceAction Momentum | ₹86.10 $\rightarrow$ ₹73.18 | -₹840.84 | -15.0% | square_off / SL |
| `PAPER-95F9E7` | **NIFTY 50** | `24200 PE` | PriceAction Momentum | ₹87.55 $\rightarrow$ ₹74.42 | -₹854.50 | -15.0% | square_off / SL |
| `PAPER-EBA4C3` | **NIFTY 50** | `24200 PE` | PriceAction Momentum | ₹86.00 $\rightarrow$ ₹73.10 | -₹839.53 | -15.0% | square_off / SL |
| `PAPER-4D0422` | **NIFTY 50** | `24200 PE` | PriceAction Momentum | ₹85.35 $\rightarrow$ ₹72.55 | -₹833.03 | -15.0% | square_off / SL |
| `PAPER-1D6885` | **NIFTY 50** | `24150 PE` | PriceAction Momentum | ₹75.45 $\rightarrow$ ₹64.13 | -₹736.71 | -15.0% | square_off / SL |
| `PAPER-5A1CF7` | **SENSEX** | `77400 PE` | PriceAction Momentum | ₹117.70 $\rightarrow$ ₹100.05 | -₹353.44 | -15.0% | square_off / SL |

#### Critical Engineering Learnings:
1. **The Midday Mean-Reversion Squeeze (12:30 - 13:30):**
   * NIFTY made an initial morning drop from 24,201 to 24,133, triggering 5 bearish PE breakdown trades.
   * Between 12:30 PM and 1:15 PM, NIFTY staged a sharp counter-trend pullback back up to 24,199 (+65 points), hitting the tight $-15\%$ stop loss on all PE positions before collapsing back down to 24,021 by 3:15 PM.
2. **Pyramiding Without Profit-Locking:**
   * 5 consecutive trades entered the exact same instrument (`NIFTY01SEP2624200PE`) at nearly identical prices (₹85 - ₹87) within 45 minutes. When the midday mean-reversion squeeze occurred, all 5 hit SL simultaneously for $-₹840 \times 5 = -₹4,104$ (92% of the day's loss).
3. **Actionable Rule — Strike-Level Lockout:**
   * Once a position is opened in a specific option contract, subsequent entries in the SAME exact strike must be blocked unless the existing position is already sitting in $\ge +10\%$ profit and risk is moved to Break-Even.

### 3. Engine Upgrades Deployed Post-Session (Commit 72-Test Verified)
1. **Low-VIX Dynamic Conviction Filter (`main.py:2060`):**
   * Automatically checks India VIX at each scan cycle.
   * When $\text{India VIX} < 11.5$, Option Buying momentum score threshold is raised from $3.5$ to **$4.5+$** (only true explosive volume expansion triggers buying).
   * Range-bound signals automatically fall back to **Hedged Credit Spreads** to capture Theta decay.
2. **Max Nominal Capital Exposure Cap (`main.py:2070`):**
   * Single-lot option premium allocation is hard-capped at **₹15,000**.
   * Expensive monthly contracts (such as BankNifty @ ₹924 LTP with ₹27,720 lot cost) are safely suppressed from dominating account risk.
3. **Same-Strike Lockout & Profit-Locked Pyramiding (`main.py:2090`):**
   * Block repeat entries on the same strike if the active trade is sitting in $< +10\%$ profit.
   * Scale-in is strictly permitted only when the existing trade has achieved $\ge +10\%$ gain with risk moved to Break-Even.
   * Strong Signal Override: High conviction setups (Score $\ge 5.0$) are permitted to scale-in dynamically.

---

## 📅 Entry 4: Friday, August 28, 2026

### 1. Market Context & Macro Regime
* **India VIX:** 10.76 - 10.78 (Extreme low-volatility consolidation regime).
* **Expiry Day:** SENSEX Weekly Expiry (0DTE).
* **Price Action Dynamics:**
  * **Morning (09:15 - 11:00 AM):** Flat horizontal consolidation within a narrow 28-point range (NIFTY 24,141 - 24,169).
  * **Midday (11:15 - 13:15 PM):** Slow low-volume drift to 24,076 followed by flat 2-hour consolidation.
  * **Afternoon (13:30 - 15:15 PM):** Sharp counter-trend short-covering squeeze; NIFTY rallied +80 points from 24,095 to 24,175, and SENSEX surged +228 points from 77,036 to 77,264 into market close.

---

### 2. Full-Day Trade Performance & Forensic Breakdown
* **Total Trades Recorded:** 6 trades
* **Total Realized Net P&L:** **🟢 +₹1,850.39 (PROFITABLE SESSION)**
* **Overall Win Rate:** 33.3% (2 Wins / 4 Losses)

#### Trade Log:
| Trade ID | Symbol | Contract | Strategy | Entry $\rightarrow$ Exit | Net P&L | Return % | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PAPER-D23A8B` | **NIFTY 50** | `24200CE / 24350CE` | Hedged Credit Spread | ₹63.90 $\rightarrow$ ₹95.85 | **+₹2,075.71** | **+50.0%** | square_off (Max Decay) |
| `PAPER-684054` | **NIFTY 50** | `24200CE / 24350CE` | Hedged Credit Spread | ₹71.90 $\rightarrow$ ₹107.85 | **+₹2,335.58** | **+50.0%** | square_off (Max Decay) |
| `PAPER-AE4368` | **NIFTY 50** | `24100 PE` | PriceAction Momentum | ₹68.50 $\rightarrow$ ₹58.23 | -₹668.37 | -15.0% | square_off / SL |
| `PAPER-58C16D` | **NIFTY 50** | `24100 PE` | PriceAction Momentum | ₹64.25 $\rightarrow$ ₹54.61 | -₹627.37 | -15.0% | square_off / SL |
| `PAPER-AA4785` | **NIFTY 50** | `24100 PE` | PriceAction Momentum | ₹64.20 $\rightarrow$ ₹54.57 | -₹626.72 | -15.0% | square_off / SL |
| `PAPER-0F84F0` | **NIFTY 50** | `24100 PE` | PriceAction Momentum | ₹65.40 $\rightarrow$ ₹55.59 | -₹638.44 | -15.0% | square_off / SL |

#### Performance by Strategy:
| Strategy | Trades | Win Rate | Net P&L | Strategic Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Hedged Credit Spreads (Selling)** | 2 | **100.0% (2W / 0L)** | **+₹4,411.29** | Flawless Theta capture in low-VIX consolidation; both positions achieved 50% target decay. |
| **PriceAction Momentum (Buying)** | 4 | **0.0% (0W / 4L)** | **-₹2,560.90** | Afternoon breakdown reversed due to Friday expiry short-covering squeeze; stopped out at -15%. |

---

### 3. Critical Quantitative Insights:
1. **The Barbell Engine Proves Its Profitability Again:**
   * In a choppy session where Option Buying lost -₹2,560, the Hedged Credit Spread strategy generated **+₹4,411 in pure Theta profit**, delivering a **net profitable green day (+₹1,850.39)** for the portfolio.
2. **Cumulative Weekly Credit Spread Record:**
   * Wednesday + Friday Credit Spreads: **10 Trades, 10 Wins (100% Win Rate), +₹15,610.54 Total Profit**.
   * Demonstrates that Option Selling in low-VIX environments is our highest-expectancy edge.
3. **Friday Expiry Afternoon Dynamics:**
   * After 13:30 PM on expiry days, breakdown moves frequently trigger violent short-covering squeezes back to VWAP. Option Buying entries after 13:30 on low-VIX days require heightened momentum confirmation to prevent late-session whip-saws.

---

## 📅 Entry 5: Monday, August 31, 2026

### 1. Market Context & Macro Regime
* **India VIX:** 11.22 - 11.25 (Persistent low-volatility environment).
* **Expiry Day:** Non-expiry session.
* **Price Action Dynamics:**
  * **Morning (09:15 - 10:45 AM):** Sharp morning bearish impulse down to Opening Range Low (SENSEX low 76,842, NIFTY low 24,013) triggering Put Buying signals.
  * **Midday (11:00 AM - 13:15 PM):** Aggressive counter-trend V-reversal short covering; SENSEX rallied **+180 points** (76,842 $\rightarrow$ 77,022) and NIFTY surged **+67 points** (24,013 $\rightarrow$ 24,080).
  * **Afternoon (13:30 - 15:15 PM):** Slow drift into close; all positions exited at automated 15:15 square-off.

---

### 2. Full-Day Trade Performance & Forensic Breakdown
* **Total Trades Recorded:** 4 trades
* **Total Realized Net P&L:** **🔴 -₹4,459.89**
* **Overall Win Rate:** 0.0% (0 Wins / 4 Losses)

#### Trade Log:
| Trade ID | Symbol | Contract Traded | Strategy | Entry $\rightarrow$ Exit Price | Net P&L | Return % | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PAPER-488B02` | **SENSEX** | `SENSEX26SEP76800PE` | PriceAction Momentum | ₹575.70 $\rightarrow$ ₹489.35 | **-₹1,729.13** | -15.0% | square_off / SL |
| `PAPER-4F5CE1` | **SENSEX** | `SENSEX26SEP76800PE` | PriceAction Momentum | ₹565.90 $\rightarrow$ ₹481.01 | **-₹1,699.89** | -15.0% | square_off / SL |
| `PAPER-477757` | **NIFTY 50** | `NIFTY01SEP2624000PE` | PriceAction Momentum | ₹53.15 $\rightarrow$ ₹45.18 | **-₹518.69** | -15.0% | square_off / SL |
| `PAPER-5901FF` | **NIFTY 50** | `NIFTY01SEP2624000PE` | PriceAction Momentum | ₹52.45 $\rightarrow$ ₹44.58 | **-₹512.18** | -15.0% | square_off / SL |

#### Performance by Symbol:
| Symbol | Trades | Win Rate | Gross Loss | Net Realized P&L | % of Day's Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SENSEX (Monthly Contract)** | 2 | 0.0% | -₹3,429.02 | <font color="#ef4444">**-₹3,429.02**</font> | **76.9%** |
| **NIFTY 50 (Weekly Contract)** | 2 | 0.0% | -₹1,030.87 | <font color="#ef4444">**-₹1,030.87**</font> | **23.1%** |

---

### 3. Critical Quantitative Learnings & Actionable Upgrades:
1. **The Monthly Option Risk Asymmetry:**
   * SENSEX was traded using a deep monthly contract @ ₹575 LTP (Lot Cost ₹11,514), causing **77% of the day's total loss** across just 2 trades (-₹3,429).
   * In contrast, NIFTY weekly options @ ₹53 LTP (Lot Cost ₹3,445) risked only ₹515 per trade.
   * *Actionable Rule:* Prioritize weekly option expiries for intraday momentum buying whenever available, and cap single-contract premium to under ₹250 LTP for index options.
2. **The Consecutive-Bar Double Entry Trap:**
   * At 10:22 & 10:36 (SENSEX) and 10:51 & 11:06 (NIFTY), the strong-signal override allowed 2 entries on consecutive 15-minute candles into the exact same strike at identical price levels.
   * *Actionable Rule:* Enforce a minimum **30-minute spacing cooldown** between repeat entries on the same underlying symbol even when Score $\ge 5.0$, preventing immediate double-allocation before the trade develops.
3. **Midday Mean-Reversion Filter in Low-VIX Regimes:**
   * When India VIX < 11.5, morning breakouts between 10:15 and 11:15 have a high failure rate due to lack of volume follow-through.

---
*(Next trading day entry will be appended below)*
