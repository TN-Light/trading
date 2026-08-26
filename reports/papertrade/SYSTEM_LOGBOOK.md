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

---
*(Next trading day entry will be appended below)*
