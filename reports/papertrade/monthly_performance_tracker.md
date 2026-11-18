# 📊 PROMETHEUS — Monthly Performance Tracker

**Tracking Period:** November 2026  
**Initial Base Capital:** Rs 100,000  
**Current Realized Net P&L:** **🟢 +Rs 146.50 (+0.15%)**  
**Last Updated:** 2026-11-18 18:36:53 IST  

> [!IMPORTANT]
> **Strict Capital Segregation Enforced (GEMINI.md §7)**:  
> **Operational Live Capital** is governed strictly by live-eligible setups (`tier IN ('S', 'A', 'B')`).  
> **Experimental Sandbox Trades (`tier = 'C'`)** run 24/7 in paper mode for telemetry discovery and **NEVER** bleed into operational capital, drawdowns, or daily risk locks.

---

## 1. Operational Live Capital Performance (Tier S, A, B)

| Metric | Value | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Operational Live Net P&L** | **+Rs 146.50** | +Rs 5,000 / month | 🟢 |
| **Operational Return %** | **+0.15%** | +30.0% / month | Active |
| **Operational Equity** | **Rs 100,146.50** | Base Rs 100,000 | Capital Protected |
| **Total Live Trades** | **1** | High-Conviction Tiers Only | Tracked |
| **Live Win Rate** | **100.0%** (1W / 0L) | > 55.0% | ✅ Above Target |
| **Live Max Drawdown** | **0.00%** | < 5.00% | Zero Capital Impairment |

---

## 2. Live Capital Daily Performance Log

| Date | Live Trades | W / L | Win Rate | Live Net P&L | Return % | Live Equity | Execution & Strategy Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2026-11-16** | 0 | 0W / 0L | 0% | **⚪ +Rs 0.0** | +0.00% | Rs 100,000 | Clean Room Session Open: 0 Live setups triggered; 3 Tier C paper tests |
| **2026-11-17** | 1 | 1W / 0L | 100% | **🟢 +Rs 146.5** | +0.15% | Rs 100,146 | SENSEX Long Tier B target hit (+Rs 146.50 net); 8 Tier C paper tests |
| **2026-11-18** | 0 | 0W / 0L | 0% | **⚪ +Rs 0.0** | +0.00% | Rs 100,146 | 0 Live setups (5 trapped setups demoted to Tier C by Trap Gates; Rs 0 live loss) |

---

## 3. Experimental Sandbox Telemetry (Tier C — Quarantined Telemetry)

> [!NOTE]
> Tier C represents exploratory paper testing setups (e.g. Asymmetric GEX Long Gamma call buys, marginal ORB tests, low-DTE spreads).
> All Tier C executions are **quarantined from real and paper operational capital** (Capital At Risk: **Rs 0.00**).

| Metric | Sandbox Value | Operational Live Capital Impact | Status |
| :--- | :---: | :---: | :---: |
| **Total Sandbox Trades** | **16** | 0 Live Orders Sent | Quarantined |
| **Sandbox Win Rate** | **50.0%** (8W / 8L) | N/A | Telemetry Only |
| **Sandbox Net P&L** | **Rs -931.46** | **Rs 0.00 (Zero Capital Bleed)** | Protected |

### Sandbox Daily Breakdown

| Date | Sandbox Trades | W / L | Win Rate | Sandbox Net P&L | Microstructure Notes & Traps Filtered |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **2026-11-16** | 3 | 1W / 2L | 33% | **🔴 -Rs 584.3** | Clean Room Session Open: 0 Live setups triggered; 3 Tier C paper tests |
| **2026-11-17** | 8 | 7W / 1L | 88% | **🟢 +Rs 2,408.3** | SENSEX Long Tier B target hit (+Rs 146.50 net); 8 Tier C paper tests |
| **2026-11-18** | 5 | 0W / 5L | 0% | **🔴 -Rs 2,755.5** | 0 Live setups (5 trapped setups demoted to Tier C by Trap Gates; Rs 0 live loss) |

---

## 4. Combined Population Reconciliation (GEMINI.md Full-Population Accounting)

| Population | Trades | Wins | Losses | Win Rate | Net P&L | Capital Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Operational Live (Tier S, A, B)** | 1 | 1 | 0 | 100.0% | **+Rs 146.50** | **Realized to Equity** |
| **Sandbox Telemetry (Tier C)** | 16 | 8 | 8 | 50.0% | Rs -931.46 | **Rs 0.00 (Segregated)** |
| **Total Ledger Population** | 17 | 9 | 8 | 52.9% | Rs -784.96 | — |

---

## 5. Data File Locations

* **CSV Raw Daily Ledger:** [`reports/papertrade/daily_performance_ledger.csv`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/daily_performance_ledger.csv)
* **Markdown Tracker:** [`reports/papertrade/monthly_performance_tracker.md`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/monthly_performance_tracker.md)
* **SQLite Master Ledger:** [`reports/papertrade/live_ledger.sqlite`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/live_ledger.sqlite)