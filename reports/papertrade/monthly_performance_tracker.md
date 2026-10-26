# 📊 PROMETHEUS — Performance Tracker & System Ledger

**Tracking Period:** August – October 2026  
**Initial Base Capital:** Rs 15,000  
**Current Realized Net P&L:** **🔴 -Rs 38,566.40 (-257.11%)**  
**Last Updated:** 2026-10-26 21:00:00 IST  

---

## 1. System Performance Overview (Lifetime N=135 vs. Crucible N=78)

### A. Lifetime Performance (August 25 – October 26, 2026)

| Metric | Value | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Net Realized P&L** | **-Rs 38,566.40** | +Rs 5,000 / month | 🔴 Drawdown |
| **Lifetime Return %** | **-257.11%** | +30.0% / month | Drawdown |
| **Total Trading Days** | **41** | 20–22 days | 41 Sessions Tracked |
| **Green / Red Days** | **11 Green / 17 Red / 13 BE** | > 65% Green | 26.8% Green Rate |
| **Total Trades** | **135** | ~3–5 / day | 48W + 87L = 135 |
| **Win Rate** | **35.6%** (48W / 87L) | > 55.0% | ⚠️ Under Review |
| **Gross Profit Factor** | **0.62** | > 1.50 | ⚠️ Under Review |
| **Gross Profit** | +Rs 47,311.35 | — | Winning Trades |
| **Gross Loss** | -Rs 76,774.93 | — | Losing Trades |

### B. Crucible Phase Performance (September 21 – October 26, 2026 — `live_ledger.sqlite`)

| Metric | Value | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Crucible Net P&L** | **-Rs 14,138.15** | Capital Preservation | -14.14% Drawdown |
| **Starting Equity** | **Rs 1,00,000.00** | — | Normalized Capital Base |
| **Current Continuous Equity**| **Rs 85,861.85** | — | Realized Balance |
| **Total Trades (N)** | **78** | — | 27W + 51L = 78 |
| **Net Win Rate** | **34.6%** (27W / 51L) | > 50.0% | Strict Net-over-Gross |
| **Gross P&L** | -Rs 6,121.02 | — | Profit: +Rs 19,893.24 / Loss: -Rs 26,014.26 |
| **Total Statutory Costs** | **Rs 8,017.13** | — | Avg Rs 102.78 / trade |
| **Tier B Net P&L (Runners)**| **🟢 +Rs 2,371.74** | Positive Expectancy | **11W / 20L (Proven Alpha)** |
| **Tier C Net P&L (Scalps)** | **🔴 -Rs 16,509.89**| Paper Shadow Only | **16W / 31L (Friction Drag)** |
| **Short Gamma Squeeze (N=38)**| **Rs -3,910.25** | Quick Breakeven Locking| **Gross +Rs 127.72 / Costs Rs 4,037.97** |

---

## 2. Daily Performance Calendar (Recent Sessions)

| Date | Trades | Result | Win Rate | Gross Profit | Gross Loss | Net P&L | Return % | Running Total | Key Observations & Rule Updates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2026-10-15** | 10 | 2W / 8L | 20% | +Rs 1,618 | -Rs 6,464 | **🔴 -Rs 6,055.7** | -6.4% | **-Rs 36,449.7** | Day 15: 10 trades executed (2 Wins / 8 Losses). SENSEX 4-trade martingale ladder (-Rs 3,667.27) resolved via directional lockout. |
| **2026-10-16** | 10 | 5W / 5L | 50% | +Rs 2,730 | -Rs 4,368 | **🔴 -Rs 2,581.8** | -2.9% | **-Rs 39,031.5** | Day 16: 10 trades executed (5 Wins / 5 Losses, 50.0% WR). Gross -Rs 1638.16, Costs Rs 943.63, Net -Rs 2581.79. Interrupted trade 8864A5 cleanly settled. |
| **2026-10-19** | 2 | 1W / 1L | 50% | +Rs 373 | -Rs 0 | **🟢 +Rs 126.2** | +0.1% | **-Rs 38,905.3** | Day 17: 2 credit spreads executed (1 Win / 1 Loss, 50.0% WR). FIN NIFTY scratch (-Rs 8.74), BANK NIFTY square-off decay (+Rs 134.92 net). Net Green session (+Rs 126.18). |
| **2026-10-21** | 0 | 0W / 0L | 0% | +Rs 0 | -Rs 0 | **⚪ +Rs 0.0** | +0.0% | **-Rs 38,905.3** | Intraday paper trading session |
| **2026-10-23** | 2 | 1W / 1L | 50% | +Rs 520 | -Rs 90 | **🟢 +Rs 332.7** | +0.4% | **-Rs 38,572.6** | Day 18: 2 trades executed (1 Win / 1 Loss, 50.0% WR). SENSEX 73100PE target hit (+Rs 336.33 net), SENSEX 72900PE trailed BE scratch (-Rs 3.65 net). Net Green session (+Rs 332.68). |
| **2026-10-26** | 2 | 1W / 1L | 50% | +Rs 201 | -Rs 0 | **🟢 +Rs 6.2** | +0.01% | **-Rs 38,566.4** | Day 19: 2 trades executed (1 Win / 1 Loss, 50.0% WR). SENSEX 73300CE trailed BE (+Rs 6.65 net), NIFTY 22650CE trailed BE (-Rs 0.41 net friction). Squeeze micro-locks saved -Rs 1,064 in MAE drawdown. Unified statutory buffers recalibrated (NIFTY >= 2.00, SENSEX/BN >= 5.50, FIN >= 2.60). Net Green session (+Rs 6.24). |

---

## 3. Data File Locations

* **SQLite Primary Ledger:** [`reports/papertrade/live_ledger.sqlite`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/live_ledger.sqlite)
* **CSV Raw Ledger:** [`reports/papertrade/daily_performance_ledger.csv`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/daily_performance_ledger.csv)
* **Live Trades CSV:** [`reports/papertrade/live_ledger.csv`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/live_ledger.csv)
* **Markdown Tracker:** [`reports/papertrade/monthly_performance_tracker.md`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/monthly_performance_tracker.md)
