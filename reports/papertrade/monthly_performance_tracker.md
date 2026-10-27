# 📊 PROMETHEUS — Performance Tracker & System Ledger

**Tracking Period:** August – October 2026  
**Initial Base Capital:** Rs 15,000  
**Current Realized Net P&L:** **🔴 -Rs 40,651.46 (-271.01%)**  
**Last Updated:** 2026-10-27 20:30:00 IST  

---

## 1. System Performance Overview (Lifetime N=139 vs. Crucible N=82)

### A. Lifetime Performance (August 25 – October 27, 2026)

| Metric | Value | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Net Realized P&L** | **-Rs 40,651.46** | +Rs 5,000 / month | 🔴 Drawdown |
| **Lifetime Return %** | **-271.01%** | +30.0% / month | Drawdown |
| **Total Trading Days** | **42** | 20–22 days | 42 Sessions Tracked |
| **Green / Red Days** | **11 Green / 18 Red / 13 BE** | > 65% Green | 26.2% Green Rate |
| **Total Trades** | **139** | ~3–5 / day | 49W + 90L = 139 |
| **Win Rate** | **35.3%** (49W / 90L) | > 55.0% | ⚠️ Under Review |
| **Gross Profit Factor** | **0.60** | > 1.50 | ⚠️ Under Review |
| **Gross Profit** | +Rs 47,467.36 | — | Winning Trades |
| **Gross Loss** | -Rs 78,657.87 | — | Losing Trades |

### B. Crucible Phase Performance (September 21 – October 27, 2026 — `live_ledger.sqlite`)

| Metric | Value | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Crucible Net P&L** | **-Rs 16,223.21** | Capital Preservation | -16.22% Drawdown |
| **Starting Equity** | **Rs 1,00,000.00** | — | Normalized Capital Base |
| **Current Continuous Equity**| **Rs 83,776.79** | — | Realized Balance |
| **Total Trades (N)** | **82** | — | 28W + 54L = 82 |
| **Net Win Rate** | **34.1%** (28W / 54L) | > 50.0% | Strict Net-over-Gross |
| **Gross P&L** | -Rs 7,847.95 | — | Profit: +Rs 21,322.31 / Loss: -Rs 29,170.26 |
| **Total Statutory Costs** | **Rs 8,375.26** | — | Avg Rs 102.14 / trade |
| **Tier A Net P&L (Spreads)** | **🔴 -Rs 412.48** | Statistical Tracking | **0W / 1L (0-DTE Gamma Spike)** |
| **Tier B Net P&L (Runners)**| **🟢 +Rs 623.01** | Positive Expectancy | **11W / 22L (Positive Edge)** |
| **Tier C Net P&L (Scalps)** | **🔴 -Rs 16,433.74**| Paper Shadow Only | **17W / 31L (Friction Drag)** |
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
| **2026-10-27** | 4 | 1W / 3L | 25% | +Rs 156 | -Rs 1,883 | **🔴 -Rs 2,085.1** | -2.43% | **-Rs 40,651.5** | Day 20: 4 trades executed (1 Win / 3 Losses, 25.0% WR). Gross -Rs 1,726.93, Costs Rs 358.13, Net -Rs 2,085.06. FINNIFTY 24500CE (Tier C) banked +Rs 76.15 net. NIFTY 22750CE (-Rs 878.60 net) & SENSEX 73600CE (-Rs 870.13 net) Tier B calls stopped out into +102k Cr Long Gamma dealer absorption. FINNIFTY Bear Call Spread (-Rs 412.48 net) Tier A stopped out on 0-DTE gamma spike. |

---

## 3. Data File Locations

* **SQLite Primary Ledger:** [`reports/papertrade/live_ledger.sqlite`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/live_ledger.sqlite)
* **CSV Raw Ledger:** [`reports/papertrade/daily_performance_ledger.csv`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/daily_performance_ledger.csv)
* **Live Trades CSV:** [`reports/papertrade/live_ledger.csv`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/live_ledger.csv)
* **Markdown Tracker:** [`reports/papertrade/monthly_performance_tracker.md`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/monthly_performance_tracker.md)
