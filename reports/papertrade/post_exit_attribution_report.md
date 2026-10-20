# Post-Exit Performance Attribution & Forensic Report
**Generated:** 2026-10-09 11:21:47 IST  
**Database:** `reports/papertrade/live_ledger.sqlite`

---

## 1. Executive Summary

| Forensic KPI | Metric Value | Benchmark Standard | Operational Verdict |
| :--- | :--- | :--- | :--- |
| **Total Observed Closed Trades** | **0** | N >= 30 | Accumulating Sample |
| **Stop Loss Exits** | **0** | — | — |
| **Premature Shakeout Rate** | **0.0%** (0 trades) | < 25.0% | 🟢 Healthy (SL outside market noise) |
| **Target Exits** | **0** | — | — |
| **Avg Alpha Left Beyond Target** | **+0.0 pts** | < 15.0 pts | 🟢 Well Calibrated |
| **Capital Preserved by SL** | **~₹0.00** | > 0 | 🟢 Net Capital Defense Positive |
| **Inactivity Kill Invalidation** | **0.0%** | < 20.0% | 🟢 Clean Stagnation Harvest |

---

## 2. Core Quantitative Findings

1. **Premature Shakeout Attribution**:
   - Total Stop-Loss trades that subsequently reversed to hit Target or reclaim Entry: **0 out of 0 (0.0%)**.
   - If this rate is < 25%, the current stop loss placement is mathematically validated and successfully protecting capital against trend continuation.

2. **Runner Preservation & Target Calibration**:
   - Average post-exit expansion beyond Target: **+0.0 pts** (Peak: **+0.0 pts**).
   - Demonstrates whether dynamic multi-ATR targets are capturing sufficient runner alpha.

3. **Inactivity Kill-Switch Reliability**:
   - Liquidations that subsequent broke out to target: **0.0%**.
   - Validates that killing stagnant option buys after 45m avoids theta cremation without forfeiting late breakouts.
