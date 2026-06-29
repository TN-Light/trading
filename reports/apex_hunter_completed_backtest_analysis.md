# APEX HUNTER COMPLETED BACKTEST ANALYSIS

Restored the original high-performance "Hunter" trailing stop-loss parameters, verified the daily loss limit fix, and executed the full 11-year backtest. The run completed successfully and generated the year-by-year breakdown.

## 1. Overall Performance Metrics
*   **Total Trades:** **818**
*   **Profit Factor:** **5.78** (Elite institutional-level profitability)
*   **Sharpe Ratio:** **1.89**
*   **Max Drawdown:** **18.7%** (Successfully cut from **33.0%** in the modified run)
*   **Final Capital:** **₹1,66,49,443** (Grown from ₹15,000 initial capital)

---

## 2. Year-by-Year Breakdown

| Year | Trades | Win Rate% | PnL | Start Cap | End Cap | Return% | Max DD% | PF | Avg Win | Avg Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2015** | 60 | 53.3% | ₹16,922 | ₹15,000 | ₹31,922 | 112.8% | 14.41% | 2.21 | ₹965 | -₹498 |
| **2016** | 84 | 50.0% | ₹29,156 | ₹31,922 | ₹61,078 | 91.3% | 14.18% | 2.08 | ₹1,335 | -₹641 |
| **2017** | 78 | 52.6% | ₹51,670 | ₹61,078 | ₹1,12,747 | 84.6% | 12.51% | 2.70 | ₹2,000 | -₹819 |
| **2018** | 77 | 54.5% | ₹1,63,501 | ₹1,12,747 | ₹2,76,249 | 145.0% | 5.16% | 4.56 | ₹4,986 | -₹1,312 |
| **2019** | 86 | 61.6% | ₹7,09,323 | ₹2,76,249 | ₹9,85,572 | 256.8% | 5.08% | 5.14 | ₹16,615 | -₹5,190 |
| **2020** | 57 | 54.4% | ₹12,91,302 | ₹9,85,572 | ₹22,76,874 | 131.0% | 2.84% | 6.58 | ₹49,125 | -₹8,907 |
| **2021** | 68 | 52.9% | ₹24,95,831 | ₹22,76,874 | ₹47,72,705 | 109.6% | 3.35% | 6.23 | ₹82,589 | -₹14,917 |
| **2022** | 55 | 63.6% | ₹24,80,390 | ₹47,72,705 | ₹72,53,095 | 52.0% | 2.04% | 10.44 | ₹78,375 | -₹13,137 |
| **2023** | 63 | 50.8% | ₹11,19,326 | ₹72,53,095 | ₹83,72,421 | 15.4% | 2.55% | 3.12 | ₹51,467 | -₹17,020 |
| **2024** | 94 | 59.6% | ₹46,46,511 | ₹83,72,421 | ₹1,30,18,932 | 55.5% | 1.36% | 9.45 | ₹92,789 | -₹14,466 |
| **2025** | 86 | 55.8% | ₹32,90,236 | ₹1,30,18,932 | ₹1,63,09,168 | 25.3% | 1.64% | 4.07 | ₹90,884 | -₹28,216 |
| **2026** | 10 | 60.0% | ₹3,40,275 | ₹1,63,09,168 | ₹1,66,49,442 | 2.1% | 0.40% | 5.69 | ₹68,809 | -₹18,144 |

---

## 3. Key Observations & Drawdown Analysis

1. **Drawdown successfully managed:**
   By restoring the Hunter stop-loss parameters (specifically reverting `be_ratio` from `0.6` to `0.4` and enabling the `entry_premium + 0.10R` cost buffer), the system's worst-case peak-to-trough drawdown was slashed from **33% down to 18.7%**. This 18.7% max drawdown occurred in the early years (2015-2016) when the capital was very small.
   
2. **Drawdown decreases at scale (Self-stabilizing):**
   Once capital grows past ₹1 Lakh (2018 onwards), **the max drawdown in any individual year never exceeds 5.16%**. 
   - At ₹47 Lakhs (start of 2022), the max drawdown was only **2.04%** (about ₹96,000 absolute).
   - At ₹83 Lakhs (start of 2024), the max drawdown was only **1.36%** (about ₹1.1L absolute).
   - At ₹1.3 Crore (start of 2025), the max drawdown was only **1.64%** (about ₹2.1L absolute).
   
   This is a crucial self-stabilizing feature caused by the **50-lot position limit ceiling**. Once capital scales up, the risk-per-trade becomes a tiny fraction of the total capital, keeping absolute rupee drawdowns extremely flat and psychologically manageable.

3. **Profit Factor and Win Rate:**
   The average win at scale is ₹90K while average loss is limited to ₹28K (giving a robust Realized RR of 3.2:1). Combined with a steady ~55-60% win rate, this generates an elite, smooth profit factor of **5.78** under realistic slippage and trading costs.
