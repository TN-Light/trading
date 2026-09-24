# 🏛️ Real Indian Index Derivatives Expiry Calendar (SEBI Rationalization Truth)

> **Mandatory Reference for Operators and Future AI Chat Sessions**  
> **Effective Regulatory Framework:** SEBI Circular `SEBI/HO/MRD/MRD-PoD-2/P/CIR/2024/134` (implemented September 1, 2025).

---

## 1. The Core Regulatory Law: "One Weekly Index Per Exchange"

To curb speculative retail frenzy across daily zero-day options, SEBI mandated that each exchange may offer **weekly derivative contracts on only ONE benchmark index**.

| Exchange | Authorized Weekly Index | Weekly Expiry Day | Monthly Expiry Day |
| :--- | :--- | :---: | :---: |
| **NSE (National Stock Exchange)** | **NIFTY 50** | **TUESDAY** | Last Tuesday of Month |
| **BSE (Bombay Stock Exchange)** | **BSE SENSEX** | **THURSDAY** | Last Thursday of Month |

### Discontinued Weekly Expiries (Monthly Contracts Only)
The following contracts **NO LONGER HAVE WEEKLY EXPIRIES**:
- **BANK NIFTY**: Weekly options discontinued. Only Monthly contracts exist (expire on the last Tuesday of the month).
- **FINNIFTY**: Weekly options discontinued. Only Monthly contracts exist (expire on the last Tuesday of the month).
- **MIDCAP NIFTY**: Weekly options discontinued. Only Monthly contracts exist (expire on the last Tuesday of the month).
- **BANKEX**: Weekly options discontinued. Only Monthly contracts exist (expire on the last Thursday of the month).

---

## 2. Weekly Market Schedule (Monday through Friday)

| Day of Week | Primary Index Focus | Expiry Status | Trading Character |
| :--- | :--- | :---: | :--- |
| **MONDAY** | NIFTY 50 & General Indices | **1-DTE Setup** | Directional trends & positioning ahead of Tuesday NIFTY expiry. |
| **TUESDAY** | **NIFTY 50** | **🔥 0-DTE WEEKLY EXPIRY** | High gamma, rapid theta decay, 0-DTE option rules strictly apply. |
| **WEDNESDAY** | BSE SENSEX & General Indices | **1-DTE Setup** | SENSEX positioning, mid-week volume consolidation. |
| **THURSDAY** | **BSE SENSEX** | **🔥 0-DTE WEEKLY EXPIRY** | BSE Monthly & Weekly Expiry. High gamma breakdown/squeeze day. |
| **FRIDAY** | All Benchmark Indices | **NORMAL SESSION (NO EXPIRY)** | **ZERO index options expire on Friday.** New contract cycle begins. |

---

## 3. Why AI Models Hallucinate & How to Prevent It

### The Root Cause of AI Hallucinations:
1. **Pre-2025 Training Bias**: Prior to the SEBI rationalization circular, BSE SENSEX expired on Friday, Bank Nifty expired on Wednesday, and FinNifty expired on Tuesday. Large language models frequently regress to this pre-2025 training distribution.
2. **Failure to Ground in Code**: Prometheus already has the true schedule mathematically encoded in `prometheus/utils/indian_market.py`:
   ```python
   WEEKLY_EXPIRY_DAYS = {
       "NIFTY 50": "Tuesday",
       "SENSEX": "Thursday",
   }
   ```

### Non-Negotiable Operational Directives:
1. **Never assert that Friday is SENSEX expiry.** Friday is a regular, non-expiry session.
2. **Never assert that Monday is SENSEX expiry.** Thursday is SENSEX weekly expiry.
3. **Today (Thursday, September 24, 2026) was BSE SENSEX Weekly & Monthly Expiry (0-DTE).**
4. **Tomorrow (Friday, September 25, 2026) is a Regular Non-Expiry Session (Day 5).**
