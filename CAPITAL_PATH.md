# Workstream 3: Capital Growth & Deployment Path

## Current Equity Baseline
**Starting Capital:** ₹15,000 INR

## Capital Tiers & System Unlocks
The PROMETHEUS architecture automatically adapts risk parameters, position sizing, and trade management based on available capital.

### Tier 0: Startup (< ₹30,000)
* **Status:** Current
* **Unlocks/Restrictions:** 
  * Max 1 concurrent position.
  * Capital efficiency override: Forces 1-strike OTM selection.
  * Extended time stop: 7 daily bars for breathing room on smaller deltas.

### Intermediate Unlock: The Multi-Position Threshold (₹30,000)
* **Unlocks/Restrictions:**
  * Unlocks up to 2 concurrent positions (allows holding two uncorrelated setups simultaneously).

### Tier 1: Standard Deployment (₹50,000)
* **Unlocks/Restrictions:**
  * Removes forced 1-strike OTM requirement (ATM strikes accessible).
  * Time stop tightened to 6 daily bars (faster cut of stagnant trades).
  * Minimum viable capital for 2.5:1 Risk/Reward without heavy slippage drag.

### Tier 2: Scaled Deployment (₹1,00,000)
* **Unlocks/Restrictions:**
  * Time stop tightened to 5 daily bars (maximum capital velocity).
  * Multi-lot scaling activated based on dynamic equity sizing.
  * Drawdown throttle operates smoothly without risking account ruin on a single 1-lot string of losses.

### Tier 3: Full Capacity (₹1,50,000+)
* **Unlocks/Restrictions:**
  * System nears structural capacity limit (₹2L) before slippage models degrade.
  * Optimized Kelly Criterion sizing fully expressed across multiple lots.

---

## Scenario Projections (Monthly Infusion Paths)

*Timeline assumes pure savings injection without compounding trading returns (conservative path).*

### Scenario A: ₹5,000 / month
* **Month 0:** ₹15,000
* **Month 3:** ₹30,000 *(Unlocks 2 Concurrent Positions)*
* **Month 7:** ₹50,000 *(Hits Tier 1 - ATM strikes, 6-bar time stop)*
* **Month 17:** ₹1,00,000 *(Hits Tier 2 - 5-bar time stop, multi-lot)*
* **Month 27:** ₹1,50,000 *(Hits Tier 3)*

### Scenario B: ₹10,000 / month (Recommended Baseline)
* **Month 0:** ₹15,000
* **Month 2:** ₹35,000 *(Unlocks 2 Concurrent Positions)*
* **Month 4:** ₹55,000 *(Hits Tier 1 - ATM strikes, 6-bar time stop)*
* **Month 9:** ₹1,05,000 *(Hits Tier 2 - 5-bar time stop, multi-lot)*
* **Month 14:** ₹1,55,000 *(Hits Tier 3)*

### Scenario C: ₹20,000 / month (Accelerated)
* **Month 0:** ₹15,000
* **Month 1:** ₹35,000 *(Unlocks 2 Concurrent Positions)*
* **Month 2:** ₹55,000 *(Hits Tier 1 - ATM strikes, 6-bar time stop)*
* **Month 5:** ₹1,15,000 *(Hits Tier 2 - 5-bar time stop, multi-lot)*
* **Month 7:** ₹1,55,000 *(Hits Tier 3)*