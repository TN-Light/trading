# Walkthrough: Spot-Anchored ORB Retest Stop-Loss & Lunch Trend-Day Bypass

We have implemented and verified the two quantitative enhancements requested to ensure Prometheus captures massive intraday breakouts and survives routine retest shakeouts while maintaining strict single-lot position sizing (`max_lots_per_trade: 1`).

---

## What Was Implemented

### 1. Spot-Anchored ORB Retest Stop-Loss (`prometheus/main.py`)
- **Root Problem Solved**: On Bank Nifty today, Trade #1 entered at `1055.05` on spot `54,778.90` (ORB High: `54,625.75`). At 10:30 AM, spot dipped -165 points to `54,613.45` to retest the breakout line. An artificial cap (`target_gain_pts * 1.5`) clamped the stop-loss to 75.9 points (`979.10`), stopping out the trade by only 6 option points before the market rallied +500 points.
- **The Solution**:
  - Defined instrument-calibrated ORB retest buffers:
    - **NIFTY BANK**: $\max(25.0, 0.30 \times \text{ATR})$
    - **SENSEX**: $\max(35.0, 0.30 \times \text{ATR})$
    - **NIFTY 50**: $\max(12.0, 0.30 \times \text{ATR})$
  - Calculated `spot_sl_level` anchored below the ORB High (`54,625.75 - 26.4 = 54,599.35`).
  - Converted the spot risk (`179.55 pts`) to option points via delta (`89.8 pts`).
  - **Dynamic Target Expansion**: If `structural_sl_pts > target_gain_pts * 1.2`, target gain is scaled up (`target_gain_pts >= structural_sl_pts * 1.2`) instead of clipping the SL inside the retest zone.
  - **Result**: Stop-loss price is placed at `965.25`. During the 10:30 retest, option low hit `979.10` (staying 13.85 points above SL), completely surviving the shakeout and riding the entire rally to target.

### 2. Institutional Trend-Day Lunch Bypass
- **Indicators (`prometheus/signals/technical.py`)**:
  - Implemented Wilder's `calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series` to quantify directional trend strength.
- **Scanner (`prometheus/signals/price_action_momentum.py`)**:
  - Checks if day qualifies as an **Institutional Trend Day**:
    1. Active ORB Breakout (High or Low).
    2. $\text{ADX}(14) \ge 25.0$.
    3. Current Volume $\ge 1.5\times \text{20-SMA Volume}$ (or index expansion $> 1.5\times \text{ATR}$).
  - Allows bar evaluation during the midday window (`11:30 <= current_time <= 13:15`) when `is_institutional_trend_day` is True.
- **Tier Classifier (`prometheus/signals/tier_classifier.py`)**:
  - During `is_lunch_dead_zone`, checks `signal.get("is_institutional_trend_day")`.
  - If True, awards **`TIER B: INSTITUTIONAL TREND DAY CONTINUATION`** with `is_live_eligible = True`.
  - Normal choppy days remain 100% strictly blocked as `Tier C (LUNCH DEAD ZONE — PAPER ONLY)`.
- **Main Execution (`prometheus/main.py`)**:
  - In both Lunch Dead Zone gate checks (lines 5104 and 5678), checks `refined.get("is_institutional_trend_day")`.
  - Logs `[Trend-Day Lunch Bypass]` and executes the signal.
- **Settings (`prometheus/config/settings.yaml`)**:
  - Added configurable parameters under `intraday.trend_day_bypass`.

---

## Verification Results

### Automated Test Suite: 47 / 47 Passed (0 Failures)

```
collected 47 items

prometheus/tests/test_trend_day_lunch_bypass_and_orb_sl.py::TestWildersADX::test_adx_computation_range PASSED
prometheus/tests/test_trend_day_lunch_bypass_and_orb_sl.py::TestWildersADX::test_adx_flat_market_low PASSED
prometheus/tests/test_trend_day_lunch_bypass_and_orb_sl.py::TestSpotAnchoredORBRetestSL::test_banknifty_retest_buffer_survival PASSED
prometheus/tests/test_trend_day_lunch_bypass_and_orb_sl.py::TestLunchDeadZoneTrendDayBypass::test_tier_classifier_promotes_trend_day_during_lunch PASSED
prometheus/tests/test_trend_day_lunch_bypass_and_orb_sl.py::TestLunchDeadZoneTrendDayBypass::test_tier_classifier_strictly_blocks_normal_day_during_lunch PASSED
prometheus/tests/test_trend_day_lunch_bypass_and_orb_sl.py::TestLunchDeadZoneTrendDayBypass::test_scanner_evaluates_bar_during_lunch_when_trend_day_active PASSED
prometheus/tests/test_dynamic_target_sl_calibration.py (4 tests) PASSED
prometheus/tests/test_price_action_momentum.py (7 tests) PASSED
prometheus/tests/test_tier_classifier.py (16 tests) PASSED
prometheus/tests/test_intraday_windows_and_trade_limits.py (10 tests) PASSED
prometheus/tests/test_trend_aware_inactivity_kill_switch.py (4 tests) PASSED

============================= 47 passed in 3.78s ==============================
```
