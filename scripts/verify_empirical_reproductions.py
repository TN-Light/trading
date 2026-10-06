"""
Empirical Reproduction & Stress Test Verification Script
Independently verifies the 5 core bug reproduction cases:
1. Credit spread SL arming when spread premium rises
2. Pre-SL tick-0 bailout noise suppression (2-pt dip does NOT trigger)
3. Max pain calculation on asymmetric chains (minimizing buyer payout)
4. NaN health composite score returning 0.0 (preventing min(100, nan) == 100 trap)
5. Commitment ratio clamping in [0.0, 1.0] across extreme bounds
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.order_manager import OrderManager, ManagedPosition
from prometheus.execution.broker import Order, OrderSide, OrderStatus
from prometheus.execution.position_health import PositionHealthEngine, PositionHealthReport
from prometheus.utils.options_math import max_pain
from prometheus.signals.oi_analyzer import OIAnalyzer


def verify_reproduction_1_credit_spread():
    print("=" * 70)
    print("REPRODUCTION 1: Credit Spread SL Arming & Inverted Trailing")
    print("=" * 70)

    # Setup mock broker and order manager
    broker_mock = MagicMock()
    om = OrderManager(broker_mock, MagicMock())
    exits_recorded = []

    def mock_on_exit(pos_id, exit_price, reason):
        exits_recorded.append((pos_id, exit_price, reason))

    pm = PositionMonitor(om)
    pm._on_exit = mock_on_exit

    # Managed credit spread position
    # Net credit = 50.0. Hard SL = 80.0 (30 pt risk). Target = 15.0 (70% decay). Breakeven = 25.0 (50% decay)
    cs_state = TrailingState(
        position_id="CS_TEST_01",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=50.0,
        initial_sl=80.0,
        current_sl=80.0,
        target=15.0,
        direction="bearish",
        tier="B",
        strategy_type="credit_spread",
        hard_sl_price=80.0,
        target_decay_price=15.0,
        breakeven_decay_price=25.0,
    )
    pm.add_position(cs_state)

    # 1. Test normal decay tick (gain): spread drops to 40.0 -> no exit
    pm._process_tick(cs_state, 40.0)
    assert len(exits_recorded) == 0, f"Unexpected exit at 40.0: {exits_recorded}"
    print("[PASS] Spread at 40.0 (decay from 50.0): No exit triggered.")

    # 2. Test Breakeven Lock trigger: spread drops to 24.0 (<= breakeven_decay 25.0)
    pm._process_tick(cs_state, 24.0)
    assert cs_state.breakeven_set is True, "Breakeven lock was not armed!"
    expected_new_sl = 50.0 * 0.85  # 42.5
    assert cs_state.current_sl == expected_new_sl, f"SL not set to 85% of credit: {cs_state.current_sl}"
    assert len(exits_recorded) == 0, f"Unexpected exit during breakeven arming: {exits_recorded}"
    print(f"[PASS] Spread at 24.0: Breakeven armed, SL ratcheted down to {cs_state.current_sl}.")

    # 3. Test Breakeven SL breach: spread bounces back to 43.0 (>= 42.5)
    pm._process_tick(cs_state, 43.0)
    assert len(exits_recorded) == 1, f"Expected 1 exit, got: {exits_recorded}"
    assert exits_recorded[0] == ("CS_TEST_01", 43.0, "breakeven_exit_credit_spread")
    print(f"[PASS] Spread bounced to 43.0: Breakeven SL hit triggered exit: {exits_recorded[0]}.")

    # 4. Fresh credit spread state testing Hard SL breach: spread spikes to 85.0 (>= 80.0)
    exits_recorded.clear()
    cs_state_hard = TrailingState(
        position_id="CS_TEST_02",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=50.0,
        initial_sl=80.0,
        current_sl=80.0,
        target=15.0,
        direction="bearish",
        tier="B",
        strategy_type="credit_spread",
        hard_sl_price=80.0,
        target_decay_price=15.0,
        breakeven_decay_price=25.0,
    )
    pm.add_position(cs_state_hard)
    pm._process_tick(cs_state_hard, 81.5)
    assert len(exits_recorded) == 1, f"Expected hard SL exit, got: {exits_recorded}"
    assert exits_recorded[0] == ("CS_TEST_02", 81.5, "stop_loss_credit_spread")
    print(f"[PASS] Spread spiked to 81.5 >= 80.0: Hard SL triggered: {exits_recorded[0]}.")

    # 5. Fresh credit spread state testing Target Decay: spread decays to 12.0 (<= 15.0)
    exits_recorded.clear()
    cs_state_tgt = TrailingState(
        position_id="CS_TEST_03",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=50.0,
        initial_sl=80.0,
        current_sl=80.0,
        target=15.0,
        direction="bearish",
        tier="B",
        strategy_type="credit_spread",
        hard_sl_price=80.0,
        target_decay_price=15.0,
        breakeven_decay_price=25.0,
    )
    pm.add_position(cs_state_tgt)
    pm._process_tick(cs_state_tgt, 14.5)
    assert len(exits_recorded) == 1, f"Expected target decay exit, got: {exits_recorded}"
    assert exits_recorded[0] == ("CS_TEST_03", 14.5, "target_decay_credit_spread")
    print(f"[PASS] Spread decayed to 14.5 <= 15.0: Target exit triggered: {exits_recorded[0]}.")

    print("=> Reproduction 1: ALL CHECKS PASSED.\n")


def verify_reproduction_2_pre_sl_bailout_noise_suppression():
    print("=" * 70)
    print("REPRODUCTION 2: Pre-SL Tick-0 Bailout Noise Suppression")
    print("=" * 70)

    engine = PositionHealthEngine()

    class MockPosition:
        def __init__(self, tier="B", symbol="NIFTY BANK", entry_premium=250.0, entry_spot=50000.0, bars_held=0):
            self.position_id = "POS_MOCK"
            self.symbol = symbol
            self.tradingsymbol = "BANKNIFTY26OCT50000CE"
            self.direction = "bullish"
            self.entry_premium = entry_premium
            self.entry_spot = entry_spot
            self.risk_distance = 35.0
            self.tier = tier
            self.entry_bar_count = bars_held
            self.bars_held = bars_held
            self.entry_iv = 0.16

    # Create adverse candle data that creates poor health score (<= -35.0)
    bars = []
    base_dt = pd.Timestamp("2026-10-06 09:15:00")
    for i in range(15):
        bars.append({
            "timestamp": base_dt + pd.Timedelta(minutes=15 * i),
            "date": base_dt + pd.Timedelta(minutes=15 * i),
            "open": 50000.0 - i * 40,
            "high": 50010.0 - i * 40,
            "low": 49400.0 - i * 40,
            "close": 49410.0 - i * 40,
            "volume": 25000 + i * 1500,
        })
    df_adverse = pd.DataFrame(bars)

    with patch.object(engine, "_eval_pillar_htf", return_value=(-50.0, "BEAR", "1H trend reversed", None)):
        # Test 2.1: Tier S on Bank Nifty, 2-pt noise dip (entry 250 -> 248.0, loss = -2.0 pts) at Tick 0 (bars_held=0)
        pos_s = MockPosition(tier="S", symbol="NIFTY BANK", entry_premium=250.0, bars_held=0)
        report_2pt = engine.evaluate_position_health(pos_s, current_premium=248.0, underlying_df_override=df_adverse)
        print(f"Tier S, Bank Nifty, 2-pt dip: Health={report_2pt.health_score}, Threats={len(report_2pt.threat_reasons)}, Action={report_2pt.suggested_action}")
        assert report_2pt.suggested_action == "HOLD", f"Expected HOLD for 2-pt noise dip, got: {report_2pt.suggested_action}"
        print("[PASS] 2-pt noise dip on Tier S Bank Nifty correctly preserved as HOLD.")

        # Test 2.2: Tier B on Sensex, 5-pt noise dip (entry 300 -> 295.0, loss = -5.0 pts) at Tick 0
        pos_b = MockPosition(tier="B", symbol="SENSEX", entry_premium=300.0, bars_held=0)
        report_5pt = engine.evaluate_position_health(pos_b, current_premium=295.0, underlying_df_override=df_adverse)
        print(f"Tier B, Sensex, 5-pt dip: Health={report_5pt.health_score}, Threats={len(report_5pt.threat_reasons)}, Action={report_5pt.suggested_action}")
        assert report_5pt.suggested_action == "HOLD", f"Expected HOLD for 5-pt noise dip, got: {report_5pt.suggested_action}"
        print("[PASS] 5-pt noise dip on Tier B Sensex correctly preserved as HOLD.")

        # Test 2.3: Tier B on Bank Nifty, 13.5-pt noise dip (< 14 pt noise floor) at Tick 0
        pos_b_bn = MockPosition(tier="B", symbol="NIFTY BANK", entry_premium=250.0, bars_held=0)
        report_13pt = engine.evaluate_position_health(pos_b_bn, current_premium=236.5, underlying_df_override=df_adverse)
        print(f"Tier B, Bank Nifty, 13.5-pt dip: Health={report_13pt.health_score}, Action={report_13pt.suggested_action}")
        assert report_13pt.suggested_action == "HOLD", f"Expected HOLD within 14 pt noise floor, got: {report_13pt.suggested_action}"
        print("[PASS] 13.5-pt dip on Bank Nifty (<14 pt buffer) correctly preserved as HOLD.")

        # Test 2.4: Meaningful loss beyond noise floor (> 14 pts, e.g. -20 pts loss) -> Now PRE_SL_BAILOUT is permitted
        report_20pt = engine.evaluate_position_health(pos_b_bn, current_premium=230.0, underlying_df_override=df_adverse)
        print(f"Tier B, Bank Nifty, 20-pt loss (> noise floor): Health={report_20pt.health_score}, Action={report_20pt.suggested_action}")
        assert report_20pt.suggested_action == "PRE_SL_BAILOUT", f"Expected PRE_SL_BAILOUT past noise floor, got: {report_20pt.suggested_action}"
        print("[PASS] 20-pt loss outside noise floor correctly triggers PRE_SL_BAILOUT.")

        # Test 2.5: Tier C setup after 1 bar with sustained structural threat
        pos_c_held = MockPosition(tier="C", symbol="NIFTY BANK", entry_premium=250.0, bars_held=1)
        report_c_held = engine.evaluate_position_health(pos_c_held, current_premium=248.0, underlying_df_override=df_adverse)
        print(f"Tier C, Bank Nifty, 2-pt dip after 1 bar: Health={report_c_held.health_score}, Action={report_c_held.suggested_action}")
        assert report_c_held.suggested_action == "PRE_SL_BAILOUT", f"Expected PRE_SL_BAILOUT for Tier C after 1 bar, got: {report_c_held.suggested_action}"
        print("[PASS] Tier C after 1 bar with structural threat correctly transitions to PRE_SL_BAILOUT.")

        # Test 2.6: Tier C setup at tick 0 (bars_held=0) with 2-pt dip -> MUST be HOLD (noise suppressed)
        pos_c_tick0 = MockPosition(tier="C", symbol="NIFTY BANK", entry_premium=250.0, bars_held=0)
        report_c_tick0 = engine.evaluate_position_health(pos_c_tick0, current_premium=248.0, underlying_df_override=df_adverse)
        print(f"Tier C, Bank Nifty, 2-pt dip at tick 0: Health={report_c_tick0.health_score}, Action={report_c_tick0.suggested_action}")
        assert report_c_tick0.suggested_action == "HOLD", f"Expected HOLD for Tier C at tick 0 inside noise envelope, got: {report_c_tick0.suggested_action}"
        print("[PASS] Tier C at tick 0 correctly suppresses 2-pt noise dip as HOLD.")

    print("=> Reproduction 2: ALL CHECKS PASSED.\n")


def verify_reproduction_3_max_pain_asymmetric_chains():
    print("=" * 70)
    print("REPRODUCTION 3: Max Pain Calculation on Asymmetric Chains")
    print("=" * 70)

    strikes = np.array([24000, 24200, 24400, 24600, 24800, 25000, 25200, 25400], dtype=float)

    # Oracle function to compute exact total buyer payout
    def buyer_payout(expiry, strikes, call_oi, put_oi):
        c_p = np.sum(call_oi * np.maximum(expiry - strikes, 0.0))
        p_p = np.sum(put_oi * np.maximum(strikes - expiry, 0.0))
        return c_p + p_p

    # Case A: Heavy Call OI clustered at 25000 (Call wall)
    call_oi_a = np.array([0, 0, 0, 0, 0, 200000, 0, 0], dtype=float)
    put_oi_a = np.zeros_like(call_oi_a)
    mp_a = max_pain(strikes, call_oi_a, put_oi_a, 24800.0)

    # For strikes <= 25000, payout is 0. For > 25000, payout > 0.
    # Minimum payout is 0, occurring at all strikes <= 25000.
    # Max pain strike must be <= 25000.
    payout_at_mp_a = buyer_payout(mp_a, strikes, call_oi_a, put_oi_a)
    min_possible_payout_a = min(buyer_payout(s, strikes, call_oi_a, put_oi_a) for s in strikes)
    assert payout_at_mp_a == min_possible_payout_a, f"Payout at MP {payout_at_mp_a} != min {min_possible_payout_a}"
    assert mp_a <= 25000.0, f"Max pain {mp_a} > 25000.0 on pure call chain!"
    print(f"[PASS] Asymmetric Call Chain: Max pain = {mp_a}, Payout = {payout_at_mp_a} (Minimal buyer payout).")

    # Case B: Heavy Put OI clustered at 24400 (Put wall)
    call_oi_b = np.zeros_like(strikes)
    put_oi_b = np.array([0, 0, 150000, 0, 0, 0, 0, 0], dtype=float)
    mp_b = max_pain(strikes, call_oi_b, put_oi_b, 24600.0)

    # For strikes >= 24400, put payout is 0. For < 24400, put payout > 0.
    payout_at_mp_b = buyer_payout(mp_b, strikes, call_oi_b, put_oi_b)
    min_possible_payout_b = min(buyer_payout(s, strikes, call_oi_b, put_oi_b) for s in strikes)
    assert payout_at_mp_b == min_possible_payout_b, f"Payout at MP {payout_at_mp_b} != min {min_possible_payout_b}"
    assert mp_b >= 24400.0, f"Max pain {mp_b} < 24400.0 on pure put chain!"
    print(f"[PASS] Asymmetric Put Chain: Max pain = {mp_b}, Payout = {payout_at_mp_b} (Minimal buyer payout).")

    # Case C: Realistic highly skewed chain with 50 random trials against brute force oracle
    np.random.seed(42)
    for trial in range(50):
        rand_ce = np.random.randint(0, 100000, size=len(strikes)).astype(float)
        rand_pe = np.random.randint(0, 100000, size=len(strikes)).astype(float)
        mp_calc = max_pain(strikes, rand_ce, rand_pe, 24700.0)

        # Brute force oracle
        all_payouts = [buyer_payout(s, strikes, rand_ce, rand_pe) for s in strikes]
        min_idx = np.argmin(all_payouts)
        expected_mp = strikes[min_idx]

        assert mp_calc == expected_mp, f"Trial {trial} failed: calculated {mp_calc} != oracle {expected_mp}"

    print("[PASS] 50 Randomized Asymmetric Option Chains matched Brute Force Oracle with 100% precision.")
    print("=> Reproduction 3: ALL CHECKS PASSED.\n")


def verify_reproduction_4_nan_composite_score():
    print("=" * 70)
    print("REPRODUCTION 4: NaN Health Composite Score Returns 0.0 (Neutral)")
    print("=" * 70)

    engine = PositionHealthEngine()

    class MockPosState:
        def __init__(self):
            self.position_id = "P_NAN_TEST"
            self.symbol = "NIFTY BANK"
            self.tradingsymbol = "BANKNIFTY26OCT50000CE"
            self.direction = "bullish"
            self.entry_premium = 250.0
            self.entry_spot = 50000.0
            self.risk_distance = 35.0
            self.tier = "B"
            self.entry_bar_count = 2
            self.bars_held = 2
            self.entry_iv = 0.16

    state = MockPosState()

    # Patch ALL individual pillar methods to return float('nan')
    with patch.object(engine, "_eval_pillar_vwap", return_value=(float("nan"), 0.0, None, None)), \
         patch.object(engine, "_eval_pillar_volume", return_value=(float("nan"), 1.0, None, None)), \
         patch.object(engine, "_eval_pillar_gex", return_value=(float("nan"), 0.0, None, None)), \
         patch.object(engine, "_eval_pillar_oi", return_value=(float("nan"), 0.0, None, None)), \
         patch.object(engine, "_eval_pillar_noise", return_value=float("nan")), \
         patch.object(engine, "_eval_pillar_theta", return_value=(float("nan"), None)), \
         patch.object(engine, "_eval_pillar_iv", return_value=(float("nan"), 0.0, None, None)), \
         patch.object(engine, "_eval_pillar_htf", return_value=(float("nan"), "NEUTRAL", None, None)):

        report = engine.evaluate_position_health(state, current_premium=250.0)

        # Check 1: Must not be NaN
        assert not math.isnan(report.health_score), "health_score is NaN!"
        # Check 2: Must be exactly 0.0 (neutral)
        assert report.health_score == 0.0, f"Expected 0.0, but got {report.health_score} (did min(100, NaN) trap occur?)"
        # Check 3: Action must be HOLD
        assert report.suggested_action == "HOLD", f"Expected HOLD on NaN composite, got: {report.suggested_action}"
        print(f"[PASS] All-NaN pillars evaluated cleanly: Health Score = {report.health_score}, Action = {report.suggested_action}")

    # Test direct math property of min(100.0, nan) vs guarded code:
    raw_nan = float("nan")
    trap_value = min(100.0, raw_nan)
    assert math.isnan(trap_value) or trap_value == 100.0
    guarded_value = 0.0 if math.isnan(raw_nan) else round(max(-100.0, min(100.0, raw_nan)), 1)
    assert guarded_value == 0.0
    print(f"[PASS] Verified Python trap demonstration: min(100.0, NaN) yields {trap_value}, while guarded yields {guarded_value}.")

    print("=> Reproduction 4: ALL CHECKS PASSED.\n")


def verify_reproduction_5_commitment_ratio_clamping():
    print("=" * 70)
    print("REPRODUCTION 5: Institutional Commitment Ratio Clamping [0.0, 1.0]")
    print("=" * 70)

    analyzer = OIAnalyzer()

    # Helper function to run analyze on synthetic ATM data
    def run_cr(tot_oi_change, tot_volume):
        df = pd.DataFrame({
            "strike": [25000],
            "option_type": ["CE"],
            "oi": [10000],
            "volume": [tot_volume],
            "oi_change": [tot_oi_change],
            "iv": [0.15],
        })
        res = analyzer.analyze(df, spot_price=25000.0)
        return res["metrics"]["commitment_ratio"]

    # Boundary 1: Zero OI change, positive volume
    cr_zero = run_cr(0, 10000)
    assert cr_zero == 0.0, f"Expected 0.0, got {cr_zero}"
    print(f"[PASS] Zero OI change -> CR = {cr_zero}")

    # Boundary 2: Normal ratio (OI change 2,000, volume 10,000) -> 0.2
    cr_normal = run_cr(2000, 10000)
    assert cr_normal == 0.20, f"Expected 0.20, got {cr_normal}"
    print(f"[PASS] 2,000 OI change / 10,000 Vol -> CR = {cr_normal}")

    # Boundary 3: Equal ratio (OI change 10,000, volume 10,000) -> 1.0
    cr_one = run_cr(10000, 10000)
    assert cr_one == 1.0, f"Expected 1.0, got {cr_one}"
    print(f"[PASS] 10,000 OI change / 10,000 Vol -> CR = {cr_one}")

    # Boundary 4: Extreme anomalous spike (OI change 50,000, volume 10,000 -> raw = 5.0) -> must clamp to 1.0
    cr_spike = run_cr(50000, 10000)
    assert cr_spike == 1.0, f"Expected clamped 1.0, got {cr_spike}"
    print(f"[PASS] Extreme spike (5x volume) -> Clamped CR = {cr_spike} (Strictly <= 1.0)")

    # Boundary 5: Negative OI change (unwinding) (abs taken -> 8,000 / 10,000 -> 0.8)
    cr_neg = run_cr(-8000, 10000)
    assert cr_neg == 0.8, f"Expected 0.8, got {cr_neg}"
    print(f"[PASS] Negative OI change unwinding -> CR = {cr_neg}")

    # Boundary 6: Zero volume -> must be 0.0 (no division by zero)
    cr_zero_vol = run_cr(5000, 0)
    assert cr_zero_vol == 0.0, f"Expected 0.0 on zero volume, got {cr_zero_vol}"
    print(f"[PASS] Zero volume -> CR = {cr_zero_vol}")

    # Boundary 7: 1,000 randomized Monte Carlo stress inputs
    np.random.seed(123)
    for _ in range(1000):
        oi_chg = np.random.uniform(-100000, 100000)
        vol = max(0.0, np.random.uniform(0, 100000))
        cr_rand = run_cr(oi_chg, vol)
        assert 0.0 <= cr_rand <= 1.0, f"CR out of bounds: {cr_rand} (oi_chg={oi_chg}, vol={vol})"

    print("[PASS] 1,000 Monte Carlo boundary stress tests: 100% strictly clamped in [0.0, 1.0].")
    print("=> Reproduction 5: ALL CHECKS PASSED.\n")


def main():
    print("=" * 80)
    print("STARTING INDEPENDENT ADVERSARIAL EMPIRICAL REPRODUCTION VERIFICATION")
    print("=" * 80 + "\n")

    verify_reproduction_1_credit_spread()
    verify_reproduction_2_pre_sl_bailout_noise_suppression()
    verify_reproduction_3_max_pain_asymmetric_chains()
    verify_reproduction_4_nan_composite_score()
    verify_reproduction_5_commitment_ratio_clamping()

    print("=" * 80)
    print("ALL 5 INDEPENDENT EMPIRICAL REPRODUCTIONS VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
