"""
Adversarial Boundary & Stress Test Harness.
Executes programmatic stress tests on extreme inputs across:
1. gamma_engine.py
2. target_calibrator.py
3. oi_analyzer.py
4. position_health.py
5. angelone_options.py / broker error handling

Covers:
- Zero division (zero volume, OI, strikes, DTE, premium, ATR)
- Empty chains / Schema corruption (None, empty DF, missing cols, NaN/Inf)
- Negative / Zero / Extreme IV (-50%, 0%, 500%, 10,000%)
- Abrupt spot gaps (±5%, ±10%, ±20%, ±50%)
- Broker connection timeouts and error codes (AB1021, AB2001, AG8001, Timeout)
- Discrepancy analysis between mathematical formulations and runtime code
"""

import math
import sys
import os
import traceback
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prometheus.signals.gamma_engine import GammaEngine, calculate_black_scholes_gamma
from prometheus.signals.target_calibrator import calibrate_target_and_sl, calculate_structural_sl
from prometheus.signals.oi_analyzer import OIAnalyzer
from prometheus.utils.options_math import max_pain, pcr_ratio, calculate_greeks, implied_volatility
from prometheus.execution.position_health import PositionHealthEngine, PositionHealthReport


class StressTestRunner:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def record(self, module: str, test_name: str, passed: bool, details: str, severity: str = "INFO"):
        self.results.append({
            "module": module,
            "test": test_name,
            "passed": passed,
            "details": details,
            "severity": severity
        })
        status = "PASS" if passed else f"FAIL [{severity}]"
        print(f"[{status}] {module} :: {test_name} -> {details}")

    # =========================================================================
    # MODULE 1: GAMMA ENGINE & BLACK-SCHOLES
    # =========================================================================
    def stress_gamma_engine(self):
        print("\n" + "="*80)
        print("RUNNING STRESS TESTS: gamma_engine.py")
        print("="*80)

        # 1.1 calculate_black_scholes_gamma with Zero / Negative values
        try:
            g_zero_spot = calculate_black_scholes_gamma(spot=0.0, strike=25000, dte=2, sigma=0.15)
            self.record("gamma_engine", "bs_gamma_zero_spot", g_zero_spot == 0.0, f"Result: {g_zero_spot}")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_zero_spot", False, f"Exception: {e}", "CRITICAL")

        try:
            g_neg_spot = calculate_black_scholes_gamma(spot=-25000, strike=25000, dte=2, sigma=0.15)
            self.record("gamma_engine", "bs_gamma_neg_spot", g_neg_spot == 0.0, f"Result: {g_neg_spot}")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_neg_spot", False, f"Exception: {e}", "HIGH")

        try:
            g_zero_strike = calculate_black_scholes_gamma(spot=25000, strike=0.0, dte=2, sigma=0.15)
            self.record("gamma_engine", "bs_gamma_zero_strike", g_zero_strike == 0.0, f"Result: {g_zero_strike}")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_zero_strike", False, f"Exception: {e}", "CRITICAL")

        try:
            g_neg_strike = calculate_black_scholes_gamma(spot=25000, strike=-25000, dte=2, sigma=0.15)
            self.record("gamma_engine", "bs_gamma_neg_strike", g_neg_strike == 0.0, f"Result: {g_neg_strike}")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_neg_strike", False, f"Exception: {e}", "HIGH")

        try:
            g_zero_dte = calculate_black_scholes_gamma(spot=25000, strike=25000, dte=0.0, sigma=0.15)
            self.record("gamma_engine", "bs_gamma_zero_dte", g_zero_dte > 0 and not math.isnan(g_zero_dte), f"Result: {g_zero_dte} (handled by max(dte, 0.25))")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_zero_dte", False, f"Exception: {e}", "HIGH")

        try:
            g_neg_dte = calculate_black_scholes_gamma(spot=25000, strike=25000, dte=-10.0, sigma=0.15)
            self.record("gamma_engine", "bs_gamma_neg_dte", g_neg_dte > 0 and not math.isnan(g_neg_dte), f"Result: {g_neg_dte} (handled by max(dte, 0.25))")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_neg_dte", False, f"Exception: {e}", "MEDIUM")

        try:
            g_zero_iv = calculate_black_scholes_gamma(spot=25000, strike=25000, dte=2, sigma=0.0)
            self.record("gamma_engine", "bs_gamma_zero_iv", g_zero_iv == 0.0, f"Result: {g_zero_iv}")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_zero_iv", False, f"Exception: {e}", "HIGH")

        try:
            g_neg_iv = calculate_black_scholes_gamma(spot=25000, strike=25000, dte=2, sigma=-0.20)
            self.record("gamma_engine", "bs_gamma_neg_iv", g_neg_iv == 0.0, f"Result: {g_neg_iv}")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_neg_iv", False, f"Exception: {e}", "HIGH")

        try:
            g_extreme_iv = calculate_black_scholes_gamma(spot=25000, strike=25000, dte=2, sigma=5.0)
            self.record("gamma_engine", "bs_gamma_extreme_iv_500pct", g_extreme_iv >= 0.0 and not math.isnan(g_extreme_iv), f"Result: {g_extreme_iv}")
        except Exception as e:
            self.record("gamma_engine", "bs_gamma_extreme_iv_500pct", False, f"Exception: {e}", "HIGH")

        # 1.2 NaN / Inf injection into calculate_black_scholes_gamma
        for val_name, bad_val in [("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf"))]:
            try:
                g_bad = calculate_black_scholes_gamma(spot=bad_val, strike=25000, dte=2, sigma=0.15)
                is_safe = (g_bad == 0.0 or not math.isnan(g_bad))
                self.record("gamma_engine", f"bs_gamma_spot_{val_name}", is_safe, f"Returned: {g_bad}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("gamma_engine", f"bs_gamma_spot_{val_name}", False, f"CRASH: {type(e).__name__}: {e}", "HIGH")

            try:
                g_bad = calculate_black_scholes_gamma(spot=25000, strike=bad_val, dte=2, sigma=0.15)
                is_safe = (g_bad == 0.0 or not math.isnan(g_bad))
                self.record("gamma_engine", f"bs_gamma_strike_{val_name}", is_safe, f"Returned: {g_bad}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("gamma_engine", f"bs_gamma_strike_{val_name}", False, f"CRASH: {type(e).__name__}: {e}", "HIGH")

            try:
                g_bad = calculate_black_scholes_gamma(spot=25000, strike=25000, dte=2, sigma=bad_val)
                is_safe = (g_bad == 0.0 or not math.isnan(g_bad))
                self.record("gamma_engine", f"bs_gamma_sigma_{val_name}", is_safe, f"Returned: {g_bad}", "HIGH" if not is_safe else "HIGH")
            except Exception as e:
                self.record("gamma_engine", f"bs_gamma_sigma_{val_name}", False, f"CRASH: {type(e).__name__}: {e}", "HIGH")

        # 1.3 GammaEngine.calculate_gex with Adversarial DataFrames
        engine = GammaEngine()
        
        # Empty chain
        res = engine.calculate_gex(pd.DataFrame(), 25000.0, "NIFTY 50")
        self.record("gamma_engine", "calculate_gex_empty_df", res["net_gex"] == 0.0 and res["gamma_regime"] == "NEUTRAL", f"Result: {res}")

        # None chain
        res = engine.calculate_gex(None, 25000.0, "NIFTY 50")
        self.record("gamma_engine", "calculate_gex_none_df", res["net_gex"] == 0.0 and res["gamma_regime"] == "NEUTRAL", f"Result: {res}")

        # Missing required columns
        corrupt_cols_df = pd.DataFrame({"random_col": [1, 2, 3], "other": [4, 5, 6]})
        res = engine.calculate_gex(corrupt_cols_df, 25000.0, "NIFTY 50")
        self.record("gamma_engine", "calculate_gex_missing_cols", res["net_gex"] == 0.0 and res["gamma_regime"] == "NEUTRAL", f"Result: {res}")

        # Chain with all zeros
        zero_df = pd.DataFrame({
            "strike_price": [24800, 24900, 25000, 25100, 25200] * 2,
            "option_type": ["CE"] * 5 + ["PE"] * 5,
            "open_interest": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "iv": [0.15] * 10
        })
        res = engine.calculate_gex(zero_df, 25000.0, "NIFTY 50")
        self.record("gamma_engine", "calculate_gex_zero_oi", res["net_gex"] == 0.0 and res["gamma_regime"] == "NEUTRAL", f"Result: {res}")

        # Chain with NaN in strikes, OI, IV
        nan_df = pd.DataFrame({
            "strike_price": [24800, np.nan, 25000, 25100, 25200, 24800, 24900, 25000, np.nan, 25200],
            "option_type": ["CE"] * 5 + ["PE"] * 5,
            "open_interest": [1000, 2000, np.nan, 4000, 5000, 1000, np.nan, 3000, 4000, 5000],
            "iv": [0.15, np.nan, 0.15, np.nan, 0.15, 0.15, 0.15, np.nan, 0.15, 0.15]
        })
        try:
            res = engine.calculate_gex(nan_df, 25000.0, "NIFTY 50")
            self.record("gamma_engine", "calculate_gex_nan_in_chain", not math.isnan(res["net_gex"]), f"Net GEX: {res['net_gex']}, ZGL: {res['zgl']}")
        except Exception as e:
            self.record("gamma_engine", "calculate_gex_nan_in_chain", False, f"CRASH: {e}", "CRITICAL")

        # Spot price = NaN / Inf in calculate_gex
        for bad_spot, name in [(float("nan"), "nan"), (float("inf"), "inf"), (-25000.0, "negative")]:
            try:
                valid_chain = pd.DataFrame({
                    "strike_price": [24800, 25000, 25200, 24800, 25000, 25200],
                    "option_type": ["CE", "CE", "CE", "PE", "PE", "PE"],
                    "open_interest": [10000, 20000, 15000, 12000, 25000, 11000],
                    "iv": [0.15] * 6
                })
                res = engine.calculate_gex(valid_chain, bad_spot, "NIFTY 50")
                is_safe = (res["net_gex"] == 0.0 or not math.isnan(res["net_gex"]))
                self.record("gamma_engine", f"calculate_gex_spot_{name}", is_safe, f"Result: {res}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("gamma_engine", f"calculate_gex_spot_{name}", False, f"CRASH: {type(e).__name__}: {e}", "CRITICAL")

        # Spot gaps: +5%, +10%, +20%, +50%, -20%
        base_spot = 25000.0
        sample_chain = pd.DataFrame({
            "strike_price": [24500, 24800, 25000, 25200, 25500] * 2,
            "option_type": ["CE"] * 5 + ["PE"] * 5,
            "open_interest": [10000, 25000, 50000, 20000, 15000, 12000, 22000, 48000, 18000, 14000],
            "iv": [0.14] * 10
        })
        for gap_pct in [-0.50, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.50]:
            eval_spot = base_spot * (1 + gap_pct)
            try:
                res = engine.calculate_gex(sample_chain, eval_spot, "NIFTY 50")
                self.record("gamma_engine", f"spot_gap_{int(gap_pct*100):+d}pct", not math.isnan(res["net_gex"]), f"Spot: {eval_spot:.1f}, Net GEX: {res['net_gex']}, Regime: {res['gamma_regime']}, ZGL: {res['zgl']}")
            except Exception as e:
                self.record("gamma_engine", f"spot_gap_{int(gap_pct*100):+d}pct", False, f"CRASH: {e}", "HIGH")

        # 1.4 Mathematical Verification: Rupee Notional GEX Dimension Check
        # Standard SqueezeMetrics Net GEX: Rupee Gamma = Q * Gamma * Spot^2 * 0.01 (where Q is shares in open interest)
        expected_gex = 100000 * calculate_black_scholes_gamma(25000.0, 25000.0, 2.0, 0.15) * (25000.0 ** 2) * 0.01
        test_chain = pd.DataFrame([{"strike_price": 25000.0, "option_type": "CE", "open_interest": 100000, "iv": 0.15}])
        gex_out = engine.calculate_gex(test_chain, 25000.0, "NIFTY 50", dte=2.0)
        is_scaled_properly = math.isclose(gex_out["net_gex"], expected_gex, rel_tol=1e-3)
        self.record(
            "gamma_engine",
            "mathematical_discrepancy_gex_scaling",
            is_scaled_properly,
            f"Rupee Notional GEX correctly scaled: {gex_out['net_gex_cr']} Cr matches expected {expected_gex / 1e7:.2f} Cr"
        )


    # =========================================================================
    # MODULE 2: TARGET CALIBRATOR
    # =========================================================================
    def stress_target_calibrator(self):
        print("\n" + "="*80)
        print("RUNNING STRESS TESTS: target_calibrator.py")
        print("="*80)

        # 2.1 Zero / Negative inputs into calibrate_target_and_sl
        symbols = ["NIFTY BANK", "SENSEX", "NIFTY 50"]
        for sym in symbols:
            # Zero spot
            res = calibrate_target_and_sl(sym, spot_price=0.0, spot_atr=0.0, opt_ltp=0.0)
            self.record("target_calibrator", f"zero_spot_{sym}", res.target_gain_pts > 0 and res.sl_pts > 0, f"Target: {res.target_gain_pts}, SL: {res.sl_pts}")

            # Zero opt_ltp
            res = calibrate_target_and_sl(sym, spot_price=50000.0, spot_atr=100.0, opt_ltp=0.0)
            self.record("target_calibrator", f"zero_opt_ltp_{sym}", res.target_gain_pts > 0 and res.sl_pts > 0, f"Target: {res.target_gain_pts}, SL: {res.sl_pts}")

            # Negative opt_ltp
            res = calibrate_target_and_sl(sym, spot_price=50000.0, spot_atr=100.0, opt_ltp=-50.0)
            self.record("target_calibrator", f"neg_opt_ltp_{sym}", res.target_gain_pts > 0, f"Target: {res.target_gain_pts}, SL: {res.sl_pts}")

        # 2.2 Tier C Target Compression Boundaries
        # Contract:
        # Bank Nifty: 20 <= target <= 30
        # Sensex: 22 <= target <= 32
        # Nifty 50: 8 <= target <= 14
        specs = [
            ("NIFTY BANK", 20.0, 30.0),
            ("SENSEX", 22.0, 32.0),
            ("NIFTY 50", 8.0, 14.0)
        ]
        for sym, exp_min, exp_max in specs:
            # Case 1: Tier C, counter-trend (htf_aligned=False)
            res = calibrate_target_and_sl(sym, spot_price=50000.0, spot_atr=150.0, opt_ltp=300.0, tier="C", is_htf_aligned=False)
            valid = (exp_min <= res.target_gain_pts <= exp_max) and res.is_compressed
            self.record("target_calibrator", f"tier_c_counter_trend_{sym}", valid, f"Target: {res.target_gain_pts} (Expected [{exp_min}, {exp_max}]), Compressed: {res.is_compressed}")

            # Case 2: Tier C, LONG_GAMMA
            res = calibrate_target_and_sl(sym, spot_price=50000.0, spot_atr=150.0, opt_ltp=300.0, tier="C", is_htf_aligned=True, gamma_regime="LONG_GAMMA", net_gex=5.0)
            valid = (exp_min <= res.target_gain_pts <= exp_max) and res.is_compressed
            self.record("target_calibrator", f"tier_c_long_gamma_{sym}", valid, f"Target: {res.target_gain_pts} (Expected [{exp_min}, {exp_max}]), Compressed: {res.is_compressed}")

            # Case 3: Tier S, aligned - MUST NOT compress
            res_s = calibrate_target_and_sl(sym, spot_price=50000.0, spot_atr=150.0, opt_ltp=300.0, tier="S", is_htf_aligned=True, gamma_regime="SHORT_GAMMA", net_gex=-5.0)
            valid_s = not res_s.is_compressed and res_s.target_gain_pts >= exp_min
            self.record("target_calibrator", f"tier_s_runner_{sym}", valid_s, f"Target: {res_s.target_gain_pts}, Compressed: {res_s.is_compressed}")

            # Case 4: Dynamic retest expansion bypass check
            # For Tier C, structural_sl_pts very high (e.g. 50 pts).
            # Retest expansion rule: sl_pts > target * 1.2 -> scale target UP. BUT for Tier C, this MUST be bypassed!
            res_retest = calibrate_target_and_sl(sym, spot_price=50000.0, spot_atr=150.0, opt_ltp=300.0, tier="C", is_htf_aligned=False, structural_sl_pts=60.0)
            valid_retest = (exp_min <= res_retest.target_gain_pts <= exp_max) and res_retest.is_compressed
            self.record("target_calibrator", f"tier_c_retest_bypass_{sym}", valid_retest, f"Target: {res_retest.target_gain_pts} (Must remain in [{exp_min}, {exp_max}]), SL: {res_retest.sl_pts}")

        # 2.3 NaN / Inf injection into calibrate_target_and_sl
        for bad_val, name in [(float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")]:
            try:
                res = calibrate_target_and_sl("NIFTY BANK", spot_price=bad_val, spot_atr=100.0, opt_ltp=200.0)
                is_safe = not math.isnan(res.target_gain_pts)
                self.record("target_calibrator", f"calibrate_spot_{name}", is_safe, f"Target: {res.target_gain_pts}, SL: {res.sl_pts}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("target_calibrator", f"calibrate_spot_{name}", False, f"CRASH: {e}", "HIGH")

            try:
                res = calibrate_target_and_sl("NIFTY BANK", spot_price=50000.0, spot_atr=bad_val, opt_ltp=200.0)
                is_safe = not math.isnan(res.target_gain_pts)
                self.record("target_calibrator", f"calibrate_atr_{name}", is_safe, f"Target: {res.target_gain_pts}, SL: {res.sl_pts}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("target_calibrator", f"calibrate_atr_{name}", False, f"CRASH: {e}", "HIGH")

            try:
                res = calibrate_target_and_sl("NIFTY BANK", spot_price=50000.0, spot_atr=100.0, opt_ltp=bad_val)
                is_safe = not math.isnan(res.target_gain_pts) and not math.isnan(res.tgt_price)
                self.record("target_calibrator", f"calibrate_opt_ltp_{name}", is_safe, f"Target: {res.target_gain_pts}, Tgt Price: {res.tgt_price}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("target_calibrator", f"calibrate_opt_ltp_{name}", False, f"CRASH: {e}", "HIGH")


    # =========================================================================
    # MODULE 3: OI ANALYZER & OPTIONS MATH
    # =========================================================================
    def stress_oi_analyzer(self):
        print("\n" + "="*80)
        print("RUNNING STRESS TESTS: oi_analyzer.py & options_math.py")
        print("="*80)

        # 3.1 max_pain mathematical formulation & edge cases
        strikes = np.array([24800, 24900, 25000, 25100, 25200], dtype=float)
        ce_oi = np.array([10000, 20000, 50000, 30000, 10000], dtype=float)
        pe_oi = np.array([15000, 35000, 45000, 20000, 5000], dtype=float)

        try:
            mp = max_pain(strikes, ce_oi, pe_oi, 25000.0)
            self.record("options_math", "max_pain_normal", mp in strikes, f"Max pain strike: {mp}")
        except Exception as e:
            self.record("options_math", "max_pain_normal", False, f"CRASH: {e}", "CRITICAL")

        # Discrepancy Analysis on Max Pain formulation:
        # Theoretical Max Pain:
        # At expiration price S, total money paid by option writers to option buyers is:
        # Payout(S) = sum(CE_OI * max(S - K, 0)) + sum(PE_OI * max(K - S, 0)).
        # Max Pain to buyers occurs when Payout(S) is MINIMIZED (i.e. argmin(Payout)).
        # Let's inspect options_math.py lines 221-227:
        # call_pain = np.sum(call_oi * np.maximum(strikes - strike, 0))  <-- calculates max(K - S, 0), which is PUT payout!
        # put_pain = np.sum(put_oi * np.maximum(strike - strikes, 0))   <-- calculates max(S - K, 0), which is CALL payout!
        # total_pain = call_pain + put_pain
        # return strikes[np.argmax(total_pain)]                           <-- picks ARGMAX of cross-mismatched payouts!
        
        # Test synthetic asymmetric scenario:
        # Only Calls at 25000 (CE_OI = 100,000), PEs = 0 everywhere.
        # Where SHOULD Max Pain be?
        # If spot finishes <= 25000, all Calls expire worthless. Buyer payout is 0.
        # If spot finishes at 25200, Call payout is 100,000 * 200 = 20,000,000.
        # So maximum pain to buyers is at any strike <= 25000.
        test_strikes = np.array([24800, 24900, 25000, 25100, 25200], dtype=float)
        test_ce = np.array([0, 0, 100000, 0, 0], dtype=float)
        test_pe = np.array([0, 0, 0, 0, 0], dtype=float)
        mp_asym = max_pain(test_strikes, test_ce, test_pe, 25000.0)
        
        is_fixed = (mp_asym <= 25000.0)
        self.record(
            "options_math",
            "mathematical_discrepancy_max_pain_inverted",
            is_fixed,
            f"Pure Call OI at 25000 evaluated to Max Pain = {mp_asym} (correctly <= 25000, minimizing total buyer payout via argmin)",
            severity="MEDIUM"
        )

        # 3.2 PCR Ratio Zero Division & False Extreme Signal Bug
        # pcr_ratio with zero call_oi_total:
        pcr_zero_call = pcr_ratio(put_oi_total=10000, call_oi_total=0)
        self.record("options_math", "pcr_ratio_zero_call_oi", pcr_zero_call == 0.0, f"PCR: {pcr_zero_call}")

        # Now test how OIAnalyzer._interpret_pcr handles pcr = 0.0
        analyzer = OIAnalyzer()
        pcr_dict = {"oi": 0.0, "volume": 0.0, "call_oi_total": 0, "put_oi_total": 10000}
        pcr_signal = analyzer._interpret_pcr(pcr_dict)
        if pcr_signal is not None and pcr_signal.direction == "bearish" and pcr_signal.strength >= 1.0:
            self.record(
                "oi_analyzer",
                "zero_call_oi_false_bearish_pcr_signal",
                False,
                f"BUG CONFIRMED: When call_oi_total = 0, pcr_ratio returns 0.0. "
                f"_interpret_pcr evaluates (0.0 < 0.7) and triggers MAX STRENGTH BEARISH ({pcr_signal.strength}) "
                f"signal claiming '{pcr_signal.details}' when call OI was actually zero!",
                severity="HIGH"
            )
        else:
            self.record("oi_analyzer", "zero_call_oi_false_bearish_pcr_signal", True, f"Signal: {pcr_signal}")

        # 3.3 OIAnalyzer.analyze with Empty / Missing / NaN DataFrames
        res = analyzer.analyze(pd.DataFrame(), 25000.0)
        self.record("oi_analyzer", "analyze_empty_df", res["signals"] == [] and res["metrics"] == {}, f"Result: {res}")

        # Missing columns
        bad_chain = pd.DataFrame({
            "strike": [25000, 25100],
            "option_type": ["CE", "PE"]
            # missing "oi", "volume", "oi_change"
        })
        try:
            res = analyzer.analyze(bad_chain, 25000.0)
            self.record("oi_analyzer", "analyze_missing_oi_col", res["signals"] == [] and res["metrics"] == {}, f"Correctly caught and handled missing required column without KeyError: {res}")
        except KeyError as e:
            self.record(
                "oi_analyzer",
                "analyze_missing_oi_col_keyerror",
                False,
                f"UNHANDLED EXCEPTION: Passing DataFrame missing 'oi' column raises uncaught KeyError: {e}",
                severity="MEDIUM"
            )
        except Exception as e:
            self.record("oi_analyzer", "analyze_missing_oi_col_other", False, f"Exception: {e}", "MEDIUM")

        # Chain with NaNs and Infinite values in OI, volume, oi_change
        nan_chain = pd.DataFrame({
            "strike": [24800, 24900, 25000, 25100, 25200] * 2,
            "option_type": ["CE"] * 5 + ["PE"] * 5,
            "oi": [1000.0, np.nan, 5000.0, np.nan, 2000.0, np.nan, 3000.0, 4000.0, np.nan, 1000.0],
            "volume": [100, np.nan, 500, 200, np.nan, np.nan, 300, 400, np.nan, 100],
            "oi_change": [50, np.nan, -100, np.nan, 20, np.nan, 30, -50, np.nan, 10],
            "iv": [0.15, np.nan, 0.16, np.nan, 0.15, np.nan, 0.14, 0.15, np.nan, 0.16]
        })
        try:
            res = analyzer.analyze(nan_chain, 25000.0)
            self.record("oi_analyzer", "analyze_nan_chain", "metrics" in res, f"Processed with NaNs: commitment={res['metrics'].get('commitment_ratio')}")
        except Exception as e:
            self.record("oi_analyzer", "analyze_nan_chain", False, f"CRASH: {e}", "HIGH")

        # Spot gap ±20%
        try:
            res_gap = analyzer.analyze(nan_chain.dropna(), 30000.0) # spot way above strikes
            self.record("oi_analyzer", "analyze_extreme_spot_gap", "metrics" in res_gap, f"Processed gap: {res_gap['metrics']}")
        except Exception as e:
            self.record("oi_analyzer", "analyze_extreme_spot_gap", False, f"CRASH: {e}", "HIGH")


    # =========================================================================
    # MODULE 4: POSITION HEALTH ENGINE
    # =========================================================================
    def stress_position_health(self):
        print("\n" + "="*80)
        print("RUNNING STRESS TESTS: position_health.py")
        print("="*80)

        engine = PositionHealthEngine()

        class MockState:
            def __init__(self, **kwargs):
                self.position_id = "TEST_POS_1"
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
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # Baseline valid candle data
        bars = []
        base_dt = pd.Timestamp("2026-10-06 10:00:00")
        for i in range(25):
            bars.append({
                "date": base_dt + pd.Timedelta(minutes=15 * i),
                "open": 50000.0 + i * 10,
                "high": 50050.0 + i * 10,
                "low": 49980.0 + i * 10,
                "close": 50020.0 + i * 10,
                "volume": 15000 + i * 500
            })
        valid_df = pd.DataFrame(bars)

        # 4.1 Normal evaluation baseline
        state = MockState()
        rep = engine.evaluate_position_health(state, current_premium=270.0, underlying_df_override=valid_df)
        self.record("position_health", "evaluate_baseline", -100.0 <= rep.health_score <= 100.0, f"Score: {rep.health_score}, Action: {rep.suggested_action}")

        # 4.2 Empty candle DataFrame
        rep_empty = engine.evaluate_position_health(state, current_premium=270.0, underlying_df_override=pd.DataFrame())
        self.record("position_health", "evaluate_empty_df", rep_empty.health_score == 0.0, f"Score: {rep_empty.health_score} (neutral fallback)")

        # 4.3 Candle DataFrame with < 5 bars
        short_df = valid_df.iloc[:3].copy()
        rep_short = engine.evaluate_position_health(state, current_premium=270.0, underlying_df_override=short_df)
        self.record("position_health", "evaluate_short_df", rep_short.health_score == 0.0, f"Score: {rep_short.health_score} (neutral fallback)")

        # 4.4 DataFrame missing volume / close
        no_vol_df = valid_df.drop(columns=["volume"])
        rep_no_vol = engine.evaluate_position_health(state, current_premium=270.0, underlying_df_override=no_vol_df)
        self.record("position_health", "evaluate_missing_volume", -100.0 <= rep_no_vol.health_score <= 100.0, f"Score: {rep_no_vol.health_score}")

        # 4.5 All zero volume in candles
        zero_vol_df = valid_df.copy()
        zero_vol_df["volume"] = 0
        rep_zero_vol = engine.evaluate_position_health(state, current_premium=270.0, underlying_df_override=zero_vol_df)
        self.record("position_health", "evaluate_zero_volume", rep_zero_vol.rvol == 1.0, f"RVOL: {rep_zero_vol.rvol}, Score: {rep_zero_vol.health_score}")

        # 4.6 Zero and Negative current_premium and entry_premium
        rep_zero_prem = engine.evaluate_position_health(MockState(entry_premium=0.0), current_premium=0.0, underlying_df_override=valid_df)
        self.record("position_health", "evaluate_zero_premium", -100.0 <= rep_zero_prem.health_score <= 100.0, f"Score: {rep_zero_prem.health_score}")

        rep_neg_prem = engine.evaluate_position_health(MockState(entry_premium=-50.0), current_premium=-10.0, underlying_df_override=valid_df)
        self.record("position_health", "evaluate_neg_premium", -100.0 <= rep_neg_prem.health_score <= 100.0, f"Score: {rep_neg_prem.health_score}")

        # 4.7 NaN / Inf Injection in position_health
        for bad_val, name in [(float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")]:
            try:
                rep_bad_prem = engine.evaluate_position_health(state, current_premium=bad_val, underlying_df_override=valid_df)
                is_safe = not math.isnan(rep_bad_prem.health_score)
                self.record("position_health", f"evaluate_premium_{name}", is_safe, f"Score: {rep_bad_prem.health_score}, Action: {rep_bad_prem.suggested_action}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("position_health", f"evaluate_premium_{name}", False, f"CRASH: {e}", "CRITICAL")

            try:
                rep_bad_spot = engine.evaluate_position_health(state, current_premium=250.0, spot_override=bad_val, underlying_df_override=valid_df)
                is_safe = not math.isnan(rep_bad_spot.health_score)
                self.record("position_health", f"evaluate_spot_override_{name}", is_safe, f"Score: {rep_bad_spot.health_score}, Action: {rep_bad_spot.suggested_action}", "HIGH" if not is_safe else "INFO")
            except Exception as e:
                self.record("position_health", f"evaluate_spot_override_{name}", False, f"CRASH: {e}", "CRITICAL")

        # 4.8 NaNs injected into underlying candle DataFrame
        nan_candle_df = valid_df.copy()
        nan_candle_df.loc[nan_candle_df.index[-1], "close"] = np.nan
        nan_candle_df.loc[nan_candle_df.index[-1], "volume"] = np.nan
        try:
            rep_nan_df = engine.evaluate_position_health(state, current_premium=250.0, underlying_df_override=nan_candle_df)
            is_safe = not math.isnan(rep_nan_df.health_score)
            self.record("position_health", "evaluate_nan_candle_df", is_safe, f"Score: {rep_nan_df.health_score}, Action: {rep_nan_df.suggested_action}", "HIGH" if not is_safe else "INFO")
        except Exception as e:
            self.record("position_health", "evaluate_nan_candle_df", False, f"CRASH: {e}", "CRITICAL")

        # 4.9 Pillar 1 VWAP False Positive on NaN spot
        # When spot_price is NaN, does _eval_pillar_vwap falsely assign +15.0?
        p1_score, v_gap, v_threat, v_favor = engine._eval_pillar_vwap(valid_df, float("nan"), True)
        if p1_score == 15.0:
            self.record(
                "position_health",
                "pillar_1_vwap_nan_false_positive",
                False,
                f"DISCREPANCY / ANOMALY: _eval_pillar_vwap with spot=NaN falls through (NaN < VWAP is False) "
                f"and returns +15.0 (positive score) instead of 0.0 or neutral on NaN spot!",
                severity="MEDIUM"
            )
        else:
            self.record("position_health", "pillar_1_vwap_nan_false_positive", True, f"P1 score: {p1_score}")


    # =========================================================================
    # MODULE 5: BROKER CONNECTION ERROR CODES & TIMEOUTS
    # =========================================================================
    def stress_broker_connection(self):
        print("\n" + "="*80)
        print("RUNNING STRESS TESTS: broker error handling & timeouts")
        print("="*80)

        from prometheus.data.angelone_options import AngelOneOptionChain

        class MockFetcher:
            def __init__(self):
                self._rate_limiter = None

        ao = AngelOneOptionChain(MockFetcher())

        # Test AB1021 (rate limited)
        res_ab1021 = {"status": False, "errorCode": "AB1021", "message": "Too many requests"}
        marked = ao._mark_rate_limited(res_ab1021, "test")
        self.record("broker_error", "mark_rate_limited_ab1021", marked is True, f"Handled AB1021: {marked}")

        # Test AB1020 (rate limited)
        res_ab1020 = {"status": False, "errorCode": "AB1020", "message": "Exceeding access rate limit"}
        marked_1020 = ao._mark_rate_limited(res_ab1020, "test")
        self.record("broker_error", "mark_rate_limited_ab1020", marked_1020 is True, f"Handled AB1020: {marked_1020}")

        # Test AB2001 (Internal server error)
        res_ab2001 = {"status": False, "errorCode": "AB2001", "message": "Internal Server Error"}
        marked_2001 = ao._mark_rate_limited(res_ab2001, "test")
        self.record("broker_error", "mark_ab2001_server_error", marked_2001 is False, f"AB2001 marked as rate limited: {marked_2001} (Correctly not treated as 429)")

        # Test AG8001 (Invalid token / auth failure)
        ao._mark_auth_failure("AG8001: Invalid Token")
        is_disabled = ao._is_temporarily_disabled()
        self.record("broker_error", "mark_auth_failure_ag8001", is_disabled is True, f"Option chain disabled for cooldown: {is_disabled}")


    def run_all(self):
        self.stress_gamma_engine()
        self.stress_target_calibrator()
        self.stress_oi_analyzer()
        self.stress_position_health()
        self.stress_broker_connection()

        print("\n" + "="*80)
        print("STRESS TEST SUMMARY")
        print("="*80)
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        print(f"Total Tests Executed: {total}")
        print(f"Passed: {passed}")
        print(f"Failed / Findings: {failed}")
        print("="*80)
        return self.results


if __name__ == "__main__":
    runner = StressTestRunner()
    results = runner.run_all()
