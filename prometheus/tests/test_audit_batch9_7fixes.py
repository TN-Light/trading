"""
Audit Certification Suite for Batch 9 (7 Remaining Bugs & Edge Cases).

Covers:
1. Bug 1: run_combined_mode ts indentation guard (no UnboundLocalError when position is str).
2. Bug 2: cs_sig classified at generation time inside _get_intraday_signal_for_execution.
3. Bug 3: RiskManager.pre_trade_check single-lot risk ceiling and credit spread margin approval.
4. Bug 4: DataEngine.fetch_options_chain scales NSE changeinOpenInterest contracts to shares via lot_size.
5. Bug 5: calibrate_target_and_sl calculates and passes noise-floor aware breakeven_trigger_pts.
6. Bug 6: _compute_intraday_bias zero-lookahead guard on final session date.
7. Bug 7: ZerodhaCostModel defaults transaction_charges to 0.03503% (NSE statutory rate).
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

from prometheus.risk.manager import RiskManager
from prometheus.signals.target_calibrator import calibrate_target_and_sl, TargetCalibrationResult
from prometheus.backtest.engine import ZerodhaCostModel
from prometheus.data.engine import DataEngine
from prometheus.main import Prometheus


def test_bug1_run_combined_mode_ts_indentation():
    """Bug 1: run_combined_mode must not reference ts if position is a string trade_id."""
    import inspect
    source = inspect.getsource(Prometheus.run_combined_mode)
    not_is_instance_line = None
    if_ts_line = None
    for line in source.splitlines():
        if "if not isinstance(position, str):" in line:
            not_is_instance_line = line
        elif not_is_instance_line and "if ts:" in line:
            if_ts_line = line
            break

    assert not_is_instance_line is not None
    assert if_ts_line is not None
    indent_parent = len(not_is_instance_line) - len(not_is_instance_line.lstrip())
    indent_child = len(if_ts_line) - len(if_ts_line.lstrip())
    assert indent_child > indent_parent, "if ts: must be indented inside if not isinstance(position, str):"


def test_bug2_credit_spread_classification_and_routing():
    """Bug 2: cs_sig must be classified so Tier C paper-only spreads don't override Live Option Buying."""
    prom = Prometheus.__new__(Prometheus)
    prom.initial_capital = 100000.0
    prom.data = MagicMock()
    prom.data.get_vix.return_value = 14.5
    prom.data.fetch_intraday.return_value = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-10 09:15", periods=20, freq="15min"),
        "open": [24000.0] * 20,
        "high": [24050.0] * 20,
        "low": [23950.0] * 20,
        "close": [24000.0] * 20,
        "volume": [1000] * 20,
    })
    prom.config = {"intraday": {"credit_spread": {"strategy_preference": "credit_spread"}}}
    prom.mode = "paper"

    # Setup mock scanner to return None for option buying
    prom._pa_momentum_scanner = MagicMock()
    prom._pa_momentum_scanner.scan.return_value = None

    # Setup mock credit spread strategy that returns an unclassified dict
    prom._credit_spread_strategy = MagicMock()
    prom._credit_spread_strategy.evaluate_spread.return_value = {
        "strategy": "CreditSpread_BEAR_CALL_SPREAD",
        "action": "SELL_BEAR_CALL_SPREAD",
        "spread_type": "BEAR_CALL_SPREAD",
        "signal_score": 5.0,  # Score 5.0 is Tier C (Paper Only)
        "net_credit": 25.0,
        "is_0dte": False,
        "tradingsymbol": "NIFTY26NOV24500CE",
    }

    # Evaluate priority
    with patch("prometheus.main.get", return_value="credit_spread"):
        chosen = prom._get_intraday_signal_for_execution("NIFTY", "15minute", False)

    # The credit spread strategy was called with vix=14.5
    prom._credit_spread_strategy.evaluate_spread.assert_called_once()
    assert prom._credit_spread_strategy.evaluate_spread.call_args[1]["vix"] == 14.5
    # The returned signal must now be classified with tier
    assert chosen is not None
    assert chosen.get("tier") in ("C", "D", "A", "B", "S")


def test_bug3_risk_manager_pre_trade_check_single_lot_and_spread():
    """Bug 3: pre_trade_check must approve 1-lot options within ₹850 risk and credit spreads within exposure limit."""
    rm = RiskManager(config={"max_single_position_pct": 15.0, "max_total_exposure_pct": 70.0}, initial_capital=100000.0)
    market_time = datetime(2026, 11, 10, 10, 0)

    with patch("prometheus.utils.indian_market.is_market_open", return_value=True):
        # 1. 1-lot option costing ₹18,000 (> 15% of ₹100k capital = ₹15,000)
        # Entry ₹276.92, SL ₹264.00 -> Risk per share = ₹12.92. Lot size = 65 -> Total Risk = ₹839.80 (<= ₹850)
        single_lot_trade = {
            "symbol": "NIFTY26NOV24500CE",
            "direction": "LONG",
            "entry_price": 276.92,
            "stop_loss": 264.00,
            "quantity": 65,
            "cost": 18000.0,
            "strategy_type": "option_buying",
            "risk_amount": 839.80,
        }
        res = rm.pre_trade_check(single_lot_trade, current_time=market_time)
        assert res.approved, f"Single-lot option should be approved under risk budget: {res.reason}"

        # 2. Credit spread requiring ₹60,000 margin (> 15% single position limit of ₹15,000, but <= 70% exposure limit)
        spread_trade = {
            "symbol": "NIFTY26NOV24500CE_SPREAD",
            "direction": "SHORT",
            "entry_price": 25.0,
            "stop_loss": 50.0,
            "quantity": 65,
            "cost": 60000.0,  # margin required
            "strategy_type": "credit_spread",
            "risk_amount": 850.0,
        }
        res_spread = rm.pre_trade_check(spread_trade, current_time=market_time)
        assert res_spread.approved, f"Credit spread within max exposure should be approved: {res_spread.reason}"


def test_bug4_data_engine_oi_change_lot_scaling():
    """Bug 4: fetch_options_chain must multiply NSE contracts by lot_size to match Angel One shares."""
    de = DataEngine.__new__(DataEngine)
    de.angelone_options = MagicMock()
    de.nse = MagicMock()
    de.store = MagicMock()
    de.get_spot_price = MagicMock(return_value=24000.0)
    de._normalize_options_chain = lambda df, s: df

    # Angel One returns DataFrame with oi_change == 0.0
    angel_df = pd.DataFrame([
        {"strike": 24000.0, "option_type": "CE", "oi": 500000, "oi_change": 0.0, "volume": 100000},
        {"strike": 24000.0, "option_type": "PE", "oi": 600000, "oi_change": 0.0, "volume": 120000},
    ])
    de.angelone_options.get_option_chain.return_value = angel_df

    # NSE returns oi_change in contracts (e.g. 100 contracts for NIFTY)
    de.nse.get_options_chain.return_value = {"mock": "data"}
    de.nse.parse_options_chain.return_value = pd.DataFrame([
        {"strike": 24000.0, "option_type": "CE", "oi_change": 100},
        {"strike": 24000.0, "option_type": "PE", "oi_change": -50},
    ])

    enriched = de.fetch_options_chain("NIFTY")
    # For NIFTY, lot size is 65. 100 contracts * 65 = 6500 shares.
    ce_oi_change = enriched[enriched["option_type"] == "CE"]["oi_change"].iloc[0]
    pe_oi_change = enriched[enriched["option_type"] == "PE"]["oi_change"].iloc[0]
    assert ce_oi_change == 6500.0, f"Expected 6500 shares, got {ce_oi_change}"
    assert pe_oi_change == -3250.0, f"Expected -3250 shares, got {pe_oi_change}"


def test_bug5_target_calibrator_breakeven_trigger_pts():
    """Bug 5: calibrate_target_and_sl must calculate and return noise-floor aware breakeven_trigger_pts."""
    res = calibrate_target_and_sl(
        symbol="BANKNIFTY",
        spot_price=50000.0,
        spot_atr=200.0,
        opt_ltp=300.0,
        edge_score=8.5,
        tier="B",
        is_htf_aligned=True,
    )
    # BANKNIFTY noise floor is 18.0 pts
    assert res.breakeven_trigger_pts >= 18.0
    assert res.breakeven_trigger_pts == round(max(18.0, res.target_gain_pts * 0.5), 1)

    # Verify 5-tuple backwards compatibility for unpacking
    tgt, sl, tgt_px, sl_px, is_comp = res
    assert tgt == res.target_gain_pts
    assert sl == res.sl_pts


def test_bug6_compute_intraday_bias_final_session():
    """Bug 6: _compute_intraday_bias must not overwrite today's bias with today's in-progress bars."""
    prom = Prometheus.__new__(Prometheus)

    # 2 sessions: Day 1 (yesterday) and Day 2 (today)
    timestamps = [
        # Day 1: 15 bars
        *[f"2026-11-09 09:{15+i:02d}:00" for i in range(15)],
        # Day 2: 15 bars
        *[f"2026-11-10 09:{15+i:02d}:00" for i in range(15)],
    ]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0 + i for i in range(30)],
        "high": [105.0 + i for i in range(30)],
        "low": [95.0 + i for i in range(30)],
        "close": [102.0 + i for i in range(30)],
        "volume": [1000] * 30,
    })

    bias_map = prom._compute_intraday_bias(df)
    # Day 1's close should determine bias for Day 2 (2026-11-10).
    # Day 2's bars must NOT project onto a non-existent Day 3 or overwrite Day 2.
    assert "2026-11-10" in bias_map
    assert "2026-11-11" not in bias_map


def test_bug7_zerodha_cost_model_default_transaction_charges():
    """Bug 7: ZerodhaCostModel default transaction_charges must be 0.03503% (NSE options rate)."""
    cm = ZerodhaCostModel()
    # 0.03503% / 100 = 0.0003503
    assert abs(cm.transaction_charges_pct - 0.0003503) < 1e-7
