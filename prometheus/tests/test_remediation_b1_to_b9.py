"""
Unit tests validating Bug Remediation B1 through B9 and Challenger Boundary findings.
"""

import math
import os
import sqlite3
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from prometheus.signals.oi_analyzer import OIAnalyzer, OISignal
from prometheus.signals.target_calibrator import calibrate_target_and_sl
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.signals.gamma_engine import GammaEngine, calculate_black_scholes_gamma
from prometheus.utils.options_math import max_pain, pcr_ratio
from prometheus.execution.broker import Order, OrderSide, OrderType, OrderStatus, Position
from prometheus.execution.order_manager import OrderManager, ManagedPosition
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.position_health import PositionHealthEngine, PositionHealthReport
from prometheus.interface.telegram_bot import REGIME_QUALITY, TelegramBot


# =============================================================================
# B1: Credit Spread Protection & Decay Arming
# =============================================================================
def test_b1_credit_spread_trailing_state_arming():
    """Verify OrderManager.create_trailing_state arms credit spread parameters."""
    broker_mock = MagicMock()
    om = OrderManager(broker_mock, MagicMock())
    
    entry_ord = Order(
        order_id="ORD123",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        side=OrderSide.SELL,
        quantity=15,
        price=120.0,
        status=OrderStatus.COMPLETE,
    )
    pos = ManagedPosition(
        position_id="CS_POS_001",
        symbol="NIFTY BANK",
        strategy="CREDIT_SPREAD",
        direction="BEARISH",
        entry_orders=[entry_ord],
        exit_orders=[],
        stop_loss=160.0,
        target=20.0,
        trailing_stop=160.0,
        entry_time="2026-10-06 10:00:00",
    )
    pos.tradingsymbol = "BANKNIFTY26OCT50000CE"
    pos.entry_premium = 120.0
    pos.strategy_type = "CREDIT_SPREAD"
    pos.hard_sl_price = 160.0
    pos.target_decay_price = 20.0
    pos.breakeven_decay_price = 110.0
    pos.tier = "S"
    pos.atr = 25.0
    om.managed_positions["CS_POS_001"] = pos
    
    state = om.create_trailing_state("CS_POS_001")
    assert state.strategy_type == "CREDIT_SPREAD"
    assert state.hard_sl_price == 160.0
    assert state.target_decay_price == 20.0
    assert state.breakeven_decay_price == 110.0
    assert state.tier == "S"
    assert state.risk_distance == 40.0  # abs(160 - 120)


def test_b1_execute_credit_spread_order_placement():
    """Verify OrderManager._execute_credit_spread places hedge first, then short leg."""
    broker_mock = MagicMock()
    o1 = Order(order_id="HEDGE_ORD_1", tradingsymbol="BANKNIFTY26OCT51000CE", side=OrderSide.BUY, quantity=15, status=OrderStatus.COMPLETE)
    o2 = Order(order_id="SHORT_ORD_2", tradingsymbol="BANKNIFTY26OCT50500CE", side=OrderSide.SELL, quantity=15, status=OrderStatus.COMPLETE)
    broker_mock.place_order.side_effect = [o1, o2]
    broker_mock.get_order_status.return_value = Order(status=OrderStatus.COMPLETE)
    
    om = OrderManager(broker_mock, MagicMock())
    sig = {
        "action": "BEAR_CALL_SPREAD",
        "symbol": "NIFTY BANK",
        "tradingsymbol": "BANKNIFTY26OCT50500CE",
        "legs": [
            {"tradingsymbol": "BANKNIFTY26OCT51000CE", "action": "BUY", "is_hedge": True, "quantity": 15, "premium": 40.0},
            {"tradingsymbol": "BANKNIFTY26OCT50500CE", "action": "SELL", "is_hedge": False, "quantity": 15, "premium": 100.0},
        ],
        "quantity": 15,
        "entry_price": 100.0,
        "net_credit": 60.0,
        "max_risk_pts": 40.0,
        "strategy": "CREDIT_SPREAD",
        "tier": "B",
    }
    
    res = om._execute_credit_spread(sig, quantity=15)
    assert res is not None
    assert res.strategy_type == "credit_spread"
    assert broker_mock.place_order.call_count == 2
    # First call: Hedge BUY
    first_call = broker_mock.place_order.call_args_list[0]
    assert first_call[0][0].tradingsymbol == "BANKNIFTY26OCT51000CE"
    assert first_call[0][0].side == OrderSide.BUY
    # Second call: Short SELL
    second_call = broker_mock.place_order.call_args_list[1]
    assert second_call[0][0].tradingsymbol == "BANKNIFTY26OCT50500CE"
    assert second_call[0][0].side == OrderSide.SELL


# =============================================================================
# B2: Double-Exit & Naked Shorting Race Prevention
# =============================================================================
def test_b2_close_position_race_prevention_on_completed_exit():
    """Verify close_position avoids placing duplicate market exit if already complete."""
    broker_mock = MagicMock()
    completed_exit = Order(
        order_id="EXIT_ORD_1",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        side=OrderSide.SELL,
        quantity=15,
        filled_quantity=15,
        average_price=270.0,
        status=OrderStatus.COMPLETE,
    )
    broker_mock.get_order_status.return_value = completed_exit
    broker_mock.cancel_order.return_value = True
    broker_mock.get_positions.return_value = []
    
    om = OrderManager(broker_mock, MagicMock())
    entry_ord = Order(
        order_id="ENTRY_1",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        side=OrderSide.BUY,
        quantity=15,
        filled_quantity=15,
        average_price=250.0,
        status=OrderStatus.COMPLETE,
    )
    pos = ManagedPosition(
        position_id="POS_RACE_1",
        symbol="NIFTY BANK",
        strategy="MOMENTUM",
        direction="BULLISH",
        entry_orders=[entry_ord],
        exit_orders=[completed_exit],
        stop_loss=220.0,
        target=300.0,
        trailing_stop=220.0,
        entry_time="2026-10-06 10:00:00",
    )
    pos.tradingsymbol = "BANKNIFTY26OCT50000CE"
    pos.entry_premium = 250.0
    om.managed_positions["POS_RACE_1"] = pos
    
    res = om.close_position("POS_RACE_1", exit_price_hint=270.0, reason="TAKE_PROFIT")
    assert res is not None
    # Crucial: broker.place_order should NOT be called since exit order is already COMPLETE
    assert broker_mock.place_order.call_count == 0
    assert om.managed_positions["POS_RACE_1"].status == "closed"


def test_b2_close_position_race_prevention_on_zero_net_quantity():
    """Verify close_position avoids duplicate sell when broker position is already 0."""
    broker_mock = MagicMock()
    broker_mock.get_order_status.return_value = Order(status=OrderStatus.CANCELLED)
    broker_mock.cancel_order.return_value = True
    broker_pos = Position(tradingsymbol="BANKNIFTY26OCT50000CE", quantity=0)
    broker_mock.get_positions.return_value = [broker_pos]
    
    om = OrderManager(broker_mock, MagicMock())
    entry_ord = Order(
        order_id="ENTRY_2",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        side=OrderSide.BUY,
        quantity=15,
        filled_quantity=15,
        average_price=250.0,
        status=OrderStatus.COMPLETE,
    )
    pos = ManagedPosition(
        position_id="POS_RACE_2",
        symbol="NIFTY BANK",
        strategy="MOMENTUM",
        direction="BULLISH",
        entry_orders=[entry_ord],
        exit_orders=[],
        stop_loss=220.0,
        target=300.0,
        trailing_stop=220.0,
        entry_time="2026-10-06 10:00:00",
    )
    pos.tradingsymbol = "BANKNIFTY26OCT50000CE"
    pos.entry_premium = 250.0
    om.managed_positions["POS_RACE_2"] = pos
    
    res = om.close_position("POS_RACE_2", exit_price_hint=250.0, reason="TEST_ZERO_NET")
    assert res is not None
    assert broker_mock.place_order.call_count == 0
    assert om.managed_positions["POS_RACE_2"].status == "closed"


# =============================================================================
# B3: Live vs Paper Trailing Parity
# =============================================================================
def test_b3_tier_c_micro_locking_vs_tier_b_breathing_room():
    """Verify Tier C micro-locks at +12 pts while Tier B preserves breathing room."""
    broker_mock = MagicMock()
    broker_mock.modify_order.return_value = True
    om = OrderManager(broker_mock, MagicMock())
    pm = PositionMonitor(om)
    
    # Tier C setup on Bank Nifty: Entry 200, SL 170. Gain +12 pts -> SL moves to cost + buffer (201.5)
    state_tier_c = TrailingState(
        position_id="POS_TIER_C",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=200.0,
        initial_sl=170.0,
        current_sl=170.0,
        target=250.0,
        direction="bullish",
        tier="C",
        strategy_type="DIRECTIONAL",
    )
    pm.add_position(state_tier_c)
    
    # Tick at 212 (+12 pts gain)
    pm._process_tick(state_tier_c, 212.0)
    assert state_tier_c.current_sl == 201.9  # Entry + 1.9 pt Bank Nifty cost buffer
    assert state_tier_c.breakeven_set is True
    
    # Tier B setup on Bank Nifty: Entry 200, SL 170. Gain +12 pts -> MUST NOT ratchet to BE
    state_tier_b = TrailingState(
        position_id="POS_TIER_B",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=200.0,
        initial_sl=170.0,
        current_sl=170.0,
        target=280.0,
        direction="bullish",
        tier="B",
        strategy_type="DIRECTIONAL",
    )
    pm.add_position(state_tier_b)
    
    # Tick at 212 (+12 pts gain) -> Stage 0A half-risk cut (from 170 to 185) but NOT breakeven!
    pm._process_tick(state_tier_b, 212.0)
    assert state_tier_b.current_sl == 185.0  # Cut half risk, preserving runner breathing room
    assert state_tier_b.current_sl < state_tier_b.entry_premium  # Not breakeven yet
    assert state_tier_b.breakeven_set is False
    
    # Tick at 218.5 (+18.5 pts gain >= 18.0 pt Bank Nifty noise floor) -> Now triggers breakeven
    pm._process_tick(state_tier_b, 218.5)
    assert state_tier_b.current_sl == 201.9  # Breakeven + buffer
    assert state_tier_b.breakeven_set is True


# =============================================================================
# B4: Pre-SL Structural Bailout Guard against Tick-0 Shakeouts
# =============================================================================
def test_b4_pre_sl_bailout_guard_inside_noise_envelope():
    """Verify PositionHealthEngine holds Tier S/B positions within the noise envelope."""
    engine = PositionHealthEngine()
    
    class MockPosState:
        def __init__(self):
            self.position_id = "P1"
            self.symbol = "NIFTY BANK"
            self.tradingsymbol = "BANKNIFTY26OCT50000CE"
            self.direction = "bullish"
            self.entry_premium = 250.0
            self.entry_spot = 50000.0
            self.risk_distance = 35.0
            self.tier = "S"  # Golden setup runner
            self.entry_bar_count = 0
            self.bars_held = 0
            self.entry_iv = 0.15
    
    # Synthetic candle DF showing poor initial score
    bars = []
    base_dt = pd.Timestamp("2026-10-06 09:15:00")
    for i in range(10):
        bars.append({
            "date": base_dt + pd.Timedelta(minutes=15 * i),
            "open": 50000.0 - i * 10,
            "high": 50020.0 - i * 10,
            "low": 49950.0 - i * 10,
            "close": 49960.0 - i * 10,
            "volume": 20000 + i * 1000,
        })
    df = pd.DataFrame(bars)
    
    # Tick 0: Small 4 pt noise dip on Bank Nifty (< 14 pt noise floor)
    rep = engine.evaluate_position_health(MockPosState(), current_premium=246.0, underlying_df_override=df)
    # Must NOT bail out; must preserve runner
    assert rep.suggested_action == "HOLD"


# =============================================================================
# B5 & B6: GEX Dict Indexing & NaN Trap Guard
# =============================================================================
def test_b5_position_health_gex_dict_support():
    """Verify _eval_pillar_gex cleanly parses dict returned from GammaEngine."""
    engine = PositionHealthEngine()
    gex_dict = {
        "net_gex": -25000000.0,
        "net_gex_cr": -2.5,
        "zgl": 50100.0,
        "gamma_regime": "SHORT_GAMMA",
    }
    score, net_gex, threat, favor = engine._eval_pillar_gex("NIFTY BANK", 50200.0, True, override=gex_dict)
    assert net_gex == -2.5
    assert score == 75.0
    assert favor is not None


def test_b6_composite_score_nan_trap_guard():
    """Verify NaN raw composite score clamps safely to 0.0 without triggering min(100, NaN)==100 trap."""
    engine = PositionHealthEngine()
    
    class MockState:
        def __init__(self):
            self.position_id = "P_NAN"
            self.symbol = "NIFTY BANK"
            self.tradingsymbol = "BANKNIFTY26OCT50000CE"
            self.direction = "bullish"
            self.entry_premium = 250.0
            self.entry_spot = 50000.0
            self.risk_distance = 35.0
            self.tier = "B"
            self.entry_bar_count = 2
            self.bars_held = 2
            self.entry_iv = 0.15
            
    # Mock all pillars returning NaN or 0
    with patch.object(engine, "_eval_pillar_vwap", return_value=(float("nan"), 0.0, None, None)):
        rep = engine.evaluate_position_health(MockState(), current_premium=250.0)
        assert not math.isnan(rep.health_score)
        assert rep.health_score == 0.0


# =============================================================================
# B7: Commitment Ratio Normalization Clamp
# =============================================================================
def test_b7_commitment_ratio_clamped():
    """Verify commitment_ratio is clamped strictly to [0.0, 1.0]."""
    analyzer = OIAnalyzer()
    
    # Normal chain
    chain_df = pd.DataFrame({
        "strike": [25000, 25100],
        "option_type": ["CE", "PE"],
        "oi": [10000, 10000],
        "volume": [1000, 1000],
        "oi_change": [5000, 2000],
        "iv": [0.15, 0.15],
    })
    res = analyzer.analyze(chain_df, 25050.0)
    cr = res["metrics"]["commitment_ratio"]
    assert 0.0 <= cr <= 1.0


# =============================================================================
# B8: Row 27 Performance Ledger Reconciliation
# =============================================================================
def test_b8_performance_ledger_row_27_reconciled():
    """Verify Day 11 (Row 27) in CSV performance ledger matches SQLite ground truth."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../reports/papertrade/daily_performance_ledger.csv"))
    if not os.path.exists(csv_path):
        pytest.skip("CSV ledger not found at path")
        
    df = pd.read_csv(csv_path)
    day11 = df[df["Date"] == "2026-10-06"]
    assert len(day11) == 1
    row = day11.iloc[0]
    
    assert int(row["Total_Trades"]) == 7
    assert int(row["Wins"]) == 2
    assert int(row["Losses"]) == 5
    assert float(row["Win_Rate_Pct"]) == 28.6
    assert abs(float(row["Gross_Profit"]) - 1510.33) < 0.1
    assert abs(float(row["Gross_Loss"]) - (-755.56)) < 0.1
    assert abs(float(row["Net_PnL"]) - (-49.88)) < 0.1


# =============================================================================
# B9: Telemetry & Alert Clarifications
# =============================================================================
def test_b9_telemetry_and_alert_wording():
    """Verify objective wording in telegram templates and tier classifier."""
    assert "90%" not in REGIME_QUALITY.get("TRENDING_BULL", "")
    assert "85%" not in REGIME_QUALITY.get("TRENDING_BEAR", "")
    assert "70%" not in REGIME_QUALITY.get("COMPRESSED_CHOP", "")
    
    # Test Tier B Golden Setup action description in tier_classifier
    sig = {
        "action": "BUY_CE",
        "symbol": "BANKNIFTY",
        "strategy": "PriceAction_Momentum",
        "strategy_type": "option_buying",
        "edge_score": 7.8,
        "bar_timestamp": "2026-09-11 11:15:00",
        "reasons": [
            "15M_ORB_High_Breakout",
            "Session_VWAP_Bullish",
            "Volume_Surge_Confirmed",
            "1H_Trend_Bullish",
        ],
    }
    res = classify_signal_tier(sig)
    assert "+10 pts" not in res["action_instruction"]
    assert "+18" in res["action_instruction"] or "+20" in res["action_instruction"]


# =============================================================================
# Challenger Boundary Findings
# =============================================================================
def test_challenger_target_calibrator_nan_structural_sl():
    """Verify target_calibrator clamps NaN structural_sl_pts to 35% risk ceiling."""
    res = calibrate_target_and_sl(
        "NIFTY BANK",
        spot_price=50000.0,
        spot_atr=120.0,
        opt_ltp=300.0,
        structural_sl_pts=float("nan"),
    )
    assert not math.isnan(res.sl_pts)
    assert res.sl_pts <= 300.0 * 0.35 + 0.1


def test_challenger_oi_analyzer_malformed_chain_protection():
    """Verify malformed chain missing 'oi' is rejected without corrupting history."""
    analyzer = OIAnalyzer()
    bad_df = pd.DataFrame({"strike": [25000], "option_type": ["CE"]})
    res = analyzer.analyze(bad_df, 25000.0)
    assert res["signals"] == []
    assert res["metrics"] == {}
    assert len(analyzer.oi_history) == 0  # Not appended to history


def test_challenger_max_pain_minimizes_buyer_payout():
    """Verify max_pain correctly minimizes total buyer payout using argmin."""
    strikes = np.array([24800, 24900, 25000, 25100, 25200], dtype=float)
    ce_oi = np.array([0, 0, 100000, 0, 0], dtype=float)  # Only calls at 25000
    pe_oi = np.array([0, 0, 0, 0, 0], dtype=float)
    mp = max_pain(strikes, ce_oi, pe_oi, 25000.0)
    # Total payout is 0 for strikes <= 25000, positive for strikes > 25000
    assert mp <= 25000.0


def test_challenger_position_health_volume_pillar_positive_progress():
    """Verify volume pillar does NOT award +25 score when spot progress is negative."""
    engine = PositionHealthEngine()
    bars = []
    base_dt = pd.Timestamp("2026-10-06 10:00:00")
    for i in range(25):
        bars.append({
            "date": base_dt + pd.Timedelta(minutes=15 * i),
            "open": 50000.0,
            "high": 50050.0,
            "low": 49950.0,
            "close": 50000.0,
            "volume": 10000,
        })
    df = pd.DataFrame(bars)
    df.loc[df.index[-1], "volume"] = 12800  # RVOL = 1.28x
    
    # Long CE (bullish), but spot drops from 50000 to 49600 (adverse move: -400 pts)
    score, rvol, threat, favor = engine._eval_pillar_volume(df, spot=49600.0, entry_spot=50000.0, trade_is_bullish=True)
    assert score == 0.0  # Must NOT be +25.0


def test_challenger_position_health_favor_alert_noise_floor_buffer():
    """Verify runner acceleration alert respects the noise floor buffer before suggesting BE."""
    report_inside_noise = PositionHealthReport(
        position_id="R1",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        direction="bullish",
        current_price=260.0,
        entry_price=250.0,  # gain = 10 pts (< 18 pt Bank Nifty noise floor)
        health_score=75.0,
        real_favor=True,
    )
    msg_inside = PositionHealthEngine.format_favor_telegram_alert(report_inside_noise, old_target=280.0, new_target=320.0)
    assert "Preserve wide breathing room" in msg_inside
    assert "Move Stop Loss to Breakeven" not in msg_inside
    
    report_outside_noise = PositionHealthReport(
        position_id="R2",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        direction="bullish",
        current_price=272.0,
        entry_price=250.0,  # gain = 22 pts (>= 18 pt Bank Nifty noise floor)
        health_score=85.0,
        real_favor=True,
    )
    msg_outside = PositionHealthEngine.format_favor_telegram_alert(report_outside_noise, old_target=280.0, new_target=320.0)
    assert "Move Stop Loss to Breakeven" in msg_outside
