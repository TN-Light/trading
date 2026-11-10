import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from prometheus.data.engine import DataEngine, NSEDirectFeed
from prometheus.signals.oi_analyzer import OIAnalyzer
from prometheus.signals.target_calibrator import calibrate_target_and_sl
from prometheus.execution.order_manager import OrderManager
from prometheus.risk.manager import RiskManager
from prometheus.papertrade.position_tracker import CostModel, PositionTracker, Position


# -------------------------------------------------------------------------
# Test 1: DataEngine.get_spot_price Angel One query
# -------------------------------------------------------------------------
def test_get_spot_price_angelone():
    engine = DataEngine.__new__(DataEngine)
    engine.kite = MagicMock()
    engine.kite.is_connected.return_value = False
    
    # Mock angelone
    mock_ao = MagicMock()
    mock_ao._ensure_connected.return_value = True
    mock_ao.INDEX_TOKENS = {"NIFTY 50": "99926000"}
    mock_ao.INDEX_EXCHANGES = {"NIFTY 50": "NSE"}
    mock_ao.STOCK_TOKENS = {}
    mock_ao.rate_limiter = MagicMock()
    mock_ao.rate_limiter.acquire.return_value = True
    mock_ao._obj.ltpData.return_value = {
        "status": True,
        "data": {"ltp": "24350.50"}
    }
    engine.angelone = mock_ao

    price = engine.get_spot_price("NIFTY")
    assert price == 24350.50
    mock_ao._obj.ltpData.assert_called_once_with("NSE", "NIFTY 50", "99926000")


# -------------------------------------------------------------------------
# Test 2: run_backtest bias computation does not throw NameError
# -------------------------------------------------------------------------
def test_run_backtest_bias_no_nameerror():
    from prometheus.main import Prometheus
    app = Prometheus.__new__(Prometheus)
    
    # Mock data_hourly and data_daily
    dates = pd.date_range("2026-01-01", periods=15, freq="1h")
    df_hourly = pd.DataFrame({
        "timestamp": dates,
        "open": [100 + i for i in range(15)],
        "high": [105 + i for i in range(15)],
        "low": [95 + i for i in range(15)],
        "close": [102 + i for i in range(15)],
        "volume": [1000 for _ in range(15)],
    })

    # Calling _compute_intraday_bias
    bias_map = app._compute_intraday_bias(df_hourly)
    assert isinstance(bias_map, dict)


# -------------------------------------------------------------------------
# Test 3: OIAnalyzer computes delta OI & ATM volume metrics
# -------------------------------------------------------------------------
def test_oi_analyzer_delta_metrics():
    analyzer = OIAnalyzer()
    
    # Construct synthetic option chain around spot 24000
    chain_df = pd.DataFrame([
        {"strike": 23800, "option_type": "PE", "oi": 50000, "oi_change": 12000, "volume": 15000},
        {"strike": 23900, "option_type": "PE", "oi": 40000, "oi_change": 8000, "volume": 20000},
        {"strike": 24000, "option_type": "PE", "oi": 60000, "oi_change": 15000, "volume": 35000},
        {"strike": 24000, "option_type": "CE", "oi": 55000, "oi_change": -5000, "volume": 40000},
        {"strike": 24100, "option_type": "CE", "oi": 70000, "oi_change": 25000, "volume": 30000},
        {"strike": 24200, "option_type": "CE", "oi": 45000, "oi_change": 10000, "volume": 12000},
    ])
    
    res = analyzer.analyze(chain_df, spot_price=24000.0)
    metrics = res.get("metrics", {})
    
    assert "call_wall_delta_oi" in metrics
    assert "put_wall_delta_oi" in metrics
    assert "atm_ce_delta_oi" in metrics
    assert "atm_pe_delta_oi" in metrics
    assert "atm_option_volume" in metrics
    assert "writer_trapping_bull" in metrics
    
    # Strongest call wall above 24000 is 24100 (oi 70000) with delta 25000
    assert metrics["strongest_resistance"] == 24100
    assert metrics["call_wall_delta_oi"] == 25000.0

    # ATM CE delta is -5000, ATM PE delta is +15000 -> writer_trapping_bull = True
    assert metrics["atm_ce_delta_oi"] == -5000.0
    assert metrics["atm_pe_delta_oi"] == 15000.0
    assert metrics["writer_trapping_bull"] is True
    assert metrics["atm_option_volume"] == 40000.0


# -------------------------------------------------------------------------
# Test 4: TargetCalibrator Golden Balance Rejection & SL Cap
# -------------------------------------------------------------------------
def test_golden_balance_rejection_and_sl_cap():
    # 1. Setup where Target * 0.80 < actual_noise_floor must be REJECTED
    # SENSEX noise floor is 22.0. If opt_ltp = 40.0, target_gain_pts = 40 * 0.45 = 18.0
    # golden_sl_cap = 18.0 * 0.80 = 14.4 < 22.0 -> trade is structurally unviable and rejected!
    res_rejected = calibrate_target_and_sl(
        symbol="SENSEX",
        spot_price=77000.0,
        spot_atr=100.0,
        opt_ltp=40.0,
        edge_score=8.5,
        tier="S",
        is_htf_aligned=True,
        gamma_regime="SHORT_GAMMA",
        net_gex=-100000.0,
        structural_sl_pts=30.0,
    )
    assert res_rejected.is_rejected is True
    assert "Reward/Risk Inverted" in res_rejected.rejection_reason

    # 2. Valid setup: Target = 50.0 pts on Bank Nifty (noise floor 20.0)
    res_valid = calibrate_target_and_sl(
        symbol="BANKNIFTY",
        spot_price=50000.0,
        spot_atr=350.0,
        opt_ltp=250.0,
        edge_score=8.5,
        tier="S",
        is_htf_aligned=True,
        gamma_regime="SHORT_GAMMA",
        net_gex=-100000.0,
        structural_sl_pts=45.0,
    )
    assert res_valid.is_rejected is False
    # SL must be clamped to golden_sl_cap (target_gain_pts * 0.80)
    assert res_valid.sl_pts <= round(res_valid.target_gain_pts * 0.80, 1)
    # Gross R:R >= 1.25:1
    assert res_valid.target_gain_pts >= (1.25 * res_valid.sl_pts - 0.1)


# -------------------------------------------------------------------------
# Test 5: OrderManager spread cost & RiskManager 100k sizing
# -------------------------------------------------------------------------
def test_order_manager_spread_cost_and_100k_sizing():
    rm = RiskManager(config={}, initial_capital=100000.0)
    om = OrderManager(broker=MagicMock(), risk_manager=rm)

    # 1. Credit spread passes margin_required as trade_cost to pre_trade_check
    spread_signal = {
        "symbol": "NIFTY",
        "action": "BEAR_CALL_SPREAD",
        "direction": "SHORT",
        "entry_price": 25.0,
        "stop_loss": 55.0,
        "strategy_type": "credit_spread",
        "margin_required": 55000.0,
        "quantity": 65,
    }
    
    with patch.object(rm, "pre_trade_check") as mock_check:
        mock_check.return_value = MagicMock(approved=False, reason="Test block", violations=[])
        om.execute_signal(spread_signal, confirm=False)
        called_arg = mock_check.call_args[0][0]
        # Must pass margin_required (55000), NOT entry * quantity (25 * 65 = 1625)
        assert called_arg["cost"] == 55000.0

    # 2. Option buying on 100k account: single-lot floor allows 1 lot for Rs 240 option (risk <= 850)
    sizing = rm.calculate_position_size(entry_price=240.0, stop_loss=227.0, lot_size=65)
    # 13 pts risk on 65 qty = Rs 845 <= Rs 850 budget cap
    assert sizing["lots"] == 1
    assert sizing["quantity"] == 65


# -------------------------------------------------------------------------
# Test 6: Statutory Rates and Spread Turnover
# -------------------------------------------------------------------------
def test_statutory_rates_and_spread_turnover():
    cm = CostModel()
    # Check 0.03503% exchange rate
    costs_single = cm.calculate_trade_cost(entry_notional=10000.0, exit_notional=12000.0, is_spread=False)
    assert costs_single > 0

    # Spread notional should reflect gross legs turnover
    costs_spread = cm.calculate_trade_cost(entry_notional=35000.0, exit_notional=20000.0, is_spread=True)
    assert costs_spread > costs_single


# -------------------------------------------------------------------------
# Test 7: NSEDirectFeed contracts flag
# -------------------------------------------------------------------------
def test_nsedirect_feed_contracts_flag():
    feed = NSEDirectFeed()
    raw_json = {
        "records": {
            "data": [
                {
                    "strikePrice": 24000,
                    "expiryDate": "26-Nov-2026",
                    "CE": {"lastPrice": 150.0, "openInterest": 1000, "totalTradedVolume": 500},
                    "PE": {"lastPrice": 120.0, "openInterest": 1200, "totalTradedVolume": 600},
                }
            ]
        }
    }
    df = feed.parse_options_chain(raw_json)
    assert not df.empty
    assert df.attrs.get("is_contracts") is True
    assert "source" in df.columns
    assert (df["source"] == "nse_direct").all()
