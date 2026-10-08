"""
Unit and integration tests for quantitative trading engine remediations:
1. PositionTracker and PositionMonitor compulsive 8-pillar health check & edge-aware trailing stops.
2. AngelOneOptionChain session baseline OI seeding from SQLite and delta_volume propagation.
3. DataEngine options chain zero-OI enrichment fallback.
4. OIAnalyzer wider-ATM / delta_oi fallback for commitment_ratio calculation.
5. LiveBridge fallback commitment_ratio computation.
6. FillSimulator 2-leg spread pricing source verification.
7. Cost buffer recalibration (4.50 Bank Nifty, 5.0 Sensex, 1.5 Nifty).
8. PositionMonitor in-trade 5-minute heartbeat notification callback.
"""
import pytest
from datetime import datetime, date
from unittest.mock import MagicMock, patch

from prometheus.papertrade.types import Position, Direction
from prometheus.papertrade.fill_simulator import FillSimulator, PriceFeed, FillResult
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.data.angelone_options import AngelOneOptionChain, ContractOISnapshot
from prometheus.signals.oi_analyzer import OIAnalyzer
from prometheus.data.engine import DataEngine
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
import pandas as pd


class DummySpreadFeed:
    def __init__(self, quotes: dict):
        self.quotes = quotes

    def get_ltp(self, instrument: str) -> float:
        return float(self.quotes.get(instrument, 0.0))

    def get_quote(self, instrument: str):
        ltp = self.get_ltp(instrument)
        if ltp > 0:
            return (ltp, ltp * 0.99, ltp * 1.01)
        return None


def test_fill_simulator_spread_decomposition_and_source():
    """Verify 2-leg spread decomposition returns live_spread_2leg."""
    feed = DummySpreadFeed({
        "BANKNIFTY50000CE": 250.0,
        "BANKNIFTY50500CE": 100.0,
    })
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=True)
    res = sim.fill("BANKNIFTY50000CE/BANKNIFTY50500CE", Direction.SHORT, side="SELL")
    assert res.source == "live_spread_2leg"
    # When selling credit spread: sell short leg at bid (250*0.99=247.5), buy long leg at ask (100*1.01=101.0)
    assert res.fill_price == pytest.approx(146.5, rel=1e-2)


def test_cost_buffer_recalibration_1lot_vs_legacy():
    """Verify cost buffer recalibration for 1-lot sizing and legacy backward compatibility."""
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))

    # 1-lot Bank Nifty (quantity=15 or 1): cost_buffer_pts should be 4.50
    pos_bn_1lot = Position(
        trade_id="POS-BN-1LOT",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT50000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=15,
        entry_price=200.0,
        entry_time=datetime.now(),
        stop_loss=170.0,
        target=250.0,
        max_bars=16,
    )
    tracker.open_positions[pos_bn_1lot.trade_id] = pos_bn_1lot
    tracker._maybe_advance_trailing_stop(pos_bn_1lot, 218.0)  # +18 pts gain
    assert pos_bn_1lot.breakeven_set is True
    assert pos_bn_1lot.stop_loss == pytest.approx(204.50, rel=1e-2)

    # Legacy 2-lot Bank Nifty (quantity=30): cost_buffer_pts should be 1.90
    pos_bn_2lot = Position(
        trade_id="POS-BN-2LOT",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT50000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=30,
        entry_price=200.0,
        entry_time=datetime.now(),
        stop_loss=170.0,
        target=250.0,
        max_bars=16,
    )
    tracker.open_positions[pos_bn_2lot.trade_id] = pos_bn_2lot
    tracker._maybe_advance_trailing_stop(pos_bn_2lot, 218.0)
    assert pos_bn_2lot.breakeven_set is True
    assert pos_bn_2lot.stop_loss == pytest.approx(201.90, rel=1e-2)


def test_weak_edge_capital_defense_lock():
    """Verify weak edge triggers early breakeven ratchet in PositionTracker."""
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))

    pos = Position(
        trade_id="POS-WEAK-EDGE",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT50000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=15,
        entry_price=200.0,
        entry_time=datetime.now(),
        stop_loss=170.0,
        target=250.0,
        max_bars=16,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Mock health engine to return weak edge (health_score = -15)
    mock_report = MagicMock()
    mock_report.health_score = -15.0
    mock_report.net_gex = 5e7
    mock_report.zgl = 51000.0
    mock_report.commitment_ratio = 0.05
    mock_report.suggested_action = "TIGHTEN_SL"
    mock_report.real_threat = False
    mock_report.real_favor = False

    tracker.health_engine = MagicMock()
    tracker.health_engine.evaluate_position_health.return_value = mock_report

    # Gain +6.0 pts (greater than cost_buffer 4.50 + 1.0 = 5.50 pts, but well below normal 18 pt threshold)
    tracker._maybe_advance_trailing_stop(pos, 206.0)
    assert pos.breakeven_set is True
    assert pos.stop_loss == pytest.approx(204.50, rel=1e-2)


def test_oi_analyzer_fallback_wider_window():
    """Verify OIAnalyzer falls back to wider window if 2% ATM window has 0 oi_change."""
    analyzer = OIAnalyzer()
    spot = 24000.0

    # Option chain where 2% ATM window (23520 - 24480) has 0 oi_change,
    # but strike 24600 (2.5% away, within 3.5% wider window) has significant oi_change and volume
    chain_data = [
        {"strike": 24000.0, "option_type": "CE", "oi": 50000, "oi_change": 0, "volume": 10000, "delta_oi": 0},
        {"strike": 24000.0, "option_type": "PE", "oi": 50000, "oi_change": 0, "volume": 10000, "delta_oi": 0},
        {"strike": 24600.0, "option_type": "CE", "oi": 50000, "oi_change": 8000, "volume": 20000, "delta_oi": 8000},
        {"strike": 24600.0, "option_type": "PE", "oi": 50000, "oi_change": 2000, "volume": 20000, "delta_oi": 2000},
    ]
    df = pd.DataFrame(chain_data)
    res = analyzer.analyze(df, spot_price=spot)

    assert "commitment_ratio" in res["metrics"]
    assert res["metrics"]["commitment_ratio"] == 0.167


def test_angelone_delta_volume_tracking():
    """Verify AngelOneOptionChain records delta_volume in fetch_market_data."""
    class FakeSmartConnect:
        def __init__(self):
            self.market_data_responses = []

        def getMarketData(self, mode, payload):
            if self.market_data_responses:
                return self.market_data_responses.pop(0)
            return {"status": False}

    class FakeFetcher:
        def __init__(self):
            self._obj = FakeSmartConnect()

        def _ensure_connected(self):
            return True

    fetcher = FakeFetcher()
    chain = AngelOneOptionChain(fetcher)

    contracts = [
        {"symboltoken": "999", "tradingsymbol": "NIFTY24000CE", "strike": 24000, "option_type": "CE", "expiry": "2026-10-29"}
    ]

    fetcher._obj.market_data_responses = [
        {"status": True, "data": {"fetched": [{"symbolToken": "999", "ltp": 100.0, "tradeVolume": 5000, "opnInterest": 20000}]}}
    ]
    df1 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df1["delta_volume"].iloc[0] == 0

    fetcher._obj.market_data_responses = [
        {"status": True, "data": {"fetched": [{"symbolToken": "999", "ltp": 105.0, "tradeVolume": 7500, "opnInterest": 22000}]}}
    ]
    df2 = chain.fetch_market_data(contracts, trading_date="2026-10-01")
    assert df2["delta_volume"].iloc[0] == 2500
    assert df2["delta_oi"].iloc[0] == 2000


def test_position_monitor_heartbeat_callback():
    """Verify PositionMonitor fires on_heartbeat at 5-minute elapsed interval."""
    broker_mock = MagicMock()
    order_mgr_mock = MagicMock()
    heartbeat_mock = MagicMock()

    pm = PositionMonitor(
        broker=order_mgr_mock,
        on_heartbeat=heartbeat_mock,
    )

    state = TrailingState(
        position_id="POS_HEARTBEAT_TEST",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        symbol="NIFTY BANK",
        entry_premium=200.0,
        initial_sl=170.0,
        current_sl=170.0,
        target=250.0,
        direction="bullish",
        entry_time="2026-10-08 09:15:00",
    )
    pm.add_position(state)

    # Patch _get_elapsed_seconds to simulate 310 seconds elapsed (milestone 1 = 5 min)
    with patch.object(pm, "_get_elapsed_seconds", return_value=310.0):
        pm._process_tick(state, 205.0)

    heartbeat_mock.assert_called_once()
    args, kwargs = heartbeat_mock.call_args
    assert args[0].position_id == "POS_HEARTBEAT_TEST"
    assert args[1] == 205.0
    assert args[2] == 310
    assert args[3] == 1  # 5-minute milestone 1


def test_angelone_lookup_previous_oi_baseline_from_sqlite():
    """Verify AngelOneOptionChain seeds session baseline from SQLite."""
    fetcher = MagicMock()
    chain = AngelOneOptionChain(fetcher)

    # With real DB in data/prometheus.db, querying 54200.0 CE on Bank Nifty before 2026-10-08
    prev_oi = chain._lookup_previous_oi_baseline(
        token="999",
        tradingsymbol="BANKNIFTY27OCT54200CE",
        trading_date="2026-10-08",
    )
    # Yesterday 2026-10-07 had 80670 recorded in data/prometheus.db
    assert prev_oi == 80670


def test_data_engine_enrich_zero_oi_from_nse():
    """Verify DataEngine fetch_options_chain enriches zero-oi Angel One chains with NSE."""
    from prometheus.data.engine import NSEDirectFeed
    engine = DataEngine()
    engine.angelone_options = MagicMock()
    engine.nse = MagicMock()
    engine.nse.parse_options_chain.side_effect = NSEDirectFeed().parse_options_chain
    engine.store = MagicMock()
    engine.get_spot_price = MagicMock(return_value=25000.0)

    # Angel One returns chain with oi_change == 0
    ao_df = pd.DataFrame([
        {"strike": 25000.0, "option_type": "CE", "oi": 10000, "oi_change": 0, "volume": 5000, "ltp": 120.0},
        {"strike": 25000.0, "option_type": "PE", "oi": 12000, "oi_change": 0, "volume": 6000, "ltp": 110.0},
    ])
    engine.angelone_options.get_option_chain.return_value = ao_df

    # NSE returns chain with non-zero changeinOpenInterest
    nse_raw = {"records": {"data": [
        {"strikePrice": 25000.0, "expiryDate": "2026-10-29", "CE": {"changeinOpenInterest": 3500, "openInterest": 10000}, "PE": {"changeinOpenInterest": -1200, "openInterest": 12000}}
    ]}}
    engine.nse.get_options_chain.return_value = nse_raw

    enriched = engine.fetch_options_chain("NIFTY 50")
    assert not enriched.empty
    ce_row = enriched[enriched["option_type"] == "CE"].iloc[0]
    pe_row = enriched[enriched["option_type"] == "PE"].iloc[0]
    assert ce_row["oi_change"] == 3500
    assert pe_row["oi_change"] == -1200


def test_live_bridge_build_signal_notification_commitment_ratio_fallback(tmp_path):
    """Verify LivePaperCapture computes fallback commitment_ratio if missing or 0.0."""
    data_engine_mock = MagicMock()
    data_engine_mock.get_spot_price.return_value = 24000.0
    data_engine_mock.fetch_options_chain.return_value = pd.DataFrame([
        {"strike": 24000.0, "option_type": "CE", "oi": 50000, "oi_change": 10000, "volume": 20000, "delta_oi": 0},
        {"strike": 24000.0, "option_type": "PE", "oi": 50000, "oi_change": 10000, "volume": 20000, "delta_oi": 0},
    ])

    csv_file = str(tmp_path / "test_ledger.csv")
    config = CaptureConfig(enabled=True, csv_path=csv_file)
    bridge = LivePaperCapture(config=config, ltp_source=MagicMock(), data_engine=data_engine_mock)

    signal = {
        "symbol": "NIFTY 50",
        "action": "BUY",
        "direction": "bullish",
        "strike": 24000,
        "option_type": "CE",
        "entry_price": 100.0,
        "stop_loss": 80.0,
        "target": 140.0,
        "commitment_ratio": 0.0,  # Zero in incoming signal
        "spot_price": 24000.0,
    }

    notif = bridge._build_signal_notification(signal)
    assert notif.commitment_ratio is not None
    assert notif.commitment_ratio == pytest.approx(0.500, rel=1e-2)


def test_position_monitor_cost_buffer_recalibration_1lot_vs_legacy():
    """Verify PositionMonitor uses 4.50 for 1-lot Bank Nifty positions while preserving legacy 1.90."""
    om = MagicMock()
    pm = PositionMonitor(broker=om)

    # 1-lot Bank Nifty (qty=15): should use cost_buffer_pts = 4.50 -> BE at entry 200 + 4.50 = 204.50
    state_1lot = TrailingState(
        position_id="POS_MON_1LOT",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=200.0,
        initial_sl=170.0,
        current_sl=170.0,
        target=250.0,
        direction="bullish",
        quantity=15,
        tier="B",
    )
    pm.add_position(state_1lot)
    pm._process_tick(state_1lot, 218.0)  # +18 pts gain (Bank Nifty noise floor)
    assert state_1lot.breakeven_set is True
    assert state_1lot.current_sl == pytest.approx(204.50, rel=1e-2)

    # Legacy 2-lot Bank Nifty (qty=30): should use cost_buffer_pts = 1.90 -> BE at 201.90
    state_2lot = TrailingState(
        position_id="POS_MON_2LOT",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=200.0,
        initial_sl=170.0,
        current_sl=170.0,
        target=250.0,
        direction="bullish",
        quantity=30,
        tier="B",
    )
    pm.add_position(state_2lot)
    pm._process_tick(state_2lot, 218.0)
    assert state_2lot.breakeven_set is True
    assert state_2lot.current_sl == pytest.approx(201.90, rel=1e-2)


def test_fill_simulator_missing_leg_quote_does_not_fill_fake_spread():
    """Verify FillSimulator refuses 0.05 fake fill when short leg quote is missing."""
    # Short leg has 0.0 LTP, Long leg has 100.0 LTP
    feed = DummySpreadFeed({
        "BANKNIFTY50000CE": 0.0,
        "BANKNIFTY50500CE": 100.0,
    })
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=True)
    res = sim.fill("BANKNIFTY50000CE/BANKNIFTY50500CE", Direction.SHORT, price_hint=140.0, side="SELL")
    # Must NOT fill at 0.05; must fall back to price_hint
    assert res.fill_price == pytest.approx(140.0, rel=1e-2)
    assert res.source != "live_spread_2leg"


def test_strong_edge_preserves_wide_trailing_breathing_room():
    """Verify strong edge scales trailing stop buffer to preserve runner breathing room."""
    # 1. PositionTracker Stage 5 HWM trail:
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))
    pos = Position(
        trade_id="POS-STRONG-RUNNER",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT50000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=15,
        entry_price=200.0,
        entry_time=datetime.now(),
        stop_loss=170.0,  # risk_distance = 30
        target=350.0,
        breakeven_set=True,
        trailing_floor=0.70,
        high_water_mark=320.0,  # progress = (320-200)/30 = 4.0R (>= 3.5R for Stage 5)
        max_bars=16,
    )
    tracker.open_positions[pos.trade_id] = pos

    mock_report = MagicMock()
    mock_report.health_score = 40.0  # Strong conviction
    mock_report.real_favor = True
    mock_report.net_gex = -2.5  # -2.5 Cr Short Gamma acceleration
    mock_report.zgl = 49000.0
    mock_report.commitment_ratio = 0.55
    mock_report.suggested_action = "EXPAND_TARGET"
    mock_report.real_threat = False

    tracker.health_engine = MagicMock()
    tracker.health_engine.evaluate_position_health.return_value = mock_report

    # High water mark is 320. For strong edge, trail_buf = 0.08 * 30 = 2.4 pts -> trail_candidate = 320 - 2.4 = 317.6
    # (versus weak edge 0.02 * 30 = 0.6 -> 319.4)
    tracker._maybe_advance_trailing_stop(pos, 320.0)
    assert pos.stop_loss == pytest.approx(317.60, rel=1e-2)

    # 2. PositionMonitor Stage 4 dynamic trail:
    om = MagicMock()
    pm = PositionMonitor(broker=om)
    pm.health_engine = MagicMock()
    pm.health_engine.evaluate_position_health.return_value = mock_report

    state = TrailingState(
        position_id="POS_MON_RUNNER",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT50000CE",
        entry_premium=200.0,
        initial_sl=170.0,
        current_sl=221.0,
        target=350.0,
        direction="bullish",
        quantity=15,
        breakeven_set=True,
        trailing_activated=True,
        trailing_stage2=True,
        trailing_stage3=True,  # In Stage 4
        premium_hwm=300.0,
    )
    pm.add_position(state)
    # Current price 300.0. Entry 200.0. Gain = 100.
    # Strong edge uses trail_mult = 0.40 -> trail_offset = 100 * 0.40 = 40 -> dynamic_sl = 300 - 40 = 260.0
    # Floor SL = 200 + 30 * 0.70 = 221.0. New SL = max(221, 260) = 260.0
    pm._process_tick(state, 300.0)
    assert state.current_sl == pytest.approx(260.0, rel=1e-2)


def test_zgl_integration_triggers_defensive_lock_on_adverse_boundary():
    """Verify spot crossing below ZGL for bullish trade triggers defensive weak-edge lock."""
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))
    pos = Position(
        trade_id="POS-ZGL-DEFENSE",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT50000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=15,
        entry_price=200.0,
        entry_spot=50500.0,
        entry_time=datetime.now(),
        stop_loss=170.0,
        target=250.0,
        max_bars=16,
    )
    tracker.open_positions[pos.trade_id] = pos

    mock_report = MagicMock()
    mock_report.health_score = -5.0  # Deteriorating health
    mock_report.net_gex = 1.0
    mock_report.zgl = 50600.0  # ZGL is above current spot (50500 < 50600 -> zgl_adverse)
    mock_report.commitment_ratio = 0.20
    mock_report.suggested_action = "HOLD"
    mock_report.real_threat = False
    mock_report.real_favor = False

    tracker.health_engine = MagicMock()
    tracker.health_engine.evaluate_position_health.return_value = mock_report

    # Gain +6.0 pts >= cost_buffer 4.50 + 1.0 = 5.50 pts. Since zgl_adverse and health <= 0, weak edge locks early
    tracker._maybe_advance_trailing_stop(pos, 206.0)
    assert pos.breakeven_set is True
    assert pos.stop_loss == pytest.approx(204.50, rel=1e-2)


def test_oi_analyzer_delta_volume_poll_volume():
    """Verify OIAnalyzer uses delta_volume without full-day volume denominator dilution."""
    analyzer = OIAnalyzer()
    spot = 24000.0

    chain_data = [
        {"strike": 24000.0, "option_type": "CE", "oi": 500000, "oi_change": 0, "volume": 10000000, "delta_oi": 5000, "delta_volume": 10000},
        {"strike": 24000.0, "option_type": "PE", "oi": 500000, "oi_change": 0, "volume": 10000000, "delta_oi": 5000, "delta_volume": 10000},
    ]
    df = pd.DataFrame(chain_data)
    res = analyzer.analyze(df, spot_price=spot)

    # eff_oi_change = 10000, eff_volume = 20000 -> raw_cr = 0.50
    assert "commitment_ratio" in res["metrics"]
    assert res["metrics"]["commitment_ratio"] == 0.500


def test_live_bridge_fallback_extracts_spot_from_chain_underlying(tmp_path):
    """Verify LivePaperCapture extracts spot from chain_df['underlying'] when entry_spot is 0."""
    data_engine_mock = MagicMock()
    data_engine_mock.get_spot_price.return_value = 0.0  # Spot lookup returns 0
    data_engine_mock.fetch_options_chain.return_value = pd.DataFrame([
        {"strike": 24000.0, "option_type": "CE", "oi": 50000, "oi_change": 10000, "volume": 20000, "underlying": 24000.0},
        {"strike": 24000.0, "option_type": "PE", "oi": 50000, "oi_change": 10000, "volume": 20000, "underlying": 24000.0},
    ])

    csv_file = str(tmp_path / "test_ledger_spot.csv")
    config = CaptureConfig(enabled=True, csv_path=csv_file)
    bridge = LivePaperCapture(config=config, ltp_source=MagicMock(), data_engine=data_engine_mock)

    signal = {
        "symbol": "NIFTY 50",
        "action": "BUY",
        "direction": "bullish",
        "strike": 24000,
        "option_type": "CE",
        "entry_price": 100.0,
        "stop_loss": 80.0,
        "target": 140.0,
        "commitment_ratio": 0.0,
        "spot_price": 0.0,  # Signal has 0 spot
    }

    notif = bridge._build_signal_notification(signal)
    assert notif.commitment_ratio is not None
    assert notif.commitment_ratio == pytest.approx(0.500, rel=1e-2)


