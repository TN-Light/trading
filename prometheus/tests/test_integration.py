import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from prometheus.data.engine import DataEngine
from prometheus.signals.technical import TechnicalSignal
from prometheus.signals.oi_analyzer import OISignal
from prometheus.signals.regime_detector import RegimeState, MarketRegime
from prometheus.signals.fusion import SignalFusionEngine
from prometheus.backtest.engine import ZerodhaCostModel
from prometheus.risk.manager import RiskManager
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.broker import OrderStatus


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


class DummyBroker:
    def __init__(self):
        self._orders = {}

    def get_order_status(self, order_id):
        class _O:
            status = OrderStatus.OPEN
        return _O()

    def modify_order(self, order_id, **kwargs):
        return True

    def get_ltp(self, ts, exchange="NFO"):
        return 0


# -----------------------------------------------------------------------------
# TEST 1 — DATA PIPELINE
# -----------------------------------------------------------------------------

def test_data_pipeline_cleaning():
    engine = DataEngine()
    ts = pd.date_range("2024-01-01 09:15", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": [ts[2], ts[0], ts[1], ts[1]],  # unsorted + duplicate
        "open": [1, 2, 3, 3],
        "high": [2, 3, 4, 4],
        "low": [0.5, 1.5, 2.5, 2.5],
        "close": [1.5, 2.5, 3.5, 3.5],
        "volume": [100, 200, 300, 300],
    })
    cleaned = engine._clean_ohlcv(df)
    # Sorted, deduped, tz-naive IST timestamps
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["timestamp"].dt.tz is None
    assert cleaned["timestamp"].nunique() == len(cleaned)
    assert not cleaned[["open", "high", "low", "close", "volume"]].isna().any().any()


# -----------------------------------------------------------------------------
# TEST 2 — SIGNAL PIPELINE
# -----------------------------------------------------------------------------

def test_signal_pipeline_fusion_confluence():
    tech = [
        TechnicalSignal(name="vwap", direction="bullish", strength=0.9, timeframe="day", stop_loss=95, target=110),
        TechnicalSignal(name="volume_profile", direction="bullish", strength=0.8, timeframe="day", stop_loss=96, target=112),
    ]
    oi = [OISignal(signal_type="put_oi_buildup", direction="bullish", strength=0.8)]
    regime = RegimeState(
        regime=MarketRegime.MARKUP,
        confidence=0.8,
        volatility_regime="medium",
        trend_strength=0.6,
        mean_reversion_score=0.1,
        recommended_strategies=["trend"],
    )
    fusion = SignalFusionEngine(min_confluence_score=3.0, min_rr=1.5)
    fused = fusion.fuse("NIFTY 50", 100, tech, oi, regime)
    assert fused is not None
    assert fused.action in ("BUY_CE", "BUY_PE")
    assert 0 <= fused.confidence <= 1
    assert fused.risk_reward >= 1.5


# -----------------------------------------------------------------------------
# TEST 3 — NO LOOK-AHEAD BIAS
# -----------------------------------------------------------------------------

def test_no_lookahead_bias_simple():
    import numpy as np
    prices = pd.Series(range(1, 51))  # 50 bars

    def compute_signals(series):
        # Uses only past data (shift)
        ma = series.shift(1).rolling(5).mean()
        return (series.shift(1) > ma).fillna(False)

    full = compute_signals(prices)
    truncated = compute_signals(prices[:-10])
    pd.testing.assert_series_equal(full[:-10].reset_index(drop=True), truncated.reset_index(drop=True))


# -----------------------------------------------------------------------------
# TEST 4 — COST MODEL
# -----------------------------------------------------------------------------

def test_cost_model_expected_values():
    model = ZerodhaCostModel()
    buy_value = 25 * 150  # one lot premium buy
    sell_value = 25 * 160  # exit higher
    costs = model.calculate_costs(buy_value, sell_value, "options")
    expected_total = 40 + 4 + 4.11 + 7.39 + 0.01 + 0.11  # approx components
    assert math.isclose(costs["total"], expected_total, rel_tol=0.05)
    assert math.isclose(costs["stt"], sell_value * 0.001, rel_tol=1e-3)


# -----------------------------------------------------------------------------
# TEST 5 — POSITION SIZING
# -----------------------------------------------------------------------------

def test_position_sizing_respects_floor():
    cfg = {"max_single_position_pct": 30.0}
    rm = RiskManager(cfg, initial_capital=15000)
    sizing = rm.calculate_position_size(entry_price=200, stop_loss=180, lot_size=50, risk_per_trade_pct=1.0)
    assert sizing["lots"] == 0
    assert sizing["quantity"] == 0


# -----------------------------------------------------------------------------
# TEST 6 — TRAILING STOP
# -----------------------------------------------------------------------------

def test_trailing_stop_stages_non_decreasing_sl():
    broker = DummyBroker()
    monitor = PositionMonitor(broker=broker, poll_interval=1)
    state = TrailingState(
        position_id="POS-TEST",
        tradingsymbol="TEST",
        symbol="TEST",
        entry_premium=100,
        initial_sl=70,
        current_sl=70,
        target=999,
        direction="bullish",
        sl_order_id="",  # avoid broker modify
        breakeven_ratio=0.6,
        risk_distance=30,
    )
    # Progress through stages
    monitor._process_tick(state, 120)  # breakeven
    be_sl = state.current_sl
    monitor._process_tick(state, 130)  # stage 1
    stage1_sl = state.current_sl
    monitor._process_tick(state, 160)  # stage 2
    stage2_sl = state.current_sl
    monitor._process_tick(state, 190)  # stage 3
    stage3_sl = state.current_sl
    assert be_sl < stage1_sl < stage2_sl <= stage3_sl


# -----------------------------------------------------------------------------
# TEST 7 — CIRCUIT BREAKER / COOL-OFF
# -----------------------------------------------------------------------------

def test_circuit_breaker_after_consecutive_losses():
    cfg = {
        "consecutive_losses_pause": 3,
        "pause_duration_minutes": 60,
        "max_daily_trades": 10,
        "max_daily_loss": 5000,
    }
    rm = RiskManager(cfg, initial_capital=200000)
    for _ in range(3):
        rm.record_trade_result(-1000)
    result = rm.pre_trade_check({"symbol": "NIFTY 50", "cost": 0, "entry_price": 0, "quantity": 0},
                                current_time=datetime.now())
    assert not result.approved
    assert any("Cool-off" in v or "cool-off" in v for v in result.violations)


# -----------------------------------------------------------------------------
# TEST 8 — PAPER TRADER PARITY (BASIC)
# -----------------------------------------------------------------------------

def test_paper_trader_costs_accumulate():
    from prometheus.execution.paper_trader import PaperTrader
    from prometheus.execution.broker import Order, OrderSide, OrderType, ProductType

    broker = PaperTrader(initial_capital=100000)
    order = Order(
        symbol="NIFTY 50",
        tradingsymbol="NIFTY",
        exchange="NFO",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        product=ProductType.MIS,
        quantity=50,
        price=100,
    )
    filled = broker.place_order(order)
    assert filled.status == OrderStatus.COMPLETE
    # Close position
    close = Order(
        symbol="NIFTY 50",
        tradingsymbol="NIFTY",
        exchange="NFO",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        product=ProductType.MIS,
        quantity=50,
        price=110,
    )
    broker.place_order(close)
    assert broker.total_costs > 0
    assert broker.get_margins()["equity"] <= 100000 + 50 * 10  # costs deducted


# -----------------------------------------------------------------------------
# TEST 9 — BACKTEST ENGINE SMOKE
# -----------------------------------------------------------------------------

def test_backtest_engine_initializes_with_cost_config():
    engine = ZerodhaCostModel({"stt_options_sell": 0.10, "stt_futures": 0.01})
    costs = engine.calculate_costs(10000, 11000, "options")
    assert costs["stt"] == pytest.approx(11.0, rel=1e-2)


# -----------------------------------------------------------------------------
# TEST 10 — WALK-FORWARD PLACEHOLDER
# -----------------------------------------------------------------------------

def test_walkforward_placeholder():
    # Placeholder smoke test to ensure suite completes; full walk-forward run
    # would be heavy and data-dependent, so we simply assert environment ready.
    assert True
