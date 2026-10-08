"""
Tests for Health-Gated Inactivity Kill Switch and Microstructure Noise Preservation.

Validates:
1. Neutral or Healthy positions (Score > -15.0, no real_threat) have the 45-min
   inactivity kill switch DEFERRED, preserving the consolidation flag up to 6 bars (90 min).
2. Degraded or Structurally Broken positions (Score <= -15.0 or real_threat)
   have the inactivity kill switch TRIGGERED immediately on bar 3 to protect capital.
3. Microstructure spot drift with noise buffer (0.35 * ATR or 35/18/8 pts) preserves consolidation
   flags without false-failing on 1-minute candle noise.
4. Live PositionMonitor synchronizes with HealthEngine to defer/trigger kill switch.
"""

from datetime import datetime
from unittest.mock import MagicMock
import pytest
import pandas as pd

from prometheus.utils.indian_market import IST
from prometheus.papertrade.types import Position, Direction, ExitReason, TradeSnapshot
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.position_health import PositionHealthReport


class MockPriceFeed:
    def __init__(self, ltp_dict=None):
        self.ltp_dict = ltp_dict or {}

    def get_ltp(self, symbol: str, instrument: str = None) -> float:
        key = instrument or symbol
        return float(self.ltp_dict.get(key, 150.0))


def _make_bearish_df_sensex():
    """Declining OHLC data where EMA9 < EMA21 and SuperTrend == -1."""
    closes = [72400.0] * 15
    for i in range(1, 11):
        closes.append(72400.0 - i * 20.0)
    closes.extend([72200.0] * 5)
    highs = [c + 15.0 for c in closes]
    lows = [c - 15.0 for c in closes]
    for i in range(15):
        highs[i] = closes[i] + 5.0
        lows[i] = closes[i] - 5.0
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-08 09:15", periods=len(closes), freq="15min"),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [50000] * len(closes),
    })


def test_inactivity_kill_switch_deferred_for_neutral_health():
    """Verify that a stagnant position with intact technical trend and NEUTRAL health (+8) is deferred."""
    feed = MockPriceFeed({"SENSEX26O0872200PE": 148.90})
    fill_sim = FillSimulator(feed=feed)

    mock_data = MagicMock()
    mock_data.fetch_historical.return_value = _make_bearish_df_sensex()

    tracker = PositionTracker(
        fill_sim=fill_sim,
        cost_model=CostModel(),
        data_engine=mock_data,
        trend_aware=True,
        max_stagnation_bars=6,
    )

    # Attach mock health engine returning NEUTRAL score (+8.0)
    mock_health_engine = MagicMock()
    mock_report = PositionHealthReport(
        position_id="PAPER-SENSEX-01",
        symbol="SENSEX",
        tradingsymbol="SENSEX26O0872200PE",
        direction="bearish",
        current_price=148.90,
        entry_price=185.00,
        health_score=8.0,
        real_threat=False,
    )
    mock_health_engine.evaluate_position_health.return_value = mock_report
    tracker.health_engine = mock_health_engine

    now = datetime.now(IST)
    pos = Position(
        trade_id="PAPER-SENSEX-01",
        symbol="SENSEX",
        instrument="SENSEX26O0872200PE",
        underlying="SENSEX",
        direction=Direction.SHORT,
        quantity=20,
        entry_price=185.00,
        stop_loss=139.80,
        target=263.30,
        entry_time=now,
        entry_spot=72186.72,
        atr=81.7,
        max_bars=10,
        bars_held=2,  # next bar makes it 3
    )
    tracker.open_position(pos)

    # Process bar 3 with spot at 72210.0 (+23.28 pts against short, within noise buffer)
    snap = TradeSnapshot(
        timestamp=now,
        symbol="SENSEX",
        instrument="",
        open=72200.0,
        high=72225.0,
        low=72180.0,
        close=72210.0,
        volume=50000.0,
    )

    closed = tracker.on_bar(snap)

    # Position must NOT be killed on bar 3 because trend is intact and health is NEUTRAL (+8.0)
    assert len(closed) == 0, "Kill switch should be deferred for structurally neutral/healthy position"
    assert "PAPER-SENSEX-01" in tracker.open_positions
    assert tracker.open_positions["PAPER-SENSEX-01"].bars_held == 3


def test_inactivity_kill_switch_triggers_for_critical_threat():
    """Verify that a stagnant position with CRITICAL threat (Score -40.0) IS killed on bar 3."""
    feed = MockPriceFeed({"SENSEX26O0872200PE": 140.00})
    fill_sim = FillSimulator(feed=feed)

    mock_data = MagicMock()
    mock_data.fetch_historical.return_value = _make_bearish_df_sensex()

    tracker = PositionTracker(
        fill_sim=fill_sim,
        cost_model=CostModel(),
        data_engine=mock_data,
        trend_aware=True,
        max_stagnation_bars=6,
    )

    # Attach mock health engine returning CRITICAL threat (-40.0)
    mock_health_engine = MagicMock()
    mock_report = PositionHealthReport(
        position_id="PAPER-THREAT-01",
        symbol="SENSEX",
        tradingsymbol="SENSEX26O0872200PE",
        direction="bearish",
        current_price=140.00,
        entry_price=185.00,
        health_score=-40.0,
        real_threat=True,
        threat_reasons=["Spot rallied above 15M VWAP", "Severe Theta Burn"],
    )
    mock_health_engine.evaluate_position_health.return_value = mock_report
    tracker.health_engine = mock_health_engine

    now = datetime.now(IST)
    pos = Position(
        trade_id="PAPER-THREAT-01",
        symbol="SENSEX",
        instrument="SENSEX26O0872200PE",
        underlying="SENSEX",
        direction=Direction.SHORT,
        quantity=20,
        entry_price=185.00,
        stop_loss=139.80,
        target=263.30,
        entry_time=now,
        entry_spot=72186.72,
        atr=81.7,
        max_bars=10,
        bars_held=2,  # next bar makes it 3
    )
    tracker.open_position(pos)

    snap = TradeSnapshot(
        timestamp=now,
        symbol="SENSEX",
        instrument="",
        open=72200.0,
        high=72225.0,
        low=72180.0,
        close=72210.0,
        volume=50000.0,
    )

    closed = tracker.on_bar(snap)

    # Position MUST be killed immediately on bar 3 due to health veto
    assert len(closed) == 1
    assert closed[0].exit_reason == ExitReason.INACTIVITY_KILL_SWITCH
    assert closed[0].trade_id == "PAPER-THREAT-01"


def test_spot_drift_with_noise_buffer_preserves_consolidation():
    """Verify _is_underlying_trend_intact allows drift within noise buffer without failing."""
    mock_data = MagicMock()
    mock_data.fetch_historical.return_value = _make_bearish_df_sensex()
    tracker = PositionTracker(fill_sim=None, data_engine=mock_data)

    # SENSEX Short position: entry 72186.72, current 72210.00 (+23.28 pts against short), ATR = 81.7
    # Noise buffer = max(35.0, 0.35 * 81.7) = 35.0 pts
    # 72210.0 <= 72186.72 + 35.0 (72221.72) -> INTACT!
    intact = tracker._is_underlying_trend_intact(
        symbol="SENSEX",
        direction=Direction.SHORT,
        current_spot=72210.00,
        entry_spot=72186.72,
        atr=81.7,
    )
    assert intact is True

    # If spot drifts beyond noise buffer (+45 pts against short): NOT intact
    intact_out = tracker._is_underlying_trend_intact(
        symbol="SENSEX",
        direction=Direction.SHORT,
        current_spot=72240.00,
        entry_spot=72186.72,
        atr=81.7,
    )
    assert intact_out is False


def test_position_monitor_defers_kill_switch_for_neutral_health():
    """Verify live PositionMonitor defers kill switch when health engine reports neutral score and trend is intact."""
    mock_broker = MagicMock()
    mock_exit_cb = MagicMock()

    mock_data = MagicMock()
    mock_data.fetch_historical.return_value = _make_bearish_df_sensex()

    mock_health = MagicMock()
    mock_report = PositionHealthReport(
        position_id="LIVE-001",
        symbol="SENSEX",
        tradingsymbol="SENSEX26O0872200PE",
        direction="bearish",
        current_price=148.90,
        entry_price=185.00,
        health_score=8.0,
        real_threat=False,
    )
    mock_health.evaluate_position_health.return_value = mock_report

    monitor = PositionMonitor(
        broker=mock_broker,
        on_exit=mock_exit_cb,
        data_engine=mock_data,
    )
    monitor.health_engine = mock_health

    state = TrailingState(
        position_id="LIVE-001",
        tradingsymbol="SENSEX26O0872200PE",
        symbol="SENSEX",
        entry_premium=185.0,
        initial_sl=139.8,
        current_sl=139.8,
        target=263.3,
        direction="bearish",
        entry_spot=72186.72,
        atr=81.7,
        entry_bar_count=3,
        trade_mode="intraday",
    )
    monitor.add_position(state)

    # Process tick at 148.90 (underwater option)
    monitor._process_tick(state, current_price=148.90)

    # Inactivity kill switch should NOT be called because trend is intact and health is NEUTRAL (+8.0)
    mock_exit_cb.assert_not_called()
