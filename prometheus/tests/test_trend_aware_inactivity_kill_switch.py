"""Unit tests for Trend-Aware Inactivity Kill Switch.

Verifies:
1. PositionTracker spares stagnant option positions during trend consolidation flags (bars 3, 4, 5)
   when underlying trend remains intact (EMA9/EMA21 and SuperTrend aligned + favorable spot drift)
   for BOTH bullish (calls) and bearish (puts) positions.
2. PositionTracker liquidates positions at bar 6 (90 min) max extension to eliminate theta decay.
3. PositionTracker liquidates positions at bar 3 if trend is broken or spot drifted adversely.
4. PositionMonitor trend-aware consolidation flag deferral and 90m cap.
"""

from datetime import datetime
from unittest.mock import MagicMock
import pandas as pd
import pytest

from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.papertrade.types import Position, Direction, ExitReason, TradeSnapshot
from prometheus.execution.position_monitor import PositionMonitor, TrailingState


class MockFeed:
    def __init__(self, ltps=None):
        self.ltps = ltps or {}

    def get_ltp(self, instrument: str) -> float:
        return float(self.ltps.get(instrument, 0.0))

    def get_quote(self, instrument: str):
        price = self.get_ltp(instrument)
        if price > 0:
            return {"bid": price - 0.5, "ask": price + 0.5, "ltp": price}
        return None


def _make_bullish_consolidation_df():
    """Generates OHLC data with a bullish trend (EMA9 > EMA21, SuperTrend == 1)
    ending in a 5-bar consolidation at 24015.0 with ATR ~31.4 (min_disp ~15.7)."""
    closes = [23800.0] * 15
    for i in range(1, 11):
        closes.append(23800.0 + i * 25.0)
    closes.extend([24015.0] * 5)
    highs = [c + 10.0 for c in closes]
    lows = [c - 10.0 for c in closes]
    for i in range(15):
        highs[i] = closes[i] + 2.0
        lows[i] = closes[i] - 2.0
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-09-15 09:15", periods=len(closes), freq="15min"),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [5000] * len(closes),
    })


def _make_bearish_consolidation_df():
    """Generates OHLC data with a bearish trend (EMA9 < EMA21, SuperTrend == -1)
    ending in a 5-bar consolidation at 23985.0 with ATR ~31.4 (min_disp ~15.7)."""
    closes = [24200.0] * 15
    for i in range(1, 11):
        closes.append(24200.0 - i * 25.0)
    closes.extend([23985.0] * 5)
    highs = [c + 10.0 for c in closes]
    lows = [c - 10.0 for c in closes]
    for i in range(15):
        highs[i] = closes[i] + 2.0
        lows[i] = closes[i] - 2.0
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-09-15 09:15", periods=len(closes), freq="15min"),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [5000] * len(closes),
    })


def test_tracker_trend_aware_spares_intact_consolidation_flag():
    """PositionTracker keeps bullish position open at bars 3, 4, 5 if trend is intact,
    then closes at bar 6 (90 min) max extension."""
    feed = MockFeed({"NIFTY26SEP24000CE": 101.0, "NIFTY 50": 24015.0})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    mock_data_engine = MagicMock()
    mock_data_engine.fetch_historical.return_value = _make_bullish_consolidation_df()

    tracker = PositionTracker(
        fill_sim=sim,
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=False,
        data_engine=mock_data_engine,
        trend_aware=True,
        max_stagnation_bars=6,
    )

    pos = Position(
        trade_id="PAPER-TREND-AWARE-01",
        symbol="NIFTY 50",
        instrument="NIFTY26SEP24000CE",
        underlying="NIFTY 50",
        direction=Direction.LONG,
        quantity=50,
        entry_price=100.0,
        entry_time=datetime(2026, 9, 15, 9, 30),
        stop_loss=70.0,
        target=160.0,
        max_bars=16,
        strategy="PriceAction_Momentum",
        trade_mode="intraday",
        bars_held=2,  # next bar makes it 3
        entry_spot=24005.0,  # spot is 24015 -> displacement +10 < min_disp ~15.7
        atr=31.4,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Bar 3 (45 min): spot 24015 (displacement +10 < 0.5*ATR=15.7), premium 101.0 (stagnant)
    snap3 = TradeSnapshot(
        timestamp=datetime(2026, 9, 15, 10, 15),
        symbol="NIFTY 50",
        instrument="",
        open=24010.0, high=24025.0, low=24005.0, close=24015.0,
        volume=10000, bar_interval="15minute",
    )
    closed3 = tracker.on_bar(snap3, is_session_end=False, is_square_off=False)
    assert len(closed3) == 0, "Position must NOT be closed on bar 3 when macro trend is intact"
    assert pos.bars_held == 3

    # Bar 4 (60 min):
    snap4 = TradeSnapshot(
        timestamp=datetime(2026, 9, 15, 10, 30),
        symbol="NIFTY 50",
        instrument="",
        open=24015.0, high=24025.0, low=24010.0, close=24015.0,
        volume=10000, bar_interval="15minute",
    )
    closed4 = tracker.on_bar(snap4, is_session_end=False, is_square_off=False)
    assert len(closed4) == 0, "Position must NOT be closed on bar 4 when macro trend is intact"
    assert pos.bars_held == 4

    # Bar 5 (75 min):
    snap5 = TradeSnapshot(
        timestamp=datetime(2026, 9, 15, 10, 45),
        symbol="NIFTY 50",
        instrument="",
        open=24015.0, high=24025.0, low=24010.0, close=24015.0,
        volume=10000, bar_interval="15minute",
    )
    closed5 = tracker.on_bar(snap5, is_session_end=False, is_square_off=False)
    assert len(closed5) == 0, "Position must NOT be closed on bar 5 when macro trend is intact"
    assert pos.bars_held == 5

    # Bar 6 (90 min): Max stagnation bars reached! Must exit to stop theta decay.
    snap6 = TradeSnapshot(
        timestamp=datetime(2026, 9, 15, 11, 0),
        symbol="NIFTY 50",
        instrument="",
        open=24015.0, high=24025.0, low=24010.0, close=24015.0,
        volume=10000, bar_interval="15minute",
    )
    closed6 = tracker.on_bar(snap6, is_session_end=False, is_square_off=False)
    assert len(closed6) == 1, "Position MUST exit on bar 6 (max 90m extension reached)"
    assert closed6[0].exit_reason == ExitReason.INACTIVITY_KILL_SWITCH
    assert closed6[0].exit_price == 101.0


def test_tracker_trend_aware_spares_bearish_consolidation_flag():
    """PositionTracker keeps bearish position (put) open at bars 3, 4, 5 if downtrend is intact,
    then closes at bar 6 (90 min) max extension."""
    feed = MockFeed({"NIFTY26SEP24000PE": 102.0, "NIFTY 50": 23985.0})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    mock_data_engine = MagicMock()
    mock_data_engine.fetch_historical.return_value = _make_bearish_consolidation_df()

    tracker = PositionTracker(
        fill_sim=sim,
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=False,
        data_engine=mock_data_engine,
        trend_aware=True,
        max_stagnation_bars=6,
    )

    pos = Position(
        trade_id="PAPER-BEAR-TREND-01",
        symbol="NIFTY 50",
        instrument="NIFTY26SEP24000PE",
        underlying="NIFTY 50",
        direction=Direction.SHORT,
        quantity=50,
        entry_price=100.0,
        entry_time=datetime(2026, 9, 15, 9, 30),
        stop_loss=70.0,
        target=160.0,
        max_bars=16,
        strategy="PriceAction_Momentum",
        trade_mode="intraday",
        bars_held=2,
        entry_spot=23995.0,  # spot is 23985 -> displacement +10 < min_disp ~15.7
        atr=31.4,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Bar 3: spot 23985 (favorable drift + intact downtrend)
    snap3 = TradeSnapshot(
        timestamp=datetime(2026, 9, 15, 10, 15),
        symbol="NIFTY 50",
        instrument="",
        open=23990.0, high=23995.0, low=23980.0, close=23985.0,
        volume=10000, bar_interval="15minute",
    )
    closed3 = tracker.on_bar(snap3, is_session_end=False, is_square_off=False)
    assert len(closed3) == 0, "Bearish position must NOT be closed on bar 3 when macro downtrend is intact"
    assert pos.bars_held == 3

    # Bar 6: Max extension reached -> exits
    pos.bars_held = 5
    snap6 = TradeSnapshot(
        timestamp=datetime(2026, 9, 15, 11, 0),
        symbol="NIFTY 50",
        instrument="",
        open=23990.0, high=23995.0, low=23980.0, close=23985.0,
        volume=10000, bar_interval="15minute",
    )
    closed6 = tracker.on_bar(snap6, is_session_end=False, is_square_off=False)
    assert len(closed6) == 1, "Bearish position MUST exit on bar 6 (max 90m extension reached)"
    assert closed6[0].exit_reason == ExitReason.INACTIVITY_KILL_SWITCH


def test_tracker_trend_aware_exits_on_adverse_drift_at_bar_3():
    """PositionTracker immediately exits on bar 3 if spot has drifted adversely against trade."""
    feed = MockFeed({"NIFTY26SEP24000CE": 98.0, "NIFTY 50": 23980.0})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    mock_data_engine = MagicMock()
    mock_data_engine.fetch_historical.return_value = _make_bullish_consolidation_df()

    tracker = PositionTracker(
        fill_sim=sim,
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=False,
        data_engine=mock_data_engine,
        trend_aware=True,
        max_stagnation_bars=6,
    )

    pos = Position(
        trade_id="PAPER-ADVERSE-01",
        symbol="NIFTY 50",
        instrument="NIFTY26SEP24000CE",
        underlying="NIFTY 50",
        direction=Direction.LONG,
        quantity=50,
        entry_price=100.0,
        entry_time=datetime(2026, 9, 15, 9, 30),
        stop_loss=70.0,
        target=160.0,
        max_bars=16,
        strategy="PriceAction_Momentum",
        trade_mode="intraday",
        bars_held=2,  # next bar makes it 3
        entry_spot=24000.0,
        atr=31.4,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Bar 3: spot drifted adversely to 23980 (< entry_spot 24000)
    snap3 = TradeSnapshot(
        timestamp=datetime(2026, 9, 15, 10, 15),
        symbol="NIFTY 50",
        instrument="",
        open=23990.0, high=23995.0, low=23970.0, close=23980.0,
        volume=10000, bar_interval="15minute",
    )
    closed = tracker.on_bar(snap3, is_session_end=False, is_square_off=False)
    assert len(closed) == 1, "Position must exit on bar 3 when spot drifts adversely"
    assert closed[0].exit_reason == ExitReason.INACTIVITY_KILL_SWITCH


def test_monitor_trend_aware_spares_intact_and_exits_at_max_bars():
    """PositionMonitor spares position on bars 3, 4, 5 if trend is intact, then exits at bar 6."""
    exits = []
    def _on_exit(pos_id, price, reason):
        exits.append((pos_id, price, reason))

    mock_data_engine = MagicMock()
    mock_data_engine.fetch_historical.return_value = _make_bullish_consolidation_df()

    monitor = PositionMonitor(broker=MagicMock(), on_exit=_on_exit, data_engine=mock_data_engine)

    state = TrailingState(
        position_id="POS-MON-TA-01",
        symbol="NIFTY 50",
        tradingsymbol="NIFTY26SEP24000CE",
        entry_premium=100.0,
        initial_sl=70.0,
        current_sl=70.0,
        target=160.0,
        direction="bullish",
        trade_mode="intraday",
        adverse_exit_enabled=False,
        entry_bar_count=3,  # 3 bars
        entry_spot=24005.0,  # spot is 24015.0 -> displacement +10 < min_disp ~15.7
        atr=31.4,
    )

    # Tick at bar 3: trend is intact -> spared
    monitor._process_tick(state, 101.0)
    assert len(exits) == 0, "PositionMonitor must spare intact trend at bar 3"

    # Advance to bar 4 & 5
    state.entry_bar_count = 5
    monitor._process_tick(state, 101.0)
    assert len(exits) == 0, "PositionMonitor must spare intact trend at bar 5"

    # Advance to bar 6 (max 90m extension reached)
    state.entry_bar_count = 6
    monitor._process_tick(state, 101.0)
    assert len(exits) == 1, "PositionMonitor must exit at bar 6 max extension"
    assert exits[0][2] == "inactivity_kill_switch"
