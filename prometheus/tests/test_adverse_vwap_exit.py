import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from prometheus.execution.position_monitor import PositionMonitor, TrailingState

def create_sample_df(spot_prices):
    base_time = datetime(2026, 8, 25, 9, 15)
    rows = []
    for i, p in enumerate(spot_prices):
        rows.append({
            'timestamp': base_time + timedelta(minutes=15 * i),
            'open': p,
            'high': p + 5,
            'low': p - 5,
            'close': p,
            'volume': 1000,
        })
    return pd.DataFrame(rows)

def test_adverse_vwap_exit_pe():
    monitor = PositionMonitor(broker=MagicMock(), poll_interval=1.0)
    mock_data = MagicMock()
    prices = [24100.0] * 18 + [24180.0, 24200.0]
    mock_data.fetch_historical.return_value = create_sample_df(prices)
    monitor._data_engine = mock_data

    state = TrailingState(
        position_id='test_pe_1',
        symbol='NIFTY 50',
        tradingsymbol='NIFTY25AUG2624150PE',
        entry_premium=35.0,
        initial_sl=28.0,
        current_sl=28.0,
        target=45.0,
        direction='bearish',
        trade_mode='intraday',
        entry_bar_count=2,
    )

    should_exit = monitor._check_adverse_indicator(state, current_price=30.0)
    assert should_exit is True
    assert 'VWAP Bullish Invalidation' in state._adverse_reason

def test_adverse_vwap_exit_ce():
    monitor = PositionMonitor(broker=MagicMock(), poll_interval=1.0)
    mock_data = MagicMock()
    prices = [24200.0] * 18 + [24150.0, 24100.0]
    mock_data.fetch_historical.return_value = create_sample_df(prices)
    monitor._data_engine = mock_data

    state = TrailingState(
        position_id='test_ce_1',
        symbol='NIFTY 50',
        tradingsymbol='NIFTY25AUG2624200CE',
        entry_premium=35.0,
        initial_sl=28.0,
        current_sl=28.0,
        target=45.0,
        direction='bullish',
        trade_mode='intraday',
        entry_bar_count=2,
    )

    should_exit = monitor._check_adverse_indicator(state, current_price=30.0)
    assert should_exit is True
    assert 'VWAP Bearish Invalidation' in state._adverse_reason

def test_adverse_vwap_no_exit_when_aligned():
    monitor = PositionMonitor(broker=MagicMock(), poll_interval=1.0)
    mock_data = MagicMock()
    prices = [24200.0] * 15 + [24150.0] * 5
    mock_data.fetch_historical.return_value = create_sample_df(prices)
    monitor._data_engine = mock_data

    state = TrailingState(
        position_id='test_pe_aligned',
        symbol='NIFTY 50',
        tradingsymbol='NIFTY25AUG2624150PE',
        entry_premium=35.0,
        initial_sl=28.0,
        current_sl=28.0,
        target=45.0,
        direction='bearish',
        trade_mode='intraday',
        entry_bar_count=2,
    )

    should_exit = monitor._check_adverse_indicator(state, current_price=38.0)
    assert should_exit is False