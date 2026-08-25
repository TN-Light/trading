import pytest
import pandas as pd
from datetime import datetime, timedelta, time as dtime
from unittest.mock import MagicMock

from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.signals.price_action_momentum import PriceActionMomentumScanner

def test_low_vix_trailing_ladder():
    monitor = PositionMonitor(broker=MagicMock(), poll_interval=1.0)
    
    state = TrailingState(
        position_id='test_low_vix_1',
        symbol='NIFTY 50',
        tradingsymbol='NIFTY25AUG2624150PE',
        entry_premium=100.0,
        initial_sl=85.0,
        current_sl=85.0,
        target=122.0,
        direction='bearish',
        trade_mode='intraday',
        low_vix_mode=True,
    )
    
    # 1. At 105 (only +5%), no change
    monitor._process_tick(state, current_price=105.0)
    assert state.breakeven_set is False
    assert state.current_sl == 85.0
    
    # 2. At 108 (+8% gain), triggers Low-VIX Breakeven
    monitor._process_tick(state, current_price=108.5)
    assert state.breakeven_set is True
    assert state.current_sl == 101.5
    
    # 3. At 114.5 (+14.5% gain), triggers Low-VIX Stage 1 (Lock +8%)
    monitor._process_tick(state, current_price=114.5)
    assert state.trailing_activated is True
    assert state.current_sl == 108.0
    
    # 4. At 118.5 (+18.5% gain), triggers Low-VIX Stage 2 (Lock +14%)
    monitor._process_tick(state, current_price=118.5)
    assert state.trailing_stage2 is True
    assert state.current_sl == 114.0

def test_expiry_fast_trigger_bullish_reclaim():
    scanner = PriceActionMomentumScanner()
    
    base_time = datetime(2026, 8, 25, 11, 0)
    rows = []
    # 15 bars: earlier bars below VWAP (~24130), last 2 bars surge to 24180
    for i in range(15):
        p = 24130.0 if i < 13 else (24150.0 if i == 13 else 24185.0)
        rows.append({
            'timestamp': base_time + timedelta(minutes=5 * i),
            'open': p - 2,
            'high': p + 5,
            'low': p - 5,
            'close': p,
            'volume': 1000 if i < 14 else 3000,
        })
    df_5m = pd.DataFrame(rows)
    
    sig = scanner.evaluate_5m_expiry_surge(df_5m, symbol='NIFTY 50')
    assert sig is not None
    assert sig['action'] == 'BUY_CE'
    assert sig['fast_expiry_surge'] is True
    assert sig['timeframe'] == '5minute'
    assert '5M_Expiry_Surge' in sig['reasons']

def test_expiry_fast_trigger_bearish_breakdown():
    scanner = PriceActionMomentumScanner()
    
    base_time = datetime(2026, 8, 25, 10, 0)
    rows = []
    # 15 bars: earlier bars around 24180, last 2 bars plunge below VWAP to 24120
    for i in range(15):
        p = 24180.0 if i < 13 else (24160.0 if i == 13 else 24125.0)
        rows.append({
            'timestamp': base_time + timedelta(minutes=5 * i),
            'open': p + 2,
            'high': p + 5,
            'low': p - 5,
            'close': p,
            'volume': 1000 if i < 14 else 3000,
        })
    df_5m = pd.DataFrame(rows)
    
    sig = scanner.evaluate_5m_expiry_surge(df_5m, symbol='NIFTY 50')
    assert sig is not None
    assert sig['action'] == 'BUY_PE'
    assert sig['fast_expiry_surge'] is True
    assert sig['timeframe'] == '5minute'