import pytest
import pandas as pd
from datetime import datetime, timedelta, time as dtime

from prometheus.signals.price_action_momentum import PriceActionMomentumScanner

def test_expiry_power_hour_time_cutoff():
    scanner = PriceActionMomentumScanner()
    
    # Create 20 15-min bars where last bar is at 14:45 PM
    base_time = datetime(2026, 8, 25, 9, 15)
    rows = []
    # 23 bars -> 09:15 to 14:45
    for i in range(23):
        p = 24100.0 + (i * 10)
        rows.append({
            'timestamp': base_time + timedelta(minutes=15 * i),
            'open': p - 5,
            'high': p + 10,
            'low': p - 5,
            'close': p,
            'volume': 2000,
        })
    df = pd.DataFrame(rows)
    
    # Non-expiry day at 14:45 -> blocked by 11:45 cutoff
    sig_non_expiry = scanner.evaluate_bar(df, symbol='NIFTY 50', is_expiry_day=False)
    assert sig_non_expiry is None
    
    # Expiry day at 14:45 -> strictly blocked by 11:45 cutoff to prevent afternoon theta traps
    sig_expiry = scanner.evaluate_bar(df, symbol='NIFTY 50', is_expiry_day=True)
    assert sig_expiry is None

def test_5m_expiry_surge_at_1445():
    scanner = PriceActionMomentumScanner()
    
    # Create 5-min bars ending at 14:45 PM
    base_time = datetime(2026, 8, 25, 13, 0)
    rows = []
    # 22 bars of 5min -> 13:00 to 14:45
    for i in range(22):
        p = 24150.0 if i < 18 else (24180.0 + (i - 18) * 15)
        rows.append({
            'timestamp': base_time + timedelta(minutes=5 * i),
            'open': p - 2,
            'high': p + 5,
            'low': p - 5,
            'close': p,
            'volume': 1000 if i < 20 else 4000,
        })
    df_5m = pd.DataFrame(rows)
    
    # 5m surge at 14:45 PM is strictly blocked by 11:45 cutoff
    sig = scanner.evaluate_5m_expiry_surge(df_5m, symbol='NIFTY 50')
    assert sig is None