import pytest
import pandas as pd
from datetime import date, datetime
from prometheus.utils.indian_market import get_expiry_date, _resolve_weekly_expiry_day_name
from prometheus.strategies.credit_spread import CreditSpreadStrategy

def test_sensex_weekly_expiry_is_friday():
    today = date(2026, 8, 26)
    day_name = _resolve_weekly_expiry_day_name('SENSEX', today)
    assert day_name == 'Friday'
    
    exp_date = get_expiry_date('SENSEX', today)
    assert exp_date == date(2026, 8, 28)

def test_credit_spread_uses_live_option_chain_pricing():
    strategy = CreditSpreadStrategy()
    
    # Mock candle data (sideways regime)
    rows = []
    base_time = datetime(2026, 8, 26, 10, 15)
    for i in range(30):
        rows.append({
            'timestamp': base_time,
            'open': 24350.0,
            'high': 24360.0,
            'low': 24340.0,
            'close': 24350.0,
            'volume': 1000,
        })
    df = pd.DataFrame(rows)
    
    # Mock AngelOne option chain
    class MockAngelOneOptions:
        def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
            if strike == 24400:
                return {'ltp': 67.5, 'bid': 67.0, 'ask': 68.0, 'tradingsymbol': 'NIFTY26AUG24400CE'}
            elif strike == 24550:
                return {'ltp': 138.0, 'bid': 137.5, 'ask': 138.5, 'tradingsymbol': 'NIFTY26AUG24550CE'}
            return None
            
    mock_chain = MockAngelOneOptions()
    sig = strategy.evaluate_spread(df, symbol='NIFTY 50', capital=15000, option_chain=mock_chain)
    
    assert sig is not None
    # Verify live prices are used
    assert sig['legs'][0]['premium'] == 138.0
    assert sig['legs'][1]['premium'] == 67.5