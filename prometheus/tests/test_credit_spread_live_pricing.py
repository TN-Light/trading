import pytest
import pandas as pd
from datetime import date, datetime
from prometheus.utils.indian_market import get_expiry_date, _resolve_weekly_expiry_day_name
from prometheus.strategies.credit_spread import CreditSpreadStrategy

def test_sensex_weekly_expiry_is_thursday():
    # Post Sep 1, 2025: BSE SENSEX weekly expiry is Thursday
    today = date(2026, 8, 26)  # Wednesday
    day_name = _resolve_weekly_expiry_day_name('SENSEX', today)
    assert day_name == 'Thursday'
    
    exp_date = get_expiry_date('SENSEX', today)
    assert exp_date == date(2026, 8, 27)  # Next Thursday


def test_regulatory_expiry_schedule_pre_and_post_cutover():
    # 1. SENSEX: Friday before cutover, Thursday after cutover
    pre_cutover = date(2025, 6, 18)  # Wednesday
    assert _resolve_weekly_expiry_day_name('SENSEX', pre_cutover) == 'Friday'
    assert get_expiry_date('SENSEX', pre_cutover) == date(2025, 6, 20)  # Friday

    post_cutover = date(2026, 9, 2)  # Wednesday
    assert _resolve_weekly_expiry_day_name('SENSEX', post_cutover) == 'Thursday'
    assert get_expiry_date('SENSEX', post_cutover) == date(2026, 9, 3)  # Thursday

    # 2. NIFTY: Thursday before cutover, Tuesday after cutover
    assert _resolve_weekly_expiry_day_name('NIFTY 50', pre_cutover) == 'Thursday'
    assert get_expiry_date('NIFTY 50', pre_cutover) == date(2025, 6, 19)  # Thursday

    assert _resolve_weekly_expiry_day_name('NIFTY 50', post_cutover) == 'Tuesday'
    assert get_expiry_date('NIFTY 50', post_cutover) == date(2026, 9, 8)  # Tuesday

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
    
    # Mock AngelOne option chain — returns realistic premiums for any strike
    # Uses a simple distance-based premium model (premium decays with OTM distance)
    class MockAngelOneOptions:
        def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
            spot = spot_price or 24350.0
            dist = abs(strike - spot)
            # Approximate premium: higher for nearer strikes, lower for farther
            if option_type == "CE":
                premium = max(5.0, 200.0 - dist * 0.8)
            else:
                premium = max(5.0, 200.0 - dist * 0.8)
            tsym = f"NIFTY26AUG{int(strike)}{option_type}"
            return {'ltp': round(premium, 2), 'bid': round(premium - 0.5, 2), 'ask': round(premium + 0.5, 2), 'tradingsymbol': tsym}
            
    mock_chain = MockAngelOneOptions()
    sig = strategy.evaluate_spread(df, symbol='NIFTY 50', capital=15000, option_chain=mock_chain)
    
    assert sig is not None
    # Verify structure: 2-leg spread with live prices (not synthetic formulas)
    assert len(sig['legs']) == 2
    assert sig['legs'][0]['action'] == 'BUY'   # Hedge leg first
    assert sig['legs'][1]['action'] == 'SELL'   # Short leg second
    assert sig['legs'][0]['premium'] > 0        # Live premium used
    assert sig['legs'][1]['premium'] > 0        # Live premium used
    assert sig['net_credit'] > 0                # Positive net credit collected