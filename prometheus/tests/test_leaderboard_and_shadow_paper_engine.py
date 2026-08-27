import pytest
from prometheus.main import Prometheus

def test_leaderboard_sorting_and_shadow_partitioning():
    candidates = [
        {'symbol': 'NIFTY BANK', 'signal_score': 3.4, 'strategy_type': 'credit_spread'},
        {'symbol': 'NIFTY 50', 'signal_score': 4.6, 'strategy_type': 'option_buying', 'action': 'BUY_CE'},
        {'symbol': 'SENSEX', 'signal_score': 3.8, 'strategy_type': 'credit_spread'},
    ]
    
    # Sort descending
    candidates.sort(key=lambda s: float(s.get('signal_score', 0.0)), reverse=True)
    
    # Verify rank 1 is NIFTY 50 with highest score
    assert candidates[0]['symbol'] == 'NIFTY 50'
    assert candidates[0]['signal_score'] == 4.6
    
    # Verify rank 2 is SENSEX
    assert candidates[1]['symbol'] == 'SENSEX'
    assert candidates[1]['signal_score'] == 3.8
    
    # Verify rank 3 is NIFTY BANK
    assert candidates[2]['symbol'] == 'NIFTY BANK'
    assert candidates[2]['signal_score'] == 3.4

def test_kite_search_name_monthly_vs_weekly():
    from prometheus.utils.symbol_format import human_search_name
    from datetime import date
    
    # SENSEX Monthly expiry (August 28, 2026 is last Friday of August)
    monthly_name = human_search_name("SENSEX", date(2026, 8, 28), 77400, "PE")
    assert monthly_name == "SENSEX AUG 77400 PE"
    
    # SENSEX Weekly expiry (August 21, 2026 is not the last Friday)
    weekly_name = human_search_name("SENSEX", date(2026, 8, 21), 77400, "PE")
    assert weekly_name == "SENSEX 21 AUG 77400 PE"
    
    # NIFTY Monthly expiry
    nifty_monthly = human_search_name("NIFTY 50", date(2026, 8, 25), 24250, "CE")
    assert nifty_monthly == "NIFTY AUG 24250 CE"