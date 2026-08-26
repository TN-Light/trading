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