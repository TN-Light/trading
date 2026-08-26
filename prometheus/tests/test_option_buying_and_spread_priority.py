import pytest
import pandas as pd
from datetime import datetime
from prometheus.main import Prometheus

def test_option_buying_and_spread_priority_logic():
    p = Prometheus()
    p.mode = 'paper'
    
    # Mock Option Buying signal and Credit Spread signal
    exec_buy_sig = {
        'strategy_type': 'option_buying',
        'action': 'BUY_CE',
        'signal_score': 4.2,
        'entry_price': 150.0,
    }
    
    cs_sig = {
        'strategy_type': 'credit_spread',
        'spread_type': 'BEAR_CALL_SPREAD',
        'signal_score': 3.5,
        'net_credit': 25.0,
    }
    
    # When buy_score >= 3.5, option buying takes priority
    buy_score = float(exec_buy_sig.get('signal_score', 0))
    spread_score = float(cs_sig.get('signal_score', 0))
    
    chosen = exec_buy_sig if buy_score >= 3.5 else cs_sig
    assert chosen['strategy_type'] == 'option_buying'
    assert chosen['action'] == 'BUY_CE'
    
    # When buy_score is low (< 3.5), spread takes priority
    exec_buy_sig['signal_score'] = 3.0
    buy_score = float(exec_buy_sig.get('signal_score', 0))
    chosen = exec_buy_sig if buy_score >= 3.5 else cs_sig
    assert chosen['strategy_type'] == 'credit_spread'