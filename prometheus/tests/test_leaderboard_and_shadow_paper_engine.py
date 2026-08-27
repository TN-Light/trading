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

def test_telegram_alert_new_signal_formatting():
    from prometheus.interface.telegram_bot import TelegramBot
    
    messages = []
    tb = TelegramBot(bot_token='', chat_id='')
    tb.send_message = lambda text: messages.append(text)
    
    # 1. Option Buying
    sig_buy = {
        'action': 'BUY_PE',
        'symbol': 'NIFTY MIDCAP SELECT',
        'instrument': 'MIDCPNIFTY29SEP2614925PE',
        'tradingsymbol': 'MIDCPNIFTY29SEP2614925PE',
        'entry_price': 181.9,
        'stop_loss': 154.6,
        'target': 221.9,
        'strike': 14925,
        'option_type': 'PE',
        'leaderboard_rank': 1,
        'strategy_type': 'option_buying',
    }
    tb.alert_new_signal(sig_buy)
    assert len(messages) == 1
    assert 'RANK #1 SIGNAL' in messages[0]
    assert 'MIDCPNIFTY SEP 14925 PE' in messages[0]
    assert '<b>Live Entry LTP (Angel One):</b> Rs 181.9' in messages[0]
    
    # 2. Credit Spread
    sig_spread = {
        'action': 'BEAR_CALL_SPREAD',
        'symbol': 'NIFTY 50',
        'net_credit': 69.45,
        'target_decay_price': 20.8,
        'hard_sl_price': 104.1,
        'leaderboard_rank': 2,
        'strategy_type': 'credit_spread',
        'legs': [
            {'action': 'BUY', 'tradingsymbol': 'NIFTY2690124500CE', 'premium': 15.0, 'strike': 24500, 'option_type': 'CE', 'is_hedge': True},
            {'action': 'SELL', 'tradingsymbol': 'NIFTY2690124350CE', 'premium': 84.45, 'strike': 24350, 'option_type': 'CE', 'is_hedge': False},
        ]
    }
    tb.alert_new_signal(sig_spread)
    assert len(messages) == 2
    assert 'RANK #2 SIGNAL' in messages[1]
    assert 'NIFTY 1 SEP 24500 CE' in messages[1]
    assert '<b>Net Live Credit (Angel One):</b> Rs 69.5/share' in messages[1]
