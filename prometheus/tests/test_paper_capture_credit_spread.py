import pytest
from prometheus.strategies.credit_spread import CreditSpreadStrategy
from prometheus.papertrade.signal_source import from_signal_dict
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig

def test_paper_capture_credit_spread_parsing():
    strategy = CreditSpreadStrategy()
    
    # Mock credit spread signal from strategy
    signal = {
        'strategy': 'Hedged_Credit_Spread',
        'strategy_type': 'credit_spread',
        'spread_type': 'BEAR_CALL_SPREAD',
        'action': 'BEAR_CALL_SPREAD',
        'direction': 'neutral_range',
        'symbol': 'NIFTY 50',
        'underlying_price': 24350.0,
        'spot_price': 24350.0,
        'entry_price': 25.0,
        'entry_premium': 25.0,
        'strike': 24400.0,
        'option_type': 'CE',
        'stop_loss': 37.5,
        'target': 7.5,
        'net_credit': 25.0,
        'strike_width': 150.0,
        'short_strike': 24400,
        'long_strike': 24550,
        'expiry': '2026-08-27',
        'lot_size': 75,
        'target_decay_price': 7.5,
        'breakeven_decay_price': 12.5,
        'hard_sl_price': 37.5,
        'margin_required': 35000.0,
        'tradingsymbol': 'NIFTY27AUG2624400CE/NIFTY27AUG2624550CE',
        'instrument': 'NIFTY27AUG2624400CE/NIFTY27AUG2624550CE',
    }
    
    notif = from_signal_dict(signal)
    assert notif.entry_price_hint == 25.0
    assert notif.strike == 24400.0
    assert notif.option_type == 'CE'
    assert notif.stop_loss == 37.5
    assert notif.target == 7.5
    assert notif.instrument != ''
    assert notif.symbol == 'NIFTY 50'