import pytest
from datetime import datetime
import pytz
import pandas as pd
from prometheus.papertrade.types import Position, Direction, TradeSnapshot, ExitReason
from prometheus.papertrade.position_tracker import PositionTracker
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.strategies.credit_spread import CreditSpreadStrategy

IST = pytz.timezone('Asia/Kolkata')

class DummyFeed:
    def get_ltp(self, s): return 30.0
    def get_quote(self, s): return (datetime.now(IST), 30.0, 29.5, 30.5)

def test_credit_spread_exempt_from_inactivity_kill_switch():
    feed = DummyFeed()
    fill_sim = FillSimulator(feed=feed)
    tracker = PositionTracker(fill_sim=fill_sim)
    
    # Credit spread held for 3 bars (45 min) with zero spot progress
    pos = Position(
        trade_id='SPREAD-001',
        symbol='NIFTY 50',
        instrument='NIFTY2692223450CE/NIFTY2692223600CE',
        underlying='NIFTY',
        direction=Direction.SHORT,
        quantity=65,
        entry_price=32.0,
        entry_time=datetime(2026, 9, 21, 10, 0, tzinfo=IST),
        stop_loss=48.0,
        target=9.6,
        max_bars=16,
        bars_held=3,
        strategy='Hedged_Credit_Spread',
        entry_spot=23380.0,
        atr=25.0,
    )
    snap = TradeSnapshot(
        timestamp=datetime(2026, 9, 21, 10, 45, tzinfo=IST),
        symbol='NIFTY 50',
        instrument=pos.instrument,
        open=30.0, high=31.0, low=29.0, close=30.0,
        bar_interval='15minute'
    )
    # Stagnant price (30.0 vs entry 32.0) held 3 bars: should NOT exit via INACTIVITY_KILL_SWITCH
    exit_px, reason = tracker._evaluate_exit(pos, snap, is_session_end=False, is_square_off=False)
    assert reason != ExitReason.INACTIVITY_KILL_SWITCH, f'Credit spread should not trigger inactivity kill switch, got {reason}'

def test_credit_spread_statistical_buffer_widened():
    strat = CreditSpreadStrategy()
    dates = pd.date_range('2026-09-21 09:15', periods=30, freq='15min', tz=IST)
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [23350.0 + i*2 for i in range(30)],
        'high': [23360.0 + i*2 for i in range(30)],
        'low': [23340.0 + i*2 for i in range(30)],
        'close': [23355.0 + i*2 for i in range(30)],
        'volume': [10000]*30
    })
    res = strat.evaluate_spread(df, symbol='NIFTY 50')
    if res:
        strike_dist = abs(res['short_strike'] - df['close'].iloc[-1])
        assert strike_dist >= 140.0, f'Strike distance {strike_dist} should be >= 140 pts OTM on Nifty'

def test_paper_capture_side_labeling_and_dedup(tmp_path):
    from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
    cfg = CaptureConfig(enabled=True, csv_path=str(tmp_path / 'test_cs.csv'), sqlite_path=str(tmp_path / 'test_cs.sqlite'))
    capture = LivePaperCapture(cfg, DummyFeed())
    notif_dict = {
        'symbol': 'NIFTY 50',
        'instrument': 'NIFTY2692223450CE/NIFTY2692223600CE',
        'tradingsymbol': 'NIFTY2692223450CE/NIFTY2692223600CE',
        'action': 'BEAR_CALL_SPREAD',
        'spread_type': 'BEAR_CALL_SPREAD',
        'strategy_type': 'credit_spread',
        'direction': 'neutral_range',
        'strategy': 'Hedged_Credit_Spread',
        'entry_price': 30.0,
        'stop_loss': 45.0,
        'target': 10.0,
        'signal_score': 8.5
    }
    tid = capture.on_signal(notif_dict)
    assert tid is not None, 'Initial spread should open'
    
    # Simulate trade closing
    closed_trade = capture._engine.tracker.close_position(tid, datetime.now(IST), 28.0, ExitReason.TARGET)
    capture._on_trade_closed(closed_trade)
    
    # Re-entry attempt with SAME instrument should be blocked
    tid2 = capture.on_signal(notif_dict)
    assert tid2 is None, 'Repeat entry on same instrument closed today should be blocked'


def test_credit_spread_offline_evaluate_exit():
    feed = DummyFeed()
    fill_sim = FillSimulator(feed=feed)
    tracker = PositionTracker(fill_sim=fill_sim)
    
    pos = Position(
        trade_id='SPREAD-TGT-01',
        symbol='SENSEX',
        instrument='SENSEX26O0173500CE/SENSEX26O0173900CE',
        underlying='SENSEX',
        direction=Direction.SHORT,
        quantity=20,
        entry_price=59.04,
        entry_time=datetime(2026, 9, 30, 12, 30, tzinfo=IST),
        stop_loss=92.25,
        target=18.45,
        max_bars=16,
        bars_held=2,
        strategy='Hedged_Credit_Spread',
    )
    
    # 1. Bar does not breach SL or Target (spread at 35.0) -> No exit
    snap_normal = TradeSnapshot(
        timestamp=datetime(2026, 9, 30, 13, 0, tzinfo=IST),
        symbol='SENSEX',
        instrument=pos.instrument,
        open=45.0, high=48.0, low=35.0, close=40.0,
        bar_interval='15minute'
    )
    px, reason = tracker._evaluate_exit(pos, snap_normal, is_session_end=False, is_square_off=False)
    assert reason is None, f"Expected no exit, got {reason}"
    
    # 2. Bar breaches target (low touched 17.45 <= 18.45) -> TARGET exit
    snap_target = TradeSnapshot(
        timestamp=datetime(2026, 9, 30, 14, 30, tzinfo=IST),
        symbol='SENSEX',
        instrument=pos.instrument,
        open=25.0, high=26.0, low=17.45, close=18.0,
        bar_interval='15minute'
    )
    px, reason = tracker._evaluate_exit(pos, snap_target, is_session_end=False, is_square_off=False)
    assert reason == ExitReason.TARGET, f"Expected TARGET exit, got {reason}"
    assert px == 18.45

    # 3. Bar breaches SL (high touched 95.0 >= 92.25) -> STOP_LOSS exit
    snap_sl = TradeSnapshot(
        timestamp=datetime(2026, 9, 30, 14, 30, tzinfo=IST),
        symbol='SENSEX',
        instrument=pos.instrument,
        open=70.0, high=95.0, low=68.0, close=94.0,
        bar_interval='15minute'
    )
    px, reason = tracker._evaluate_exit(pos, snap_sl, is_session_end=False, is_square_off=False)
    assert reason == ExitReason.STOP_LOSS, f"Expected STOP_LOSS exit, got {reason}"
    assert px == 92.25


def test_credit_spread_trailing_stop_strictly_excluded():
    feed = DummyFeed()
    fill_sim = FillSimulator(feed=feed)
    tracker = PositionTracker(fill_sim=fill_sim)
    
    pos = Position(
        trade_id='SPREAD-TRAIL-01',
        symbol='SENSEX',
        instrument='SENSEX26O0173500CE/SENSEX26O0173900CE',
        underlying='SENSEX',
        direction=Direction.SHORT,
        quantity=20,
        entry_price=59.04,
        entry_time=datetime(2026, 9, 30, 12, 30, tzinfo=IST),
        stop_loss=92.25,
        target=18.45,
        max_bars=16,
        bars_held=4,
        strategy='Hedged_Credit_Spread',
    )
    
    # Simulate favorable spread price decay from 59.04 to 25.0
    original_sl = pos.stop_loss
    tracker._maybe_advance_trailing_stop(pos, current_price=25.0)
    assert pos.stop_loss == original_sl, "Trailing stop should NOT be modified for credit spreads"
    assert not pos.breakeven_set, "Breakeven flag should not be set by trailing stop on credit spreads"


def test_telegram_2leg_spread_alerts(monkeypatch):
    from prometheus.interface.telegram_bot import TelegramBot
    
    sent_messages = []
    bot = TelegramBot(bot_token="", chat_id="")
    monkeypatch.setattr(bot, 'send_message', lambda msg: sent_messages.append(msg))
    
    # 1. Test alert_trade_closed for 2-leg spread
    bot.alert_trade_closed({
        'symbol': 'SENSEX',
        'instrument': 'SENSEX26O0173500CE/SENSEX26O0173900CE',
        'side': 'SHORT',
        'quantity': 20,
        'entry_price': 59.04,
        'exit_price': 17.45,
        'gross_pnl': 831.80,
        'net_pnl': 711.11,
        'return_pct': 60.22,
        'exit_reason': 'target',
        'holding_duration_seconds': 7200,
        'costs': 120.69,
    })
    
    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert "SENSEX 1 OCT 73500 CE" in msg
    assert "SENSEX 1 OCT 73900 CE" in msg
    assert "BUY back Leg 1 (Short) first" in msg
    assert "Exit Spread on Kite" in msg
    
    # 2. Test alert_new_signal for 2-leg spread
    sent_messages.clear()
    bot.alert_new_signal({
        'symbol': 'SENSEX',
        'tradingsymbol': 'SENSEX26O0173500CE/SENSEX26O0173900CE',
        'action': 'BEAR_CALL_SPREAD',
        'entry': 59.04,
        'sl': 92.25,
        'target': 18.45,
        'lots': 1,
        'quantity': 20,
        'strategy': 'Hedged_Credit_Spread',
        'signal_score': 8.1,
        'legs': [
            {'symbol': 'SENSEX26O0173500CE', 'action': 'SELL', 'premium': 96.5, 'is_hedge': False},
            {'symbol': 'SENSEX26O0173900CE', 'action': 'BUY', 'premium': 37.46, 'is_hedge': True},
        ]
    })
    
    assert len(sent_messages) == 1
    new_sig_msg = sent_messages[0]
    assert "SENSEX 1 OCT 73500 CE" in new_sig_msg
    assert "SENSEX 1 OCT 73900 CE" in new_sig_msg
    assert "Add BUY Hedge leg FIRST" in new_sig_msg

