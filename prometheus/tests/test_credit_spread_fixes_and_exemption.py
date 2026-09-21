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
