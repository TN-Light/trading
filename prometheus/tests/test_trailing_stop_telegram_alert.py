import pytest
from datetime import datetime, time
from prometheus.papertrade.position_tracker import PositionTracker, Position, Direction, CostModel
from prometheus.papertrade.fill_simulator import FillSimulator

class MockFeed:
    def get_ltp(self, symbol, instrument):
        return 74.0

class MockTelegram:
    def __init__(self):
        self.alerts = []

    def alert_trailing_stop_updated(self, symbol, instrument, old_sl, new_sl, stage, current_price, gain_pts, cost_pts):
        self.alerts.append({
            "symbol": symbol,
            "instrument": instrument,
            "old_sl": old_sl,
            "new_sl": new_sl,
            "stage": stage,
            "current_price": current_price,
            "gain_pts": gain_pts,
            "cost_pts": cost_pts,
        })

    def send_message(self, text):
        pass

def test_trailing_stop_breakeven_alert_with_brokerage():
    mock_tg = MockTelegram()

    def on_sl_update(pos, old_sl, new_sl, stage, current_price, gain_pts, cost_buffer_pts):
        mock_tg.alert_trailing_stop_updated(
            pos.symbol, pos.instrument, old_sl, new_sl, stage, current_price, gain_pts, cost_buffer_pts
        )

    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0),
        enable_trailing=True,
        on_sl_update=on_sl_update,
    )

    pos = Position(
        trade_id="TEST-1",
        symbol="NIFTY 50",
        instrument="NIFTY08SEP2623800PE",
        direction=Direction.SHORT,
        quantity=65,
        entry_price=62.50,
        stop_loss=53.12,
        target=76.25,
        entry_time=datetime.now(),
        underlying="NIFTY",
        max_bars=16,
    )
    tracker.open_position(pos)

    # Current price moves to 73.50 (+11.0 pts, >= 10.0 + 0.9 cost buffer)
    tracker._maybe_advance_trailing_stop(pos, 73.50)

    # Verify SL moved to Entry + 0.9 cost buffer
    assert pos.breakeven_set is True
    assert pos.stop_loss == pytest.approx(63.40, 0.01)

    # Verify Telegram alert was emitted
    assert len(mock_tg.alerts) == 1
    alert = mock_tg.alerts[0]
    assert alert["symbol"] == "NIFTY 50"
    assert alert["instrument"] == "NIFTY08SEP2623800PE"
    assert alert["old_sl"] == 53.12
    assert alert["new_sl"] == pytest.approx(63.40, 0.01)
    assert alert["stage"] == "breakeven"
    assert alert["gain_pts"] == 11.0
    assert alert["cost_pts"] == 0.9
