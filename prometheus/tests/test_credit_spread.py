"""
Unit tests for CreditSpreadStrategy and Inverted Decay Trailing in PositionMonitor.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta
import pytest

from prometheus.strategies.credit_spread import CreditSpreadStrategy
from prometheus.execution.position_monitor import PositionMonitor, TrailingState


def _generate_range_candles(n_bars=30, start_dt=datetime(2026, 8, 24, 9, 15)):
    """Generate realistic tight sideways OHLCV candles."""
    timestamps = [start_dt + timedelta(minutes=15 * i) for i in range(n_bars)]
    base_price = 24000.0
    
    rows = []
    current = base_price
    for i, ts in enumerate(timestamps):
        # Oscillate in tight 20-point range
        step = np.sin(i) * 6.0
        open_p = current
        high_p = open_p + 8.0
        low_p = open_p - 8.0
        close_p = open_p + step
        vol = 50000
        
        rows.append({
            "timestamp": ts,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
        })
        current = close_p
        
    return pd.DataFrame(rows)


from unittest.mock import MagicMock

def test_credit_spread_generation():
    """Verify CreditSpreadStrategy generates valid 2-leg spread on sideways market."""
    strategy = CreditSpreadStrategy()
    # Prior bars (20 bars) + today's 6 bars (09:15 to 10:30)
    prior_df = _generate_range_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15))
    today_df = _generate_range_candles(n_bars=6, start_dt=datetime(2026, 8, 24, 9, 15))
    df = pd.concat([prior_df, today_df], ignore_index=True)

    # Mock live option chain quotes
    class MockChain:
        def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
            return {"ltp": 45.0, "bid": 44.0, "ask": 46.0, "tradingsymbol": f"{symbol}{strike}{option_type}"}

    spread = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=50000.0, option_chain=MockChain())
    assert spread is not None, "Expected valid credit spread in sideways regime"
    assert spread["strategy_type"] == "credit_spread"
    assert spread["net_credit"] > 0
    assert spread["strike_width"] > 0
    assert len(spread["legs"]) == 2
    
    # Verify leg ordering (Hedge leg BUY first, short leg SELL second)
    hedge_leg = spread["legs"][0]
    short_leg = spread["legs"][1]
    assert hedge_leg["action"] == "BUY"
    assert hedge_leg["is_hedge"] is True
    assert short_leg["action"] == "SELL"
    assert short_leg["is_hedge"] is False
    
    # Verify exit thresholds
    assert spread["target_decay_price"] < spread["net_credit"]
    assert spread["hard_sl_price"] > spread["net_credit"]


def test_credit_spread_inverted_trailing_target():
    """Verify PositionMonitor exits credit spread when 70% decay target is hit."""
    exits = []
    def _on_exit(pos_id, price, reason):
        exits.append((pos_id, price, reason))

    mock_broker = MagicMock()
    monitor = PositionMonitor(broker=mock_broker, on_exit=_on_exit)
    
    state = TrailingState(
        position_id="CS-TEST-001",
        symbol="NIFTY 50",
        tradingsymbol="NIFTY26AUG24150PE/23950PE",
        entry_premium=40.0,  # Net credit = 40
        initial_sl=60.0,     # Hard SL = 60
        current_sl=60.0,
        target=12.0,         # Target = 12 (70% decay)
        direction="neutral_range",
        strategy_type="credit_spread",
        target_decay_price=12.0,
        breakeven_decay_price=20.0,
        hard_sl_price=60.0,
    )
    
    # Tick 1: Spread at 30.0 -> no exit
    monitor._process_tick(state, 30.0)
    assert len(exits) == 0
    
    # Tick 2: Spread decays to 11.5 (below target 12.0) -> Target Exit
    monitor._process_tick(state, 11.5)
    assert len(exits) == 1
    assert exits[0][2] == "target_decay_credit_spread"


def test_credit_spread_inverted_trailing_hard_sl():
    """Verify PositionMonitor exits credit spread when hard stop loss is breached."""
    exits = []
    def _on_exit(pos_id, price, reason):
        exits.append((pos_id, price, reason))

    mock_broker = MagicMock()
    monitor = PositionMonitor(broker=mock_broker, on_exit=_on_exit)
    
    state = TrailingState(
        position_id="CS-TEST-002",
        symbol="NIFTY 50",
        tradingsymbol="NIFTY26AUG24150PE/23950PE",
        entry_premium=40.0,
        initial_sl=60.0,
        current_sl=60.0,
        target=12.0,
        direction="neutral_range",
        strategy_type="credit_spread",
        target_decay_price=12.0,
        breakeven_decay_price=20.0,
        hard_sl_price=60.0,
    )
    
    # Spread price jumps to 62.0 (above hard SL 60.0) -> Hard SL Exit
    monitor._process_tick(state, 62.0)
    assert len(exits) == 1
    assert exits[0][2] == "stop_loss_credit_spread"
