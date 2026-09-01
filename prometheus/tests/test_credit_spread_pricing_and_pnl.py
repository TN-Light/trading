"""Unit tests for multi-leg credit spread fill simulation and PnL calculation."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from prometheus.papertrade.types import Position, Direction, ExitReason
from prometheus.papertrade.fill_simulator import FillSimulator, PriceFeed
from prometheus.papertrade.position_tracker import PositionTracker, CostModel


class DummyPriceFeed:
    """Mock price feed providing individual leg quotes."""
    def __init__(self, prices: dict):
        self.prices = prices

    def get_ltp(self, instrument: str) -> float:
        return float(self.prices.get(instrument, 0.0))

    def get_quote(self, instrument: str):
        ltp = self.get_ltp(instrument)
        if ltp > 0:
            return (ltp, ltp * 0.999, ltp * 1.001)
        return None


def test_fill_simulator_2leg_spread_pricing():
    feed = DummyPriceFeed({
        "NIFTY24050CE": 30.0,
        "NIFTY24200CE": 10.0,
    })
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    
    # 2-leg combined instrument
    res = sim.fill("NIFTY24050CE/NIFTY24200CE", Direction.SHORT, side="BUY")
    # Expected spread value: 30.0 - 10.0 = 20.0
    assert res.fill_price == 20.0
    assert res.source == "live_spread_2leg"


def test_credit_spread_pnl_decay_profit():
    feed = DummyPriceFeed({
        "NIFTY24050CE": 15.0,
        "NIFTY24200CE": 5.0,
    })
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    tracker = PositionTracker(fill_sim=sim, cost_model=CostModel(cost_per_side_bps=0.0), enable_trailing=False)

    pos = Position(
        trade_id="TEST-SPREAD-01",
        symbol="NIFTY 50",
        instrument="NIFTY24050CE/NIFTY24200CE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        quantity=50,
        entry_price=25.0,  # 25.0 net credit collected at entry
        entry_time=datetime(2026, 9, 1, 10, 0),
        stop_loss=37.5,    # 1.5x entry credit
        target=12.5,       # 50% target decay
        max_bars=16,
        strategy="Hedged_Credit_Spread",
    )
    tracker.open_positions[pos.trade_id] = pos

    # Exit at decayed cost of 10.0 (Spread decayed from 25.0 -> 10.0)
    trade = tracker.close_position("TEST-SPREAD-01", exit_price=10.0, exit_reason=ExitReason.TARGET, timestamp=datetime(2026, 9, 1, 15, 15))
    
    assert trade is not None
    # Realized gross PnL = (entry_credit - exit_cost) * qty = (25.0 - 10.0) * 50 = +750.0
    assert trade.gross_pnl == 750.0
    assert trade.net_pnl == 750.0
    assert trade.return_pct == 60.0  # (15.0 / 25.0) * 100% = +60.0%


def test_credit_spread_pnl_expansion_loss():
    feed = DummyPriceFeed({})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    tracker = PositionTracker(fill_sim=sim, cost_model=CostModel(cost_per_side_bps=0.0), enable_trailing=False)

    pos = Position(
        trade_id="TEST-SPREAD-02",
        symbol="NIFTY 50",
        instrument="NIFTY24050CE/NIFTY24200CE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        quantity=50,
        entry_price=20.0,  # 20.0 net credit collected at entry
        entry_time=datetime(2026, 9, 1, 10, 0),
        stop_loss=30.0,    # 1.5x entry credit
        target=10.0,       # 50% target decay
        max_bars=16,
        strategy="Hedged_Credit_Spread",
    )
    tracker.open_positions[pos.trade_id] = pos

    # Exit at expanded cost of 30.0 (Stop loss hit)
    trade = tracker.close_position("TEST-SPREAD-02", exit_price=30.0, exit_reason=ExitReason.STOP_LOSS, timestamp=datetime(2026, 9, 1, 15, 15))
    
    assert trade is not None
    # Realized gross PnL = (entry_credit - exit_cost) * qty = (20.0 - 30.0) * 50 = -500.0
    assert trade.gross_pnl == -500.0
    assert trade.net_pnl == -500.0
    assert trade.return_pct == -50.0


def test_option_buying_pnl_still_correct():
    feed = DummyPriceFeed({})
    sim = FillSimulator(feed=feed, slippage_bps=0, use_bid_ask=False)
    tracker = PositionTracker(fill_sim=sim, cost_model=CostModel(cost_per_side_bps=0.0), enable_trailing=False)

    pos = Position(
        trade_id="TEST-BUY-01",
        symbol="NIFTY 50",
        instrument="NIFTY24000PE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        quantity=50,
        entry_price=50.0,  # 50.0 premium paid
        entry_time=datetime(2026, 9, 1, 10, 0),
        stop_loss=42.5,    # -15% SL
        target=65.0,       # +30% Target
        max_bars=16,
        strategy="PriceAction_Momentum",
    )
    tracker.open_positions[pos.trade_id] = pos

    # Option bought at 50, sold at 65 (Target hit)
    trade = tracker.close_position("TEST-BUY-01", exit_price=65.0, exit_reason=ExitReason.TARGET, timestamp=datetime(2026, 9, 1, 15, 15))
    
    assert trade is not None
    # Option Buying gross PnL = (exit - entry) * qty = (65.0 - 50.0) * 50 = +750.0
    assert trade.gross_pnl == 750.0
    assert trade.net_pnl == 750.0
    assert trade.return_pct == 30.0
