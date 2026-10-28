"""
Unit and regression tests for P0 Put Target Inversion Bug Fix.
Verifies that:
1. Option Buying (both CE and PE) always expands target ABOVE entry price (effective_target > fill_px).
2. Stop loss for option buying is strictly BELOW entry price (effective_sl < fill_px).
3. Option Selling (Credit Spreads) retains decay target (effective_target < fill_px) and risk stop loss (effective_sl > fill_px).
4. Put Buying does not exit immediately on ticks slightly above fill price.
5. Post-exit tracker treats Put Buying as long premium (alpha left on table = MFE - exit_price).
"""

import pytest
from datetime import datetime, timezone, timedelta
from prometheus.papertrade import PaperTradeEngine
from prometheus.papertrade.recorder import TradeRecorder
from prometheus.papertrade.types import Direction, TradeSnapshot, ExitReason
from prometheus.papertrade.signal_source import SignalNotification
from prometheus.papertrade.post_exit_tracker import PostExitTracker

IST = timezone(timedelta(hours=5, minutes=30))


class DummyFeed:
    """Mock feed for testing."""
    def __init__(self, prices=None):
        self.prices = prices or {}

    def get_ltp(self, token_or_symbol):
        return self.prices.get(token_or_symbol, 0.0)

    def get_quote(self, token_or_symbol):
        return None


def create_test_engine(feed_prices=None):
    feed = DummyFeed(feed_prices or {})
    engine = PaperTradeEngine(
        feed=feed,
        signal_source=None,
        recorder=TradeRecorder(sqlite_path=None, csv_path=None),
        lot_size_override=1,
        enable_trailing=True,
    )
    return engine, feed


def test_put_buying_target_and_sl_expansion():
    """
    Assert that Put Buying (Direction.SHORT, single-leg PE) sets target ABOVE entry
    and stop loss BELOW entry, rather than treating it as a credit spread.
    This directly reproduces today's SENSEX trade parameters:
    Hint: 424.80, Target: 456.80 (+32.0 pts), SL: 276.12 (-148.68 pts).
    """
    engine, feed = create_test_engine({"SENSEX26OCT73100PE": 418.67})

    sig = SignalNotification(
        symbol="SENSEX",
        instrument="SENSEX26OCT73100PE",
        underlying="SENSEX",
        direction=Direction.SHORT,
        strike=73100.0,
        option_type="PE",
        expiry="2026-10-30",
        entry_price_hint=424.80,
        stop_loss=276.12,
        target=456.80,
        target_gain_pts=32.0,
        signal_score=8.5,
        signal_confidence=0.8,
        max_bars=16,
        trade_mode="intraday",
        strategy="PriceAction",
        bar_timestamp=datetime.now(IST),
        metadata={},
    )

    trade_id = engine.process_new_signal(sig)
    assert trade_id is not None
    assert trade_id in engine.tracker.open_positions

    pos = engine.tracker.open_positions[trade_id]

    # Target MUST expand above entry_price by target_gain_pts (32.0)
    # Target should be entry_price + 32.0, NOT 0.90x entry_price
    expected_tgt = round(pos.entry_price + 32.0, 2)
    assert pos.target == expected_tgt
    assert pos.target > pos.entry_price

    # Stop Loss MUST be below entry_price by risk distance (424.80 - 276.12 = 148.68)
    expected_sl = round(pos.entry_price - 148.68, 2)
    assert pos.stop_loss == expected_sl
    assert pos.stop_loss < pos.entry_price


def test_put_buying_does_not_prematurely_exit_on_minor_tick():
    """
    Verify that an incoming tick at 431.10 (which triggered the premature exit today)
    does NOT trigger ExitReason.TARGET on the corrected position.
    """
    engine, feed = create_test_engine({"SENSEX26OCT73100PE": 418.67})

    sig = SignalNotification(
        symbol="SENSEX",
        instrument="SENSEX26OCT73100PE",
        underlying="SENSEX",
        direction=Direction.SHORT,
        strike=73100.0,
        option_type="PE",
        expiry="2026-10-30",
        entry_price_hint=424.80,
        stop_loss=276.12,
        target=456.80,
        target_gain_pts=32.0,
        signal_score=8.5,
        signal_confidence=0.8,
        max_bars=16,
        trade_mode="intraday",
        strategy="PriceAction",
        bar_timestamp=datetime.now(IST),
        metadata={},
    )

    trade_id = engine.process_new_signal(sig)
    assert trade_id is not None
    pos = engine.tracker.open_positions[trade_id]

    # Feed updates with tick that previously caused false exit: 431.10
    feed.prices["SENSEX26OCT73100PE"] = 431.10
    snap = TradeSnapshot(
        timestamp=datetime.now(IST),
        symbol="SENSEX",
        instrument="SENSEX26OCT73100PE",
        open=pos.entry_price,
        high=431.10,
        low=pos.entry_price,
        close=431.10,
        volume=100,
        bar_interval="15minute",
    )

    closed = engine.process_bar(snap)
    # Trade MUST remain open — 431.10 is well below target pos.target (> 450)!
    assert closed == [], "Trade must NOT exit at 431.10"
    assert trade_id in engine.tracker.open_positions

    # Now simulate true target hit at 460.00 (which exceeds pos.target ~451.30)
    feed.prices["SENSEX26OCT73100PE"] = 460.00
    snap_tgt = TradeSnapshot(
        timestamp=datetime.now(IST),
        symbol="SENSEX",
        instrument="SENSEX26OCT73100PE",
        open=431.10,
        high=460.00,
        low=430.00,
        close=455.00,
        volume=100,
        bar_interval="15minute",
    )
    closed = engine.process_bar(snap_tgt)
    assert len(closed) == 1
    assert closed[0].exit_reason == ExitReason.TARGET


def test_credit_spread_still_retains_decay_target():
    """
    Assert that Credit Spreads (defined-risk option selling) still have target DECAY
    below entry credit, and stop loss SPIKE above entry credit.
    """
    engine, feed = create_test_engine({"BANKNIFTY24OCT51000CE/BANKNIFTY24OCT51200CE": 100.0})

    sig = SignalNotification(
        symbol="BANKNIFTY",
        instrument="BANKNIFTY24OCT51000CE/BANKNIFTY24OCT51200CE",
        underlying="BANKNIFTY",
        direction=Direction.SHORT,
        strike=51000.0,
        option_type="CE",
        expiry="2026-10-30",
        entry_price_hint=100.0,
        stop_loss=150.0,  # 1.5x credit
        target=50.0,     # 50% decay
        target_gain_pts=0.0,
        signal_score=7.0,
        signal_confidence=0.75,
        max_bars=16,
        trade_mode="intraday",
        strategy="Hedged_Credit_Spread",
        bar_timestamp=datetime.now(IST),
        metadata={},
    )

    trade_id = engine.process_new_signal(sig)
    assert trade_id is not None
    pos = engine.tracker.open_positions[trade_id]

    # For Credit Spreads, target is DECAY: target < entry_price
    assert pos.target < pos.entry_price
    # Stop Loss is SPIKE: stop_loss > entry_price
    assert pos.stop_loss > pos.entry_price


def test_post_exit_tracker_put_buying_alpha_left():
    """
    Assert that PostExitTracker treats Put Buying as long premium,
    measuring post-exit alpha left on table when premium rallies higher.
    """
    tracker = PostExitTracker(recorder=None, max_watch_seconds=3600)

    # Register closed Put Buying trade as a dict
    now = datetime.now(IST)
    trade_dict = {
        "trade_id": "PAPER-TEST-PE-01",
        "symbol": "SENSEX",
        "instrument": "SENSEX26OCT73100PE",
        "underlying": "SENSEX",
        "direction": "SHORT",
        "tier": "C",
        "entry_price": 418.67,
        "exit_price": 431.10,
        "exit_reason": "TARGET",
        "exit_time": now,
        "target": 450.67,
        "stop_loss": 269.99,
        "entry_spot": 79500.0,
        "holding_duration_seconds": 18,
    }
    item = tracker.register_closed_trade(trade_dict)
    assert item is not None

    # Simulate price rally in Put option to 500.0
    updates = tracker.on_tick("SENSEX26OCT73100PE", 500.0, timestamp=now + timedelta(seconds=60))
    assert len(updates) == 1
    updated = updates[0]

    # Post exit MFE should be 500.0
    assert updated.post_exit_mfe == 500.0
    # Alpha left on table should be 500.0 - 431.10 = 68.90 pts
    assert updated.alpha_left_pts == 68.90
    assert updated.target_reclaimed is True
