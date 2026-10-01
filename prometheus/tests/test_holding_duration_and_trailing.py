"""
Unit tests for holding duration standardization and tier-differentiated trailing stop mechanics.

Verifies:
1. Intra-bar quick exit yields positive duration (holding_duration_seconds > 0).
2. Tier C triggers offensive micro-lock at +12 pts gain (Bank Nifty/Sensex) or +5 pts (Nifty).
3. Tier S/B preserves half-risk cut and does not prematurely micro-lock inside spread noise.
4. Credit spreads remain strictly exempt from trailing ratchets.
5. CapitalBracketManager and RiskManager single-lot limit (max_lots_per_trade: 1) and risk guards.
"""

import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from prometheus.utils.indian_market import IST
from prometheus.papertrade.types import (
    Position, Direction, ExitReason, TradeSnapshot, PaperTrade,
)
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.papertrade.engine import PaperTradeEngine
from prometheus.papertrade.signal_source import SignalNotification, SignalSource
from prometheus.papertrade.recorder import TradeRecorder
from prometheus.risk.manager import RiskManager
from prometheus.risk.position_sizer import CapitalBracketManager


class MockFeed:
    """Mock price feed for FillSimulator."""
    def __init__(self, ltp_dict=None):
        self.ltp_dict = ltp_dict or {}

    def get_ltp(self, symbol, instrument=None):
        key = instrument or symbol
        return self.ltp_dict.get(key, 100.0)


class MockSignalSource(SignalSource):
    def __init__(self, signals):
        self._signals = list(signals)

    def next_batch(self):
        s = self._signals
        self._signals = []
        return s

    def close(self):
        pass


# ==============================================================================
# 1. Holding Duration Standardization Tests
# ==============================================================================

def test_holding_duration_standardization_positive_on_quick_exit():
    """Verify intra-bar quick exit yields positive wall-clock duration (> 0s)."""
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
    )

    entry_time = datetime.now(IST)
    pos = Position(
        trade_id="DUR-TEST-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=30,
        entry_price=1000.0,
        entry_time=entry_time,
        stop_loss=975.0,
        target=1050.0,
        max_bars=16,
        tier="B",
    )
    tracker.open_position(pos)

    # Simulate close_position called with the exact same bar timestamp (the historical bug trigger)
    # The standardized exit_time must be datetime.now(IST) and duration >= 1.
    same_bar_timestamp = entry_time
    trade = tracker.close_position(
        trade_id="DUR-TEST-01",
        timestamp=same_bar_timestamp,
        exit_price=975.0,
        exit_reason=ExitReason.STOP_LOSS,
    )

    assert trade is not None
    assert trade.holding_duration_seconds > 0
    assert trade.holding_duration_seconds >= 1
    assert trade.exit_time is not None
    assert trade.exit_time >= trade.entry_time


def test_holding_duration_with_engine_and_bar_timestamp():
    """Verify PaperTradeEngine sets entry_time=datetime.now(IST) and preserves bar_timestamp."""
    bar_ts = datetime(2026, 10, 1, 10, 30, 0)
    sig = SignalNotification(
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        strike=55000.0,
        option_type="CE",
        expiry="2026-10-29",
        entry_price_hint=1014.72,
        stop_loss=992.66,
        target=1076.82,
        signal_score=0.85,
        signal_confidence=0.80,
        max_bars=16,
        trade_mode="intraday",
        strategy="PriceAction_Momentum",
        bar_timestamp=bar_ts,
        tier="C",
        target_gain_pts=62.1,
    )

    feed = MockFeed({"BANKNIFTY26OCT55000CE": 1014.72})
    recorder = TradeRecorder(sqlite_path=None, csv_path=None)
    engine = PaperTradeEngine(
        feed=feed,
        signal_source=MockSignalSource([sig]),
        recorder=recorder,
        enable_trailing=True,
    )

    trade_ids = engine.gather_new_signals()
    assert len(trade_ids) == 1
    tid = engine.process_new_signal(trade_ids[0])
    assert tid is not None

    pos = engine.tracker.open_positions[tid]
    # Verify entry_time is IST-aware wall-clock time
    assert pos.entry_time.tzinfo is not None
    now_ist = datetime.now(IST)
    assert abs((now_ist - pos.entry_time).total_seconds()) < 5.0
    # Verify bar_timestamp is recorded separately on the position
    assert getattr(pos, "bar_timestamp", None) is not None
    assert pos.bar_timestamp.year == 2026

    # Feed an exit bar with the same bar timestamp
    snap = TradeSnapshot(
        timestamp=bar_ts,
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        open=1010.0,
        high=1015.0,
        low=990.0,  # breaches SL 992.66
        close=991.0,
        bar_interval="15minute",
    )
    closed = engine.process_bar(snap)
    assert len(closed) == 1
    trade = closed[0]

    assert trade.holding_duration_seconds > 0
    assert trade.holding_duration_seconds >= 1
    assert trade.exit_time.tzinfo is not None
    assert trade.bar_timestamp is not None


# ==============================================================================
# 2. Tier C Offensive Micro-Lock Tests
# ==============================================================================

def test_tier_c_banknifty_offensive_microlock_at_12pts():
    """Verify Tier C on Bank Nifty triggers offensive micro-lock at +12 pts gain."""
    alerts = []
    def on_sl_update(pos, old_sl, new_sl, stage, current_price, gain_pts, cost_buffer_pts):
        alerts.append({
            "stage": stage,
            "old_sl": old_sl,
            "new_sl": new_sl,
            "current_price": current_price,
            "gain_pts": gain_pts,
        })

    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
        on_sl_update=on_sl_update,
    )

    pos = Position(
        trade_id="TIER-C-BN-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=30,
        entry_price=1000.0,
        entry_time=datetime.now(IST),
        stop_loss=975.0,  # initial risk = 25.0 pts
        target=1025.0,   # scalp target
        max_bars=16,
        tier="C",
        target_gain_pts=25.0,
    )
    tracker.open_position(pos)

    # Step 1: Gain is +11.0 pts (below 12.0 pts threshold)
    tracker._maybe_advance_trailing_stop(pos, 1011.0)
    assert pos.breakeven_set is False

    # Step 2: Gain reaches +12.0 pts (triggers Tier C offensive micro-lock)
    tracker._maybe_advance_trailing_stop(pos, 1012.0)
    assert pos.breakeven_set is True
    # Bank Nifty cost buffer is 1.9 pts -> new_sl = 1000.0 + 1.9 = 1001.90
    assert pos.stop_loss == pytest.approx(1001.90, 0.01)
    assert len(alerts) == 1
    assert alerts[0]["stage"] == "breakeven"
    assert alerts[0]["new_sl"] == pytest.approx(1001.90, 0.01)


def test_tier_c_sensex_offensive_microlock_at_12pts():
    """Verify Tier C on Sensex triggers offensive micro-lock at +12 pts gain."""
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
    )

    pos = Position(
        trade_id="TIER-C-SX-01",
        symbol="SENSEX",
        instrument="SENSEX26OCT75000CE",
        underlying="SENSEX",
        direction=Direction.LONG,
        quantity=20,
        entry_price=200.0,
        entry_time=datetime.now(IST),
        stop_loss=175.0,
        target=225.0,
        max_bars=16,
        tier="C",
        target_gain_pts=25.0,
    )
    tracker.open_position(pos)

    # At +12.0 pts gain (current_price = 212.0)
    tracker._maybe_advance_trailing_stop(pos, 212.0)
    assert pos.breakeven_set is True
    # Sensex cost buffer is 3.0 pts -> new_sl = 200.0 + 3.0 = 203.0
    assert pos.stop_loss == pytest.approx(203.0, 0.01)


def test_tier_c_nifty_offensive_microlock_at_5pts():
    """Verify Tier C on Nifty 50 triggers offensive micro-lock at +5 pts gain."""
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
    )

    pos = Position(
        trade_id="TIER-C-NF-01",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT24000CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(IST),
        stop_loss=90.0,
        target=112.0,
        max_bars=16,
        tier="C",
        target_gain_pts=12.0,
    )
    tracker.open_position(pos)

    # Gain is +4.5 pts (< 5.0 pts)
    tracker._maybe_advance_trailing_stop(pos, 104.50)
    assert pos.breakeven_set is False

    # Gain is +5.0 pts (>= 5.0 pts)
    tracker._maybe_advance_trailing_stop(pos, 105.0)
    assert pos.breakeven_set is True
    # Nifty cost buffer is 0.9 pts -> new_sl = 100.0 + 0.9 = 100.90
    assert pos.stop_loss == pytest.approx(100.90, 0.01)


# ==============================================================================
# 3. Tier S/B Defensive Runner Ladder Tests
# ==============================================================================

def test_tier_b_banknifty_preserves_half_risk_cut_survives_noise():
    """
    Verify Tier B on Bank Nifty:
    - At +12 pts (0.48R): cuts risk 50% (SL moves to 987.50, NOT breakeven).
    - Preserves >20 pt cushion outside 10-15 pt spread noise.
    - Pullback to 1001.0 does NOT stop out (would have stopped out Tier C).
    - At >= +18 pts (min_be_gain): ratchets safely to breakeven (1001.90).
    """
    alerts = []
    def on_sl_update(pos, old_sl, new_sl, stage, current_price, gain_pts, cost_buffer_pts):
        alerts.append({
            "stage": stage,
            "old_sl": old_sl,
            "new_sl": new_sl,
            "current_price": current_price,
            "gain_pts": gain_pts,
        })

    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
        on_sl_update=on_sl_update,
    )

    pos = Position(
        trade_id="TIER-B-BN-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=30,
        entry_price=1000.0,
        entry_time=datetime.now(IST),
        stop_loss=975.0,  # initial risk = 25.0 pts
        target=1080.0,   # runner target
        max_bars=16,
        tier="B",
        target_gain_pts=80.0,
    )
    tracker.open_position(pos)

    # Step 1: Gain reaches +12.0 pts (0.48R progress)
    tracker._maybe_advance_trailing_stop(pos, 1012.0)

    # Must be in half_risk stage, NOT breakeven!
    assert pos.half_risk_set is True
    assert pos.breakeven_set is False
    # SL cut 50% of risk: 1000.0 - 0.5 * 25.0 = 987.50
    assert pos.stop_loss == pytest.approx(987.50, 0.01)
    assert len(alerts) == 1
    assert alerts[0]["stage"] == "half_risk"
    assert alerts[0]["new_sl"] == pytest.approx(987.50, 0.01)

    # Step 2: Micro-pullback to 1001.0 (inside spread noise envelope)
    # Tier C would have exited here at 1001.90. Tier B survives with 13.5 pt buffer!
    assert 1001.0 > pos.stop_loss

    # Step 3: Rally extends to +18.5 pts (>= min_be_gain 18.0 pts)
    tracker._maybe_advance_trailing_stop(pos, 1018.5)

    # Now breakeven sets safely outside noise floor
    assert pos.breakeven_set is True
    assert pos.stop_loss == pytest.approx(1001.90, 0.01)
    assert len(alerts) == 2
    assert alerts[1]["stage"] == "breakeven"


def test_tier_s_multi_stage_runner_trailing():
    """Verify Tier S setup progresses through multi-stage runner locks."""
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
    )

    pos = Position(
        trade_id="TIER-S-BN-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=30,
        entry_price=1000.0,
        entry_time=datetime.now(IST),
        stop_loss=975.0,  # risk = 25.0 pts
        target=1100.0,
        max_bars=16,
        tier="S",
        target_gain_pts=100.0,
    )
    tracker.open_position(pos)

    # Stage 0A: +12 pts -> Half-risk cut
    tracker._maybe_advance_trailing_stop(pos, 1012.0)
    assert pos.stop_loss == pytest.approx(987.50, 0.01)
    assert pos.breakeven_set is False

    # Stage 0B: +20 pts (>= min_be_gain 18.0) -> Breakeven
    tracker._maybe_advance_trailing_stop(pos, 1020.0)
    assert pos.stop_loss == pytest.approx(1001.90, 0.01)
    assert pos.breakeven_set is True

    # Stage 3: 1.0R progress (+25 pts, current 1025.0) -> Lock 20%
    tracker._maybe_advance_trailing_stop(pos, 1025.0)
    # SL = 1000 + 0.20 * 25 = 1005.0
    assert pos.stop_loss == pytest.approx(1005.0, 0.01)

    # Stage 4: 2.0R progress (+50 pts, current 1050.0) -> Lock 50%
    tracker._maybe_advance_trailing_stop(pos, 1050.0)
    # SL = 1000 + 0.50 * 25 = 1012.50
    assert pos.stop_loss == pytest.approx(1012.50, 0.01)

    # Stage 5: 3.0R progress (+75 pts, current 1075.0) -> Lock 70%
    tracker._maybe_advance_trailing_stop(pos, 1075.0)
    # SL = 1000 + 0.70 * 25 = 1017.50
    assert pos.stop_loss == pytest.approx(1017.50, 0.01)


# ==============================================================================
# 4. Credit Spread Trailing Exemption Tests
# ==============================================================================

def test_credit_spreads_strictly_exempt_from_trailing():
    """Verify Credit Spreads are completely bypassed by trailing ratchets."""
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=MockFeed(), slippage_bps=0),
        cost_model=CostModel(cost_per_side_bps=0.0),
        enable_trailing=True,
    )

    pos = Position(
        trade_id="SPREAD-TEST-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY56500CE/BANKNIFTY56700CE",
        underlying="BANKNIFTY",
        direction=Direction.SHORT,
        quantity=30,
        entry_price=40.0,
        entry_time=datetime.now(IST),
        stop_loss=60.0,
        target=10.0,
        max_bars=16,
        strategy="Hedged_Credit_Spread",
        tier="C",
    )
    tracker.open_position(pos)

    # Even at large price changes, stop_loss remains untouched
    tracker._maybe_advance_trailing_stop(pos, 80.0)
    assert pos.stop_loss == 60.0
    assert pos.breakeven_set is False
    assert pos.half_risk_set is False


# ==============================================================================
# 5. Single-Lot Limit & Risk Audit Verification
# ==============================================================================

def test_risk_manager_strictly_enforces_single_lot_limit():
    """Verify RiskManager single-lot mandate (max_lots_per_trade: 1)."""
    rm = RiskManager(
        config={"max_lots_per_trade": 1, "max_single_position_pct": 30.0},
        initial_capital=200000.0,
    )
    assert rm.max_lots_per_trade == 1

    sizing = rm.calculate_position_size(
        entry_price=100.0,
        stop_loss=90.0,
        lot_size=30,  # Bank Nifty lot size
    )
    assert sizing["lots"] == 1
    assert sizing["quantity"] == 30


def test_capital_bracket_manager_risk_parameters():
    """Verify CapitalBracketManager provides structured risk boundaries."""
    cbm = CapitalBracketManager(config={
        "brackets": {
            "tier_1": {
                "name": "Small_15K",
                "max_capital": 15000,
                "max_loss_per_trade": 600,
                "min_rr": 1.5,
                "base_target": 2.0,
                "sl_atr_mult": 1.0,
            },
            "tier_2": {
                "name": "Standard_50K",
                "max_capital": 50000,
                "max_loss_per_trade": 1500,
                "min_rr": 2.0,
                "base_target": 2.5,
                "sl_atr_mult": 1.2,
            },
        }
    })

    bracket_15k = cbm.get_bracket(15000.0)
    assert bracket_15k.name == "Small_15K"
    assert bracket_15k.max_loss_per_trade == 600

    bracket_50k = cbm.get_bracket(35000.0)
    assert bracket_50k.name == "Standard_50K"
    assert bracket_50k.max_loss_per_trade == 1500
