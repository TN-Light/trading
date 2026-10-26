"""
Test Suite: October 26 Cost Buffer Recalibration and Squeeze Tag Certification
Validates:
1. Unified statutory fee hurdle floors across all 4 index underlyings.
2. Protection against negative net rupee breakevens on 1-lot option buying (GEMINI.md Rule 2).
3. is_short_gamma_squeeze tracking tag creation, persistence, and classification.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.types import Position, PaperTrade, Direction, ExitReason
from prometheus.papertrade.post_exit_tracker import PostExitTracker
from prometheus.papertrade.recorder import TRADE_COLUMNS


def test_unified_statutory_cost_buffers():
    """Verify that get_statutory_cost_buffer returns the safe calibrated floors."""
    # 1-lot Option Buying
    assert PositionTracker.get_statutory_cost_buffer("NIFTY 50", qty=65, is_spread=False) == 2.00
    assert PositionTracker.get_statutory_cost_buffer("SENSEX", qty=20, is_spread=False) == 5.50
    assert PositionTracker.get_statutory_cost_buffer("BANKNIFTY", qty=30, is_spread=False) == 5.50
    assert PositionTracker.get_statutory_cost_buffer("FINNIFTY", qty=65, is_spread=False) == 2.60

    # Multi-lot Option Buying
    assert PositionTracker.get_statutory_cost_buffer("NIFTY 50", qty=130, is_spread=False) == 1.20
    assert PositionTracker.get_statutory_cost_buffer("SENSEX", qty=40, is_spread=False) == 3.50
    assert PositionTracker.get_statutory_cost_buffer("BANKNIFTY", qty=60, is_spread=False) == 3.50
    assert PositionTracker.get_statutory_cost_buffer("FINNIFTY", qty=130, is_spread=False) == 1.80

    # Credit Spreads
    assert PositionTracker.get_statutory_cost_buffer("NIFTY 50", qty=65, is_spread=True) == 2.80
    assert PositionTracker.get_statutory_cost_buffer("SENSEX", qty=20, is_spread=True) == 8.50
    assert PositionTracker.get_statutory_cost_buffer("BANKNIFTY", qty=30, is_spread=True) == 8.50
    assert PositionTracker.get_statutory_cost_buffer("FINNIFTY", qty=65, is_spread=True) == 4.20


def test_nifty_single_lot_breakeven_yields_positive_gross():
    """Verify that a 2.00 pt Nifty buffer on 65 qty generates +Rs 130.00 gross, overcoming Rs 91.41 statutory friction."""
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))
    pos = Position(
        trade_id="TEST-NIFTY-2PT",
        symbol="NIFTY 50",
        instrument="NIFTY27OCT2622650CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=173.72,
        entry_time=datetime.now(),
        stop_loss=116.22,
        target=187.72,
        max_bars=16,
        tier="C",
    )
    tracker.open_positions[pos.trade_id] = pos

    # Micro-lock triggers at gain >= 5.0 pts
    tracker._maybe_advance_trailing_stop(pos, 179.85)
    assert pos.breakeven_set is True
    assert pos.stop_loss == pytest.approx(175.72, abs=1e-2)  # 173.72 + 2.00

    # Gross points gained at exit: 2.00 pts * 65 = Rs 130.00 (comfortably exceeds Rs 91.41 statutory friction)
    gross_pnl = (pos.stop_loss - pos.entry_price) * pos.quantity
    assert gross_pnl == pytest.approx(130.00, abs=1e-2)
    assert gross_pnl > 91.41


def test_is_short_gamma_squeeze_tag_lifecycle():
    """Verify that is_short_gamma_squeeze is set when net_gex < 0 and flows to PaperTrade and PostExitItem."""
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))
    pos = Position(
        trade_id="TEST-SQUEEZE-001",
        symbol="NIFTY 50",
        instrument="NIFTY27OCT2622650CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=173.72,
        entry_time=datetime.now(),
        stop_loss=116.22,
        target=187.72,
        max_bars=16,
        net_gex=-203901255170.0,  # Negative Net GEX -> Short Gamma Squeeze
        tier="C",
    )
    tracker.open_position(pos)
    assert pos.is_short_gamma_squeeze == 1

    # Close position
    trade = tracker.close_position(
        trade_id="TEST-SQUEEZE-001",
        timestamp=datetime.now(),
        exit_price=175.72,
        exit_reason=ExitReason.TRAILING_STOP,
    )
    assert trade is not None
    assert trade.is_short_gamma_squeeze == 1

    # Verify flow to PostExitTracker
    pe_tracker = PostExitTracker(recorder=None)
    watch_item = pe_tracker.register_closed_trade(trade)
    assert watch_item is not None
    assert watch_item.is_short_gamma_squeeze == 1


def test_recorder_columns_include_squeeze_tag():
    """Verify that TRADE_COLUMNS includes is_short_gamma_squeeze."""
    assert "is_short_gamma_squeeze" in TRADE_COLUMNS
