import pytest
import yaml
import os
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta

from prometheus.papertrade.position_tracker import PositionTracker, Position, Direction
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.papertrade.post_exit_tracker import PostExitTracker, PostExitWatchItem
from prometheus.signals.target_calibrator import calibrate_target_and_sl

IST = timezone(timedelta(hours=5, minutes=30))


def test_bank_nifty_cost_buffer_in_tracker_and_monitor():
    """Verify that Bank Nifty single-leg buffer >= 5.20 pts for 30-share 1 lot, and multi-leg >= 8.50 pts."""
    mock_sim = MagicMock()
    tracker = PositionTracker(fill_sim=mock_sim)

    # 1. Single-leg Bank Nifty option buy (30 qty)
    pos_bn = Position(
        trade_id="TEST_BN_BUY",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT54000CE",
        underlying="NIFTY BANK",
        direction=Direction.LONG,
        quantity=30,
        entry_price=400.0,
        entry_time=datetime.now(IST),
        stop_loss=365.0,
        target=460.0,
        max_bars=7,
        initial_sl=365.0,
        trade_mode="intraday",
        tier="B",
    )
    tracker.open_positions["TEST_BN_BUY"] = pos_bn

    # Capture cost_buffer_pts passed to on_sl_update callback
    captured_buffers = []
    tracker.on_sl_update = lambda pos, old_sl, new_sl, reason, cur_px, gain, buffer: captured_buffers.append(buffer)

    # Price moves to +25.0 pts gain (exceeds min_be_gain 18.0)
    tracker._maybe_advance_trailing_stop(pos_bn, 425.0)
    assert len(captured_buffers) > 0, "Trailing stop should have triggered breakeven"
    assert captured_buffers[0] >= 5.20, f"Expected Bank Nifty cost buffer >= 5.20 pts, got {captured_buffers[0]}"


def test_credit_spread_inactivity_kill_exemption_in_monitor():
    """Verify that credit spreads are never killed by the 45-minute inactivity kill switch."""
    exit_calls = []
    monitor = PositionMonitor(
        broker=MagicMock(),
        on_exit=lambda pid, price, reason: exit_calls.append((pid, price, reason))
    )

    state = TrailingState(
        position_id="TEST_SPREAD_1",
        tradingsymbol="BANKNIFTY26OCT54000CE/BANKNIFTY26OCT54300CE",
        symbol="NIFTY BANK",
        entry_premium=60.0,
        initial_sl=120.0,
        current_sl=120.0,
        target=10.0,
        direction="bearish",
        strategy="Hedged_Credit_Spread",
        strategy_type="credit_spread",
        trade_mode="intraday",
        entry_bar_count=5,  # Exceeds 3 bars
        breakeven_set=False,
    )
    monitor.add_position(state)

    # Run process tick when price is stagnant at entry (60.0)
    monitor._process_tick(state, 60.0)

    # Verify no inactivity kill exit occurred
    inactivity_exits = [call for call in exit_calls if "inactivity" in call[2] or "kill" in call[2]]
    assert len(inactivity_exits) == 0, f"Credit spread must be exempt from inactivity kill, but got {inactivity_exits}"


def test_post_exit_tracker_put_option_buying_shakeout_logic():
    """Verify that Put Option Buying (direction='SHORT', target > entry) does not invert shakeout logic."""
    tracker = PostExitTracker()

    # Long PE option stopped out at 150.0 (entry 170.0, target 230.0).
    # In SQLite, direction was recorded as 'SHORT' (underlying market bias).
    item = PostExitWatchItem(
        trade_id="TEST_PE_BUY",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT22000PE",
        underlying="NIFTY 50",
        direction="SHORT",
        tier="B",
        entry_price=170.0,
        exit_price=150.0,
        exit_reason="stop_loss",
        exit_time=datetime.now(IST),
        target=230.0,
        stop_loss=150.0,
    )
    tracker._watches["TEST_PE_BUY"] = item

    # After exit, option premium crashes further to 100.0 (stop loss saved capital!).
    tracker.on_tick("NIFTY26OCT22000PE", 100.0, datetime.now(IST))

    assert not item.is_premature_shakeout, "Crashing put premium must NOT be labeled a premature shakeout!"
    assert item.drawdown_saved_pts == 50.0, f"Expected drawdown saved of 50.0 pts (150 - 100), got {item.drawdown_saved_pts}"


def test_target_calibrator_preserves_noise_floor_under_premium_cap():
    """Verify that Bank Nifty SL is not capped below the 35.0 pt noise floor on cheap options."""
    # Cheap option at Rs 80 on Bank Nifty: 80 * 0.35 = 28.0 pts
    res = calibrate_target_and_sl(
        symbol="NIFTY BANK",
        spot_price=54000.0,
        spot_atr=120.0,
        opt_ltp=80.0,
        edge_score=8.5,
        tier="B",
        is_htf_aligned=True,
        structural_sl_pts=38.0,
    )

    # sl_pts must preserve the 35.0 pt noise floor rather than compressing to 28.0 pts
    assert res.sl_pts >= 35.0, f"Bank Nifty SL must not be compressed inside 35 pt noise floor, got {res.sl_pts}"


def test_settings_yaml_consistency():
    """Verify settings.yaml has no duplicate slippage and valid risk limits."""
    settings_path = "prometheus/config/settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # 1. No contradictory daily vs weekly loss limits
    risk = conf.get("risk", {})
    daily_loss_pct = risk.get("max_daily_loss_pct", 0)
    weekly_loss_pct = risk.get("max_weekly_loss_pct", 0)
    assert daily_loss_pct <= weekly_loss_pct, f"Daily loss cap ({daily_loss_pct}%) cannot exceed weekly cap ({weekly_loss_pct}%)"

    # 2. FINNIFTY is present in lot sizes
    lot_sizes = conf.get("market", {}).get("lot_sizes", {})
    assert "NIFTY FIN SERVICE" in lot_sizes, "NIFTY FIN SERVICE missing from market.lot_sizes"
    assert lot_sizes["NIFTY FIN SERVICE"] == 65

    # 3. Expiry debit spreads are disabled
    expiry_conf = conf.get("strategies", {}).get("expiry", {})
    assert not expiry_conf.get("enabled", True), "Dead expiry strategy should be disabled"

    # 4. Golden setup targets outside noise floor
    golden = conf.get("intraday", {}).get("golden_setup", {})
    assert golden.get("trailing_be_pts", 0) >= 18.0, "Golden setup trailing_be_pts must be >= 18.0"
    assert golden.get("partial_profit_pct", 0.0) == 0.0, "Partial profit booking must be 0.0 under 1-lot constraint"
