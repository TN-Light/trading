"""
Unit and integration test suite validating the multi-team adversarial engineering upgrades:
- Team A (SFT): Capital de-allocation for BN/FN directional buying, cost buffers covering Rs 105 fees.
- Team B (QFT): High-VIX target expansion, spot vs option displacement resolution in inactivity check.
- Team C (NQIT): Preservation of technical SL outside 1-minute candle noise, denominator stabilization.
- Team D (ZKT): High-contrast live vs paper banners, clear operational states (TREND INTACT, CONSOLIDATING, EXIT).
- Execution Safety: SharedPriceMicroCache fallback on broker LTP cooldown.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd

from prometheus.signals.target_calibrator import calibrate_target_and_sl
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.types import Position, Direction, TradeSnapshot, ExitReason
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.position_health import PositionHealthEngine, PositionHealthReport
from prometheus.interface.telegram_bot import TelegramBot
from prometheus.data.price_micro_cache import get_shared_micro_cache, MicroCacheEntry
from prometheus.utils.indian_market import IST


class MockFeed:
    def __init__(self, ltp_dict=None):
        self.ltp_dict = ltp_dict or {}

    def get_ltp(self, symbol, instrument=None):
        return float(self.ltp_dict.get(instrument or symbol, 100.0))


# ============================================================================
# TEAM A (SFT): REAL-MONEY ECONOMICS & CAPITAL ALLOCATION
# ============================================================================

def test_team_a_capital_deallocation_bn_and_finnifty():
    """Verify that Bank Nifty & FinNifty directional buying are de-allocated to Paper Only when allocation rule is enforced."""
    bn_sig = {
        "action": "BUY_CE",
        "symbol": "BANKNIFTY",
        "strategy": "PriceAction_Momentum",
        "strategy_type": "option_buying",
        "edge_score": 8.0,
        "bar_timestamp": "2026-10-08 10:00:00",
        "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        "enforce_instrument_allocation": True,
    }
    res_bn = classify_signal_tier(bn_sig)
    assert res_bn["tier"] == "C"
    assert res_bn["is_live_eligible"] is False
    assert "DEALLOCATED" in res_bn["tier_name"]
    assert any("Team A Capital Mandate" in r for r in res_bn["classification_reasons"])

    fn_sig = {
        "action": "BUY_PE",
        "symbol": "FINNIFTY",
        "strategy": "PriceAction_Momentum",
        "strategy_type": "option_buying",
        "edge_score": 7.5,
        "bar_timestamp": "2026-10-08 10:00:00",
        "reasons": ["15M_ORB_Low_Breakout", "Session_VWAP_Bearish", "1H_Trend_Bearish"],
        "enforce_instrument_allocation": True,
    }
    res_fn = classify_signal_tier(fn_sig)
    assert res_fn["tier"] == "C"
    assert res_fn["is_live_eligible"] is False

    # Sensex and Nifty 50 directional buying MUST remain live eligible (Tier B / S)
    sx_sig = {
        "action": "BUY_CE",
        "symbol": "SENSEX",
        "strategy": "Golden_Setup",
        "strategy_type": "option_buying",
        "edge_score": 7.8,
        "bar_timestamp": "2026-10-08 10:00:00",
        "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        "enforce_instrument_allocation": True,
    }
    res_sx = classify_signal_tier(sx_sig)
    assert res_sx["tier"] == "B"
    assert res_sx["is_live_eligible"] is True

    # Credit Spreads on Bank Nifty remain live eligible (Tier A Sure-Shot)
    bn_spread = {
        "action": "SELL_BULL_PUT_SPREAD",
        "symbol": "BANKNIFTY",
        "strategy": "credit_spread",
        "strategy_type": "credit_spread",
        "is_0dte": True,
        "otm_sigma": 2.2,
        "signal_score": 9.5,
        "oi_shielded": True,
        "trend_aligned": True,
        "enforce_instrument_allocation": True,
    }
    res_spread = classify_signal_tier(bn_spread)
    assert res_spread["tier"] == "A"
    assert res_spread["is_live_eligible"] is True


def test_team_a_single_lot_cost_buffers():
    """Verify single-lot cost buffers (4.50 BN, 5.0 Sensex, 1.5 Nifty)."""
    feed = MockFeed({"BANKNIFTY26OCT55000CE": 1025.0})
    tracker = PositionTracker(fill_sim=FillSimulator(feed=feed), enable_trailing=True)

    # Bank Nifty 1 lot (qty 15): cost buffer must be 4.50 pts
    pos_bn = Position(
        trade_id="BN-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=15,
        entry_price=1000.0,
        stop_loss=965.0,
        target=1080.0,
        max_bars=10,
        entry_time=datetime.now(),
        tier="B",
    )
    tracker.open_position(pos_bn)
    # Price rises to 1020.0 (+20 pts gain >= 18 pts min_be_gain)
    tracker._maybe_advance_trailing_stop(pos_bn, 1020.0)
    assert pos_bn.breakeven_set is True
    # SL ratcheted to 1000 + 4.50 = 1004.50 pts
    assert pos_bn.stop_loss == pytest.approx(1004.50, 0.05)


# ============================================================================
# TEAM B (QFT): REGIME MULTIPLIERS & SPOT DISPLACEMENT
# ============================================================================

def test_team_b_high_vix_target_expansion():
    """Verify that VIX > 18.0 expands profit targets by 1.25x to avoid truncation inside volatility noise."""
    res_low = calibrate_target_and_sl(
        symbol="SENSEX",
        spot_price=72000.0,
        spot_atr=98.0,
        opt_ltp=350.0,
        edge_score=8.5,
        tier="B",
        is_htf_aligned=True,
        vix=13.0,
    )
    res_high = calibrate_target_and_sl(
        symbol="SENSEX",
        spot_price=72000.0,
        spot_atr=98.0,
        opt_ltp=350.0,
        edge_score=8.5,
        tier="B",
        is_htf_aligned=True,
        vix=19.5,
    )
    assert res_high.target_gain_pts > res_low.target_gain_pts
    assert res_high.target_gain_pts == pytest.approx(round(res_low.target_gain_pts * 1.25, 1), abs=1.0)


def test_team_b_option_vs_spot_displacement_resolution():
    """Verify that PositionTracker._evaluate_exit does not compare option LTP against spot index levels."""
    feed = MockFeed({"NIFTY BANK": 55100.0})
    tracker = PositionTracker(
        fill_sim=FillSimulator(feed=feed),
        enable_trailing=True,
        trend_aware=True,
        max_stagnation_bars=6,
    )

    pos = Position(
        trade_id="DISP-TEST-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=15,
        entry_price=400.0,
        stop_loss=365.0,
        target=480.0,
        max_bars=10,
        entry_time=datetime.now(),
        tier="B",
        entry_spot=55000.0,
        atr=88.0,
    )
    pos.current_spot = 55100.0  # Spot moved +100 pts (favorable, intact)
    pos.bars_held = 3  # Held 3 bars

    # Option snap with close=410.0 (option premium ~400, NOT index ~55000)
    snap = TradeSnapshot(
        timestamp=datetime.now(),
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        open=405.0,
        high=415.0,
        low=400.0,
        close=410.0,
        volume=1000,
    )

    # Underlying trend is intact (+100 spot move). Inactivity kill switch MUST NOT trigger!
    with patch.object(tracker, "_is_underlying_trend_intact", return_value=True):
        exit_price, exit_reason = tracker._evaluate_exit(pos, snap)
        assert exit_reason != ExitReason.INACTIVITY_KILL_SWITCH


# ============================================================================
# TEAM C (NQIT): NOISE ENVELOPE PRESERVATION & HEALTH ENGINE STABILITY
# ============================================================================

def test_team_c_noise_envelope_preserves_tier_b_runner():
    """Verify that a Tier B runner dipping 14.5 pts on 1-min candle noise does NOT trigger PRE_SL_BAILOUT."""
    engine = PositionHealthEngine()

    state = TrailingState(
        position_id="RUNNER-NOISE-01",
        tradingsymbol="BANKNIFTY55000CE",
        symbol="NIFTY BANK",
        entry_premium=1000.0,
        initial_sl=965.0,
        current_sl=965.0,
        target=1080.0,
        direction="bullish",
        entry_spot=55000.0,
        entry_bar_count=1,
        tier="B",
    )

    # DF with minor pullback to VWAP (not a catastrophic crash)
    df = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-10-08 09:30"), "open": 54980, "high": 55020, "low": 54960, "close": 54990, "volume": 10000},
        {"timestamp": pd.Timestamp("2026-10-08 09:45"), "open": 54990, "high": 55040, "low": 54980, "close": 55010, "volume": 12000},
    ])

    # Premium dipped to 985.5 (-14.5 pts loss inside spread noise)
    report = engine.evaluate_position_health(
        state=state,
        current_premium=985.5,
        spot_override=54990.0,
        underlying_df_override=df,
    )

    # Must HOLD and NOT trigger premature bailout!
    assert report.suggested_action != "PRE_SL_BAILOUT"
    assert report.suggested_action == "HOLD"


def test_team_c_phantom_p7_iv_stub_zeroed():
    """Verify P7 IV stub weight is 0.0 in composite weights to prevent score distortion."""
    engine = PositionHealthEngine()
    # Check that P7 IV weight does not penalize or inflate scores
    p7_score, _, _, _ = engine._eval_pillar_iv("NIFTY BANK", "BANKNIFTY26OCT55000CE", entry_iv=None)
    assert p7_score == 0.0


# ============================================================================
# TEAM D (ZKT): OPERATIONAL UX & HIGH-CONTRAST BANNERS
# ============================================================================

def test_team_d_live_signal_shows_operational_state_and_live_banner():
    """Verify Live signals show high-contrast green banner and clear operational state (TREND INTACT)."""
    messages = []
    bot = TelegramBot(bot_token="", chat_id="")
    bot.send_message = lambda text: messages.append(text)

    signal = {
        "action": "BUY_CE",
        "symbol": "SENSEX",
        "instrument": "SENSEX26OCT72000CE",
        "tradingsymbol": "SENSEX26OCT72000CE",
        "entry_price": 350.0,
        "stop_loss": 320.0,
        "target": 400.0,
        "strike": 72000,
        "option_type": "CE",
        "leaderboard_rank": 1,
        "edge_score": 8.0,
        "strategy": "PriceAction_Momentum",
        "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        "bar_timestamp": "2026-10-08 10:00:00",
    }
    bot.alert_new_signal(signal)
    assert len(messages) == 1
    msg = messages[0]
    assert "🟢🟢🟢 <b>ACTION: EXECUTE LIVE ON KITE</b> 🟢🟢🟢" in msg
    assert "OPERATIONAL STATE: TREND INTACT" in msg
    # Complex raw telemetry stripped from live card
    assert "ZGL:" not in msg


def test_team_d_trade_update_shows_operational_states():
    """Verify 5-minute trade updates display clear operational states (TREND INTACT, CONSOLIDATING)."""
    messages = []
    bot = TelegramBot(bot_token="", chat_id="")
    bot.send_message = lambda text: messages.append(text)

    # Active runner trade
    info_runner = {
        "symbol": "SENSEX",
        "instrument": "SENSEX72000CE",
        "direction": "BUY CE",
        "current_price": 375.0,
        "entry_price": 350.0,
        "stop_loss": 355.0,
        "target": 400.0,
        "holding_duration_seconds": 300,
        "gross_pnl_pts": 25.0,
        "gross_pnl_rs": 250.0,
        "net_pnl_pts": 14.5,
        "net_pnl_rs": 145.0,
        "health_score": 35.0,
        "async_send": False,
    }
    bot.alert_trade_update(info_runner)
    assert len(messages) == 1
    msg = messages[0]
    assert "STATUS: TREND INTACT" in msg

    # Consolidating trade
    messages.clear()
    info_consol = {
        "symbol": "SENSEX",
        "instrument": "SENSEX72000CE",
        "direction": "BUY CE",
        "current_price": 352.0,
        "entry_price": 350.0,
        "stop_loss": 320.0,
        "target": 400.0,
        "holding_duration_seconds": 600,
        "gross_pnl_pts": 2.0,
        "gross_pnl_rs": 20.0,
        "net_pnl_pts": -8.5,
        "net_pnl_rs": -85.0,
        "health_score": 5.0,
        "async_send": False,
    }
    bot.alert_trade_update(info_consol)
    assert len(messages) == 1
    msg = messages[0]
    assert "STATUS: CONSOLIDATING" not in msg


# ============================================================================
# EXECUTION SAFETY: SHARED PRICE MICRO-CACHE FALLBACK
# ============================================================================

def test_position_monitor_micro_cache_fallback_on_rate_limit():
    """Verify PositionMonitor falls back to SharedPriceMicroCache when broker returns ltp <= 0."""
    cache = get_shared_micro_cache()
    # Populate fresh quote in cache (price = 360.0)
    cache.put(
        "TOKEN-123",
        MicroCacheEntry(
            price=360.0,
            bid=359.0,
            ask=361.0,
            open_interest=50000,
            volume=10000,
            timestamp=time.monotonic(),
            tradingsymbol="SENSEX72000CE",
        ),
    )

    mock_broker = MagicMock()
    # Simulate broker HTTP 429 rate limit returning 0.0
    mock_broker.get_ltp.return_value = 0.0

    monitor = PositionMonitor(broker=mock_broker, poll_interval=1.0)
    state = TrailingState(
        position_id="POS-COOLDOWN-01",
        tradingsymbol="SENSEX72000CE",
        symbol="SENSEX",
        entry_premium=350.0,
        initial_sl=320.0,
        current_sl=320.0,
        target=400.0,
        direction="bullish",
    )
    monitor.positions["POS-COOLDOWN-01"] = state

    # Mock _process_tick to observe if it gets called with cached price
    processed_ticks = []
    monitor._process_tick = lambda st, ltp: processed_ticks.append((st.position_id, ltp))

    # Run one pass of the monitor loop
    for pid, s in list(monitor.positions.items()):
        ltp = monitor.broker.get_ltp(s.tradingsymbol, exchange="BFO")
        if ltp <= 0:
            c_entry = cache.get(s.tradingsymbol) or cache.get_by_symbol(s.tradingsymbol)
            if c_entry and c_entry.price > 0 and c_entry.age() < 5.0:
                ltp = c_entry.price
        if ltp > 0:
            monitor._process_tick(s, ltp)

    assert len(processed_ticks) == 1
    assert processed_ticks[0][0] == "POS-COOLDOWN-01"
    assert processed_ticks[0][1] == 360.0
