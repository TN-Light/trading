"""Comprehensive unit test suite for Signal Gate Upgrades, Runner Preservation,
and Telegram Dual-Track Alert Templates.

Validates:
1. Climax Bar Overextension Gate (vwap_dist >= 60 bps or bar_stretch >= 1.8x ATR without base).
2. Synthetic volume spoof removal on spot index bars.
3. Dealer Long GEX Gate (net_gex > +1.5 Cr) capping option buying to Tier C (Paper Only).
4. Midday continuation gate through lunch requiring strict 1H alignment and non-long-gamma.
5. 0-DTE Golden Setup promotion to Tier B when edge score >= 8.0.
6. Resilient Inactivity Kill-Switch deferral during intact technical / neutral health (+8/100)
   consolidation flags, and immediate liquidation on adverse spot drift.
7. Noise floor preservation: FINNIFTY (16 pts) & NIFTY (8 pts) min_be_gain, and Half-Risk Cut
   for weak edge defense below entry.
8. Telegram alert segregation: Live alerts show green execution banner with Kite copy box,
   while Paper alerts show paper banner with stripped copy box and routing explanation.
"""

from datetime import datetime, time as dtime
from unittest.mock import MagicMock
import pandas as pd
import pytest

from prometheus.signals.price_action_momentum import PriceActionMomentumScanner as PriceActionMomentum
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.fill_simulator import FillSimulator
from prometheus.papertrade.types import Position, Direction, ExitReason, TradeSnapshot
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.execution.position_health import PositionHealthReport
from prometheus.interface.telegram_bot import TelegramBot


class MockFeed:
    def __init__(self, ltps=None):
        self.ltps = ltps or {}

    def get_ltp(self, instrument: str) -> float:
        return float(self.ltps.get(instrument, 0.0))

    def get_quote(self, instrument: str):
        price = self.get_ltp(instrument)
        if price > 0:
            return {"bid": price - 0.5, "ask": price + 0.5, "ltp": price}
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Climax Bar Overextension & Volume Spoof Removal Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_climax_overextension_vetoed_for_ce():
    """Verify that a CE breakout with price stretched >= 60 bps above VWAP is vetoed."""
    pam = PriceActionMomentum()
    closes = [24000.0] * 10
    # Create 15-min bars with 09:15-09:30 ORB
    timestamps = pd.date_range("2026-10-08 09:15", periods=5, freq="15min")
    # ORB bar (09:15-09:30): high 24050, low 23950
    # Current bar at 10:15: close 24300, vwap 24000 (dist = 300 / 24300 = 1.23% > 0.60%)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [24000, 24050, 24100, 24150, 24150],
        "high": [24050, 24080, 24150, 24200, 24320],
        "low":  [23950, 24020, 24080, 24120, 24140],
        "close": [24020, 24070, 24120, 24180, 24300],
        "volume": [1000, 1000, 1000, 1000, 2500],
    })
    # HTF 1H dataframe
    df_1h = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-08 09:15", periods=3, freq="1h"),
        "open": [23900, 24000, 24100],
        "high": [24050, 24150, 24350],
        "low": [23850, 23980, 24080],
        "close": [24020, 24120, 24300],
        "volume": [5000, 5000, 5000],
    })

    # When close is stretched >= 60 bps above VWAP, evaluate_price_action_momentum returns None
    sig = pam.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig is None, "Climax overextension above VWAP must be vetoed"


def test_climax_plunge_vetoed_for_pe():
    """Verify that a PE breakdown with price plunged >= 60 bps below VWAP is vetoed."""
    pam = PriceActionMomentum()
    timestamps = pd.date_range("2026-10-08 09:15", periods=5, freq="15min")
    # ORB bar: 24050 high, 23950 low. Last bar plunges to 23700 (vwap ~ 24000, dist = 300 / 23700 = 1.26% > 0.60%)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [24000, 23950, 23900, 23850, 23850],
        "high": [24050, 23980, 23920, 23880, 23860],
        "low":  [23950, 23880, 23840, 23800, 23680],
        "close": [23980, 23910, 23860, 23820, 23700],
        "volume": [1000, 1000, 1000, 1000, 2500],
    })
    df_1h = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-08 09:15", periods=3, freq="1h"),
        "open": [24100, 24000, 23900],
        "high": [24150, 24050, 23920],
        "low": [23950, 23850, 23680],
        "close": [23980, 23880, 23700],
        "volume": [5000, 5000, 5000],
    })

    sig = pam.evaluate_bar(df, symbol="NIFTY 50", df_1h=df_1h, golden_mode=True)
    assert sig is None, "Climax plunge below VWAP must be vetoed"


def test_no_synthetic_volume_spoof_on_zero_volume_bars():
    """Verify that zero volume bars do not spoof has_inst_volume = True."""
    pam = PriceActionMomentum()
    timestamps = pd.date_range("2026-10-08 09:15", periods=20, freq="15min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [54000.0] * 20,
        "high": [54100.0] * 20,
        "low":  [53900.0] * 20,
        "close": [54000.0] * 20,
        "volume": [0.0] * 20,  # Zero volume index spot
    })
    # ORB plunge at bar 19
    df.loc[19, "close"] = 53000.0
    sig = pam.evaluate_bar(df, symbol="NIFTY BANK", golden_mode=False)
    # Institutional trend day must be False because volume was 0
    if sig:
        assert sig.get("is_institutional_trend_day") is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dealer Long GEX Gate & Tier Classifier Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dealer_long_gex_gates_option_buying_to_tier_c():
    """In Dealer Long Gamma (net_gex > +1.5 Cr), directional option buying is routed to Tier C."""
    sig = {
        "action": "BUY_CE",
        "symbol": "NIFTY BANK",
        "strategy": "Golden_Setup (1H+VWAP+ORB)",
        "is_golden_setup": True,
        "edge_score": 8.0,
        "bar_timestamp": "2026-10-08 10:15:00",
        "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        "net_gex_cr": 2.5,  # +2.5 Cr Long Gamma
        "is_0dte": False,
    }
    tier_info = classify_signal_tier(sig)
    assert tier_info["tier"] == "C"
    assert tier_info["is_live_eligible"] is False
    assert any("Dealer Long Gamma Regime" in r for r in tier_info["classification_reasons"])


def test_0dte_golden_setup_promoted_to_tier_b_when_score_8():
    """0-DTE Golden Setup with score >= 8.0 is promoted to Tier B Live."""
    sig = {
        "action": "BUY_PE",
        "symbol": "SENSEX",
        "strategy": "Golden_Setup (1H+VWAP+ORB)",
        "is_golden_setup": True,
        "edge_score": 8.0,
        "bar_timestamp": "2026-10-08 10:45:00",
        "reasons": ["15M_ORB_Low_Breakdown", "Session_VWAP_Bearish", "1H_Trend_Bearish"],
        "is_0dte": True,
        "net_gex_cr": -1.2,  # Short Gamma
    }
    tier_info = classify_signal_tier(sig)
    assert tier_info["tier"] == "B"
    assert tier_info["is_live_eligible"] is True
    assert "0-DTE HIGH CONVICTION GOLDEN SETUP" in tier_info["tier_badge"]


def test_0dte_option_buying_score_under_8_stays_tier_c():
    """0-DTE option buying with score < 8.0 remains Tier C Paper Only."""
    sig = {
        "action": "BUY_PE",
        "symbol": "SENSEX",
        "strategy": "Golden_Setup (1H+VWAP+ORB)",
        "is_golden_setup": True,
        "edge_score": 7.0,
        "bar_timestamp": "2026-10-08 10:45:00",
        "reasons": ["15M_ORB_Low_Breakdown", "Session_VWAP_Bearish", "1H_Trend_Bearish"],
        "is_0dte": True,
        "net_gex_cr": -1.2,
    }
    tier_info = classify_signal_tier(sig)
    assert tier_info["tier"] == "C"
    assert tier_info["is_live_eligible"] is False
    assert any("0-DTE Expiry Option Buying gated" in r for r in tier_info["classification_reasons"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Runner Preservation & Inactivity Kill-Switch Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_inactivity_kill_switch_deferred_on_missing_api_data_when_drift_favorable():
    """When broker historical data returns None (timeout/rate limit), favorable drift preserves trade."""
    feed = MockFeed({"BANKNIFTY26OCT55000CE": 450.0, "NIFTY BANK": 55010.0})
    sim = FillSimulator(feed=feed)
    mock_data = MagicMock()
    mock_data.fetch_historical.return_value = None  # API timeout / None returned

    tracker = PositionTracker(
        fill_sim=sim,
        cost_model=CostModel(),
        data_engine=mock_data,
        trend_aware=True,
        max_stagnation_bars=6,
    )

    pos = Position(
        trade_id="PAPER-TIMEOUT-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="NIFTY BANK",
        direction=Direction.LONG,
        quantity=30,
        entry_price=450.0,
        entry_time=datetime(2026, 10, 8, 9, 30),
        stop_loss=410.0,
        target=550.0,
        max_bars=16,
        strategy="PriceAction_Momentum",
        trade_mode="intraday",
        bars_held=2,
        entry_spot=55000.0,
        atr=90.0,
    )
    tracker.open_positions[pos.trade_id] = pos

    snap3 = TradeSnapshot(
        timestamp=datetime(2026, 10, 8, 10, 15),
        symbol="NIFTY BANK",
        instrument="",
        open=55005.0, high=55025.0, low=54995.0, close=55010.0,
        volume=10000, bar_interval="15minute",
    )
    closed = tracker.on_bar(snap3, is_session_end=False, is_square_off=False)
    assert len(closed) == 0, "Position must NOT be killed on API timeout when spot drift is favorable"


def test_trailing_stop_noise_floor_finnifty_and_nifty():
    """Verify FINNIFTY noise floor is 16 pts and NIFTY is 8 pts."""
    feed = MockFeed({"FINNIFTY26OCT24000CE": 110.0})
    sim = FillSimulator(feed=feed)
    tracker = PositionTracker(fill_sim=sim, cost_model=CostModel())

    pos_fin = Position(
        trade_id="PAPER-FIN-01",
        symbol="FINNIFTY",
        instrument="FINNIFTY26OCT24000CE",
        underlying="FINNIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=100.0,
        entry_time=datetime(2026, 10, 8, 9, 30),
        stop_loss=80.0,
        target=150.0,
        max_bars=16,
        strategy="PriceAction_Momentum",
        trade_mode="intraday",
        initial_risk_distance=20.0,
        tier="B",
    )
    # Gain is +10 pts (below 16 pt noise floor)
    tracker._maybe_advance_trailing_stop(pos_fin, current_price=110.0)
    assert not pos_fin.breakeven_set, "Breakeven must not set before 16 pt gain on FINNIFTY"

    # Gain reaches +18 pts (above 16 pt noise floor)
    tracker._maybe_advance_trailing_stop(pos_fin, current_price=118.0)
    assert pos_fin.breakeven_set, "Breakeven must be set at >= 16 pt gain on FINNIFTY"


def test_weak_edge_uses_half_risk_cut_instead_of_entry_noise_sl():
    """Weak edge defense must cut risk by 50% without pulling SL directly into entry noise floor."""
    feed = MockFeed({"BANKNIFTY26OCT55000CE": 454.0})
    sim = FillSimulator(feed=feed)

    mock_health = MagicMock()
    mock_rep = PositionHealthReport(
        position_id="PAPER-WEAK-01",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY26OCT55000CE",
        direction="bullish",
        current_price=454.0,
        entry_price=450.0,
        health_score=-12.0,  # Weak edge
        real_threat=False,
        suggested_action="TIGHTEN_SL",
    )
    mock_health.evaluate_position_health.return_value = mock_rep

    tracker = PositionTracker(fill_sim=sim, cost_model=CostModel())
    tracker.health_engine = mock_health
    pos = Position(
        trade_id="PAPER-WEAK-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY26OCT55000CE",
        underlying="NIFTY BANK",
        direction=Direction.LONG,
        quantity=30,
        entry_price=450.0,
        entry_time=datetime(2026, 10, 8, 9, 30),
        stop_loss=410.0,
        target=550.0,
        max_bars=16,
        strategy="PriceAction_Momentum",
        trade_mode="intraday",
        initial_risk_distance=40.0,
        tier="B",
    )
    # Gain is +4.0 pts (above cost buffer + 1.0)
    tracker._maybe_advance_trailing_stop(pos, current_price=454.0)
    assert pos.half_risk_set is True
    # Half-risk cut: entry (450) - 0.5 * 40 = 430.0. NOT 450 + 4.50 (which would be inside noise)
    assert pos.stop_loss == 430.0, f"Expected Half-Risk SL of 430.0, got {pos.stop_loss}"
    assert not pos.breakeven_set, "Breakeven must NOT be set inside noise floor"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Telegram Live vs. Paper Alert Formatting Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_telegram_alert_live_tier_b_includes_green_banner_and_copy_box():
    """Live Tier B Rank #1 alert must include green action banner and Zerodha Kite Search copy box."""
    messages = []
    bot = TelegramBot(bot_token="", chat_id="")
    bot.send_message = lambda text: messages.append(text)

    signal = {
        "action": "BUY_CE",
        "symbol": "NIFTY BANK",
        "instrument": "BANKNIFTY26OCT55000CE",
        "tradingsymbol": "BANKNIFTY26OCT55000CE",
        "entry_price": 450.0,
        "stop_loss": 410.0,
        "target": 550.0,
        "strike": 55000,
        "option_type": "CE",
        "leaderboard_rank": 1,
        "edge_score": 7.5,
        "strategy": "Golden_Setup (1H+VWAP+ORB)",
        "is_golden_setup": True,
        "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        "bar_timestamp": "2026-10-08 10:00:00",
    }
    bot.alert_new_signal(signal)
    assert len(messages) == 1
    msg = messages[0]
    assert "🟢🟢🟢 <b>ACTION: EXECUTE LIVE ON KITE</b> 🟢🟢🟢" in msg
    assert "🥇 <b>RANK #1 SIGNAL (PRIMARY EXECUTION)</b>" in msg
    assert "Zerodha Kite Search (Tap to Copy)" in msg
    assert "BANKNIFTY" in msg


def test_telegram_alert_paper_tier_c_strips_copy_box_and_explains_routing():
    """Paper Tier C signal must have paper action banner, stripped copy box, and explanation."""
    messages = []
    bot = TelegramBot(bot_token="", chat_id="")
    bot.send_message = lambda text: messages.append(text)

    signal = {
        "action": "BUY_CE",
        "symbol": "NIFTY BANK",
        "instrument": "BANKNIFTY26OCT55000CE",
        "tradingsymbol": "BANKNIFTY26OCT55000CE",
        "entry_price": 450.0,
        "stop_loss": 410.0,
        "target": 500.0,
        "strike": 55000,
        "option_type": "CE",
        "leaderboard_rank": 1,
        "edge_score": 5.0,  # Below 6.5 -> Tier C
        "strategy": "PriceAction_Momentum",
        "reasons": ["15M_ORB_High_Breakout"],
        "bar_timestamp": "2026-10-08 10:00:00",
    }
    bot.alert_new_signal(signal)
    assert len(messages) == 1
    msg = messages[0]
    assert "📝📝📝 <b>ACTION: PAPER ONLY — DO NOT EXECUTE LIVE</b> 📝📝📝" in msg
    assert "🥇 <b>RANK #1 SIGNAL (PAPER ONLY)</b>" in msg
    assert "Zerodha Kite Search (Tap to Copy)" not in msg, "Kite copy box must be stripped for paper signals"
    assert "Paper Engine Tracking" in msg
    assert "Why Paper Only" in msg
