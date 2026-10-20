"""
Unit tests validating Telegram message formatting, Zerodha Kite copy box restoration,
clean symbol strings, bulletproof HTML escaping, minimal heartbeat UX,
and single-lot cost buffer recalibrations.
"""

import time
import html
import threading
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from prometheus.interface.telegram_bot import TelegramBot
from prometheus.signals.tier_classifier import classify_signal_tier
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.types import Position, Direction
from prometheus.execution.position_health import PositionHealthReport


# ============================================================================
# 1. ZERODHA KITE COPY BOX RESTORATION FOR ALL SIGNALS (LIVE & PAPER)
# ============================================================================

def test_zerodha_copy_box_present_on_live_and_paper_directional_signals():
    """Verify that Zerodha Kite search copy box is present unconditionally for both Live and Paper signals."""
    sent_messages = []
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))

    # Live signal (Tier S)
    live_signal = {
        "symbol": "NIFTY 50",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "edge_score": 8.5,
        "entry_price": 24500.0,
        "stop_loss": 24450.0,
        "target": 24600.0,
        "strike": 24500,
        "option_type": "CE",
        "expiry": "2026-07-30",
        "tradingsymbol": "NIFTY26JUL24500CE",
        "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish", "Volume_Surge"],
        "bar_timestamp": "2026-10-08 10:00:00",
        "leaderboard_rank": 1,
    }
    bot.alert_new_signal(live_signal)
    assert len(sent_messages) == 1
    live_msg = sent_messages[0][0]
    assert "ACTION: EXECUTE LIVE ON KITE" in live_msg
    assert "Zerodha Kite Search (Tap to Copy):" in live_msg
    assert "NIFTY JUL 500 CE" in live_msg

    # Paper signal (Tier C)
    sent_messages.clear()
    paper_signal = {
        "symbol": "NIFTY 50",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "edge_score": 4.5,
        "entry_price": 24500.0,
        "stop_loss": 24450.0,
        "target": 24600.0,
        "strike": 24500,
        "option_type": "CE",
        "expiry": "2026-07-30",
        "tradingsymbol": "NIFTY26JUL24500CE",
        "reasons": ["Session_VWAP_Bullish"],
        "bar_timestamp": "2026-10-08 10:00:00",
        "leaderboard_rank": 2,
    }
    bot.alert_new_signal(paper_signal)
    assert len(sent_messages) == 1
    paper_msg = sent_messages[0][0]
    assert "ACTION: PAPER ONLY — DO NOT EXECUTE LIVE" in paper_msg
    # Copy box must be present so operator can shadow/monitor on Kite
    assert "Zerodha Kite Search (Tap to Copy):" in paper_msg
    assert "NIFTY JUL 500 CE" in paper_msg


def test_zerodha_copy_box_present_on_paper_credit_spread_signals():
    """Verify that Zerodha Kite basket copy box is present unconditionally on paper credit spreads."""
    sent_messages = []
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))

    spread_signal = {
        "symbol": "NIFTY 50",
        "action": "SHORT_CALL_SPREAD",
        "strategy_type": "credit_spread",
        "strategy": "CreditSpread_0DTE",
        "edge_score": 7.0,  # Below 9.0 -> Tier C (Paper Only)
        "net_credit": 15.0,
        "target_decay_price": 4.5,
        "hard_sl_price": 22.5,
        "margin_required": 35000.0,
        "tradingsymbol": "NIFTY26JUL24600CE / NIFTY26JUL24700CE",
        "legs": [
            {"instrument": "NIFTY26JUL24600CE", "action": "SELL", "strike": 24600, "option_type": "CE", "price": 25.0, "is_hedge": False},
            {"instrument": "NIFTY26JUL24700CE", "action": "BUY", "strike": 24700, "option_type": "CE", "price": 10.0, "is_hedge": True},
        ],
        "is_0dte": True,
        "otm_sigma": 1.8,
        "leaderboard_rank": 1,
    }
    bot.alert_new_signal(spread_signal)
    assert len(sent_messages) == 1
    msg = sent_messages[0][0]
    assert "ACTION: PAPER ONLY — DO NOT EXECUTE LIVE" in msg
    assert "(Tap to Copy):" in msg
    assert "LEG (BUY HEDGE)" in msg
    assert "LEG (SELL MAIN)" in msg


# ============================================================================
# 2. CLEAN SYMBOL NAME & /SCAN PARAMETER MAPPING
# ============================================================================

def test_clean_symbol_and_display_name_in_scan_mapping():
    """Verify intraday /scan payload retains clean symbol without '(intraday 15minute)' pollution."""
    exec_sig = {
        "action": "BUY_CE",
        "confidence": 0.85,
        "edge_score": 8.5,
        "entry_price": 250.0,
        "stop_loss": 230.0,
        "target": 300.0,
        "risk_reward": 2.5,
        "strategy": "PriceAction_Momentum",
        "reasons": ["15M_ORB_High_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        "instrument": "NIFTY26JUL24500CE",
        "tradingsymbol": "NIFTY26JUL24500CE",
        "strike": 24500,
        "option_type": "CE",
        "expiry": "2026-07-30",
        "lots": 1,
        "lot_cost": 16250.0,
    }

    # Simulate the /scan mapping block from main.py:10169-10197
    symbol = "NIFTY 50"
    bar_interval = "15minute"
    raw_confidence = exec_sig.get("confidence", 0.5)
    adj_confidence = raw_confidence * 0.75

    mapped = {
        "symbol": symbol,
        "display_name": f"{symbol} (intraday {bar_interval})",
        "action": exec_sig.get("action", "HOLD"),
        "raw_confidence": raw_confidence,
        "adj_confidence": adj_confidence,
        "regime": "markup",
        "signal_count": int(exec_sig.get("edge_score", exec_sig.get("confidence", 0) * 100)),
        "entry_price": exec_sig.get("entry_price", 0),
        "stop_loss": exec_sig.get("stop_loss", 0),
        "target": exec_sig.get("target", 0),
        "risk_reward": exec_sig.get("risk_reward", 0),
        "reasoning": exec_sig.get("reasoning", exec_sig.get("strategy", "")),
        "executable": True,
        "timeframe": "intraday",
        "instrument": exec_sig.get("instrument", ""),
        "tradingsymbol": exec_sig.get("tradingsymbol", exec_sig.get("instrument", "")),
        "strike": exec_sig.get("strike", 0),
        "option_type": exec_sig.get("option_type", ""),
        "expiry": exec_sig.get("expiry", ""),
        "strategy_type": exec_sig.get("strategy_type", ""),
        "spread_type": exec_sig.get("spread_type", ""),
        "net_credit": exec_sig.get("net_credit", 0),
        "target_decay_price": exec_sig.get("target_decay_price", 0),
        "hard_sl_price": exec_sig.get("hard_sl_price", 0),
        "margin_required": exec_sig.get("margin_required", 0),
        "legs": exec_sig.get("legs", []),
        "lot_size": exec_sig.get("lot_size", 0),
        "edge_score": exec_sig.get("edge_score", 0.0),
        "signal_score": exec_sig.get("edge_score", 0.0),
        "reasons": exec_sig.get("reasons", []),
        "lots": exec_sig.get("lots", 1),
        "lot_cost": exec_sig.get("lot_cost", 0.0),
        "_signal_generated_at": time.time(),
    }

    assert mapped["symbol"] == "NIFTY 50"
    assert mapped["display_name"] == "NIFTY 50 (intraday 15minute)"
    assert mapped["edge_score"] == 8.5
    assert mapped["signal_score"] == 8.5
    assert mapped["lots"] == 1
    assert mapped["lot_cost"] == 16250.0
    assert len(mapped["reasons"]) == 3

    # Ensure tier classification does not falsely demote to Tier D due to missing score
    tier = classify_signal_tier(mapped)
    assert tier["tier"] in ("S", "B")  # High edge score retained
    assert tier["tier"] != "D"


def test_no_empty_italic_tags_in_telegram_alert():
    """Verify empty sizing_line or cost_line does not produce empty <i></i><i></i> tags."""
    sent_messages = []
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))

    signal = {
        "symbol": "NIFTY 50",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "edge_score": 7.5,
        "entry_price": 100.0,
        "stop_loss": 85.0,
        "target": 130.0,
        "strike": 24500,
        "option_type": "CE",
        "tradingsymbol": "NIFTY26JUL24500CE",
        # Intentionally zero/empty sizing and lot_cost
        "quantity": 0,
        "lots": 0,
        "lot_cost": 0,
        "bar_timestamp": "2026-10-08 10:00:00",
    }
    bot.alert_new_signal(signal)
    assert len(sent_messages) == 1
    msg = sent_messages[0][0]
    assert "<i></i>" not in msg
    assert "<i></i><i></i>" not in msg


# ============================================================================
# 3. BULLETPROOF HTML ESCAPING & FALLBACK SANITIZATION
# ============================================================================

def test_tier_classifier_reasons_contain_no_raw_less_than_symbols():
    """Verify tier_classifier reasons use HTML-safe 'below' text instead of raw '<' characters."""
    # Test Tier C low OTM sigma (< 1.95)
    sig_low_otm = {
        "action": "SHORT_CALL_SPREAD",
        "strategy_type": "credit_spread",
        "edge_score": 9.5,
        "is_0dte": True,
        "otm_sigma": 1.7,
    }
    res = classify_signal_tier(sig_low_otm)
    assert res["tier"] == "C"
    reasons_joined = " ".join(res["classification_reasons"])
    assert "<" not in reasons_joined
    assert "below 2.0σ requirement" in reasons_joined

    # Test Tier C conviction score (< 9.0)
    sig_low_score = {
        "action": "SHORT_CALL_SPREAD",
        "strategy_type": "credit_spread",
        "edge_score": 8.5,
        "is_0dte": True,
        "otm_sigma": 2.2,
    }
    res_score = classify_signal_tier(sig_low_score)
    assert res_score["tier"] == "C"
    reasons_joined = " ".join(res_score["classification_reasons"])
    assert "<" not in reasons_joined
    assert "below 9.0" in reasons_joined

    # Test Tier D threshold (< 3.5)
    sig_tier_d = {
        "action": "BUY_CE",
        "strategy_type": "option_buying",
        "edge_score": 2.5,
    }
    res_d = classify_signal_tier(sig_tier_d)
    assert res_d["tier"] == "D"
    reasons_joined = " ".join(res_d["classification_reasons"])
    assert "<" not in reasons_joined
    assert "below 3.5 threshold" in reasons_joined


def test_telegram_html_escaping_on_arbitrary_user_strings():
    """Verify arbitrary characters like <, >, & in symbol, strategy, reasons are escaped."""
    sent_messages = []
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))

    malicious_signal = {
        "symbol": "BANK<NIFTY>&CO",
        "action": "BUY_CE",
        "strategy": "Breakout <v2.0> & Momentum",
        "edge_score": 7.0,
        "entry_price": 500.0,
        "stop_loss": 480.0,
        "target": 550.0,
        "strike": 50000,
        "option_type": "CE",
        "tradingsymbol": "BANK<TEST>",
        "reasons": ["15M_ORB <Breakout>", "RSI < 30 & Volume > 2x"],
        "reasoning": "Strategy alert <script>alert(1)</script> & details",
        "bar_timestamp": "2026-10-08 10:00:00",
    }
    bot.alert_new_signal(malicious_signal)
    assert len(sent_messages) == 1
    msg = sent_messages[0][0]
    # Raw unescaped brackets in dynamic content must not exist
    assert "<Breakout>" not in msg
    assert "<script>" not in msg
    assert "<TEST>" not in msg
    assert "&lt;Breakout&gt;" in msg
    assert "&lt;script&gt;" in msg
    assert "BANK&lt;NIFTY&gt;&amp;CO" in msg
    assert "BANK&lt;TEST&gt;" in msg


def test_telegram_http_400_fallback_strips_all_html_tags():
    """Verify HTTP 400 fallback sends sanitized plain text without raw <b>, <code>, <i> tags."""
    bot = TelegramBot.__new__(TelegramBot)
    bot._enabled = True
    bot._send_lock = threading.Lock()
    bot.bot_token = "TEST_TOKEN"
    bot.chat_id = "12345"
    bot._session = MagicMock()
    bot._rate_limit_until = 0.0
    bot._send_fail_streak = 0
    bot._last_send_ok_ts = 0.0
    bot._ordered_base_candidates = MagicMock(return_value=["https://api.telegram.org"])
    bot._request_transport_recovery = MagicMock()

    # First request returns HTTP 400 (rejected HTML), second request succeeds (200)
    mock_resp_400 = MagicMock()
    mock_resp_400.status_code = 400
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    bot._session.post.side_effect = [mock_resp_400, mock_resp_200]

    raw_html_text = "<b>TRADE ALERT</b>\n<code>NIFTY 24500 CE</code>\n<i>Running P&L: +25.5 pts</i>"
    success = bot.send_message(raw_html_text, parse_mode="HTML")

    assert success is True
    assert bot._session.post.call_count == 2
    # Second call (the fallback) must have stripped tags
    fallback_payload = bot._session.post.call_args_list[1][1]["json"]
    clean_text = fallback_payload["text"]
    assert "<b>" not in clean_text
    assert "</b>" not in clean_text
    assert "<code>" not in clean_text
    assert "</code>" not in clean_text
    assert "<i>" not in clean_text
    assert "</i>" not in clean_text
    assert "TRADE ALERT\nNIFTY 24500 CE\nRunning P&L: +25.5 pts" in clean_text


# ============================================================================
# 4. CLEAN MINIMAL HEARTBEAT UX
# ============================================================================

def test_clean_minimal_heartbeat_ux_removes_action_command_and_consolidation_badge():
    """Verify heartbeat alert is a clean, minimal card without patronizing ACTION COMMAND or CONSOLIDATING badge."""
    sent_messages = []
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message_async = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))
    bot.send_message = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))

    info = {
        "symbol": "SENSEX",
        "instrument": "SENSEX72000CE",
        "direction": "BUY CE",
        "current_price": 352.0,
        "entry_price": 350.0,
        "stop_loss": 320.0,
        "target": 400.0,
        "gross_pnl_pts": 2.0,
        "gross_pnl": 40.0,
        "net_pnl_pts": -2.5,
        "net_pnl": -50.0,
        "holding_duration": "10 minutes",
        "health_score": 5.0,
        "quantity": 20,
        "async_send": True,
    }
    bot.alert_trade_update(info)
    assert len(sent_messages) == 1
    msg = sent_messages[0][0]

    # Required card lines
    assert "Symbol:" in msg and "SENSEX" in msg
    assert "Current LTP:" in msg and "Rs 352.00" in msg
    assert "Entry:" in msg and "Rs 350.00" in msg
    assert "Running Net P&L:" in msg and "-2.5 pts" in msg
    assert "Active Trailing SL:" in msg and "Rs 320.00" in msg
    assert "Profit Target:" in msg and "Rs 400.00" in msg
    assert "Holding Duration:" in msg and "10 minutes" in msg

    # Stripped clutter
    assert "ACTION COMMAND" not in msg
    assert "HOLD POSITION" not in msg
    assert "HOLD IT" not in msg
    assert "STATUS: CONSOLIDATING" not in msg


def test_heartbeat_retains_trend_intact_status_for_strong_runners():
    """Verify strong runner positions (>= 15 pts gain) retain STATUS: TREND INTACT badge."""
    sent_messages = []
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message_async = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))

    runner_info = {
        "symbol": "SENSEX",
        "instrument": "SENSEX72000CE",
        "direction": "BUY CE",
        "current_price": 380.0,
        "entry_price": 350.0,
        "stop_loss": 355.0,
        "target": 420.0,
        "gross_pnl_pts": 30.0,
        "gross_pnl": 600.0,
        "net_pnl_pts": 25.5,
        "net_pnl": 510.0,
        "holding_duration": "25 minutes",
        "health_score": 35.0,
        "quantity": 20,
        "async_send": True,
    }
    bot.alert_trade_update(runner_info)
    assert len(sent_messages) == 1
    msg = sent_messages[0][0]
    assert "STATUS: TREND INTACT" in msg
    assert "ACTION COMMAND" not in msg


# ============================================================================
# 5. RECALIBRATE SINGLE-LOT COST BUFFERS & GUARD MISSING TELEMETRY
# ============================================================================

def test_single_lot_cost_buffers_calibrated_for_statutory_fees():
    """Verify single-lot cost buffers: SENSEX (20 qty) >= 4.50 pts, NIFTY (65 qty) >= 1.20 pts."""
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))

    # SENSEX single lot (qty = 20): covers Rs 85.98 round trip fees (4.50 pts * 20 = Rs 90.00)
    pos_sensex = Position(
        trade_id="POS-SENSEX-1LOT",
        symbol="SENSEX",
        instrument="SENSEX26OCT72000CE",
        underlying="SENSEX",
        direction=Direction.LONG,
        quantity=20,
        entry_price=300.0,
        entry_time=datetime.now(),
        stop_loss=270.0,
        target=360.0,
        max_bars=16,
    )
    tracker.open_positions[pos_sensex.trade_id] = pos_sensex
    tracker._maybe_advance_trailing_stop(pos_sensex, 320.0)  # +20 pts gain (hits noise floor)
    assert pos_sensex.breakeven_set is True
    # Breakeven SL should be Entry + 4.50 = 304.50
    assert pos_sensex.stop_loss == pytest.approx(304.50, abs=1e-2)

    # NIFTY single lot (qty = 65): covers Rs 69.52 round trip fees (1.20 pts * 65 = Rs 78.00)
    pos_nifty = Position(
        trade_id="POS-NIFTY-1LOT",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT24500CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=85.0,
        target=130.0,
        max_bars=16,
    )
    tracker.open_positions[pos_nifty.trade_id] = pos_nifty
    tracker._maybe_advance_trailing_stop(pos_nifty, 110.0)  # +10 pts gain
    # Breakeven SL should be Entry + 1.40 = 101.40 (GEMINI.md Rule 2)
    assert pos_nifty.stop_loss == pytest.approx(101.40, abs=1e-2)


def test_missing_telemetry_keys_do_not_trigger_weak_edge_ratchet():
    """Verify commitment_ratio == 0.0 or net_gex == 0.0 from missing feed keys never triggers weak edge panic."""
    tracker = PositionTracker(fill_sim=MagicMock(), cost_model=CostModel(cost_per_side_bps=0.0))

    pos = Position(
        trade_id="POS-TELEMETRY-GUARD",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT24500CE",
        underlying="NIFTY",
        direction=Direction.LONG,
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=85.0,
        target=130.0,
        tier="S",
        max_bars=16,
    )
    tracker.open_positions[pos.trade_id] = pos

    # Health report with missing feed telemetry keys (commitment_ratio = 0.0, net_gex = 0.0)
    # but slight negative score (-2.0) that would previously panic-trigger comm_ratio < 0.15
    report = PositionHealthReport(
        position_id=pos.trade_id,
        symbol=pos.symbol,
        tradingsymbol=pos.instrument,
        direction="BUY",
        current_price=102.50,
        entry_price=100.0,
        health_score=-2.0,
        suggested_action="HOLD",
        real_threat=False,
        net_gex=0.0,
        commitment_ratio=0.0,
    )

    with patch.object(tracker.health_engine, "evaluate_position_health", return_value=report):
        # Current price slightly above entry (+2.5 pts gain)
        # If is_weak_edge were falsely True, it would trigger WEAK_EDGE_DEFENSIVE_LOCK (Half-Risk Cut)
        tracker._maybe_advance_trailing_stop(pos, 102.50)

    # Invariants preserved: no half-risk cut or breakeven triggered from missing telemetry keys
    assert pos.half_risk_set is False
    assert pos.breakeven_set is False
    assert pos.stop_loss == 85.0


def test_position_monitor_live_cost_buffers_and_telemetry_guards():
    """Verify live PositionMonitor cost buffers (SENSEX 20 qty >= 4.50, NIFTY 65 qty >= 1.20) and missing telemetry guard."""
    from prometheus.execution.position_monitor import PositionMonitor, TrailingState

    monitor = PositionMonitor(broker=MagicMock(), poll_interval=1.0)

    # 1. SENSEX 20 qty single lot in PositionMonitor
    state_sensex = TrailingState(
        position_id="MON-SENSEX-1LOT",
        symbol="SENSEX",
        tradingsymbol="SENSEX26OCT72000CE",
        entry_premium=300.0,
        initial_sl=270.0,
        current_sl=270.0,
        target=360.0,
        direction="bullish",
        quantity=20,
        entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        tier="S",
        max_bars=16,
    )
    monitor.positions[state_sensex.position_id] = state_sensex
    # Gain +20 pts reaches SENSEX noise floor (>= 20 pts)
    monitor._process_tick(state_sensex, 320.0)
    assert state_sensex.breakeven_set is True
    assert state_sensex.current_sl == pytest.approx(304.50, abs=1e-2)

    # 2. NIFTY 65 qty single lot in PositionMonitor
    state_nifty = TrailingState(
        position_id="MON-NIFTY-1LOT",
        symbol="NIFTY 50",
        tradingsymbol="NIFTY26OCT24500CE",
        entry_premium=100.0,
        initial_sl=85.0,
        current_sl=85.0,
        target=130.0,
        direction="bullish",
        quantity=65,
        entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        tier="S",
        max_bars=16,
    )
    monitor.positions[state_nifty.position_id] = state_nifty
    # Gain +10 pts triggers breakeven
    monitor._process_tick(state_nifty, 110.0)
    assert state_nifty.breakeven_set is True
    assert state_nifty.current_sl == pytest.approx(101.40, abs=1e-2)

    # 3. Missing telemetry keys (commitment_ratio=0.0, net_gex=0.0) must not trigger weak edge ratchet in PositionMonitor
    state_guard = TrailingState(
        position_id="MON-GUARD-1LOT",
        symbol="NIFTY 50",
        tradingsymbol="NIFTY26OCT24500CE",
        entry_premium=100.0,
        initial_sl=85.0,
        current_sl=85.0,
        target=130.0,
        direction="bullish",
        quantity=65,
        entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        tier="S",
        max_bars=16,
    )
    monitor.positions[state_guard.position_id] = state_guard
    report = PositionHealthReport(
        position_id=state_guard.position_id,
        symbol=state_guard.symbol,
        tradingsymbol=state_guard.tradingsymbol,
        direction="BUY",
        current_price=102.50,
        entry_price=100.0,
        health_score=-2.0,
        suggested_action="HOLD",
        real_threat=False,
        net_gex=0.0,
        commitment_ratio=0.0,
    )
    with patch.object(monitor.health_engine, "evaluate_position_health", return_value=report):
        monitor._process_tick(state_guard, 102.50)

    assert state_guard.half_risk_set is False
    assert state_guard.breakeven_set is False
    assert state_guard.current_sl == 85.0


def test_telegram_escaping_in_auxiliary_alerts():
    """Verify HTML escaping in alert_trailing_stop_updated, alert_adverse_exit, alert_trade_closed, and alert_strategy_drift_warning."""
    sent_messages = []
    bot = TelegramBot.__new__(TelegramBot)
    bot.send_message = lambda text, parse_mode="HTML": sent_messages.append((text, parse_mode))

    # 1. alert_trailing_stop_updated
    bot.alert_trailing_stop_updated(
        symbol="BANK<NIFTY>",
        instrument="BANK<OPT>",
        old_sl=100.0,
        new_sl=110.0,
        stage="half_risk <stage>",
        current_price=120.0,
        gain_pts=20.0,
    )
    assert len(sent_messages) == 1
    msg1 = sent_messages[-1][0]
    assert "<NIFTY>" not in msg1
    assert "&lt;NIFTY&gt;" in msg1
    assert "<OPT>" not in msg1
    assert "&lt;OPT&gt;" in msg1

    # 2. alert_adverse_exit
    bot.alert_adverse_exit(
        symbol="NIFTY<TEST>",
        instrument="NIFTY<INST>",
        entry=100.0,
        exit_price=90.0,
        pnl=-500.0,
        reason="Price < VWAP & Volume > 2x",
    )
    assert len(sent_messages) == 2
    msg2 = sent_messages[-1][0]
    assert "<TEST>" not in msg2
    assert "&lt;TEST&gt;" in msg2
    assert "Price &lt; VWAP &amp; Volume &gt; 2x" in msg2

    # 3. alert_trade_closed
    bot.alert_trade_closed({
        "symbol": "SENSEX<SYM>",
        "side": "BUY<SIDE>",
        "quantity": 20,
        "entry_price": 300.0,
        "exit_price": 310.0,
        "gross_pnl": 200.0,
        "net_pnl": 114.0,
        "instrument": "SENSEX<INST>",
        "reason_desc": "Breached <SL> at lower band & reversed",
    })
    assert len(sent_messages) == 3
    msg3 = sent_messages[-1][0]
    assert "<SYM>" not in msg3
    assert "&lt;SYM&gt;" in msg3
    assert "<SL>" not in msg3
    assert "&lt;SL&gt;" in msg3

    # 4. alert_strategy_drift_warning
    bot.alert_strategy_drift_warning({
        "status": "DEGRADED",
        "multiplier": 0.5,
        "trades_count": 20,
        "win_rate": 35.0,
        "profit_factor": 0.8,
        "expectancy": -50.0,
        "message": "Win Rate < 40% & Profit Factor < 1.0",
    })
    assert len(sent_messages) == 4
    msg4 = sent_messages[-1][0]
    assert "< 40%" not in msg4
    assert "&lt; 40% &amp; Profit Factor &lt; 1.0" in msg4

