"""Tests for Telegram Exit Alert Overhaul, Phase 2 (VPR60 + DTE Strike), and Phase 3 (Gamma Engine)."""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from prometheus.signals.gamma_engine import GammaEngine, calculate_black_scholes_gamma
from prometheus.interface.telegram_bot import TelegramBot
from prometheus.papertrade.types import ExitReason


def test_black_scholes_gamma():
    """Verify Black-Scholes gamma calculation matches theoretical properties."""
    # ATM gamma should be highest
    g_atm = calculate_black_scholes_gamma(spot=25000, strike=25000, dte=2, sigma=0.15)
    g_otm = calculate_black_scholes_gamma(spot=25000, strike=25500, dte=2, sigma=0.15)
    assert g_atm > 0
    assert g_otm > 0
    assert g_atm > g_otm


def test_gamma_engine_gex_and_zgl():
    """Verify GammaEngine computes net GEX, ZGL, and gamma regime."""
    engine = GammaEngine()
    mock_chain = pd.DataFrame([
        {'strike_price': 24800, 'option_type': 'CE', 'open_interest': 50000, 'iv': 0.15},
        {'strike_price': 25000, 'option_type': 'CE', 'open_interest': 200000, 'iv': 0.14},
        {'strike_price': 25200, 'option_type': 'CE', 'open_interest': 100000, 'iv': 0.15},
        {'strike_price': 24800, 'option_type': 'PE', 'open_interest': 150000, 'iv': 0.15},
        {'strike_price': 25000, 'option_type': 'PE', 'open_interest': 80000, 'iv': 0.14},
        {'strike_price': 25200, 'option_type': 'PE', 'open_interest': 30000, 'iv': 0.15},
    ])
    res = engine.calculate_gex(mock_chain, spot_price=25000.0, symbol="NIFTY 50", dte=1.0)
    assert "net_gex" in res
    assert "net_gex_cr" in res
    assert "call_gex_cr" in res
    assert "put_gex_cr" in res
    assert "zgl" in res
    assert res["gamma_regime"] in ("LONG_GAMMA", "SHORT_GAMMA")
    assert res["call_gex_cr"] > 0
    assert res["put_gex_cr"] < 0


def test_gamma_engine_handles_empty_chain():
    """Verify GammaEngine handles empty DataFrame without crashing."""
    engine = GammaEngine()
    res = engine.calculate_gex(pd.DataFrame(), spot_price=25000.0, symbol="NIFTY 50")
    assert res["net_gex"] == 0.0
    assert res["gamma_regime"] == "NEUTRAL"


def test_telegram_exit_alert_inactivity_kill_switch():
    """Verify Telegram exit alert formats 45-min inactivity kill-switch properly."""
    bot = TelegramBot(bot_token="test_token", chat_id="test_chat")
    bot.send_message = MagicMock(return_value=True)

    trade_info = {
        "trade_id": "PAPER-12345",
        "symbol": "NIFTY BANK",
        "instrument": "BANKNIFTY29SEP2656200CE",
        "side": "BUY CE",
        "quantity": 30,
        "entry_price": 763.11,
        "exit_price": 768.80,
        "return_pct": 0.20,
        "gross_pnl": 170.63,
        "net_pnl": 45.95,
        "costs": {"total": 124.68},
        "holding_duration_seconds": 2700,
        "exit_reason": "inactivity_kill_switch",
    }

    bot.alert_trade_closed(trade_info)

    assert bot.send_message.called
    msg = bot.send_message.call_args[0][0]

    # Verify critical components in the message
    assert "45-MIN INACTIVITY KILL-SWITCH" in msg
    assert "BANKNIFTY" in msg
    assert "56200" in msg
    assert "763.11" in msg
    assert "768.80" in msg
    assert "+45.95" in msg
    assert "ACTION REQUIRED ON KITE" in msg
    assert "Exit position immediately on Kite" in msg


def test_telegram_exit_alert_target_hit():
    """Verify Telegram exit alert formats target hit properly."""
    bot = TelegramBot(bot_token="test_token", chat_id="test_chat")
    bot.send_message = MagicMock(return_value=True)

    trade_info = {
        "trade_id": "PAPER-99999",
        "symbol": "NIFTY 50",
        "instrument": "NIFTY24SEP2625100CE",
        "side": "BUY CE",
        "quantity": 75,
        "entry_price": 120.0,
        "exit_price": 180.0,
        "return_pct": 50.0,
        "gross_pnl": 4500.0,
        "net_pnl": 4420.0,
        "costs": {"total": 80.0},
        "holding_duration_seconds": 1800,
        "exit_reason": "target",
    }

    bot.alert_trade_closed(trade_info)

    assert bot.send_message.called
    msg = bot.send_message.call_args[0][0]

    assert "TARGET ACHIEVED" in msg
    assert "Book profit on Kite now" in msg
    assert "NIFTY" in msg


def test_telegram_alert_new_signal_displays_telemetry():
    """Verify Telegram alert_new_signal displays VPR60 and GEX/ZGL when available."""
    bot = TelegramBot(bot_token="test_token", chat_id="test_chat")
    bot.send_message = MagicMock(return_value=True)

    signal = {
        "symbol": "NIFTY BANK",
        "action": "BUY_CE",
        "entry_price": 760.0,
        "stop_loss": 620.0,
        "target": 930.0,
        "strike": 56200,
        "option_type": "CE",
        "expiry": "2026-09-29",
        "instrument": "BANKNIFTY29SEP2656200CE",
        "tradingsymbol": "BANKNIFTY29SEP2656200CE",
        "strategy": "Golden_Setup (1H+VWAP+ORB)",
        "confidence": 0.75,
        "signal_score": 4.5,
        "commitment_ratio": 0.42,
        "vpr60": 0.72,
        "net_gex": 12500000.0,
        "zgl": 55800.0,
    }

    bot.alert_new_signal(signal)

    assert bot.send_message.called
    msg = bot.send_message.call_args[0][0]

    assert "Commitment: 0.42" in msg
    assert "VPR₆₀: 72%" in msg
    assert "GEX: +1.2Cr" in msg
    assert "ZGL: 55800" in msg
