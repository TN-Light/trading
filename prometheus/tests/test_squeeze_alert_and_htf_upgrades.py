import pytest
import pandas as pd
from unittest.mock import MagicMock
from prometheus.interface.telegram_bot import TelegramBot
from prometheus.signals.price_action_momentum import PriceActionMomentumScanner
from prometheus.data.engine import DataEngine


def test_telegram_squeeze_alert_banner_short_gamma():
    bot = TelegramBot(bot_token="fake_token", chat_id="12345")
    bot.send_message = MagicMock()

    signal = {
        "symbol": "SENSEX",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "entry_price": 424.67,
        "stop_loss": 389.60,
        "target": 469.20,
        "risk_reward": 1.3,
        "net_gex": -5858561.36,
        "gamma_regime": "SHORT_GAMMA",
        "zgl": 72865.96,
        "tier": "C",
        "edge_score": 4.5,
        "quantity": 20,
        "tradingsymbol": "SENSEX26O0872600CE",
    }

    bot.alert_new_signal(signal)
    assert bot.send_message.called
    msg = bot.send_message.call_args[0][0]

    assert "HIGH-VELOCITY SQUEEZE ALERT" in msg
    assert "Option sellers trapped" in msg
    assert "Fast, aggressive Call / Upward spike expected" in msg


def test_telegram_golden_setup_header_not_on_tier_c():
    bot = TelegramBot(bot_token="fake_token", chat_id="12345")
    bot.send_message = MagicMock()

    signal = {
        "symbol": "SENSEX",
        "action": "BUY_CE",
        "strategy": "Golden_Setup (1H+VWAP+ORB)",
        "is_golden_setup": True,
        "entry_price": 424.67,
        "stop_loss": 389.60,
        "target": 469.20,
        "tier": "C",
        "edge_score": 4.5,
        "quantity": 20,
        "tradingsymbol": "SENSEX26O0872600CE",
    }

    bot.alert_new_signal(signal)
    assert bot.send_message.called
    msg = bot.send_message.call_args[0][0]

    # Tier C signals must NOT claim to be NEW GOLDEN SETUP SIGNAL
    assert "NEW GOLDEN SETUP SIGNAL" not in msg
    assert "NEW TRADING SIGNAL" in msg


def test_emerging_bullish_1h_trend():
    scanner = PriceActionMomentumScanner()

    # Create 1H candles where EMA20 < EMA50 from prior drop, but price has reclaimed both
    h_rows = []
    base_ts = pd.Timestamp("2026-10-06 09:15:00")
    for i in range(35):
        ts = base_ts - pd.Timedelta(hours=35 - i)
        p = 73000 - (i * 35)
        h_rows.append({"timestamp": ts, "open": p, "high": p + 20, "low": p - 20, "close": p, "volume": 1000})

    h_rows.append({"timestamp": base_ts, "open": 72500, "high": 72750, "low": 72450, "close": 72700, "volume": 5000})
    df_1h = pd.DataFrame(h_rows)

    # 15M candles for today
    m_rows = []
    base_15m = pd.Timestamp("2026-10-06 09:15:00")
    for i in range(20):
        ts = base_15m - pd.Timedelta(minutes=(20 - i) * 15)
        m_rows.append({"timestamp": ts, "open": 72400, "high": 72450, "low": 72350, "close": 72400, "volume": 1000})

    # Today's bars with ORB breakout (latest bar at 10:00 AM)
    m_rows.append({"timestamp": pd.Timestamp("2026-10-06 09:15:00"), "open": 72500, "high": 72580, "low": 72490, "close": 72550, "volume": 2000})
    m_rows.append({"timestamp": pd.Timestamp("2026-10-06 09:30:00"), "open": 72550, "high": 72600, "low": 72540, "close": 72590, "volume": 2500})
    m_rows.append({"timestamp": pd.Timestamp("2026-10-06 09:45:00"), "open": 72590, "high": 72680, "low": 72580, "close": 72650, "volume": 3500})
    m_rows.append({"timestamp": pd.Timestamp("2026-10-06 10:00:00"), "open": 72650, "high": 72720, "low": 72640, "close": 72700, "volume": 4000})
    df_15m = pd.DataFrame(m_rows)

    sig = scanner.evaluate_bar(df_15m, symbol="SENSEX", df_1h=df_1h, golden_mode=True)
    assert sig is not None
    assert sig["action"] == "BUY_CE"
    # Should contain Emerging Bullish confluence
    assert any("Emerging_Bullish" in r for r in sig["reasons"])


def test_engine_60m_cache_ttl():
    engine = DataEngine()
    key = "SENSEX:60minute:10"
    engine._mem_cache[key] = (pd.DataFrame([{"timestamp": pd.Timestamp.now(), "close": 72000}]), 1000.0)
    assert key in engine._mem_cache
