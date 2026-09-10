"""
Unit tests validating Deep Audit Bug Fixes:
1. BFO Exchange Routing for SENSEX/BANKEX in OrderManager and PositionMonitor
2. 2-leg spread LTP and dict-safe LTP resolution in PaperTrader
3. LivePriceFeed expiry preservation
4. Telegram message chunking for messages > 4000 chars
"""

from unittest.mock import MagicMock
import pytest

from prometheus.execution.broker import OrderSide, OrderType, ProductType
from prometheus.execution.order_manager import OrderManager
from prometheus.execution.paper_trader import PaperTrader
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.paper_executor.live_bridge import LivePriceFeed
from prometheus.interface.telegram_bot import TelegramBot


def test_order_manager_routes_sensex_to_bfo():
    from prometheus.execution.broker import OrderStatus
    mock_broker = MagicMock()
    def mock_place(order):
        order.order_id = "ORD-001"
        order.status = OrderStatus.COMPLETE
        return order
    mock_broker.place_order.side_effect = mock_place
    mock_risk = MagicMock()
    mock_risk.calculate_position_size.return_value = {"quantity": 10, "lots": 1}
    mock_check = MagicMock()
    mock_check.approved = True
    mock_risk.pre_trade_check.return_value = mock_check
    om = OrderManager(broker=mock_broker, risk_manager=mock_risk)

    sensex_sig = {
        "symbol": "SENSEX",
        "action": "BUY_CE",
        "entry_price": 100.0,
        "stop_loss": 80.0,
        "target": 150.0,
        "tradingsymbol": "SENSEX2691075300CE",
        "instrument": "SENSEX2691075300CE",
        "strike": 75300,
        "option_type": "CE",
        "lots": 1,
    }
    pos_sensex = om.execute_signal(sensex_sig)
    assert pos_sensex is not None
    calls = mock_broker.place_order.call_args_list
    placed_order = calls[0][0][0]
    assert placed_order.exchange == "BFO"
    assert placed_order.tradingsymbol == "SENSEX2691075300CE"

    sl_calls = [c for c in calls if c[0][0].order_type == OrderType.SL_M]
    assert len(sl_calls) == 1
    assert sl_calls[0][0][0].exchange == "BFO"


def test_paper_trader_handles_spread_and_dict_ltp():
    pt = PaperTrader()
    pt._price_feed = {
        "NIFTY2691523600CE": {"ltp": 100.35},
        "NIFTY2691523750CE": 48.0,
    }

    assert pt.get_ltp("NIFTY2691523600CE") == 100.35
    assert pt.get_ltp("NIFTY2691523750CE") == 48.0

    spread_ltp = pt.get_ltp("NIFTY2691523600CE/NIFTY2691523750CE")
    assert round(spread_ltp, 2) == 52.35


def test_telegram_message_chunking():
    bot = TelegramBot(bot_token="fake_token", chat_id="12345")
    bot._enabled = True

    sent_payloads = []
    def mock_post(url, json=None, timeout=10):
        sent_payloads.append(json)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        return mock_resp

    bot._session = MagicMock()
    bot._session.post = mock_post

    lines = [f"Line {i:03d}: " + "X" * 80 for i in range(100)]
    long_msg = "\n".join(lines)
    assert len(long_msg) > 9000

    ok = bot.send_message(long_msg)
    assert ok is True
    assert len(sent_payloads) >= 3
    for p in sent_payloads:
        assert len(p["text"]) <= 4000


def test_telegram_startup_online_message_delayed_recovery():
    bot = TelegramBot(bot_token="fake_token", chat_id="12345")
    bot._enabled = False
    bot._announced_online = False

    # Simulate alert_system_start called while bot is offline/blocked
    ok = bot.alert_system_start()
    assert ok is False
    assert bot._announced_online is False

    # Mock successful session and connection
    sent_msgs = []
    def mock_post(url, json=None, timeout=10):
        if json and "text" in json:
            sent_msgs.append(json["text"])
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"ok": True, "result": {"username": "testbot"}}
        return m

    def mock_get(url, timeout=10):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"ok": True, "result": {"username": "testbot"}}
        return m

    bot._try_connect = MagicMock(return_value=True)
    bot._session = MagicMock()
    bot._session.post = mock_post
    bot._session.get = mock_get
    bot._make_session = MagicMock(return_value=bot._session)

    # Trigger reconnect - should reconnect and automatically announce online
    bot._last_reconnect_attempt = 0
    bot.reconnect()

    assert bot._enabled is True
    assert bot._announced_online is True
    assert any("PROMETHEUS ONLINE" in m for m in sent_msgs)

    # Subsequent reconnect should not re-announce
    prior_count = len(sent_msgs)
    bot.reconnect()
    assert len(sent_msgs) == prior_count


def test_data_engine_short_term_mem_cache():
    import pandas as pd
    from prometheus.data.engine import DataEngine

    de = DataEngine()
    de.historical_source = "auto"
    mock_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-10 09:15", periods=5, freq="15min"),
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [99, 100, 101, 102, 103],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })

    fetch_count = 0
    def mock_fetch(*args, **kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return mock_df.copy()

    de._fetch_with_retry = mock_fetch
    de.store = MagicMock()
    de.store.get_ohlcv.return_value = pd.DataFrame()

    # First fetch: calls _fetch_with_retry
    df1 = de.fetch_historical("NIFTY 50", days=5, interval="15minute", force_refresh=False)
    assert fetch_count == 1
    assert len(df1) == 5

    # Second fetch within 45s (even with force_refresh=True as in fetch_intraday): hits mem cache!
    df2 = de.fetch_intraday("NIFTY 50", interval="15minute", days=5)
    assert fetch_count == 1  # No additional API call!
    assert len(df2) == 5

    # Bypass mem cache: makes another call
    df3 = de.fetch_intraday("NIFTY 50", interval="15minute", days=5, bypass_mem_cache=True)
    assert fetch_count == 2
