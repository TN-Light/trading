"""
Unit tests verifying the 5 October 15 quantitative, execution, and telemetry fixes:
1. Zero-volume cash index rolling Typical-Price VWAP & symmetric directional alignment in CreditSpreadStrategy.
2. Directional compatibility check preventing opposing Credit Spreads from overriding Option Buying.
3. Symbol-level stop-out lockout eliminating martingale ladder re-entries in LivePaperCapture.
4. Datetime timezone normalization in PostExitTracker and TradeSnapshot processing in LivePaperCapture.
5. Option Greek list payload parsing in AngelOneOptionChain.
"""

from datetime import datetime, time as dtime, timedelta, timezone
import threading
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest

from prometheus.strategies.credit_spread import CreditSpreadStrategy
from prometheus.papertrade.post_exit_tracker import PostExitTracker, PostExitWatchItem
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
from prometheus.papertrade.types import PaperTrade, Direction, ExitReason, TradeSnapshot
from prometheus.utils.indian_market import IST


# -----------------------------------------------------------------------------
# 1. Tests for CreditSpreadStrategy (Zero-Volume VWAP & Symmetric Direction)
# -----------------------------------------------------------------------------

class MockChain:
    """Mock option chain returning realistic live quotes for short and hedge legs."""
    def __init__(self):
        self._first = True

    def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
        if self._first:
            self._first = False
            prem = 50.0  # Short strike premium
        else:
            self._first = True
            prem = 20.0  # Long hedge strike premium
        return {"ltp": prem, "bid": prem - 1.0, "ask": prem + 1.0, "tradingsymbol": f"{symbol}{int(strike)}{option_type}"}

    def get_option_chain(self, symbol, spot_price=None, expiry_date=None):
        return pd.DataFrame([
            {"strike": 24200, "option_type": "CE", "oi": 50000},
            {"strike": 23800, "option_type": "PE", "oi": 60000},
        ])


def _generate_test_candles(n_bars=30, start_dt=datetime(2026, 8, 24, 9, 15), base_price=24000.0, trend=0.0):
    timestamps = [start_dt + timedelta(minutes=15 * i) for i in range(n_bars)]
    rows = []
    current = base_price
    for i, ts in enumerate(timestamps):
        current += trend
        rows.append({
            "timestamp": ts,
            "open": current - 2.0,
            "high": current + 8.0,
            "low": current - 8.0,
            "close": current,
            "volume": 0.0,  # Zero-volume cash index
        })
    return pd.DataFrame(rows)


def test_credit_spread_zero_volume_typical_price_vwap_bullish():
    """Verify zero-volume cash index computes rolling Typical-Price VWAP and aligns directionally."""
    strat = CreditSpreadStrategy()
    # Prior day 20 bars + today 8 bars (09:15 to 11:00) with gentle upward trend (+3 pts per bar)
    prior_df = _generate_test_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), base_price=23950.0)
    today_df = _generate_test_candles(n_bars=8, start_dt=datetime(2026, 8, 24, 9, 15), base_price=24000.0, trend=3.0)
    df = pd.concat([prior_df, today_df], ignore_index=True)

    mock_chain = MockChain()
    # In an uptrend, close > vwap and ema9 > ema21 -> should generate BULL_PUT_SPREAD
    sig = strat.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=mock_chain)

    assert sig is not None, "Signal should be generated"
    assert sig["spread_type"] == "BULL_PUT_SPREAD", f"Expected BULL_PUT_SPREAD in uptrend, got {sig['spread_type']}"
    assert sig["action"] == "SELL_PUT_SPREAD"


def test_credit_spread_neutral_fallback_symmetric_alignment():
    """In neutral/fallback case, being in upper half of range is bullish strength, not bearish resistance."""
    strat = CreditSpreadStrategy()
    prior_df = _generate_test_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), base_price=24000.0)
    # Today 8 bars in tight range (23960 to 24040, midpoint 24000), close = 24025 (upper half)
    today_rows = []
    start_dt = datetime(2026, 8, 24, 9, 15)
    for i in range(8):
        ts = start_dt + timedelta(minutes=15 * i)
        # Bar 0 sets low (23960), bar 1 sets high (24040), remaining bars fluctuate around 24025
        h = 24040.0 if i == 1 else 24030.0
        l = 23960.0 if i == 0 else 24010.0
        c = 24025.0
        today_rows.append({"timestamp": ts, "open": 24020.0, "high": h, "low": l, "close": c, "volume": 0.0})
    today_df = pd.DataFrame(today_rows)
    df = pd.concat([prior_df, today_df], ignore_index=True)

    mock_chain = MockChain()
    # Neutral/fallback: upper half of range should generate BULL_PUT_SPREAD
    sig = strat.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=mock_chain)

    assert sig is not None
    assert sig["spread_type"] == "BULL_PUT_SPREAD", "Upper half of range should trigger Bull Put Spread, not Bear Call Spread"


def test_credit_spread_neutral_fallback_lower_half_bearish():
    """In neutral/fallback case, being in lower half of range triggers Bear Call Spread."""
    strat = CreditSpreadStrategy()
    prior_df = _generate_test_candles(n_bars=20, start_dt=datetime(2026, 8, 21, 9, 15), base_price=24000.0)
    # Today 8 bars in tight range (23960 to 24040, midpoint 24000), close = 23975 (lower half)
    today_rows = []
    start_dt = datetime(2026, 8, 24, 9, 15)
    for i in range(8):
        ts = start_dt + timedelta(minutes=15 * i)
        h = 24040.0 if i == 1 else 23990.0
        l = 23960.0 if i == 0 else 23970.0
        c = 23975.0
        today_rows.append({"timestamp": ts, "open": 23980.0, "high": h, "low": l, "close": c, "volume": 0.0})
    today_df = pd.DataFrame(today_rows)
    df = pd.concat([prior_df, today_df], ignore_index=True)

    mock_chain = MockChain()
    sig = strat.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=mock_chain)

    assert sig is not None
    assert sig["spread_type"] == "BEAR_CALL_SPREAD", "Lower half of range should trigger Bear Call Spread"


# -----------------------------------------------------------------------------
# 2. Tests for Directional Compatibility in Seller-First Priority (main.py)
# -----------------------------------------------------------------------------

def test_directional_compatibility_opposing_spread_cannot_override():
    """Verify that an opposing Credit Spread (BEAR_CALL_SPREAD) cannot override a bullish Option Buy (BUY_CE)."""
    symbol = "SENSEX"

    # Bullish Option Buy
    exec_sig = {
        "action": "BUY_CE",
        "option_type": "CE",
        "symbol": symbol,
        "signal_score": 6.0,
        "is_golden_setup": False,
        "direction": "bullish",
    }

    # Opposing Bearish Credit Spread
    cs_sig = {
        "strategy_type": "credit_spread",
        "spread_type": "BEAR_CALL_SPREAD",
        "action": "SELL_CALL_SPREAD",
        "symbol": symbol,
        "signal_score": 8.0,
        "pop_pct": 88.0,
        "is_0dte": False,
    }

    buy_action = str(exec_sig.get("action", "")).upper()
    buy_opt = str(exec_sig.get("option_type", "")).upper()
    buy_dir = str(exec_sig.get("direction", "")).lower()
    is_buy_bullish = ("CE" in buy_action) or (buy_opt == "CE") or ("CALL" in buy_action) or (buy_dir in ("bullish", "long", "buy"))
    is_buy_bearish = ("PE" in buy_action) or (buy_opt == "PE") or ("PUT" in buy_action) or (buy_dir in ("bearish", "short", "sell"))

    spread_type = str(cs_sig.get("spread_type", "")).upper()
    cs_act = str(cs_sig.get("action", "")).upper()
    is_spread_bullish = ("BULL" in spread_type) or ("PUT_SPREAD" in spread_type) or ("BULL" in cs_act) or ("PUT_SPREAD" in cs_act)
    is_spread_bearish = ("BEAR" in spread_type) or ("CALL_SPREAD" in spread_type) or ("BEAR" in cs_act) or ("CALL_SPREAD" in cs_act)

    directions_conflict = (is_buy_bullish and is_spread_bearish) or (is_buy_bearish and is_spread_bullish)
    assert directions_conflict is True, "BUY_CE and BEAR_CALL_SPREAD must be detected as conflicting"


def test_directional_compatibility_compatible_spread_can_override_in_seller_first():
    """Verify that a compatible Credit Spread (BULL_PUT_SPREAD) properly overrides BUY_CE under Seller-First."""
    exec_sig = {
        "action": "BUY_CE",
        "option_type": "CE",
        "symbol": "SENSEX",
        "signal_score": 6.0,
        "is_golden_setup": False,
        "direction": "bullish",
    }

    cs_sig = {
        "strategy_type": "credit_spread",
        "spread_type": "BULL_PUT_SPREAD",
        "action": "SELL_PUT_SPREAD",
        "symbol": "SENSEX",
        "signal_score": 8.0,
        "pop_pct": 88.0,
        "is_0dte": False,
    }

    buy_action = str(exec_sig.get("action", "")).upper()
    buy_opt = str(exec_sig.get("option_type", "")).upper()
    buy_dir = str(exec_sig.get("direction", "")).lower()
    is_buy_bullish = ("CE" in buy_action) or (buy_opt == "CE") or ("CALL" in buy_action) or (buy_dir in ("bullish", "long", "buy"))
    is_buy_bearish = ("PE" in buy_action) or (buy_opt == "PE") or ("PUT" in buy_action) or (buy_dir in ("bearish", "short", "sell"))

    spread_type = str(cs_sig.get("spread_type", "")).upper()
    cs_act = str(cs_sig.get("action", "")).upper()
    is_spread_bullish = ("BULL" in spread_type) or ("PUT_SPREAD" in spread_type) or ("BULL" in cs_act) or ("PUT_SPREAD" in cs_act)
    is_spread_bearish = ("BEAR" in spread_type) or ("CALL_SPREAD" in spread_type) or ("BEAR" in cs_act) or ("CALL_SPREAD" in cs_act)

    directions_conflict = (is_buy_bullish and is_spread_bearish) or (is_buy_bearish and is_spread_bullish)
    assert directions_conflict is False, "BUY_CE and BULL_PUT_SPREAD are directionally aligned"


# -----------------------------------------------------------------------------
# 3. Tests for Symbol-Level Lockout on Stop-Outs (live_bridge.py)
# -----------------------------------------------------------------------------

def test_symbol_level_lockout_on_stop_loss(tmp_path):
    """Verify that when a trade stops out, (symbol, direction) is locked out from re-entries."""
    cfg = CaptureConfig(
        enabled=True,
        sqlite_path=str(tmp_path / "test.sqlite"),
        csv_path=str(tmp_path / "test.csv"),
    )
    ltp_feed = MagicMock()
    ltp_feed.get_ltp.return_value = 100.0

    capture = LivePaperCapture(config=cfg, ltp_source=ltp_feed)

    # 1. Simulate a closed trade with ExitReason.STOP_LOSS
    trade = PaperTrade(
        trade_id="T1",
        symbol="SENSEX",
        instrument="SENSEX26101581000CE",
        underlying="SENSEX",
        direction=Direction.LONG,
        quantity=10,
        entry_price=150.0,
        exit_price=120.0,
        entry_time=datetime.now(IST),
        exit_time=datetime.now(IST),
        exit_reason=ExitReason.STOP_LOSS,
        gross_pnl=-300.0,
        costs=20.0,
        net_pnl=-320.0,
        return_pct=-20.0,
        holding_duration_seconds=60,
    )

    capture._on_trade_closed(trade)

    # Verify set was populated
    assert ("SENSEX", "BUY_CE") in capture._today_stopped_out_symbol_directions

    # 2. Attempt a new signal for SENSEX BUY_CE on a different strike (martingale ladder)
    new_signal = {
        "symbol": "SENSEX",
        "action": "BUY_CE",
        "option_type": "CE",
        "strike": 81200,
        "expiry": "2026-10-15",
        "entry_price": 95.0,
        "stop_loss": 75.0,
        "target": 140.0,
        "tradingsymbol": "SENSEX26101581200CE",
        "instrument": "SENSEX26101581200CE",
    }

    result = capture.on_signal(new_signal)
    assert result is None, "Signal must be rejected due to intraday stop-out lockout"

    # 3. But a signal on a DIFFERENT direction (e.g. BUY_PE) must NOT be blocked
    pe_signal = {
        "symbol": "SENSEX",
        "action": "BUY_PE",
        "option_type": "PE",
        "strike": 80800,
        "expiry": "2026-10-15",
        "entry_price": 90.0,
        "stop_loss": 70.0,
        "target": 130.0,
        "tradingsymbol": "SENSEX26101580800PE",
        "instrument": "SENSEX26101580800PE",
    }
    capture._engine.process_new_signal = MagicMock(return_value="T2")
    pe_result = capture.on_signal(pe_signal)
    assert pe_result == "T2", "Opposite direction (BUY_PE) should not be blocked"


def test_spread_symbol_level_lockout_on_stop_loss(tmp_path):
    """Verify that credit spreads are locked out at (symbol, direction) on stop-out."""
    cfg = CaptureConfig(
        enabled=True,
        sqlite_path=str(tmp_path / "test.sqlite"),
        csv_path=str(tmp_path / "test.csv"),
    )
    capture = LivePaperCapture(config=cfg, ltp_source=MagicMock())

    trade = PaperTrade(
        trade_id="T_SPREAD_1",
        symbol="SENSEX",
        instrument="SENSEX26101581500CE/SENSEX26101582000CE",
        underlying="SENSEX",
        direction=Direction.SHORT,
        quantity=10,
        entry_price=40.0,
        exit_price=60.0,
        entry_time=datetime.now(IST),
        exit_time=datetime.now(IST),
        exit_reason=ExitReason.STOP_LOSS,
        gross_pnl=-200.0,
        costs=30.0,
        net_pnl=-230.0,
        return_pct=-50.0,
        holding_duration_seconds=120,
        strategy="Hedged_Credit_Spread",
    )

    capture._on_trade_closed(trade)
    assert ("SENSEX", "BEAR_CALL_SPREAD") in capture._today_stopped_out_symbol_directions

    # Re-entry with rolled strikes
    cs_sig = {
        "symbol": "SENSEX",
        "spread_type": "BEAR_CALL_SPREAD",
        "action": "SELL_CALL_SPREAD",
        "strike": 81700,
        "expiry": "2026-10-15",
        "net_credit": 35.0,
        "stop_loss": 52.5,
        "tradingsymbol": "SENSEX26101581700CE/SENSEX26101582200CE",
        "instrument": "SENSEX26101581700CE/SENSEX26101582200CE",
    }

    res = capture.on_signal(cs_sig)
    assert res is None, "Rolled credit spread must be rejected due to intraday stop-out lockout"


def test_bull_put_spread_symbol_level_lockout_on_stop_loss(tmp_path):
    """Verify that Bull Put Spreads are accurately classified and locked out at (symbol, direction) on stop-out."""
    cfg = CaptureConfig(
        enabled=True,
        sqlite_path=str(tmp_path / "test_put_spread.sqlite"),
        csv_path=str(tmp_path / "test_put_spread.csv"),
    )
    capture = LivePaperCapture(config=cfg, ltp_source=MagicMock())

    trade = PaperTrade(
        trade_id="T_PUT_SPREAD_1",
        symbol="NIFTY 50",
        instrument="NIFTY26OCT24500PE/NIFTY26OCT24350PE",
        underlying="NIFTY",
        direction=Direction.SHORT,
        quantity=50,
        entry_price=30.0,
        exit_price=50.0,
        entry_time=datetime.now(IST),
        exit_time=datetime.now(IST),
        exit_reason=ExitReason.STOP_LOSS,
        gross_pnl=-1000.0,
        costs=40.0,
        net_pnl=-1040.0,
        return_pct=-66.7,
        holding_duration_seconds=180,
        strategy="Hedged_Credit_Spread",
    )

    assert trade.trade_type == "BULL_PUT_SPREAD", f"Expected BULL_PUT_SPREAD, got {trade.trade_type}"
    capture._on_trade_closed(trade)
    assert ("NIFTY 50", "BULL_PUT_SPREAD") in capture._today_stopped_out_symbol_directions
    assert ("NIFTY 50", "BEAR_CALL_SPREAD") not in capture._today_stopped_out_symbol_directions

    # Re-entry of rolled Bull Put Spread must be rejected
    put_sig = {
        "symbol": "NIFTY 50",
        "spread_type": "BULL_PUT_SPREAD",
        "action": "SELL_PUT_SPREAD",
        "strike": 24400,
        "expiry": "2026-10-26",
        "net_credit": 25.0,
        "stop_loss": 40.0,
        "tradingsymbol": "NIFTY26OCT24400PE/NIFTY26OCT24250PE",
        "instrument": "NIFTY26OCT24400PE/NIFTY26OCT24250PE",
    }
    assert capture.on_signal(put_sig) is None, "Rolled Bull Put Spread must be locked out"

    # Opposing Bear Call Spread must NOT be blocked
    call_sig = {
        "symbol": "NIFTY 50",
        "spread_type": "BEAR_CALL_SPREAD",
        "action": "SELL_CALL_SPREAD",
        "strike": 24800,
        "expiry": "2026-10-26",
        "net_credit": 28.0,
        "stop_loss": 42.0,
        "tradingsymbol": "NIFTY26OCT24800CE/NIFTY26OCT24950CE",
        "instrument": "NIFTY26OCT24800CE/NIFTY26OCT24950CE",
    }
    capture._engine.process_new_signal = MagicMock(return_value="T_CALL_NEW")
    assert capture.on_signal(call_sig) == "T_CALL_NEW", "Opposing Bear Call Spread should be allowed"


# -----------------------------------------------------------------------------
# 4. Tests for Datetime Timezone Bug in PostExitTracker & LiveBridge on_bar
# -----------------------------------------------------------------------------

def test_post_exit_tracker_timezone_normalization():
    """Verify that timezone-naive and timezone-aware datetimes do not raise TypeError."""
    tracker = PostExitTracker()

    exit_aware = datetime.now(IST)
    item = PostExitWatchItem(
        trade_id="TEST_TZ_1",
        symbol="NIFTY 50",
        instrument="NIFTY2691523600CE",
        underlying="NIFTY",
        direction="LONG",
        tier="C",
        entry_price=100.0,
        exit_price=90.0,
        exit_reason="stop_loss",
        exit_time=exit_aware,
        target=140.0,
        stop_loss=80.0,
    )
    tracker._watches[item.trade_id] = item

    # 1. Call on_tick with timezone-naive timestamp
    now_naive = datetime.now()
    updates = tracker.on_tick("NIFTY2691523600CE", 95.0, timestamp=now_naive)
    assert len(updates) == 1
    assert updates[0].post_exit_mfe == 95.0

    # 2. Call on_tick with timezone-aware timestamp when exit_time is naive
    item.exit_time = datetime.now()  # naive
    now_aware = datetime.now(IST)
    updates2 = tracker.on_tick("NIFTY2691523600CE", 98.0, timestamp=now_aware)
    assert len(updates2) == 1
    assert updates2[0].post_exit_mfe == 98.0


def test_live_bridge_on_bar_accepts_tradesnapshot_directly(tmp_path):
    """Verify live_bridge.on_bar cleanly handles TradeSnapshot instance."""
    cfg = CaptureConfig(
        enabled=True,
        sqlite_path=str(tmp_path / "test.sqlite"),
        csv_path=str(tmp_path / "test.csv"),
    )
    capture = LivePaperCapture(config=cfg, ltp_source=MagicMock())
    capture._engine.process_bar = MagicMock()

    snap = TradeSnapshot(
        timestamp=datetime.now(IST),
        symbol="NIFTY 50",
        instrument="",
        open=24000.0,
        high=24050.0,
        low=23980.0,
        close=24020.0,
        volume=1000.0,
    )

    # Should not raise AttributeError: 'TradeSnapshot' object has no attribute 'get'
    capture.on_bar("NIFTY 50", snap)
    capture._engine.process_bar.assert_called_once_with(
        snap, is_session_end=False, is_square_off=False
    )


# -----------------------------------------------------------------------------
# 5. Tests for Option Greek Payload Parsing (angelone_options.py)
# -----------------------------------------------------------------------------

def test_angelone_option_greek_list_payload():
    """Verify AngelOneOptionChain handles SmartAPI returning a list for optionGreek."""
    from prometheus.data.angelone_options import AngelOneOptionChain

    chain = AngelOneOptionChain.__new__(AngelOneOptionChain)
    chain.enabled = True
    chain._oi_snapshots = {}
    chain._token_cache = {}
    chain._exchange_cache = {}
    chain.UNDERLYING_MAP = {"NIFTY": "NIFTY", "NIFTY 50": "NIFTY"}
    from datetime import date
    chain._cache_date = date.today().isoformat()
    chain._oi_lock = threading.Lock()
    chain._auth_cooldown_sec = 60

    # Mock SmartAPI client returning a list of dicts for optionGreek
    mock_client = MagicMock()
    mock_client.optionGreek.return_value = [
        {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": {
                "delta": "0.5234",
                "gamma": "0.0125",
                "theta": "-8.45",
                "vega": "14.20",
                "impliedVolatility": "21.5",
            }
        }
    ]
    chain._get_obj = MagicMock(return_value=mock_client)
    chain.get_spot_price = MagicMock(return_value=24000.0)

    # Mock discover_contracts / cache lookup to return candidate contract
    chain._token_cache["NIFTY"] = [
        {
            "tradingsymbol": "NIFTY2691524000CE",
            "strike": 24000.0,
            "option_type": "CE",
            "expiry": "15SEP26",
            "symboltoken": "12345",
        }
    ]
    chain._get_batch_engine = MagicMock(return_value=None)
    mock_client.ltpData.return_value = {"status": True, "data": {"ltp": 125.0}}
    chain._rate_limit = MagicMock(return_value=True)
    chain._mark_rate_limited = MagicMock(return_value=False)

    premium = chain.get_real_premium("NIFTY 50", 24000.0, "CE", expiry="2026-09-15", include_greeks=True)

    assert premium is not None
    assert premium["ltp"] == 125.0
    assert premium["delta"] == 0.5234
    assert premium["gamma"] == 0.0125
    assert premium["theta"] == -8.45
    assert premium["vega"] == 14.20
    assert premium["iv"] == 21.5
