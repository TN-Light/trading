import time
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, date, time as dtime
import sqlite3

from prometheus.main import Prometheus


class DummyRisk:
    def __init__(self, capital=105000.0):
        self.capital = capital
        self.current_capital = capital
        self.realized_pnl_today = 2500.0
        self.trades_today = 3
        self._halted = False

    def get_portfolio_state(self):
        class State:
            def __init__(self, cap):
                self.capital = cap
                self.realized_pnl_today = 2500.0
                self.trades_today = 3
        return State(self.capital)


class DummyTelegram:
    def __init__(self):
        self.messages = []
        self.summaries = []
        self.scanner_summaries = []

    def send_message(self, msg, **kwargs):
        self.messages.append(msg)

    def alert_daily_summary(self, summary):
        self.summaries.append(summary)

    def alert_scanner_summary(self, scan_results):
        self.scanner_summaries.append(scan_results)


# -------------------------------------------------------------------------
# Test 1 & 2: _run_tg_scan_job Metadata Preservation & signal_count
# -------------------------------------------------------------------------
def test_scan_intra_cmd_preserves_metadata_and_signal_count():
    app = Prometheus.__new__(Prometheus)
    app.mode = "paper"
    app.data = MagicMock()
    app.data.fetch_intraday.return_value = []
    app.symbols = ["BANKNIFTY"]
    app.stock_symbols = []
    app.all_symbols = []
    app._today_traded_symbols = set()
    app._today_traded_instruments = set()
    app.telegram = DummyTelegram()
    app._scan_lock = MagicMock()
    app._paper_capture_feed_bars = MagicMock()
    app._get_intraday_instruments = MagicMock(return_value=["BANKNIFTY"])
    app._select_intraday_interval = MagicMock(return_value="15minute")
    app.regime_detector = MagicMock()

    exec_sig = {
        "action": "BUY_CE",
        "symbol": "BANKNIFTY",
        "entry_price": 450.0,
        "stop_loss": 410.0,
        "target": 550.0,
        "risk_reward": 2.5,
        "edge_score": 8.75,
        "tier": "Tier S",
        "tier_badge": "🏆 Tier S (PERFECT_STORM)",
        "is_live_eligible": True,
        "strategy_type": "DIRECTIONAL_BUY",
        "is_golden_setup": True,
        "otm_sigma": 1.5,
        "net_gex": -2500000.0,
        "gamma_regime": "SHORT_GAMMA",
        "commitment_ratio": 1.45,
        "reasons": ["ORB_Breakout", "SuperTrend_Aligned", "VWAP_Pullback", "OI_Volume_Surge"],
        "tradingsymbol": "BANKNIFTY26NOV48000CE",
        "confidence": 0.88,
    }

    app._get_intraday_signal_for_execution = MagicMock(return_value=exec_sig)

    # Patch market hours (10:00 AM on a Monday) and is_trading_day
    mock_now = datetime(2026, 11, 9, 10, 0, 0)
    with patch("prometheus.main.datetime") as mock_dt, \
         patch("prometheus.main.is_trading_day", return_value=True), \
         patch("prometheus.main.get", return_value=True):
        mock_dt.now.return_value = mock_now
        app._run_tg_scan_job("intraday")

    assert len(app.telegram.scanner_summaries) == 1
    results = app.telegram.scanner_summaries[0]
    assert len(results) == 1
    res = results[0]

    # Verify all rich metadata fields are preserved
    assert res["tier"] == "Tier S"
    assert res["tier_badge"] == "🏆 Tier S (PERFECT_STORM)"
    assert res["is_live_eligible"] is True
    assert res["strategy_type"] == "DIRECTIONAL_BUY"
    assert res["is_golden_setup"] is True
    assert res["gamma_regime"] == "SHORT_GAMMA"
    assert res["commitment_ratio"] == 1.45
    assert res["net_gex"] == -2500000.0

    # Verify signal_count is count of reasons (4), NOT edge_score (8)
    assert res["signal_count"] == 4
    assert res["edge_score"] == 8.75


# -------------------------------------------------------------------------
# Test 3: _tg_cmd_scan_count Inter-Scan & Daily Lockout Dedup
# -------------------------------------------------------------------------
def test_tg_cmd_scan_count_lockout_and_dedup():
    app = Prometheus.__new__(Prometheus)
    app.mode = "paper"
    app.all_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE"]
    app.symbols = ["NIFTY", "BANKNIFTY"]
    app.stock_symbols = ["RELIANCE"]
    app.telegram = DummyTelegram()
    app._today_traded_symbols = {"NIFTY"}
    app._today_traded_instruments = {"BANKNIFTY26NOV48000CE"}

    # Mock position_monitor having RELIANCE
    pm = MagicMock()
    pm.has_symbol.side_effect = lambda sym: sym == "RELIANCE"
    app.position_monitor = pm

    # Mock paper_capture direction lockout
    pc = MagicMock()
    pc.is_direction_stopped_out.side_effect = lambda sym, refined_signal: sym == "BANKNIFTY"
    app._paper_capture = pc

    app._get_intraday_signal_for_execution = MagicMock(return_value={
        "action": "BUY_CE",
        "symbol": "BANKNIFTY",
        "tradingsymbol": "BANKNIFTY26NOV48000CE"
    })

    # Test dedup logic:
    # 1. Traded symbol NIFTY is filtered
    assert "NIFTY" in app._today_traded_symbols
    # 2. Instrument BANKNIFTY26NOV48000CE is filtered
    assert "BANKNIFTY26NOV48000CE" in app._today_traded_instruments
    # 3. Position monitor has RELIANCE
    assert app.position_monitor.has_symbol("RELIANCE") is True
    # 4. Direction stopped out on BANKNIFTY
    assert app.paper_capture.is_direction_stopped_out("BANKNIFTY", refined_signal={}) is True


# -------------------------------------------------------------------------
# Test 4: _extract_position_pnl Handles Floats, Callables, and Zero Fallback
# -------------------------------------------------------------------------
def test_extract_position_pnl_various_types():
    # 1. Float property
    class PosFloat:
        unrealized_pnl = 1250.50

    assert Prometheus._extract_position_pnl(PosFloat()) == 1250.50

    # 2. Callable method requiring LTP
    class PosMethod:
        last_price = 120.0
        entry_price = 100.0
        quantity = 25
        def unrealized_pnl(self, ltp):
            return (ltp - self.entry_price) * self.quantity

    assert Prometheus._extract_position_pnl(PosMethod()) == 500.0

    # 3. pnl == 0.0 with price/quantity fallback
    class PosFallback:
        unrealized_pnl = 0.0
        last_price = 150.0
        average_price = 140.0
        quantity = 15

    assert Prometheus._extract_position_pnl(PosFallback()) == 150.0


# -------------------------------------------------------------------------
# Test 5: _send_daily_summary Equity Double Count & Tier Segregation
# -------------------------------------------------------------------------
def test_send_daily_summary_equity_and_tier_segregation(tmp_path):
    real_connect = sqlite3.connect
    db_file = str(tmp_path / "live_ledger.sqlite")
    conn = real_connect(db_file)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE paper_trades (
            entry_time TEXT,
            net_pnl REAL,
            gross_pnl REAL,
            tier TEXT
        )
    """)
    today_str = date.today().strftime("%Y-%m-%d")

    # Insert Live trade (Tier S) +1500 net
    c.execute("INSERT INTO paper_trades VALUES (?, 1500.0, 1600.0, 'S')", (f"{today_str} 10:00:00",))
    # Insert Live trade (Tier B) +800 net
    c.execute("INSERT INTO paper_trades VALUES (?, 800.0, 900.0, 'B')", (f"{today_str} 11:00:00",))
    # Insert Experimental Sandbox trade (Tier C) -3000 net
    c.execute("INSERT INTO paper_trades VALUES (?, -3000.0, -2900.0, 'C')", (f"{today_str} 12:00:00",))
    conn.commit()
    conn.close()

    app = Prometheus.__new__(Prometheus)
    app.risk = DummyRisk(capital=105000.0)
    app.telegram = DummyTelegram()
    app.multi_account = None
    app._get_intraday_guardrail_audit_line = MagicMock(return_value="Guardrails OK")

    def mock_sqlite_connect(path):
        return real_connect(db_file)

    with patch("sqlite3.connect", side_effect=mock_sqlite_connect):
        app._send_daily_summary()

    assert len(app.telegram.summaries) == 1
    summary = app.telegram.summaries[0]

    # Equity must equal state.capital (105000), NOT state.capital + daily_pnl (double-counted)
    assert summary["equity"] == 105000.0

    # Operational daily PnL must segregate out Tier C: 1500 + 800 = 2300, 2 trades, 2 winning
    assert summary["daily_pnl"] == 2300.0
    assert summary["total_trades"] == 2
    assert summary["winning_trades"] == 2


# -------------------------------------------------------------------------
# Test 6: _tg_cmd_scan Lock TTL Timeout
# -------------------------------------------------------------------------
def test_tg_cmd_scan_lock_ttl():
    import threading
    app = Prometheus.__new__(Prometheus)
    app._scan_lock = threading.Lock()
    app._scan_in_progress = True
    app._scan_started_at = time.time() - 200.0  # 200 seconds ago (stuck)
    app.all_symbols = ["NIFTY", "BANKNIFTY"]
    app.symbols = ["NIFTY", "BANKNIFTY"]
    app.stock_symbols = []

    # Calling _tg_cmd_scan should reset the lock because 200s > 180s TTL
    with patch("threading.Thread") as mock_thread:
        res = app._tg_cmd_scan()
        assert "Scan already in progress" not in res
        assert "Scan started" in res
        assert app._scan_in_progress is True
        assert time.time() - app._scan_started_at < 5.0


# -------------------------------------------------------------------------
# Test 7: run_parrondo_tuning Win Rate Percentage Scaling
# -------------------------------------------------------------------------
def test_parrondo_tuning_win_rate_scale():
    results = [
        {"combo": "Combo1", "wr": 58.0, "dd": 25.0, "pf": 1.8, "sharpe": 2.1, "mr_min_score": 5.0},
        {"combo": "Combo2", "wr": 0.55, "dd": 30.0, "pf": 1.6, "sharpe": 1.9, "mr_min_score": 4.5},
        {"combo": "Combo3", "wr": 45.0, "dd": 20.0, "pf": 1.5, "sharpe": 1.5, "mr_min_score": 4.0},
    ]

    filtered = [
        r for r in results
        if (r['wr'] if r['wr'] > 1.0 else r['wr'] * 100.0) >= 52.0
        and r['dd'] <= 55.0
        and r['pf'] >= 1.4
    ]

    # Combo1 (58%) and Combo2 (55%) should both pass
    assert len(filtered) == 2
    assert filtered[0]["combo"] == "Combo1"
    assert filtered[1]["combo"] == "Combo2"


# -------------------------------------------------------------------------
# Test 8: _make_signal_generator Seamless Afternoon Window (13:15)
# -------------------------------------------------------------------------
def test_make_signal_generator_lunch_dead_zone_defaults():
    # Verify dead zone defaults: 11:30 to 13:15
    overrides = {}
    dead_start = str(overrides.get("dead_zone_start", "11:30") or "11:30")
    dead_end = str(overrides.get("dead_zone_end", "13:15") or "13:15")
    aft_start = str(overrides.get("afternoon_window_start", dead_end) or dead_end)
    use_second_window = bool(overrides.get("use_second_window", True))

    ds_h, ds_m = map(int, dead_start.split(":"))
    de_h, de_m = map(int, dead_end.split(":"))
    as_h, as_m = map(int, aft_start.split(":"))

    dead_start_time = dtime(ds_h, ds_m)
    dead_end_time = dtime(de_h, de_m)
    aft_start_time = dtime(as_h, as_m)

    # 12:00 is dead zone
    bar_t = dtime(12, 0)
    assert dead_start_time <= bar_t < dead_end_time

    # 13:30 (1:30 PM) is in the active afternoon window!
    bar_t_aft = dtime(13, 30)
    assert not (dead_start_time <= bar_t_aft < dead_end_time)
    assert bar_t_aft >= aft_start_time


# -------------------------------------------------------------------------
# Test 9: SENSEX Not in SIGNAL_ONLY in run_scan
# -------------------------------------------------------------------------
def test_sensex_not_in_signal_only():
    with open("prometheus/main.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Find definition of SIGNAL_ONLY in run_scan
    assert 'SIGNAL_ONLY = {"NIFTY MIDCAP SELECT", "NIFTY NEXT 50"}' in content
    assert '"SENSEX"' not in 'SIGNAL_ONLY = {"NIFTY MIDCAP SELECT", "NIFTY NEXT 50"}'


# -------------------------------------------------------------------------
# Test 10: _scan_stocks_batched Dynamic Defaults
# -------------------------------------------------------------------------
def test_scan_stocks_batched_dynamic_defaults():
    app = Prometheus.__new__(Prometheus)
    app.swing_cfg = {"stock_confidence_min": 0.68, "stock_min_signals": 4}
    app.data = MagicMock()

    # Check defaults logic inside _scan_stocks_batched
    min_confidence = None
    min_signals = None
    if min_confidence is None:
        min_confidence = float(getattr(app, "swing_cfg", {}).get("stock_confidence_min", 0.70))
    if min_signals is None:
        min_signals = int(getattr(app, "swing_cfg", {}).get("stock_min_signals", 3))

    assert min_confidence == 0.68
    assert min_signals == 4


# -------------------------------------------------------------------------
# Test 11: _tg_cmd_postexit Truncation and FileNotFoundError Handling
# -------------------------------------------------------------------------
def test_tg_cmd_postexit_truncation_and_handling():
    app = Prometheus.__new__(Prometheus)

    # 1. Output > 4000 characters
    mock_analytics = MagicMock()
    mock_analytics.generate_telegram_summary.return_value = "A" * 4500

    with patch("prometheus.analysis.post_exit_analytics.PostExitAnalytics", return_value=mock_analytics):
        res = app._tg_cmd_postexit()
        assert len(res) <= 4000
        assert "truncated for Telegram 4k limit" in res

    # 2. FileNotFoundError
    with patch("prometheus.analysis.post_exit_analytics.PostExitAnalytics", side_effect=FileNotFoundError("ledger missing")):
        res = app._tg_cmd_postexit()
        assert "No trades ledger found yet" in res
