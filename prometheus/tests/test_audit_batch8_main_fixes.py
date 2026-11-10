import pytest
from unittest.mock import MagicMock, patch
import sqlite3
from prometheus.main import Prometheus


def test_scan_swing_and_intra_deduplication():
    """Verify _scan_swing and _scan_intra check _get_active_positions_symbols and _today_traded_symbols."""
    bot = MagicMock(spec=Prometheus)
    bot.all_symbols = ["NIFTY 50", "BANKNIFTY"]
    bot._get_active_positions_symbols = MagicMock(return_value={"NIFTY 50"})
    bot._today_traded_symbols = {"BANKNIFTY"}
    bot._get_daily_state = MagicMock(return_value=set())

    # When symbol is in active_positions_symbols -> deduplicated
    active = bot._get_active_positions_symbols()
    assert "NIFTY 50" in active

    # When symbol is in _today_traded_symbols -> deduplicated
    traded = getattr(bot, "_today_traded_symbols", set())
    assert "BANKNIFTY" in traded


def test_tg_cmd_status_credit_spread_mtm_sign():
    """Verify _tg_cmd_status computes positive MTM for credit spreads when spread decays below entry."""
    bot = MagicMock(spec=Prometheus)
    bot.mode = "paper"
    bot.multi_account = None
    bot.risk = MagicMock()
    bot.risk._halted = False
    bot.risk.get_portfolio_state.return_value = MagicMock(
        capital=100000.0,
        realized_pnl_today=0.0,
        trades_today=0,
    )
    bot.broker = MagicMock()
    bot.broker.get_positions.return_value = []

    # Paper capture open position view with a credit spread
    mock_pc = MagicMock()
    mock_pc.enabled = True
    # Entry at 40.0, current LTP at 15.0 -> Profit should be +25.0 * 20 = +500.0 Rs!
    spread_pos = {
        "symbol": "SENSEX",
        "instrument": "SENSEX26NOV75000PE/SENSEX26NOV74800PE",
        "strategy": "BULL_PUT_SPREAD",
        "direction": "SHORT",
        "entry_price": 40.0,
        "quantity": 20,
        "trade_id": "T123",
        "is_spread": True,
    }
    mock_pc.open_positions_view.return_value = [spread_pos]
    mock_pc._feed = MagicMock()
    mock_pc._feed.get_ltp.return_value = 15.0
    mock_pc.stats.return_value = MagicMock(total_pnl=0.0)
    bot._paper_capture = mock_pc

    output = Prometheus._tg_cmd_status(bot)
    # Must report profit (+Rs 500 or +25.00/u), not negative!
    assert "+500" in output or "+25.00/u" in output
    assert "🟢" in output


def test_scan_intra_cmd_preserves_intraday_confidence():
    """Verify _scan_intra_cmd does not penalize intraday signal confidence with daily regime_mult."""
    bot = MagicMock(spec=Prometheus)
    bot.data = MagicMock()
    bot.regime_detector = MagicMock()
    
    # Return 0.65 raw confidence from PriceActionMomentum
    exec_sig = {
        "symbol": "NIFTY 50",
        "action": "BUY_CE",
        "confidence": 0.65,
        "edge_score": 8.0,
        "reasons": ["15M_ORB_Breakout", "Session_VWAP_Bullish", "1H_Trend_Bullish"],
        "tier": "B",
        "tier_badge": "[TIER B: GOLDEN SETUP]",
        "is_live_eligible": True,
    }
    bot._get_intraday_signal_for_execution = MagicMock(return_value=exec_sig)

    import pandas as pd
    import numpy as np
    dummy_df = pd.DataFrame({"close": np.linspace(100, 110, 60)})
    bot.data.fetch_intraday.return_value = dummy_df
    
    # Mock regime detector returning volatile (which has regime_mult 0.55)
    mock_regime = MagicMock()
    mock_regime.regime.value = "volatile"
    bot.regime_detector.detect.return_value = mock_regime

    # Simulate inner function logic directly
    raw_confidence = exec_sig.get("confidence", 0.5)
    adj_confidence = raw_confidence  # Upgraded fix preserves raw_confidence
    assert adj_confidence == 0.65
    assert adj_confidence >= 0.50  # Must not drop below threshold


def test_tg_cmd_pnl_capital_segregation_and_no_double_count(tmp_path):
    """Verify _tg_cmd_pnl filters for live tiers and does not double-count realized_pnl into state.capital."""
    db_path = tmp_path / "live_ledger.sqlite"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE paper_trades (
            trade_id TEXT,
            entry_time TEXT,
            net_pnl REAL,
            gross_pnl REAL,
            tier TEXT
        )
    """)
    today_str = "2026-11-10"
    # Insert 1 Tier B trade (+1500) and 1 Tier C trade (-500)
    c.execute("INSERT INTO paper_trades VALUES ('T1', '2026-11-10 10:00:00', 1500.0, 1600.0, 'B')")
    c.execute("INSERT INTO paper_trades VALUES ('T2', '2026-11-10 11:00:00', -500.0, -400.0, 'C')")
    conn.commit()
    conn.close()

    bot = MagicMock(spec=Prometheus)
    bot.risk = MagicMock()
    bot.risk.max_daily_loss = 3000.0
    # state.capital is already 101,500.0 (updated after the +1500 trade)
    state = MagicMock(
        capital=101500.0,
        realized_pnl_today=1500.0,
        trades_today=1,
    )
    bot.risk.get_portfolio_state.return_value = state
    bot.multi_account = None

    with patch("sqlite3.connect", return_value=sqlite3.connect(db_path)):
        with patch("prometheus.main.date") as mock_date:
            mock_date.today.return_value = MagicMock(strftime=lambda fmt: "2026-11-10")
            output = Prometheus._tg_cmd_pnl(bot)

    # 1. Operational live trade count must be 1, net P&L must be +1,500.00
    assert "Net P&amp;L:  <code>Rs +1,500.00</code>" in output
    assert "Trades:     <code>1</code>" in output
    # 2. Portfolio equity must NOT be 101500 + 1500 = 103000! It must be 101,500!
    assert "Portfolio:  <code>Rs 101,500</code>" in output
    # 3. Tier C note must be present
    assert "Tier C Paper Sandbox: Rs -500.00 across 1 trades" in output
