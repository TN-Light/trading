"""
prometheus/tests/test_lockout_sync_and_daily_alpha_brake.py
==========================================================
Tests for:
1. Synchronization of Telegram alerts and leaderboard ranking with Intraday Martingale Lockout.
2. Prevention of "EXECUTE LIVE" alerts on symbol/direction pairs stopped out earlier in the session.
3. Daily alpha goal lock and consecutive stop-out brake telemetry in LivePaperCapture.
4. Adaptive volatility ceiling in TargetCalibrator when VIX > 18.0.
"""

import pytest
from prometheus.paper_executor.live_bridge import LivePaperCapture, CaptureConfig
from prometheus.interface.telegram_bot import TelegramBot
from prometheus.signals.target_calibrator import calibrate_target_and_sl


class MockBotEngine:
    def __init__(self):
        pass


def test_live_bridge_is_direction_stopped_out():
    """Verify LivePaperCapture accurately flags stopped-out directions."""
    class DummyFeed:
        def get_ltp(self, inst): return 100.0

    cfg = CaptureConfig(enabled=True)
    capture = LivePaperCapture(ltp_source=DummyFeed(), config=cfg)

    # Initially empty
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE") is False

    # Simulate stop-out recorded with default Tier C
    capture._today_stopped_out_symbol_directions.add(("SENSEX", "BUY_CE"))
    # Default (Tier C) is stopped out
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE") is True
    # Explicit Tier C candidate is stopped out
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE", tier="C") is True
    # High-conviction Tier B / Tier S candidate is NOT stopped out by a Tier C stop-out!
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE", tier="B") is False
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE", tier="S") is False

    # Now simulate Tier B stop-out
    capture._today_stopped_out_symbol_directions.add(("SENSEX", "BUY_CE", "B"))
    # Now both Tier B and Tier C are stopped out
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE", tier="B") is True
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE", tier="S") is True
    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_CE", tier="C") is True

    assert capture.is_direction_stopped_out("SENSEX", direction="BUY_PE") is False
    assert capture.is_direction_stopped_out("NIFTY 50", direction="BUY_CE") is False

    # Also test via refined_signal dict
    sig_sensex_ce_b = {"symbol": "SENSEX", "action": "BUY_CE", "tier": "B"}
    sig_sensex_pe = {"symbol": "SENSEX", "action": "BUY_PE", "tier": "B"}
    assert capture.is_direction_stopped_out("SENSEX", refined_signal=sig_sensex_ce_b) is True
    assert capture.is_direction_stopped_out("SENSEX", refined_signal=sig_sensex_pe) is False


def test_telegram_alert_suppresses_execute_live_on_martingale_lockout():
    """Verify Telegram alert replaces 'EXECUTE LIVE' with 'MARTINGALE LOCKOUT ACTIVE'."""
    bot = TelegramBot(bot_token="dummy", chat_id="12345")
    sent_messages = []
    bot.send_message = lambda msg: sent_messages.append(msg)

    # 1. Normal Tier B signal (Rank #1) -> Should declare EXECUTE LIVE ON KITE
    normal_signal = {
        "symbol": "NIFTY 50",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "signal_score": 7.5,
        "entry_price": 150.0,
        "stop_loss": 130.0,
        "target": 180.0,
        "strike": 22800,
        "option_type": "CE",
        "expiry": "2026-10-29",
        "tier": "B",
        "leaderboard_rank": 1,
        "is_live_eligible": True,
        "is_martingale_locked_out": False,
    }
    bot.alert_new_signal(normal_signal, source="auto")
    assert len(sent_messages) == 1
    assert "ACTION: EXECUTE LIVE ON KITE" in sent_messages[0]
    assert "RANK #1 SIGNAL (PRIMARY EXECUTION)" in sent_messages[0]

    # 2. Martingale Locked-Out signal -> Must declare PAPER ONLY [MARTINGALE LOCKOUT ACTIVE]
    locked_signal = {
        "symbol": "SENSEX",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "signal_score": 7.5,
        "entry_price": 145.0,
        "stop_loss": 95.5,
        "target": 210.2,
        "strike": 73700,
        "option_type": "CE",
        "expiry": "2026-10-29",
        "tier": "B",
        "leaderboard_rank": 1,
        "is_live_eligible": False,
        "is_martingale_locked_out": True,
    }
    bot.alert_new_signal(locked_signal, source="auto")
    assert len(sent_messages) == 2
    alert_text = sent_messages[1]
    assert "ACTION: EXECUTE LIVE ON KITE" not in alert_text
    assert "ACTION: PAPER ONLY — [MARTINGALE LOCKOUT ACTIVE]" in alert_text
    assert "Directional Martingale Lockout" in alert_text
    assert "Per GEMINI.md Rule 4" in alert_text


def test_telegram_spread_alert_suppresses_execute_live_on_martingale_lockout():
    """Verify Telegram credit spread alert also respects Martingale Lockout."""
    bot = TelegramBot(bot_token="dummy", chat_id="12345")
    sent_messages = []
    bot.send_message = lambda msg: sent_messages.append(msg)

    locked_spread = {
        "symbol": "SENSEX",
        "action": "SELL_CALL_SPREAD",
        "strategy_type": "credit_spread",
        "net_credit": 25.0,
        "target_decay_price": 7.5,
        "hard_sl_price": 37.5,
        "leaderboard_rank": 1,
        "is_live_eligible": False,
        "is_martingale_locked_out": True,
        "legs": [
            {"action": "SELL", "tradingsymbol": "SENSEX26OCT74000CE", "premium": 35.0, "strike": 74000, "option_type": "CE", "is_hedge": False},
            {"action": "BUY", "tradingsymbol": "SENSEX26OCT74200CE", "premium": 10.0, "strike": 74200, "option_type": "CE", "is_hedge": True},
        ]
    }
    bot.alert_new_signal(locked_spread, source="auto")
    assert len(sent_messages) == 1
    alert_text = sent_messages[0]
    assert "ACTION: EXECUTE LIVE ON KITE" not in alert_text
    assert "ACTION: PAPER ONLY — [MARTINGALE LOCKOUT ACTIVE]" in alert_text


def test_target_calibrator_adaptive_volatility_ceiling():
    """Verify TargetCalibrator expands allowable SL ceiling during high-VIX regimes."""
    # Normal VIX (15.5): Nifty ceiling is strictly 25.0 pts
    res_normal = calibrate_target_and_sl(
        symbol="NIFTY 50",
        spot_price=22800.0,
        spot_atr=80.0,
        opt_ltp=150.0,
        edge_score=7.0,
        tier="B",
        vix=15.5,
        structural_sl_pts=30.0,
    )
    # SL 30.0 > 25.0 cap in normal VIX -> Rejected
    assert res_normal.is_rejected is True

    # High VIX (24.0) with spot_atr=90.0:
    # vol_sl_cap = 0.40 * 90.0 = 36.0 pts. SL 30.0 <= 36.0 cap -> Accepted!
    res_high_vix = calibrate_target_and_sl(
        symbol="NIFTY 50",
        spot_price=22800.0,
        spot_atr=90.0,
        opt_ltp=150.0,
        edge_score=7.0,
        tier="B",
        vix=24.0,
        structural_sl_pts=30.0,
    )
    assert res_high_vix.is_rejected is False
    assert res_high_vix.sl_pts == 30.0


def test_live_bridge_today_net_pnl_and_consecutive_losses():
    """Verify get_today_realized_net_pnl and get_today_consecutive_losses query live ledger."""
    class DummyFeed:
        def get_ltp(self, inst): return 100.0

    cfg = CaptureConfig(enabled=True)
    capture = LivePaperCapture(ltp_source=DummyFeed(), config=cfg)
    pnl = capture.get_today_realized_net_pnl()
    consec = capture.get_today_consecutive_losses()
    # Today Oct 29 has trades in live_ledger.sqlite
    assert isinstance(pnl, float)
    assert isinstance(consec, int)
    assert pnl == -615.0
    assert consec == 1


def test_tier_b_candidate_not_locked_out_by_tier_c_stopout():
    """Verify that a Tier B Institutional Trend Day candidate is NOT locked out
    when only an earlier Tier C shadow trade stopped out on the same symbol/direction."""
    class DummyFeed:
        def get_ltp(self, inst): return 100.0

    cfg = CaptureConfig(enabled=True)
    capture = LivePaperCapture(ltp_source=DummyFeed(), config=cfg)

    # 1. Simulate Tier C stop-out recorded earlier
    capture._today_stopped_out_symbol_directions.add(("SENSEX", "BUY_CE", "C"))

    # 2. Incoming candidate has Tier B setup (e.g. Institutional Trend Day)
    cand_b = {
        "symbol": "SENSEX",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "signal_score": 7.5,
        "adx": 27.1,
        "is_institutional_trend_day": True,
        "entry_price": 145.0,
        "stop_loss": 95.5,
        "target": 210.2,
        "tier": "B",
        "is_live_eligible": True,
    }

    # Must NOT be locked out!
    is_locked = capture.is_direction_stopped_out("SENSEX", refined_signal=cand_b, tier="B")
    assert is_locked is False

    # 3. But an incoming Tier C candidate on the same symbol IS locked out!
    cand_c = {
        "symbol": "SENSEX",
        "action": "BUY_CE",
        "strategy": "PriceAction_Momentum",
        "signal_score": 6.5,
        "tier": "C",
        "is_live_eligible": False,
    }
    is_locked_c = capture.is_direction_stopped_out("SENSEX", refined_signal=cand_c, tier="C")
    assert is_locked_c is True


def test_tier_aware_consecutive_losses_ignores_tier_c():
    """Verify get_today_consecutive_losses with tier filter ignores Tier C shadow trades."""
    class DummyFeed:
        def get_ltp(self, inst): return 100.0

    cfg = CaptureConfig(enabled=True)
    capture = LivePaperCapture(ltp_source=DummyFeed(), config=cfg)
    
    # Across all trades, today has 1 consecutive loss at the end (FinNifty Tier C)
    total_consec = capture.get_today_consecutive_losses()
    assert total_consec == 1

    # But among Tier S / Tier A / Tier B trades, FinNifty Tier C is ignored!
    # NIFTY 50 Tier B was the last Tier B trade (which was stopped by inactivity kill)
    tier_sb_consec = capture.get_today_consecutive_losses(tiers=("S", "A", "B"))
    assert tier_sb_consec == 1


def test_tier_segregated_realized_net_pnl(tmp_path):
    """Verify get_today_realized_net_pnl segregates Tier C experimental losses from live tiers."""
    import sqlite3
    class DummyFeed:
        def get_ltp(self, inst): return 100.0

    test_db = tmp_path / "test_ledger.sqlite"
    with sqlite3.connect(str(test_db)) as conn:
        conn.execute("""
            CREATE TABLE paper_trades (
                entry_time TEXT,
                tier TEXT,
                net_pnl REAL
            )
        """)
        from datetime import datetime
        import pytz
        today_str = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
        # Live Tier B win
        conn.execute("INSERT INTO paper_trades VALUES (?, 'B', 3080.0)", (today_str,))
        # Experimental Tier C loss
        conn.execute("INSERT INTO paper_trades VALUES (?, 'C', -5000.0)", (today_str,))
        conn.commit()

    cfg = CaptureConfig(enabled=True, sqlite_path=str(test_db))
    capture = LivePaperCapture(ltp_source=DummyFeed(), config=cfg)

    # Total blended includes Tier C (-1920.0)
    assert capture.get_today_realized_net_pnl() == 3080.0 - 5000.0

    # Live-only (S, A, B) preserves pure +3080.0 profit!
    live_pnl = capture.get_today_realized_net_pnl(tiers=("S", "A", "B"))
    assert live_pnl == 3080.0
