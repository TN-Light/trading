"""
Unit and integration tests for PostExitTracker and PostExitAnalytics.
Verifies:
  1. Registration and initial state.
  2. Milestone capture (5m, 15m, 30m, 60m).
  3. Premature shakeout detection vs capital preservation validation.
  4. MFE expansion beyond Target.
  5. Auto-unwatch lifecycle and SQLite persistence.
  6. PostExitAnalytics metric aggregation.
"""

from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import MagicMock
import time

from prometheus.papertrade.post_exit_tracker import PostExitTracker, PostExitWatchItem
from prometheus.papertrade.recorder import TradeRecorder
from prometheus.analysis.post_exit_analytics import PostExitAnalytics, PostExitMetrics
from prometheus.data.price_micro_cache import get_shared_micro_cache, MicroCacheEntry

IST = timezone(timedelta(hours=5, minutes=30))


def test_post_exit_registration(tmp_path):
    db_file = tmp_path / "test_ledger.sqlite"
    recorder = TradeRecorder(sqlite_path=str(db_file), csv_path=None)
    tracker = PostExitTracker(recorder=recorder, max_watch_seconds=3600)

    closed_trade = {
        "trade_id": "TRADE-PE-01",
        "symbol": "NIFTY 50",
        "instrument": "NIFTY26OCT24000CE",
        "underlying": "NIFTY",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 200.0,
        "exit_price": 185.0,
        "exit_reason": "stop_loss",
        "exit_time": datetime.now(IST),
        "target": 240.0,
        "stop_loss": 185.0,
        "entry_spot": 24000.0,
        "holding_duration_seconds": 600,
    }

    item = tracker.register_closed_trade(closed_trade)
    assert item is not None
    assert tracker.active_watch_count == 1
    assert item.trade_id == "TRADE-PE-01"
    assert item.status == "WATCHING"
    assert item.post_exit_mfe == 185.0
    assert item.post_exit_mae == 185.0

    # Verify initial record in SQLite
    obs = recorder.get_post_exit_observations()
    assert len(obs) == 1
    assert obs[0]["trade_id"] == "TRADE-PE-01"
    assert obs[0]["exit_reason"] == "stop_loss"


def test_post_exit_milestone_snapshots(tmp_path):
    recorder = TradeRecorder(sqlite_path=str(tmp_path / "test_ledger.sqlite"), csv_path=None)
    tracker = PostExitTracker(recorder=recorder, max_watch_seconds=3600)

    exit_time = datetime(2026, 10, 9, 10, 0, 0, tzinfo=IST)
    trade = {
        "trade_id": "TRADE-MILESTONE",
        "symbol": "SENSEX",
        "instrument": "SENSEX72000CE",
        "underlying": "SENSEX",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 300.0,
        "exit_price": 350.0,
        "exit_reason": "target",
        "exit_time": exit_time,
        "target": 350.0,
        "stop_loss": 270.0,
    }
    tracker.register_closed_trade(trade)

    # 1. Tick at T+2m (120s elapsed): neither 5m nor 15m reached
    tracker.on_tick("SENSEX72000CE", 355.0, timestamp=exit_time + timedelta(seconds=120))
    item = tracker._watches["TRADE-MILESTONE"]
    assert item.price_t5m is None
    assert item.post_exit_mfe == 355.0

    # 2. Tick at T+6m (360s elapsed): price_t5m captured
    tracker.on_tick("SENSEX72000CE", 360.0, timestamp=exit_time + timedelta(seconds=360))
    assert item.price_t5m == 360.0
    assert item.price_t15m is None

    # 3. Tick at T+16m (960s elapsed): price_t15m captured
    tracker.on_tick("SENSEX72000CE", 375.0, timestamp=exit_time + timedelta(seconds=960))
    assert item.price_t15m == 375.0
    assert item.price_t30m is None

    # 4. Tick at T+32m (1920s elapsed): price_t30m captured
    tracker.on_tick("SENSEX72000CE", 390.0, timestamp=exit_time + timedelta(seconds=1920))
    assert item.price_t30m == 390.0
    assert item.price_t60m is None

    # 5. Tick at T+61m (3660s elapsed): price_t60m captured and item completed
    tracker.on_tick("SENSEX72000CE", 410.0, timestamp=exit_time + timedelta(seconds=3660))
    assert tracker.active_watch_count == 0  # Completed and cleaned up

    # Verify SQLite finalized state
    obs = recorder.get_post_exit_observations()
    assert len(obs) == 1
    assert obs[0]["status"] == "COMPLETED"
    assert obs[0]["price_t5m"] == 360.0
    assert obs[0]["price_t15m"] == 375.0
    assert obs[0]["price_t30m"] == 390.0
    assert obs[0]["price_t60m"] == 410.0
    assert obs[0]["post_exit_mfe"] == 410.0


def test_premature_shakeout_detection(tmp_path):
    recorder = TradeRecorder(sqlite_path=str(tmp_path / "test_ledger.sqlite"), csv_path=None)
    tracker = PostExitTracker(recorder=recorder)

    exit_time = datetime(2026, 10, 9, 10, 0, 0, tzinfo=IST)
    trade = {
        "trade_id": "TRADE-SHAKEOUT",
        "symbol": "NIFTY 50",
        "instrument": "NIFTY24000CE",
        "underlying": "NIFTY",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 100.0,
        "exit_price": 85.0,
        "exit_reason": "stop_loss",
        "exit_time": exit_time,
        "target": 130.0,
        "stop_loss": 85.0,
    }
    tracker.register_closed_trade(trade)

    # Price dips to 80 (noise), then reverses sharply to 135 (reaching Target)
    tracker.on_tick("NIFTY24000CE", 80.0, timestamp=exit_time + timedelta(minutes=5))
    tracker.on_tick("NIFTY24000CE", 135.0, timestamp=exit_time + timedelta(minutes=20))

    item = tracker._watches["TRADE-SHAKEOUT"]
    assert item.target_reclaimed is True
    assert item.entry_reclaimed is True
    assert item.is_premature_shakeout is True
    assert item.post_exit_mae == 80.0
    assert item.post_exit_mfe == 135.0


def test_capital_preservation_sl_validation(tmp_path):
    recorder = TradeRecorder(sqlite_path=str(tmp_path / "test_ledger.sqlite"), csv_path=None)
    tracker = PostExitTracker(recorder=recorder)

    exit_time = datetime(2026, 10, 9, 10, 0, 0, tzinfo=IST)
    trade = {
        "trade_id": "TRADE-PRESERVE",
        "symbol": "NIFTY BANK",
        "instrument": "BANKNIFTY55000CE",
        "underlying": "BANKNIFTY",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 500.0,
        "exit_price": 450.0,
        "exit_reason": "stop_loss",
        "exit_time": exit_time,
        "target": 600.0,
        "stop_loss": 450.0,
    }
    tracker.register_closed_trade(trade)

    # Price collapses from 450 to 250 without ever reclaiming entry
    tracker.on_tick("BANKNIFTY55000CE", 380.0, timestamp=exit_time + timedelta(minutes=10))
    tracker.on_tick("BANKNIFTY55000CE", 250.0, timestamp=exit_time + timedelta(minutes=25))

    item = tracker._watches["TRADE-PRESERVE"]
    assert item.target_reclaimed is False
    assert item.entry_reclaimed is False
    assert item.is_premature_shakeout is False
    # SL saved 450 - 250 = 200 pts of catastrophic drawdown
    assert item.drawdown_saved_pts == 200.0


def test_target_exit_alpha_left_attribution(tmp_path):
    recorder = TradeRecorder(sqlite_path=str(tmp_path / "test_ledger.sqlite"), csv_path=None)
    tracker = PostExitTracker(recorder=recorder)

    exit_time = datetime(2026, 10, 9, 10, 0, 0, tzinfo=IST)
    trade = {
        "trade_id": "TRADE-RUNNER",
        "symbol": "SENSEX",
        "instrument": "SENSEX72200PE",
        "underlying": "SENSEX",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 200.0,
        "exit_price": 260.0,
        "exit_reason": "target",
        "exit_time": exit_time,
        "target": 260.0,
        "stop_loss": 170.0,
    }
    tracker.register_closed_trade(trade)

    # Contract runs from 260 to 340 post-exit
    tracker.on_tick("SENSEX72200PE", 340.0, timestamp=exit_time + timedelta(minutes=15))

    item = tracker._watches["TRADE-RUNNER"]
    assert item.post_exit_mfe == 340.0
    # Left 340 - 260 = 80 pts beyond target
    assert item.alpha_left_pts == 80.0


def test_micro_cache_polling(tmp_path):
    recorder = TradeRecorder(sqlite_path=str(tmp_path / "test_ledger.sqlite"), csv_path=None)
    tracker = PostExitTracker(recorder=recorder)

    cache = get_shared_micro_cache()
    cache.put(
        "SENSEX72500CE",
        MicroCacheEntry(
            price=275.0,
            bid=274.0,
            ask=276.0,
            open_interest=20000,
            volume=5000,
            timestamp=time.monotonic(),
            tradingsymbol="SENSEX72500CE",
        ),
    )

    trade = {
        "trade_id": "TRADE-CACHE-TEST",
        "symbol": "SENSEX",
        "instrument": "SENSEX72500CE",
        "underlying": "SENSEX",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 250.0,
        "exit_price": 260.0,
        "exit_reason": "target",
        "exit_time": datetime.now(IST),
        "target": 260.0,
    }
    tracker.register_closed_trade(trade)

    # Poll micro cache
    updates = tracker.poll_micro_cache()
    assert updates == 1
    assert tracker._watches["TRADE-CACHE-TEST"].post_exit_mfe == 275.0


def test_sqlite_persistence_and_analytics(tmp_path):
    db_file = str(tmp_path / "test_ledger.sqlite")
    recorder = TradeRecorder(sqlite_path=db_file, csv_path=None)
    tracker = PostExitTracker(recorder=recorder)

    # Populate 3 simulated closed trades
    exit_time = datetime(2026, 10, 9, 10, 0, 0, tzinfo=IST)
    # Trade 1: SL hit, but reversed to hit target (Premature Shakeout)
    tracker.register_closed_trade({
        "trade_id": "T1-SHAKEOUT",
        "symbol": "NIFTY 50",
        "instrument": "NIFTY24000CE",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 100.0,
        "exit_price": 85.0,
        "exit_reason": "stop_loss",
        "exit_time": exit_time,
        "target": 130.0,
        "stop_loss": 85.0,
    })
    tracker.on_tick("NIFTY24000CE", 135.0, timestamp=exit_time + timedelta(minutes=20))

    # Trade 2: SL hit, collapsed (Good SL)
    tracker.register_closed_trade({
        "trade_id": "T2-GOOD-SL",
        "symbol": "BANK NIFTY",
        "instrument": "BN50000CE",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 400.0,
        "exit_price": 360.0,
        "exit_reason": "stop_loss",
        "exit_time": exit_time,
        "target": 480.0,
        "stop_loss": 360.0,
    })
    tracker.on_tick("BN50000CE", 200.0, timestamp=exit_time + timedelta(minutes=20))

    # Trade 3: Target hit, ran another 40 pts
    tracker.register_closed_trade({
        "trade_id": "T3-TARGET",
        "symbol": "SENSEX",
        "instrument": "SX72000PE",
        "direction": "LONG",
        "tier": "B",
        "entry_price": 200.0,
        "exit_price": 250.0,
        "exit_reason": "target",
        "exit_time": exit_time,
        "target": 250.0,
        "stop_loss": 170.0,
    })
    tracker.on_tick("SX72000PE", 290.0, timestamp=exit_time + timedelta(minutes=20))

    # Run PostExitAnalytics
    analytics = PostExitAnalytics(db_path=db_file)
    metrics = analytics.compute_metrics()

    assert metrics.total_trades == 3
    assert metrics.sl_exits == 2
    assert metrics.premature_shakeouts == 1
    assert metrics.premature_shakeout_rate == 50.0
    assert metrics.target_exits == 1
    assert metrics.avg_alpha_left_pts == 40.0
    assert metrics.drawdown_saved_pts == 160.0  # 360 - 200 = 160

    # Summary card formatting
    summary = analytics.generate_telegram_summary()
    assert "POST-EXIT TRACKING FORENSICS" in summary
    assert "Premature Shakeouts:" in summary

    # Markdown report generation
    report_file = tmp_path / "report.md"
    report = analytics.generate_markdown_report(output_path=str(report_file))
    assert "# Post-Exit Performance Attribution & Forensic Report" in report
    assert report_file.exists()
