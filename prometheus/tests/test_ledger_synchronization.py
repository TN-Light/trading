"""
prometheus/tests/test_ledger_synchronization.py
=============================================
Automated Continuous Integration Test for Ledger Integrity and Multi-Leg Spreads.

Enforces GEMINI.md Section 1 & Section 2 invariants:
  - SQLite vs CSV zero-drift guarantee (COUNT and SUM(net_pnl) match 100%).
  - daily_performance_ledger.csv matches SQLite daily aggregations.
  - Multi-leg credit spread cost buffers cover 2x statutory fees.
"""

import os
import sqlite3
import csv
from datetime import datetime
import pytest
from prometheus.papertrade.position_tracker import PositionTracker
from prometheus.papertrade.types import Position, Direction

SQLITE_PATH = os.path.abspath("reports/papertrade/live_ledger.sqlite")
CSV_PATH = os.path.abspath("reports/papertrade/live_ledger.csv")
DAILY_CSV_PATH = os.path.abspath("reports/papertrade/daily_performance_ledger.csv")


def test_sqlite_csv_exact_match():
    """Asserts that live_ledger.sqlite and live_ledger.csv match 100%."""
    assert os.path.exists(SQLITE_PATH), f"Missing {SQLITE_PATH}"
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
    
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("SELECT trade_id, net_pnl FROM paper_trades ORDER BY entry_time ASC")
    sqlite_rows = c.fetchall()
    conn.close()
    
    csv_rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            csv_rows.append((r["trade_id"], float(r.get("net_pnl", 0) or 0)))
            
    assert len(sqlite_rows) == len(csv_rows), f"Row count mismatch: SQLite {len(sqlite_rows)} vs CSV {len(csv_rows)}"
    
    total_sql_net = sum(r[1] for r in sqlite_rows)
    total_csv_net = sum(r[1] for r in csv_rows)
    assert abs(total_sql_net - total_csv_net) < 1e-4, f"Net PnL mismatch: SQL {total_sql_net:.2f} vs CSV {total_csv_net:.2f}"
    
    # Sort both by trade_id so comparison is independent of entry_time vs exit_time write order
    sqlite_rows = sorted(sqlite_rows, key=lambda r: r[0])
    csv_rows = sorted(csv_rows, key=lambda r: r[0])

    # Assert every trade_id matches in sequence
    for (sql_id, sql_net), (csv_id, csv_net) in zip(sqlite_rows, csv_rows):
        assert sql_id == csv_id, f"Trade ID mismatch: {sql_id} vs {csv_id}"
        assert abs(sql_net - csv_net) < 1e-4, f"Trade PnL mismatch for {sql_id}: {sql_net} vs {csv_net}"


def test_daily_performance_matches_sqlite():
    """Asserts that daily_performance_ledger.csv reflects actual SQLite daily metrics."""
    assert os.path.exists(DAILY_CSV_PATH), f"Missing {DAILY_CSV_PATH}"
    
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT 
            substr(entry_time, 1, 10) as dt,
            count(*) as total_trades,
            sum(case when net_pnl > 0 then 1 else 0 end) as wins,
            sum(case when net_pnl <= 0 then 1 else 0 end) as losses,
            sum(net_pnl) as net_pnl
        FROM paper_trades
        GROUP BY dt
    """)
    sqlite_daily = {r[0]: r for r in c.fetchall()}
    conn.close()
    
    csv_daily = {}
    with open(DAILY_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dt = r["Date"].strip()
            csv_daily[dt] = (
                dt,
                int(r["Total_Trades"]),
                int(r["Wins"]),
                int(r["Losses"]),
                float(r["Net_PnL"])
            )
            
    # Check that all SQLite dates exist in CSV and numbers match
    for dt, (s_dt, s_tot, s_w, s_l, s_net) in sqlite_daily.items():
        assert dt in csv_daily, f"Date {dt} missing from daily_performance_ledger.csv"
        c_dt, c_tot, c_w, c_l, c_net = csv_daily[dt]
        assert s_tot == c_tot, f"Trade count mismatch on {dt}: SQLite {s_tot} vs CSV {c_tot}"
        assert s_w == c_w, f"Wins mismatch on {dt}: SQLite {s_w} vs CSV {c_w}"
        assert s_l == c_l, f"Losses mismatch on {dt}: SQLite {s_l} vs CSV {c_l}"
        assert abs(s_net - c_net) < 1e-2, f"Net PnL mismatch on {dt}: SQLite {s_net:.2f} vs CSV {c_net:.2f}"


def test_multi_leg_spread_cost_buffer_calibration():
    """Verifies that 2-leg credit spreads enforce doubled cost buffers covering round-trip fees."""
    from unittest.mock import MagicMock
    tracker = PositionTracker(fill_sim=MagicMock())
    
    # 1. FinNifty 2-leg spread (60 qty)
    fin_spread = Position(
        trade_id="TEST-FIN-SPREAD",
        symbol="NIFTY FIN SERVICE",
        instrument="FINNIFTY27OCT2625550CE/FINNIFTY27OCT2625700CE",
        underlying="FINNIFTY",
        direction=Direction.SHORT,
        quantity=60,
        entry_price=28.25,
        entry_time=datetime(2026, 10, 19, 13, 24, 30),
        stop_loss=42.15,
        target=8.43,
        max_bars=10,
        strategy="Hedged_Credit_Spread"
    )
    
    # Check trailing ratchet at 35% decay (LTP = 18.00, entry 28.25 -> decay 10.25 pts)
    tracker._maybe_advance_trailing_stop(fin_spread, current_price=18.0)
    assert fin_spread.breakeven_set is True
    # Cost buffer for FinNifty spread is 4.20 pts
    # be_sl = 28.25 - 4.20 = 24.05, min(24.05, 28.25 * 0.90 = 25.43) -> 24.05
    assert fin_spread.stop_loss <= 24.05, f"Expected spread SL <= 24.05, got {fin_spread.stop_loss}"
    
    # 2. Bank Nifty 2-leg spread (15 qty)
    bn_spread = Position(
        trade_id="TEST-BN-SPREAD",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY27OCT2656300CE/BANKNIFTY27OCT2656600CE",
        underlying="BANKNIFTY",
        direction=Direction.SHORT,
        quantity=15,
        entry_price=72.38,
        entry_time=datetime(2026, 10, 19, 13, 24, 27),
        stop_loss=103.43,
        target=20.69,
        max_bars=10,
        strategy="Hedged_Credit_Spread"
    )
    tracker._maybe_advance_trailing_stop(bn_spread, current_price=45.0)
    assert bn_spread.breakeven_set is True
    # Cost buffer for BN spread is 8.50 pts
    # be_sl = 72.38 - 8.50 = 63.88
    assert bn_spread.stop_loss <= 63.88, f"Expected BN spread SL <= 63.88, got {bn_spread.stop_loss}"
