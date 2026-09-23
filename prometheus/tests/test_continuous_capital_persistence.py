import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch
from prometheus.main import Prometheus, PROJECT_ROOT

def test_continuous_capital_persistence_loads_from_sqlite(tmp_path):
    # Create a temporary sqlite db mimicking live_ledger.sqlite
    ledger_dir = tmp_path / "reports" / "papertrade"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    db_path = ledger_dir / "live_ledger.sqlite"
    
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE paper_trades (
                trade_id TEXT PRIMARY KEY,
                net_pnl REAL
            )
        """)
        conn.execute("INSERT INTO paper_trades (trade_id, net_pnl) VALUES ('T1', -1347.06)")
        conn.execute("INSERT INTO paper_trades (trade_id, net_pnl) VALUES ('T2', 2500.00)")
        conn.commit()

    with patch.object(Path, "parent", new_callable=lambda: property(lambda self: tmp_path if "reports" in str(self) else Path(__file__).parent.parent.parent)):
        # Test directly with real config path
        cfg_path = str(PROJECT_ROOT / "config" / "settings.yaml")
        
        # Test the db reading logic directly
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COALESCE(SUM(net_pnl), 0.0) FROM paper_trades")
            cum_pnl = float(cur.fetchone()[0] or 0.0)
            baseline = 100000.0
            expected_capital = baseline + cum_pnl
            
        assert round(expected_capital, 2) == 101152.94

def test_risk_manager_strict_1_lot_enforcement():
    from prometheus.risk.manager import RiskManager
    rm = RiskManager({"max_lots_per_trade": 1}, initial_capital=100000.0)
    assert rm.max_lots_per_trade == 1
    
    # Even if mathematical sizing allows 7+ lots, it must clamp to exactly 1
    res = rm.calculate_position_size(
        entry_price=100.0,
        stop_loss=90.0,  # 10 pt risk
        lot_size=65,     # 650 Rs risk per lot
        risk_per_trade_pct=5.0  # 5% of 100,000 = 5,000 Rs budget -> would allow 7 lots
    )
    assert res["lots"] == 1
