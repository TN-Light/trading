# ============================================================================
# PROMETHEUS — Data Engine: SQLite Storage Layer
# ============================================================================
"""
Persistent storage for OHLCV data, options chain snapshots, signals, and trades.
Uses SQLite — zero server setup, works everywhere.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime, date
from typing import Optional, List
from contextlib import contextmanager


class DataStore:
    """SQLite-based data storage for Prometheus."""

    def __init__(self, db_path: str = "data/prometheus.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        """Thread-safe connection context manager."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create all tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript("""
                -- OHLCV candle data
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    oi INTEGER DEFAULT 0,
                    UNIQUE(symbol, interval, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_ts
                    ON ohlcv(symbol, interval, timestamp);

                -- Options chain snapshots
                CREATE TABLE IF NOT EXISTS options_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    strike REAL NOT NULL,
                    option_type TEXT NOT NULL,
                    ltp REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER DEFAULT 0,
                    oi INTEGER DEFAULT 0,
                    oi_change INTEGER DEFAULT 0,
                    iv REAL,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    UNIQUE(timestamp, symbol, expiry, strike, option_type)
                );
                CREATE INDEX IF NOT EXISTS idx_options_symbol_expiry
                    ON options_chain(symbol, expiry, timestamp);

                -- Trade log
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    strategy TEXT,
                    signal_strength REAL,
                    stop_loss REAL,
                    target REAL,
                    status TEXT DEFAULT 'open',
                    exit_price REAL,
                    exit_timestamp TEXT,
                    pnl REAL,
                    notes TEXT
                );

                -- Signal log
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strength REAL NOT NULL,
                    strategy TEXT,
                    timeframe TEXT,
                    entry_price REAL,
                    stop_loss REAL,
                    target REAL,
                    reasoning TEXT,
                    acted_on INTEGER DEFAULT 0
                );

                -- Daily portfolio snapshot
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    capital REAL NOT NULL,
                    deployed REAL NOT NULL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    total_pnl REAL,
                    open_positions INTEGER,
                    win_count INTEGER DEFAULT 0,
                    loss_count INTEGER DEFAULT 0,
                    max_drawdown REAL DEFAULT 0
                );

                -- VIX history
                CREATE TABLE IF NOT EXISTS vix_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL
                );

                -- FII/DII data
                CREATE TABLE IF NOT EXISTS fii_dii (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    fii_buy REAL,
                    fii_sell REAL,
                    fii_net REAL,
                    dii_buy REAL,
                    dii_sell REAL,
                    dii_net REAL,
                    fii_index_futures_long INTEGER,
                    fii_index_futures_short INTEGER,
                    fii_index_options_long INTEGER,
                    fii_index_options_short INTEGER
                );

                -- Managed positions (live trailing stop state persistence)
                CREATE TABLE IF NOT EXISTS managed_positions (
                    position_id TEXT PRIMARY KEY,
                    tradingsymbol TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    entry_premium REAL NOT NULL,
                    initial_sl REAL NOT NULL,
                    current_sl REAL NOT NULL,
                    target REAL NOT NULL,
                    sl_order_id TEXT,
                    entry_time TEXT NOT NULL,
                    status TEXT DEFAULT 'open',
                    realized_pnl REAL DEFAULT 0,
                    breakeven_set INTEGER DEFAULT 0,
                    trailing_activated INTEGER DEFAULT 0,
                    trailing_stage2 INTEGER DEFAULT 0,
                    trailing_stage3 INTEGER DEFAULT 0,
                    premium_hwm REAL DEFAULT 0,
                    entry_bar_count INTEGER DEFAULT 0,
                    max_bars INTEGER DEFAULT 7,
                    breakeven_ratio REAL DEFAULT 0.6,
                    risk_distance REAL DEFAULT 0,
                    entry_orders_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    # -----------------------------------------------------------------------
    # OHLCV Data
    # -----------------------------------------------------------------------
    def save_ohlcv(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save OHLCV dataframe to database."""
        with self._connection() as conn:
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, interval, timestamp, open, high, low, close, volume, oi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, interval, str(row.get("timestamp", row.name)),
                    row["open"], row["high"], row["low"], row["close"],
                    int(row.get("volume", 0)), int(row.get("oi", 0))
                ))

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50000
    ) -> pd.DataFrame:
        """Retrieve OHLCV data as DataFrame."""
        query = "SELECT * FROM ohlcv WHERE symbol = ? AND interval = ?"
        params: list = [symbol, interval]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # -----------------------------------------------------------------------
    # Options Chain
    # -----------------------------------------------------------------------
    def save_options_chain(self, df: pd.DataFrame):
        """Save options chain snapshot."""
        with self._connection() as conn:
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO options_chain
                    (timestamp, symbol, expiry, strike, option_type,
                     ltp, bid, ask, volume, oi, oi_change, iv,
                     delta, gamma, theta, vega)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row["timestamp"]), row["symbol"], str(row["expiry"]),
                    row["strike"], row["option_type"],
                    row.get("ltp"), row.get("bid"), row.get("ask"),
                    int(row.get("volume", 0)), int(row.get("oi", 0)),
                    int(row.get("oi_change", 0)), row.get("iv"),
                    row.get("delta"), row.get("gamma"),
                    row.get("theta"), row.get("vega")
                ))

    def get_options_chain(
        self,
        symbol: str,
        expiry: str,
        timestamp: Optional[str] = None
    ) -> pd.DataFrame:
        """Get options chain data."""
        if timestamp:
            query = """
                SELECT * FROM options_chain
                WHERE symbol = ? AND expiry = ? AND timestamp = ?
                ORDER BY strike, option_type
            """
            params = [symbol, expiry, timestamp]
        else:
            query = """
                SELECT * FROM options_chain
                WHERE symbol = ? AND expiry = ?
                AND timestamp = (SELECT MAX(timestamp) FROM options_chain WHERE symbol = ? AND expiry = ?)
                ORDER BY strike, option_type
            """
            params = [symbol, expiry, symbol, expiry]

        with self._connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # -----------------------------------------------------------------------
    # Trades
    # -----------------------------------------------------------------------
    def log_trade(self, trade: dict):
        """Log a trade to the database."""
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO trades
                (timestamp, symbol, instrument, action, quantity, price,
                 order_type, strategy, signal_strength, stop_loss, target, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("timestamp", str(datetime.now())),
                trade["symbol"], trade["instrument"], trade["action"],
                trade["quantity"], trade["price"], trade["order_type"],
                trade.get("strategy"), trade.get("signal_strength"),
                trade.get("stop_loss"), trade.get("target"), trade.get("notes")
            ))

    def close_trade(self, trade_id: int, exit_price: float, pnl: float):
        """Close a trade with exit price and P&L."""
        with self._connection() as conn:
            conn.execute("""
                UPDATE trades SET
                    status = 'closed',
                    exit_price = ?,
                    exit_timestamp = ?,
                    pnl = ?
                WHERE id = ?
            """, (exit_price, str(datetime.now()), pnl, trade_id))

    def get_open_trades(self) -> pd.DataFrame:
        """Get all open trades."""
        with self._connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp",
                conn
            )

    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history for last N days."""
        cutoff = str(datetime.now().date() - pd.Timedelta(days=days))
        with self._connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp",
                conn, params=[cutoff]
            )

    # -----------------------------------------------------------------------
    # Signals
    # -----------------------------------------------------------------------
    def log_signal(self, signal: dict):
        """Log a generated signal."""
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO signals
                (timestamp, symbol, signal_type, direction, strength,
                 strategy, timeframe, entry_price, stop_loss, target, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get("timestamp", str(datetime.now())),
                signal["symbol"], signal["signal_type"], signal["direction"],
                signal["strength"], signal.get("strategy"),
                signal.get("timeframe"), signal.get("entry_price"),
                signal.get("stop_loss"), signal.get("target"),
                signal.get("reasoning")
            ))

    # -----------------------------------------------------------------------
    # Portfolio Snapshots
    # -----------------------------------------------------------------------
    def save_portfolio_snapshot(self, snapshot: dict):
        """Save daily portfolio snapshot."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO portfolio_snapshots
                (date, capital, deployed, unrealized_pnl, realized_pnl,
                 total_pnl, open_positions, win_count, loss_count, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot["date"], snapshot["capital"], snapshot["deployed"],
                snapshot.get("unrealized_pnl", 0), snapshot.get("realized_pnl", 0),
                snapshot.get("total_pnl", 0), snapshot.get("open_positions", 0),
                snapshot.get("win_count", 0), snapshot.get("loss_count", 0),
                snapshot.get("max_drawdown", 0)
            ))

    def get_equity_curve(self) -> pd.DataFrame:
        """Get portfolio equity curve over time."""
        with self._connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM portfolio_snapshots ORDER BY date",
                conn
            )

    # -----------------------------------------------------------------------
    # Daily stats
    # -----------------------------------------------------------------------
    def get_daily_pnl(self, trade_date: Optional[str] = None) -> float:
        """Get total P&L for a given day."""
        if trade_date is None:
            trade_date = str(date.today())
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE date(timestamp) = ? AND status = 'closed'",
                [trade_date]
            )
            return cursor.fetchone()[0]

    # -----------------------------------------------------------------------
    # Managed Positions (live trailing stop state persistence)
    # -----------------------------------------------------------------------
    def save_position_state(self, state: dict):
        """Upsert a managed position's trailing stop state."""
        now = str(datetime.now())
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO managed_positions
                (position_id, tradingsymbol, symbol, direction, strategy,
                 entry_premium, initial_sl, current_sl, target, sl_order_id,
                 entry_time, status,
                 breakeven_set, trailing_activated, trailing_stage2,
                 trailing_stage3, premium_hwm,
                 entry_bar_count, max_bars, breakeven_ratio, risk_distance,
                 entry_orders_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM managed_positions WHERE position_id = ?), ?),
                        ?)
            """, (
                state["position_id"], state.get("tradingsymbol", ""),
                state.get("symbol", ""), state.get("direction", ""),
                state.get("strategy", ""),
                state.get("entry_premium", 0), state.get("initial_sl", 0),
                state.get("current_sl", 0), state.get("target", 0),
                state.get("sl_order_id", ""), state.get("entry_time", now),
                state.get("status", "open"),
                int(state.get("breakeven_set", False)),
                int(state.get("trailing_activated", False)),
                int(state.get("trailing_stage2", False)),
                int(state.get("trailing_stage3", False)),
                state.get("premium_hwm", 0),
                state.get("entry_bar_count", 0), state.get("max_bars", 7),
                state.get("breakeven_ratio", 0.6), state.get("risk_distance", 0),
                state.get("entry_orders_json", ""),
                state["position_id"], now,
                now,
            ))

    def load_open_positions(self) -> List[dict]:
        """Load all open managed positions for crash recovery."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM managed_positions WHERE status = 'open'"
            )
            return [dict(row) for row in cursor.fetchall()]

    def close_position_state(self, position_id: str, pnl: float = 0):
        """Mark a managed position as closed."""
        with self._connection() as conn:
            conn.execute("""
                UPDATE managed_positions
                SET status = 'closed', realized_pnl = ?, updated_at = ?
                WHERE position_id = ?
            """, (pnl, str(datetime.now()), position_id))
