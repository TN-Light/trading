"""
Persists every paper-trade to SQLite + CSV.

Schema mirrors the production ``prometheus/data/store.py`` trades-table
convention but adds the columns required for off-line analysis
(entry_time, exit_time, holding_duration_seconds, return_pct, costs, etc.).
Keeps SQLite + CSV in sync so the user can pick whichever is convenient.

CSV is the human-readable side (open in Excel). SQLite is the programmatic
side — supports multi-session queries and resumability.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Optional, List

from prometheus.papertrade.types import PaperTrade, TradeStats
from prometheus.utils.logger import logger


# Column order — kept explicit so CSV output is stable.
TRADE_COLUMNS = [
    "trade_id",
    "symbol",
    "instrument",
    "underlying",
    "direction",
    "quantity",
    "entry_price",
    "exit_price",
    "entry_time",
    "exit_time",
    "exit_reason",
    "gross_pnl",
    "costs",
    "net_pnl",
    "return_pct",
    "holding_duration_seconds",
    "strategy",
    "signal_score",
    "signal_confidence",
    "stop_loss",
    "target",
]


class TradeRecorder:
    """Append-only recorder for closed PaperTrades + stats snapshots.

    Args:
        sqlite_path: optional path to a SQLite DB; creates one if missing.
            Pass ``None`` to disable SQLite persistence.
        csv_path: optional path to a CSV file; writes/appends one row per
            trade. Pass ``None`` to disable CSV persistence.
    """

    def __init__(
        self,
        sqlite_path: Optional[str] = "reports/papertrade_trades.sqlite",
        csv_path: Optional[str] = "reports/papertrade_trades.csv",
    ):
        self.sqlite_path = str(sqlite_path) if sqlite_path else None
        self.csv_path = str(csv_path) if csv_path else None
        self._db: Optional[sqlite3.Connection] = None
        if self.sqlite_path:
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._create_schema()
        if self.csv_path:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            # Create header if file doesn't exist yet
            if not Path(self.csv_path).exists() or Path(self.csv_path).stat().st_size == 0:
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(TRADE_COLUMNS)

    # ------------------------------------------------------------------
    def _create_schema(self) -> None:
        assert self._db is not None
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                instrument TEXT,
                underlying TEXT,
                direction TEXT,
                quantity INTEGER,
                entry_price REAL,
                exit_price REAL,
                entry_time TEXT,
                exit_time TEXT,
                exit_reason TEXT,
                gross_pnl REAL,
                costs REAL,
                net_pnl REAL,
                return_pct REAL,
                holding_duration_seconds INTEGER,
                strategy TEXT,
                signal_score REAL,
                signal_confidence REAL,
                stop_loss REAL,
                target REAL,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_stats_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                open_positions INTEGER,
                total_pnl REAL,
                win_rate REAL,
                profit_factor REAL,
                expectancy REAL,
                max_drawdown_pnl REAL,
                max_drawdown_pct REAL,
                avg_holding_duration_seconds REAL,
                exit_reason_counts TEXT,
                snapshot_json TEXT
            )
        """)
        # Bug C.2 (2026-25-07 audit): open-position persistence table.
        # Mirrored by a separate method so it can be re-run idempotently
        # (e.g. when ``load_open_positions`` is called on an old DB).
        self._create_open_positions_schema()
        self._db.commit()

    # ------------------------------------------------------------------
    def record_trade(self, trade: PaperTrade) -> None:
        """Append a closed trade to both stores."""
        d = trade.to_dict()

        if self._db is not None:
            cols = TRADE_COLUMNS
            try:
                self._db.execute(
                    "INSERT OR REPLACE INTO paper_trades ({cols}) VALUES ({ph})".format(
                        cols=",".join(cols),
                        ph=",".join(["?"] * len(cols)),
                    ),
                    [d.get(c, "" if c == "exit_reason" else 0) for c in cols],
                )
                self._db.commit()
            except Exception as e:
                logger.error(f"TradeRecorder: SQLite insert failed for {trade.trade_id}: {e}")

        if self.csv_path:
            try:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([d.get(c, "") for c in TRADE_COLUMNS])
            except Exception as e:
                logger.error(f"TradeRecorder: CSV append failed for {trade.trade_id}: {e}")

    # ------------------------------------------------------------------
    def record_stats_snapshot(self, stats: TradeStats) -> None:
        """Take a point-in-time snapshot of the running metrics table.

        Useful for tracking metric drift over time (e.g. compare win-rate
        after 10 trades vs 100).
        """
        import json
        from datetime import datetime
        captured = datetime.utcnow().isoformat()
        snapshot_json = json.dumps(stats.to_dict(), default=str)

        if self._db is not None:
            try:
                self._db.execute(
                    """INSERT INTO paper_stats_snapshots
                       (captured_at, total_trades, winning_trades, losing_trades,
                        open_positions, total_pnl, win_rate, profit_factor,
                        expectancy, max_drawdown_pnl, max_drawdown_pct,
                        avg_holding_duration_seconds, exit_reason_counts, snapshot_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (captured, stats.total_trades, stats.winning_trades,
                     stats.losing_trades, stats.open_positions, stats.total_pnl,
                     stats.win_rate, stats.profit_factor, stats.expectancy,
                     stats.max_drawdown_pnl, stats.max_drawdown_pct,
                     stats.avg_holding_duration_seconds,
                     json.dumps(stats.exit_reason_counts, default=str),
                     snapshot_json),
                )
                self._db.commit()
            except Exception as e:
                logger.error(f"TradeRecorder: stats snapshot failed: {e}")

    # ------------------------------------------------------------------
    def load_previously_closed_trades(self) -> List[PaperTrade]:
        """Load all rows from the SQLite ``paper_trades`` table.

        Used by the engine to resumable a session that crashed mid-day —
        we keep these as historical context only; open positions are not
        recreated (they're gone with the process).
        """
        from datetime import datetime
        from prometheus.papertrade.types import (
            Direction as Dir, ExitReason as ER,
        )

        trades: List[PaperTrade] = []
        if self._db is None:
            return trades
        for row in self._db.execute("SELECT * FROM paper_trades"):
            d = dict(row)
            try:
                trades.append(PaperTrade(
                    trade_id=d["trade_id"],
                    symbol=d["symbol"],
                    instrument=d["instrument"],
                    underlying=d["underlying"],
                    direction=Dir(d["direction"]),
                    quantity=int(d["quantity"]),
                    entry_price=float(d["entry_price"]),
                    exit_price=float(d["exit_price"]),
                    entry_time=datetime.fromisoformat(d["entry_time"]),
                    exit_time=datetime.fromisoformat(d["exit_time"]),
                    exit_reason=ER(d["exit_reason"]),
                    gross_pnl=float(d["gross_pnl"]),
                    costs=float(d["costs"]),
                    net_pnl=float(d["net_pnl"]),
                    return_pct=float(d["return_pct"]),
                    holding_duration_seconds=int(d["holding_duration_seconds"]),
                    strategy=d.get("strategy", ""),
                    signal_score=float(d.get("signal_score") or 0),
                    signal_confidence=float(d.get("signal_confidence") or 0),
                    stop_loss=float(d.get("stop_loss") or 0),
                    target=float(d.get("target") or 0),
                ))
            except Exception as e:
                logger.warning(f"TradeRecorder: skip unloadable row {d.get('trade_id')}: {e}")
                continue
        return trades

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Bug C.2 (2026-07-25 audit): open-position persistence
    # ------------------------------------------------------------------
    # Before this patch, ``TradeRecorder`` only persisted CLOSED trades
    # (``record_trade``) — open positions lived solely in
    # ``PositionTracker.open_positions`` (in-memory dict). On any restart
    # (process crash, daily service bounce, configuration reload) every
    # open paper_capture position was silently abandoned: no exit was
    # ever booked, no telegram alert fired, and ``process_bar`` had no
    # idea the position existed. Production paper-mode P&L silently lost
    # every entry that survived past a restart boundary — exactly the
    # "ghost position" failure mode the LivePaperCapture rewrite was
    # supposed to eliminate.
    #
    # Fix: add a ``paper_open_positions`` table mirroring the open
    # ``Position`` dataclass fields. The tracker's ``open_position``
    # inserts a row; ``close_position`` deletes it. On startup, the
    # ``LivePaperCapture`` adapter calls ``load_open_positions`` to
    # rehydrate the in-memory dict from disk. Closed rows still flow
    # to ``paper_trades`` (the existing table) via the immutable
    # ``record_trade`` path — so historical reporting is unchanged.
    def _create_open_positions_schema(self) -> None:
        """Idempotent migration: create ``paper_open_positions`` if missing.

        Existing databases (pre-patch) won't have the table; the
        ``CREATE TABLE IF NOT EXISTS`` makes the migration safe on every
        startup without a separate ``ALTER TABLE`` step.
        """
        assert self._db is not None
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_open_positions (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                instrument TEXT,
                underlying TEXT,
                direction TEXT,
                quantity INTEGER,
                entry_price REAL,
                entry_time TEXT,
                stop_loss REAL,
                target REAL,
                max_bars INTEGER,
                bars_held INTEGER DEFAULT 0,
                max_bars_allowed TEXT,
                breakeven_set INTEGER DEFAULT 0,
                trailing_floor REAL DEFAULT 0,
                high_water_mark REAL DEFAULT 0,
                strategy TEXT DEFAULT '',
                signal_score REAL DEFAULT 0,
                signal_confidence REAL DEFAULT 0,
                trade_mode TEXT DEFAULT 'intraday',
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.commit()

    def record_open_position(self, position_dict: dict) -> None:
        """Upsert one row into ``paper_open_positions``.

        Called by ``PositionTracker.open_position`` immediately after a
        duplicate-rejected insert succeeds.
        ``position_dict`` should be the output of ``Position.to_dict``
        (already JSON-serializable — datetime fields as ISO strings,
        enums as their values).
        """
        if self._db is None:
            return
        cols = [
            "trade_id", "symbol", "instrument", "underlying", "direction",
            "quantity", "entry_price", "entry_time", "stop_loss", "target",
            "max_bars", "bars_held", "max_bars_allowed", "breakeven_set",
            "trailing_floor", "high_water_mark", "strategy",
            "signal_score", "signal_confidence", "trade_mode",
        ]
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO paper_open_positions ({cols}) VALUES ({ph})".format(
                    cols=",".join(cols),
                    ph=",".join(["?"] * len(cols)),
                ),
                [
                    position_dict.get("trade_id"),
                    position_dict.get("symbol"),
                    position_dict.get("instrument"),
                    position_dict.get("underlying"),
                    position_dict.get("direction"),
                    int(position_dict.get("quantity") or 0),
                    float(position_dict.get("entry_price") or 0),
                    position_dict.get("entry_time"),
                    float(position_dict.get("stop_loss") or 0),
                    float(position_dict.get("target") or 0),
                    int(position_dict.get("max_bars") or 0),
                    int(position_dict.get("bars_held") or 0),
                    str(position_dict.get("max_bars_allowed") or ""),
                    int(bool(position_dict.get("breakeven_set"))),
                    float(position_dict.get("trailing_floor") or 0),
                    float(position_dict.get("high_water_mark") or 0),
                    position_dict.get("strategy") or "",
                    float(position_dict.get("signal_score") or 0),
                    float(position_dict.get("signal_confidence") or 0),
                    position_dict.get("trade_mode") or "intraday",
                ],
            )
            self._db.commit()
        except Exception as e:
            logger.error(
                f"TradeRecorder: open-position insert failed for "
                f"{position_dict.get('trade_id')}: {e}"
            )

    def delete_open_position(self, trade_id: str) -> None:
        """Remove a row from ``paper_open_positions`` on close.

        Called by ``PositionTracker.close_position`` after the trade is
        deleted from the in-memory dict (and the closed PaperTrade has
        been recorded to ``paper_trades`` via the existing path).
        """
        if self._db is None:
            return
        try:
            self._db.execute(
                "DELETE FROM paper_open_positions WHERE trade_id = ?",
                (trade_id,),
            )
            self._db.commit()
        except Exception as e:
            logger.error(
                f"TradeRecorder: open-position delete failed for {trade_id}: {e}"
            )

    def load_open_positions(self) -> List[dict]:
        """Read every row from ``paper_open_positions``.

        Returns a list of dicts (one per open position at last shutdown).
        The ``LivePaperCapture`` adapter rehydrates these into
        ``engine.tracker.open_positions`` on init — see
        ``paper_executor/live_bridge.py``.
        """
        if self._db is None:
            return []
        try:
            self._create_open_positions_schema()
            rows = self._db.execute(
                "SELECT * FROM paper_open_positions"
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"TradeRecorder: load_open_positions failed: {e}")
            return []

