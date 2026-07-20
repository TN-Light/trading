"""
Computes and progressively updates aggregate paper-trade metrics.

Metrics computed:

* Total / winning / losing trade counts
* Win rate (%)
* Average win and average loss (Rs)
* Profit factor = sum_wins / abs(sum_losses)
* Expectancy per trade = wr * avg_win - (1-wr) * avg_loss
* Total net PnL (Rs) and total return %
* Max drawdown (peak-to-trough on cumulative realized PnL, Rs and %)
* Average holding duration (seconds)
* Per-exit-reason breakdown (strategy-level diagnostic)
* Open position snapshot count

The state is recomputed on every record_close() call so the metrics are
always consistent with the closed-trades record. We deliberately don't keep
incremental-only state because drift over thousands of trades would become
hard to reason about; trade counts are bounded by daily volume so the cost
is negligible.
"""

from __future__ import annotations

from typing import List

from prometheus.papertrade.types import (
    PaperTrade, TradeStats, Position, Direction, ExitReason,
)


class MetricsEngine:
    """Computes aggregate trade statistics.

    Pure-computation class. The engine or CLI can call ``snapshot()``
    whenever it needs a current view. Cost: O(N) in number of closed trades,
    where N is bounded by total trade count per session (typically < 1k/day).
    """

    def __init__(self):
        self.closed_trades: List[PaperTrade] = []

    def record_close(self, trade: PaperTrade) -> None:
        self.closed_trades.append(trade)

    def record_open_positions(self, count: int) -> None:
        # Snapshotio-wise we just keep the latest count supplied
        self._latest_open_positions = max(0, int(count))
        # ``snapshot`` uses this

    def snapshot(self, open_positions: int = 0) -> TradeStats:
        """Recompute and return the current TradeStats snapshot.

        Args:
            open_positions: count of positions still open (since this engine
                doesn't track the open list itself — caller passes it).
        """
        trades = self.closed_trades
        n = len(trades)
        if n == 0:
            stats = TradeStats(open_positions=open_positions)
            return stats

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl < 0]
        sum_wins = sum(t.net_pnl for t in wins)
        sum_losses = sum(t.net_pnl for t in losses)

        wr = (len(wins) / n) * 100.0 if n else 0.0
        avg_win = (sum_wins / len(wins)) if wins else 0.0
        avg_loss = (sum_losses / len(losses)) if losses else 0.0
        avg_win_pct = (
            sum(t.return_pct for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss_pct = (
            sum(t.return_pct for t in losses) / len(losses) if losses else 0.0
        )
        pf = (
            sum_wins / abs(sum_losses) if sum_losses != 0 else float("inf") if sum_wins > 0 else 0.0
        )
        # Both avg_win and avg_loss are SIGNED (avg_loss is negative). The
        # classic expectancy formula treats wins/losses as positive
        # magnitudes, so when avg_loss is already signed we use the
        # "+" sign in front of (1-wr) * avg_loss (subtracting a negative ==
        # adding its absolute value).
        expectancy = (
            (wr / 100.0) * avg_win + (1.0 - wr / 100.0) * avg_loss
        )

        # Drawdown: peak-to-trough over the cumulative realized PnL curve
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        max_dd_pct = 0.0
        # The denominator for pct drawdown: peak of the equity curve. If the
        # curve never rises above zero (only losing trades, running loss
        # never_booked_a_profit), peak stays 0 and we fall back to the worst
        # cumulative loss as the reference notional.
        max_dd_ref = 0.0
        for t in trades:
            cum += t.net_pnl
            if cum > peak:
                peak = cum
            if cum < max_dd_ref:
                max_dd_ref = -cum   # positive number, magnitude of worst running loss
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        # Pct drawdown denominator: peak if any peak, else worst-loss magnitude
        denom = peak if peak > 0 else max(1.0, max_dd_ref)
        if denom > 0:
            max_dd_pct = (max_dd / denom) * 100.0

        avg_dur = sum(t.holding_duration_seconds for t in trades) / n

        exit_counts: dict = {}
        for t in trades:
            # Always serialize ExitReason to a plain str for the dict
            k = t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason)
            exit_counts[k] = exit_counts.get(k, 0) + 1
        # silence "unused" lint — ExitReason imported for the hasattr check above
        _ = ExitReason.STOP_LOSS

        return TradeStats(
            total_trades=n,
            winning_trades=len(wins),
            losing_trades=len(losses),
            open_positions=open_positions,
            total_pnl=sum(t.net_pnl for t in trades),
            total_costs=sum(t.costs for t in trades),
            gross_pnl=sum(t.gross_pnl for t in trades),
            total_return_pct=sum(t.return_pct for t in trades),
            avg_win_pnl=avg_win,
            avg_loss_pnl=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win_pnl=max((t.net_pnl for t in wins), default=0.0),
            largest_loss_pnl=min((t.net_pnl for t in losses), default=0.0),
            win_rate=wr,
            profit_factor=pf,
            expectancy=expectancy,
            cumulative_pnl=max(0.0, cum) if cum > 0 else cum,
            peak_pnl=peak,
            max_drawdown_pnl=max_dd,
            max_drawdown_pct=max_dd_pct,
            avg_holding_duration_seconds=avg_dur,
            exit_reason_counts=exit_counts,
        )
