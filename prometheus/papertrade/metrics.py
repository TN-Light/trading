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


def record_daily_performance(
    date_str: str = None,
    total_trades: int = 0,
    wins: int = 0,
    losses: int = 0,
    gross_profit: float = 0.0,
    gross_loss: float = 0.0,
    net_pnl: float = 0.0,
    capital: float = 15000.0,
    notes: str = "",
) -> dict:
    """Record or update a single day's trading performance in CSV & Markdown."""
    import os
    import pandas as pd
    from datetime import datetime

    ledger_dir = os.path.join("reports", "papertrade")
    os.makedirs(ledger_dir, exist_ok=True)
    csv_path = os.path.join(ledger_dir, "daily_performance_ledger.csv")
    md_path = os.path.join(ledger_dir, "monthly_performance_tracker.md")

    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    daily_return = (net_pnl / capital * 100.0) if capital > 0 else 0.0

    row = {
        "Date": date_str,
        "Total_Trades": int(total_trades),
        "Wins": int(wins),
        "Losses": int(losses),
        "Win_Rate_Pct": round(win_rate, 1),
        "Gross_Profit": round(gross_profit, 2),
        "Gross_Loss": round(gross_loss, 2),
        "Net_PnL": round(net_pnl, 2),
        "Capital": round(capital, 2),
        "Daily_Return_Pct": round(daily_return, 2),
        "Cumulative_PnL": round(net_pnl, 2),
        "Notes": notes,
    }

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "Date" in df.columns:
                df = df[df["Date"] != date_str]
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
        except Exception:
            df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])

    df["Date_dt"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date_dt").reset_index(drop=True)
    df["Cumulative_PnL"] = df["Net_PnL"].cumsum().round(2)
    df = df.drop(columns=["Date_dt"])
    df.to_csv(csv_path, index=False)

    _generate_monthly_tracker_md(df, md_path)
    return row


def _generate_monthly_tracker_md(df, md_path: str):
    """Generate comprehensive monthly performance tracker markdown."""
    from datetime import datetime

    total_days = len(df)
    total_trades = int(df["Total_Trades"].sum())
    total_wins = int(df["Wins"].sum())
    total_losses = int(df["Losses"].sum())
    win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0

    total_gross_profit = float(df["Gross_Profit"].sum())
    total_gross_loss = float(df["Gross_Loss"].sum())
    total_net_pnl = float(df["Net_PnL"].sum())
    total_return_pct = (total_net_pnl / 15000.0 * 100.0)

    green_days = int((df["Net_PnL"] > 0).sum())
    red_days = int((df["Net_PnL"] < 0).sum())
    breakeven_days = total_days - green_days - red_days
    profit_factor = (abs(total_gross_profit) / abs(total_gross_loss)) if abs(total_gross_loss) > 0 else 0.0

    net_pnl_str = f"+Rs {total_net_pnl:,.2f}" if total_net_pnl >= 0 else f"-Rs {abs(total_net_pnl):,.2f}"
    status_emoji = "🟢" if total_net_pnl >= 0 else "🔴"

    lines = [
        "# 📊 PROMETHEUS — Monthly Performance Tracker",
        "",
        "**Tracking Period:** August 2026  ",
        "**Initial Base Capital:** Rs 15,000  ",
        f"**Current Realized Net P&L:** **{status_emoji} {net_pnl_str} ({total_return_pct:+.2f}%)**  ",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}  ",
        "",
        "---",
        "",
        "## 1. Month-to-Date Performance Overview",
        "",
        "| Metric | Value | Benchmark / Target | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Net Realized P&L** | **{net_pnl_str}** | +Rs 5,000 / month | {status_emoji} |",
        f"| **Month Return %** | **{total_return_pct:+.2f}%** | +30.0% / month | {'Active' if total_return_pct >= 0 else 'Drawdown'} |",
        f"| **Total Trading Days** | **{total_days}** | 20–22 days | In Progress |",
        f"| **Green / Red Days** | **{green_days} Green / {red_days} Red / {breakeven_days} BE** | > 65% Green | {(green_days/total_days*100) if total_days > 0 else 0:.1f}% |",
        f"| **Total Trades** | **{total_trades}** | ~3–5 / day | Tracked |",
        f"| **Win Rate** | **{win_rate:.1f}%** ({total_wins}W / {total_losses}L) | > 55.0% | {'✅ Above Target' if win_rate >= 55 else '⚠️ Under Review'} |",
        f"| **Profit Factor** | **{profit_factor:.2f}** | > 1.50 | {'✅ Good' if profit_factor >= 1.5 else '⚠️ Under Review'} |",
        f"| **Gross Profit** | +Rs {total_gross_profit:,.2f} | — | Winning Trades |",
        f"| **Gross Loss** | -Rs {abs(total_gross_loss):,.2f} | — | Losing Trades |",
        "",
        "---",
        "",
        "## 2. Daily Performance Log",
        "",
        "| Date | Trades | W / L | Win Rate | Gross Profit | Gross Loss | Net P&L | Daily Return | Cumulative P&L | Key Notes & Market Context |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for _, r in df.iterrows():
        pnl = float(r["Net_PnL"])
        cum = float(r["Cumulative_PnL"])
        day_emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
        pnl_fmt = f"+Rs {pnl:,.1f}" if pnl >= 0 else f"-Rs {abs(pnl):,.1f}"
        cum_fmt = f"+Rs {cum:,.1f}" if cum >= 0 else f"-Rs {abs(cum):,.1f}"
        ret_fmt = f"{float(r['Daily_Return_Pct']):+.1f}%"
        lines.append(
            f"| **{r['Date']}** | {int(r['Total_Trades'])} | {int(r['Wins'])}W / {int(r['Losses'])}L | "
            f"{float(r['Win_Rate_Pct']):.0f}% | +Rs {float(r['Gross_Profit']):,.0f} | -Rs {abs(float(r['Gross_Loss'])):,.0f} | "
            f"**{day_emoji} {pnl_fmt}** | {ret_fmt} | **{cum_fmt}** | {r.get('Notes', '')} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Data File Locations")
    lines.append("")
    lines.append("* **CSV Raw Ledger:** [`reports/papertrade/daily_performance_ledger.csv`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/daily_performance_ledger.csv)")
    lines.append("* **Markdown Tracker:** [`reports/papertrade/monthly_performance_tracker.md`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/monthly_performance_tracker.md)")

    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass
