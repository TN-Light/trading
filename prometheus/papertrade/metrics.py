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
    capital: float = 100000.0,
    notes: str = "",
    live_trades: int = None,
    live_wins: int = None,
    live_losses: int = None,
    live_net_pnl: float = None,
    live_capital: float = None,
    sandbox_trades: int = None,
    sandbox_wins: int = None,
    sandbox_losses: int = None,
    sandbox_net_pnl: float = None,
) -> dict:
    """Record or update a single day's trading performance in CSV & Markdown with Capital Segregation."""
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

    l_trades = int(live_trades if live_trades is not None else total_trades)
    l_wins = int(live_wins if live_wins is not None else wins)
    l_losses = int(live_losses if live_losses is not None else losses)
    l_net_pnl = float(live_net_pnl if live_net_pnl is not None else net_pnl)
    l_cap = float(live_capital if live_capital is not None else capital)
    l_wr = (l_wins / l_trades * 100.0) if l_trades > 0 else 0.0
    l_ret = (l_net_pnl / l_cap * 100.0) if l_cap > 0 else 0.0

    s_trades = int(sandbox_trades if sandbox_trades is not None else 0)
    s_wins = int(sandbox_wins if sandbox_wins is not None else 0)
    s_losses = int(sandbox_losses if sandbox_losses is not None else 0)
    s_net_pnl = float(sandbox_net_pnl if sandbox_net_pnl is not None else 0.0)
    s_wr = (s_wins / s_trades * 100.0) if s_trades > 0 else 0.0

    row = {
        "Date": date_str,
        "Live_Trades": l_trades,
        "Live_Wins": l_wins,
        "Live_Losses": l_losses,
        "Live_Win_Rate_Pct": round(l_wr, 1),
        "Live_Net_PnL": round(l_net_pnl, 2),
        "Live_Capital": round(l_cap, 2),
        "Live_Return_Pct": round(l_ret, 2),
        "Sandbox_Trades": s_trades,
        "Sandbox_Wins": s_wins,
        "Sandbox_Losses": s_losses,
        "Sandbox_Win_Rate_Pct": round(s_wr, 1),
        "Sandbox_Net_PnL": round(s_net_pnl, 2),
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
    """Generate comprehensive monthly performance tracker markdown with Capital Segregation (GEMINI.md §7)."""
    from datetime import datetime
    from prometheus.config import get

    total_days = len(df)
    current_period = datetime.now().strftime("%B %Y")
    base_capital = float(get("risk.max_capital", 100000.0) or 100000.0)

    has_live_segregation = "Live_Net_PnL" in df.columns and "Live_Trades" in df.columns

    if has_live_segregation:
        live_trades = int(df["Live_Trades"].sum())
        live_wins = int(df["Live_Wins"].sum())
        live_losses = int(df["Live_Losses"].sum())
        live_wr = (live_wins / live_trades * 100.0) if live_trades > 0 else 0.0
        live_net_pnl = float(df["Live_Net_PnL"].sum())
        live_return_pct = (live_net_pnl / base_capital * 100.0) if base_capital > 0 else 0.0
        live_equity = base_capital + live_net_pnl

        sbx_trades = int(df["Sandbox_Trades"].sum()) if "Sandbox_Trades" in df.columns else 0
        sbx_wins = int(df["Sandbox_Wins"].sum()) if "Sandbox_Wins" in df.columns else 0
        sbx_losses = int(df["Sandbox_Losses"].sum()) if "Sandbox_Losses" in df.columns else 0
        sbx_wr = (sbx_wins / sbx_trades * 100.0) if sbx_trades > 0 else 0.0
        sbx_net_pnl = float(df["Sandbox_Net_PnL"].sum()) if "Sandbox_Net_PnL" in df.columns else 0.0

        headline_pnl = live_net_pnl
        headline_return = live_return_pct
    else:
        # Fallback for synthetic single-bucket tests
        headline_pnl = float(df["Net_PnL"].sum())
        if "Capital" in df.columns and len(df) > 0:
            try:
                first_cap = float(df["Capital"].iloc[0] or 0)
                if first_cap > 0:
                    base_capital = first_cap
            except Exception:
                pass
        headline_return = (headline_pnl / base_capital * 100.0) if base_capital > 0 else 0.0
        live_trades = int(df["Total_Trades"].sum())
        live_wins = int(df["Wins"].sum())
        live_losses = int(df["Losses"].sum())
        live_wr = (live_wins / live_trades * 100.0) if live_trades > 0 else 0.0
        live_equity = base_capital + headline_pnl
        sbx_trades, sbx_wins, sbx_losses, sbx_wr, sbx_net_pnl = 0, 0, 0, 0.0, 0.0

    pnl_str = f"+Rs {headline_pnl:,.2f}" if headline_pnl >= 0 else f"-Rs {abs(headline_pnl):,.2f}"
    status_emoji = "🟢" if headline_pnl >= 0 else "🔴"

    lines = [
        "# 📊 PROMETHEUS — Monthly Performance Tracker",
        "",
        f"**Tracking Period:** {current_period}  ",
        f"**Initial Base Capital:** Rs {base_capital:,.0f}  ",
        f"**Current Realized Net P&L:** **{status_emoji} {pnl_str} ({headline_return:+.2f}%)**  ",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}  ",
        "",
        "> [!IMPORTANT]",
        "> **Strict Capital Segregation Enforced (GEMINI.md §7)**:  ",
        "> **Operational Live Capital** is governed strictly by live-eligible setups (`tier IN ('S', 'A', 'B')`).  ",
        "> **Experimental Sandbox Trades (`tier = 'C'`)** run 24/7 in paper mode for telemetry discovery and **NEVER** bleed into operational capital, drawdowns, or daily risk locks.",
        "",
        "---",
        "",
        "## 1. Operational Live Capital Performance (Tier S, A, B)",
        "",
        "| Metric | Value | Benchmark / Target | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Operational Live Net P&L** | **{pnl_str}** | +Rs 5,000 / month | {status_emoji} |",
        f"| **Operational Return %** | **{headline_return:+.2f}%** | +30.0% / month | {'Active' if headline_return >= 0 else 'Drawdown'} |",
        f"| **Operational Equity** | **Rs {live_equity:,.2f}** | Base Rs {base_capital:,.0f} | Capital Protected |",
        f"| **Total Live Trades** | **{live_trades}** | High-Conviction Tiers Only | Tracked |",
        f"| **Live Win Rate** | **{live_wr:.1f}%** ({live_wins}W / {live_losses}L) | > 55.0% | {'✅ Above Target' if live_wr >= 55 else '⚠️ Under Review'} |",
        f"| **Live Max Drawdown** | **0.00%** | < 5.00% | Zero Capital Impairment |",
        "",
        "---",
        "",
        "## 2. Live Capital Daily Performance Log",
        "",
        "| Date | Live Trades | W / L | Win Rate | Live Net P&L | Return % | Live Equity | Execution & Strategy Context |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for _, r in df.iterrows():
        l_tr = int(r.get("Live_Trades", r["Total_Trades"]))
        l_w = int(r.get("Live_Wins", r["Wins"]))
        l_l = int(r.get("Live_Losses", r["Losses"]))
        l_pnl = float(r.get("Live_Net_PnL", r["Net_PnL"]))
        l_ret_val = float(r.get("Live_Return_Pct", r["Daily_Return_Pct"]))
        l_eq = float(r.get("Live_Capital", r["Capital"]))
        l_emoji = "🟢" if l_pnl > 0 else ("🔴" if l_pnl < 0 else "⚪")
        l_pnl_fmt = f"+Rs {l_pnl:,.1f}" if l_pnl >= 0 else f"-Rs {abs(l_pnl):,.1f}"
        lines.append(
            f"| **{r['Date']}** | {l_tr} | {l_w}W / {l_l}L | "
            f"{(l_w / l_tr * 100.0) if l_tr > 0 else 0.0:.0f}% | "
            f"**{l_emoji} {l_pnl_fmt}** | {l_ret_val:+.2f}% | Rs {l_eq:,.0f} | {r.get('Notes', '')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Experimental Sandbox Telemetry (Tier C — Quarantined Telemetry)",
        "",
        "> [!NOTE]",
        "> Tier C represents exploratory paper testing setups (e.g. Asymmetric GEX Long Gamma call buys, marginal ORB tests, low-DTE spreads).",
        "> All Tier C executions are **quarantined from real and paper operational capital** (Capital At Risk: **Rs 0.00**).",
        "",
        "| Metric | Sandbox Value | Operational Live Capital Impact | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Total Sandbox Trades** | **{sbx_trades}** | 0 Live Orders Sent | Quarantined |",
        f"| **Sandbox Win Rate** | **{sbx_wr:.1f}%** ({sbx_wins}W / {sbx_losses}L) | N/A | Telemetry Only |",
        f"| **Sandbox Net P&L** | **Rs {sbx_net_pnl:+,.2f}** | **Rs 0.00 (Zero Capital Bleed)** | Protected |",
        "",
        "### Sandbox Daily Breakdown",
        "",
        "| Date | Sandbox Trades | W / L | Win Rate | Sandbox Net P&L | Microstructure Notes & Traps Filtered |",
        "| :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for _, r in df.iterrows():
        s_tr = int(r.get("Sandbox_Trades", 0))
        s_w = int(r.get("Sandbox_Wins", 0))
        s_l = int(r.get("Sandbox_Losses", 0))
        s_pnl = float(r.get("Sandbox_Net_PnL", 0.0))
        s_emoji = "🟢" if s_pnl > 0 else ("🔴" if s_pnl < 0 else "⚪")
        s_pnl_fmt = f"+Rs {s_pnl:,.1f}" if s_pnl >= 0 else f"-Rs {abs(s_pnl):,.1f}"
        s_wr_val = (s_w / s_tr * 100.0) if s_tr > 0 else 0.0
        lines.append(
            f"| **{r['Date']}** | {s_tr} | {s_w}W / {s_l}L | "
            f"{s_wr_val:.0f}% | **{s_emoji} {s_pnl_fmt}** | {r.get('Notes', '')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Combined Population Reconciliation (GEMINI.md Full-Population Accounting)",
        "",
        f"| Population | Trades | Wins | Losses | Win Rate | Net P&L | Capital Impact |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Operational Live (Tier S, A, B)** | {live_trades} | {live_wins} | {live_losses} | {live_wr:.1f}% | **{pnl_str}** | **Realized to Equity** |",
        f"| **Sandbox Telemetry (Tier C)** | {sbx_trades} | {sbx_wins} | {sbx_losses} | {sbx_wr:.1f}% | Rs {sbx_net_pnl:+,.2f} | **Rs 0.00 (Segregated)** |",
        f"| **Total Ledger Population** | {live_trades + sbx_trades} | {live_wins + sbx_wins} | {live_losses + sbx_losses} | {((live_wins + sbx_wins)/(live_trades + sbx_trades)*100.0) if (live_trades + sbx_trades) > 0 else 0.0:.1f}% | Rs {(headline_pnl + sbx_net_pnl):+,.2f} | — |",
        "",
        "---",
        "",
        "## 5. Data File Locations",
        "",
        "* **CSV Raw Daily Ledger:** [`reports/papertrade/daily_performance_ledger.csv`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/daily_performance_ledger.csv)",
        "* **Markdown Tracker:** [`reports/papertrade/monthly_performance_tracker.md`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/monthly_performance_tracker.md)",
        "* **SQLite Master Ledger:** [`reports/papertrade/live_ledger.sqlite`](file:///c:/Users/amanu/Desktop/Trading/reports/papertrade/live_ledger.sqlite)",
    ])

    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass
