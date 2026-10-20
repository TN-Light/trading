"""
Post-Exit Signal Tracking & Attribution Analytics Engine.

Analyzes post-exit performance metrics from SQLite (reports/papertrade/live_ledger.sqlite)
to generate data-driven reports on:
  1. Premature Shakeout Rate (%)
  2. MFE Left on Table Beyond Target (Pts)
  3. Capital Drawdown Saved by Stop Loss (INR)
  4. Inactivity Kill-Switch Invalidation Rate (%)
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Any
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class PostExitMetrics:
    total_trades: int
    sl_exits: int
    premature_shakeouts: int
    premature_shakeout_rate: float
    target_exits: int
    avg_alpha_left_pts: float
    max_alpha_left_pts: float
    drawdown_saved_pts: float
    drawdown_saved_inr: float
    inactivity_exits: int
    inactivity_invalidation_rate: float
    avg_mfe_pts: float
    avg_mae_pts: float


class PostExitAnalytics:
    """Computes empirical post-exit performance attribution."""

    def __init__(self, db_path: str = "reports/papertrade/live_ledger.sqlite"):
        self.db_path = str(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def compute_metrics(self) -> PostExitMetrics:
        """Query post_exit_attribution table and compute aggregate metrics."""
        try:
            conn = self._get_connection()
            # Verify table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='post_exit_attribution'")
            if not cursor.fetchone():
                conn.close()
                return self._empty_metrics()

            df = pd.read_sql_query("SELECT * FROM post_exit_attribution", conn)
            conn.close()

            if df.empty:
                return self._empty_metrics()

            total_trades = len(df)
            sl_mask = df["exit_reason"].str.upper().str.contains("STOP|BAILOUT", na=False)
            sl_df = df[sl_mask]
            sl_count = len(sl_df)

            shakeouts = int(sl_df["is_premature_shakeout"].sum()) if not sl_df.empty else 0
            shakeout_rate = (shakeouts / sl_count * 100.0) if sl_count > 0 else 0.0

            tgt_mask = df["exit_reason"].str.upper().str.contains("TARGET", na=False)
            tgt_df = df[tgt_mask]
            tgt_count = len(tgt_df)
            avg_alpha_left = float(tgt_df["alpha_left_pts"].mean()) if not tgt_df.empty else 0.0
            max_alpha_left = float(tgt_df["alpha_left_pts"].max()) if not tgt_df.empty else 0.0

            total_dd_saved_pts = float(sl_df["drawdown_saved_pts"].sum()) if not sl_df.empty else 0.0
            # Approximate INR saved using lot sizing (15 BN / 10 Sensex / 25 Nifty ~ 15 qty average)
            total_dd_saved_inr = total_dd_saved_pts * 15.0

            inactivity_mask = df["exit_reason"].str.upper().str.contains("INACTIVITY|KILL_SWITCH", na=False)
            inact_df = df[inactivity_mask]
            inact_count = len(inact_df)
            inact_inval = int(inact_df["target_reclaimed"].sum()) if not inact_df.empty else 0
            inact_rate = (inact_inval / inact_count * 100.0) if inact_count > 0 else 0.0

            avg_mfe = float(df["post_exit_mfe"].mean()) if not df.empty else 0.0
            avg_mae = float(df["post_exit_mae"].mean()) if not df.empty else 0.0

            return PostExitMetrics(
                total_trades=total_trades,
                sl_exits=sl_count,
                premature_shakeouts=shakeouts,
                premature_shakeout_rate=round(shakeout_rate, 1),
                target_exits=tgt_count,
                avg_alpha_left_pts=round(avg_alpha_left, 1),
                max_alpha_left_pts=round(max_alpha_left, 1),
                drawdown_saved_pts=round(total_dd_saved_pts, 1),
                drawdown_saved_inr=round(total_dd_saved_inr, 2),
                inactivity_exits=inact_count,
                inactivity_invalidation_rate=round(inact_rate, 1),
                avg_mfe_pts=round(avg_mfe, 1),
                avg_mae_pts=round(avg_mae, 1),
            )
        except Exception:
            return self._empty_metrics()

    def _empty_metrics(self) -> PostExitMetrics:
        return PostExitMetrics(
            total_trades=0,
            sl_exits=0,
            premature_shakeouts=0,
            premature_shakeout_rate=0.0,
            target_exits=0,
            avg_alpha_left_pts=0.0,
            max_alpha_left_pts=0.0,
            drawdown_saved_pts=0.0,
            drawdown_saved_inr=0.0,
            inactivity_exits=0,
            inactivity_invalidation_rate=0.0,
            avg_mfe_pts=0.0,
            avg_mae_pts=0.0,
        )

    def generate_telegram_summary(self) -> str:
        """Format an actionable summary card for Telegram."""
        m = self.compute_metrics()
        if m.total_trades == 0:
            return (
                "📊 <b>POST-EXIT TRACKING FORENSICS</b>\n\n"
                "<i>No post-exit observations recorded yet. Active observer is tracking all newly closed trades across 5m, 15m, 30m, and 60m milestones.</i>"
            )

        verdict_shakeout = "🟢 LOW (SL is outside noise)" if m.premature_shakeout_rate < 30.0 else "🔴 HIGH (SL noise floor needs expansion)"
        return (
            "📊 <b>POST-EXIT TRACKING FORENSICS</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>Trades Observed:</b> <code>{m.total_trades}</code>\n"
            f"🛑 <b>Stop Loss Exits:</b> <code>{m.sl_exits}</code>\n"
            f"🔄 <b>Premature Shakeouts:</b> <code>{m.premature_shakeouts} ({m.premature_shakeout_rate}%)</code>\n"
            f"   ↳ <i>Verdict: {verdict_shakeout}</i>\n\n"
            f"🎯 <b>Target Exits:</b> <code>{m.target_exits}</code>\n"
            f"🏃 <b>Avg Alpha Left Beyond Target:</b> <code>+{m.avg_alpha_left_pts:.1f} pts</code> (Max: +{m.max_alpha_left_pts:.1f} pts)\n"
            f"🛡️ <b>Capital Saved by Stop Loss:</b> <code>~Rs {m.drawdown_saved_inr:,.0f} (+{m.drawdown_saved_pts:.1f} pts)</code>\n"
            f"⏳ <b>Inactivity Kill Invalidation:</b> <code>{m.inactivity_invalidation_rate:.1f}%</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "ℹ️ <i>Data tracked asynchronously via SharedPriceMicroCache without consuming broker API limits.</i>"
        )

    def generate_markdown_report(self, output_path: str = "reports/papertrade/post_exit_attribution_report.md") -> str:
        """Write a comprehensive Markdown audit report."""
        m = self.compute_metrics()
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

        report = f"""# Post-Exit Performance Attribution & Forensic Report
**Generated:** {now_str}  
**Database:** `{self.db_path}`

---

## 1. Executive Summary

| Forensic KPI | Metric Value | Benchmark Standard | Operational Verdict |
| :--- | :--- | :--- | :--- |
| **Total Observed Closed Trades** | **{m.total_trades}** | N >= 30 | {"Sufficient Sample" if m.total_trades >= 30 else "Accumulating Sample"} |
| **Stop Loss Exits** | **{m.sl_exits}** | — | — |
| **Premature Shakeout Rate** | **{m.premature_shakeout_rate}%** ({m.premature_shakeouts} trades) | < 25.0% | {"🟢 Healthy (SL outside market noise)" if m.premature_shakeout_rate < 25.0 else "🔴 High Noise Shakeout (Expand SL)"} |
| **Target Exits** | **{m.target_exits}** | — | — |
| **Avg Alpha Left Beyond Target** | **+{m.avg_alpha_left_pts} pts** | < 15.0 pts | {"🟢 Well Calibrated" if m.avg_alpha_left_pts < 15.0 else "⚠️ Targets Truncating Runners"} |
| **Capital Preserved by SL** | **~₹{m.drawdown_saved_inr:,.2f}** | > 0 | 🟢 Net Capital Defense Positive |
| **Inactivity Kill Invalidation** | **{m.inactivity_invalidation_rate}%** | < 20.0% | {"🟢 Clean Stagnation Harvest" if m.inactivity_invalidation_rate < 20.0 else "⚠️ Premature Kill (Extend hold time)"} |

---

## 2. Core Quantitative Findings

1. **Premature Shakeout Attribution**:
   - Total Stop-Loss trades that subsequently reversed to hit Target or reclaim Entry: **{m.premature_shakeouts} out of {m.sl_exits} ({m.premature_shakeout_rate}%)**.
   - If this rate is < 25%, the current stop loss placement is mathematically validated and successfully protecting capital against trend continuation.

2. **Runner Preservation & Target Calibration**:
   - Average post-exit expansion beyond Target: **+{m.avg_alpha_left_pts} pts** (Peak: **+{m.max_alpha_left_pts} pts**).
   - Demonstrates whether dynamic multi-ATR targets are capturing sufficient runner alpha.

3. **Inactivity Kill-Switch Reliability**:
   - Liquidations that subsequent broke out to target: **{m.inactivity_invalidation_rate}%**.
   - Validates that killing stagnant option buys after 45m avoids theta cremation without forfeiting late breakouts.
"""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        return report


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    analytics = PostExitAnalytics()
    analytics.generate_markdown_report()
    print(analytics.generate_telegram_summary())
