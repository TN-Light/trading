# ============================================================================
# PROMETHEUS — Interface: CLI Dashboard
# ============================================================================
"""
Terminal-based trading dashboard using Rich library.
Displays real-time P&L, positions, signals, regime, and risk status.
"""

from typing import Dict, List, Optional
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.live import Live
    from rich.columns import Columns
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _safe_console() -> "Console":
    """Create a Rich Console safe for Windows cp1252 terminals."""
    import sys
    if sys.platform == "win32":
        import io
        # Force UTF-8 output to avoid cp1252 encoding crashes
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return Console(force_terminal=True, highlight=False)


class CLIDashboard:
    """Terminal dashboard for PROMETHEUS trading system."""

    def __init__(self):
        if not HAS_RICH:
            print("[WARNING] 'rich' library not installed. Using basic output.")
            print("Install with: pip install rich")
            self.console = None
        else:
            self.console = _safe_console()

    def show_header(self):
        """Display system header."""
        if not self.console:
            print("=" * 60)
            print("  PROMETHEUS Trading System v1.0")
            print("=" * 60)
            return

        header = Panel(
            "[bold cyan]PROMETHEUS[/bold cyan] Trading System v1.0\n"
            "[dim]Indian F&O | Intelligent Trading[/dim]",
            border_style="cyan",
            box=box.DOUBLE,
        )
        self.console.print(header)

    def show_portfolio_summary(self, portfolio: Dict):
        """Display portfolio summary."""
        capital = portfolio.get("initial_capital", 200000)
        equity = portfolio.get("equity", capital)
        pnl_today = portfolio.get("daily_pnl", 0)
        pnl_total = equity - capital
        open_positions = portfolio.get("open_positions", 0)
        margin_used = portfolio.get("margin_used_pct", 0)

        pnl_color = "green" if pnl_today >= 0 else "red"
        total_color = "green" if pnl_total >= 0 else "red"

        if not self.console:
            print(f"\nPortfolio: Rs {equity:,.0f} | Today: Rs {pnl_today:+,.0f} | "
                  f"Total: Rs {pnl_total:+,.0f} | Positions: {open_positions}")
            return

        table = Table(title="Portfolio", box=box.ROUNDED, border_style="blue")
        table.add_column("Metric", style="cyan", min_width=20)
        table.add_column("Value", justify="right", min_width=15)

        table.add_row("Initial Capital", f"Rs {capital:,.0f}")
        table.add_row("Current Equity", f"Rs {equity:,.0f}")
        table.add_row("Today's P&L", f"[{pnl_color}]Rs {pnl_today:+,.0f}[/{pnl_color}]")
        table.add_row("Total P&L", f"[{total_color}]Rs {pnl_total:+,.0f}[/{total_color}]")
        table.add_row("Open Positions", str(open_positions))
        table.add_row("Margin Used", f"{margin_used:.1f}%")

        self.console.print(table)

    def show_positions(self, positions: List[Dict]):
        """Display open positions."""
        if not positions:
            if self.console:
                self.console.print("[dim]No open positions[/dim]")
            else:
                print("No open positions")
            return

        if not self.console:
            print("\n--- Open Positions ---")
            for p in positions:
                print(f"  {p.get('symbol', '')} | Qty: {p.get('quantity', 0)} | "
                      f"Entry: {p.get('entry', 0):.2f} | LTP: {p.get('ltp', 0):.2f} | "
                      f"P&L: Rs {p.get('pnl', 0):+,.0f}")
            return

        table = Table(title="Open Positions", box=box.ROUNDED, border_style="blue")
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("LTP", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Strategy", style="dim")

        for p in positions:
            pnl = p.get("pnl", 0)
            pnl_color = "green" if pnl >= 0 else "red"
            table.add_row(
                p.get("symbol", ""),
                str(p.get("quantity", 0)),
                f"{p.get('entry', 0):.2f}",
                f"{p.get('ltp', 0):.2f}",
                f"[{pnl_color}]{pnl:+,.0f}[/{pnl_color}]",
                f"{p.get('sl', 0):.2f}",
                f"{p.get('target', 0):.2f}",
                p.get("strategy", ""),
            )

        self.console.print(table)

    def show_signal(self, signal: Dict):
        """Display a trading signal prominently."""
        action = signal.get("action", "HOLD")
        symbol = signal.get("symbol", "")
        confidence = signal.get("confidence", 0)
        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        target = signal.get("target", 0)
        rr = signal.get("risk_reward", 0)
        reasoning = signal.get("reasoning", "")
        regime = signal.get("regime", "")

        if action == "HOLD":
            if self.console:
                self.console.print("[dim]Signal: HOLD — No actionable setup[/dim]")
            else:
                print("Signal: HOLD — No actionable setup")
            return

        action_color = "green" if "CE" in action else "red" if "PE" in action else "yellow"

        if not self.console:
            print(f"\n{'='*50}")
            print(f"  SIGNAL: {action} {symbol}")
            print(f"  Confidence: {confidence:.0%} | R:R = 1:{rr:.1f}")
            print(f"  Entry: {entry:.2f} | SL: {sl:.2f} | Target: {target:.2f}")
            print(f"  Regime: {regime}")
            print(f"  {reasoning}")
            print(f"{'='*50}")
            return

        signal_text = (
            f"[bold {action_color}]{action}[/bold {action_color}] "
            f"[bold]{symbol}[/bold]\n\n"
            f"Confidence: [bold]{confidence:.0%}[/bold] | "
            f"R:R = [bold]1:{rr:.1f}[/bold]\n"
            f"Entry: [bold]{entry:.2f}[/bold] | "
            f"SL: [bold red]{sl:.2f}[/bold red] | "
            f"Target: [bold green]{target:.2f}[/bold green]\n"
            f"Regime: [dim]{regime}[/dim]\n\n"
            f"[dim]{reasoning}[/dim]"
        )

        panel = Panel(
            signal_text,
            title=f"[bold {action_color}]NEW SIGNAL[/bold {action_color}]",
            border_style=action_color,
            box=box.HEAVY,
        )
        self.console.print(panel)

    def show_regime(self, regime_info: Dict):
        """Display current market regime."""
        regime = regime_info.get("regime", "UNKNOWN")
        confidence = regime_info.get("confidence", 0)
        volatility = regime_info.get("volatility_state", "")
        trend = regime_info.get("trend_strength", 0)
        strategy = regime_info.get("recommended_strategy", "")

        regime_colors = {
            "ACCUMULATION": "blue",
            "MARKUP": "green",
            "DISTRIBUTION": "yellow",
            "MARKDOWN": "red",
            "VOLATILE": "magenta",
            "UNKNOWN": "dim",
        }
        color = regime_colors.get(regime, "dim")

        if not self.console:
            print(f"\nRegime: {regime} ({confidence:.0%}) | "
                  f"Vol: {volatility} | Strategy: {strategy}")
            return

        self.console.print(
            f"[bold {color}]Regime: {regime}[/bold {color}] "
            f"({confidence:.0%}) | "
            f"Volatility: {volatility} | "
            f"Trend: {trend:+.2f} | "
            f"Strategy: [bold]{strategy}[/bold]"
        )

    def show_risk_status(self, risk_info: Dict):
        """Display risk management status."""
        daily_pnl = risk_info.get("daily_pnl", 0)
        daily_limit = risk_info.get("daily_limit", 5000)
        weekly_pnl = risk_info.get("weekly_pnl", 0)
        weekly_limit = risk_info.get("weekly_limit", 10000)
        consec_losses = risk_info.get("consecutive_losses", 0)
        system_halted = risk_info.get("system_halted", False)

        if not self.console:
            status = "HALTED" if system_halted else "ACTIVE"
            print(f"\nRisk: {status} | "
                  f"Daily: Rs {daily_pnl:+,.0f}/{daily_limit:,.0f} | "
                  f"Weekly: Rs {weekly_pnl:+,.0f}/{weekly_limit:,.0f}")
            return

        # Risk meter
        daily_usage = abs(daily_pnl) / daily_limit * 100 if daily_limit > 0 else 0
        bar_color = "green" if daily_usage < 50 else "yellow" if daily_usage < 80 else "red"

        status_text = "[bold red]HALTED[/bold red]" if system_halted else "[bold green]ACTIVE[/bold green]"

        self.console.print(
            f"Risk Status: {status_text} | "
            f"Daily [{bar_color}]Rs {daily_pnl:+,.0f}[/{bar_color}]/{daily_limit:,.0f} "
            f"({daily_usage:.0f}%) | "
            f"Weekly Rs {weekly_pnl:+,.0f}/{weekly_limit:,.0f} | "
            f"Consecutive Losses: {consec_losses}"
        )

    def show_ai_insight(self, insight: Dict):
        """Display AI analysis insight."""
        sentiment = insight.get("sentiment", "neutral")
        reasoning = insight.get("reasoning", "")
        confidence = insight.get("confidence", 0)

        color = "green" if sentiment == "bullish" else "red" if sentiment == "bearish" else "yellow"

        if not self.console:
            print(f"\nAI: {sentiment.upper()} ({confidence:.0%}) — {reasoning}")
            return

        self.console.print(Panel(
            f"[{color}]{sentiment.upper()}[/{color}] ({confidence:.0%})\n{reasoning}",
            title="AI Intelligence",
            border_style="magenta",
        ))

    def show_trade_history(self, trades: List[Dict], limit: int = 10):
        """Display recent trade history."""
        if not trades:
            if self.console:
                self.console.print("[dim]No trade history[/dim]")
            else:
                print("No trade history")
            return

        recent = trades[-limit:]

        if not self.console:
            print(f"\n--- Last {len(recent)} Trades ---")
            for t in recent:
                pnl = t.get("net_pnl", 0)
                icon = "W" if pnl > 0 else "L"
                print(f"  [{icon}] {t.get('symbol', '')} | "
                      f"P&L: Rs {pnl:+,.0f} | {t.get('strategy', '')}")
            return

        table = Table(title=f"Last {len(recent)} Trades", box=box.ROUNDED, border_style="blue")
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Direction")
        table.add_column("P&L", justify="right")
        table.add_column("Strategy", style="dim")

        for t in recent:
            pnl = t.get("net_pnl", 0)
            pnl_color = "green" if pnl > 0 else "red"
            table.add_row(
                t.get("timestamp", "")[:16],
                t.get("symbol", ""),
                t.get("direction", ""),
                f"[{pnl_color}]Rs {pnl:+,.0f}[/{pnl_color}]",
                t.get("strategy", ""),
            )

        self.console.print(table)

    def show_backtest_results(self, result):
        """Display backtest results."""
        if not self.console:
            print(result.summary())
            return

        self.console.print(Panel(
            result.summary(),
            title="[bold]Backtest Results[/bold]",
            border_style="cyan",
            box=box.HEAVY,
        ))

    def prompt_confirmation(self, signal: Dict) -> bool:
        """Ask user to confirm a trade in semi-auto mode."""
        self.show_signal(signal)

        response = input("\nExecute this trade? (y/n): ").strip().lower()
        return response in ("y", "yes")

    def show_status_line(self, text: str):
        """Show a single status line."""
        if self.console:
            self.console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] {text}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
