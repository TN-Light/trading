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

    def show_scanner_table(self, scan_results: list):
        """Display ranked multi-index scanner results with regime-adjusted confidence."""
        if not scan_results:
            if self.console:
                self.console.print("[dim]No signals found across any index[/dim]")
            else:
                print("No signals found across any index")
            return

        # Sort by adjusted confidence (highest first)
        scan_results.sort(key=lambda x: x.get("adj_confidence", 0), reverse=True)

        # Regime quality tiers from backtest data
        REGIME_QUALITY = {
            "markup": ("HIGH", "green"),
            "markdown": ("HIGH", "green"),
            "accumulation": ("MED", "yellow"),
            "distribution": ("MED", "yellow"),
            "volatile": ("LOW", "magenta"),
            "unknown": ("WEAK", "red"),
        }

        if not self.console:
            print(f"\n{'=' * 80}")
            print("  PROMETHEUS — Multi-Index Scanner (Ranked by Confidence)")
            print(f"{'=' * 80}")
            print(f"  {'Symbol':<20} {'Action':<10} {'Confluence':>10} {'Regime':<14} "
                  f"{'Quality':>8} {'Signals':>8} {'Adj Conf':>9}")
            print(f"  {'-' * 75}")
            for r in scan_results:
                regime = r.get("regime", "unknown")
                quality, _ = REGIME_QUALITY.get(regime, ("???", "dim"))
                warn = " << CAUTION" if quality in ("WEAK", "LOW") else ""
                print(f"  {r['symbol']:<20} {r['action']:<10} {r['raw_confidence']:>9.0%} "
                      f"{regime:<14} {quality:>8} {r['signal_count']:>5}/10 "
                      f"{r['adj_confidence']:>8.0%}{warn}")
            print(f"{'=' * 80}")
            return

        table = Table(
            title="[bold cyan]Multi-Index Scanner[/bold cyan] (Ranked by Confidence)",
            box=box.ROUNDED,
            border_style="cyan",
            show_lines=True,
        )
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Symbol", style="bold", min_width=18)
        table.add_column("Action", min_width=10)
        table.add_column("Confluence", justify="right", min_width=10)
        table.add_column("Regime", min_width=14)
        table.add_column("Quality", justify="center", min_width=8)
        table.add_column("Signals", justify="right", min_width=7)
        table.add_column("Adj Conf", justify="right", min_width=9)
        table.add_column("Alert", min_width=12)

        for i, r in enumerate(scan_results, 1):
            action = r.get("action", "HOLD")
            regime = r.get("regime", "unknown")
            quality, q_color = REGIME_QUALITY.get(regime, ("???", "dim"))
            raw_conf = r.get("raw_confidence", 0)
            adj_conf = r.get("adj_confidence", 0)
            sig_count = r.get("signal_count", 0)

            # Action color
            a_color = "green" if "CE" in action else "red" if "PE" in action else "yellow"

            # Confidence bar (visual)
            bar_len = int(adj_conf * 10)
            conf_bar = "█" * bar_len + "░" * (10 - bar_len)

            # Alert column
            alert = ""
            if quality == "WEAK":
                alert = "[bold red]!! AVOID !![/bold red]"
            elif quality == "LOW":
                alert = "[magenta]CAUTION[/magenta]"
            elif adj_conf >= 0.75:
                alert = "[bold green]STRONG[/bold green]"
            elif action == "HOLD":
                alert = "[dim]no setup[/dim]"

            # Executable flag
            executable = "" if r.get("executable", True) else " [dim](signal only)[/dim]"

            table.add_row(
                str(i),
                f"{r['symbol']}{executable}",
                f"[{a_color}]{action}[/{a_color}]",
                f"{raw_conf:.0%} {conf_bar}",
                f"[{q_color}]{regime.upper()}[/{q_color}]",
                f"[{q_color}]{quality}[/{q_color}]",
                f"{sig_count}/10",
                f"[bold]{adj_conf:.0%}[/bold]",
                alert,
            )

        self.console.print()
        self.console.print(table)

        # Legend
        self.console.print(
            "\n[dim]Quality: HIGH = markup/markdown (58-62% WR) | "
            "MED = accum/distrib | LOW = volatile | WEAK = unknown (26% WR)[/dim]"
        )
        self.console.print(
            "[dim]Adj Conf = Confluence x Regime Multiplier. "
            "Ranked highest-first. HOLD signals excluded from ranking.[/dim]"
        )

    def show_multi_account_summary(self, accounts: list):
        """Display comparison table of all simulated paper trading accounts."""
        if not accounts:
            return
        if not self.console:
            for acc in accounts:
                print(f"  {acc['label']}: Rs {acc['equity']:,.0f} ({acc['return_pct']:+.1f}%)")
            return

        table = Table(
            title="Multi-Account Paper Trading",
            box=box.ROUNDED,
            border_style="cyan",
        )
        table.add_column("Account", style="bold")
        table.add_column("Capital", justify="right")
        table.add_column("Equity", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("WR", justify="right")
        table.add_column("Open", justify="right")
        table.add_column("Costs", justify="right")

        for acc in accounts:
            pnl = acc.get("pnl", 0)
            ret = acc.get("return_pct", 0)
            color = "green" if pnl >= 0 else "red"
            table.add_row(
                acc["label"],
                f"Rs {acc['initial']:,.0f}",
                f"Rs {acc['equity']:,.0f}",
                f"[{color}]Rs {pnl:+,.0f}[/{color}]",
                f"[{color}]{ret:+.1f}%[/{color}]",
                str(acc.get("trades", 0)),
                f"{acc.get('win_rate', 0):.0f}%",
                str(acc.get("open_positions", 0)),
                f"Rs {acc.get('total_costs', 0):,.0f}",
            )

        self.console.print()
        self.console.print(table)

    def show_bracket_status(self, bracket_info: Dict, funnel_stats: Optional[Dict] = None):
        """Display current capital bracket, RR parameters, and funnel stats."""
        b_name = bracket_info.get("name", "UNKNOWN")
        min_rr = bracket_info.get("min_rr", 0)
        max_loss = bracket_info.get("max_loss_per_trade", 0)
        cap = bracket_info.get("capital", 0)
        
        if not self.console:
            print(f"\nBracket: {b_name} | Capital: Rs {cap:,.0f} | Min RR: {min_rr} | Max Risk/Trade: Rs {max_loss:,.0f}")
            if funnel_stats:
                print(f"Funnel: Raw {funnel_stats.get('raw', 0)} -> RR Pass {funnel_stats.get('rr_pass', 0)} -> Executed {funnel_stats.get('final', 0)}")
            return

        panel_text = (
            f"[bold cyan]Active Bracket:[/bold cyan] [bold]{b_name}[/bold] (Capital: Rs {cap:,.0f})\n"
            f"[bold cyan]Min R:R:[/bold cyan] {min_rr}x | "
            f"[bold cyan]Max Risk/Trade:[/bold cyan] Rs {max_loss:,.0f}"
        )
        
        if funnel_stats:
            raw = funnel_stats.get('raw', 0)
            rr_pass = funnel_stats.get('rr_pass', 0)
            final = funnel_stats.get('final', 0)
            panel_text += f"\n[bold cyan]Live Funnel:[/bold cyan] {raw} Raw -> {rr_pass} Passed RR -> {final} Final Trades"
            
        self.console.print(Panel(
            panel_text,
            title="[bold yellow]Capital Allocation & RR[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        ))
