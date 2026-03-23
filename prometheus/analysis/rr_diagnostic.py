import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass

from prometheus.utils.logger import logger

@dataclass
class FunnelStats:
    raw_signals: int = 0
    confluence_passed: int = 0
    regime_passed: int = 0
    rr_passed: int = 0
    kelly_passed: int = 0
    final_trades: int = 0

class SignalFunnelAnalyzer:
    """Tracks the exact funnel of drop-offs during signal generation."""
    
    def __init__(self):
        self.stats = FunnelStats()

    def record_raw(self):
        self.stats.raw_signals += 1
        
    def record_confluence_pass(self):
        self.stats.confluence_passed += 1
        
    def record_regime_pass(self):
        self.stats.regime_passed += 1
        
    def record_rr_pass(self):
        self.stats.rr_passed += 1
        
    def record_kelly_pass(self):
        self.stats.kelly_passed += 1
        
    def record_final_trade(self):
        self.stats.final_trades += 1

    def print_report(self):
        """Prints the signal funnel dropoff report and saves to file."""
        s = self.stats
        report = []
        report.append("\n" + "="*50)
        report.append("  SIGNAL FUNNEL DIAGNOSTIC REPORT")
        report.append("="*50)
        report.append(f"  1. Raw Setups Explored : {s.raw_signals}")
        
        c_drop = s.raw_signals - s.confluence_passed
        c_pct = (c_drop / s.raw_signals * 100) if s.raw_signals else 0
        report.append(f"  2. Confluence Passed   : {s.confluence_passed} (Dropped {c_drop} = {c_pct:.1f}%)")
        
        r_drop = s.confluence_passed - s.regime_passed
        r_pct = (r_drop / s.confluence_passed * 100) if s.confluence_passed else 0
        report.append(f"  3. Regime Passed       : {s.regime_passed} (Dropped {r_drop} = {r_pct:.1f}%)")
        
        rr_drop = s.regime_passed - s.rr_passed
        rr_pct = (rr_drop / s.regime_passed * 100) if s.regime_passed else 0
        report.append(f"  4. RR/Target Passed    : {s.rr_passed} (Dropped/Forced {rr_drop} = {rr_pct:.1f}%)")
        
        k_drop = s.rr_passed - s.kelly_passed
        k_pct = (k_drop / s.rr_passed * 100) if s.rr_passed else 0
        report.append(f"  5. Kelly EV Passed     : {s.kelly_passed} (Dropped {k_drop} = {k_pct:.1f}%)")
        
        t_drop = s.kelly_passed - s.final_trades
        t_pct = (t_drop / s.kelly_passed * 100) if s.kelly_passed else 0
        report.append(f"  6. Final Trades Entered: {s.final_trades} (Dropped by max positions/time {t_drop} = {t_pct:.1f}%)")
        report.append("="*50)
        
        out_str = "\n".join(report)
        print(out_str)
        try:
            with open("funnel_report.txt", "a") as f:
                f.write(out_str + "\n")
        except Exception as e:
            pass
        
        
class RRPerformanceAnalyzer:
    """Analyzes the actual empirical RR realized from historical trades."""
    
    def __init__(self, trades: List[Dict]):
        self.trades = trades
    
    def calculate_actual_rr(self, trade: Dict) -> float:
        """Calculate the actual achieved R:R for a single trade."""
        entry = float(trade.get('entry_price', 0))
        exit_p = float(trade.get('exit_price', trade.get('exit_price', 0)))
        qty = float(trade.get('quantity', 0))
        direction = trade.get('direction', 'bullish')

        if entry <= 0 or qty <= 0:
            return 0.0

        # Determine the assumed risk per unit (1R) using provided stop loss if available
        sl = float(trade.get('stop_loss', trade.get('initial_sl', 0)))
        if sl > 0 and sl != entry:
            risk_per_unit = abs(entry - sl)
        else:
            # Fallback: use average absolute loss across losers as 1R proxy; handled in calculate_metrics
            risk_per_unit = None

        # Compute realized P&L for the trade
        if 'net_pnl' in trade:
            pnl = float(trade['net_pnl'])
        else:
            price_diff = exit_p - entry
            if direction == 'bearish':
                price_diff = -price_diff
            pnl = price_diff * qty

        if risk_per_unit is None or risk_per_unit == 0:
            # Caller will normalize using avg_loss; return pnl as-is for later scaling
            return pnl

        risk_amount = risk_per_unit * qty
        return pnl / risk_amount

    def calculate_metrics(self):
        if not self.trades:
            print("No trades available for RR analysis.")
            return

        df = pd.DataFrame(self.trades)

        # First attempt per-trade RR using explicit stop loss when available
        df['realized_rr'] = df.apply(lambda row: self.calculate_actual_rr(row), axis=1)

        # For trades lacking SL info, scale using average loser size
        winners = df[df.get('net_pnl', pd.Series([0]*len(df))) > 0]
        losers = df[df.get('net_pnl', pd.Series([0]*len(df))) < 0]
        avg_loss = abs(losers['net_pnl'].mean()) if not losers.empty else 1.0
        df.loc[df['realized_rr'] == 0, 'realized_rr'] = df.loc[df['realized_rr'] == 0, 'net_pnl'] / avg_loss
        
        w_rr = df[df['realized_rr'] > 0]['realized_rr'].mean()
        l_rr = df[df['realized_rr'] < 0]['realized_rr'].mean()
        all_rr = df['realized_rr'].mean()
        
        pct_reached_2_5 = len(df[df['realized_rr'] >= 2.5]) / len(df) * 100
        
        time_stops = len(df[df['exit_reason'].str.contains('time_stop', na=False)]) / len(df) * 100
        trail_stops = len(df[df['exit_reason'].str.contains('trailing', na=False)]) / len(df) * 100
        target_hits = len(df[df['exit_reason'].str.contains('target', na=False)]) / len(df) * 100

        print("\n" + "="*50)
        print("  ACTUAL RR PERFORMANCE REPORT")
        print("="*50)
        print(f"  Average RR on WINNERS      : {w_rr:.2f}:1")
        print(f"  Average RR on LOSERS       : {l_rr:.2f}:1")
        print(f"  Average RR ACROSS ALL      : {all_rr:.2f}:1")
        print(f"  Trades achieving >= 2.5:1  : {pct_reached_2_5:.1f}%")
        print("-"*50)
        print(f"  Closed by TARGET           : {target_hits:.1f}%")
        print(f"  Closed by TIME STOP        : {time_stops:.1f}%")
        print(f"  Closed by TRAILING STOP    : {trail_stops:.1f}%")
        print("="*50)
        
        # Save histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['realized_rr'], bins=20, color='cyan', edgecolor='black', alpha=0.7)
        plt.axvline(2.5, color='red', linestyle='dashed', linewidth=2, label='2.5 Target')
        plt.axvline(w_rr, color='green', linestyle='dashed', linewidth=2, label=f'Avg Win ({w_rr:.1f})')
        plt.title('Actual Realized R:R Distribution')
        plt.xlabel('Realized R:R')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('rr_heatmap.png')
        plt.close()
        
        df[['entry_time', 'symbol', 'direction', 'net_pnl', 'realized_rr', 'exit_reason']].to_csv('actual_rr_distribution.csv', index=False)

if __name__ == '__main__':
    from prometheus.backtest.engine import BacktestResult
    import json
    
    # Simple test to verify functionality
    print("Diagnostics module loaded.")
