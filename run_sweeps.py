import argparse
import sys
import os
import pandas as pd
from tabulate import tabulate

import concurrent.futures

from prometheus.main import Prometheus
try:
    from prometheus.analysis.rr_diagnostic import global_funnel
except ImportError:
    global_funnel = None

def run_series_a():
    """Series A: RR Sensitivity Sweep for the 15K bracket."""
    print("="*60)
    print("SERIES A: RR Sensitivity Sweep (15K Bracket)")
    print("="*60)
    
    rr_values = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    records = []

    for rr in rr_values:
        print(f"\n[Testing min_rr = {rr}]")
        try:
            p = Prometheus()
            
            # Override the 15K bracket config
            bracket = None
            for b in p.risk.bracket_manager.brackets:
                if "15K" in b.name:
                    bracket = b
                    break
            
            if bracket:
                bracket.min_rr = rr
            
            # Override global_funnel
            if global_funnel:
                global_funnel.reset()

            res = p.run_backtest(
                symbol="NIFTY 50",
                days=365,
                strategy="trend",
            )
            
            if res:
                records.append({
                    "Min RR": rr,
                    "Trades": res.total_trades,
                    "Win Rate (%)": res.win_rate,
                    "Profit Factor": res.profit_factor,
                    "Total Return (%)": res.total_return_pct,
                    "Max DD (%)": res.max_drawdown_pct,
                    "Sharpe": res.sharpe_ratio
                })
        except Exception as e:
            import traceback
            print("\n!!! EXCEPTION CAUGHT !!!")
            traceback.print_exc()
            break
            
    print("\n" + "="*60)
    print("SERIES A RESULTS: RR SENSITIVITY (15K BRACKET)")
    print("="*60)
    df = pd.DataFrame(records)
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

def run_series_b():
    """Series B: Cross Bracket Comparison."""
    print("="*60)
    print("SERIES B: Cross Bracket Comparison")
    print("="*60)
    brackets = ["15K", "30K", "50K", "1L", "2L"]
    records = []
    
    for b_name in brackets:
        print(f"\n[Testing Bracket: {b_name}]")
        p = Prometheus()
        
        b = None
        for bracket in p.risk.bracket_manager.brackets:
            if b_name in bracket.name:
                b = bracket
                break
                
        if not b:
            continue
        
        # Force capital to match bracket to simulate the bracket
        cap = b.max_capital if b.max_capital < float('inf') else 250000
        p.initial_capital = cap
        p.capital = cap
        
        if global_funnel:
            global_funnel.reset()

        try:
            res = p.run_backtest(
                symbol="NIFTY 50",
                days=365,
                strategy="trend",
            )
            print(f"-----> RES FOR {b_name} IS: {res}")
            if res:
                records.append({
                    "Bracket": b_name,
                    "Capital": cap,
                    "Min RR": b.min_rr,
                    "Trades": res.total_trades,
                    "Win Rate (%)": res.win_rate,
                    "Profit Factor": res.profit_factor,
                    "Total Return (%)": res.total_return_pct,
                    "Max DD (%)": res.max_drawdown_pct
                })
        except BaseException as e:
            import traceback
            import sys
            print("\n\n\n!!! EXCEPTION CAUGHT in SERIES B !!!", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            break
            
    print("\n" + "="*60)
    print("SERIES B RESULTS: CROSS BRACKET")
    print("="*60)
    df = pd.DataFrame(records)
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    df.to_csv("series_b_results.csv", index=False)

def run_series_c():
    """Series C: Factor Isolation."""
    print("="*60)
    print("SERIES C: Factor Isolation (Not Fully Implemented)")
    print("="*60)
    print("This series will test isolating individual factors like confluence, time stop, etc.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, choices=["a", "b", "c", "all"], default="a")
    args = parser.parse_args()
    
    if args.series in ["a", "all"]:
        run_series_a()
    if args.series in ["b", "all"]:
        run_series_b()
    if args.series in ["c", "all"]:
        run_series_c()
