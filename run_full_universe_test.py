import os
import sys
import subprocess
import yaml
import re
import concurrent.futures

def run_backtest_for_symbol(symbol, main_py, env):
    cmd = [sys.executable, main_py, "backtest", "--symbol", symbol, "--days", "365", "--intraday"]
    try:
        res = subprocess.run(cmd, capture_output=True, env=env)
        
        output = ""
        if res.stdout: output += res.stdout.decode("utf-8", errors="replace")
        if res.stderr: output += "\n" + res.stderr.decode("utf-8", errors="replace")
        
        trades, wr, pnl_val = 0, 0.0, 0.0
        
        m_trades = re.search(r"Total Trades:\s*(\d+)", output)
        if m_trades: trades = int(m_trades.group(1))
        
        m_wr = re.search(r"Win Rate:\s*([\d\.]+)%", output)
        if m_wr: wr = float(m_wr.group(1))
        
        m_pnl = re.search(r"PnL Rs\s+([+\-\d,]+)", output)
        if m_pnl: pnl_val = m_pnl.group(1).replace(",", "")
        
        return {
            "symbol": symbol,
            "trades": trades,
            "win_rate": wr,
            "pnl": pnl_val
        }
    except Exception as e:
        return {"symbol": symbol, "trades": 0, "win_rate": 0, "pnl": f"Error: {str(e)}"}

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(root, "prometheus", "config", "settings.yaml")
    
    with open(settings_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    indices = cfg.get("market", {}).get("indices", [])
    intraday = cfg.get("intraday", {}).get("instruments", [])
    
    universe = list(dict.fromkeys(indices + intraday))
    
    print("=======================================================================")
    print(f" PROMETHEUS — Full Universe 365D Engine Backtest ({len(universe)} symbols)")
    print("=======================================================================\n")
    print("Testing locally in parallel. Please wait...")
    
    main_py = os.path.join(root, "prometheus", "main.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_backtest_for_symbol, sym, main_py, env): sym for sym in universe}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            res = future.result()
            results.append(res)
            print(f"[{completed}/{len(universe)}] {res['symbol']} finished. -> Trades: {res['trades']} | WR: {res['win_rate']}% | PnL: Rs {res['pnl']}")
            
    print("\n\n====================== FINAL SUMMARY ======================")
    print(f"{'SYMBOL':<20} | {'TRADES':<10} | {'WIN RATE':<10} | {'PNL':<15}")
    print("-" * 62)
    tot_trades = 0
    
    for row in sorted(results, key=lambda x: x["symbol"]):
        t = row["trades"]
        if isinstance(t, int): tot_trades += t
        print(f"{row['symbol']:<20} | {row['trades']:<10} | {str(row['win_rate'])+'%':<10} | Rs {row['pnl']:<15}")
        
    print("-" * 62)
    print(f"{'TOTAL TRADES':<20} | {tot_trades:<10} | {'':<10} |")
    print("===========================================================")

if __name__ == "__main__":
    main()
