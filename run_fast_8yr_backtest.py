import os
import sys
import pickle
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Setup path so we can import prometheus modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

CHUNK_DIR = "backtest_chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)

def get_chunks():
    print("Pre-computing chunk intervals from 8-year dataset...")
    # Load primary DF just to get the index bounds for chunking
    df = pd.read_csv('dataset/NIFTY 50_5minute.csv', parse_dates=['date'])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter to last 8 years (2018 - 2026)
    df = df[df['date'].dt.year >= 2018].reset_index(drop=True)
    
    # Chunk by Quarter for fast granular resumptions
    df['Year'] = df['date'].dt.year
    df['Quarter'] = df['date'].dt.quarter
    
    chunks = []
    groups = df.groupby(['Year', 'Quarter'])
    for (year, quarter), group in groups:
        start_idx = group.index.min()
        end_idx = group.index.max()
        chunks.append({
            "year": year,
            "quarter": quarter,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_date": group['date'].min(),
            "end_date": group['date'].max()
        })
    return chunks

def worker_job(chunk):
    y = chunk["year"]
    q = chunk["quarter"]
    chunk_id = f"{y}_Q{q}"
    out_file = os.path.join(CHUNK_DIR, f"trades_{chunk_id}.pkl")
    
    if os.path.exists(out_file):
        print(f"[{chunk_id}] Already processed. Found cache. Skipping.")
        return chunk_id, True, None
    
    print(f"[{chunk_id}] Starting timeframe: {chunk['start_date']} to {chunk['end_date']} ...")
    
    try:
        from prometheus.main import Prometheus
        from prometheus.data.engine import DataEngine
        from loguru import logger
        logger.remove() # Silence noisy logs inside worker threads
        
        # Load slice for this specific process
        df = pd.read_csv('dataset/NIFTY 50_5minute.csv', parse_dates=['date'])
        df.rename(columns={'date': 'timestamp'}, inplace=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[df['timestamp'].dt.year >= 2018].reset_index(drop=True)
        
        # Add 1000 bars (about 14 days) BEFORE start_idx for indicator warmup (RSI, EMA, etc)
        actual_start = max(0, chunk["start_idx"] - 1000)
        data_slice = df.iloc[actual_start:chunk["end_idx"]+1].copy()
        
        # We need daily data for Regime detection checks internally
        # We can dynamically synthesize it from our CSV to ensure no internet bottleneck
        ds = data_slice.set_index("timestamp")
        data_daily = ds.resample('D').agg({
            'open':'first',
            'high':'max',
            'low':'min',
            'close':'last',
            'volume':'sum'
        }).dropna().reset_index()
        
        prom = Prometheus()
        # Ensure 'apex' is injected explicitly
        param_overrides = {"apex": True}
        
        # Mute intra-process standard printing
        import builtins
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        
        # Trigger the precise Backtest execution for this specific slice
        result, engine = prom._run_intraday_backtest_on_slice(
            data_slice=data_slice,
            data_daily=data_daily,
            symbol="NIFTY 50",
            bar_interval="5minute",
            strategy_name=f"APEX_{chunk_id}",
            parrondo=False,
            dd_throttle=True,
            param_overrides=param_overrides,
            verbose=False
        )
        
        # Restore prints
        builtins.print = original_print
        
        # Filter trades that actually occurred in this chunk (exclude the 1000 bar warmup)
        valid_start = chunk["start_date"]
        # Enforce timezone-naive comparisons
        if getattr(valid_start, "tzinfo", None) is not None:
            valid_start = valid_start.tz_localize(None)
            
        chunk_trades = []
        for t in engine.trades:
            entry_dt = pd.to_datetime(t.entry_time)
            if getattr(entry_dt, "tzinfo", None) is not None:
                entry_dt = entry_dt.tz_localize(None)
                
            if entry_dt >= valid_start:
                chunk_trades.append(t)
                
        # Commit to disk (Pickle object payload)
        with open(out_file, "wb") as f:
            pickle.dump(chunk_trades, f)
            
        print(f"[{chunk_id}] Finished beautifully. Found {len(chunk_trades)} trades in Q{q}-{y}.")
        return chunk_id, True, len(chunk_trades)

    except Exception as e:
        # Restore prints in case of error
        import builtins
        builtins.print = getattr(builtins, "original_print", builtins.print)
        err = traceback.format_exc()
        print(f"[{chunk_id}] ERROR FATAL CRASH: {err}")
        return chunk_id, False, err

if __name__ == '__main__':
    print("=" * 60)
    print("   PROMETHEUS 8-YEAR PARALLEL ACCELERATOR")
    print("=" * 60)
    chunks = get_chunks()
    
    # Only use safe CPU max so we don't freeze the system
    num_cores = max(1, min(cpu_count() - 1, 10)) 
    print(f"\nDistributing {len(chunks)} Quarters across {num_cores} Parallel Cores...")
    
    start_time = datetime.now()
    
    # Map jobs to cpu pool
    with Pool(num_cores) as pool:
        results = pool.map(worker_job, chunks)
        
    end_time = datetime.now()
    print(f"\nAll Compute Nodes Terminated. Total time: {end_time - start_time}")
    
    print("\n--- Aggregating Chunk Data ---")
    all_trades = []
    failed = []
    for r in results:
        cid, success, val = r
        if success:
            f_path = os.path.join(CHUNK_DIR, f"trades_{cid}.pkl")
            if os.path.exists(f_path):
                with open(f_path, "rb") as f:
                    chunk_subset = pickle.load(f)
                    all_trades.extend(chunk_subset)
        else:
            failed.append((cid, val))
            
    print(f"Grand Total Trades Captured (2018-2026): {len(all_trades)}")
    if failed:
        print("\nWARNING: Some chunks failed processing!")
        for f in failed:
            print(f"Chunk {f[0]} Error snippet: {str(f[1])[:200]}")
    
    # Compute Final Global Underlying Target Direction Accuracy
    direction_correct_count = sum(1 for t in all_trades if getattr(t, 'underlying_direction_correct', False))
    total_trades = len(all_trades)
    
    print("\n" + "="*50)
    print("    GLOBAL APEX DIRECTIONAL ACCURACY (8-YEARS)")
    print("="*50)
    if total_trades > 0:
        dir_acc = (direction_correct_count / total_trades) * 100
        print(f"   Overall Trades Executed  : {total_trades}")
        print(f"   True Signal Vector Hit   : {direction_correct_count}/{total_trades}")
        print(f"   Accuracy ExpectancyEdge  : {dir_acc:.1f}%")
        print("\n   *(Data securely cached in /backtest_chunks/)*")
    else:
        print("   No valid trades found mathematically across the entire timespan.")
    print("="*50)
