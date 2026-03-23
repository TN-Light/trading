"""Run a full universe backtest and find toxic patterns across ALL trades simultaneously."""
import os
import sys
import yaml
import pandas as pd
from itertools import combinations
import concurrent.futures

# Make sure we can import prometheus
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prometheus.main import Prometheus

def run_single_backtest(symbol):
    try:
        # Import inside the thread to avoid sharing state if Prometheus is not thread-safe
        from prometheus.main import Prometheus
        bot = Prometheus()
        bot.mode = "backtest"
        
        # Determine timeframe
        intraday_cfg = bot.config.get("intraday", {})
        intraday_symbols = intraday_cfg.get("instruments", [])
        is_intraday = symbol in intraday_symbols
        # We actually want just SWING or just INTRADAY? The user said "analysis the losses".
        # Let's assess BOTH or just SWING? Let's do SWING for now, since it's the profitable one 
        # where we want to clean up the bad apples.
        
        # Force swing setting to get the major trades
        data = bot.data.fetch_historical(symbol, days=365, interval="day")
        if data.empty:
            return None
            
        result = bot._run_backtest_on_slice(
            data, symbol, "full_universe", param_overrides=None, verbose=False
        )
        # handle case where it might return a tuple
        if isinstance(result, tuple):
            result, df_trades = result
        else:
            df_trades = getattr(result, "trades_df", None)
            if df_trades is None and hasattr(result, "trades"):
                # If it's a list of dicts/objects
                trade_dicts = [t if isinstance(t, dict) else t.__dict__ for t in result.trades]
                df_trades = pd.DataFrame(trade_dicts)

        if df_trades is not None and not df_trades.empty:
            # Normalize column names for the toxic analysis
            if 'entry' in df_trades.columns and 'entry_time' not in df_trades.columns:
                df_trades = df_trades.rename(columns={'entry': 'entry_time', 'exit': 'exit_time', 'pnl': 'net_pnl'})
            
            if 'net_pnl' not in df_trades.columns:
                df_trades['net_pnl'] = 0.0
            if 'win' not in df_trades.columns:
                df_trades['win'] = df_trades['net_pnl'] > 0
                
            df_trades['symbol'] = symbol
            return df_trades
        return None
    except Exception as e:
        print(f"Error on {symbol}: {e}")
        return None

def find_toxic_patterns_global(df):
    """Scan for patterns exclusively found in losing trades across entire dataset."""
    df['win'] = df['net_pnl'] > 0
    df['year'] = pd.to_datetime(df['entry_time']).dt.year

    signal_cols = [c for c in df.columns if c.startswith('signal_')]
    if signal_cols:
        df['sig_count'] = df[signal_cols].sum(axis=1)
    else:
        df['sig_count'] = df.get('signal_count', 0)

    losers = df[~df.win]
    winners = df[df.win]

    print(f"\n{'#'*80}")
    print(f"  GLOBAL TOXIC PATTERN MINING: ALL SYMBOLS (365 DAYS)")
    print(f"  {len(losers)} losses | {len(winners)} wins | Looking for LOSS-ONLY patterns")
    print(f"{'#'*80}")

    toxic = []

    # === SCORE RANGE + DIRECTION ===
    print(f"\n--- 1. SCORE THRESHOLD + DIRECTION TOXIC ---")
    for d in ['bullish', 'bearish']:
        sub_d = df[df.direction == d]
        sc = 'bull_score' if d == 'bullish' else 'bear_score'
        for lo, hi in [(0.01, 1.0), (1.0, 2.0), (2.0, 2.5), (0.01, 1.5), (0.01, 2.0)]:
            if sc in sub_d.columns:
                mask = (sub_d[sc] >= lo) & (sub_d[sc] < hi)
                sub = sub_d[mask]
                if len(sub) >= 3 and sub.win.sum() == 0:
                    toxic.append({'pattern': f'{d} + {sc} in [{lo},{hi})', 'count': len(sub),
                                  'total_loss': sub.net_pnl.sum()})
                    print(f"  TOXIC: {d} with {sc} in [{lo:.1f}, {hi:.1f}): {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === REGIME + DIRECTION ===
    print(f"\n--- 2. REGIME + DIRECTION TOXIC ---")
    if 'regime_at_entry' in df.columns:
        for direction in df['direction'].unique():
            for regime in df['regime_at_entry'].unique():
                mask = (df.direction == direction) & (df.regime_at_entry == regime)
                sub = df[mask]
                if len(sub) >= 3 and sub.win.sum() == 0:
                    toxic.append({'pattern': f'{direction} + {regime}', 'count': len(sub),
                                  'total_loss': sub.net_pnl.sum()})
                    print(f"  TOXIC: {direction} in {regime}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === SPECIFIC SYMBOL + REGIME ===
    print(f"\n--- 3. SPECIFIC SYMBOL + REGIME ---")
    if 'regime_at_entry' in df.columns:
        for sym in df['symbol'].unique():
            for regime in df['regime_at_entry'].unique():
                mask = (df.symbol == sym) & (df.regime_at_entry == regime)
                sub = df[mask]
                if len(sub) >= 3 and sub.win.sum() == 0:
                    toxic.append({'pattern': f'{sym} + {regime}', 'count': len(sub),
                                  'total_loss': sub.net_pnl.sum()})
                    print(f"  TOXIC: {sym} in {regime}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === SPECIFIC SYMBOL + DIRECTION ===
    print(f"\n--- 4. SPECIFIC SYMBOL + DIRECTION ---")
    for sym in df['symbol'].unique():
        for d in ['bullish', 'bearish']:
            mask = (df.symbol == sym) & (df.direction == d)
            sub = df[mask]
            if len(sub) >= 3 and sub.win.sum() == 0:
                toxic.append({'pattern': f'{sym} + {d}', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum()})
                print(f"  TOXIC: {sym} going {d}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {len(toxic)} TOXIC PATTERNS FOR UNIVERSE")
    print(f"{'='*70}")

    if toxic:
        toxic_df = pd.DataFrame(toxic).sort_values('total_loss')
        print(f"\n  TOP 20 MOST DESTRUCTIVE (sorted by total damage):")
        for _, row in toxic_df.head(20).iterrows():
            print(f"    {row['pattern']:45s} | {row['count']:3d} trades | PnL={row['total_loss']:+8.0f}")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(root, "prometheus", "config", "settings.yaml")
    
    with open(settings_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    indices = cfg.get("market", {}).get("indices", [])
    intraday = cfg.get("intraday", {}).get("instruments", [])
    universe = list(dict.fromkeys(indices + intraday))
    
    print(f"Fetching 365 days of Swing trade data for {len(universe)} symbols...")
    all_dfs = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for df in executor.map(run_single_backtest, universe):
            if df is not None:
                all_dfs.append(df)
                
    if not all_dfs:
        print("No trades found.")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Aggregated {len(master_df)} total trades. Running analysis...")
    
    master_df.to_csv("master_universe_trades_swing.csv", index=False)
    
    find_toxic_patterns_global(master_df)

if __name__ == "__main__":
    main()
