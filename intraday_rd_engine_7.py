import pandas as pd
import numpy as np
from pathlib import Path

def load_and_prep_data(symbol, timeframe='15min'):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists(): return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    if timeframe != '5min':
        df = df.resample(timeframe).agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
    df['date'] = df.index.date
    df['time'] = df.index.time
    return df

def generate_alpha_features(df):
    
    # VWAP anchoring (Session VWAP)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume'] = df['volume'].replace(0, 1)
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']

    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    daily = df.groupby('date').agg({'high':'max', 'low':'min', 'open': 'first', 'close': 'last'})
    daily['range'] = daily['high'] - daily['low']
    daily['avg_range_20'] = daily['range'].rolling(20).mean().shift(1)
    # Using 95% threshold for >10 trades / month
    daily['is_compression'] = daily['range'].shift(1) < (0.95 * daily['avg_range_20'])
    
    daily['prev_close'] = daily['close'].shift(1)
    daily['gap_pct'] = abs(daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
    
    df = df.reset_index().merge(daily[['is_compression', 'gap_pct']], on='date', how='left').set_index('datetime')
    
    # Rolling min/max to indicate micro-breakouts within the day
    df['rolling_max_10'] = df['high'].rolling(10).max().shift(1)
    df['rolling_min_10'] = df['low'].rolling(10).min().shift(1)

    return df.dropna()

def test_v8_strategy(df, tax_drag=15.0):
    # Phase 8: Post-Compression VWAP Alignment with High Frequency Micro-Breaks
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr'].values
    vwap_arr = df['vwap'].values
    rmax_arr = df['rolling_max_10'].values
    rmin_arr = df['rolling_min_10'].values
    is_comp_arr = df['is_compression'].values
    gap_pct_arr = df['gap_pct'].values

    entry_price, sl_px, trade_dir = 0, 0, 0
    max_favorable = 0
    
    SL_ATR = 2.0  
    
    for i in range(25, len(df)-1):
        if not in_trade:
            # We want volume/frequency. Allow 09:45 to 13:00 entries
            valid_time = (time_arr[i].hour > 9 or (time_arr[i].hour == 9 and time_arr[i].minute >= 45)) and \
                         (time_arr[i].hour <= 13)
            
            if not valid_time: continue
            
            # The edge: If previous day was compressed AND today did not open with a huge gap
            if not is_comp_arr[i] or gap_pct_arr[i] > 0.4:
                continue

            # Directional Micro-breakout aligned with VWAP
            if close_arr[i] > rmax_arr[i] and close_arr[i] > vwap_arr[i]:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] 
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                max_favorable = entry_price
                
            elif close_arr[i] < rmin_arr[i] and close_arr[i] < vwap_arr[i]:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                max_favorable = entry_price
                
        else:
            
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 15:
                exit_px = open_arr[i]
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir})
                in_trade = False
                continue
                
            if trade_dir == 1 and high_arr[i] > max_favorable: max_favorable = high_arr[i]
            elif trade_dir == -1 and low_arr[i] < max_favorable: max_favorable = low_arr[i]
            
            # Fast breakeven at 1.0 ATR
            be_dist = 1.0 * atr_arr[i]
            if trade_dir == 1 and (max_favorable - entry_price) > be_dist:
                sl_px = max(sl_px, entry_price + 2)
            elif trade_dir == -1 and (entry_price - max_favorable) > be_dist:
                sl_px = min(sl_px, entry_price - 2)

            exit_px = None
            if trade_dir == 1 and low_arr[i] <= sl_px: exit_px = sl_px
            elif trade_dir == -1 and high_arr[i] >= sl_px: exit_px = sl_px
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir})
                in_trade = False
                
    return pd.DataFrame(trades)

def evaluate(t_df, tax, name, lot):
    if len(t_df) == 0: 
        print(f'{name}: No trades.')
        return
    t_df['net_pnl'] = t_df['pnl'] - tax
    wins = t_df[t_df['net_pnl'] > 0]
    losses = t_df[t_df['net_pnl'] <= 0]
    wr = len(wins)/(len(t_df)+1e-9)*100
    avg_w = wins['net_pnl'].mean() if len(wins)>0 else 0
    avg_l = abs(losses['net_pnl'].mean()) if len(losses)>0 else 0
    total_rs = (t_df['net_pnl'] * lot).sum()
    pf = wins['net_pnl'].sum() / abs(losses['net_pnl'].sum()) if len(losses)>0 else 99
    
    months = 11 * 12 # ~132 months
    t_freq = len(t_df) / months
    
    print(f"{name:20s} | TRADES: {len(t_df):4d} ({t_freq:4.1f}/mo) | WR: {wr:5.2f}% | PF: {pf:4.2f} | NET WIN: {avg_w:5.1f} | NET LOSS: {avg_l:5.1f} | PNL: Rs {total_rs:,.0f}")

print('=== 15-MIN R&D (PHASE 8: VOLATILITY COMPRESSION + VWAP + MICRO-BREAKOUT) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_alpha_features(df)
        df_tc = test_v8_strategy(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} MicBreak', lot)