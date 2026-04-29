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
    
    df['is_morning_ib'] = np.where((df.index.hour == 9) | ((df.index.hour == 10) & (df.index.minute == 0)), 1, 0)
    morning_data = df[df['is_morning_ib'] == 1]
    
    ib_high = morning_data.groupby('date')['high'].max().reset_index(name='ib_high')
    ib_low = morning_data.groupby('date')['low'].min().reset_index(name='ib_low')
    df = df.reset_index().merge(ib_high, on='date', how='left').merge(ib_low, on='date', how='left').set_index('datetime')
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    # Get Daily Context
    daily = df.groupby('date').agg({'high':'max', 'low':'min', 'open': 'first', 'close': 'last'})
    daily['range'] = daily['high'] - daily['low']
    daily['prev_close'] = daily['close'].shift(1)
    
    # Identify NR4 (Narrowest Range of Last 4 Days)
    daily['range_1'] = daily['range'].shift(1)
    daily['range_2'] = daily['range'].shift(2)
    daily['range_3'] = daily['range'].shift(3)
    daily['range_4'] = daily['range'].shift(4)
    # To increase frequency, we define "Compression" as the range being narrower than the average of the last 3 days
    daily['avg_range_3'] = (daily['range_1'] + daily['range_2'] + daily['range_3']) / 3
    daily['is_compressed'] = daily['range_1'] < daily['avg_range_3']
    
    df = df.reset_index().merge(daily[['is_compressed']], on='date', how='left').set_index('datetime')
    
    return df.dropna()

def test_v9_strategy(df, tax_drag=15.0):
    # Phase 9: Wide-Stop Trend Holding with Simple Compression Filter (Target: ~10 trades/mo, >45% WR, PF > 1.2)
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr'].values
    ib_h_arr = df['ib_high'].values
    ib_l_arr = df['ib_low'].values
    comp_arr = df['is_compressed'].values

    entry_price, sl_px, trade_dir = 0, 0, 0
    max_favorable = 0
    
    # We maintain wide stops to not get chopped by noise/taxes
    SL_ATR = 3.0  
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # We want volume/frequency. Allow 10:15 to 11:30 entries for morning expansion
            valid_time = (time_arr[i].hour == 10 and time_arr[i].minute >= 15) or \
                         (time_arr[i].hour == 11 and time_arr[i].minute <= 30)
            
            if not valid_time: continue
            
            # Simple, relaxed compression filter. Yesterday must be quieter than average.
            if not comp_arr[i]:
                continue

            # Core Logic: Simple IB Breakout
            if close_arr[i] > ib_h_arr[i]:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] 
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                max_favorable = entry_price
                
            elif close_arr[i] < ib_l_arr[i]:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                max_favorable = entry_price
                
        else:
            
            # Strict EOD Exit at 15:15
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 15:
                exit_px = open_arr[i]
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir})
                in_trade = False
                continue
                
            if trade_dir == 1 and high_arr[i] > max_favorable: max_favorable = high_arr[i]
            elif trade_dir == -1 and low_arr[i] < max_favorable: max_favorable = low_arr[i]
            
            # Breakeven Trail: Lock to Entry + Tax exactly, once profit hits 1.5 ATR
            be_dist = 1.5 * atr_arr[i]
            if trade_dir == 1 and (max_favorable - entry_price) > be_dist:
                sl_px = max(sl_px, entry_price + tax_drag + 1)
            elif trade_dir == -1 and (entry_price - max_favorable) > be_dist:
                sl_px = min(sl_px, entry_price - tax_drag - 1)

            # Intraday SL execution
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

print('=== 15-MIN R&D (PHASE 9: WIDE STOP IB BREAKOUT + SIMPLE COMPRESSION) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_alpha_features(df)
        df_tc = test_v9_strategy(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} Simple IB', lot)