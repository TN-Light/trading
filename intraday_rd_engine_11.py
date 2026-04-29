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

def generate_price_features(df):
    # NO INDICATORS. Pure Price Action & Time.
    
    # 1. Daily Context
    daily = df.groupby('date').agg({'high':'max', 'low':'min', 'open': 'first', 'close': 'last'})
    daily['prev_close'] = daily['close'].shift(1)
    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    daily['prev_range'] = daily['high'].shift(1) - daily['low'].shift(1)
    
    # 2. Initial Balance (09:15 to 10:15)
    df['is_ib'] = np.where((df.index.hour == 9) | ((df.index.hour == 10) & (df.index.minute == 0)), 1, 0)
    morning_data = df[df['is_ib'] == 1]
    
    ib_high = morning_data.groupby('date')['high'].max().reset_index(name='ib_high')
    ib_low = morning_data.groupby('date')['low'].min().reset_index(name='ib_low')
    
    df = df.reset_index().merge(ib_high, on='date', how='left').merge(ib_low, on='date', how='left').set_index('datetime')
    df = df.merge(daily[['prev_close', 'prev_high', 'prev_low', 'prev_range']], on='date', how='left')
    
    df['ib_range'] = df['ib_high'] - df['ib_low']
    
    # Compression: IB Range is less than 50% of yesterday's total range (meaning the morning is compressed)
    df['is_compressed_morning'] = df['ib_range'] < (0.50 * df['prev_range'])
    
    return df.dropna()

def test_pure_algo(df, tax_drag=15.0):
    # Phase 11: Pure Price Algorithm + Holy Grail Time Stop (No Indicators)
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    ib_h_arr = df['ib_high'].values
    ib_l_arr = df['ib_low'].values
    ib_r_arr = df['ib_range'].values
    comp_arr = df['is_compressed_morning'].values

    entry_price, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    bars_held = 0
    
    # Using the "Old Holy Grail" mechanism: The TIME STOP
    TIME_STOP_BARS = 6 # 90 minutes max holding time
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # We look for trades passing 10:15 up to 13:00 
            if time_arr[i].hour < 10 or (time_arr[i].hour == 10 and time_arr[i].minute < 15): continue
            if time_arr[i].hour >= 13: continue
            
            # To get enough trades, we don't strictly require high compression, but we can test variations.
            
            # The "Opening Range Fakeout" (Trap) Algorithm
            # Long Trap: Price broke IB High previously, but current close is back BELOW it.
            broke_ib_h_recently = (high_arr[i-1] > ib_h_arr[i-1]) or (high_arr[i] > ib_h_arr[i])
            failed_ib_h = close_arr[i] < ib_h_arr[i]
            
            # Short Trap: Price broke IB Low previously, but current close is back ABOVE it.
            broke_ib_l_recently = (low_arr[i-1] < ib_l_arr[i-1]) or (low_arr[i] < ib_l_arr[i])
            failed_ib_l = close_arr[i] > ib_l_arr[i]
            
            # The "ORB" (Breakout) Algorithm
            orb_up = close_arr[i] > ib_h_arr[i]
            orb_dn = close_arr[i] < ib_l_arr[i]

            # Choose the Algorithm: Let's test Pure ORB with structural logic + Time Stop
            # Why ORB? In Indian Markets, compressed mornings (IB < 50% of prev range) tend to trend PM.
            if comp_arr[i] and orb_up:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] 
                # Pure structural risk: Risk 50% of the IB range
                sl_px = entry_price - (0.5 * ib_r_arr[i])
                # Pure structural target: Match the full IB Range projected upwards (1:2 R:R)
                tp_px = entry_price + (1.0 * ib_r_arr[i])
                bars_held = 0
                
            elif comp_arr[i] and orb_dn:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (0.5 * ib_r_arr[i])
                tp_px = entry_price - (1.0 * ib_r_arr[i])
                bars_held = 0
                
        else:
            bars_held += 1
            curr_close = close_arr[i]
            exit_px = None

            if trade_dir == 1:
                # EOD
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close
                # Stop Loss
                elif low_arr[i] <= sl_px: exit_px = sl_px
                # Take Profit
                elif high_arr[i] >= tp_px: exit_px = tp_px
                # TIME STOP (The secret sauce from the old holy grail)
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close 
                
            else:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir, 'bars': bars_held})
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

print('=== 15-MIN R&D (PHASE 11: PURE PRICE ALGO + HOLY GRAIL TIME STOP) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_price_features(df)
        df_tc = test_pure_algo(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} Pure Algo', lot)