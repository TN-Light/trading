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
    daily = df.groupby('date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    daily['prev_close'] = daily['close'].shift(1)
    daily['gap_pct'] = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
    
    # 09:15 Candle Properties (First 15 minutes)
    first_candle = df[df.index.time == pd.to_datetime('09:15:00').time()]
    fc_agg = first_candle.groupby('date').agg({'open':'first', 'close':'last', 'high':'max', 'low':'min'})
    fc_agg['fc_bullish'] = fc_agg['close'] > fc_agg['open']
    fc_agg['fc_bearish'] = fc_agg['close'] < fc_agg['open']
    fc_agg['fc_range'] = fc_agg['high'] - fc_agg['low']
    fc_agg['fc_high'] = fc_agg['high']
    fc_agg['fc_low'] = fc_agg['low']
    
    df = df.reset_index().merge(daily[['prev_close', 'gap_pct']], on='date', how='left')
    df = df.merge(fc_agg[['fc_bullish', 'fc_bearish', 'fc_range', 'fc_high', 'fc_low']], on='date', how='left').set_index('datetime')
    
    return df.dropna()

def test_gap_drive_algo(df, tax_drag=15.0):
    # Phase 13: Pure Algorithmic Open Drive & Gap Fill Bounce
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    pdc_arr = df['prev_close'].values
    gap_arr = df['gap_pct'].values
    fc_bull = df['fc_bullish'].values
    fc_bear = df['fc_bearish'].values
    fc_r_arr = df['fc_range'].values
    fc_h_arr = df['fc_high'].values
    fc_l_arr = df['fc_low'].values

    entry_price, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # We ONLY enter at 09:30 (Second Candle) for Open Drives
            # Or we look for Gap Fills into PDC until 11:00
            
            trade_taken = False
            
            # STRATEGY A: THE OPEN DRIVE (Trend Day Math)
            if time_arr[i].hour == 9 and time_arr[i].minute == 30:
                # Strong Gap Up + Bullish first candle closing near it's high
                if gap_arr[i] > 0.2 and fc_bull[i] and close_arr[i-1] > (fc_h_arr[i] - 0.2*fc_r_arr[i]):
                    in_trade = True; trade_dir = 1
                    entry_price = open_arr[i+1] 
                    sl_px = fc_l_arr[i] # Stop is below the 1st candle
                    tp_px = entry_price + (2.0 * fc_r_arr[i]) # Target is 2x the opening candle range
                    trade_taken = True
                    
                # Strong Gap Down + Bearish first candle closing near it's low
                elif gap_arr[i] < -0.2 and fc_bear[i] and close_arr[i-1] < (fc_l_arr[i] + 0.2*fc_r_arr[i]):
                    in_trade = True; trade_dir = -1
                    entry_price = open_arr[i+1]
                    sl_px = fc_h_arr[i]
                    tp_px = entry_price - (2.0 * fc_r_arr[i])
                    trade_taken = True
            
            # STRATEGY B: THE GAP FILL BOUNCE (Mean Reversion into Structural Level)
            elif time_arr[i].hour < 11:
                # Up Gap filling down to touching Previous Day Close
                if gap_arr[i] > 0.3 and low_arr[i] <= pdc_arr[i] and close_arr[i] > pdc_arr[i]:
                    in_trade = True; trade_dir = 1
                    entry_price = open_arr[i+1]
                    sl_px = entry_price - 50 # Fixed point structural stop 
                    tp_px = entry_price + 100 # Fixed point target (1:2)
                    trade_taken = True
                    
                # Down Gap filling up connecting PDC
                elif gap_arr[i] < -0.3 and high_arr[i] >= pdc_arr[i] and close_arr[i] < pdc_arr[i]:
                    in_trade = True; trade_dir = -1
                    entry_price = open_arr[i+1]
                    sl_px = entry_price + 50
                    tp_px = entry_price - 100
                    trade_taken = True
                    
        else:
            exit_px = None

            if trade_dir == 1:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif high_arr[i] >= tp_px: exit_px = tp_px
            else:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
                
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
    
    months = 11 * 12 
    t_freq = len(t_df) / months
    
    print(f"{name:20s} | TRADES: {len(t_df):4d} ({t_freq:4.1f}/mo) | WR: {wr:5.2f}% | PF: {pf:4.2f} | NET WIN: {avg_w:5.1f} | NET LOSS: {avg_l:5.1f} | PNL: Rs {total_rs:,.0f}")

print('=== 15-MIN R&D (PHASE 13: OPEN DRIVE & GAP FILL ALGO) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_price_features(df)
        df_tc = test_gap_drive_algo(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} Gap Algo', lot)