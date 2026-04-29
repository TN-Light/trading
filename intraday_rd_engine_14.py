import pandas as pd
import numpy as np
from pathlib import Path

def test_time_of_day_algo(symbol, tax_drag=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists(): return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    # Needs 15min resampling for stability
    df = df.resample('15min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    df['date'] = df.index.date
    df['time'] = df.index.time
    
    daily = df.groupby('date').agg({'open':'first'})
    daily = daily.rename(columns={'open':'daily_open'})
    df = df.reset_index().merge(daily, on='date', how='left')
    
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    d_open_arr = df['daily_open'].values
    
    entry_price, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    
    # No ATR. Using pure points. Nifty = 50, Bank = 120, Fin = 50
    pt_stop = 50 if '50' in symbol else (120 if 'BANK' in symbol else 50)
    
    for i in range(1, len(df)-1):
        if not in_trade:
            # THE 10:15 AM LIQUIDITY SWEEP ALGORITHM
            # At exactly 10:15 AM, assess the morning trend relative to the Open price.
            if time_arr[i].hour == 10 and time_arr[i].minute == 15:
                
                # If market pushed UP from the open by a solid margin (liquidity grabbed above)
                if close_arr[i] > (d_open_arr[i] + (pt_stop * 0.5)):
                    in_trade = True; trade_dir = -1
                    entry_price = open_arr[i+1] # Enter exactly at 10:30 open
                    sl_px = entry_price + pt_stop
                    tp_px = d_open_arr[i] # Target is the opening price (Mean Reversion)
                
                # If market flushed DOWN from the open (liquidity grabbed below)
                elif close_arr[i] < (d_open_arr[i] - (pt_stop * 0.5)):
                    in_trade = True; trade_dir = 1
                    entry_price = open_arr[i+1]
                    sl_px = entry_price - pt_stop
                    tp_px = d_open_arr[i] # Target is the opening price
        else:
            exit_px = None
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 10:
                exit_px = close_arr[i]
            elif trade_dir == 1:
                if low_arr[i] <= sl_px: exit_px = sl_px
                elif high_arr[i] >= tp_px: exit_px = tp_px
            elif trade_dir == -1:
                if high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl})
                in_trade = False
                
    return pd.DataFrame(trades)

def evaluate(t_df, tax, name, lot):
    if len(t_df) == 0: return
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

print('=== 15-MIN R&D (PHASE 14: 10:15 AM LIQUIDITY SWEEP MEAN REVERSION) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df_tc = test_time_of_day_algo(sym, tax)
    if df_tc is not None:
        evaluate(df_tc, tax, f'{sym[:9]} 10:15 Sweep', lot)