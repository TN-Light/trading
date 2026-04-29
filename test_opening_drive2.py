import pandas as pd
import numpy as np
from pathlib import Path

def test_opening_drive(symbol, tax_drag=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    df['time'] = df.index.time
    df['date_only'] = pd.to_datetime(df.index.date)
    
    daily = df.resample('1D').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    daily['trend_up'] = daily['close'] > daily['close'].shift(1)
    daily['trend_dn'] = daily['close'] < daily['close'].shift(1)
    
    df = df.join(daily.shift(1)[['trend_up', 'trend_dn']], on='date_only')
    
    df['vol'] = df['volume'].replace(0,1)
    df['typ'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typ'] * df['vol']
    df['cum_pv'] = df.groupby('date_only')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['vol'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    vwap = df['vwap'].values
    t_up = df['trend_up'].fillna(False).values
    t_dn = df['trend_dn'].fillna(False).values
    
    trades = []
    in_trade = False
    
    entry_px, sl_px, tp_px, trade_dir = 0, 0, 0, 0

    for i in range(50, len(df)-1):
        if not in_trade:
            # Enter precisely at 09:35 (after 1st 5-min candle settles direction)
            if time_arr[i].hour == 9 and time_arr[i].minute == 35:
                # If first 20 mins closes strongly in direction of HTF trend
                if t_up[i] and close_arr[i] > vwap[i] and close_arr[i] > open_arr[i]:
                    in_trade = True; trade_dir = 1
                    entry_px = open_arr[i+1]
                    tp_px = entry_px + 28  # Win 28
                    sl_px = entry_px - 22  # Lose 22
                    
                elif t_dn[i] and close_arr[i] < vwap[i] and close_arr[i] < open_arr[i]:
                    in_trade = True; trade_dir = -1
                    entry_px = open_arr[i+1]
                    tp_px = entry_px - 28
                    sl_px = entry_px + 22
                    
        else:
            if trade_dir == 1:
                if high_arr[i] >= tp_px:
                    trades.append({'pnl': tp_px - entry_px})
                    in_trade = False
                elif low_arr[i] <= sl_px:
                    trades.append({'pnl': sl_px - entry_px})
                    in_trade = False
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10:
                    trades.append({'pnl': close_arr[i] - entry_px})
                    in_trade = False
            else:
                if low_arr[i] <= tp_px:
                    trades.append({'pnl': entry_px - tp_px})
                    in_trade = False
                elif high_arr[i] >= sl_px:
                    trades.append({'pnl': entry_px - sl_px})
                    in_trade = False
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10:
                    trades.append({'pnl': entry_px - close_arr[i]})
                    in_trade = False
                
    t_df = pd.DataFrame(trades)
    if len(t_df) == 0: return
    t_df['net'] = t_df['pnl'] - tax_drag
    wins = t_df[t_df['net'] > 0]
    losses = t_df[t_df['net'] <= 0]
    wr = len(wins)/(len(t_df)+1e-9)*100
    pf = wins['net'].sum() / abs(losses['net'].sum()+1e-9)
    print(f"{symbol} - Trades: {len(t_df)}, WR: {wr:.2f}%, PF: {pf:.2f}, GrossWR: {(t_df['pnl']>0).mean()*100:.2f}%, Avg Net: {t_df['net'].mean():.2f}")

test_opening_drive('NIFTY 50')