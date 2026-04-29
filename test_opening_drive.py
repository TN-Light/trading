import pandas as pd
import numpy as np
from pathlib import Path

def test_opening_drive(symbol, tax_drag=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists():
        return
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    df['time'] = df.index.time
    df['date_only'] = df.index.date
    
    daily = df.resample('1D').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    daily['trend_up'] = daily['close'] > daily['close'].shift(1)
    daily['trend_dn'] = daily['close'] < daily['close'].shift(1)
    
    df = df.join(daily.shift(1)[['trend_up', 'trend_dn']], on='date_only')
    
    df['vwap'] = (df['high'] + df['low'] + df['close'])/3 * df['volume'].replace(0,1)
    df['cum_pv'] = df.groupby('date_only')['vwap'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['volume'].replace(0,1).cumsum()
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
            # Entry strictly between 09:30 and 10:00
            if time_arr[i].hour == 9 and time_arr[i].minute == 30:
                # Up setup: Yesterday up, today's first 15 mins closed > VWAP and bullish
                if t_up[i] and close_arr[i] > vwap[i] and close_arr[i] > open_arr[i]:
                    in_trade = True; trade_dir = 1
                    entry_px = open_arr[i+1]
                    tp_px = entry_px + 28  # Target
                    sl_px = entry_px - 22  # Stop
                    bars_held = 0
                    
                # Down setup: Yesterday down, today's first 15 mins closed < VWAP and bearish
                elif t_dn[i] and close_arr[i] < vwap[i] and close_arr[i] < open_arr[i]:
                    in_trade = True; trade_dir = -1
                    entry_px = open_arr[i+1]
                    tp_px = entry_px - 28
                    sl_px = entry_px + 22
                    bars_held = 0
                    
        else:
            bars_held += 1
            exit_px = None
            if trade_dir == 1:
                if high_arr[i] >= tp_px: exit_px = tp_px
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
            else:
                if low_arr[i] <= tp_px: exit_px = tp_px
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]

            if exit_px is not None:
                trades.append({'pnl': (exit_px - entry_px) if trade_dir == 1 else (entry_px - exit_px)})
                in_trade = False
                
    t_df = pd.DataFrame(trades)
    if len(t_df) == 0: return
    t_df['gross'] = t_df['pnl']
    t_df['net'] = t_df['pnl'] - tax_drag
    wins = t_df[t_df['net'] > 0]
    losses = t_df[t_df['net'] <= 0]
    wr = len(wins)/len(t_df)*100
    pf = wins['net'].sum() / abs(losses['net'].sum()+1e-9)
    print(f"{symbol} (Open Drive) - Trades: {len(t_df)}, WR: {wr:.2f}%, Gross WR: {(t_df['gross']>0).mean()*100:.2f}%, PF: {pf:.2f}, AvgNet: {t_df['net'].mean():.2f}, TotalPnL: Rs {t_df['net'].sum()*25:,.0f}")

test_opening_drive('NIFTY 50')