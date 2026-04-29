import pandas as pd
import numpy as np
from pathlib import Path

def generate_cash_machine_signals(symbol='NIFTY 50', capital=15000, slippage_tax=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    df = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df['time'] = df.index.time
    
    df['ema21'] = df['close'].ewm(span=21).mean()
    tr = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
    tr = np.maximum(tr, abs(df['low'] - df['close'].shift(1)))
    df['atr'] = tr.rolling(14).mean()
    
    df['dist'] = (df['close'] - df['ema21']) / df['atr']
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    dist = df['dist'].values
    
    trades = []
    in_trade = False
    
    entry_px, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    TARGET_POINTS = 30
    STOP_LOSS_POINTS = 60
    
    for i in range(50, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 14 or time_arr[i].hour < 9 or (time_arr[i].hour==9 and time_arr[i].minute<30): continue
            
            if dist[i] < -1.5 and close_arr[i] > open_arr[i]:
                in_trade = True; trade_dir = 1
                entry_px = close_arr[i]
                tp_px = entry_px + TARGET_POINTS
                sl_px = entry_px - STOP_LOSS_POINTS
                
            elif dist[i] > 1.5 and close_arr[i] < open_arr[i]:
                in_trade = True; trade_dir = -1
                entry_px = close_arr[i]
                tp_px = entry_px - TARGET_POINTS
                sl_px = entry_px + STOP_LOSS_POINTS
                
        else:
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
                gross_pnl = (exit_px - entry_px) if trade_dir == 1 else (entry_px - exit_px)
                trades.append({'gross_pts': gross_pnl})
                in_trade = False
                
    tdf = pd.DataFrame(trades)
    tdf['net_pts'] = tdf['gross_pts'] - slippage_tax
    wins = tdf[tdf['net_pts'] > 0]
    wr = len(wins) / len(tdf) * 100
    print(f"{symbol} 15m Reversal - Trades: {len(tdf)}, Net WR: {wr:.2f}%, Avg Net: {tdf['net_pts'].mean():.2f}")

generate_cash_machine_signals('NIFTY 50')