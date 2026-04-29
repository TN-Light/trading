import pandas as pd
import numpy as np
from pathlib import Path

def test_mean_rev(symbol, tax_drag=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists():
        print(f"Path not found: {path}")
        return
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    df['time'] = df.index.time
    df['date_only'] = df.index.date
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['atr'] = df['tr'].rolling(14).mean()
    
    df['typ'] = (df['high'] + df['low'] + df['close'])/3
    df['vol'] = df['volume'].replace(0, 1)
    df['pv'] = df['typ'] * df['vol']
    df['cum_pv'] = df.groupby('date_only')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['vol'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    
    # Overextension
    df['dist'] = df['close'] - df['vwap']
    
    df['long_setup'] = (df['dist'] < -2.5 * df['atr']) & (df['close'] > df['open']) # Reversal candle
    df['short_setup'] = (df['dist'] > 2.5 * df['atr']) & (df['close'] < df['open']) 
    
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr = df['atr'].values
    
    long_s = df['long_setup'].values
    short_s = df['short_setup'].values
    vwap = df['vwap'].values
    
    entry_px, sl_px, tp_px, trade_dir = 0, 0, 0, 0

    for i in range(50, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 14 or time_arr[i].hour < 9 or (time_arr[i].hour==9 and time_arr[i].minute<30): continue
            
            if long_s[i]:
                in_trade = True; trade_dir = 1
                entry_px = open_arr[i+1]
                # Target is partway to VWAP, Stop is wide
                tp_px = entry_px + max(25, atr[i] * 1.5)
                sl_px = entry_px - atr[i] * 2.0
                bars_held = 0
            
            elif short_s[i]:
                in_trade = True; trade_dir = -1
                entry_px = open_arr[i+1]
                tp_px = entry_px - max(25, atr[i] * 1.5)
                sl_px = entry_px + atr[i] * 2.0
                bars_held = 0
        else:
            bars_held += 1
            exit_px = None
            if trade_dir == 1:
                if high_arr[i] >= tp_px: exit_px = tp_px
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 15: exit_px = close_arr[i]
            else:
                if low_arr[i] <= tp_px: exit_px = tp_px
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 15: exit_px = close_arr[i]

            if exit_px is not None:
                pnl = (exit_px - entry_px) if trade_dir == 1 else (entry_px - exit_px)
                trades.append({'pnl': pnl, 'bars': bars_held})
                in_trade = False
                
    t_df = pd.DataFrame(trades)
    if len(t_df) == 0: return
    t_df['net'] = t_df['pnl'] - tax_drag
    wins = t_df[t_df['net'] > 0]
    losses = t_df[t_df['net'] <= 0]
    wr = len(wins)/(len(t_df)+1e-9)*100
    pf = wins['net'].sum() / abs(losses['net'].sum()+1e-9)
    print(f"{symbol} (VWAP MeanReversion) - Trades: {len(t_df)}, WR: {wr:.2f}%, PF: {pf:.2f}, AvgNet: {t_df['net'].mean():.2f}")

test_mean_rev('NIFTY 50')