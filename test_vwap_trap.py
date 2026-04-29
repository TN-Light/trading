import pandas as pd
import numpy as np
from pathlib import Path

def test_trap(symbol, tp_val, sl_val):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists():
        return
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna().sort_values('datetime').set_index('datetime')
    
    df['time'] = df.index.time
    df['date_only'] = pd.to_datetime(df.index.date)
    
    df['vol'] = df['volume'].replace(0,1)
    df['pv'] = ((df['high'] + df['low'] + df['close']) / 3) * df['vol']
    df['vwap'] = df.groupby('date_only')['pv'].cumsum() / df.groupby('date_only')['vol'].cumsum()
    
    tr = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
    tr = np.maximum(tr, abs(df['low'] - df['close'].shift(1)))
    df['atr'] = tr.rolling(14).mean()
    
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    vwap = df['vwap'].values
    ema9 = df['ema9'].values
    ema21 = df['ema21'].values
    atr = df['atr'].values
    
    trades = []
    in_trade = False
    entry_px, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    
    for i in range(50, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 15 or time_arr[i].hour < 9 or (time_arr[i].hour==9 and time_arr[i].minute<30): continue
            
            # Micro Pullback Trend Follow
            # Up: Green candle after a Red candle while above VWAP and EMAs are strongly bullish
            if ema9[i] > ema21[i] and close_arr[i] > vwap[i]:
                if close_arr[i] > open_arr[i] and close_arr[i-1] < open_arr[i-1]:
                    in_trade = True; trade_dir = 1
                    entry_px = close_arr[i]
                    tp_px = entry_px + tp_val
                    sl_px = entry_px - sl_val
                    
            elif ema9[i] < ema21[i] and close_arr[i] < vwap[i]:
                if close_arr[i] < open_arr[i] and close_arr[i-1] > open_arr[i-1]:
                    in_trade = True; trade_dir = -1
                    entry_px = close_arr[i]
                    tp_px = entry_px - tp_val
                    sl_px = entry_px + sl_val
                
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
                    
    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        tdf['net'] = tdf['pnl'] - 15  # Tax
        wins = tdf[tdf['net'] > 0]
        wr = len(wins) / len(tdf) * 100
        gross_wr = (tdf['pnl']>0).mean()*100
        pf = wins['net'].sum() / abs(tdf[tdf['net'] < 0]['net'].sum() + 1e-9)
        print(f"TP {tp_val} / SL {sl_val} | TRADES: {len(tdf)} | Net WR: {wr:.2f}% | GrossWR: {gross_wr:.2f}% | PF: {pf:.2f} | Avg Net: {tdf['net'].mean():.2f}")

for t, s in [(20, 60), (25, 50), (15, 45)]:
    test_trap('NIFTY 50', t, s)
