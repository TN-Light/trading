import pandas as pd
import numpy as np
from pathlib import Path

def test_engine(symbol, tax_drag=15.0):
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
    
    # Features
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # VWAP
    df['typ'] = (df['high'] + df['low'] + df['close'])/3
    df['vol'] = df['volume'].replace(0, 1)
    df['pv'] = df['typ'] * df['vol']
    df['cum_pv'] = df.groupby('date_only')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['vol'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    
    # Strategy: VWAP Bounce in Strong Trend
    df['trend_up'] = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema50']) & (df['ema50'] > df['vwap'])
    df['trend_down'] = (df['ema9'] < df['ema21']) & (df['ema21'] < df['ema50']) & (df['ema50'] < df['vwap'])
    
    # Touch VWAP but close above/below
    df['vwap_touch_up'] = (df['low'] <= df['vwap']) & (df['close'] > df['vwap'])
    df['vwap_touch_down'] = (df['high'] >= df['vwap']) & (df['close'] < df['vwap'])
    
    # Run test for VWAP Bounce
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    t_up = df['trend_up'].values
    t_dn = df['trend_down'].values
    v_up = df['vwap_touch_up'].values
    v_dn = df['vwap_touch_down'].values
    atr = df['atr'].values
    
    entry_px, sl_px, tp_px, trade_dir = 0, 0, 0, 0

    for i in range(50, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 14 or time_arr[i].hour < 9 or (time_arr[i].hour==9 and time_arr[i].minute<45): continue
            
            if t_up[i] and v_up[i]:
                in_trade = True
                trade_dir = 1
                entry_px = open_arr[i+1]
                sl_px = entry_px - atr[i] * 1.5  
                tp_px = entry_px + atr[i] * 2.0  
                bars_held = 0
            
            elif t_dn[i] and v_dn[i]:
                in_trade = True
                trade_dir = -1
                entry_px = open_arr[i+1]
                sl_px = entry_px + atr[i] * 1.5
                tp_px = entry_px - atr[i] * 2.0
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
    if len(t_df) == 0:
        print("No trades")
        return
    t_df['net'] = t_df['pnl'] - tax_drag
    wins = t_df[t_df['net'] > 0]
    losses = t_df[t_df['net'] <= 0]
    wr = len(wins)/(len(t_df)+1e-9)*100
    pf = wins['net'].sum() / abs(losses['net'].sum()+1e-9)
    print(f"{symbol} - Trades: {len(t_df)}, WR: {wr:.2f}%, PF: {pf:.2f}, AvgNet: {t_df['net'].mean():.2f}")

test_engine('NIFTY 50')
test_engine('BANK NIFTY')
test_engine('SENSEX')