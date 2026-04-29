import pandas as pd
import numpy as np
from pathlib import Path

def generate_signals(symbol, tax_drag=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists():
        print(f"Dataset for {symbol} not found")
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    df['time'] = df.index.time
    df['date_only'] = pd.to_datetime(df.index.date)
    
    # Calculate VWAP
    df['vol'] = df['volume'].replace(0,1)
    df['pv'] = ((df['high'] + df['low'] + df['close']) / 3) * df['vol']
    df['vwap'] = df.groupby('date_only')['pv'].cumsum() / df.groupby('date_only')['vol'].cumsum()
    
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    
    t_up = (df['ema9'] > df['ema21']) & (df['close'] > df['vwap'])
    t_dn = (df['ema9'] < df['ema21']) & (df['close'] < df['vwap'])
    
    # RSI for pullbacks
    delta = df['close'].diff()
    up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=14).mean() / down.ewm(com=14).mean()
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR for volatility sizing 
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['atr'] = df['tr'].rolling(14).mean()
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    rsi = df['rsi'].values
    t_up = t_up.values
    t_dn = t_dn.values
    atr = df['atr'].values
    
    trades = []
    in_trade = False
    
    entry_px, sl_px, tp_px, trade_dir = 0, 0, 0, 0

    for i in range(50, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 14 or time_arr[i].hour < 10: continue
            
# Trend is UP but price just pulled back (RSI dip on 5min)     
            if t_up[i] and rsi[i] < 45 and close_arr[i] > open_arr[i]:
                in_trade = True; trade_dir = 1
                entry_px = open_arr[i+1]
                # Modest safe target (e.g. 25 points to yield +10 after tax)    
                tp_px = entry_px + 28
                # Wide SL to give room (e.g. max 75)
                sl_px = entry_px - 82

            # Trend is DOWN but price just pulled back (RSI rally on 5min) 
            elif t_dn[i] and rsi[i] > 55 and close_arr[i] < open_arr[i]:        
                in_trade = True; trade_dir = -1
                entry_px = open_arr[i+1]
                tp_px = entry_px - 28
                sl_px = entry_px + 82
                
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
    if len(t_df) == 0: return None
    
    t_df['tax'] = tax_drag
    t_df['net'] = t_df['pnl'] - t_df['tax']
    return t_df

def print_metrics(t_df, sym, lot):
    if t_df is None or len(t_df) == 0: return
    wins = t_df[t_df['net'] > 0]
    losses = t_df[t_df['net'] <= 0]
    wr = len(wins)/(len(t_df)+1e-9)*100
    pf = wins['net'].sum() / abs(losses['net'].sum()+1e-9)
    freq = len(t_df) / (11*12) # Approximate months
    print(f"=== {sym} 15k Options Cash Machine ===")
    print(f"Total Trades: {len(t_df)} (~{freq:.1f}/month)")
    if wr > 60:
        print(f"NET WIN RATE: {wr:.2f}% (> 60% Target Achieved)")
    else:
        print(f"WIN RATE: {wr:.2f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Avg Net Points per Trade: {t_df['net'].mean():.2f}")
    print(f"Estimated Net Profit (Rs): Rs {t_df['net'].sum() * lot:,.0f} (Base Cap: 15k)")

df_tc = generate_signals('NIFTY 50')
print_metrics(df_tc, 'NIFTY 50', 65)