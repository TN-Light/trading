import pandas as pd
import numpy as np
from pathlib import Path

def load_and_prep_data(symbol, timeframe='5min'):
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

def generate_alpha_features(df):
    # Previous Day High/Low
    daily = df.groupby('date').agg({'high': 'max', 'low': 'min'}).shift(1).rename(columns={'high': 'pdh', 'low': 'pdl'})
    df = df.reset_index().merge(daily, on='date', how='left').set_index('datetime')
    
    # Intraday High/Low up to current bar
    df['hod'] = df.groupby('date')['high'].cummax()
    df['lod'] = df.groupby('date')['low'].cummin()
    
    # VWAP & StdDev Bands
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume'] = df['volume'].replace(0, 1) 
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    df['cum_pv2'] = ((df['typical_price'] ** 2) * df['volume']).groupby(df['date']).cumsum()
    df['vwap_var'] = np.maximum(0, (df['cum_pv2'] / df['cum_vol']) - (df['vwap'] ** 2))
    df['vwap_std'] = np.sqrt(df['vwap_var'])
    df['vwap_z'] = np.where(df['vwap_std'] > 0, (df['close'] - df['vwap']) / df['vwap_std'], 0)
    
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).ewm(com=13).mean() / -df['close'].diff().clip(upper=0).ewm(com=13).mean())))
    
    return df.dropna()

def test_reversal_strategy(df, tax_drag=15.0):
    # Strategy: Extreme Reversal (VWAP > 2.5 Z-score or < -2.5 Z-score, RSI Extreme)
    # Designed for HIGH Win Rate, smaller strict targets
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    z_arr = df['vwap_z'].values
    vwap_arr = df['vwap'].values
    atr_arr = df['atr'].values
    rsi_arr = df['rsi'].values
    
    entry_price, sl_px, tp_px, trade_dir, bars_held = 0, 0, 0, 0, 0

    SL_ATR = 0.5  # TIGHT STOP to reduce Average Net Loss dramatically
    TP_ATR = 1.0  # Conservative Target to ensure high win-rate
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # 9:45 to 14:30 only
            if time_arr[i].hour < 9 or (time_arr[i].hour == 9 and time_arr[i].minute < 45) or time_arr[i].hour > 14:
                continue
                
            # Mean Reversion Long
            if z_arr[i] < -2.5 and rsi_arr[i] < 25:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1]
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                tp_px = entry_price + (TP_ATR * atr_arr[i])
                bars_held = 0
                
            # Mean Reversion Short
            elif z_arr[i] > 2.5 and rsi_arr[i] > 75:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                tp_px = entry_price - (TP_ATR * atr_arr[i])
                bars_held = 0
                
        else:
            bars_held += 1
            exit_px = None
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
            elif trade_dir == 1:
                if low_arr[i] <= sl_px: exit_px = sl_px
                elif high_arr[i] >= tp_px: exit_px = tp_px
            else:
                if high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
                
            if exit_px is not None:
                trades.append({'pnl': (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)})
                in_trade = False
                
    return pd.DataFrame(trades)

def test_sweep_strategy(df, tax_drag=15.0):
    # Strategy: PDH / PDL Liquidity Sweeps
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    pdh_arr = df['pdh'].values
    pdl_arr = df['pdl'].values
    atr_arr = df['atr'].values
    
    entry_price, sl_px, tp_px, trade_dir, bars_held = 0, 0, 0, 0, 0
    
    SL_ATR = 0.5
    TP_ATR = 1.2
    
    for i in range(14, len(df)-1):
        if not in_trade:
            if time_arr[i].hour < 9 or (time_arr[i].hour == 9 and time_arr[i].minute < 30) or time_arr[i].hour > 14:
                continue
                
            # Sweep Short (Took out PDH, then rejected below it)
            if high_arr[i] > pdh_arr[i] and close_arr[i] < pdh_arr[i]:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = high_arr[i] + 1.0 # Stop above the sweep wick
                tp_px = entry_price - (TP_ATR * atr_arr[i])
                bars_held = 0
                
            # Sweep Long (Took out PDL, then rejected above it)
            elif low_arr[i] < pdl_arr[i] and close_arr[i] > pdl_arr[i]:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1]
                sl_px = low_arr[i] - 1.0 # Stop below the sweep wick
                tp_px = entry_price + (TP_ATR * atr_arr[i])
                bars_held = 0
        else:
            bars_held += 1
            exit_px = None
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
            elif trade_dir == 1:
                if low_arr[i] <= sl_px: exit_px = sl_px
                elif high_arr[i] >= tp_px: exit_px = tp_px
            else:
                if high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
            if exit_px is not None:
                trades.append({'pnl': (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)})
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
    print(f"{name:25s} | TRADES: {len(t_df):4d} | WR: {wr:5.2f}% | NET WIN: {avg_w:5.1f} | NET LOSS: {avg_l:5.1f} | PNL: Rs {total_rs:,.0f}")

print('=== 5-MIN TIMEFRAME R&D (TESTING MEAN REVERSION & SWEEPS) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '5min')
    if df is not None:
        df = generate_alpha_features(df)
        df_rev = test_reversal_strategy(df, tax)
        df_swp = test_sweep_strategy(df, tax)
        evaluate(df_rev, tax, f'{sym[:9]} Reversal (Z>2.5)', lot)
        evaluate(df_swp, tax, f'{sym[:9]} Liq Sweep', lot)
