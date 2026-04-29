import pandas as pd
import numpy as np
from pathlib import Path

def test_options_momentum(symbol, tax_drag=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists(): return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    # HTF Filter: Daily Volatility Compression (NR4)
    daily = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    daily['range'] = daily['high'] - daily['low']
    daily['nr4'] = daily['range'] < daily['range'].shift(1).rolling(3).min()
    
    df['date_only'] = pd.to_datetime(df.index.date)
    daily_shifted = daily.shift(1)[['nr4']]
    df = df.join(daily_shifted, on='date_only')
    df['nr4'] = df['nr4'].fillna(False)

    df['time'] = df.index.time

    # Calculate Volume Profile / Imbalance
    df['volume'] = df['volume'].replace(0, 1)
    
    # Volume spike (Relative to 20-bar average)
    df['vol_ma'] = df['volume'].rolling(20).mean().shift(1)
    df['vol_spike'] = df['volume'] > (1.5 * df['vol_ma'])
    
    # Body Size for Explosion
    df['body'] = abs(df['close'] - df['open'])
    df['body_ma'] = df['body'].rolling(20).mean().shift(1)
    df['body_spike'] = df['body'] > (1.5 * df['body_ma'])
    
    df['bull_candle'] = df['close'] > df['open']
    df['bear_candle'] = df['close'] < df['open']

    # Liquidity Sweep: breaking previous 10-bar max/min
    df['recent_max'] = df['high'].rolling(10).max().shift(1)
    df['recent_min'] = df['low'].rolling(10).min().shift(1)
    df['sweep_high'] = df['high'] >= df['recent_max']
    df['sweep_low'] = df['low'] <= df['recent_min']
    
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    vol_spike = df['vol_spike'].values
    body_spike = df['body_spike'].values
    sweep_h = df['sweep_high'].values
    sweep_l = df['sweep_low'].values
    bull_c = df['bull_candle'].values
    bear_c = df['bear_candle'].values
    nr4_arr = df['nr4'].values

    entry_price, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    bars_held = 0
    
    # Options dynamics parameters:
    TIME_STOP_BARS = 3   # 15 minutes max
    PT_STOP = 20         # Tight stop
    PT_TARGET = 35       # Rapid gamma burst target
    
    for i in range(20, len(df)-1):
        if not in_trade:
            # We want to trade during active hours
            if time_arr[i].hour >= 15: continue
            
            # Require HTF Volatility Compression (NR4)
            if not nr4_arr[i]: continue

            # Setup 1: Sweep Rejection (Trapped Traders)
            # Long Setup: Price swept the low, massive volume, but closed bullish and extended
            if sweep_l[i] and bull_c[i] and body_spike[i]:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1]
                sl_px = entry_price - PT_STOP
                tp_px = entry_price + PT_TARGET
                bars_held = 0
                
            # Short Setup: Price swept the high, massive volume, but closed bearish and extended
            elif sweep_h[i] and bear_c[i] and body_spike[i]:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + PT_STOP
                tp_px = entry_price - PT_TARGET
                bars_held = 0
                
        else:
            bars_held += 1
            exit_px = None
            
            # Fast Option Exit Logic
            if trade_dir == 1:
                # EOD or Time Stop
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 15: exit_px = close_arr[i]
                elif bars_held >= TIME_STOP_BARS: exit_px = close_arr[i]
                elif high_arr[i] >= tp_px: exit_px = tp_px
                elif low_arr[i] <= sl_px: exit_px = sl_px
            else:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 15: exit_px = close_arr[i]
                elif bars_held >= TIME_STOP_BARS: exit_px = close_arr[i]
                elif low_arr[i] <= tp_px: exit_px = tp_px
                elif high_arr[i] >= sl_px: exit_px = sl_px
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'bars': bars_held})
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

print('=== 5-MIN R&D (NEXUS 15k OPTIONS CASH MACHINE) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 65)]: # strict test on 65 NIFTY lot
    df_tc = test_options_momentum(sym, tax)
    if df_tc is not None:
        evaluate(df_tc, tax, f'{sym[:9]} 15k Ops', lot)