import pandas as pd
import numpy as np
from pathlib import Path

def generate_nmb_signals(symbol='NIFTY BANK', capital=15000, friction_points=5.0):
    """
    Nexus Momentum Burst (NMB) System
    Strictly Time-Based Volatility Expansion (13:30 - 14:30)
    Optimized for Option Buying (< 20 min hold times)
    """
    # 1. Load Data
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists():
        path = Path(f'dataset/BANK NIFTY_5minute.csv')
        if not path.exists():
            print(f"Dataset missing for {symbol}.")
            return None
            
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = 'date' if 'date' in df.columns else 'datetime'
    df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    df['time'] = df.index.time
    df['date_only'] = pd.to_datetime(df.index.date)
    df['vol'] = df['volume'].replace(0, 1)
    
    # 2. Volume Baseline (20-period SMA of Volume)
    df['vol_20ma'] = df['vol'].rolling(20).mean()
    
    # ---------------------------------------------------------
    # NEW ML FEATURE ENGINEERING: CONTEXT & REGIME
    # ---------------------------------------------------------
    
    # 1. Daily Aggregation for ATR and Gap
    daily_df = df.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
    }).dropna()
    
    daily_df['prev_close'] = daily_df['close'].shift(1)
    
    # True Range & 14-Day ATR
    daily_df['tr'] = np.maximum(daily_df['high'] - daily_df['low'], 
                     np.maximum(abs(daily_df['high'] - daily_df['prev_close']), 
                                abs(daily_df['low'] - daily_df['prev_close'])))
    daily_df['atr_14'] = daily_df['tr'].rolling(14).mean()
    
    # Gap Percentage
    daily_df['gap_pct'] = ((daily_df['open'] - daily_df['prev_close']) / daily_df['prev_close']) * 100
    
    # Map Daily metrics back to the 5-minute dataframe
    df['atr_14'] = df['date_only'].map(daily_df['atr_14'])
    df['gap_pct'] = df['date_only'].map(daily_df['gap_pct'])
    
    # 2. Intraday VWAP (Trend Context)
    df['typ_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_price'] = df['typ_price'] * df['vol']
    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['vol'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    
    # 3. Compression Box Build (11:30 to 13:30)
    # We need to know the High and Low of this specific window for every day
    box_highs, box_lows, box_widths, valid_boxes = {}, {}, {}, {}
    
    # --- TEMPORARY RELAXATION FOR ML DATA GENERATION ---
    # box_width <= 140 (Strict) -> box_width <= 250 (Relaxed)
    # avg_vol < avg_vol_20ma -> avg_vol < (1.2 * avg_vol_20ma)
    
    for date, group in df.groupby('date_only'):
        # Filter for the compression window
        window = df[(df['date_only'] == date) & 
                    (df['time'] >= pd.to_datetime('11:30').time()) & 
                    (df['time'] <= pd.to_datetime('13:25').time())] # Up to the candle right before 13:30
        
        if len(window) < 10: # Ensure we have enough data (should be 23 candles)
            valid_boxes[date] = False
            continue
            
        box_high = window['high'].max()
        box_low = window['low'].min()
        box_width = box_high - box_low
        
        avg_vol = window['vol'].mean()
        avg_vol_20ma = window['vol_20ma'].mean()
        
        # --- THE FLOODGATE PROTOCOL FOR ML TRAINING DATA ---
        # We accept ALL days, regardless of how wide the box is.
        # We want the XGBoost model to explicitly learn that wide boxes fail.
        
        if box_width > 0: # Simply ensure the data is valid
            box_highs[date] = box_high
            box_lows[date] = box_low
            box_widths[date] = box_width
            valid_boxes[date] = True
        else:
            valid_boxes[date] = False
            
    df['box_high'] = df['date_only'].map(box_highs)
    df['box_low'] = df['date_only'].map(box_lows)
    df['box_valid'] = df['date_only'].map(valid_boxes).fillna(False)

    # 4. Backtest Engine
    trades = []
    in_trade = False
    entry_px, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    trade_start_idx = 0
    
    # Pre-extract numpy arrays for speed
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    vol_arr = df['vol'].values
    vol_20ma_arr = df['vol_20ma'].values
    time_arr = df['time'].values
    
    box_high_arr = df['box_high'].values
    box_low_arr = df['box_low'].values
    box_valid_arr = df['box_valid'].values
    
    # Pre-extract new arrays for the execution loop speed
    atr_arr = df['atr_14'].values
    gap_arr = df['gap_pct'].values
    vwap_arr = df['vwap'].values
    
    ml_dataset = []
    
    for i in range(20, len(df) - 1):
        if not in_trade:
            # ONLY trade between 13:30 and 14:30
            trade_window = (time_arr[i].hour == 13 and time_arr[i].minute >= 30) or \
                           (time_arr[i].hour == 14 and time_arr[i].minute <= 30)
                           
            if trade_window and box_valid_arr[i]:
                # ---------------------------------------------------------
                # THE LIQUIDITY TRAP REVERSAL (LTR) LOGIC
                # ---------------------------------------------------------
                
                # SHORT TRAP (Retail bought the high breakout, got trapped)
                # Condition: The candle's high pierced the box, but it closed bearishly back inside.
                if high_arr[i] > box_high_arr[i] and close_arr[i] < box_high_arr[i] and close_arr[i] < open_arr[i]:
                    in_trade = True
                    trade_dir = -1  # We go SHORT (Buy Put Option)
                    entry_px = close_arr[i]
                    sl_px = high_arr[i]  # Stop loss is just above the fakeout wick
                    risk = sl_px - entry_px if sl_px > entry_px else 10
                    tp_px = entry_px - (1.5 * risk) # Conservative 1:1.5 target for higher win rate
                    trade_start_idx = i
                    
                    ml_data = {
                        'datetime': df.index[i],
                        'box_width': box_widths.get(df['date_only'].iloc[i], 0),
                        'volume_surge_ratio': vol_arr[i] / (vol_20ma_arr[i] + 1e-9),
                        'day_of_week': df.index[i].dayofweek,
                        'dir': -1,
                        'gap_pct': gap_arr[i],
                        'atr_ratio': box_widths.get(df['date_only'].iloc[i], 0) / (atr_arr[i] + 1e-9),
                        'vwap_dist_pct': ((close_arr[i] - vwap_arr[i]) / vwap_arr[i]) * 100
                    }
                    continue

                # LONG TRAP (Retail sold the low breakdown, got trapped)
                # Condition: The candle's low pierced the box, but it closed bullishly back inside.
                elif low_arr[i] < box_low_arr[i] and close_arr[i] > box_low_arr[i] and close_arr[i] > open_arr[i]:
                    in_trade = True
                    trade_dir = 1   # We go LONG (Buy Call Option)
                    entry_px = close_arr[i]
                    sl_px = low_arr[i] # Stop loss is just below the fakeout wick
                    risk = entry_px - sl_px if entry_px > sl_px else 10
                    tp_px = entry_px + (1.5 * risk) # Conservative 1:1.5 target
                    trade_start_idx = i
                    
                    ml_data = {
                        'datetime': df.index[i],
                        'box_width': box_widths.get(df['date_only'].iloc[i], 0),
                        'volume_surge_ratio': vol_arr[i] / (vol_20ma_arr[i] + 1e-9),
                        'day_of_week': df.index[i].dayofweek,
                        'dir': 1,
                        'gap_pct': gap_arr[i],
                        'atr_ratio': box_widths.get(df['date_only'].iloc[i], 0) / (atr_arr[i] + 1e-9),
                        'vwap_dist_pct': ((close_arr[i] - vwap_arr[i]) / vwap_arr[i]) * 100
                    }
                    continue
                    
        else: # Manage open trade
            bars_held = i - trade_start_idx
            exit_px = None
            hit_target = 0
            
            if trade_dir == 1: # LONG
                if high_arr[i] >= tp_px: 
                    exit_px = tp_px
                    hit_target = 1
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif bars_held >= 4: exit_px = close_arr[i] # 20 min Time Stop!
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
                
                if exit_px is not None:
                    gross_pts = exit_px - entry_px
                    net_pts = gross_pts - friction_points
                    trades.append({'gross_pts': gross_pts, 'net_pts': net_pts, 'bars': bars_held, 'dir': 1})
                    
                    ml_data['target_hit'] = hit_target
                    ml_data['net_pts'] = net_pts
                    ml_dataset.append(ml_data)
                    
                    in_trade = False
                    
            else: # SHORT
                if low_arr[i] <= tp_px: 
                    exit_px = tp_px
                    hit_target = 1
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif bars_held >= 4: exit_px = close_arr[i] # 20 min Time Stop!
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
                
                if exit_px is not None:
                    gross_pts = entry_px - exit_px
                    net_pts = gross_pts - friction_points
                    trades.append({'gross_pts': gross_pts, 'net_pts': net_pts, 'bars': bars_held, 'dir': -1})
                    
                    ml_data['target_hit'] = hit_target
                    ml_data['net_pts'] = net_pts
                    ml_dataset.append(ml_data)
                    
                    in_trade = False

    t_df = pd.DataFrame(trades)
    m_df = pd.DataFrame(ml_dataset)
    
    if t_df.empty:
        print("No trades generated even after relaxation.")
        return
        
    m_df.to_csv('dataset/nmb_ml_training_data.csv', index=False)
    print(f"Saved {len(m_df)} raw trades to 'dataset/nmb_ml_training_data.csv' for ML pipeline.")
        
    wins = tdf_net = t_df[t_df['net_pts'] > 0]
    losses = t_df[t_df['net_pts'] <= 0]
    
    wr = len(wins) / len(t_df) * 100
    pf = wins['net_pts'].sum() / (abs(losses['net_pts'].sum()) + 1e-9)
    net_profit_rs = t_df['net_pts'].sum() * 15 # Bank Nifty Lot (15 for ATM Option Proxy)
    
    print(f"=======================================================")
    print(f" NMB ENGINE (Momentum Burst): {symbol.upper()}")
    print(f"=======================================================")
    print(f" Total Trades : {len(t_df)}")
    print(f" WIN RATE     : {wr:.2f}%")
    print(f" Avg Net Win  : {wins['net_pts'].mean():.2f} pts")
    print(f" Avg Net Loss : {abs(losses['net_pts'].mean()):.2f} pts")
    print(f" Average Hold : {t_df['bars'].mean()*5:.0f} Minutes")
    print(f" Profit Factor: {pf:.2f}")
    print(f" Estimated Rs : Rs {net_profit_rs:,.2f} on {capital} Cap")
    print(f"=======================================================\n")
    
if __name__ == '__main__':
    generate_nmb_signals('NIFTY BANK', 15000, 5.0)