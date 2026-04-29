import pandas as pd
import numpy as np
from pathlib import Path

def load_and_prep_data(symbol, timeframe='15min'):
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
    
    # 1. Structural Regime Context 
    daily = df.groupby('date').agg({'high':'max', 'low':'min', 'open': 'first', 'close': 'last'})
    daily['prev_close'] = daily['close'].shift(1)
    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    
    # Gap Bias
    daily['gap_pct'] = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
    daily['is_gap_up'] = daily['gap_pct'] > 0.15 # Strong gap up
    daily['is_gap_dn'] = daily['gap_pct'] < -0.15 # Strong gap down
    
    # Range Context
    daily['range'] = daily['high'] - daily['low']
    daily['range_ma_10'] = daily['range'].rolling(10).mean().shift(1)
    daily['is_compression'] = daily['range'].shift(1) < daily['range_ma_10'] # Relaxed filter (just < average)
    
    df = df.reset_index().merge(daily[['is_gap_up', 'is_gap_dn', 'is_compression', 'prev_high', 'prev_low']], on='date', how='left').set_index('datetime')

    # 2. VWAP and Momentum
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume'] = df['volume'].replace(0, 1)
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['session_vwap'] = df['cum_pv'] / df['cum_vol']
    
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    return df.dropna()

def test_v10_strategy(df, tax_drag=15.0):
    # Phase 10: "The Holy Grail Architecture"
    # - Strict Gap fading/following rules
    # - Compression filter for frequency > 10
    # - Structural ATR Risk (2.0)
    # - Asymmetric trailing stop
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr'].values
    vwap_arr = df['session_vwap'].values
    ema9_arr = df['ema_9'].values
    ema21_arr = df['ema_21'].values
    
    gap_up_arr = df['is_gap_up'].values
    gap_dn_arr = df['is_gap_dn'].values
    comp_arr = df['is_compression'].values
    pd_h_arr = df['prev_high'].values
    pd_l_arr = df['prev_low'].values

    entry_price, sl_px, target_px, trade_dir = 0, 0, 0, 0
    max_favorable = 0
    
    SL_ATR = 1.0  # Tightened back to 1.0 because High-WR wide-stops result in terrible PFs
    TARGET_ATR = 3.0 # Aiming for 3:1 RR to rescue the PF
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # Entry Window: 09:30 to 12:00
            valid_time = (time_arr[i].hour == 9 and time_arr[i].minute >= 30) or \
                         (time_arr[i].hour == 10) or \
                         (time_arr[i].hour == 11)
            
            if not valid_time: continue
            
            if not comp_arr[i]: continue # Ensure we trade mainly off compressed days

            # Strategy 1: Gap Fading into VWAP
            # If gap up, but breaking below EMA 9 and VWAP, fade it targeting previous day close (which is assumed below PD_H mostly)
            is_fading_up = gap_up_arr[i] and close_arr[i] < ema9_arr[i] and close_arr[i] < vwap_arr[i] and close_arr[i] < pd_h_arr[i]
            
            # Strategy 2: Gap Following
            # If gap up, and accelerating above EMA 9 and VWAP
            is_following_up = gap_up_arr[i] and close_arr[i] > ema9_arr[i] and ema9_arr[i] > ema21_arr[i] and close_arr[i] > pd_h_arr[i]
            
            is_fading_dn = gap_dn_arr[i] and close_arr[i] > ema9_arr[i] and close_arr[i] > vwap_arr[i] and close_arr[i] > pd_l_arr[i]
            is_following_dn = gap_dn_arr[i] and close_arr[i] < ema9_arr[i] and ema9_arr[i] < ema21_arr[i] and close_arr[i] < pd_l_arr[i]

            if is_following_up or is_fading_dn:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] 
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                target_px = entry_price + (TARGET_ATR * atr_arr[i])
                max_favorable = entry_price
                
            elif is_following_dn or is_fading_up:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                target_px = entry_price - (TARGET_ATR * atr_arr[i])
                max_favorable = entry_price
                
        else:
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 15:
                exit_px = open_arr[i]
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir})
                in_trade = False
                continue
                
            if trade_dir == 1 and high_arr[i] > max_favorable: max_favorable = high_arr[i]
            elif trade_dir == -1 and low_arr[i] < max_favorable: max_favorable = low_arr[i]
            
            # Breakeven Trail at 1.0 ATR
            be_dist = 1.0 * atr_arr[i]
            if trade_dir == 1 and (max_favorable - entry_price) > be_dist:
                sl_px = max(sl_px, entry_price + tax_drag + 1)
            elif trade_dir == -1 and (entry_price - max_favorable) > be_dist:
                sl_px = min(sl_px, entry_price - tax_drag - 1)

            exit_px = None
            if trade_dir == 1:
                if high_arr[i] >= target_px: exit_px = target_px
                elif low_arr[i] <= sl_px: exit_px = sl_px
            elif trade_dir == -1:
                if low_arr[i] <= target_px: exit_px = target_px
                elif high_arr[i] >= sl_px: exit_px = sl_px
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir})
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
    
    months = 11 * 12 # ~132 months
    t_freq = len(t_df) / months
    
    print(f"{name:20s} | TRADES: {len(t_df):4d} ({t_freq:4.1f}/mo) | WR: {wr:5.2f}% | PF: {pf:4.2f} | NET WIN: {avg_w:5.1f} | NET LOSS: {avg_l:5.1f} | PNL: Rs {total_rs:,.0f}")

print('=== 15-MIN R&D (PHASE 10: GAP STRATEGIES + ATR RISK ARCHITECTURE) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_alpha_features(df)
        df_tc = test_v10_strategy(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} Gap Alpha', lot)