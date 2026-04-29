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
    
    # 1. IB Generation (Expanding IB window to 10:15)
    df['is_morning'] = np.where((df.index.hour == 9) | ((df.index.hour == 10) & (df.index.minute <= 15)), 1, 0)
    morning_data = df[df['is_morning'] == 1]
    
    ib_high = morning_data.groupby('date')['high'].max().reset_index(name='ib_high')
    ib_low = morning_data.groupby('date')['low'].min().reset_index(name='ib_low')
    df = df.reset_index().merge(ib_high, on='date', how='left').merge(ib_low, on='date', how='left').set_index('datetime')
    
    # 2. ADX indicator implementation (Momentum threshold)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['dn_move'] = df['low'].shift(1) - df['low']
    
    df['pdm'] = np.where((df['up_move'] > df['dn_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['ndm'] = np.where((df['dn_move'] > df['up_move']) & (df['dn_move'] > 0), df['dn_move'], 0)
    
    # Using simple moving averages instead of wilders smoothing for speed/simplicity here
    df['pdi'] = 100 * (df['pdm'].rolling(14).mean() / df['atr'])
    df['ndi'] = 100 * (df['ndm'].rolling(14).mean() / df['atr'])
    df['dx'] = 100 * (abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi']))
    df['adx'] = df['dx'].rolling(14).mean()

    # 3. Supertrend (14, 3) Implementation 
    hl2 = (df['high'] + df['low']) / 2
    df['band_up'] = hl2 + (3.0 * df['atr'])
    df['band_dn'] = hl2 - (3.0 * df['atr'])
    
    # Calculate supertrend states (simplified forward-fill logic for backtesting)
    df['trend'] = 1 
    df.loc[df['close'] < df['band_dn'].shift(1), 'trend'] = -1
    df['trend'] = df['trend'].ffill() 

    # 4. Moving Averages
    df['ema8'] = df['close'].ewm(span=8).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    
    # 5. Daily Aggregation for Context
    daily = df.groupby('date').agg({'high':'max', 'low':'min', 'open': 'first', 'close': 'last'})
    daily['prev_close'] = daily['close'].shift(1)
    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    daily['gap'] = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
    daily['gap_up'] = daily['gap'] > 0.1 # Need significant gap, not just fractional
    daily['gap_dn'] = daily['gap'] < -0.1
    
    df = df.reset_index().merge(daily[['gap', 'gap_up', 'gap_dn', 'prev_high', 'prev_low']], on='date', how='left').set_index('datetime')
    
    return df.dropna()

def test_momentum_strategy(df, tax_drag=15.0):
    # Phase 7: Momentum Burst Engine (ADX + Fast EMA Cross + Wide Targets)
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr'].values
    ib_h_arr = df['ib_high'].values
    ib_l_arr = df['ib_low'].values
    pdi_arr = df['pdi'].values
    ndi_arr = df['ndi'].values
    adx_arr = df['adx'].values
    ema8_arr = df['ema8'].values
    ema21_arr = df['ema21'].values
    gap_up_arr = df['gap_up'].values
    gap_dn_arr = df['gap_dn'].values
    trend_arr = df['trend'].values

    entry_price, sl_px, target_px, trade_dir = 0, 0, 0, 0
    max_favorable = 0
    bars_held = 0
    
    SL_ATR = 1.5           # Tighter structural stop since ADX guarantees momentum
    TARGET_ATR_T1 = 2.0    # First target 
    
    for i in range(25, len(df)-1):
        if not in_trade:
            # Entry Window: 10:15 / 10:30 / 10:45 / 11:00 / 11:15
            # Expanding to cover late morning moves
            valid_time = (time_arr[i].hour == 10 and time_arr[i].minute >= 15) or \
                         (time_arr[i].hour == 11 and time_arr[i].minute <= 30)
                         
            if not valid_time: continue
            
            # ADX Momentum Filter: ADX must be rising and strong > 20
            # AND DMI crossover must agree with trend
            strong_momentum = (adx_arr[i] > 20) and (adx_arr[i] > adx_arr[i-1])
            
            # Trend context
            bull_align = (ema8_arr[i] > ema21_arr[i]) and (pdi_arr[i] > ndi_arr[i]) and trend_arr[i] == 1
            bear_align = (ema8_arr[i] < ema21_arr[i]) and (ndi_arr[i] > pdi_arr[i]) and trend_arr[i] == -1
            
            # IB Breakout logic
            if strong_momentum and bull_align and close_arr[i] > ib_h_arr[i]:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] 
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                target_px = entry_price + (TARGET_ATR_T1 * atr_arr[i])
                max_favorable = entry_price
                bars_held = 0
                
            elif strong_momentum and bear_align and close_arr[i] < ib_l_arr[i]:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                target_px = entry_price - (TARGET_ATR_T1 * atr_arr[i])
                max_favorable = entry_price
                bars_held = 0
                
        else:
            bars_held += 1
            
            # EOD Exit (Strict square off at 15:15)
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 15:
                exit_px = open_arr[i]
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir})
                in_trade = False
                continue
                
            # Intraday Update Favorable Excursion
            if trade_dir == 1 and high_arr[i] > max_favorable: max_favorable = high_arr[i]
            elif trade_dir == -1 and low_arr[i] < max_favorable: max_favorable = low_arr[i]
            
            # Breakeven Trap: Move SL to Entry + 5 points automatically if up 1.0 ATR
            be_threshold = 1.0 * atr_arr[i]
            if trade_dir == 1 and (max_favorable - entry_price) > be_threshold:
                sl_px = max(sl_px, entry_price + 5)
            elif trade_dir == -1 and (entry_price - max_favorable) > be_threshold:
                sl_px = min(sl_px, entry_price - 5)

            # Execution Stops / Targets
            exit_px = None
            if trade_dir == 1:
                # Target check
                if high_arr[i] >= target_px:
                    exit_px = target_px
                # Stop Loss Check
                elif low_arr[i] <= sl_px: 
                    exit_px = sl_px
            else:
                if low_arr[i] <= target_px:
                    exit_px = target_px
                elif high_arr[i] >= sl_px:
                    exit_px = sl_px
                
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

print('=== 15-MIN R&D (PHASE 7: ADX > 20 BREAKOUT ENGINE + BREAKEVEN TRAP) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_alpha_features(df)
        df_tc = test_momentum_strategy(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} ADX Boss', lot)