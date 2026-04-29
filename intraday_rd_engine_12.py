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

def generate_price_features(df):
    daily = df.groupby('date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    daily['prev_close'] = daily['close'].shift(1)
    
    # 09:15 to 10:15 Initial Balance
    df['is_ib'] = np.where((df.index.hour == 9) | ((df.index.hour == 10) & (df.index.minute == 0)), 1, 0)
    morning = df[df['is_ib'] == 1]
    
    ib_high = morning.groupby('date')['high'].max().reset_index(name='ib_high')
    ib_low = morning.groupby('date')['low'].min().reset_index(name='ib_low')
    
    df = df.reset_index().merge(ib_high, on='date', how='left').merge(ib_low, on='date', how='left').set_index('datetime')
    df = df.merge(daily[['prev_high', 'prev_low', 'prev_close']], on='date', how='left')
    return df.dropna()

def test_trap_algo(df, tax_drag=15.0):
    # Phase 12: Pure Price Gap + Trap Algorithm
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    ib_h_arr = df['ib_high'].values
    ib_l_arr = df['ib_low'].values

    entry_price, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    bars_held = 0
    TIME_STOP_BARS = 6 
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # Look for trades from 10:15 to 13:30
            if time_arr[i].hour < 10 or (time_arr[i].hour == 10 and time_arr[i].minute < 15): continue
            if time_arr[i].hour > 13: continue
            
            # THE TRAP ALGORITHM (Turtle Soup on IB):
            # A trap is when the previous bar broke the structural high (IB High), 
            # but the CURRENT bar closes back below it. 
            
            # Bear Trap (Trapped Shorts): Look for a move below IB Low that reclaims IB Low
            bear_trap = (low_arr[i-1] < ib_l_arr[i]) and (close_arr[i] > ib_l_arr[i])

            # Bull Trap (Trapped Longs): Look for a move above IB High that loses IB High
            bull_trap = (high_arr[i-1] > ib_h_arr[i]) and (close_arr[i] < ib_h_arr[i])

            trade_taken = False
            
            if bear_trap:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] 
                # SL is the absolute low of the trap sequence
                sl_px = min(low_arr[i-1], low_arr[i]) - 2 
                # Target is the IB High
                tp_px = ib_h_arr[i]
                trade_taken = True
                
            elif bull_trap:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                # SL is the absolute high of the trap sequence
                sl_px = max(high_arr[i-1], high_arr[i]) + 2
                # Target is the IB Low
                tp_px = ib_l_arr[i]
                trade_taken = True
                
            if trade_taken:
                bars_held = 0
                # Filter invalid setups where risk is too large or target is already met
                if trade_dir == 1 and (tp_px <= entry_price or sl_px >= entry_price):
                    in_trade = False
                elif trade_dir == -1 and (tp_px >= entry_price or sl_px <= entry_price):
                    in_trade = False
                    
        else:
            bars_held += 1
            curr_close = close_arr[i]
            exit_px = None

            if trade_dir == 1:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif high_arr[i] >= tp_px: exit_px = tp_px
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close 
            else:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir, 'bars': bars_held})
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

print('=== 15-MIN R&D (PHASE 12: THE TRAPPED TRADER / TURTLE SOUP ALGO) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_price_features(df)
        df_tc = test_trap_algo(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} Trap Algo', lot)