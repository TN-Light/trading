import pandas as pd
import numpy as np
from pathlib import Path

def test_mibb_algo(symbol, tax_drag=15.0):
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists(): return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    # 15min data
    df = df.resample('15min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    df['date'] = df.index.date
    df['time'] = df.index.time
    
    # Needs prev bar data
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['is_inside_bar'] = (df['high'] < df['prev_high']) & (df['low'] > df['prev_low'])
    
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    is_ib_arr = df['is_inside_bar'].values
    ph_arr = df['prev_high'].values
    pl_arr = df['prev_low'].values
    
    entry_price, sl_px, tp_px, trade_dir = 0, 0, 0, 0
    trigger_h, trigger_l = 0, 0
    active_setup = False
    
    for i in range(1, len(df)-1):
        if not in_trade:
            # We look for the first inside bar between 09:30 and 11:00
            if time_arr[i].hour >= 11: 
                active_setup = False
                continue
                
            # If an inside bar is formed, we frame the setup!
            if is_ib_arr[i]:
                trigger_h = high_arr[i]
                trigger_l = low_arr[i]
                active_setup = True
                continue # wait for the next bar to break it
                
            if active_setup:
                # Check for Breakouts of the previously established Inside Bar
                # Use Open-To-Close holding with Stop at the opposite end of the Inside Bar
                
                # Symmetrical Target (2x the risk structural geometry)
                ib_range = trigger_h - trigger_l
                
                if close_arr[i] > trigger_h:
                    in_trade = True; trade_dir = 1
                    entry_price = open_arr[i+1]
                    sl_px = entry_price - ib_range # 100% of IB Range risk
                    tp_px = entry_price + (2.0 * ib_range) # 200% Profit Target
                    active_setup = False
                    
                elif close_arr[i] < trigger_l:
                    in_trade = True; trade_dir = -1
                    entry_price = open_arr[i+1]
                    sl_px = entry_price + ib_range
                    tp_px = entry_price - (2.0 * ib_range)
                    active_setup = False
                    
        else:
            exit_px = None
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 10:
                exit_px = close_arr[i]
            elif trade_dir == 1:
                if high_arr[i] >= tp_px: exit_px = tp_px
                elif low_arr[i] <= sl_px: exit_px = sl_px
            elif trade_dir == -1:
                if low_arr[i] <= tp_px: exit_px = tp_px
                elif high_arr[i] >= sl_px: exit_px = sl_px
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl})
                in_trade = False
                
    return pd.DataFrame(trades)

def evaluate(t_df, tax, name, lot):
    if len(t_df) == 0: return
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

print('=== 15-MIN R&D (PHASE 15: THE INSIDE BAR BREAKOUT MASTER ALGORITHM) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df_tc = test_mibb_algo(sym, tax)
    if df_tc is not None:
        evaluate(df_tc, tax, f'{sym[:9]} Inside Bar', lot)