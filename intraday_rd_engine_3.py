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
    # Morning Range (First 60 mins -> 9:15 to 10:15)
    # Using 15 min bars: 9:15, 9:30, 9:45, 10:00 (4 bars)
    df['is_morning'] = np.where((df.index.hour == 9) | ((df.index.hour == 10) & (df.index.minute == 0)), 1, 0)
    morning_data = df[df['is_morning'] == 1]
    
    ib_high = morning_data.groupby('date')['high'].max().reset_index(name='ib_high')
    ib_low = morning_data.groupby('date')['low'].min().reset_index(name='ib_low')
    df = df.reset_index().merge(ib_high, on='date', how='left').merge(ib_low, on='date', how='left').set_index('datetime')
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume'] = df['volume'].replace(0, 1)
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']

    # EMA Trend
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()

    return df.dropna()

def test_open_to_close_strategy(df, tax_drag=15.0):
    # Phase 4: High WR Volatility Trend Open-to-Close
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
    vwap_arr = df['vwap'].values
    ema9_arr = df['ema9'].values
    ema21_arr = df['ema21'].values
    ema50_arr = df['ema50'].values

    entry_price, sl_px, trade_dir, date_of_trade = 0, 0, 0, None
    
    SL_ATR = 3.0  # ULTRA-WIDE stop. Only panic exit if structural reversal occurs.
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # Entry Window: ONLY check at exactly 10:15
            if not (time_arr[i].hour == 10 and time_arr[i].minute == 15):
                continue
            
            # Trend Confluence & Breakout
            trend_up = (ema9_arr[i] > ema21_arr[i]) and (ema21_arr[i] > ema50_arr[i]) and (close_arr[i] > vwap_arr[i])
            trend_dn = (ema9_arr[i] < ema21_arr[i]) and (ema21_arr[i] < ema50_arr[i]) and (close_arr[i] < vwap_arr[i])

            # Explosive breakout Check (> IB High)
            if trend_up and close_arr[i] > ib_h_arr[i]:
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] # execution is 10:30 OPEN
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                # No Target = hold till EOD
                
            elif trend_dn and close_arr[i] < ib_l_arr[i]:
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                
        else:
            # EOD Exit (Exit on the 15:15 bar open)
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 15:
                exit_px = open_arr[i]
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'dir': trade_dir})
                in_trade = False
                continue
                
            # Stop Loss hit during the day
            exit_px = None
            if trade_dir == 1 and low_arr[i] <= sl_px:
                exit_px = sl_px
            elif trade_dir == -1 and high_arr[i] >= sl_px:
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
    
    print(f"{name:20s} | TRADES: {len(t_df):4d} | WR: {wr:5.2f}% | PF: {pf:4.2f} | NET WIN: {avg_w:5.1f} | NET LOSS: {avg_l:5.1f} | PNL: Rs {total_rs:,.0f}")

print('=== 15-MIN TIMEFRAME R&D (PHASE 4: OPEN-TO-CLOSE TREND) ===')
for sym, tax, lot in [('NIFTY 50', 15.0, 25), ('NIFTY BANK', 35.0, 15), ('NIFTY FIN SERVICE', 15.0, 40)]: 
    df = load_and_prep_data(sym, '15min')
    if df is not None:
        df = generate_alpha_features(df)
        df_tc = test_open_to_close_strategy(df, tax)
        evaluate(df_tc, tax, f'{sym[:9]} O2C Trend', lot)