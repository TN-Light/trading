import pandas as pd
import numpy as np
from pathlib import Path

def generate_dtvc_signals(symbol='NIFTY BANK', capital=30000, friction_points=15.0):
    """
    Dual-Timeframe Volatility Capture (DTVC) Engine
    Designed for Bank Nifty High-ATR Environment
    """
    # 1. Load Data
    path = Path(f'dataset/{symbol}_5minute.csv')
    if not path.exists():
        # Try alternate naming
        path = Path(f'dataset/BANK NIFTY_5minute.csv')
        if not path.exists():
            print(f"Dataset missing for {symbol}. Run standard historical fetcher to get 5m data if needed.")
            return None
            
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = 'date' if 'date' in df.columns else 'datetime'
    df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    df['time'] = df.index.time
    df['date_only'] = pd.to_datetime(df.index.date)
    
    # 2. Macro Filter (15m Proxy on 5m Chart)
    # 15m 20-EMA is exactly a 5m 60-EMA
    df['ema_macro'] = df['close'].ewm(span=60, adjust=False).mean()
    
    # Daily VWAP
    df['typ_px'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol'] = df['volume'].replace(0, 1)
    df['pv'] = df['typ_px'] * df['vol']
    df['vwap'] = df.groupby('date_only')['pv'].cumsum() / df.groupby('date_only')['vol'].cumsum()
    
    # Macro Trend Logic
    df['macro_up'] = (df['close'] > df['vwap']) & (df['close'] > df['ema_macro'])
    df['macro_dn'] = (df['close'] < df['vwap']) & (df['close'] < df['ema_macro'])
    
    # 3. Micro Trigger (5m Bollinger Bands & RSI)
    # Bollinger Bands 20, 2
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_up'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_dn'] = df['bb_mid'] - (2 * df['bb_std'])
    
    # RSI 14
    delta = df['close'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    ema_up = up.ewm(alpha=1/14, min_periods=14).mean()
    ema_down = down.ewm(alpha=1/14, min_periods=14).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR 14
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.DataFrame({'1': tr1, '2': tr2, '3': tr3}).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
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
    time_arr = df['time'].values
    
    macro_up_arr = df['macro_up'].values
    macro_dn_arr = df['macro_dn'].values
    bb_up_arr = df['bb_up'].values
    bb_dn_arr = df['bb_dn'].values
    rsi_arr = df['rsi'].values
    atr_arr = df['atr'].values
    
    for i in range(60, len(df) - 1):
        # Time Filters: Don't enter in the first 15 mins or last hour
        valid_entry_time = (time_arr[i].hour > 9 or (time_arr[i].hour == 9 and time_arr[i].minute >= 30)) and \
                           (time_arr[i].hour < 14 or (time_arr[i].hour == 14 and time_arr[i].minute <= 15))
                           
        if not in_trade:
            if not valid_entry_time:
                continue
                
            # LONG ENTRY: 
            # 1. Macro Trend UP
            # 2. Price touched/pierced Lower BB
            # 3. RSI dipped < 40 and hooked up (rsi[i] > rsi[i-1]) or closed green
            if macro_up_arr[i]:
                bb_touch = low_arr[i] <= bb_dn_arr[i]
                rsi_condition = rsi_arr[i] < 45 and (rsi_arr[i] > rsi_arr[i-1]) # Using <45 to avoid missing too many
                reversal_candle = close_arr[i] > open_arr[i]
                
                if bb_touch and rsi_condition and reversal_candle:
                    in_trade = True
                    trade_dir = 1
                    entry_px = close_arr[i]
                    # Dynamic Risk
                    sl_px = entry_px - (1.5 * atr_arr[i])
                    tp_px = entry_px + (2.0 * atr_arr[i])
                    trade_start_idx = i
                    continue
                    
            # SHORT ENTRY:
            if macro_dn_arr[i]:
                bb_touch = high_arr[i] >= bb_up_arr[i]
                rsi_condition = rsi_arr[i] > 55 and (rsi_arr[i] < rsi_arr[i-1]) # Using >55
                reversal_candle = close_arr[i] < open_arr[i]
                
                if bb_touch and rsi_condition and reversal_candle:
                    in_trade = True
                    trade_dir = -1
                    entry_px = close_arr[i]
                    # Dynamic Risk
                    sl_px = entry_px + (1.5 * atr_arr[i])
                    tp_px = entry_px - (2.0 * atr_arr[i])
                    trade_start_idx = i
                    continue
                    
        else: # Manage open trade
            bars_held = i - trade_start_idx
            exit_px = None
            
            if trade_dir == 1: # LONG
                if high_arr[i] >= tp_px: exit_px = tp_px
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif bars_held >= 6: exit_px = close_arr[i] # 30 min Time Stop
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
                
                if exit_px is not None:
                    gross_pts = exit_px - entry_px
                    net_pts = gross_pts - friction_points
                    trades.append({'gross_pts': gross_pts, 'net_pts': net_pts, 'bars': bars_held, 'dir': 1})
                    in_trade = False
                    
            else: # SHORT
                if low_arr[i] <= tp_px: exit_px = tp_px
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif bars_held >= 6: exit_px = close_arr[i] # 30 min Time Stop
                elif time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = close_arr[i]
                
                if exit_px is not None:
                    gross_pts = entry_px - exit_px
                    net_pts = gross_pts - friction_points
                    trades.append({'gross_pts': gross_pts, 'net_pts': net_pts, 'bars': bars_held, 'dir': -1})
                    in_trade = False

    t_df = pd.DataFrame(trades)
    if t_df.empty:
        print("No trades generated.")
        return
        
    wins = tdf_net = t_df[t_df['net_pts'] > 0]
    losses = t_df[t_df['net_pts'] <= 0]
    
    wr = len(wins) / len(t_df) * 100
    pf = wins['net_pts'].sum() / (abs(losses['net_pts'].sum()) + 1e-9)
    net_profit_rs = t_df['net_pts'].sum() * 15 # Bank Nifty Lot
    
    print(f"=======================================================")
    print(f" DTVC ENGINE: {symbol.upper()}")
    print(f"=======================================================")
    print(f" Total Trades : {len(t_df)}")
    print(f" WIN RATE     : {wr:.2f}%")
    print(f" Avg Net Win  : {wins['net_pts'].mean():.2f} pts")
    print(f" Avg Net Loss : {abs(losses['net_pts'].mean()):.2f} pts")
    print(f" Max Drawdown : {abs(losses['net_pts'].min()):.2f} pts (Worst Single Loss)")
    print(f" Average Hold : {t_df['bars'].mean()*5:.0f} Minutes")
    print(f" Profit Factor: {pf:.2f}")
    print(f" TOTAL NET Rs : Rs {net_profit_rs:,.2f} on {capital} Cap")
    print(f"=======================================================\n")
    
if __name__ == '__main__':
    generate_dtvc_signals('NIFTY BANK', 30000, 15.0)
