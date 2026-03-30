import pandas as pd
import numpy as np

def generate_signals(df):
    """
    HOLY GRAIL ENGINE - Momentum Pullback with Smart Time Stops
    Fixed Asymmetric Risk Profile by capping maximum holding time.
    """
    # 1. Feature Generation
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=4, adjust=False).mean()
    ema_down = down.ewm(com=4, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi_fast'] = 100 - (100 / (1 + rs))

    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    df['tr'] = pd.DataFrame({'1':tr1, '2':tr2, '3':tr3}).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume'] = df['volume'].replace(0, 1)
    df['date'] = df.index.date
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']

    df['adx_up'] = df['high'] - df['high'].shift(1)
    df['adx_down'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['adx_up'] > df['adx_down']) & (df['adx_up'] > 0), df['adx_up'], 0)
    df['minus_dm'] = np.where((df['adx_down'] > df['adx_up']) & (df['adx_down'] > 0), df['adx_down'], 0)
    atr14 = df['tr'].rolling(14).mean()
    plus_di = 100 * (pd.Series(df['plus_dm']).ewm(alpha=1/14).mean() / atr14)
    minus_di = 100 * (pd.Series(df['minus_dm']).ewm(alpha=1/14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    # Optimal Strategy Configuration
    PULLBACK = 40
    BOUNCE = 45
    SL_ATR = 2.5
    TP_ATR = 0.8
    TIME_STOP_BARS = 16  # The critical fix! Maps to 80 minutes hold time max.
    
    trades = []
    in_trade = False
    
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    ema21_arr = df['ema21'].values
    ema9_arr = df['ema9'].values
    vwap_arr = df['vwap'].values
    rsi_arr = df['rsi_fast'].values
    atr_arr = df['atr'].values
    adx_arr = df['adx'].values
    time_arr = df.index.time
    
    entry_price, sl_px, tp_px, trade_dir, bars_held = 0, 0, 0, 0, 0

    for i in range(14, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 15 or time_arr[i].hour < 9 or (time_arr[i].hour == 9 and time_arr[i].minute < 30):
                continue
            
            uptrend = (ema9_arr[i] > ema21_arr[i]) and (close_arr[i] > vwap_arr[i]) and (adx_arr[i] > 15)
            downtrend = (ema9_arr[i] < ema21_arr[i]) and (close_arr[i] < vwap_arr[i]) and (adx_arr[i] > 15)

            if uptrend and (rsi_arr[i-1] < PULLBACK) and (rsi_arr[i] >= BOUNCE):
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1]
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                tp_px = entry_price + (TP_ATR * atr_arr[i])
                bars_held = 0

            elif downtrend and (rsi_arr[i-1] > (100 - PULLBACK)) and (rsi_arr[i] <= (100 - BOUNCE)):
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                tp_px = entry_price - (TP_ATR * atr_arr[i])
                bars_held = 0

        else:
            bars_held += 1
            curr_close = close_arr[i]
            exit_px = None

            if trade_dir == 1:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif high_arr[i] >= tp_px: exit_px = tp_px
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close  # The Time Stop triggers
                
                if exit_px is not None:
                    trades.append({'pnl': exit_px - entry_price, 'bars_held': bars_held})
                    in_trade = False
            else:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close  # The Time Stop triggers
                
                if exit_px is not None:
                    trades.append({'pnl': entry_price - exit_px, 'bars_held': bars_held})
                    in_trade = False

    return trades

if __name__ == "__main__":
    from pathlib import Path
    
    df = pd.read_csv(Path('dataset/NIFTY 50_5minute.csv'))
    df.columns = [c.lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    df = df.between_time('09:15', '15:30')
    all_days = df.index.normalize().unique()
    if len(all_days) > 500: df = df.loc[df.index >= all_days[-500]]
    
    trades = generate_signals(df)
    t_df = pd.DataFrame(trades)
    
    print("--- ULTIMATE HOLY GRAIL METRICS (TIME STOP INCLUDED) ---")
    print(f"Trades Count: {len(t_df)}")
    print(f"Win Rate: {(t_df['pnl'] > 0).mean() * 100:.2f}%")
    print(f"Average Win: {t_df[t_df['pnl'] > 0]['pnl'].mean():.2f} pts")
    print(f"Average Loss: {abs(t_df[t_df['pnl'] < 0]['pnl'].mean()):.2f} pts")
    loss_sum = abs(t_df[t_df['pnl'] < 0]['pnl'].sum())
    pf = t_df[t_df['pnl'] > 0]['pnl'].sum() / loss_sum if loss_sum != 0 else 99
    print(f"Profit Factor: {pf:.2f}")
    print(f"Average Bars Held: {t_df['bars_held'].mean():.1f} bars (~{t_df['bars_held'].mean()*5:.0f} mins)")
