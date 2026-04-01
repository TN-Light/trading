import yaml
import pyotp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from SmartApi import SmartConnect

def get_angel_data():
    with open('prometheus/config/credentials.yaml', 'r') as f:
        creds = yaml.safe_load(f)['angelone']
        
    obj = SmartConnect(api_key=creds['api_key'])
    totp = pyotp.TOTP(creds['totp_secret']).now()
    data = obj.generateSession(creds['client_code'], creds['password'], totp)
    
    if data['status'] == False:
        print("Login failed:", data)
        return None

    # Fetch last 15 days
    from_date = (datetime(2026, 3, 30) - timedelta(days=15)).strftime("%Y-%m-%d %H:%M")
    to_date = "2026-03-30 15:30"
    
    historic_param={
        "exchange": "NSE",
        "symboltoken": "99926000",
        "interval": "FIVE_MINUTE",
        "fromdate": from_date, 
        "todate": to_date
    }
    
    res = obj.getCandleData(historic_param)
    if res['status'] == False:
        print("Data fetch failed:", res)
        return None
        
    df = pd.DataFrame(res['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

def generate_signals(df):
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

    PULLBACK = 40
    BOUNCE = 45
    SL_ATR = 2.5
    TP_ATR = 0.8
    TIME_STOP_BARS = 16
    
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
    date_arr = df.index.date
    idx_arr = df.index
    
    entry_price, sl_px, tp_px, trade_dir, bars_held = 0, 0, 0, 0, 0
    entry_time = None

    for i in range(14, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 15 or time_arr[i].hour < 9 or (time_arr[i].hour == 9 and time_arr[i].minute < 30):
                continue
            
            uptrend = (ema9_arr[i] > ema21_arr[i]) and (close_arr[i] > vwap_arr[i]) and (adx_arr[i] > 15)
            downtrend = (ema9_arr[i] < ema21_arr[i]) and (close_arr[i] < vwap_arr[i]) and (adx_arr[i] > 15)

            if uptrend and (rsi_arr[i-1] < PULLBACK) and (rsi_arr[i] >= BOUNCE):
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1]; entry_time = idx_arr[i+1]
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                tp_px = entry_price + (TP_ATR * atr_arr[i])
                bars_held = 0

            elif downtrend and (rsi_arr[i-1] > (100 - PULLBACK)) and (rsi_arr[i] <= (100 - BOUNCE)):
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]; entry_time = idx_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                tp_px = entry_price - (TP_ATR * atr_arr[i])
                bars_held = 0

        else:
            bars_held += 1
            curr_close = close_arr[i]
            exit_px = None
            exit_reason = ""
            exit_time = idx_arr[i]

            if trade_dir == 1:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close; exit_reason = "EOD"
                elif low_arr[i] <= sl_px: exit_px = sl_px; exit_reason = "SL"
                elif high_arr[i] >= tp_px: exit_px = tp_px; exit_reason = "TP"
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close; exit_reason = "TIME_STOP"
                
                if exit_px is not None:
                    if entry_time.date() == datetime(2026, 3, 30).date():
                        trades.append({'direction': 'LONG', 'entry_time': entry_time, 'entry_price': entry_price, 
                                       'exit_time': exit_time, 'exit_price': exit_px, 'pnl': exit_px - entry_price, 'reason': exit_reason})
                    in_trade = False
            else:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10: exit_px = curr_close; exit_reason = "EOD"
                elif high_arr[i] >= sl_px: exit_px = sl_px; exit_reason = "SL"
                elif low_arr[i] <= tp_px: exit_px = tp_px; exit_reason = "TP"
                elif bars_held >= TIME_STOP_BARS: exit_px = curr_close; exit_reason = "TIME_STOP"
                
                if exit_px is not None:
                    if entry_time.date() == datetime(2026, 3, 30).date():
                        trades.append({'direction': 'SHORT', 'entry_time': entry_time, 'entry_price': entry_price, 
                                       'exit_time': exit_time, 'exit_price': exit_px, 'pnl': entry_price - exit_px, 'reason': exit_reason})
                    in_trade = False

    return trades

df = get_angel_data()
if df is not None:
    df.dropna(inplace=True)
    trades = generate_signals(df)
    print("--- TRADES ON MARCH 30, 2026 ---")
    if not trades:
        print("No trades triggered today.")
    else:
        tdf = pd.DataFrame(trades)
        pd.set_option('display.max_columns', None)
        print(tdf.to_string(index=False))
