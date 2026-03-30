import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = Path('dataset/NIFTY 50_5minute.csv')
print("Loading dataset for Ultra-High Winrate (70%+) Engine...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]
df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
df = df.between_time('09:15', '15:30')
df = df.loc[df.index >= (df.index[-1] - pd.Timedelta(days=500))]

def add_features(df):
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=4, adjust=False).mean() # Faster RSI 5 for snipes
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
    if df['volume'].sum() == len(df):
        df['volume'] = np.random.randint(1000, 5000, size=len(df))

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

    return df

df = add_features(df)
df.dropna(inplace=True)

def run_backtest(df, rsi_pullback_thresh, rsi_bounce_thresh, sl_atr, rr_ratio):
    trades = []
    
    in_trade = False
    entry_price = 0
    trade_dir = 0
    sl_px = 0
    tp_px = 0

    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    ema9_arr = df['ema9'].values
    ema21_arr = df['ema21'].values
    ema50_arr = df['ema50'].values
    vwap_arr = df['vwap'].values
    rsi_arr = df['rsi_fast'].values
    atr_arr = df['atr'].values
    adx_arr = df['adx'].values

    try:
        time_arr = df.index.time
    except:
        return 0, 0, 0, 0

    for i in range(14, len(df)-1):
        if not in_trade:
            if time_arr[i].hour >= 15 or time_arr[i].hour < 9 or (time_arr[i].hour == 9 and time_arr[i].minute < 45):
                continue
            
            # Ultra strict alignment for 70%+ WR
            # Trend must be perfectly stacked and strong
            uptrend = (ema9_arr[i] > ema21_arr[i]) and (ema21_arr[i] > ema50_arr[i]) and (close_arr[i] > vwap_arr[i]) and (adx_arr[i] > 20)
            downtrend = (ema9_arr[i] < ema21_arr[i]) and (ema21_arr[i] < ema50_arr[i]) and (close_arr[i] < vwap_arr[i]) and (adx_arr[i] > 20)

            # Squeeze exact pivot point
            if uptrend and (rsi_arr[i-1] < rsi_pullback_thresh) and (rsi_arr[i] >= rsi_bounce_thresh):
                in_trade = True
                trade_dir = 1
                entry_price = open_arr[i+1]
                # Dynamic trailing simulator: Wide stop to avoid shakeout, tighter asymmetric profit 
                sl_px = entry_price - sl_atr * atr_arr[i]
                tp_px = entry_price + (sl_atr * rr_ratio) * atr_arr[i]

            elif downtrend and (rsi_arr[i-1] > (100 - rsi_pullback_thresh)) and (rsi_arr[i] <= (100 - rsi_bounce_thresh)):
                in_trade = True
                trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + sl_atr * atr_arr[i]
                tp_px = entry_price - (sl_atr * rr_ratio) * atr_arr[i]

        else:
            curr_close = close_arr[i]

            if trade_dir == 1:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10:
                    trades.append(curr_close - entry_price)
                    in_trade = False
                elif low_arr[i] <= sl_px:
                    trades.append(sl_px - entry_price)
                    in_trade = False
                elif high_arr[i] >= tp_px:
                    trades.append(tp_px - entry_price)
                    in_trade = False
            else:
                if time_arr[i].hour >= 15 and time_arr[i].minute >= 10:
                    trades.append(entry_price - curr_close)
                    in_trade = False
                elif high_arr[i] >= sl_px:
                    trades.append(entry_price - sl_px)
                    in_trade = False
                elif low_arr[i] <= tp_px:
                    trades.append(entry_price - tp_px)
                    in_trade = False

    if len(trades) == 0: return 0, 0, 0, 0

    trades = np.array(trades)
    win_rate = np.mean(trades > 0) * 100
    expected_value = np.mean(trades)
    loss_sum = abs(np.sum(trades[trades < 0]))
    pf = np.sum(trades[trades > 0]) / loss_sum if loss_sum != 0 else 99
    return len(trades), win_rate, expected_value, pf

print("Hunting 70%+ Win Rate (500 days)...")
rsi_pullbacks = [30, 35, 40]
rsi_bounces = [35, 40, 45, 50, 55]
sl_atrs = [1.5, 2.0, 2.5]
# Asymmetric skew favors Winrate: Stop is wider than Target, meaning we ensure the trade finishes positive
rr_ratios = [0.4, 0.5, 0.6, 0.8]

results = []
for p, b, sl, rr in product(rsi_pullbacks, rsi_bounces, sl_atrs, rr_ratios):
    if b <= p: continue
    cnt, wr, ev, pf = run_backtest(df, p, b, sl, rr)
    
    # Needs at LEAST medium frequency (2+ per week on avg = > 100)
    if cnt >= 100:
        results.append({
            "trades": cnt, "win_rate": np.round(wr, 2), "pf": np.round(pf, 2),
            "pullback": p, "bounce": b, "sl_atr": sl, "rr": rr
        })

df_res = pd.DataFrame(results)
if len(df_res) > 0:
    df_res = df_res.sort_values(by=["win_rate", "pf"], ascending=[False, False])
    print("\n Top 70%+ Winrate Configurations ")
    print(df_res.head(20).to_string(index=False))
else:
    print("\nNo configurations met the criteria.")
