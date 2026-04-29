import pandas as pd
df = pd.read_csv("NIFTY_BANK_5minute_anchored.csv")
df['datetime'] = pd.to_datetime(df['timestamp'])
df.set_index('datetime', inplace=True)
df['time'] = df.index.time
df['date_only'] = pd.to_datetime(df.index.date)

box_highs, box_lows = {}, {}

for date, group in df.groupby('date_only'):
    window = group[(group['time'] >= pd.to_datetime('11:30').time()) &
                   (group['time'] <= pd.to_datetime('13:25').time())]
    if len(window) >= 10:
        box_highs[date] = window['high'].max()
        box_lows[date] = window['low'].min()

df['box_high'] = df['date_only'].map(box_highs)
df['box_low'] = df['date_only'].map(box_lows)

trades = 0
for i in range(len(df)):
    t = df['time'].iloc[i]
    if (t.hour == 13 and t.minute >= 30) or (t.hour == 14 and t.minute <= 30):
        h = df['high'].iloc[i]
        bh = df['box_high'].iloc[i]
        c = df['close'].iloc[i]
        o = df['open'].iloc[i]
        if h > bh and c < bh and c < o: trades += 1
        
print("Short traps found:", trades)

trades2 = 0
for i in range(len(df)):
    t = df['time'].iloc[i]
    if (t.hour == 13 and t.minute >= 30) or (t.hour == 14 and t.minute <= 30):
        l = df['low'].iloc[i]
        bl = df['box_low'].iloc[i]
        c = df['close'].iloc[i]
        o = df['open'].iloc[i]
        if l < bl and c > bl and c > o: trades2 += 1

print("Long traps found:", trades2)
