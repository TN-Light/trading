import pandas as pd
df = pd.read_csv("NIFTY_BANK_5minute_anchored.csv")
df['datetime'] = pd.to_datetime(df['timestamp'])
df.set_index('datetime', inplace=True)
df['time'] = df.index.time
df['date_only'] = pd.to_datetime(df.index.date)

box_highs, box_lows, box_widths, valid_boxes = {}, {}, {}, {}

for date, group in df.groupby('date_only'):
    window = group[(group['time'] >= pd.to_datetime('11:30').time()) &
                   (group['time'] <= pd.to_datetime('13:25').time())]

    if len(window) < 10:
        valid_boxes[date] = False
        continue
    box_high = window['high'].max()
    box_low = window['low'].min()
    box_width = box_high - box_low
    box_widths[date] = box_width
    valid_boxes[date] = True

df['box_valid'] = df['date_only'].map(valid_boxes).fillna(False)

print("Valid boxes count:", df['box_valid'].sum())
print("Number of days:", len(df['date_only'].unique()))
