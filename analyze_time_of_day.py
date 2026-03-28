import os
import glob
import pickle
import pandas as pd

def categorize_time_bin(dt):
    h = dt.hour
    m = dt.minute
    
    # 9:15 to 15:30 breaks down into 30 min intervals
    if h == 9 and 15 <= m < 45: return '09:15 - 09:45'
    elif (h == 9 and m >= 45) or (h == 10 and m < 15): return '09:45 - 10:15'
    elif h == 10 and 15 <= m < 45: return '10:15 - 10:45'
    elif (h == 10 and m >= 45) or (h == 11 and m < 15): return '10:45 - 11:15'
    elif h == 11 and 15 <= m < 45: return '11:15 - 11:45'
    elif (h == 11 and m >= 45) or (h == 12 and m < 15): return '11:45 - 12:15'
    elif h == 12 and 15 <= m < 45: return '12:15 - 12:45'
    elif (h == 12 and m >= 45) or (h == 13 and m < 15): return '12:45 - 13:15'
    elif h == 13 and 15 <= m < 45: return '13:15 - 13:45'
    elif (h == 13 and m >= 45) or (h == 14 and m < 15): return '13:45 - 14:15'
    elif h == 14 and 15 <= m < 45: return '14:15 - 14:45'
    elif (h == 14 and m >= 45) or (h == 15 and m < 15): return '14:45 - 15:15'
    elif h == 15 and 15 <= m <= 30: return '15:15 - 15:30'
    else: return 'Pre/Post Market'

print("Loading cached 8-year cash trades...")
chunk_files = sorted(glob.glob('backtest_chunks/trades_*.pkl'))
all_trades = []
for f in chunk_files:
    with open(f, 'rb') as file:
        all_trades.extend(pickle.load(file))

results = []
for t in all_trades:
    dt = pd.to_datetime(t.entry_time)
    t_bin = categorize_time_bin(dt)
    
    results.append({
        'bin': t_bin,
        'correct': getattr(t, 'underlying_direction_correct', False)
    })

df = pd.DataFrame(results)
total = len(df)

summary = df.groupby('bin').agg(
    Trades=('correct', 'count'),
    Wins=('correct', 'sum')
)

summary['Accuracy (%)'] = (summary['Wins'] / summary['Trades']) * 100
summary['% of Total'] = (summary['Trades'] / total) * 100

# Reorder index chronologically
bins_order = [
    '09:15 - 09:45', '09:45 - 10:15', '10:15 - 10:45', '10:45 - 11:15',
    '11:15 - 11:45', '11:45 - 12:15', '12:15 - 12:45', '12:45 - 13:15',
    '13:15 - 13:45', '13:45 - 14:15', '14:15 - 14:45', '14:45 - 15:15',
    '15:15 - 15:30', 'Pre/Post Market'
]
existing_bins = [b for b in bins_order if b in summary.index]
summary = summary.reindex(existing_bins)

print("\n=========================================================================")
print("          TIME-OF-DAY DIRECTIONAL ACCURACY DECOMPOSITION")
print("=========================================================================")
print(summary[['Trades', '% of Total', 'Accuracy (%)']].to_string(formatters={
    '% of Total': '{:.1f}'.format,
    'Accuracy (%)': '{:.1f}'.format
}))
print("=========================================================================")
