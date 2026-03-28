import os
import glob
import pickle
import pandas as pd
import numpy as np

print("Loading cached trades...")
chunk_files = sorted(glob.glob('backtest_chunks/trades_*.pkl'))
all_trades = []
for f in chunk_files:
    with open(f, 'rb') as file:
        trades = pickle.load(file)
        all_trades.extend(trades)

print(f"Total trades loaded: {len(all_trades)}")

print("Loading NIFTY 50 5-minute dataset...")
df = pd.read_csv('dataset/NIFTY 50_5minute.csv', parse_dates=['date'])
df = df.sort_values("date").reset_index(drop=True)

# Calculate returns over prior 12, 24, 48 bars
# We will shift by 1 later, or just look up directly safely via index.
# Ret = Close_t / Close_{t-n} - 1
df['ret_12'] = df['close'] / df['close'].shift(12) - 1
df['ret_24'] = df['close'] / df['close'].shift(24) - 1
df['ret_48'] = df['close'] / df['close'].shift(48) - 1

df.set_index('date', inplace=True)

results = []
missing = 0

for t in all_trades:
    entry_time = pd.to_datetime(t.entry_time)
    
    if entry_time in df.index:
        idx = df.index.get_loc(entry_time)
        # Evaluate on the signal bar (idx - 1)
        if idx > 0:
            signal_row = df.iloc[idx - 1]
            
            # Extract returns
            r12 = signal_row['ret_12']
            r24 = signal_row['ret_24']
            r48 = signal_row['ret_48']
            
            if pd.isna(r12) or pd.isna(r24) or pd.isna(r48):
                missing += 1
                continue
            
            trade_dir = 1 if t.direction == 'bullish' else -1
            
            # Alignment: True if sign of return matches trade direction
            align_12 = 1 if (np.sign(r12) == trade_dir) else 0
            align_24 = 1 if (np.sign(r24) == trade_dir) else 0
            align_48 = 1 if (np.sign(r48) == trade_dir) else 0
            
            num_aligned = align_12 + align_24 + align_48
            
            # Map to user's requested score
            if num_aligned == 3:
                score = 2    # All three agree
            elif num_aligned == 2:
                score = 1    # Two of three agree
            elif num_aligned == 1:
                score = 0    # Mixed / One agrees, two oppose
            else:
                score = -1   # All three opposed
            
            correct = getattr(t, 'underlying_direction_correct', False)
            
            results.append({
                'correct': correct,
                'score': score,
                'num_aligned': num_aligned
            })
    else:
        missing += 1

print(f"Successfully processed {len(results)} trades. (Missing/NaN: {missing})")

res_df = pd.DataFrame(results)

print("\n=========================================================================")
print("       CATEGORY 4: CROSS-TIMEFRAME MOMENTUM ALIGNMENT SPLIT")
print("=========================================================================")
for s in [2, 1, 0, -1]:
    subset = res_df[res_df['score'] == s]
    count = len(subset)
    if count > 0:
        acc = subset['correct'].mean() * 100
        print(f"Alignment Score {s:2d} | Trades: {count:4d} | Accuracy: {acc:.1f}%")
    else:
        print(f"Alignment Score {s:2d} | Trades:    0 | Accuracy: N/A")

print("=========================================================================")

# Breakdown by alignment count for clarity
print("\n[Detailed Alignment Match Count]")
for count in [3, 2, 1, 0]:
    subset = res_df[res_df['num_aligned'] == count]
    size = len(subset)
    if size > 0:
        acc = subset['correct'].mean() * 100
        print(f"Agreed Timeframes: {count} | Trades: {size:4d} | Accuracy: {acc:.1f}%")
