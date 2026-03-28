import os
import glob
import pickle
import pandas as pd
import numpy as np

# Load the trades
chunk_files = sorted(glob.glob('backtest_chunks/trades_*.pkl'))
all_trades = []
for f in chunk_files:
    with open(f, 'rb') as file:
        trades = pickle.load(file)
        all_trades.extend(trades)

print(f"Total trades loaded: {len(all_trades)}")

print("Loading dataset/NIFTY 50_5minute.csv...")
df = pd.read_csv('dataset/NIFTY 50_5minute.csv', parse_dates=['date'])
df = df.sort_values("date").reset_index(drop=True)

# Feature 1: 1-bar return autocorrelation over the prior 20 bars
print("Computing 1-bar return autocorrelation...")
df['ret'] = df['close'].pct_change()
df['ret_lag1'] = df['ret'].shift(1)
df['autocorr_20'] = df['ret'].rolling(window=20).corr(df['ret_lag1'])

# Feature 2: Volume acceleration (d(volume_ratio)/dt) on the signal bar over 3 bars
print("Computing volume acceleration...")
df['vol_mean_20'] = df['volume'].replace(0, np.nan).rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['vol_mean_20']
df['vol_accel_3'] = df['volume_ratio'] - df['volume_ratio'].shift(3)

# Feature 3: Candle body fraction
print("Computing candle body fraction...")
df['high_low_range'] = df['high'] - df['low']
# Avoid division by zero
df['high_low_range'] = df['high_low_range'].replace(0, np.nan)
df['body_fraction'] = abs(df['close'] - df['open']) / df['high_low_range']

# Shift features by 1 so we get the value right *before* the trade if the trade entered at open, 
# or ensure we get the exact signal bar feature. The user says:
# "compute 40-60 features on every bar using only information available at bar close"
# Since entry is likely at the next bar, let's map trade entry time to the dataset.
# The trade entry time is usually the bar of the trade. If it's the signal bar, we might need to adjust.
# Let's map features based on the exact time.

# Index by date for fast lookup
df.set_index('date', inplace=True)

results = []
missing = 0
for t in all_trades:
    # Convert entry time string to datetime
    entry_time = pd.to_datetime(t.entry_time)
    
    # In Prometheus, signal bar is typically entry_time - 1 bar. Let's get the feature on the signal bar.
    # To be safe, let's look at the bar exactly AT entry_time, or the one right before it. 
    # Let's just use the index lookup using bisect or get the latest bar before or at entry time.
    
    # We will get the bar that closed right AT the entry time if it's intraday, or the prior bar if it's next bar entry.
    # We will compute the features AT the entry time bar (assuming that's the signal bar, as entry_time in the trade object might be the signal bar).
    
    # Let's try exact match first
    if entry_time in df.index:
        idx = df.index.get_loc(entry_time)
        # Features must be evaluated on the signal bar (idx - 1) available *exactly at bar close* prior to entry
        if idx > 0:
            row_signal = df.iloc[idx - 1]
            
            autocorr = row_signal['autocorr_20']
            vol_accel = row_signal['vol_accel_3']
            body_frac = row_signal['body_fraction']
        
        correct = getattr(t, 'underlying_direction_correct', False)
        
        results.append({
            'correct': correct,
            'autocorr_20': autocorr,
            'vol_accel_3': vol_accel,
            'body_fraction': body_frac
        })
    else:
        missing += 1

print(f"Matched {len(results)} trades to market data features. (Missing: {missing})")

res_df = pd.DataFrame(results)

print("\n--- Feature 1: 1-bar Return Autocorrelation (Prior 20 bars) ---")
valid_auto = res_df.dropna(subset=['autocorr_20'])
pos_auto = valid_auto[valid_auto['autocorr_20'] > 0]
neg_auto = valid_auto[valid_auto['autocorr_20'] < 0]

print(f"Positive Autocorrelation (> 0): {len(pos_auto)} trades, Accuracy: {pos_auto['correct'].mean()*100:.1f}%")
print(f"Negative Autocorrelation (< 0): {len(neg_auto)} trades, Accuracy: {neg_auto['correct'].mean()*100:.1f}%")

print("\n--- Feature 2: Volume Acceleration (d(vol_ratio)/dt over 3 bars) ---")
valid_vol = res_df.dropna(subset=['vol_accel_3'])
accel_vol = valid_vol[valid_vol['vol_accel_3'] > 0]
decel_vol = valid_vol[valid_vol['vol_accel_3'] < 0]

if len(valid_vol) == 0:
    print("NO VOLUME DATA AVAILABLE IN UNDERLYING INDEX DATASET (All NaN).")
else:
    print(f"Accelerating Volume (> 0): {len(accel_vol)} trades, Accuracy: {accel_vol['correct'].mean()*100:.1f}%")
    print(f"Decelerating Volume (< 0): {len(decel_vol)} trades, Accuracy: {decel_vol['correct'].mean()*100:.1f}%")

print("\n--- Feature 3: Candle Body Fraction ---")
valid_body = res_df.dropna(subset=['body_fraction'])
high_body = valid_body[valid_body['body_fraction'] > 0.6]
low_body = valid_body[valid_body['body_fraction'] <= 0.6]

print(f"Body Fraction > 0.6: {len(high_body)} trades, Accuracy: {high_body['correct'].mean()*100:.1f}%")
print(f"Body Fraction <= 0.6: {len(low_body)} trades, Accuracy: {low_body['correct'].mean()*100:.1f}%")

