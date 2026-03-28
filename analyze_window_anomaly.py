import os
import glob
import pickle
import pandas as pd
import numpy as np
import yfinance as yf

# Function to isolate just our anomaly timeframe
def is_target_window(dt):
    h = dt.hour
    m = dt.minute
    # targeting 10:45 to 11:15
    if (h == 10 and m >= 45) or (h == 11 and m < 15):
        return True
    return False

# 1. Fetch market regimes (Yearly VIX & NIFTY Return)
print("Fetching macro data to calculate yearly context...")
nifty = yf.download('^NSEI', start='2018-01-01', end='2026-03-31', progress=False)
vix = yf.download('^INDIAVIX', start='2018-01-01', end='2026-03-31', progress=False)

if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

yearly_regime = {}
for year in range(2018, 2027):
    n_slice = nifty[nifty.index.year == year]
    v_slice = vix[vix.index.year == year]
    if len(n_slice) > 0:
        y_ret = ((n_slice['Close'].iloc[-1] / n_slice['Close'].iloc[0]) - 1) * 100
        avg_vix = v_slice['Close'].mean()
        yearly_regime[str(year)] = {'NIFTY_Ret': y_ret, 'Avg_VIX': avg_vix}

# 2. Load trades and filter to the 10:45--11:15 anomaly window
print("Loading cached 8-year cash trades...")
chunk_files = sorted(glob.glob('backtest_chunks/trades_*.pkl'))
all_trades = []
for f in chunk_files:
    with open(f, 'rb') as file:
        all_trades.extend(pickle.load(file))

target_trades = []
for t in all_trades:
    dt = pd.to_datetime(t.entry_time)
    if is_target_window(dt):
        target_trades.append(t)

print(f"\nLocked onto target window (10:45 - 11:15). Trades found: {len(target_trades)}")

# ----------------------------------------------------------------------------------
# TASK 1: TEMPORAL STABILITY DECOMPOSITION
# ----------------------------------------------------------------------------------
stability_res = []
for t in target_trades:
    dt = pd.to_datetime(t.entry_time)
    stability_res.append({
        'year': str(dt.year),
        'correct': getattr(t, 'underlying_direction_correct', False)
    })

df_stab = pd.DataFrame(stability_res)
yearly_counts = df_stab.groupby('year').agg(
    n_trades=('correct', 'count'),
    wins=('correct', 'sum')
)
yearly_counts['accuracy%'] = (yearly_counts['wins'] / yearly_counts['n_trades']) * 100

# Merge in macro regimes
final_stab = []
for yr, row in yearly_counts.iterrows():
    r = yearly_regime.get(yr, {'NIFTY_Ret': np.nan, 'Avg_VIX': np.nan})
    final_stab.append({
        'Year': yr,
        'n_trades': row['n_trades'],
        'accuracy%': row['accuracy%'],
        'NIFTY_Return%': r['NIFTY_Ret'],
        'Avg_VIX': r['Avg_VIX']
    })

df_temporal = pd.DataFrame(final_stab)

print("\n=========================================================================")
print("     TASK 1: TEMPORAL STABILITY (10:45 - 11:15 WINDOW EXCLUSIVE)")
print("=========================================================================")
print(df_temporal.to_string(index=False, float_format="%.1f"))


# ----------------------------------------------------------------------------------
# TASK 2: FEATURE SPLIT RE-RUN STRICTLY ON THE TARGET WINDOW
# ----------------------------------------------------------------------------------
print("\nLoading NIFTY 50 5-minute dataset to re-run features...")
df_price = pd.read_csv('dataset/NIFTY 50_5minute.csv', parse_dates=['date'])
df_price = df_price.sort_values("date").reset_index(drop=True)

# 1-bar Autocorr
df_price['ret'] = df_price['close'].pct_change()
df_price['ret_lag1'] = df_price['ret'].shift(1)
df_price['autocorr_20'] = df_price['ret'].rolling(window=20).corr(df_price['ret_lag1'])

# Body Fraction
df_price['high_low_range'] = (df_price['high'] - df_price['low']).replace(0, np.nan)
df_price['body_fraction'] = abs(df_price['close'] - df_price['open']) / df_price['high_low_range']

# Momentum Alignment (12, 24, 48 bars)
df_price['ret_12'] = df_price['close'] / df_price['close'].shift(12) - 1
df_price['ret_24'] = df_price['close'] / df_price['close'].shift(24) - 1
df_price['ret_48'] = df_price['close'] / df_price['close'].shift(48) - 1

df_price.set_index('date', inplace=True)

feat_results = []
for t in target_trades:
    entry_time = pd.to_datetime(t.entry_time)
    if entry_time in df_price.index:
        idx = df_price.index.get_loc(entry_time)
        if idx > 0:
            signal_row = df_price.iloc[idx - 1]
            
            trade_dir = 1 if t.direction == 'bullish' else -1
            r12 = signal_row['ret_12']
            r24 = signal_row['ret_24']
            r48 = signal_row['ret_48']
            
            score = np.nan
            if not (pd.isna(r12) or pd.isna(r24) or pd.isna(r48)):
                align_12 = 1 if (np.sign(r12) == trade_dir) else 0
                align_24 = 1 if (np.sign(r24) == trade_dir) else 0
                align_48 = 1 if (np.sign(r48) == trade_dir) else 0
                num_aligned = align_12 + align_24 + align_48
                if num_aligned == 3: score = 2
                elif num_aligned == 2: score = 1
                elif num_aligned == 1: score = 0
                else: score = -1
                
            feat_results.append({
                'correct': getattr(t, 'underlying_direction_correct', False),
                'autocorr_20': signal_row['autocorr_20'],
                'body_fraction': signal_row['body_fraction'],
                'align_score': score
            })

res_feat = pd.DataFrame(feat_results)

print("\n=========================================================================")
print("      TASK 2: SENSORY MODALITY FEATURE SPLITS (TARGET WINDOW ONLY)")
print("=========================================================================")

print("--- Modality A: 1-bar Return Autocorrelation (Prior 20 bars) ---")
valid_auto = res_feat.dropna(subset=['autocorr_20'])
pos_auto = valid_auto[valid_auto['autocorr_20'] > 0]
neg_auto = valid_auto[valid_auto['autocorr_20'] < 0]

print(f"Positive Autocorr (> 0): {len(pos_auto)} trades, Accuracy: {pos_auto['correct'].mean()*100:.1f}%")
print(f"Negative Autocorr (< 0): {len(neg_auto)} trades, Accuracy: {neg_auto['correct'].mean()*100:.1f}%")

print("\n--- Modality B: Candle Body Fraction ---")
valid_body = res_feat.dropna(subset=['body_fraction'])
high_body = valid_body[valid_body['body_fraction'] > 0.6]
low_body = valid_body[valid_body['body_fraction'] <= 0.6]

print(f"Body Fraction > 0.6: {len(high_body)} trades, Accuracy: {high_body['correct'].mean()*100:.1f}%")
print(f"Body Fraction <= 0.6: {len(low_body)} trades, Accuracy: {low_body['correct'].mean()*100:.1f}%")

print("\n--- Modality C: Cross-Timeframe Momentum Alignment (Macro) ---")
valid_align = res_feat.dropna(subset=['align_score'])
for s in [2, 1, 0, -1]:
    subset = valid_align[valid_align['align_score'] == s]
    if len(subset) > 0:
        print(f"Alignment Score {s:2.0f} : {len(subset):3d} trades, Accuracy: {subset['correct'].mean()*100:.1f}%")

print("=========================================================================")
