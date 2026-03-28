import os
import glob
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("Starting Weekend Optimization...")
print("Loading data...")

# 1. Load price data
df_price = pd.read_csv('dataset/NIFTY 50_5minute.csv', parse_dates=['date'])
df_price = df_price.sort_values("date").reset_index(drop=True)

# 2. Extract features
df_price['ret'] = df_price['close'].pct_change()
df_price['ret_lag1'] = df_price['ret'].shift(1)
df_price['autocorr_20'] = df_price['ret'].rolling(window=20).corr(df_price['ret_lag1'])

df_price['high_low_range'] = (df_price['high'] - df_price['low']).replace(0, np.nan)
df_price['body'] = (df_price['close'] - df_price['open']).abs()
df_price['body_fraction'] = df_price['body'] / df_price['high_low_range']

df_price['ret_12'] = df_price['close'].pct_change(12)
df_price['ret_24'] = df_price['close'].pct_change(24)
df_price['ret_48'] = df_price['close'].pct_change(48)

df_price['bar_dir'] = np.sign(df_price['close'] - df_price['open'])

# Range expansion
df_price['avg_5_range'] = df_price['high_low_range'].shift(1).rolling(5).mean()
df_price['range_expansion'] = df_price['high_low_range'] / df_price['avg_5_range']

df_price.set_index('date', inplace=True)

# 3. Load trades
chunk_files = sorted(glob.glob('backtest_chunks/trades_*.pkl'))
all_trades = []
for f in chunk_files:
    with open(f, 'rb') as file:
        all_trades.extend(pickle.load(file))

# Need to map over trades and create a unified dataset
trades_data = []
for t in all_trades:
    dt = pd.to_datetime(t.entry_time)
    
    if dt in df_price.index:
        row = df_price.loc[dt]
    else:
        # Fallback to nearest before
        try:
            idx_loc = df_price.index.get_indexer([dt], method='pad')[0]
            row = df_price.iloc[idx_loc]
        except:
            continue
        
    t_dir = 1 if t.direction == 'bullish' else -1
    
    # Alignment score logic
    r12 = row['ret_12']
    r24 = row['ret_24']
    r48 = row['ret_48']
    
    score = np.nan
    if not (pd.isna(r12) or pd.isna(r24) or pd.isna(r48)):
        align_12 = 1 if (np.sign(r12) == t_dir) else 0
        align_24 = 1 if (np.sign(r24) == t_dir) else 0
        align_48 = 1 if (np.sign(r48) == t_dir) else 0
        num_aligned = align_12 + align_24 + align_48
        if num_aligned == 3: score = 2
        elif num_aligned == 2: score = 1
        elif num_aligned == 1: score = 0
        else: score = -1
        
    # Streak
    # get the last 5 bars ending at this dt
    idx_loc = df_price.index.get_indexer([dt], method='pad')[0]
    if idx_loc >= 4:
        last_5_dirs = df_price['bar_dir'].iloc[idx_loc-4:idx_loc+1].values
        streak = np.sum(last_5_dirs == t_dir)
    else:
        streak = np.nan
        
    trades_data.append({
        'time': dt,
        'year': dt.year,
        'correct': getattr(t, 'underlying_direction_correct', False),
        'autocorr_20': row['autocorr_20'],
        'body_fraction': row['body_fraction'],
        'align_score': score,
        'streak': streak,
        'range_expansion': row['range_expansion']
    })

df_trades = pd.DataFrame(trades_data)
df_trades['time'] = pd.to_datetime(df_trades['time'])
df_trades['hour'] = df_trades['time'].dt.hour
df_trades['minute'] = df_trades['time'].dt.minute

# Filtering helper
def in_window(df, h_start, m_start, h_end, m_end):
    time_val = df['hour'] * 60 + df['minute']
    start_val = h_start * 60 + m_start
    end_val = h_end * 60 + m_end
    # inclusive of start, exclusive of end
    return df[(time_val >= start_val) & (time_val < end_val)]

df_1045_1115 = in_window(df_trades, 10, 45, 11, 15)

print(f"Total trades: {len(df_trades)}")
print(f"Trades in 10:45-11:15: {len(df_1045_1115)}")
print("\n" + "="*70)

# EXPERIMENT 1
print("EXPERIMENT 1 - Threshold Sensitivity Heat Map")
ac_thresholds = [-0.05, -0.02, 0.00, 0.02, 0.05, 0.08]
bf_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Print header row
header = "BF\\AC\t" + "\t".join([f">{ac:5.2f}" for ac in ac_thresholds])
print(header)

for bf in bf_thresholds:
    row_str = f"<={bf:4.2f}\t"
    for ac in ac_thresholds:
        subset = df_1045_1115[(df_1045_1115['autocorr_20'] > ac) & (df_1045_1115['body_fraction'] <= bf)]
        if len(subset) == 0:
            row_str += "  0.0%\t"
        else:
            acc = subset['correct'].mean() * 100
            # format precisely to keep columns aligned
            row_str += f"{acc:5.1f}%\t"
    print(row_str)

print("\nOptional: N-count corresponding to the heatmap")
header_n = "BF\\AC\t" + "\t".join([f">{ac:5.2f}" for ac in ac_thresholds])
print(header_n)
for bf in bf_thresholds:
    row_str = f"<={bf:4.2f}\t"
    for ac in ac_thresholds:
        subset = df_1045_1115[(df_1045_1115['autocorr_20'] > ac) & (df_1045_1115['body_fraction'] <= bf)]
        row_str += f"{len(subset):5d}\t"
    print(row_str)

print("\n" + "="*70)
# EXPERIMENT 2
print("EXPERIMENT 2 - Window Boundary Sensitivity")
windows = [
    ("10:30-11:00", 10, 30, 11, 00),
    ("10:45-11:15", 10, 45, 11, 15),
    ("11:00-11:30", 11, 00, 11, 30),
    ("10:30-11:30", 10, 30, 11, 30),
    ("10:45-11:30", 10, 45, 11, 30)
]
for name, h1, m1, h2, m2 in windows:
    win_df = in_window(df_trades, h1, m1, h2, m2)
    n = len(win_df)
    if n > 0:
        acc = win_df['correct'].mean() * 100
        print(f"{name:12} : {n:3} trades | {acc:.1f}% accuracy")
    else:
        print(f"{name:12} :   0 trades")

print("\n" + "="*70)
# EXPERIMENT 3
print("EXPERIMENT 3 - Feature Expansion (within 10:45-11:15)")

# Cat 4
valid_align = df_1045_1115.dropna(subset=['align_score'])
for s in [2, 1, 0, -1]:
    sub = valid_align[valid_align['align_score'] == s]
    if len(sub) > 0:
        print(f"Alignment Score {s:2.0f} : {len(sub):3d} trades | Accuracy: {sub['correct'].mean()*100:.1f}%")

print("---")
# Streak
valid_streak = df_1045_1115.dropna(subset=['streak'])
sub_high = valid_streak[valid_streak['streak'] >= 3]
sub_low = valid_streak[valid_streak['streak'] < 3]
print(f"Streak >= 3 : {len(sub_high):3d} trades | Accuracy: {sub_high['correct'].mean()*100:.1f}%")
print(f"Streak <  3 : {len(sub_low):3d} trades | Accuracy: {sub_low['correct'].mean()*100:.1f}%")

print("---")
# Range Expansion
valid_re = df_1045_1115.dropna(subset=['range_expansion'])
sub_exp = valid_re[valid_re['range_expansion'] > 1.5]
sub_comp = valid_re[valid_re['range_expansion'] < 0.8]
print(f"Range Exp > 1.5 : {len(sub_exp):3d} trades | Accuracy: {sub_exp['correct'].mean()*100:.1f}%")
print(f"Range Exp < 0.8 : {len(sub_comp):3d} trades | Accuracy: {sub_comp['correct'].mean()*100:.1f}%")

print("\n" + "="*70)
# EXPERIMENT 4
print("EXPERIMENT 4 - Logistic Regression Cross-Validation")

df_lr = df_1045_1115.dropna(subset=['autocorr_20', 'body_fraction', 'correct'])
X_ac = df_lr['autocorr_20'].values
X_bf = df_lr['body_fraction'].values
X_int = X_ac * X_bf
X = np.column_stack([X_ac, X_bf, X_int])
y = df_lr['correct'].astype(int).values

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accs = []
coef_ints = []

# we use standard scaler to normalize to make coefficients more interpretable?
# actually we can just fit
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = LogisticRegression(class_weight=None, C=1.0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accs.append(np.mean(preds == y_test))
    coef_ints.append(clf.coef_[0][2])

print(f"Mean CV Accuracy: {np.mean(accs)*100:.1f}%")
print(f"Mean Interaction Coef (beta_3): {np.mean(coef_ints):.4f}")

try:
    import statsmodels.api as sm
    X_sm = sm.add_constant(X)
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=0)
    print(f"\nFull Model Interaction Coef (beta_3): {result.params[3]:.4f}")
    print(f"Interaction p-value: {result.pvalues[3]:.4f}")
except ImportError:
    clf_full = LogisticRegression()
    clf_full.fit(X, y)
    print(f"Full Model Interaction Coef (beta_3): {clf_full.coef_[0][2]:.4f}")

print("\n" + "="*70)
# EXPERIMENT 5
print("EXPERIMENT 5 - Quarterly/Yearly Accuracy of Joint Condition")
# joint condition: autocorr > 0 AND body_fraction <= 0.60
joint_df = df_1045_1115[(df_1045_1115['autocorr_20'] > 0) & (df_1045_1115['body_fraction'] <= 0.60)]

years = sorted(joint_df['year'].unique())
for y in years:
    sub = joint_df[joint_df['year'] == y]
    acc = sub['correct'].mean()*100 if len(sub) > 0 else 0
    print(f"{y}: {len(sub):3d} trades | {acc:.1f}%")

print("\nTotal joint trades:", len(joint_df))
print(f"Total joint accuracy: {joint_df['correct'].mean()*100:.1f}%")
