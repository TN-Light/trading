import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)"""
    if len(df) <= period:
        return pd.Series(index=df.index, dtype=float)

    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    # Calculate +DM and -DM
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    
    # Resolving overlapping DMs
    mask = (plus_dm > minus_dm)
    df_plus_dm = plus_dm.copy()
    df_plus_dm[~mask] = 0
    df_minus_dm = minus_dm.copy()
    df_minus_dm[mask] = 0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def smooth(series, period):
        return series.ewm(alpha=1/period, min_periods=period).mean()
        
    atr = smooth(tr, period)
    plus_di = 100 * (smooth(df_plus_dm, period) / atr)
    minus_di = 100 * (smooth(df_minus_dm, period) / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = smooth(dx, period)
    
    return adx

print("Downloading NIFTY 50 and INDIA VIX data (2018-2026)...")
nifty = yf.download('^NSEI', start='2017-10-01', end='2026-03-31', progress=False)
vix = yf.download('^INDIAVIX', start='2017-10-01', end='2026-03-31', progress=False)

# Flatten columns if MultiIndex (from newer yf versions)
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# Calculate ADX on NIFTY
nifty['ADX'] = calculate_adx(nifty)

# Prepare empty list for our quarterly rows
results = []

# Map out chunks
chunk_files = sorted(glob.glob('backtest_chunks/trades_*.pkl'))

for chunk_path in chunk_files:
    filename = os.path.basename(chunk_path)
    year_str = filename.split('_')[1]
    quarter_str = filename.split('_')[2].split('.')[0]
    
    year = int(year_str)
    q_num = int(quarter_str[1])
    
    # Define start and end dates for the quarter
    q_start_month = (q_num - 1) * 3 + 1
    q_end_month = q_num * 3
    q_start = pd.Timestamp(f'{year}-{q_start_month:02d}-01')
    q_end = q_start + pd.offsets.QuarterEnd(0)
    
    # Get Market Data slices
    n_slice = nifty.loc[q_start:q_end]
    v_slice = vix.loc[q_start:q_end]
    
    if len(n_slice) < 5:
        continue
    
    q_return = ((n_slice['Close'].iloc[-1] / n_slice['Close'].iloc[0]) - 1) * 100
    q_avg_vix = v_slice['Close'].mean() if not v_slice.empty else np.nan
    q_avg_adx = n_slice['ADX'].mean() if not n_slice['ADX'].isna().all() else np.nan
    
    # Load accurate trades
    with open(chunk_path, 'rb') as f:
        trades = pickle.load(f)
        
    num_trades = len(trades)
    if num_trades == 0:
        continue
        
    correct_trades = sum(1 for t in trades if hasattr(t, 'underlying_direction_correct') and t.underlying_direction_correct)
    accuracy = (correct_trades / num_trades) * 100
    
    results.append({
        'Quarter': f"{year}_{quarter_str}",
        'Trades': num_trades,
        'Accuracy (%)': accuracy,
        'Q_Return (%)': q_return,
        'Avg VIX': q_avg_vix,
        'Avg ADX': q_avg_adx
    })

df_res = pd.DataFrame(results)

# Calculate correlations
cov_adx = df_res['Accuracy (%)'].corr(df_res['Avg ADX'])
cov_vix = df_res['Accuracy (%)'].corr(df_res['Avg VIX'])
cov_ret = df_res['Accuracy (%)'].corr(df_res['Q_Return (%)'])
cov_abs_ret = df_res['Accuracy (%)'].corr(df_res['Q_Return (%)'].abs())

print("\n=========================================================================")
print("          QUARTERLY DIRECTIONAL ACCURACY vs MARKET REGIMES")
print("=========================================================================")
print(df_res.to_string(index=False, float_format="%.2f"))

print("\n=========================================================================")
print("                           REGIME CORRELATIONS")
print("=========================================================================")
print(f"Correlation: Accuracy <-> ADX (Trend Strength) : {cov_adx:+.2f}")
print(f"Correlation: Accuracy <-> VIX (Volatility)     : {cov_vix:+.2f}")
print(f"Correlation: Accuracy <-> Abs Return (Move Size): {cov_abs_ret:+.2f}")
print(f"Correlation: Accuracy <-> Return (Direction)   : {cov_ret:+.2f}")

# Detect the best regime
best_qrts = df_res[df_res['Accuracy (%)'] > 50.0]
if not best_qrts.empty:
    print("\n[PROFITABLE REGIME PROFILING - Quarters > 50% Accuracy]")
    print(f"Average ADX during winning quarters: {best_qrts['Avg ADX'].mean():.2f}")
    print(f"Average VIX during winning quarters: {best_qrts['Avg VIX'].mean():.2f}")
    
worst_qrts = df_res[df_res['Accuracy (%)'] <= 50.0]
if not worst_qrts.empty:
    print("\n[UNPROFITABLE REGIME PROFILING - Quarters < 50% Accuracy]")
    print(f"Average ADX during losing quarters: {worst_qrts['Avg ADX'].mean():.2f}")
    print(f"Average VIX during losing quarters: {worst_qrts['Avg VIX'].mean():.2f}")
print("=========================================================================")
