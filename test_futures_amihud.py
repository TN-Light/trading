import yfinance as yf
import pandas as pd
import numpy as np

print("Downloading S&P 500 E-mini Futures (ES=F) 5-minute data (last 60 days)...")
# We use ES=F as a proxy for highly liquid index futures since yf doesn't carry NIFTY futures.
df = yf.download("ES=F", interval="5m", period="60d", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"Downloaded {len(df)} bars of futures data with real volume.")

# Compute Returns
df['ret'] = df['Close'].pct_change()
df['abs_ret'] = df['ret'].abs()

# Compute Amihud Illiquidity = |return| / volume
# Volume can sometimes be 0 or small, replace 0 with np.nan
df['Volume_safe'] = df['Volume'].replace(0, np.nan)
df['Amihud'] = df['abs_ret'] / df['Volume_safe']

# Drop NaN
df = df.dropna(subset=['ret', 'Amihud'])

# Filter out zero-return bars to get clean Amihud readings
df = df[df['abs_ret'] > 0.00005] # at least 0.5 bps move

# Compute 1-bar future return to check predicting power/autocorrelation
df['ret_next'] = df['Close'].pct_change().shift(-1)
# 1-bar autocorrelation feature
df['ret_lag1'] = df['ret'].shift(1)

# Shift Amihud to the bar prior to target, or we are just checking autocorrelation conditioned on Amihud.
# The user asked: "Split the bars by Amihud quartile and check whether the lowest illiquidity quartile (largest volume relative to move) shows any autocorrelation structure."
# We want to check Corr(ret_t, ret_{t+1}) grouped by Amihud_t quartile.

# Calculate Amihud Quartiles
df['Amihud_Q'] = pd.qcut(df['Amihud'], 4, labels=['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest'])

print("\n=========================================================================")
print("       AMIHUD ILLIQUIDITY AUTOCORRELATION (S&P 500 FUTURES proxy)")
print("=========================================================================")

results = []
for q in ['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest']:
    subset = df[df['Amihud_Q'] == q].copy()
    
    # Calculate autocorrelation between this bar's return and NEXT bar's return
    # E.g., if a bar has low illiquidity (high volume), does the return continue?
    # This is Corr(ret_t, ret_{t+1})
    autocorr = subset['ret'].corr(subset['ret_next'])
    
    # Also check traditional 1-bar autocorrelation measured on the subset
    results.append({
        'Quartile': q,
        'Bars': len(subset),
        'Median_Amihud': subset['Amihud'].median(),
        'Median_Vol': subset['Volume'].median(),
        'Next_Bar_Autocorr': autocorr
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False, float_format="%.5f"))
print("=========================================================================")

# Check if Q1 > 0.05
q1_auto = res_df.loc[res_df['Quartile'] == 'Q1_Lowest', 'Next_Bar_Autocorr'].values[0]
if q1_auto > 0.05:
    print(f"\n[PASS] Q1 Autocorrelation is {q1_auto:.4f} (> 0.05). Volume-based features contain genuine signal.")
else:
    print(f"\n[FAIL] Q1 Autocorrelation is {q1_auto:.4f} (< 0.05). Even in futures, 5m autocorrelation is too weak.")
