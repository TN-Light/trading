import urllib.request
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import pyotp
try:
    from SmartApi import SmartConnect
except ImportError:
    print("SmartApi not found, skipping AngelOne.")

print("Loading Angel One credentials...")
try:
    with open('prometheus/config/credentials.yaml') as f:
        creds = yaml.safe_load(f).get('angelone', {})
        
    api_key = creds.get('api_key')
    client_code = creds.get('client_code')
    password = creds.get('password')
    totp_secret = creds.get('totp_secret')

    obj = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    data = obj.generateSession(client_code, password, totp)
    if data['status'] == False:
        print("Login Failed: ", data)
        exit()
    else:
        print("Angel One Login Successful.")
except Exception as e:
    print(f"Error during login: {e}")
    exit()

print("Fetching latest instrument master...")
url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
response = urllib.request.urlopen(url)
instrument_list = json.loads(response.read())

# Find NIFTY futures
nifty_futs = [x for x in instrument_list if x['name'] == 'NIFTY' and x['instrumenttype'] == 'FUTIDX']
nifty_futs.sort(key=lambda x: datetime.strptime(x['expiry'], '%d%b%Y'))

# Get the nearest expiry (front month)
front_month = nifty_futs[0]
print(f"Selected Front Month Future: {front_month['symbol']} (Token: {front_month['token']}, Expiry: {front_month['expiry']})")

# Fetch historical data for the last 60 days
to_date = datetime.now()
from_date = to_date - timedelta(days=60)

print(f"Fetching 5-minute data from {from_date.strftime('%Y-%m-%d %H:%M')} to {to_date.strftime('%Y-%m-%d %H:%M')}...")
historicParam = {
    "exchange": front_month['exch_seg'],
    "symboltoken": front_month['token'],
    "interval": "FIVE_MINUTE",
    "fromdate": from_date.strftime('%Y-%m-%d %H:%M'), 
    "todate": to_date.strftime('%Y-%m-%d %H:%M')
}

res = obj.getCandleData(historicParam)
if res['status'] == False:
    print("Failed to fetch historical data:", res)
    exit()

columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(res['data'], columns=columns)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Downloaded {len(df)} 5-minute bars.")

# Calculate Returns
df['ret'] = df['close'].pct_change()
df['abs_ret'] = df['ret'].abs()

# Compute Amihud Illiquidity = |return| / volume
df['volume_safe'] = df['volume'].replace(0, np.nan)
df['Amihud'] = df['abs_ret'] / df['volume_safe']

# Drop NaN and filter out zero-return bars
df = df.dropna(subset=['ret', 'Amihud'])
df = df[df['abs_ret'] > 0.00005] # at least 0.5 bps move

# Shift return for autocorrelation
df['ret_next'] = df['close'].pct_change().shift(-1)

# Quartiles
df['Amihud_Q'] = pd.qcut(df['Amihud'], 4, labels=['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest'])

print("\n=========================================================================")
print(f"       AMIHUD ILLIQUIDITY AUTOCORRELATION (NIFTY FUT: {front_month['symbol']})")
print("=========================================================================")

results = []
for q in ['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest']:
    subset = df[df['Amihud_Q'] == q].copy()
    if len(subset) > 0:
        autocorr = subset['ret'].corr(subset['ret_next'])
        results.append({
            'Quartile': q,
            'Bars': len(subset),
            'Median_Amihud': subset['Amihud'].median(),
            'Median_Vol': subset['volume'].median(),
            'Next_Bar_Autocorr': autocorr
        })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False, float_format="%.5f"))
print("=========================================================================")

if len(res_df) > 0:
    q1_auto = res_df.loc[res_df['Quartile'] == 'Q1_Lowest', 'Next_Bar_Autocorr'].values[0]
    if q1_auto and q1_auto > 0.05:
        print(f"\n[PASS] Q1 Autocorrelation is {q1_auto:.4f} (> 0.05). Indian NIFTY futures volume contains genuine signal!")
    else:
        print(f"\n[FAIL] Q1 Autocorrelation is {q1_auto:.4f} (< 0.05). Indian NIFTY 5-minute futures are efficiently priced.")
