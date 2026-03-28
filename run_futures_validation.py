import urllib.request
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import pyotp
import sys
try:
    from SmartApi import SmartConnect
except ImportError:
    print("SmartApi not found. Install smartapi-python.")
    sys.exit()

def get_angel_one_client():
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
        if data.get('status') == False:
            print("Login Failed: ", data)
            return None
        return obj
    except Exception as e:
        print(f"Error connecting to Angel One: {e}")
        return None

def calculate_amihud_autocorr(df, label=""):
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['abs_ret'] = df['ret'].abs()
    
    # Amihud Illiquidity = |return| / volume
    df['volume_safe'] = df['volume'].replace(0, np.nan)
    df['Amihud'] = df['abs_ret'] / df['volume_safe']
    
    # Drop NaN and filter zero-return bars
    df = df.dropna(subset=['ret', 'Amihud'])
    df = df[df['abs_ret'] > 0.00005]
    
    # Next bar return
    df['ret_next'] = df['close'].pct_change().shift(-1)
    
    # Quartiles
    try:
        df['Amihud_Q'] = pd.qcut(df['Amihud'], 4, labels=['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest'])
    except ValueError:
        print(f"[{label}] Not enough variance to compute Amihud quartiles.")
        return None
        
    print(f"\n=========================================================================")
    print(f"       AMIHUD ILLIQUIDITY AUTOCORRELATION ({label})")
    print(f"=========================================================================")
    
    results = []
    for q in ['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest']:
        subset = df[df['Amihud_Q'] == q].copy()
        if len(subset) > 5:
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
    
    return res_df

def calculate_category4_30m(df):
    df = df.copy()
    
    # Calculate returns over 4, 8, and 16 bars (2h, 4h, 8h)
    df['ret_4'] = df['close'] / df['close'].shift(4) - 1
    df['ret_8'] = df['close'] / df['close'].shift(8) - 1
    df['ret_16'] = df['close'] / df['close'].shift(16) - 1
    
    # Target move: the return of the NEXT 30m bar
    df['ret_next'] = df['close'].pct_change().shift(-1)
    
    df = df.dropna(subset=['ret_4', 'ret_8', 'ret_16', 'ret_next']).copy()
    
    # Continuous alignment scoring
    # +2 if all agree, +1 if 2 agree, 0 if mixed, -1 if opposed.
    # To map this strictly: "signal" is the majority vote.
    
    results = []
    for idx, row in df.iterrows():
        r4 = np.sign(row['ret_4'])
        r8 = np.sign(row['ret_8'])
        r16 = np.sign(row['ret_16'])
        next_ret = row['ret_next']
        
        sum_signs = r4 + r8 + r16
        
        if sum_signs == 3:
            score = 2
            dir_pred = 1
        elif sum_signs == -3:
            score = 2
            dir_pred = -1
        elif sum_signs == 1: # 2 pos, 1 neg
            score = 1
            dir_pred = 1
        elif sum_signs == -1: # 2 neg, 1 pos
            score = 1
            dir_pred = -1
        else: # e.g. 0s involved
            score = 0
            dir_pred = np.sign(sum_signs) if sum_signs != 0 else 1
            
        correct = (np.sign(next_ret) == dir_pred)
        
        results.append({
            'score': score,
            'correct': correct
        })
        
    res_df = pd.DataFrame(results)
    
    print(f"\n=========================================================================")
    print(f"       CATEGORY 4: CROSS-TIMEFRAME ALIGNMENT (30-MINUTE CONTINUOUS)")
    print(f"=========================================================================")
    for score in [2, 1, 0, -1]:
        subset = res_df[res_df['score'] == score]
        count = len(subset)
        if count > 0:
            acc = subset['correct'].mean() * 100
            print(f"Alignment Score {score:2d} | Observations: {count:4d} | Accuracy: {acc:.1f}%")
        else:
            print(f"Alignment Score {score:2d} | Observations:    0 | Accuracy: N/A")
    print(f"=========================================================================")


def main():
    print("Checking Angel One API Status...")
    obj = get_angel_one_client()
    if not obj:
        print("\n[!] FATAL: Cannot connect to Angel One API.")
        print("This is likely due to the weekend maintenance window where Angel One's authentication servers block connections (ConnectionTimeout).")
        print("Run this script again during market hours computationally.")
        return

    print("Fetching latest instrument master...")
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    response = urllib.request.urlopen(url)
    instrument_list = json.loads(response.read())

    nifty_futs = [x for x in instrument_list if x['name'] == 'NIFTY' and x['instrumenttype'] == 'FUTIDX']
    nifty_futs.sort(key=lambda x: datetime.strptime(x['expiry'], '%d%b%Y'))

    front_month = nifty_futs[0]
    print(f"\nSelected Front Month Future: {front_month['symbol']} (Token: {front_month['token']}, Expiry: {front_month['expiry']})")

    # Fetch historical data (60 days max for five minute in single shot or chunked)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=60)
    
    print(f"Fetching 5-minute data from {from_date.strftime('%Y-%m-%d %H:%M')}...")
    historicParam = {
        "exchange": front_month['exch_seg'],
        "symboltoken": front_month['token'],
        "interval": "FIVE_MINUTE",
        "fromdate": from_date.strftime('%Y-%m-%d 09:15'), 
        "todate": to_date.strftime('%Y-%m-%d 15:30')
    }

    res = obj.getCandleData(historicParam)
    if not res or res.get('status') == False:
        print("Failed to fetch historical data:", res)
        # If API limit triggers, try 30 days
        print("Retrying with 30 days...")
        historicParam['fromdate'] = (to_date - timedelta(days=30)).strftime('%Y-%m-%d 09:15')
        res = obj.getCandleData(historicParam)
        if not res or res.get('status') == False:
            print("Failed 30 day fetch.")
            return

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_5m = pd.DataFrame(res['data'], columns=columns)
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
    df_5m.set_index('timestamp', inplace=True)
    print(f"Downloaded {len(df_5m)} 5-minute bars.")
    
    # -----------------------
    # STEP 1: 5-Minute Validation
    # -----------------------
    res_5m = calculate_amihud_autocorr(df_5m, label="5-MINUTE NIFTY FUTURES")
    if res_5m is not None and not res_5m.empty:
        q1_auto_5m = res_5m.loc[res_5m['Quartile'] == 'Q1_Lowest', 'Next_Bar_Autocorr'].values[0]
        if q1_auto_5m > 0.05:
            print(f"\n[PASS] 5m Q1 Autocorrelation is {q1_auto_5m:.4f} (> 0.05).")
            print("-> Proceed with 5-minute futures!")
        else:
            print(f"\n[FAIL] 5m Q1 Autocorrelation is {q1_auto_5m:.4f} (< 0.05).")
            print("-> 5M EMH holds on NIFTY too. Moving to 30-minute validation.")
            
    # -----------------------
    # STEP 2: 30-Minute Generation & Validation
    # -----------------------
    print("\nResampling 5-minute order flow to 30-minute gestation blocks...")
    df_30m = df_5m.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"Generated {len(df_30m)} 30-minute bars.")
    res_30m = calculate_amihud_autocorr(df_30m, label="30-MINUTE NIFTY FUTURES")
    
    if res_30m is not None and not res_30m.empty:
        q1_auto_30m = res_30m.loc[res_30m['Quartile'] == 'Q1_Lowest', 'Next_Bar_Autocorr'].values[0]
        if q1_auto_30m > 0.05:
            print(f"\n[PASS] 30m Q1 Autocorrelation is {q1_auto_30m:.4f} (> 0.05).")
            print("-> The 30m gestation horizon is confirmed to hold order flow footprint!")
            
    calculate_category4_30m(df_30m)

if __name__ == "__main__":
    main()
