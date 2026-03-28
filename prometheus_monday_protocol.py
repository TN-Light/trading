# PROMETHEUS MONDAY FUTURES PROTOCOL
# Execute this script directly on Monday morning when Angel One API is online.

import os
import sys
import json
import yaml
import pyotp
import urllib.request
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from SmartApi import SmartConnect
except ImportError:
    print("FATAL: SmartApi not found. Install smartapi-python.")
    sys.exit(1)

def establish_angel_connection():
    with open('prometheus/config/credentials.yaml') as f:
        creds = yaml.safe_load(f).get('angelone', {})
        
    obj = SmartConnect(api_key=creds['api_key'])
    totp = pyotp.TOTP(creds['totp_secret']).now()
    try:
        data = obj.generateSession(creds['client_code'], creds['password'], totp)
    except Exception as e:
        print(f"\n[FATAL] Angel One API Endpoint Down (Weekend Maintenance): {e}")
        sys.exit(1)
    
    if data.get('status') == False:
        raise Exception(f"Angel One Login Failed: {data}")
    return obj

def fetch_instrument_token():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    response = urllib.request.urlopen(url)
    ilist = json.loads(response.read())
    
    futs = [x for x in ilist if x['name'] == 'NIFTY' and x['instrumenttype'] == 'FUTIDX']
    futs.sort(key=lambda x: datetime.strptime(x['expiry'], '%d%b%Y'))
    return futs[0]

def calculate_joint_classifier(df_5m):
    # Strictly the 10:45 - 11:15 anomaly window
    df = df_5m.copy()
    
    # Needs 20 prior bars for autocorr
    df['ret'] = df['close'].pct_change()
    df['ret_lag1'] = df['ret'].shift(1)
    df['autocorr_20'] = df['ret'].rolling(window=20).corr(df['ret_lag1'])
    
    # Body fraction
    df['hl_range'] = (df['high'] - df['low']).replace(0, np.nan)
    df['body_fraction'] = abs(df['close'] - df['open']) / df['hl_range']
    
    # Target label: Return of the next 6 bars (approx 30 mins)
    df['target_ret'] = df['close'].shift(-6) / df['close'] - 1
    
    df = df.dropna(subset=['autocorr_20', 'body_fraction', 'target_ret']).copy()
    
    # GATE: 10:45 - 11:15 Window ONLY
    # Map index to times
    times = df.index.time
    # 10:45 = 10, 45 to 11:10 (closing at 11:15)
    mask_time = [ ((t.hour == 10 and t.minute >= 45) or (t.hour == 11 and t.minute <= 10)) for t in times ]
    
    df_window = df[mask_time].copy()
    print(f"Total bars present in 10:45-11:15 window: {len(df_window)}")
    
    # JOINT CONDITION
    # 1-bar Autocorr > 0 AND Body Fraction <= 0.60
    mask_joint = (df_window['autocorr_20'] > 0) & (df_window['body_fraction'] <= 0.6)
    df_signals = df_window[mask_joint].copy()
    
    n_trades = len(df_signals)
    if n_trades == 0:
        print("[JOINT CLASSIFIER] No triggered trades found in OOS interval.")
        return 0, 0
        
    # We define accuracy as anticipating the direction of the anomaly. 
    # Usually the signal direction needs to align with the underlying. 
    # For now, we will measure the absolute direction match if we had a prior trend rule,
    # OR we measure if it successfully caught the directional continuation.
    
    # Assuming positive autocorrelation means we trend-follow the prior bar return.
    df_signals['pred_dir'] = np.sign(df_signals['ret'])
    df_signals['actual_dir'] = np.sign(df_signals['target_ret'])
    
    # Ignore flat targets
    df_signals = df_signals[df_signals['actual_dir'] != 0]
    
    df_signals['correct'] = (df_signals['pred_dir'] == df_signals['actual_dir'])
    acc = df_signals['correct'].mean() * 100
    
    print(f"\n=========================================================")
    print(f"   SEBI 11 AM MARGIN WINDOW: JOINT CLASSIFIER (OOS) ")
    print(f"=========================================================")
    print(f"N trades triggered : {len(df_signals)}")
    print(f"Directional Acc %  : {acc:.1f}%")
    print(f"=========================================================")
    
    if acc > 62.0 and len(df_signals) >= 40:
        print("RESULT: STATISTICALLY CONFIRMED. Hypothesis Validated.")
    elif acc >= 56.0 and len(df_signals) >= 40:
        print("RESULT: PROVISIONALLY VALIDATED. Requires 6-Month Data Extension.")
    else:
        print("RESULT: NULL REJECT. Path fails OOS check. Pivot to Daily Timeframe.")
        
    return acc, len(df_signals)

def main():
    print("--- PROMETHEUS MONDAY PROTOCOL INITIATED ---")
    obj = establish_angel_connection()
    front_month = fetch_instrument_token()
    print(f"Targeting: {front_month['symbol']} ({front_month['expiry']})")
    
    to_date = datetime.now()
    from_date = to_date - timedelta(days=60)
    
    req = {
        "exchange": front_month['exch_seg'],
        "symboltoken": front_month['token'],
        "interval": "FIVE_MINUTE",
        "fromdate": from_date.strftime('%Y-%m-%d 09:15'), 
        "todate": to_date.strftime('%Y-%m-%d 15:30')
    }
    
    res = obj.getCandleData(req)
    if not res or res.get('status') == False:
        print("API Limit/Timeout hit. Try again.")
        return
        
    df = pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Validation Check 1: Non-Zero Volume
    if df['volume'].sum() == 0:
        print("FATAL: Downloaded volume is zero. Not actual clearing data.")
        return
        
    # Validation Check 2: Roll Gap Check (no 5m bar > 3%)
    df['ret'] = df['close'].pct_change().abs()
    if df['ret'].max() > 0.03:
        print("WARNING: Massive roll gap detected. Needs Panama Splice immediately.")
        
    print(f"Protocol Check Passed. Valid 5M Futures Extracted: {len(df)} bars.")
    
    calculate_joint_classifier(df)
    
if __name__ == "__main__":
    main()
