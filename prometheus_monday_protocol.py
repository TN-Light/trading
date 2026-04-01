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
    with open("prometheus/config/credentials.yaml") as f:
        creds = yaml.safe_load(f).get("angelone", {})

    obj = SmartConnect(api_key=creds["api_key"])
    totp = pyotp.TOTP(creds["totp_secret"]).now()
    try:
        data = obj.generateSession(creds["client_code"], creds["password"], totp)
    except Exception as e:
        print(f"\n[FATAL] Angel One API Endpoint Down (Weekend Maintenance): {e}")
        sys.exit(1)

    if data.get("status") == False:
        raise Exception(f"Angel One Login Failed: {data}")
    return obj

def fetch_instrument_token():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    response = urllib.request.urlopen(url)
    ilist = json.loads(response.read())

    futs = [x for x in ilist if x["name"] == "NIFTY" and x["instrumenttype"] == "FUTIDX"]
    futs.sort(key=lambda x: datetime.strptime(x["expiry"], "%d%b%Y"))
    return futs[0]


def calculate_amihud_q1_autocorr(df_5m):
    """Compute lag-1 return autocorrelation within the lowest Amihud quartile on 30m bars."""
    df = df_5m.copy()
    if df.empty:
        return np.nan, 0

    bars_30m = df.resample("30min", origin="start_day", offset="15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    bars_30m = bars_30m.dropna(subset=["open", "high", "low", "close"])
    if bars_30m.empty:
        return np.nan, 0

    bars_30m["ret"] = bars_30m["close"].pct_change()
    bars_30m["amihud"] = bars_30m["ret"].abs() / bars_30m["volume"].replace(0, np.nan)

    valid = bars_30m.dropna(subset=["ret", "amihud"]).copy()
    if len(valid) < 3:
        return np.nan, len(valid)

    q1_cutoff = valid["amihud"].quantile(0.25)
    q1 = valid[valid["amihud"] <= q1_cutoff].copy()
    if len(q1) < 3:
        return np.nan, len(q1)

    autocorr = q1["ret"].corr(q1["ret"].shift(1))
    return (float(autocorr), len(q1)) if pd.notna(autocorr) else (np.nan, len(q1))

def calculate_joint_classifier(df_5m):
    # Strictly the 10:45 - 11:10 anomaly window
    df = df_5m.copy()

    # Needs 20 prior bars for autocorr
    df["ret"] = df["close"].pct_change()
    df["ret_lag1"] = df["ret"].shift(1)
    df["autocorr_20"] = df["ret"].rolling(window=20).corr(df["ret_lag1"])       

    # Body fraction
    df["hl_range"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_fraction"] = abs(df["close"] - df["open"]) / df["hl_range"]        

    # Category 4 Timeframes
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)
    df["ret_48"] = df["close"].pct_change(48)

    # Session VWAP Logic
    df["date"] = df.index.date
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vol_tp"] = df["volume"] * df["typical_price"]
    df["vwap"] = df.groupby("date")["vol_tp"].cumsum() / df.groupby("date")["volume"].cumsum()
    
    # Target label: Return of the next 6 bars (approx 30 mins)
    df["target_ret"] = df["close"].shift(-6) / df["close"] - 1

    df = df.dropna(subset=["autocorr_20", "body_fraction", "target_ret"]).copy()

    # GATE: Included bar timestamps are 10:45, 10:50, 10:55, 11:00, 11:05, 11:10
    # Map index to times
    times = df.index.time
    mask_time = [ ((t.hour == 10 and t.minute >= 45) or (t.hour == 11 and t.minute <= 10)) for t in times ]

    df_window = df[mask_time].copy()
    print(f"Total bars present in 10:45-11:10 window: {len(df_window)}")        

    # JOINT CONDITION (UPDATED FROM OPTIMIZATION)
    # 1-bar Autocorr >= 0.02 AND Body Fraction <= 0.60
    mask_joint = (df_window["autocorr_20"] >= 0.02) & (df_window["body_fraction"] <= 0.60)
    df_signals = df_window[mask_joint].copy()

    n_trades = len(df_signals)
    if n_trades == 0:
        print("[JOINT CLASSIFIER] No triggered trades found in OOS interval.")  
        return 0, 0

    # Direction logic: Track VWAP bias
    df_signals["vwap_bias"] = np.where(df_signals["close"] > df_signals["vwap"], 1, -1)
    df_signals["pred_dir"] = df_signals["vwap_bias"]
    df_signals["actual_dir"] = np.sign(df_signals["target_ret"])

    # Category 4 Alignment inversion logic
    def get_cat4_score(row):
        score = 0
        if pd.isna(row["ret_12"]) or pd.isna(row["ret_24"]) or pd.isna(row["ret_48"]): return np.nan
        if np.sign(row["ret_12"]) == row["pred_dir"]: score += 1
        if np.sign(row["ret_24"]) == row["pred_dir"]: score += 1
        if np.sign(row["ret_48"]) == row["pred_dir"]: score += 1
        return score

    df_signals["cat4_score"] = df_signals.apply(get_cat4_score, axis=1)

    # Ignore flat targets
    df_signals = df_signals[df_signals["actual_dir"] != 0].copy()

    df_signals["correct"] = (df_signals["pred_dir"] == df_signals["actual_dir"])
    acc = df_signals["correct"].mean() * 100

    cat4_low = df_signals[df_signals["cat4_score"] <= 1]
    cat4_mid = df_signals[df_signals["cat4_score"] == 2]
    
    acc_low = cat4_low["correct"].mean() * 100 if len(cat4_low) > 0 else 0
    acc_mid = cat4_mid["correct"].mean() * 100 if len(cat4_mid) > 0 else 0

    print(f"\n=========================================================")       
    print(f"   SEBI 11 AM MARGIN WINDOW: JOINT CLASSIFIER (OOS) ")
    print(f"=========================================================")
    print(f"Condition: Autocorr >= 0.02 AND Body Frac <= 0.60")
    print(f"N trades triggered : {len(df_signals)}")
    print(f"Directional Acc %  : {acc:.1f}%")
    print(f"---------------------------------------------------------")
    print(f"   SECONDARY HYPOTHESIS: CAT 4 ALIGNMENT INVERSION")
    print(f"Score 0-1 (Conflict)  : {len(cat4_low)} trades | Acc: {acc_low:.1f}%")
    print(f"Score 2 (Mid Align)   : {len(cat4_mid)} trades | Acc: {acc_mid:.1f}%")
    if len(cat4_low) > 0 and len(cat4_mid) > 0:
        print(f"Difference            : {(acc_low - acc_mid):.1f}%")
    print(f"=========================================================")

    n_signals = len(df_signals)
    if n_signals < 40:
        print("RESULT: EXTEND. Sample size below n=40; continue accumulation.")
    elif acc >= 58.0:
        print("RESULT: STATISTICALLY CONFIRMED. Hypothesis Validated.")
    elif acc >= 53.0:
        print("RESULT: PROVISIONALLY VALIDATED. Requires 6-Month Data Extension.")
    else:
        print("RESULT: REJECT. Below 53% with n>=40. Pivot to Daily Timeframe.")

    return acc, n_signals

def main():
    print("--- PROMETHEUS MONDAY PROTOCOL INITIATED ---")
    obj = establish_angel_connection()
    front_month = fetch_instrument_token()
    print(f"Targeting: {front_month['symbol']} ({front_month['expiry']})")       

    to_date = datetime.now()
    from_date = to_date - timedelta(days=60)

    req = {
        "exchange": front_month["exch_seg"],
        "symboltoken": front_month["token"],
        "interval": "FIVE_MINUTE",
        "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
        "todate": to_date.strftime("%Y-%m-%d 15:30")
    }

    res = obj.getCandleData(req)
    if not res or res.get("status") == False:
        print("API Limit/Timeout hit. Try again.")
        return

    df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Validation Check 1: Non-Zero Volume
    if df["volume"].sum() == 0:
        print("FATAL: Downloaded volume is zero. Not actual clearing data.")    
        return

    # Validation Check 2: Roll Gap Check (no 5m bar > 3%)
    df["ret"] = df["close"].pct_change().abs()
    if df["ret"].max() > 0.03:
        print("WARNING: Massive roll gap detected. Needs Panama Splice immediately.")

    print(f"Protocol Check Passed. Valid 5M Futures Extracted: {len(df)} bars.")

    amihud_q1_ac, amihud_q1_n = calculate_amihud_q1_autocorr(df)
    if pd.isna(amihud_q1_ac):
        print(f"Amihud Q1 Autocorr (30m): N/A (insufficient valid bars, n={amihud_q1_n})")
    else:
        print(f"Amihud Q1 Autocorr (30m): {amihud_q1_ac:.4f} (n={amihud_q1_n})")

    acc, n_trades = calculate_joint_classifier(df)
    print(f"Summary: Joint classifier accuracy={acc:.1f}% | n={n_trades}")

if __name__ == "__main__":
    main()

