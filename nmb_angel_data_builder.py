import requests
import json
import pandas as pd
import datetime
import pyotp
import time
from SmartApi import SmartConnect

# --- CREDENTIALS ---
API_KEY = "pokXVmse"
CLIENT_CODE = "AAAD473947"
PASSWORD = "2121"
TOTP_SECRET = "AUGICFODVQYVEGWJYY5LBMSCWA"

def get_angel_session():
    print("Initiating Angel One SmartApi Login...")
    smartApi = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    response = smartApi.generateSession(CLIENT_CODE, PASSWORD, totp)
    if response['status']:
        print("Login Successful!")
        return smartApi
    else:
        print("Login Failed: ", response)
        return None

def fetch_token_map():
    print("Fetching Angel One Token Master List...")
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    res = requests.get(url)
    data = res.json()
    token_map = {}
    for item in data:
        if item['exch_seg'] in ['NSE', 'BSE']:
            token_map[item['symbol']] = item['token']
    return token_map

def get_historical_data(smartApi, exchange, symbol, token, interval, from_date, to_date):
    try:
        historicParam = {
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": from_date, 
            "todate": to_date
        }
        res = smartApi.getCandleData(historicParam)
        if res and res.get('status') and res.get('data'):
            df = pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def calculate_vwap(df):
    if df.empty: return df
    q = df['volume']
    p = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (p * q).cumsum() / q.cumsum()
    return df

def main():
    smartApi = get_angel_session()
    if not smartApi: return
    
    token_map = fetch_token_map()
    
    symbols_needed = {
        "NIFTY 50": ("NSE", "Nifty 50"),
        "NIFTY BANK": ("NSE", "Nifty Bank"),
        "FINNIFTY": ("NSE", "Nifty Fin Service"),
        "HDFCBANK": ("NSE", "HDFCBANK-EQ"),
        "ICICIBANK": ("NSE", "ICICIBANK-EQ"),
        "RELIANCE": ("NSE", "RELIANCE-EQ"),
        "INFY": ("NSE", "INFY-EQ"),
        "BAJFINANCE": ("NSE", "BAJFINANCE-EQ")
    }
    
    # We will fetch past 100 days in chunks of 30 days
    intervals = []
    end_dt = datetime.datetime.now()
    for i in range(4):
        start_dt = end_dt - datetime.timedelta(days=30)
        intervals.append((start_dt.strftime("%Y-%m-%d %H:%M"), end_dt.strftime("%Y-%m-%d %H:%M")))
        end_dt = start_dt
    intervals.reverse()
    
    dfs = {}
    for name, (exch, sym) in symbols_needed.items():
        token = token_map.get(sym)
        if not token:
            print(f"Token not found for {sym}")
            continue
        
        print(f"Fetching {name} ({sym}) [{token}]...")
        frames = []
        for st, en in intervals:
            df_chunk = get_historical_data(smartApi, exch, sym, token, "FIVE_MINUTE", st, en)
            if not df_chunk.empty:
                frames.append(df_chunk)
            time.sleep(0.5)
        
        if frames:
            full_df = pd.concat(frames).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
            full_df = calculate_vwap(full_df)
            dfs[name] = full_df
            print(f"Loaded {len(full_df)} rows for {name}")
        else:
            print(f"Failed to fetch data for {name}")
            
    # Combine Bank Nifty with HDFC and ICICI
    if "NIFTY BANK" in dfs and "HDFCBANK" in dfs and "ICICIBANK" in dfs:
        bn = dfs["NIFTY BANK"].copy()
        bn = bn.merge(dfs["HDFCBANK"][['timestamp', 'vwap']].rename(columns={'vwap':'hdfc_vwap'}), on='timestamp', how='left')
        bn = bn.merge(dfs["ICICIBANK"][['timestamp', 'vwap']].rename(columns={'vwap':'icici_vwap'}), on='timestamp', how='left')
        bn['hdfc_vwap_div'] = (dfs["HDFCBANK"]['close'] - bn['hdfc_vwap']) / bn['hdfc_vwap'] * 100
        bn['icici_vwap_div'] = (dfs["ICICIBANK"]['close'] - bn['icici_vwap']) / bn['icici_vwap'] * 100
        bn.to_csv("NIFTY_BANK_5minute_anchored.csv", index=False)
        print("Written NIFTY_BANK_5minute_anchored.csv")

    # Combine Nifty 50 with Reliance and Infy
    if "NIFTY 50" in dfs and "RELIANCE" in dfs and "INFY" in dfs:
        n50 = dfs["NIFTY 50"].copy()
        n50 = n50.merge(dfs["RELIANCE"][['timestamp', 'vwap']].rename(columns={'vwap':'reliance_vwap'}), on='timestamp', how='left')
        n50 = n50.merge(dfs["INFY"][['timestamp', 'vwap']].rename(columns={'vwap':'infy_vwap'}), on='timestamp', how='left')
        n50.to_csv("NIFTY_50_5minute_anchored.csv", index=False)
        print("Written NIFTY_50_5minute_anchored.csv")

    # Combine FinNifty with Bajaj Finance and HDFC
    if "FINNIFTY" in dfs and "BAJFINANCE" in dfs and "HDFCBANK" in dfs:
        fn = dfs["FINNIFTY"].copy()
        fn = fn.merge(dfs["BAJFINANCE"][['timestamp', 'vwap']].rename(columns={'vwap':'bajaj_vwap'}), on='timestamp', how='left')
        fn = fn.merge(dfs["HDFCBANK"][['timestamp', 'vwap']].rename(columns={'vwap':'hdfc_vwap'}), on='timestamp', how='left')
        fn.to_csv("FINNIFTY_5minute_anchored.csv", index=False)
        print("Written FINNIFTY_5minute_anchored.csv")

if __name__ == "__main__":
    main()
