import time
import datetime
import threading
import json
import pandas as pd
import requests
import pyotp
import os
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# --- CREDENTIALS ---
API_KEY = "pokXVmse"
CLIENT_CODE = "AAAD473947"
PASSWORD = "2121"
TOTP_SECRET = "AUGICFODVQYVEGWJYY5LBMSCWA"

class OptionsIngester:
    def __init__(self):
        self.api = SmartConnect(api_key=API_KEY)
        self.session = None
        self.feed_token = None
        self.token_master = []
        self.ws = None
        
        # Real-time Storage
        self.live_data = {}
        self.csv_path = "dataset/live_options_context.csv"

        # Symbol configs: (Exchange, Symbol_Name, Strike_Step)
        self.index_config = {
            "NIFTY_50": {"exch": "NSE", "sym": "Nifty 50", "step": 50},
            "NIFTY_BANK": {"exch": "NSE", "sym": "Nifty Bank", "step": 100},
            "FINNIFTY": {"exch": "NSE", "sym": "Nifty Fin Service", "step": 50}
        }
        self.tracked_tokens = {} # token: {"index": "NIFTY_50", "type": "CE/PE/SPOT"}
        
    def login(self):
        print("[System] Authenticating with Angel One...")
        totp = pyotp.TOTP(TOTP_SECRET).now()
        data = self.api.generateSession(CLIENT_CODE, PASSWORD, totp)
        if data['status']:
            self.session = data['data']
            self.feed_token = self.api.getfeedToken()
            print("[System] Login Successful!")
        else:
            print("[System] Login Failed:", data)
            exit(1)

    def load_token_master(self):
        print("[System] Downloading Token Master (this takes a moment)...")
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        self.token_master = requests.get(url).json()

    def get_spot_ltp(self, exchange, symbol):
        # We need the token for the spot to fetch LTP
        token = next((i['token'] for i in self.token_master if i['symbol'] == symbol and i['exch_seg'] == exchange), None)
        if not token: return None
        
        try:
            res = self.api.ltpData(exchange, symbol, token)
            return res['data']['ltp']
        except:
            return None

    def setup_tracked_tokens(self):
        print("[System] Locating ATM Strikes for Indices...")
        nfo_tokens = [t for t in self.token_master if t['exch_seg'] == 'NFO' and t['instrumenttype'] == 'OPTIDX']
        
        for idx_name, cfg in self.index_config.items():
            spot_ltp = self.get_spot_ltp(cfg['exch'], cfg['sym'])
            if not spot_ltp:
                print(f"[Warning] Could not get spot for {idx_name}")
                continue
                
            atm_strike = round(spot_ltp / cfg['step']) * cfg['step']
            print(f"[{idx_name}] Spot: {spot_ltp} | ATM Strike: {atm_strike}")
            
            # Find closest nearest expiry for this symbol
            prefix = "NIFTY" if idx_name == "NIFTY_50" else "BANKNIFTY" if idx_name == "NIFTY_BANK" else "FINNIFTY"
            
            # Filter tokens by prefix and strike
            opts = [t for t in nfo_tokens if t['name'] == prefix and float(t['strike']) == float(atm_strike)*100]
            if not opts: continue
            
            # Sort by expiry to get current week
            opts.sort(key=lambda x: datetime.datetime.strptime(x['expiry'], '%d%b%Y'))
            ce_token, pe_token = None, None
            
            for o in opts:
                if o['symbol'].endswith('CE') and not ce_token: ce_token = o
                if o['symbol'].endswith('PE') and not pe_token: pe_token = o
                if ce_token and pe_token: break
            
            if ce_token and pe_token:
                self.tracked_tokens[ce_token['token']] = {"index": idx_name, "type": "CE", "symbol": ce_token['symbol']}
                self.tracked_tokens[pe_token['token']] = {"index": idx_name, "type": "PE", "symbol": pe_token['symbol']}
                print(f"[{idx_name}] Tracked CE: {ce_token['symbol']} | PE: {pe_token['symbol']}")
                
                # Initialize live_data structures
                if idx_name not in self.live_data:
                    self.live_data[idx_name] = {"CE_LTP": 0, "PE_LTP": 0, "CE_OI": 1, "PE_OI": 1, "ATM_Strike": atm_strike}

    def _on_data(self, wsapp, msg):
        try:
            token = msg.get("token")
            if token in self.tracked_tokens:
                meta = self.tracked_tokens[token]
                idx_name = meta["index"]
                
                # Best effort parsing of tick data (LTP and OI)
                if "last_traded_price" in msg:
                    self.live_data[idx_name][f"{meta['type']}_LTP"] = msg["last_traded_price"] / 100.0
                if "open_interest" in msg:
                    self.live_data[idx_name][f"{meta['type']}_OI"] = msg["open_interest"]
        except Exception as e:
            pass

    def _manage_connection(self):
        token_list = [{"exchangeType": 2, "tokens": list(self.tracked_tokens.keys())}]
        while True:
            try:
                print("[System] (Re)Initializing SmartWebSocketV2...")
                if self.ws:
                    try:
                        self.ws.close()
                    except:
                        pass

                self.ws = SmartWebSocketV2(self.session['jwtToken'], API_KEY, CLIENT_CODE, self.feed_token)
                self.ws.on_data = self._on_data

                def _run_ws():
                    try:
                        self.ws.connect()
                    except Exception as e:
                        print(f"[Warning] WebSocket Error: {e}")

                ws_thread = threading.Thread(target=_run_ws, daemon=True)
                ws_thread.start()

                time.sleep(3)
                print("[System] Subscribing to Token Streams...")
                self.ws.subscribe("my_corr_id", 3, token_list)

                # Monitor connection health by checking if thread is alive
                while ws_thread.is_alive():
                    time.sleep(5)

                print("[Warning] WebSocket connection lost. Reconnecting in 5 seconds...")
            except Exception as e:
                print(f"[Error] Connection manager failed: {e}. Retrying...")
            
            time.sleep(5)

    def start_websocket(self):
        if not self.tracked_tokens:
            print("[System] No tokens to track. Exiting.")
            return

        m_thread = threading.Thread(target=self._manage_connection, daemon=True)
        m_thread.start()

    def log_5m_data(self):
        os.makedirs("dataset", exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("Datetime,Index,Spot_Price,ATM_Strike,CE_LTP,PE_LTP,CE_OI,PE_OI\n")

        last_timestamp = None
        
        while True:
            now = datetime.datetime.now()
            # Wait until next 5-minute boundary
            sleep_sec = 300 - (now.minute % 5) * 60 - now.second
            target_time = now + datetime.timedelta(seconds=sleep_sec)
            
            # Stop ingestion after 15:15 (Interday square-off / Market End logic)
            if target_time.time() > datetime.time(15, 15):
                print("[System] Market close logic triggered (Post 15:15). Shutting down for the day.")
                break

            time.sleep(sleep_sec)

            timestamp = target_time.strftime("%Y-%m-%d %H:%M:00")
            
            # Prevent double-logging if Windows time.sleep returns milliseconds early
            if timestamp == last_timestamp:
                time.sleep(1)
                continue
            last_timestamp = timestamp

            rows = []
            for idx_name, data in self.live_data.items():
                cfg = self.index_config.get(idx_name)
                
                spot_ltp = self.get_spot_ltp(cfg['exch'], cfg['sym']) if cfg else None
                if spot_ltp is not None:
                    data['Spot_LTP'] = spot_ltp
                else:
                    spot_ltp = data.get('Spot_LTP', 0.0)
                
                rows.append(f"{timestamp},{idx_name},{spot_ltp},{data['ATM_Strike']},{data['CE_LTP']:.2f},{data['PE_LTP']:.2f},{data['CE_OI']},{data['PE_OI']}\n")
                
                pcr = data["PE_OI"] / data["CE_OI"] if data["CE_OI"] > 0 else 0 
                straddle = data["CE_LTP"] + data["PE_LTP"]
                print(f"[{timestamp}] Logged {idx_name} -> Spot: {spot_ltp} | ATM: {data['ATM_Strike']} | Straddle: {straddle:.2f} | PCR: {pcr:.4f}")

            try:
                with open(self.csv_path, "a") as f:
                    f.writelines(rows)
            except PermissionError:
                print(f"[Error] The file {self.csv_path} is currently locked by another program (like Excel). Could not save {timestamp} data.")
            except Exception as e:
                print(f"[Error] Failed to write to CSV: {e}")

    def run(self):
        self.login()
        self.load_token_master()
        self.setup_tracked_tokens()
        self.start_websocket()
        
        print("\n[System] Ingestion Engine Live! Aggregating 5-Minute Structural Features...")
        self.log_5m_data()

if __name__ == "__main__":
    ingester = OptionsIngester()
    ingester.run()
