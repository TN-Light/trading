import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, time

class HolyGrailSignalGenerator:
    """
    HOLY GRAIL ENGINE - Momentum Pullback with Smart Time Stops
    Fixed Asymmetric Risk Profile by capping maximum holding time.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Optimal Strategy Configuration
        self.PULLBACK = 40
        self.BOUNCE = 45
        self.SL_ATR = 2.5
        self.TP_ATR = 0.8
        self.TIME_STOP_BARS = 16
        
    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculates features and returns the latest live entry setup if present.
        """
        if df is None or len(df) < 30:
            return []
            
        # 1. Feature Generation
        df = df.copy()
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=4, adjust=False).mean()
        ema_down = down.ewm(com=4, adjust=False).mean()
        rs = ema_up / ema_down
        df['rsi_fast'] = 100 - (100 / (1 + rs))

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        df['tr'] = pd.DataFrame({'1':tr1, '2':tr2, '3':tr3}).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['volume'] = df['volume'].replace(0, 1)

        # Check only the most recently closed bar for LIVE signal generation.
        # i.e., index - 1 or - 2. We use the last closed bar:
        last_i = len(df) - 1

        # Safe VWAP calc per date
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
            bar_time = df.index[last_i].time()
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp']).dt.date.values
            bar_time = pd.to_datetime(df['timestamp'].iloc[last_i]).time()
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date']).dt.date.values
            bar_time = pd.to_datetime(df['date'].iloc[last_i]).time()
        else:
            return []
            
        df['date'] = dates

        cum_vol = []
        cum_pv = []
        c_v = 0
        c_pv = 0
        last_d = None
        
        for i in range(len(df)):
            d = dates[i]
            if d != last_d:
                c_v = 0
                c_pv = 0
                last_d = d
            c_v += df['volume'].iloc[i]
            c_pv += df['typical_price'].iloc[i] * df['volume'].iloc[i]
            cum_vol.append(c_v)
            cum_pv.append(c_pv)
            
        df['cum_vol'] = cum_vol
        df['cum_pv'] = cum_pv
        df['vwap'] = df['cum_pv'] / df['cum_vol']

        df['adx_up'] = df['high'] - df['high'].shift(1)
        df['adx_down'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['adx_up'] > df['adx_down']) & (df['adx_up'] > 0), df['adx_up'], 0)
        df['minus_dm'] = np.where((df['adx_down'] > df['adx_up']) & (df['adx_down'] > 0), df['adx_down'], 0)
        atr14 = df['tr'].rolling(14).mean()
        plus_di = 100 * (pd.Series(df['plus_dm']).ewm(alpha=1/14).mean() / atr14)   
        minus_di = 100 * (pd.Series(df['minus_dm']).ewm(alpha=1/14).mean() / atr14) 
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14).mean()

        signals = []
        
        if bar_time.hour >= 15 or bar_time.hour < 9 or (bar_time.hour == 9 and bar_time.minute < 30):
            return signals # Out of session window
            return signals

        uptrend = (df['ema9'].iloc[last_i] > df['ema21'].iloc[last_i]) and \
                  (df['close'].iloc[last_i] > df['vwap'].iloc[last_i]) and \
                  (df['adx'].iloc[last_i] > 15)
                  
        downtrend = (df['ema9'].iloc[last_i] < df['ema21'].iloc[last_i]) and \
                    (df['close'].iloc[last_i] < df['vwap'].iloc[last_i]) and \
                    (df['adx'].iloc[last_i] > 15)
        
        curr_p = df['close'].iloc[last_i]
        curr_atr = df['atr'].iloc[last_i]

        if uptrend and (df['rsi_fast'].iloc[last_i-1] < self.PULLBACK) and (df['rsi_fast'].iloc[last_i] >= self.BOUNCE):
            sl = curr_p - (self.SL_ATR * curr_atr)
            tp = curr_p + (self.TP_ATR * curr_atr)
            
            signals.append({
                'direction': 'bullish',
                'entry_price': curr_p,
                'stop_loss': sl,
                'target': tp,
                'risk_reward': abs(tp - curr_p) / abs(curr_p - sl) if abs(curr_p - sl) > 0 else 0,
                'time_stop_bars': self.TIME_STOP_BARS,
                'confidence': 0.85
            })

        elif downtrend and (df['rsi_fast'].iloc[last_i-1] > (100 - self.PULLBACK)) and (df['rsi_fast'].iloc[last_i] <= (100 - self.BOUNCE)):
            sl = curr_p + (self.SL_ATR * curr_atr)
            tp = curr_p - (self.TP_ATR * curr_atr)
            
            signals.append({
                'direction': 'bearish',
                'entry_price': curr_p,
                'stop_loss': sl,
                'target': tp,
                'risk_reward': abs(tp - curr_p) / abs(sl - curr_p) if abs(sl - curr_p) > 0 else 0,
                'time_stop_bars': self.TIME_STOP_BARS,
                'confidence': 0.85
            })

        return signals
