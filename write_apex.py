import pandas as pd
import numpy as np

content = """# ============================================================================
# PROMETHEUS APEX — Signal Generator Wrapping
# ============================================================================  
\"\"\"
Unified wrapper that feeds tick data into the APEX components.
\"\"\"

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .qrd_estimator import QrdEstimator
from .aes_fusion import AesFusionEngine
from .strike_gravity import StrikeGravityMapper
from .expiry_clock import ExpiryClock

from prometheus.signals.technical import calculate_vwap, calculate_session_vwap, calculate_ema, calculate_supertrend, calculate_rsi
from prometheus.utils.indian_market import get_atm_strike, days_to_expiry, get_lot_size, get_strike_interval
from prometheus.utils.options_math import black_scholes_price, calculate_greeks

class ApexSignalGenerator:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.qrd = QrdEstimator()
        self.aes = AesFusionEngine()
        self.gravity = StrikeGravityMapper()
        self.expiry = ExpiryClock()

    def generate(self, data_so_far: pd.DataFrame, current_oi_chain: Dict = None) -> Optional[Dict]:
        \"\"\"
        Calculates the APEX 0-100 edge score and returns standard dict for the backtest engine.
        \"\"\"
        if len(data_so_far) < 50:
            return None

        current_bar = data_so_far.iloc[-1]
        dt = pd.to_datetime(current_bar.get("timestamp", data_so_far.index[-1]))

        # --- 1. 5-Component Technical Stack ---
        df = data_so_far.copy()
        
        vwap_data = calculate_vwap(df)
        session_vwap_data = calculate_session_vwap(df)
        ema9 = calculate_ema(df, 9)
        ema21 = calculate_ema(df, 21)
        supertrend_data = calculate_supertrend(df, 10, 3.0)
        rsi_data = calculate_rsi(df, 14)
        
        curr_vwap = float(vwap_data["vwap"].iloc[-1])
        curr_svwap = float(session_vwap_data["session_vwap"].iloc[-1])
        curr_ema9 = float(ema9.iloc[-1])
        curr_ema21 = float(ema21.iloc[-1])
        st_dir = float(supertrend_data["direction"].iloc[-1])
        curr_rsi = float(rsi_data.iloc[-1])
        
        close = float(current_bar["close"])
        
        bull_score = 0
        bear_score = 0
        
        if close > curr_vwap: bull_score += 1
        elif close < curr_vwap: bear_score += 1
            
        if close > curr_svwap: bull_score += 1
        elif close < curr_svwap: bear_score += 1
            
        if curr_ema9 > curr_ema21: bull_score += 1
        elif curr_ema9 < curr_ema21: bear_score += 1
            
        if st_dir == 1: bull_score += 1
        elif st_dir == -1: bear_score += 1
            
        if curr_rsi > 55: bull_score += 1
        elif curr_rsi < 45: bear_score += 1
        
        # Confluence filter
        if bull_score >= 3 and bear_score < 2:
            signal_direction = 1.0
        elif bear_score >= 3 and bull_score < 2:
            signal_direction = -1.0
        else:
            return None

        # --- 2. QRD and Context ---
        qrd_state = self.qrd.estimate(data_so_far)
        time_alpha = self.expiry.evaluate_window(dt, self.symbol)

        atr = self.qrd._calc_atr(data_so_far)
        target_price_index = close + atr * 3.0 if signal_direction > 0 else close - atr * 3.0
        
        gravity_penalty = self.gravity.calculate_gravity_penalty(
            close, target_price_index, atr, current_oi_chain or {}     
        )

        features = {
            "confluence_score": max(bull_score, bear_score), 
            "macro_alignment": 1.0
        }

        # --- 3. Edge Calculation & Routing ---
        edge_score, factors = self.aes.calculate_edge_score(
            signal_direction=signal_direction,
            qrd_state=qrd_state,
            signal_features=features,
            gravity_penalty=gravity_penalty,
            is_expiry_thursday=time_alpha["is_expiry_day"],
            is_opening_session=time_alpha["is_morning_squeeze"]
        )

        tier, quantity = self.aes.get_bounded_sizing(edge_score, available_capital=15000, lot_premium=5000)
        if tier.startswith("REJECT"):
            return None

        # --- 4. Option Pricing & Dual-Trigger Stop ---
        try:
            dte = max(1, days_to_expiry(self.symbol, from_date=dt.date()))
        except:
            dte = 1

        atm_strike = get_atm_strike(close, self.symbol)
        
        if signal_direction > 0:
            strike = atm_strike
            opt_type = "CE"
        else:
            strike = atm_strike
            opt_type = "PE"

        T = float(dte) / 365.0
        daily_vol = max(1e-6, float(atr) / close)
        sigma = daily_vol * np.sqrt(78 * 252) # intraday 5min approx
        if pd.isna(sigma) or sigma == 0:
            sigma = 0.15
        r = 0.065

        premium = black_scholes_price(close, strike, T, r, sigma, opt_type)
        if pd.isna(premium) or premium <= 0:
            return None
            
        greeks = calculate_greeks(close, strike, T, r, sigma, opt_type)
        delta = abs(greeks.get("delta", 0.5))

        # Stop loss: dual-trigger (40% premium decay OR 1.5x ATR adverse)
        atr_sl_index = float(atr) * 1.5
        atr_sl_premium_drop = float(delta) * atr_sl_index
        
        premium_sl_atr = max(0.5, premium - atr_sl_premium_drop)
        premium_sl_decay = premium * 0.60 # 40% decay means 60% remaining
        
        # Pick the tighter stop (highest price value)
        final_sl = max(premium_sl_atr, premium_sl_decay)
        
        # Target: 3.0x ATR
        target_index_move = float(atr) * 3.0
        premium_target = premium + (float(delta) * target_index_move)

        return {
            "action": f"BUY_{opt_type}",
            "direction": "bullish" if signal_direction > 0 else "bearish",      
            "entry_price": round(float(premium), 2),
            "stop_loss": round(float(final_sl), 2),
            "target": round(float(premium_target), 2),
            "quantity": quantity * get_lot_size(self.symbol), 
            "edge_score": float(edge_score),
            "tier": tier,
            "aes_factors": factors,
            "regime": qrd_state.dominant_regime,
            "strategy": "apex_intraday",
            "max_bars": 12,
            "delta": delta,
            "strike": strike,
            "instrument_type": "options"
        }
"""
with open("prometheus/signals/apex_generator.py", "w", encoding="utf-8") as f:
    f.write(content)
