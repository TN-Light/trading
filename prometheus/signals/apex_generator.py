# ============================================================================
# PROMETHEUS APEX — Signal Generator Wrapping
# ============================================================================  
"""
Unified wrapper that feeds tick data into the APEX components.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .qrd_estimator import QrdEstimator
from .aes_fusion import AesFusionEngine
from .strike_gravity import StrikeGravityMapper
from .expiry_clock import ExpiryClock

from prometheus.signals.technical import calculate_vwap, calculate_session_vwap, calculate_ema, calculate_supertrend, calculate_rsi
from prometheus.utils.indian_market import (
    get_atm_strike,
    days_to_expiry,
    get_lot_size,
    get_strike_interval,
    is_weekly_expiry_day,
    is_expiry_thursday_session,
    is_monthly_expiry_session,
    get_expiry_date,
)
from prometheus.utils.options_math import black_scholes_price, calculate_greeks

from loguru import logger

class ApexSignalGenerator:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.qrd = QrdEstimator()
        self.aes = AesFusionEngine()
        self.gravity = StrikeGravityMapper()
        self.expiry = ExpiryClock()
        self.stats = {
            "bars_checked": 0,
            "reject_confluence": 0,
            "reject_compression": 0,
            "reject_aes": 0,
            "reject_tvi": 0,
            "reject_pricing": 0,
            "signals_passed": 0
        }

    def precompute(self, data: pd.DataFrame):
        """Precompute indicators for faster backtesting (O(1) lookups instead of O(N))."""
        df = data.copy()
        self._pre_vwap = calculate_vwap(df)
        self._pre_session_vwap = calculate_session_vwap(df)
        self._pre_ema9 = calculate_ema(df["close"], 9)
        self._pre_ema21 = calculate_ema(df["close"], 21)
        self._pre_supertrend = calculate_supertrend(df, 10, 3.0)
        self._pre_rsi = calculate_rsi(df, 14)
        self._is_precomputed = True
        logger.info(f"[{self.symbol} APEX] Precomputed indicators for {len(df)} rows.")

    def _log_stats(self):
        logger.info(f"[{self.symbol} APEX SIGNAL STATS] Checked: {self.stats['bars_checked']} | "
                    f"Passed: {self.stats['signals_passed']} | "
                    f"Rejects -> Confluence: {self.stats['reject_confluence']}, "
                    f"Compression: {self.stats['reject_compression']}, "
                    f"AES: {self.stats['reject_aes']}, "
                    f"TVI: {self.stats['reject_tvi']}")

    def generate(self, data_so_far: pd.DataFrame, current_oi_chain: Dict = None) -> Optional[Dict]:
        """
        Calculates the APEX 0-100 edge score and returns standard dict for the backtest engine.
        """
        if len(data_so_far) < 50:
            return None
            
        self.stats["bars_checked"] += 1
        
        # Periodically dump stats so we see them in the log without needing a destructor hook
        if self.stats["bars_checked"] % 500 == 0:
            self._log_stats()

        current_bar = data_so_far.iloc[-1]
        dt = pd.to_datetime(current_bar.get("timestamp", data_so_far.index[-1]))
        bar_date = dt.date()

        # Expiry-session context tags for runtime routing and forward validation.
        try:
            dte = max(0, int(days_to_expiry(self.symbol, from_date=bar_date)))
        except Exception:
            dte = 1
        is_expiry_day = bool(is_weekly_expiry_day(self.symbol, bar_date))
        is_expiry_thursday = bool(is_expiry_thursday_session(self.symbol, bar_date))
        is_monthly_expiry = bool(is_monthly_expiry_session(self.symbol, bar_date))

        # Gamma Ambush mode: only allow entries in the 10:45-11:15 window.
        if is_expiry_thursday:
            hhmm = dt.hour * 100 + dt.minute
            if hhmm < 1045 or hhmm > 1115:
                self.stats["reject_aes"] += 1
                return None

        # --- 1. 5-Component Technical Stack ---
        df = data_so_far
        idx_len = len(df)
        
        if getattr(self, "_is_precomputed", False) and idx_len <= len(self._pre_vwap):
            vwap_data = self._pre_vwap.iloc[:idx_len]
            session_vwap_data = self._pre_session_vwap.iloc[:idx_len]
            ema9 = self._pre_ema9.iloc[:idx_len]
            ema21 = self._pre_ema21.iloc[:idx_len]
            supertrend_data = self._pre_supertrend.iloc[:idx_len]
            rsi_data = self._pre_rsi.iloc[:idx_len]
        else:
            df_copy = df.copy()
            vwap_data = calculate_vwap(df_copy)
            session_vwap_data = calculate_session_vwap(df_copy)
            ema9 = calculate_ema(df_copy["close"], 9)
            ema21 = calculate_ema(df_copy["close"], 21)
            supertrend_data = calculate_supertrend(df_copy, 10, 3.0)
            rsi_data = calculate_rsi(df_copy, 14)
        
        curr_vwap = float(vwap_data["vwap"].iloc[-1])
        curr_svwap = float(session_vwap_data["vwap"].iloc[-1])
        curr_ema9 = float(ema9.iloc[-1])
        curr_ema21 = float(ema21.iloc[-1])
        st_dir = float(supertrend_data["supertrend_direction"].iloc[-1])
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
            
        # RSI Divergence OR Fresh EMA Cross requirement decouple
        bull_div = False
        bear_div = False
        if len(df) >= 15:
            recent_close_min = df["close"].iloc[-4:-1].min()
            past_close_min = df["close"].iloc[-15:-4].min()
            recent_rsi_min = rsi_data.iloc[-4:-1].min()
            past_rsi_min = rsi_data.iloc[-15:-4].min()
            bull_div = (recent_close_min <= past_close_min * 1.001) and (recent_rsi_min > past_rsi_min)

            recent_close_max = df["close"].iloc[-4:-1].max()
            past_close_max = df["close"].iloc[-15:-4].max()
            recent_rsi_max = rsi_data.iloc[-4:-1].max()
            past_rsi_max = rsi_data.iloc[-15:-4].max()
            bear_div = (recent_close_max >= past_close_max * 0.999) and (recent_rsi_max < past_rsi_max)

        # Fresh EMA Cross logic (within last 3 bars)
        bull_ema_cross = False
        bear_ema_cross = False
        if len(df) >= 4:
            e9_vals = ema9.iloc[-4:].values
            e21_vals = ema21.iloc[-4:].values
            # indices: 0 (-4), 1 (-3), 2 (-2), 3 (-1: current)
            # For i in range(1, 4): checks (-2 vs -1), (-3 vs -2), (-4 vs -3)
            # prev = idx-1, curr = idx
            for idx in range(1, 4):
                e9_prev = float(e9_vals[idx-1])
                e21_prev = float(e21_vals[idx-1])
                e9_curr = float(e9_vals[idx])
                e21_curr = float(e21_vals[idx])
                
                if e9_prev <= e21_prev and e9_curr > e21_curr:
                    bull_ema_cross = True
                    break
                if e9_prev >= e21_prev and e9_curr < e21_curr:
                    bear_ema_cross = True
                    break

        if bull_div or bull_ema_cross: bull_score += 1
        if bear_div or bear_ema_cross: bear_score += 1
        
        # Confluence filter
        if bull_score >= 3 and bear_score < 2:
            signal_direction = 1.0
        elif bear_score >= 3 and bull_score < 2:
            signal_direction = -1.0
        else:
            self.stats["reject_confluence"] += 1
            return None

        # Thursday expiry directional flow gate aligned to session VWAP.
        if is_expiry_thursday:
            gate_vwap = curr_svwap if not pd.isna(curr_svwap) else curr_vwap
            if signal_direction > 0 and close <= gate_vwap:
                self.stats["reject_confluence"] += 1
                return None
            if signal_direction < 0 and close >= gate_vwap:
                self.stats["reject_confluence"] += 1
                return None

        # --- 2. QRD and Context ---
        qrd_state = self.qrd.estimate(data_so_far)
        time_alpha = self.expiry.evaluate_window(dt, self.symbol)

        atr = self.qrd._calc_atr(data_so_far)

        # COMPRESSION COIL & RETEST GATE
        # Replaces velocity gate to prevent buying exhaustion peaks.
        # We demand either a breakout from a tight coil, or a retest of the EMA21.
        if len(data_so_far) >= 60:
            coil_len = 6
            past_highs = df["high"].iloc[- coil_len - 1 : -1].values
            past_lows = df["low"].iloc[- coil_len - 1 : -1].values
            coil_high = past_highs.max()
            coil_low = past_lows.min()
            
            # Ratio-based compression: 6-bar range / 60-bar range < 0.35
            long_highs = df["high"].iloc[-61:-1].values
            long_lows = df["low"].iloc[-61:-1].values
            long_range = float(long_highs.max() - long_lows.min())
            
            coil_range = float(coil_high - coil_low)
            is_coiled = False
            if long_range > 0:
                compression_ratio = coil_range / long_range
                is_coiled = compression_ratio < 0.35
            is_breakout = (signal_direction > 0 and close > coil_high) or (signal_direction < 0 and close < coil_low)
            
            # Retest condition: Price recently tapped the EMA21 but rejected it (leaving a wick)
            retest_bull = (float(current_bar["low"]) <= curr_ema21) and (close > curr_ema21)
            retest_bear = (float(current_bar["high"]) >= curr_ema21) and (close < curr_ema21)
            
            # Additional structural confirmation for retest
            retest_valid = (signal_direction > 0 and retest_bull) or (signal_direction < 0 and retest_bear)
            
            if not ((is_coiled and is_breakout) or retest_valid):
                self.stats["reject_compression"] += 1
                return None

        target_price_index = close + atr * 3.0 if signal_direction > 0 else close - atr * 3.0

        entry_trigger = float(current_bar["high"]) * 1.0005 if signal_direction > 0 else float(current_bar["low"]) * (1.0 - 0.0005)
        underlying_sl = float(close) - float(atr) * 2.0 if signal_direction > 0 else float(close) + float(atr) * 2.0

        
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
            self.stats["reject_aes"] += 1
            return None

        # Monthly expiry policy: suppress options unless conviction is very high.
        if is_monthly_expiry and edge_score <= 72:
            self.stats["reject_aes"] += 1
            return None

        # --- 4. Option Pricing & Dual-Trigger Stop ---
        atm_strike = get_atm_strike(close, self.symbol)

        if signal_direction > 0:
            opt_type = "CE"
        else:
            opt_type = "PE"

        pricing_dte = max(1, int(dte))
        T = float(pricing_dte) / 365.0
        daily_vol = max(1e-6, float(atr) / close)
        sigma = daily_vol * np.sqrt(78 * 252) # intraday 5min approx
        if pd.isna(sigma) or sigma == 0:
            sigma = 0.15
        r = 0.065

        strike = atm_strike
        premium = None
        greeks = None

        # Gamma Ambush strike routing: OTM-first with strict premium band.
        if is_expiry_thursday:
            interval = get_strike_interval(self.symbol)
            if opt_type == "CE":
                offsets = [1, 2, 3, 0, -1]
            else:
                offsets = [-1, -2, -3, 0, 1]

            premium_floor = 15.0
            premium_ceiling = 50.0
            target_mid = (premium_floor + premium_ceiling) / 2.0

            best = None
            for off in offsets:
                candidate_strike = atm_strike + (off * interval)
                candidate_premium = black_scholes_price(close, candidate_strike, T, r, sigma, opt_type)
                if pd.isna(candidate_premium) or candidate_premium <= 0:
                    continue

                if premium_floor <= float(candidate_premium) <= premium_ceiling:
                    distance = abs(float(candidate_premium) - target_mid)
                    if best is None or distance < best[0]:
                        best = (distance, candidate_strike, float(candidate_premium))

            if best is None:
                self.stats["reject_pricing"] += 1
                return None

            strike = best[1]
            premium = best[2]
            greeks = calculate_greeks(close, strike, T, r, sigma, opt_type)
        else:
            premium = black_scholes_price(close, strike, T, r, sigma, opt_type)
            if pd.isna(premium) or premium <= 0:
                self.stats["reject_pricing"] += 1
                return None
            greeks = calculate_greeks(close, strike, T, r, sigma, opt_type)

        delta = abs(greeks.get("delta", 0.5))

        # --- TVI GATE (Theta Vulnerability Index) ---
        # The DTE multiplier is the piece that makes TVI accurate for near-expiry options.
        # Weekly options with 1-2 DTE run 3-5x the annualized estimate intraday.
        theta = abs(greeks.get("theta", 0.0))
        DTE_MULTS = {0: 5.0, 1: 4.5, 2: 3.5, 3: 2.5, 4: 2.0}
        implied_dte = int(min(pricing_dte, 4))
        dte_multiplier = DTE_MULTS.get(implied_dte, 1.0)
        
        # Max bars allowed by velocity gate
        max_duration_bars = 8 
        # Intraday holding time in days (e.g. 5min bars * 8)
        holding_time_days = max_duration_bars / 75.0 
        
        # Worst-case theta erosion
        worst_case_theta_loss = theta * dte_multiplier * holding_time_days
        
        # Stop loss buffer (how much premium can drop before 40% decay)
        sl_buffer = premium * 0.40
        
        tvi = sl_buffer / max(0.01, worst_case_theta_loss)
        
        # A TVI below 1.5 means theta alone can fire the SL without any adverse price movement
        if tvi < 1.5:
            self.stats["reject_tvi"] += 1
            return None

        # Stop loss: dual-trigger (40% premium decay OR 1.5x ATR adverse)
        atr_sl_index = float(atr) * 1.5
        atr_sl_premium_drop = float(delta) * atr_sl_index
        
        premium_sl_atr = max(0.5, premium - atr_sl_premium_drop)
        premium_sl_decay = premium * 0.60 # 40% decay means 60% remaining
        
        # Pick the tighter stop (highest price value)
        final_sl = max(premium_sl_atr, premium_sl_decay)
        
        # Target: 3.0x ATR
        target_index_move = float(atr) * 3.0
        underlying_target = float(close) + target_index_move if signal_direction > 0 else float(close) - target_index_move
        premium_target = premium + (float(delta) * target_index_move)

        self.stats["signals_passed"] += 1

        lot_size = get_lot_size(self.symbol)
        lots = max(1, int(quantity // max(lot_size, 1)))

        expiry_date_str = ""
        try:
            expiry_date_str = get_expiry_date(self.symbol, from_date=bar_date).isoformat()
        except Exception:
            expiry_date_str = ""

        return {
            "action": f"BUY_{opt_type}",
            "direction": "bullish" if signal_direction > 0 else "bearish",      
            "entry_price": round(float(premium), 2),
            "stop_loss": round(float(final_sl), 2),
            "underlying_sl": round(float(underlying_sl), 2),
            "underlying_target": round(float(underlying_target), 2),
            "entry_trigger": round(float(entry_trigger), 2),
            "target": round(float(premium_target), 2),
            "quantity": quantity * get_lot_size(self.symbol), 
            "edge_score": float(edge_score),
            "tier": tier,
            "aes_factors": factors,
            "regime": qrd_state.dominant_regime,
            "strategy": "apex_intraday",
            "max_bars": 8,  # Reduced max hold time to match velocity gate expectation
            "delta": delta,
            "strike": strike,
            "instrument_type": "options",
            "option_type": opt_type,
            "expiry": expiry_date_str,
            "dte": int(dte),
            "is_expiry_day": is_expiry_day,
            "is_expiry_thursday": is_expiry_thursday,
            "is_monthly_expiry": is_monthly_expiry,
            "bar_timestamp": dt.isoformat(),
            "spot_at_signal": round(float(close), 2),
            "lot_size": lot_size,
            "lots": lots,
            "lot_cost": round(float(premium) * lot_size, 2),
        }
