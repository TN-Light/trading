# ============================================================================
# PROMETHEUS APEX — Quantum Regime Detection (QRD)
# ============================================================================
"""
Quantum Regime Detection: Replaces binary regime classification with a 
continuous probability tensor.

Markets exist in superposition: e.g., 60% trending, 40% reverting.
This module outputs proportional conviction scores based on ADX, PCR, 
and intrinsic volatility, allowing downstream sizing to adapt continuously.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class QuantumRegime:
    """Continuous regime probability tensor."""
    trend_prob: float         # 0.0 to 1.0
    reversion_prob: float     # 0.0 to 1.0
    volatile_prob: float      # 0.0 to 1.0
    
    # Combined directional conviction (-1.0 to 1.0)
    directional_bias: float   
    
    # The dominant state (for descriptive logging only, not for hard gating)
    dominant_regime: str      
    
    # Details for AES decomposition
    factors: Dict[str, float] 

class QrdEstimator:
    def __init__(self, ema_fast=9, ema_slow=21, atr_period=14, adx_period=14, hurst_window=20, micro_window=3, vol_window=10):
        self.adx_period = adx_period
        self.vol_window = vol_window
        
    def estimate(
        self, 
        df: pd.DataFrame, 
        pcr: float = 1.0, 
        vix: float = 15.0
    ) -> QuantumRegime:
        """
        Evaluate market superposition based on pure quant mechanics.
        """
        if len(df) < self.adx_period * 2:
            return self._neutral_regime()
            
        # 1. Trend component via ADX & DMI
        adx, pdi, mdi = self._calc_adx(df)
        trend_strength = min(1.0, max(0.0, (adx - 15) / 25))  # Normalizes ADX 15-40 to 0-1
        
        # Directional bias from DMI and Price vs SMA
        dmi_bias = (pdi - mdi) / (pdi + mdi + 1e-9)
        close = df["close"].iloc[-1]
        sma50 = df["close"].rolling(50).mean().iloc[-1]
        price_bias = 1.0 if close > sma50 else -1.0
        
        # 2. Volatility component via VIX and ATR normalized
        atr = self._calc_atr(df)
        atr_pct = (atr / close) * 100
        vol_score = min(1.0, (vix / 20.0) * (atr_pct / 1.5)) 
        
        # 3. Mean Reversion via Hurst / RSI Extremes
        rsi = self._calc_rsi(df["close"])
        rsi_extreme = abs(rsi - 50) / 50  # 0 at 50, 1 at 0/100
        reversion_score = min(1.0, rsi_extreme + (1 - trend_strength) * 0.5)
        
        # Integrate PCR (Options Sentiment) if available
        # PCR > 1.2 implies bearish sentiment (reversion if price up)
        # PCR < 0.8 implies bullish sentiment
        
        # Normalize into a probability tensor (sum = 1.0)
        raw_trend = trend_strength * 1.5
        raw_rev = reversion_score * 1.0
        raw_vol = vol_score * 0.8
        
        total = raw_trend + raw_rev + raw_vol
        if total == 0:
             return self._neutral_regime()
             
        t_prob = raw_trend / total
        r_prob = raw_rev / total
        v_prob = raw_vol / total
        
        dir_bias = np.clip((dmi_bias * 0.6 + price_bias * 0.4) * t_prob, -1.0, 1.0)
        
        dom = "TREND" if t_prob > max(r_prob, v_prob) else ("REVERSION" if r_prob > v_prob else "VOLATILE")
        
        return QuantumRegime(
            trend_prob=float(t_prob),
            reversion_prob=float(r_prob),
            volatile_prob=float(v_prob),
            directional_bias=float(dir_bias),
            dominant_regime=dom,
            factors={
                "adx_strength": float(trend_strength),
                "vol_score": float(vol_score),
                "rsi_extreme": float(rsi_extreme),
                "pcr_impact": float(pcr)
            }
        )

    def _neutral_regime(self) -> QuantumRegime:
        return QuantumRegime(0.33, 0.33, 0.34, 0.0, "UNKNOWN", {})
        
    def _calc_atr(self, df: pd.DataFrame, period=14) -> float:
        high = df["high"]
        low = df["low"]
        close_prev = df["close"].shift(1)
        tr = np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
        return tr.rolling(period).mean().iloc[-1]
        
    def _calc_rsi(self, series: pd.Series, period=14) -> float:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs.iloc[-1]))
        
    def _calc_adx(self, df: pd.DataFrame, period=14) -> Tuple[float, float, float]:
        high = df["high"]
        low = df["low"]
        close_prev = df["close"].shift(1)
        
        tr = np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
        plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
        minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)
        
        tr_smooth = pd.Series(tr).rolling(period).sum()
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(period).mean().iloc[-1]
        
        return adx, plus_di.iloc[-1], minus_di.iloc[-1]

