import pandas as pd
import numpy as np

def compute_oi_volume_features(df):
    """
    Workstream B: OI and Volume Feature Substrates on NIFTY Futures.
    Expects dataframe with columns: ['close', 'volume', 'oi', 'cash_close'] (resampled to 5m or 30m).
    """
    df = df.copy()
    
    # Needs minimum history for rolling windows
    if len(df) < 21:
        return df
        
    df['ret'] = df['close'].pct_change()
    vol_mean_20 = df['volume'].rolling(20).mean()
    
    # Feature B1: Amihud illiquidity with OI/Volume normalization
    df['norm_volume'] = df['volume'] / vol_mean_20.replace(0, 1)
    df['amihud_oi_norm'] = df['ret'].abs() / df['norm_volume'].replace(0, 1)
    
    # Feature B2: OI rate of change (Informed positioning signal)
    oi_mean_20 = df['oi'].rolling(20).mean()
    df['oi_roc'] = (df['oi'] - df['oi'].shift(1)) / oi_mean_20.replace(0, 1)
    
    # Feature B3: Basis signal (Needs synchronous cash index close)
    if 'cash_close' in df.columns:
        df['basis'] = df['close'] - df['cash_close']
        df['basis_roc'] = (df['basis'] - df['basis'].shift(3)) / 3
        
    # Feature B4: Volume-OI divergence
    # Positive score = Noise trading; Negative score = Informed directional commitment
    vol_std = df['volume'].rolling(20).std().replace(0, 1)
    df['vol_z'] = (df['volume'] - vol_mean_20) / vol_std
    
    oi_roc_mean = df['oi_roc'].rolling(20).mean()
    oi_roc_std = df['oi_roc'].rolling(20).std().replace(0, 1)
    df['oi_roc_z'] = (df['oi_roc'] - oi_roc_mean) / oi_roc_std
    
    df['vol_oi_div'] = df['vol_z'] - df['oi_roc_z']
    
    return df