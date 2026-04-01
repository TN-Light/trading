import pandas as pd
import numpy as np

def analyze_pcr_mean_reversion(df_daily):
    """
    Workstream F, Hypothesis F1: PCR Mean-Reversion Analysis.
    Expects df with 'nifty_close' and 'pcr_volume'
    """
    df = df_daily.copy()
    
    # 30-day rolling max/min for dynamic percentile 
    df['pcr_min'] = df['pcr_volume'].rolling(30).min()
    df['pcr_max'] = df['pcr_volume'].rolling(30).max()
    
    # Range bound PCR context (0 to 1)
    df['pcr_percentile'] = (df['pcr_volume'] - df['pcr_min']) / (df['pcr_max'] - df['pcr_min'] + 1e-9)
    
    # Triggers
    df['signal_peak_fear'] = df['pcr_percentile'] >= 0.80      # Rebound predictor
    df['signal_complacency'] = df['pcr_percentile'] <= 0.20    # Vulnerability predictor
    
    # Forward 5-day return horizon
    df['fwd_5d_ret'] = df['nifty_close'].pct_change(5).shift(-5)
    
    return df

def detect_strike_gravity(options_chain_df, current_spot):
    """
    Workstream F, Hypothesis F3: OI Concentration.
    Checks if active strikes have OI concentrations > 1.5x the mean limit.
    """
    mean_ce_oi = options_chain_df['ce_oi'].mean()
    mean_pe_oi = options_chain_df['pe_oi'].mean()
    
    # Out of the money pulls
    gravitational_resistances = options_chain_df[
        (options_chain_df['ce_oi'] > 1.5 * mean_ce_oi) & 
        (options_chain_df['strike'] > current_spot)
    ]
    
    gravitational_supports = options_chain_df[
        (options_chain_df['pe_oi'] > 1.5 * mean_pe_oi) & 
        (options_chain_df['strike'] < current_spot)
    ]
    
    return gravitational_resistances, gravitational_supports