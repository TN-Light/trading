import pandas as pd
import numpy as np
from pathlib import Path
import os

INDEX_CONFIG = {
    'NIFTY_BANK': {
        'file': 'NIFTY_BANK_5minute_anchored.csv', 
        'anchors': ['hdfc_vwap_div', 'icici_vwap_div']
    },
    'NIFTY_50': {
        'file': 'NIFTY_50_5minute_anchored.csv', 
        'anchors': ['reliance_vwap_div', 'infy_vwap_div']
    },
    'FINNIFTY': {
        'file': 'FINNIFTY_5minute_anchored.csv', 
        'anchors': ['hdfc_vwap_div', 'bajaj_vwap_div']
    }
}

def _merge_institutional_options_context(m_df, index_name):
    # 1. Load the locally harvested options CSV
    try:
        live_opt = pd.read_csv("dataset/live_options_context.csv")
    except FileNotFoundError:
        # Failsafe: Add empty columns while data builds up
        m_df['straddle_premium'] = np.nan
        m_df['pcr'] = np.nan
        return m_df 
        
    # 2. Filter for the specific index
    if 'Index' in live_opt.columns:
        opt_df = live_opt[live_opt['Index'] == index_name].copy()
    else:
        opt_df = live_opt[live_opt['index'] == index_name].copy()
    if opt_df.empty:
        m_df['straddle_premium'] = np.nan
        m_df['pcr'] = np.nan
        return m_df

    if 'timestamp' in opt_df.columns:
        opt_df['timestamp'] = pd.to_datetime(opt_df['timestamp'])
    elif 'Datetime' in opt_df.columns:
        opt_df['timestamp'] = pd.to_datetime(opt_df['Datetime'])
        
    # 3. Sort both dataframes to ensure chronological integrity
    m_df['datetime'] = pd.to_datetime(m_df['datetime'])
    m_df = m_df.sort_values('datetime')
    opt_df = opt_df.sort_values('timestamp')

    # Pre-calculate Straddle and PCR if not present
    if 'CE_LTP' in opt_df.columns and 'PE_LTP' in opt_df.columns:
        opt_df['straddle_premium'] = opt_df['CE_LTP'] + opt_df['PE_LTP']
    if 'CE_OI' in opt_df.columns and 'PE_OI' in opt_df.columns:
        opt_df['pcr'] = np.where(opt_df['CE_OI'] > 0, opt_df['PE_OI'] / opt_df['CE_OI'], 0)

    # 4. As-Of Merge: Maps the most recently published PCR/Straddle
    # to the exact moment the spot signal triggered (Look-ahead bias eliminated)
    merged_df = pd.merge_asof(
        m_df, 
        opt_df[['timestamp', 'straddle_premium', 'pcr']],
        left_on='datetime',
        right_on='timestamp',
        direction='backward', # ALWAYS backward to prevent AI cheating
        tolerance=pd.Timedelta(minutes=5) # Ensure data isn't stale
    )
    
    if 'timestamp' in merged_df.columns:
        merged_df = merged_df.drop(columns=['timestamp'])
        
    return merged_df

def generate_nmb_signals(symbol, capital=15000, friction_points=5.0):
    """
    Tri-Index Liquidity Trap Reversal (LTR) Engine
    Strictly Time-Based Volatility Expansion (13:30 - 14:30)
    Optimized for Option Buying with Causal Institutional VWAP Anchors
    """
    config = INDEX_CONFIG[symbol]
    path = Path(config['file'])
    
    if not path.exists():
        print(f"Dataset missing for {symbol} at {path}. Run Angel Builder first.")
        return None

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = 'date' if 'date' in df.columns else 'timestamp'
    if date_col not in df.columns and 'datetime' in df.columns:
        date_col = 'datetime'
        
    df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')

    df['time'] = df.index.time
    df['date_only'] = pd.to_datetime(df.index.date)
    df['vol'] = df['volume'].replace(0, 1)

    df['vol_20ma'] = df['vol'].rolling(20).mean()

    # 1. Daily Aggregation for ATR and Gap
    daily_df = df.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
    }).dropna()

    daily_df['prev_close'] = daily_df['close'].shift(1)
    daily_df['tr'] = np.maximum(daily_df['high'] - daily_df['low'],
                     np.maximum(abs(daily_df['high'] - daily_df['prev_close']), 
                                abs(daily_df['low'] - daily_df['prev_close']))) 
    daily_df['atr_14'] = daily_df['tr'].rolling(14).mean()
    daily_df['gap_pct'] = ((daily_df['open'] - daily_df['prev_close']) / daily_df['prev_close']) * 100

    df['atr_14'] = df['date_only'].map(daily_df['atr_14'])
    df['gap_pct'] = df['date_only'].map(daily_df['gap_pct'])

    # 2. Intraday VWAP (Trend Context)
    df['typ_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_price'] = df['typ_price'] * df['vol']
    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['vol'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']

    # 3. Compression Box Build (11:30 to 13:30)
    box_highs, box_lows, box_widths, valid_boxes = {}, {}, {}, {}

    for date, group in df.groupby('date_only'):
        window = df[(df['date_only'] == date) &
                    (df['time'] >= pd.to_datetime('11:30').time()) &
                    (df['time'] <= pd.to_datetime('13:25').time())]

        if len(window) < 10:
            valid_boxes[date] = False
            continue

        box_high = window['high'].max()
        box_low = window['low'].min()
        box_width = box_high - box_low

        if box_width > 0:
            box_highs[date] = box_high
            box_lows[date] = box_low
            box_widths[date] = box_width
            valid_boxes[date] = True
        else:
            valid_boxes[date] = False

    df['box_high'] = df['date_only'].map(box_highs)
    df['box_low'] = df['date_only'].map(box_lows)
    df['box_valid'] = df['date_only'].map(valid_boxes).fillna(False)

    df.dropna(subset=['high', 'low', 'close', 'open', 'box_high', 'box_low', 'vwap', 'atr_14'], inplace=True)

    # Pre-extract causal anchor arrays
    anchor_1_arr = df[config['anchors'][0].lower()].fillna(0).values
    anchor_2_arr = df[config['anchors'][1].lower()].fillna(0).values

    # Pre-extract main arrays
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    vol_arr = df['vol'].values
    vol_20ma_arr = df['vol_20ma'].values
    time_arr = df['time'].values
    box_high_arr = df['box_high'].values
    box_low_arr = df['box_low'].values
    box_valid_arr = df['box_valid'].values
    atr_arr = df['atr_14'].values
    gap_arr = df['gap_pct'].values
    vwap_arr = df['vwap'].values

    trades = []
    ml_dataset = []
    universal_paper_ledger = []
    in_trade = False
    trade_dir = 0
    trade_start_idx = 0
    entry_px, sl_px, tp_px = 0, 0, 0
    ml_data = {}

    for i in range(20, len(df) - 1):
        if not in_trade:
            val_time = time_arr[i]
            trade_window = (val_time.hour == 13 and val_time.minute >= 30) or \
                           (val_time.hour == 14 and val_time.minute <= 30)

            if trade_window and box_valid_arr[i]:
                # SHORT TRAP (Retail bought the high breakout, got trapped)
                if high_arr[i] > box_high_arr[i] and close_arr[i] < box_high_arr[i] and close_arr[i] < open_arr[i]:
                    in_trade = True
                    trade_dir = -1  # Buy Put
                    entry_px = close_arr[i]
                    sl_px = high_arr[i]
                    risk = sl_px - entry_px if sl_px > entry_px else 10
                    tp_px = entry_px - (1.5 * risk)
                    trade_start_idx = i

                    ml_data = {
                        'datetime': df.index[i],
                        'box_width': box_widths.get(df['date_only'].iloc[i], 0),
                        'dir': -1,
                        'gap_pct': gap_arr[i],
                        'atr_ratio': box_widths.get(df['date_only'].iloc[i], 0) / (atr_arr[i] + 1e-9),
                        'vwap_dist_pct': ((close_arr[i] - vwap_arr[i]) / vwap_arr[i]) * 100,
                        'anchor_1_vwap_div': anchor_1_arr[i],
                        'anchor_2_vwap_div': anchor_2_arr[i]
                    }

                    # Universal Asset Approximations (For Paper Testing, SHORT Signal)
                    future_entry = entry_px - 50
                    universal_paper_ledger.append({
                        'datetime': df.index[i],
                        'index_symbol': symbol,
                        'signal': 'SHORT',
                        'spot_entry': round(entry_px, 2),
                        'spot_stoploss': round(sl_px, 2),
                        'spot_target': round(tp_px, 2),
                        'instrument_futures_entry': round(future_entry, 2),
                        'instrument_opt_buy_PE': 150,
                        'instrument_opt_sell_CE': 150,
                        'box_width': box_widths.get(df['date_only'].iloc[i], 0),
                        'gap_pct': gap_arr[i],
                        'vwap_dist_pct': ((close_arr[i] - vwap_arr[i]) / vwap_arr[i]) * 100,
                        'anchor_1_vwap_div': anchor_1_arr[i],
                        'anchor_2_vwap_div': anchor_2_arr[i]
                    })
                    continue

                # LONG TRAP (Retail sold the low breakdown, got trapped)
                elif low_arr[i] < box_low_arr[i] and close_arr[i] > box_low_arr[i] and close_arr[i] > open_arr[i]:
                    in_trade = True
                    trade_dir = 1   # Buy Call
                    entry_px = close_arr[i]
                    sl_px = low_arr[i]
                    risk = entry_px - sl_px if entry_px > sl_px else 10
                    tp_px = entry_px + (1.5 * risk)
                    trade_start_idx = i

                    ml_data = {
                        'datetime': df.index[i],
                        'box_width': box_widths.get(df['date_only'].iloc[i], 0),
                        'dir': 1,
                        'gap_pct': gap_arr[i],
                        'atr_ratio': box_widths.get(df['date_only'].iloc[i], 0) / (atr_arr[i] + 1e-9),
                        'vwap_dist_pct': ((close_arr[i] - vwap_arr[i]) / vwap_arr[i]) * 100,
                        'anchor_1_vwap_div': anchor_1_arr[i],
                        'anchor_2_vwap_div': anchor_2_arr[i]
                    }

                    # Universal Asset Approximations (For Paper Testing, LONG Signal)
                    future_entry = entry_px + 50
                    universal_paper_ledger.append({
                        'datetime': df.index[i],
                        'index_symbol': symbol,
                        'signal': 'LONG',
                        'spot_entry': round(entry_px, 2),
                        'spot_stoploss': round(sl_px, 2),
                        'spot_target': round(tp_px, 2),
                        'instrument_futures_entry': round(future_entry, 2),
                        'instrument_opt_buy_CE': 150,
                        'instrument_opt_sell_PE': 150,
                        'box_width': box_widths.get(df['date_only'].iloc[i], 0),
                        'gap_pct': gap_arr[i],
                        'vwap_dist_pct': ((close_arr[i] - vwap_arr[i]) / vwap_arr[i]) * 100,
                        'anchor_1_vwap_div': anchor_1_arr[i],
                        'anchor_2_vwap_div': anchor_2_arr[i]
                    })
                    continue

        else: # Manage open trade
            bars_held = i - trade_start_idx
            exit_px = None
            hit_target = 0
            val_time = time_arr[i]

            if trade_dir == 1: # LONG
                if high_arr[i] >= tp_px:
                    exit_px = tp_px
                    hit_target = 1
                elif low_arr[i] <= sl_px: exit_px = sl_px
                elif bars_held >= 4: exit_px = close_arr[i]
                elif val_time.hour >= 15 and val_time.minute >= 10: exit_px = close_arr[i]

                if exit_px is not None:
                    gross_pts = exit_px - entry_px
                    net_pts = gross_pts - friction_points
                    trades.append({'gross_pts': gross_pts, 'net_pts': net_pts, 'bars': bars_held, 'dir': 1})
                    
                    ml_data_copy = ml_data.copy()
                    ml_data_copy['target_hit'] = hit_target
                    ml_data_copy['net_pts'] = net_pts
                    ml_dataset.append(ml_data_copy)
                    in_trade = False

            else: # SHORT
                if low_arr[i] <= tp_px:
                    exit_px = tp_px
                    hit_target = 1
                elif high_arr[i] >= sl_px: exit_px = sl_px
                elif bars_held >= 4: exit_px = close_arr[i]
                elif val_time.hour >= 15 and val_time.minute >= 10: exit_px = close_arr[i]

                if exit_px is not None:
                    gross_pts = entry_px - exit_px
                    net_pts = gross_pts - friction_points
                    trades.append({'gross_pts': gross_pts, 'net_pts': net_pts, 'bars': bars_held, 'dir': -1})

                    ml_data_copy = ml_data.copy()
                    ml_data_copy['target_hit'] = hit_target
                    ml_data_copy['net_pts'] = net_pts
                    ml_dataset.append(ml_data_copy)
                    in_trade = False

    t_df = pd.DataFrame(trades)
    m_df = pd.DataFrame(ml_dataset)

    if t_df.empty or m_df.empty:
        print(f"No trades generated for {symbol}.")
        return

    # Merge Institutional Order Flow Data (Options) seamlessly
    m_df = _merge_institutional_options_context(m_df, symbol)

    os.makedirs('dataset', exist_ok=True)
    out_csv = f'dataset/{symbol}_ml_training_data.csv'
    m_df.to_csv(out_csv, index=False)
    print(f"Saved {len(m_df)} raw trades to {out_csv} for ML pipeline.")

    # Universal Paper Ledger Export
    if universal_paper_ledger:
        os.makedirs("dataset", exist_ok=True)
        paper_df = pd.DataFrame(universal_paper_ledger)
        paper_df.to_csv(f"dataset/universal_ledger_{symbol}.csv", index=False)
        print(f"[{symbol}] Created Universal Ledger: {len(paper_df)} Multi-Asset Entries.")

if __name__ == '__main__':
    for symbol in INDEX_CONFIG.keys():
        print(f"\nProcessing LTR Signals for: {symbol}")
        generate_nmb_signals(symbol=symbol, capital=15000, friction_points=5.0)

