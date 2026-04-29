import pandas as pd
import numpy as np
from pathlib import Path

def load_and_prep_data(symbol, timeframe='15min'):
    # Load data from the dataset folder
    path = Path(f'dataset/{symbol}_5minute.csv')
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    
    if timeframe != '5min':
        # Up-sample to wider timeframe to defeat the STT drag
        df = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
    df['date'] = df.index.date
    df['time'] = df.index.time
    
    return df

def generate_alpha_features(df):
    # 1. Initial Balance (IB) - First 60 minutes range (09:15 - 10:15)
    # Get bars that are between 9:15 and 10:15
    mask_915_1015 = ((df.index.hour == 9) & (df.index.minute >= 15)) | ((df.index.hour == 10) & (df.index.minute <= 15))
    ib_highs = df[mask_915_1015]
    ib_high = ib_highs.groupby('date')['high'].max().reset_index(name='ib_high')
    ib_low = ib_highs.groupby('date')['low'].min().reset_index(name='ib_low')
    
    df = df.reset_index().merge(ib_high, on='date', how='left').merge(ib_low, on='date', how='left').set_index('datetime')
    
    # 2. VWAP & StdDev Bands
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume'] = df['volume'].replace(0, 1) 
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    
    price_sq_vol = (df['typical_price'] ** 2) * df['volume']
    df['cum_pv2'] = price_sq_vol.groupby(df['date']).cumsum()
    vwap_var = np.maximum(0, (df['cum_pv2'] / df['cum_vol']) - (df['vwap'] ** 2))
    df['vwap_std'] = np.sqrt(vwap_var)
    
    # 3. ATR
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # 4. Trend Quality (EMA Ribbon)
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema34'] = df['close'].ewm(span=34, adjust=False).mean()
    
    df['trend_up'] = (df['ema8'] > df['ema21']) & (df['ema21'] > df['ema34'])
    df['trend_down'] = (df['ema8'] < df['ema21']) & (df['ema21'] < df['ema34'])
    
    return df.dropna()

def test_ib_breakout_strategy(df, tax_drag=15.0):
    """
    Experimental Hybrid Strategy targeting > 60% WR.
    Rules: Wait for deeply confirmed trend + breakout of morning IB + VWAP pullback.
    """
    trades = []
    in_trade = False
    
    time_arr = df['time'].values
    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    ib_h_arr = df['ib_high'].values
    ib_l_arr = df['ib_low'].values
    vwap_arr = df['vwap'].values
    atr_arr = df['atr'].values
    trend_up_arr = df['trend_up'].values
    trend_dn_arr = df['trend_down'].values
    
    entry_price, sl_px, tp_px, trade_dir, bars_held = 0, 0, 0, 0, 0
    
    # Very restrictive RR to ensure massive hits when right
    SL_ATR = 1.0  
    TP_ATR = 1.5  # Win 1.5R to absorb the tax. Since we hunt exact spots, WR should rise.
    MAX_HOLD_BARS = 24 # End of day or max 24 bars (6 hours on 15m)
    
    for i in range(14, len(df)-1):
        if not in_trade:
            # 1. TIME FILTER: Avoid choppy morning. Only take trades 10:30 onwards when IB is locked in.
            if time_arr[i].hour < 10 or (time_arr[i].hour == 10 and time_arr[i].minute < 30):
                continue
            if time_arr[i].hour >= 14 and time_arr[i].minute > 30: 
                continue # Do not open extremely late trades
                
            # 2. LONG ENTRY (IB Breakout confirmed + Pullback)
            if trend_up_arr[i] and (close_arr[i] > ib_h_arr[i]) and (close_arr[i] < vwap_arr[i] + (1.0 * atr_arr[i])):
                in_trade = True; trade_dir = 1
                entry_price = open_arr[i+1] # Entering next bar
                sl_px = entry_price - (SL_ATR * atr_arr[i])
                tp_px = entry_price + (TP_ATR * atr_arr[i])
                bars_held = 0
                
            # 3. SHORT ENTRY
            elif trend_dn_arr[i] and (close_arr[i] < ib_l_arr[i]) and (close_arr[i] > vwap_arr[i] - (1.0 * atr_arr[i])):
                in_trade = True; trade_dir = -1
                entry_price = open_arr[i+1]
                sl_px = entry_price + (SL_ATR * atr_arr[i])
                tp_px = entry_price - (TP_ATR * atr_arr[i])
                bars_held = 0
                
        else:
            bars_held += 1
            exit_px = None
            
            # EOD EXIT
            if time_arr[i].hour >= 15 and time_arr[i].minute >= 10:
                exit_px = close_arr[i]
            # TIME_STOP
            elif bars_held >= MAX_HOLD_BARS:
                exit_px = close_arr[i]
            elif trade_dir == 1:
                if low_arr[i] <= sl_px: exit_px = sl_px
                elif high_arr[i] >= tp_px: exit_px = tp_px
            else:
                if high_arr[i] >= sl_px: exit_px = sl_px
                elif low_arr[i] <= tp_px: exit_px = tp_px
                
            if exit_px is not None:
                pnl = (exit_px - entry_price) if trade_dir == 1 else (entry_price - exit_px)
                trades.append({'pnl': pnl, 'bars': bars_held, 'dir': trade_dir})
                in_trade = False
                
    t_df = pd.DataFrame(trades)
    if len(t_df) == 0: 
        print("No trades triggered.")
        return t_df
    
    t_df['gross_pnl'] = t_df['pnl']
    t_df['net_pnl'] = t_df['pnl'] - tax_drag # Apply the reality tax
    
    return t_df

def print_metrics(t_df, title, lot_size=25, years=11):
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    if len(t_df) == 0:
        print("No trades taken.")
        return
        
    wins = t_df[t_df['net_pnl'] > 0]
    losses = t_df[t_df['net_pnl'] <= 0]
    
    wr = len(wins) / len(t_df) * 100
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['net_pnl'].mean()) if len(losses) > 0 else 0
    pf = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())) if len(losses) > 0 and abs(losses['net_pnl'].sum()) > 0 else 99
    
    returns = t_df['net_pnl'] * lot_size
    sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    
    print(f"Trades Count: {len(t_df)} (Avg {int(len(t_df)/(years*12))}/month)")
    print(f"Win Rate:     {wr:.2f}%")
    print(f"Avg Net Win:  {avg_win:.2f} pts")
    print(f"Avg Net Loss: {avg_loss:.2f} pts")
    print(f"Profit Factr: {pf:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Total Net PnL (Rs): Rs {returns.sum():,.2f}")

if __name__ == '__main__':
    print("Phase 1 & 2 R&D: Mining Structural Edges (15m Timeframe)")
    
    df_nifty = load_and_prep_data('NIFTY 50', '15min')
    df_nifty = generate_alpha_features(df_nifty)
    res_nifty = test_ib_breakout_strategy(df_nifty, tax_drag=15.0)
    print_metrics(res_nifty, "NIFTY 50 - IB Breakout + VWAP (15min) [Tax: 15pt]", lot_size=25, years=11)

    df_bank = load_and_prep_data('NIFTY BANK', '15min')
    df_bank = generate_alpha_features(df_bank)
    res_bank = test_ib_breakout_strategy(df_bank, tax_drag=35.0)
    print_metrics(res_bank, "NIFTY BANK - IB Breakout + VWAP (15min) [Tax: 35pt]", lot_size=15, years=11)
