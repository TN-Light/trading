import itertools
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CSV_PATH = Path('dataset/NIFTY 50_5minute.csv')
STARTING_CAPITAL = 15000.0

LOT_SIZE = 65  
BANK_NIFTY_LOT_SIZE = 30  # 2026 revised lot size for Tier 3 scaling

# TAX NOTE: Profits from intraday F&O trading are classified as non-speculative 
# business income and are taxed according to the trader's applicable income tax 
# slab rate (up to 30%). They do not qualify for lower capital gains tax exemptions.
STT_RATE_OPTIONS = 0.0015  # 0.15% on sell leg
BROKERAGE_FLAT = 40.0
SLIPPAGE_RATE_A = 0.003  # 0.3%
SLIPPAGE_RATE_B = 0.002  # 0.2%
NOMINAL_ATM_PREMIUM = 110.0  
DELTA_PROXY = 0.50  

def load_nifty_data(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['datetime'], errors='coerce')

    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    df = df.between_time('09:15', '15:30')

    # 5m base indicators
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr14_5m'] = df['tr'].rolling(14).mean()

    # 5m RSI
    delta_5m = df['close'].diff()
    gain_5m = delta_5m.clip(lower=0)
    loss_5m = -delta_5m.clip(upper=0)
    avg_gain_5m = gain_5m.rolling(2).mean()
    avg_loss_5m = loss_5m.rolling(2).mean()
    rs_5m = avg_gain_5m / avg_loss_5m
    df['rsi2_5m'] = 100 - (100 / (1 + rs_5m))

    # VWAP simulation
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    grouped = df.groupby(df.index.date)
    vwap_list = []
    for d, day_df in grouped:
        if ('volume' in day_df.columns) and (day_df['volume'].sum() > 0):       
            cum_vol = day_df['volume'].cumsum()
            cum_pv = (day_df['typical_price'] * day_df['volume']).cumsum()      
            vwap = cum_pv / cum_vol
        else:
            vwap = day_df['typical_price'].expanding().mean()
        vwap_list.append(vwap)
    df['vwap'] = pd.concat(vwap_list)

    # 15m synthesis
    df_15m = df.resample('15min', origin='start').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    df_15m['tr'] = np.maximum(df_15m['high'] - df_15m['low'], np.maximum(abs(df_15m['high'] - df_15m['close'].shift(1)), abs(df_15m['low'] - df_15m['close'].shift(1))))
    df_15m['atr14_15m'] = df_15m['tr'].rolling(14).mean()
    df_15m['ema200_15m'] = df_15m['close'].ewm(span=200, adjust=False).mean()   

    # 15m RSI
    delta_15m = df_15m['close'].diff()
    gain_15m = delta_15m.clip(lower=0)
    loss_15m = -delta_15m.clip(upper=0)
    avg_gain_15m = gain_15m.rolling(2).mean()
    avg_loss_15m = loss_15m.rolling(2).mean()
    rs_15m = avg_gain_15m / avg_loss_15m
    df_15m['rsi2_15m'] = 100 - (100 / (1 + rs_15m))

    return df, df_15m

def calculate_daily_structures(df):
    daily_df = df.resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    daily_df['pdh'] = daily_df['high'].shift(1)
    daily_df['pdl'] = daily_df['low'].shift(1)
    daily_df['pdc'] = daily_df['close'].shift(1)

    daily_df['pp'] = (daily_df['pdh'] + daily_df['pdl'] + daily_df['pdc']) / 3.0
    daily_df['r1'] = (2 * daily_df['pp']) - daily_df['pdl']
    daily_df['r2'] = daily_df['pp'] + (daily_df['pdh'] - daily_df['pdl'])       
    daily_df['s1'] = (2 * daily_df['pp']) - daily_df['pdh']
    daily_df['s2'] = daily_df['pp'] - (daily_df['pdh'] - daily_df['pdl'])       
    
    # NR7
    range_series = daily_df['high'] - daily_df['low']
    daily_df['nr7_prev'] = False
    for i in range(7, len(range_series)):
        current_range = range_series.iloc[i-1]
        is_nr7 = all(current_range < range_series.iloc[i-j] for j in range(2, 8))
        daily_df.iloc[i, daily_df.columns.get_loc('nr7_prev')] = is_nr7

    df_aug = df.copy()
    df_aug['date_only'] = df_aug.index.floor('D')
    daily_df = daily_df.rename_axis('date_only')
    df_aug = df_aug.join(daily_df[['pdh', 'pdl', 'pp', 'r1', 'r2', 's1', 's2', 'nr7_prev']], on='date_only')
    return df_aug

def compute_metrics(trades, daily_equity, daily_dates, initial_cap, final_cap): 
    if not trades or not daily_equity:
        return {'trades': 0, 'win_rate': 0, 'net_return': 0, 'cagr': 0, 'max_dd': 0, 'profit_factor': 0}

    df_t = pd.DataFrame(trades)
    wins = int((df_t['pnl'] > 0).sum())
    total = len(df_t)

    win_rate = (wins / total) * 100.0 if total > 0 else 0.0
    loss_rate = 100.0 - win_rate

    gross_profit = df_t.loc[df_t['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(df_t.loc[df_t['pnl'] < 0, 'pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg_win = df_t.loc[df_t['pnl'] > 0, 'pnl'].mean() if wins > 0 else 0.0      
    avg_loss = df_t.loc[df_t['pnl'] < 0, 'pnl'].mean() if (total - wins) > 0 else 0.0

    net_return = ((final_cap - initial_cap) / initial_cap) * 100.0

    eq = pd.Series(daily_equity)
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = float(dd.min() * 100)

    tdays = len(daily_equity)
    years = max(tdays / 252.0, 1e-9)
    cagr = float(((final_cap / initial_cap) ** (1.0 / years) - 1.0) * 100.0) if final_cap > 0 else -100.0

    eq_series = pd.Series(daily_equity, index=daily_dates).sort_index().pct_change().fillna(0.0)
    std_d = float(eq_series.std(ddof=0))
    down_std = float(eq_series[eq_series < 0].std(ddof=0)) if len(eq_series[eq_series < 0]) > 0 else 0.0

    sharpe = (float(eq_series.mean()) / std_d) * np.sqrt(252) if std_d > 0 else 0.0
    sortino = (float(eq_series.mean()) / down_std) * np.sqrt(252) if down_std > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'trades': total,
        'win_rate': round(win_rate, 2),
        'loss_rate': round(loss_rate, 2),
        'profit_factor': round(pf, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'net_return': round(net_return, 2),
        'cagr': round(cagr, 2),
        'max_dd': round(max_dd, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'final_cap': round(final_cap, 2),
    }

def run_nexus_engine(df, df_15m, max_days=None):
    if max_days:
        uniq = pd.Series(df.index.date).drop_duplicates()
        keep = set(uniq.iloc[-max_days:].tolist())
        df = df[pd.Index(df.index.date).isin(keep)]

    days = [d for d, _ in df.groupby(df.index.date)]
    capital = STARTING_CAPITAL

    trades, daily_equity, daily_dates = [], [], []
    current_consecutive_losses, pause_week_id = 0, None

    for current_date in days:
        day_df_5m = df[df.index.date == current_date]
        if len(day_df_5m) < 15:
            daily_equity.append(capital)
            daily_dates.append(pd.to_datetime(current_date))
            continue

        week_id = current_date.isocalendar()[1]

        # IRON RULE 1: Two-Loss Pause & 50% Cash Rule
        if pause_week_id == week_id or (NOMINAL_ATM_PREMIUM * LOT_SIZE > STARTING_CAPITAL * 0.5):
            daily_equity.append(capital)
            daily_dates.append(pd.to_datetime(current_date))
            continue
        elif pause_week_id != week_id:
            current_consecutive_losses = 0
            pause_week_id = None

        ib_high, ib_low = None, None
        active_trade = None
        playbook_b_rsi_long_trigger = playbook_b_rsi_short_trigger = False      

        for i, (ts, row_5m) in enumerate(day_df_5m.iterrows()):
            hm = ts.hour * 100 + ts.minute

            if hm == 930:
                morning_slice = day_df_5m.loc[day_df_5m.index.time < pd.to_datetime('09:30').time()]
                if len(morning_slice) > 0:
                    ib_high = float(morning_slice['high'].max())
                    ib_low = float(morning_slice['low'].min())

            if active_trade is None:
                if not ((930 <= hm <= 1045) or (1400 <= hm <= 1445)):
                    continue

                is_15m_close = (hm % 15 == 0)
                ts_cand = ts - pd.Timedelta(minutes=15)
                row_15m = df_15m.loc[ts_cand] if ts_cand in df_15m.index else None

                sig_a = sig_b = None

                # PLAYBOOK A Eval
                if is_15m_close and row_15m is not None and ib_high and ib_low: 
                    v_val = row_5m['vwap']
                    c1_L, c1_S = row_15m['close'] > v_val, row_15m['close'] < v_val
                    atr15 = row_15m['atr14_15m']
                    c2 = pd.notna(atr15) and (row_15m['high'] - row_15m['low']) > (1.2 * atr15)
                    c3_L, c3_S = row_15m['close'] > ib_high, row_15m['close'] < ib_low
                    r15 = row_15m['rsi2_15m']
                    c4_L, c4_S = pd.notna(r15) and r15 < 80, pd.notna(r15) and r15 > 20

                    if c1_L and c2 and c3_L and c4_L: sig_a = 'LONG'
                    elif c1_S and c2 and c3_S and c4_S: sig_a = 'SHORT'

                # PLAYBOOK B Eval
                r5 = row_5m['rsi2_5m']
                if pd.notna(r5):
                    if r5 < 10: playbook_b_rsi_long_trigger = True
                    if r5 > 90: playbook_b_rsi_short_trigger = True

                if (playbook_b_rsi_long_trigger or playbook_b_rsi_short_trigger) and row_15m is not None:
                    ema200 = row_15m['ema200_15m']
                    c_grn, c_red = row_5m['close'] > row_5m['open'], row_5m['close'] < row_5m['open']

                    structs = [x for x in [row_5m['pdh'], row_5m['pdl'], row_5m['pp'], row_5m['r1'], row_5m['r2'], row_5m['s1'], row_5m['s2'], row_5m['vwap'], ema200] if pd.notna(x)]
                    valid_struct = min([abs(row_5m['close'] - L) for L in structs]) <= 15.0 if structs else False

                    if playbook_b_rsi_long_trigger and (row_15m['close'] > ema200 and row_15m['close'] > row_5m['vwap']) and c_grn and valid_struct:
                        sig_b, playbook_b_rsi_long_trigger = 'LONG', False      
                    elif playbook_b_rsi_short_trigger and (row_15m['close'] < ema200 and row_15m['close'] < row_5m['vwap']) and c_red and valid_struct:
                        sig_b, playbook_b_rsi_short_trigger = 'SHORT', False    

                if sig_a or sig_b:
                    slip = SLIPPAGE_RATE_A if sig_a else SLIPPAGE_RATE_B        
                    entry_pr = NOMINAL_ATM_PREMIUM * (1 + slip)
                    active_trade = {
                        'playbook': 'A' if sig_a else 'B', 'dir': sig_a or sig_b,
                        'nifty_entry_px': row_5m['close'], 'premium_entry': entry_pr,
                        'sl_stage': 1, 'stop_px': entry_pr * 0.8, 'trail_nifty_base': row_5m['close']
                    }

            else:
                dm = 1 if active_trade['dir'] == 'LONG' else -1
                sim_premium = active_trade['premium_entry'] + ((row_5m['close'] - active_trade['nifty_entry_px']) * dm * DELTA_PROXY)

                sim_premium_low = active_trade['premium_entry'] + ((row_5m['low'] - active_trade['nifty_entry_px']) * dm * DELTA_PROXY)
                if dm == -1:
                    sim_premium_low = active_trade['premium_entry'] + ((row_5m['high'] - active_trade['nifty_entry_px']) * dm * DELTA_PROXY)

                exit_reason, exit_premium = None, None

                if sim_premium_low <= active_trade['stop_px']:
                    exit_reason, exit_premium = 'STOP', active_trade['stop_px'] 
                elif hm >= 1458:
                    # 2:58 PM Hard Time Stop prevents brokers from initiating auto
                    # square-off (~15:15 - 15:20) and avoids the ₹50+GST penalty fee.
                    exit_reason, exit_premium = 'TIME_STOP', sim_premium        

                if exit_reason:
                    net_pnl = ((exit_premium - active_trade['premium_entry']) * LOT_SIZE) - (exit_premium * LOT_SIZE * STT_RATE_OPTIONS) - BROKERAGE_FLAT       
                    trades.append({'pnl': net_pnl})
                    capital += net_pnl

                    current_consecutive_losses = current_consecutive_losses + 1 if net_pnl < 0 else 0
                    if current_consecutive_losses >= 2: pause_week_id = week_id 
                    active_trade = None
                    break
                else:
                    dist = active_trade['premium_entry'] * 0.20
                    if active_trade['sl_stage'] == 1 and sim_premium >= (active_trade['premium_entry'] + dist):
                        active_trade['sl_stage'] = 2
                        active_trade['stop_px'] = active_trade['premium_entry'] 

                    if active_trade['sl_stage'] >= 2:
                        active_trade['sl_stage'] = 3
                        if (dm == 1 and row_5m['close'] > active_trade['trail_nifty_base']) or (dm == -1 and row_5m['close'] < active_trade['trail_nifty_base']):
                            active_trade['trail_nifty_base'] = row_5m['close']  

                        if pd.notna(row_5m['atr14_5m']):
                            new_stop = active_trade['premium_entry'] + ((active_trade['trail_nifty_base'] - active_trade['nifty_entry_px']) * dm * DELTA_PROXY) - (1.5 * row_5m['atr14_5m'] * DELTA_PROXY)
                            if new_stop > active_trade['stop_px']: active_trade['stop_px'] = new_stop

        daily_equity.append(capital)
        daily_dates.append(pd.to_datetime(current_date))

    return compute_metrics(trades, daily_equity, daily_dates, STARTING_CAPITAL, capital)

def run_walkforward(df, df_15m):
    days = pd.Series(df.index.date).drop_duplicates().tolist()
    steps = [days[i:i+252] for i in range(0, len(days), 252)]

    folds = {}
    for i, fold_days in enumerate(steps, 1):
        if len(fold_days) < 120: continue
        f_df = df[pd.Index(df.index.date).isin(fold_days)]
        m = run_nexus_engine(f_df, df_15m)
        m['start_date'] = str(fold_days[0])
        m['end_date'] = str(fold_days[-1])
        folds[f"Fold_{i}"] = m
    return folds

if __name__ == '__main__':
    if not CSV_PATH.exists():
        logger.error(f"Cannot find {CSV_PATH}")
    else:
        df, df_15m = load_nifty_data(CSV_PATH)
        df = calculate_daily_structures(df)

        results = {}

        logger.info("=== Running NEXUS 60-Day Horizon ===")
        results['60_days'] = run_nexus_engine(df, df_15m, 60)

        logger.info("=== Running NEXUS 120-Day Horizon ===")
        results['120_days'] = run_nexus_engine(df, df_15m, 120)

        logger.info("=== Running NEXUS Full 8-Year Horizon ===")
        results['8_years'] = run_nexus_engine(df, df_15m, None)

        logger.info("=== Running NEXUS Walk-Forward (1-Year Folds) ===")        
        results['Walk_Forward'] = run_walkforward(df, df_15m)

        out_file = Path('reports/nexus/nexus_multi_horizon.json')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(json.dumps(results, indent=2))
        logger.info(f"Results written directly to: {out_file}")
