"""Deep analysis of losing trades across all 3 indices (full history).
Uses _run_backtest_on_slice which handles all internal setup."""
import sys, os
sys.path.insert(0, '.')

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from prometheus.main import Prometheus
import pandas as pd


def run_and_extract(symbol, days):
    """Use Prometheus._run_backtest_on_slice to run backtest and get engine.trades."""
    p = Prometheus()

    data_daily = p.data.fetch_historical(symbol, days=days, interval='1d')
    if data_daily is None or len(data_daily) < 100:
        print(f"  ERROR: Insufficient data for {symbol}")
        return pd.DataFrame()

    result, engine = p._run_backtest_on_slice(
        data_slice=data_daily,
        symbol=symbol,
        strategy_name="default",
        param_overrides={},
        verbose=False,
        parrondo=True,
        dd_throttle=True,
    )

    trades_data = []
    for t in engine.trades:
        entry_dt = pd.to_datetime(t.entry_time)
        exit_dt = pd.to_datetime(t.exit_time)
        hold_min = (exit_dt - entry_dt).total_seconds() / 60

        trades_data.append({
            'entry_time': str(t.entry_time),
            'exit_time': str(t.exit_time),
            'direction': t.direction,
            'net_pnl': t.net_pnl,
            'gross_pnl': t.gross_pnl,
            'costs': t.costs,
            'exit_reason': getattr(t, 'exit_reason', 'unknown'),
            'regime_at_entry': getattr(t, 'regime_at_entry', 'unknown'),
            'bull_score': getattr(t, 'bull_score', 0),
            'bear_score': getattr(t, 'bear_score', 0),
            'atr_at_entry': getattr(t, 'atr_at_entry', 0),
            'strategy': getattr(t, 'strategy', ''),
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'hold_minutes': hold_min,
            'signal_liqsweep': getattr(t, 'signal_liqsweep', False),
            'signal_fvg': getattr(t, 'signal_fvg', False),
            'signal_vp': getattr(t, 'signal_vp', False),
            'signal_ote': getattr(t, 'signal_ote', False),
            'signal_rsi_div': getattr(t, 'signal_rsi_div', False),
            'signal_vol_surge': getattr(t, 'signal_vol_surge', False),
            'signal_vol_confirm': getattr(t, 'signal_vol_confirm', False),
            'signal_vwap': getattr(t, 'signal_vwap', False),
            'signal_bias': getattr(t, 'signal_bias', False),
            'entry_type': getattr(t, 'entry_type', 'immediate'),
        })

    return pd.DataFrame(trades_data)


def analyze(df, symbol):
    df['win'] = df['net_pnl'] > 0
    df['year'] = pd.to_datetime(df['entry_time']).dt.year
    losers = df[~df.win]
    winners = df[df.win]

    print(f"\n{'='*80}")
    print(f"  DEEP LOSS ANALYSIS: {symbol} | {len(df)} trades | {len(winners)} W / {len(losers)} L")
    print(f"{'='*80}")

    # 1. EXIT REASON
    print(f"\n--- 1. EXIT REASON BREAKDOWN ---")
    for reason in sorted(df['exit_reason'].unique()):
        sub = df[df.exit_reason == reason]
        w = sub[sub.win]
        l = sub[~sub.win]
        avg_l = l.net_pnl.mean() if len(l) > 0 else 0
        avg_w = w.net_pnl.mean() if len(w) > 0 else 0
        print(f"  {reason:15s}: {len(sub):3d} trades | WR={len(w)/len(sub)*100:5.1f}% | "
              f"AvgWin={avg_w:+7.0f} | AvgLoss={avg_l:+7.0f} | TotalPnL={sub.net_pnl.sum():+8.0f}")

    # 2. REGIME AT ENTRY
    print(f"\n--- 2. REGIME AT ENTRY ---")
    for regime in sorted(df['regime_at_entry'].unique()):
        sub = df[df.regime_at_entry == regime]
        w = sub[sub.win]
        l = sub[~sub.win]
        avg_l = l.net_pnl.mean() if len(l) > 0 else 0
        print(f"  {regime:14s}: {len(sub):3d} trades | WR={len(w)/len(sub)*100:5.1f}% | "
              f"Losers={len(l):3d} | LossAvg={avg_l:+7.0f} | TotalPnL={sub.net_pnl.sum():+8.0f}")

    # 3. YEAR-BY-YEAR
    print(f"\n--- 3. YEAR-BY-YEAR ---")
    for year in sorted(df['year'].unique()):
        sub = df[df.year == year]
        w = sub[sub.win]
        wr = len(w)/len(sub)*100 if len(sub) > 0 else 0
        print(f"  {year}: {len(sub):3d} trades | WR={wr:5.1f}% | PnL={sub.net_pnl.sum():+8.0f} | AvgPnL={sub.net_pnl.mean():+7.0f}")

    # 4. SIGNAL ANALYSIS
    print(f"\n--- 4. SIGNAL PRESENCE (WINNERS vs LOSERS) ---")
    signal_cols = [c for c in df.columns if c.startswith('signal_')]
    for sig in signal_cols:
        present = df[df[sig] == True]
        absent = df[df[sig] == False]
        if len(present) > 5:
            wr_p = present.win.mean() * 100
            wr_a = absent.win.mean() * 100 if len(absent) > 0 else 0
            edge = wr_p - wr_a
            print(f"  {sig:20s}: Present={len(present):3d} WR={wr_p:5.1f}% | "
                  f"Absent={len(absent):3d} WR={wr_a:5.1f}% | Edge={edge:+5.1f}%")
        else:
            print(f"  {sig:20s}: Only {len(present)} trades -- too few")

    # 5. DIRECTION
    print(f"\n--- 5. DIRECTION ---")
    for d in df['direction'].unique():
        sub = df[df.direction == d]
        w = sub[sub.win]
        print(f"  {d:8s}: {len(sub):3d} trades | WR={len(w)/len(sub)*100:5.1f}% | "
              f"PnL={sub.net_pnl.sum():+8.0f}")

    # 6. HOLD DURATION
    print(f"\n--- 6. HOLD DURATION ---")
    if len(winners) > 0:
        print(f"  Winners avg: {winners.hold_minutes.mean():7.0f} min ({winners.hold_minutes.mean()/1440:.1f} days)")
    if len(losers) > 0:
        print(f"  Losers avg:  {losers.hold_minutes.mean():7.0f} min ({losers.hold_minutes.mean()/1440:.1f} days)")

    # 7. SCORE AT ENTRY
    print(f"\n--- 7. SIGNAL SCORE AT ENTRY ---")
    for d in ['bullish', 'bearish']:
        sub = df[df.direction == d]
        w = sub[sub.win]
        l = sub[~sub.win]
        sc = 'bull_score' if d == 'bullish' else 'bear_score'
        if len(w) > 0 and len(l) > 0:
            print(f"  {d} Winners {sc}: {w[sc].mean():.2f} | Losers: {l[sc].mean():.2f} | "
                  f"W ATR={w.atr_at_entry.mean():.1f} | L ATR={l.atr_at_entry.mean():.1f}")

    # 8. STRATEGY TYPE
    print(f"\n--- 8. STRATEGY TYPE ---")
    df['strat_type'] = df['strategy'].apply(lambda x: 'trend' if 'pro_' in str(x) else ('mean-rev' if 'mr_' in str(x) else 'other'))
    for st in df['strat_type'].unique():
        sub = df[df.strat_type == st]
        w = sub[sub.win]
        print(f"  {st:10s}: {len(sub):3d} trades | WR={len(w)/len(sub)*100:5.1f}% | PnL={sub.net_pnl.sum():+8.0f}")

    # 9. TOP 20 BIGGEST LOSERS
    print(f"\n--- 9. TOP 20 BIGGEST LOSSES ---")
    worst = df.nsmallest(20, 'net_pnl')
    for _, t in worst.iterrows():
        sigs = []
        for sig in [c for c in df.columns if c.startswith('signal_')]:
            if t[sig]: sigs.append(sig.replace('signal_', ''))
        sig_str = ','.join(sigs) if sigs else 'none'
        print(f"  {str(t.entry_time)[:10]} | {t.direction:8s} | {t.regime_at_entry:14s} | "
              f"PnL={t.net_pnl:+7.0f} | {t.exit_reason:10s} | "
              f"Score={t.bull_score:.1f}/{t.bear_score:.1f} | Sigs={sig_str}")

    # 10. CONSECUTIVE LOSSES
    print(f"\n--- 10. CONSECUTIVE LOSS STREAKS ---")
    consec = 0; max_consec = 0; streak_pnl = 0; max_streak_pnl = 0
    for _, t in df.iterrows():
        if not t.win:
            consec += 1; streak_pnl += t.net_pnl
        else:
            if consec > max_consec:
                max_consec = consec; max_streak_pnl = streak_pnl
            consec = 0; streak_pnl = 0
    print(f"  Max consecutive losses: {max_consec}")
    print(f"  Max streak PnL: {max_streak_pnl:+.0f}")

    # 11. COST IMPACT
    print(f"\n--- 11. COST IMPACT ---")
    killed_by_costs = losers[losers.gross_pnl > 0]
    pct = len(killed_by_costs)/len(losers)*100 if len(losers) > 0 else 0
    print(f"  Losers killed by costs (gross>0, net<0): {len(killed_by_costs)}/{len(losers)} ({pct:.1f}%)")
    print(f"  PnL destroyed by costs: {killed_by_costs.net_pnl.sum():+.0f}")
    print(f"  Avg cost/trade: {df.costs.mean():.0f} | Total costs: {df.costs.sum():.0f}")

    # 12. REGIME x EXIT REASON CROSS-TAB (LOSSES)
    print(f"\n--- 12. REGIME x EXIT REASON (LOSS HOTSPOTS) ---")
    loss_hotspots = []
    for regime in sorted(df['regime_at_entry'].unique()):
        for reason in sorted(df['exit_reason'].unique()):
            sub = df[(df.regime_at_entry == regime) & (df.exit_reason == reason) & (~df.win)]
            if len(sub) >= 3:
                loss_hotspots.append((regime, reason, len(sub), sub.net_pnl.sum()))
    loss_hotspots.sort(key=lambda x: x[3])
    for regime, reason, cnt, pnl in loss_hotspots:
        print(f"  {regime:14s} + {reason:10s}: {cnt:3d} losses | TotalLoss={pnl:+8.0f}")

    # 13. ATR QUARTILE
    print(f"\n--- 13. ATR QUARTILE ANALYSIS ---")
    try:
        df['atr_q'] = pd.qcut(df['atr_at_entry'], q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'], duplicates='drop')
        for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
            sub = df[df.atr_q == q]
            if len(sub) > 0:
                w = sub[sub.win]
                print(f"  {q:8s}: {len(sub):3d} trades | WR={len(w)/len(sub)*100:5.1f}% | "
                      f"AvgPnL={sub.net_pnl.mean():+7.0f} | ATR={sub.atr_at_entry.min():.0f}-{sub.atr_at_entry.max():.0f}")
    except Exception as e:
        print(f"  Could not compute quartiles: {e}")

    return df


if __name__ == '__main__':
    for symbol in ['NIFTY 50', 'NIFTY BANK', 'SENSEX']:
        print(f"\n{'#'*80}")
        print(f"  PROCESSING: {symbol}")
        print(f"{'#'*80}")
        df = run_and_extract(symbol, 6750)
        if len(df) > 0:
            analyze(df, symbol)
            fname = f"trade_analysis_{symbol.replace(' ', '_')}_full.csv"
            df.to_csv(fname, index=False)
            print(f"\n  >> Saved: {fname}")
