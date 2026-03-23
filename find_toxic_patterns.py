"""Find patterns that exist EXCLUSIVELY in losing trades - never in a single winner."""
import pandas as pd
import numpy as np
from itertools import combinations

def find_toxic_patterns(csv_file, symbol):
    df = pd.read_csv(csv_file)
    df['win'] = df['net_pnl'] > 0
    df['year'] = pd.to_datetime(df['entry_time']).dt.year

    signal_cols = [c for c in df.columns if c.startswith('signal_')]
    df['sig_count'] = df[signal_cols].sum(axis=1)

    # Focus on OOS era (2016+) since we agreed 2007-2015 is irrelevant
    df_full = df.copy()
    df = df[df.year >= 2016].copy()

    losers = df[~df.win]
    winners = df[df.win]

    print(f"\n{'#'*80}")
    print(f"  TOXIC PATTERN MINING: {symbol} (2016-2026 only)")
    print(f"  {len(losers)} losses | {len(winners)} wins | Looking for LOSS-ONLY patterns")
    print(f"{'#'*80}")

    toxic = []

    # === 1. DIRECTION + REGIME ===
    print(f"\n--- 1. DIRECTION + REGIME TOXIC COMBOS ---")
    for direction in df['direction'].unique():
        for regime in df['regime_at_entry'].unique():
            mask = (df.direction == direction) & (df.regime_at_entry == regime)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'{direction} + {regime}', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'dir+regime'})
                print(f"  TOXIC: {direction} in {regime}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 2. DIRECTION + REGIME + EXIT REASON ===
    print(f"\n--- 2. TRIPLE COMBO: DIRECTION + REGIME + EXIT REASON ---")
    for direction in df['direction'].unique():
        for regime in df['regime_at_entry'].unique():
            for reason in df['exit_reason'].unique():
                mask = (df.direction == direction) & (df.regime_at_entry == regime) & (df.exit_reason == reason)
                sub = df[mask]
                if len(sub) >= 5 and sub.win.sum() == 0:
                    toxic.append({'pattern': f'{direction} + {regime} + {reason}', 'count': len(sub),
                                  'total_loss': sub.net_pnl.sum(), 'type': 'triple'})
                    print(f"  TOXIC: {direction} + {regime} + {reason}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 3. SIGNAL PAIR COMBOS ===
    print(f"\n--- 3. SIGNAL PAIR TOXIC COMBOS ---")
    for sig_a, sig_b in combinations(signal_cols, 2):
        mask = (df[sig_a] == True) & (df[sig_b] == True)
        sub = df[mask]
        if len(sub) >= 5 and sub.win.sum() == 0:
            na = sig_a.replace('signal_', '')
            nb = sig_b.replace('signal_', '')
            toxic.append({'pattern': f'{na} + {nb}', 'count': len(sub),
                          'total_loss': sub.net_pnl.sum(), 'type': 'signal_pair'})
            print(f"  TOXIC: {na} + {nb}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 4. SIGNAL + DIRECTION ===
    print(f"\n--- 4. SIGNAL + DIRECTION TOXIC ---")
    for sig in signal_cols:
        for d in ['bullish', 'bearish']:
            mask = (df[sig] == True) & (df.direction == d)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'{sig.replace("signal_", "")} + {d}', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'sig+dir'})
                print(f"  TOXIC: {sig.replace('signal_', '')} + {d}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 5. SIGNAL + REGIME ===
    print(f"\n--- 5. SIGNAL + REGIME TOXIC ---")
    for sig in signal_cols:
        for regime in df['regime_at_entry'].unique():
            mask = (df[sig] == True) & (df.regime_at_entry == regime)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'{sig.replace("signal_", "")} + {regime}', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'sig+regime'})
                print(f"  TOXIC: {sig.replace('signal_', '')} + {regime}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 6. SIGNAL COUNT + DIRECTION ===
    print(f"\n--- 6. SIGNAL COUNT + DIRECTION TOXIC ---")
    for d in ['bullish', 'bearish']:
        for n in range(0, 6):
            mask = (df.direction == d) & (df.sig_count == n)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'{d} + exactly {n} signals', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'sigcount+dir'})
                print(f"  TOXIC: {d} with exactly {n} signals: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 7. SIGNAL COUNT + REGIME ===
    print(f"\n--- 7. SIGNAL COUNT + REGIME TOXIC ---")
    for regime in df['regime_at_entry'].unique():
        for n in range(0, 6):
            mask = (df.regime_at_entry == regime) & (df.sig_count == n)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'{regime} + exactly {n} signals', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'sigcount+regime'})
                print(f"  TOXIC: {regime} + exactly {n} signals: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 8. SCORE THRESHOLDS ===
    print(f"\n--- 8. SCORE THRESHOLD TOXIC ---")
    for d in ['bullish', 'bearish']:
        sub_d = df[df.direction == d]
        sc = 'bull_score' if d == 'bullish' else 'bear_score'
        for lo, hi in [(0.01, 1.0), (1.0, 2.0), (2.0, 2.5), (0.01, 1.5), (0.01, 2.0)]:
            mask = (sub_d[sc] >= lo) & (sub_d[sc] < hi)
            sub = sub_d[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'{d} + {sc} in [{lo},{hi})', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'score_range'})
                print(f"  TOXIC: {d} with {sc} in [{lo:.1f}, {hi:.1f}): {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 9. DIRECTION + REGIME + LOW SIG COUNT ===
    print(f"\n--- 9. DIRECTION + REGIME + LOW SIGNAL COUNT ---")
    for direction in df['direction'].unique():
        for regime in df['regime_at_entry'].unique():
            for max_sig in [0, 1, 2]:
                mask = (df.direction == direction) & (df.regime_at_entry == regime) & (df.sig_count <= max_sig)
                sub = df[mask]
                if len(sub) >= 5 and sub.win.sum() == 0:
                    toxic.append({'pattern': f'{direction} + {regime} + sigs<={max_sig}', 'count': len(sub),
                                  'total_loss': sub.net_pnl.sum(), 'type': 'dir+regime+lowsig'})
                    print(f"  TOXIC: {direction} + {regime} + <={max_sig} sigs: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 10. STRATEGY TYPE SPECIFIC ===
    print(f"\n--- 10. STRATEGY TYPE TOXIC ---")
    df['strat_type'] = df['strategy'].apply(lambda x: 'trend' if 'pro_' in str(x) else ('mr' if 'mr_' in str(x) else 'other'))
    for st in df['strat_type'].unique():
        for regime in df['regime_at_entry'].unique():
            mask = (df.strat_type == st) & (df.regime_at_entry == regime)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'strat={st} + {regime}', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'strat+regime'})
                print(f"  TOXIC: strat={st} in {regime}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    for st in df['strat_type'].unique():
        for d in ['bullish', 'bearish']:
            mask = (df.strat_type == st) & (df.direction == d)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'strat={st} + {d}', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'strat+dir'})
                print(f"  TOXIC: strat={st} + {d}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 11. YEAR-SPECIFIC TOXIC ===
    print(f"\n--- 11. YEAR-SPECIFIC TOXIC (entire year, one direction) ---")
    for year in sorted(df['year'].unique()):
        for d in ['bullish', 'bearish']:
            mask = (df.year == year) & (df.direction == d)
            sub = df[mask]
            if len(sub) >= 5 and sub.win.sum() == 0:
                toxic.append({'pattern': f'year={year} + {d}', 'count': len(sub),
                              'total_loss': sub.net_pnl.sum(), 'type': 'year+dir'})
                print(f"  TOXIC: {year} + {d}: {len(sub)} trades, ALL LOSSES, PnL={sub.net_pnl.sum():+.0f}")

    # === 12. LOSS STREAK FINGERPRINTS ===
    print(f"\n--- 12. LOSS STREAK FINGERPRINTS (5+ consecutive losses) ---")
    streaks = []
    current = []
    for _, row in df.iterrows():
        if not row['win']:
            current.append(row)
        else:
            if len(current) >= 5:
                streaks.append(pd.DataFrame(current))
            current = []
    if len(current) >= 5:
        streaks.append(pd.DataFrame(current))

    print(f"  Found {len(streaks)} streaks of 5+ losses")
    for i, streak in enumerate(streaks):
        dirs = streak['direction'].value_counts().to_dict()
        regs = streak['regime_at_entry'].value_counts().to_dict()
        avg_sig = streak['sig_count'].mean()
        print(f"  Streak {i+1}: {len(streak)} losses | {str(streak.iloc[0]['entry_time'])[:10]} to "
              f"{str(streak.iloc[-1]['entry_time'])[:10]} | PnL={streak['net_pnl'].sum():+.0f} | "
              f"AvgSigs={avg_sig:.1f} | Dir={dirs} | Regime={regs}")

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {len(toxic)} TOXIC PATTERNS FOR {symbol} (2016-2026)")
    print(f"{'='*70}")

    if toxic:
        toxic_df = pd.DataFrame(toxic).sort_values('total_loss')
        print(f"\n  TOP 25 MOST DESTRUCTIVE (sorted by total damage):")
        for _, row in toxic_df.head(25).iterrows():
            print(f"    {row['pattern']:55s} | {row['count']:3d} trades | PnL={row['total_loss']:+8.0f}")

    return toxic


if __name__ == '__main__':
    from loguru import logger
    logger.remove()

    all_toxic = {}
    for sym, csv in [('NIFTY_50', 'trade_analysis_NIFTY_50_full.csv'),
                     ('NIFTY_BANK', 'trade_analysis_NIFTY_BANK_full.csv'),
                     ('SENSEX', 'trade_analysis_SENSEX_full.csv')]:
        all_toxic[sym] = find_toxic_patterns(csv, sym)

    # Cross-index
    print(f"\n{'#'*80}")
    print(f"  CROSS-INDEX: Toxic patterns found in 2+ indices")
    print(f"{'#'*80}")
    sets = {}
    for sym in all_toxic:
        sets[sym] = set(t['pattern'] for t in all_toxic[sym])

    all_three = sets['NIFTY_50'] & sets['NIFTY_BANK'] & sets['SENSEX']
    two_plus = (sets['NIFTY_50'] & sets['NIFTY_BANK']) | (sets['NIFTY_50'] & sets['SENSEX']) | (sets['NIFTY_BANK'] & sets['SENSEX'])

    if all_three:
        print(f"\n  === UNIVERSAL TOXIC (all 3 indices): {len(all_three)} patterns ===")
        for p in sorted(all_three):
            total_c = 0; total_p = 0
            for sym in all_toxic:
                for t in all_toxic[sym]:
                    if t['pattern'] == p:
                        total_c += t['count']; total_p += t['total_loss']
            print(f"    {p:55s} | {total_c:3d} trades | PnL={total_p:+.0f}")

    if two_plus - all_three:
        print(f"\n  === FOUND IN 2 INDICES: {len(two_plus - all_three)} patterns ===")
        for p in sorted(two_plus - all_three):
            indices = []
            if p in sets['NIFTY_50']: indices.append('N50')
            if p in sets['NIFTY_BANK']: indices.append('BNK')
            if p in sets['SENSEX']: indices.append('SEN')
            total_c = 0; total_p = 0
            for sym in all_toxic:
                for t in all_toxic[sym]:
                    if t['pattern'] == p:
                        total_c += t['count']; total_p += t['total_loss']
            print(f"    {p:55s} | {'+'.join(indices):10s} | {total_c:3d} trades | PnL={total_p:+.0f}")
