import sqlite3
import pandas as pd

conn = sqlite3.connect('reports/papertrade/live_ledger.sqlite')
df = pd.read_sql_query("SELECT * FROM paper_trades WHERE entry_time >= '2026-11-17' ORDER BY entry_time ASC", conn)

print(f"Total trades on 2026-11-17: {len(df)}")
for idx, row in df.iterrows():
    tid = row['trade_id']
    sym = row['symbol']
    inst = row['instrument']
    tier = row['tier']
    en_t = row['entry_time']
    ex_t = row['exit_time']
    en_p = row['entry_price']
    ex_p = row['exit_price']
    net = row['net_pnl']
    gross = row['gross_pnl']
    cost = row['costs']
    reason = row['exit_reason']
    score = row['signal_score']
    strat = row['strategy']
    sl = row['stop_loss']
    tgt = row['target']
    dur = row['holding_duration_seconds']
    gex = row['net_gex']
    regime = row['gex_regime']
    print(f"Trade #{idx+1}: {tid} | {sym} | {inst} | Tier {tier} | Net: Rs {net:.2f} (Gross: Rs {gross:.2f}, Cost: Rs {cost:.2f}) | {reason} ({dur}s) | Entry: {en_t} @ {en_p:.2f} -> Exit: {ex_t} @ {ex_p:.2f} | Strat: {strat} | Score: {score} | SL: {sl} | TGT: {tgt} | GEX: {gex} ({regime})")
