import sqlite3
import pandas as pd

conn = sqlite3.connect('reports/papertrade/live_ledger.sqlite')
df = pd.read_sql_query("SELECT * FROM paper_trades WHERE entry_time >= '2026-11-16' ORDER BY entry_time ASC", conn)

print("=== Clean-Room by GEX Regime ===")
for reg, g in df.groupby('gex_regime'):
    wins = (g['net_pnl'] > 0).sum()
    print(f"{reg}: N={len(g)}, Wins={wins}, Losses={len(g)-wins}, WR={wins/len(g)*100:.1f}%, Net=Rs {g['net_pnl'].sum():.2f}")

print("\n=== Clean-Room SENSEX by GEX Regime ===")
sx = df[df['symbol'] == 'SENSEX']
for reg, g in sx.groupby('gex_regime'):
    wins = (g['net_pnl'] > 0).sum()
    print(f"{reg}: N={len(g)}, Wins={wins}, Losses={len(g)-wins}, WR={wins/len(g)*100:.1f}%, Net=Rs {g['net_pnl'].sum():.2f}")

print("\n=== Clean-Room NIFTY BANK by GEX Regime ===")
bn = df[df['symbol'] == 'NIFTY BANK']
for reg, g in bn.groupby('gex_regime'):
    wins = (g['net_pnl'] > 0).sum()
    print(f"{reg}: N={len(g)}, Wins={wins}, Losses={len(g)-wins}, WR={wins/len(g)*100:.1f}%, Net=Rs {g['net_pnl'].sum():.2f}")
