import sqlite3

conn = sqlite3.connect('reports/papertrade/live_ledger.sqlite')
c = conn.cursor()

print("=== Active Table: paper_trades ===")
c.execute('SELECT COUNT(*), MIN(entry_time), MAX(entry_time) FROM paper_trades')
cnt, min_t, max_t = c.fetchone()
print(f"Count: {cnt}, Min Entry: {min_t}, Max Entry: {max_t}")

c.execute("""
    SELECT 
        COUNT(*),
        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
        ROUND(SUM(gross_pnl), 2) as gross,
        ROUND(SUM(costs), 2) as costs,
        ROUND(SUM(net_pnl), 2) as net
    FROM paper_trades
""")
print("Active Summary:", c.fetchone())

c.execute("""
    SELECT 
        tier,
        COUNT(*),
        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
        ROUND(SUM(gross_pnl), 2) as gross,
        ROUND(SUM(costs), 2) as costs,
        ROUND(SUM(net_pnl), 2) as net
    FROM paper_trades
    GROUP BY tier
""")
print("Active by Tier:", c.fetchall())

print("\n=== Persistent Capital Calculation ===")
c.execute("SELECT COALESCE(SUM(net_pnl), 0.0) FROM paper_trades WHERE tier IN ('S', 'A', 'B')")
cum_pnl = c.fetchone()[0]
print(f"Live Eligible (Tier S/A/B) Cumulative Net PnL: Rs {cum_pnl:+.2f}")
print(f"Persistent Live Capital (Baseline Rs 100,000): Rs {100000.0 + cum_pnl:,.2f}")

print("\n=== Archive Table: legacy_paper_trades ===")
c.execute('SELECT COUNT(*), MIN(entry_time), MAX(entry_time) FROM legacy_paper_trades')
cnt_leg, min_leg, max_leg = c.fetchone()
print(f"Count: {cnt_leg}, Min Entry: {min_leg}, Max Entry: {max_leg}")

c.execute("""
    SELECT 
        COUNT(*),
        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
        ROUND(SUM(gross_pnl), 2) as gross,
        ROUND(SUM(costs), 2) as costs,
        ROUND(SUM(net_pnl), 2) as net
    FROM legacy_paper_trades
""")
print("Legacy Summary:", c.fetchone())
