import os
import shutil
import sqlite3
import pandas as pd

SQLITE_PATH = "reports/papertrade/live_ledger.sqlite"
CSV_PATH = "reports/papertrade/live_ledger.csv"
DAILY_CSV_PATH = "reports/papertrade/daily_performance_ledger.csv"

BACKUP_SQLITE_PATH = "reports/papertrade/legacy_ledger_pre_nov16.sqlite"
BACKUP_CSV_PATH = "reports/papertrade/legacy_ledger_pre_nov16.csv"

print("--- Step 1: Immutable Backup ---")
# Make complete byte-for-byte copy of existing SQLite database
shutil.copyfile(SQLITE_PATH, BACKUP_SQLITE_PATH)
print(f"Backed up SQLite to {BACKUP_SQLITE_PATH} ({os.path.getsize(BACKUP_SQLITE_PATH)} bytes)")

# Make complete copy of existing CSV
shutil.copyfile(CSV_PATH, BACKUP_CSV_PATH)
print(f"Backed up CSV to {BACKUP_CSV_PATH} ({os.path.getsize(BACKUP_CSV_PATH)} bytes)")

print("\n--- Step 2: Database Archive & Purge ---")
conn = sqlite3.connect(SQLITE_PATH)
c = conn.cursor()

# 1. Create legacy_paper_trades table inside live_ledger.sqlite
c.execute("DROP TABLE IF EXISTS legacy_paper_trades")
c.execute("""
    CREATE TABLE legacy_paper_trades AS 
    SELECT * FROM paper_trades WHERE entry_time < '2026-11-16'
""")
c.execute("SELECT COUNT(*) FROM legacy_paper_trades")
legacy_count = c.fetchone()[0]
print(f"Archived {legacy_count} trades into legacy_paper_trades table.")

# 2. Create legacy_post_exit_attribution table inside live_ledger.sqlite
c.execute("DROP TABLE IF EXISTS legacy_post_exit_attribution")
c.execute("""
    CREATE TABLE legacy_post_exit_attribution AS 
    SELECT * FROM post_exit_attribution WHERE exit_time < '2026-11-16'
""")
c.execute("SELECT COUNT(*) FROM legacy_post_exit_attribution")
legacy_attr_count = c.fetchone()[0]
print(f"Archived {legacy_attr_count} rows into legacy_post_exit_attribution table.")

# 3. Purge legacy rows from active paper_trades
c.execute("DELETE FROM paper_trades WHERE entry_time < '2026-11-16'")
c.execute("SELECT COUNT(*) FROM paper_trades")
active_count = c.fetchone()[0]
print(f"Purged pre-Nov 16 trades. Active paper_trades count: {active_count} (Expected: 12)")
assert active_count == 12, f"Expected 12 active trades, found {active_count}"

# 4. Purge legacy rows from active post_exit_attribution
c.execute("DELETE FROM post_exit_attribution WHERE exit_time < '2026-11-16'")
c.execute("SELECT COUNT(*) FROM post_exit_attribution")
active_attr_count = c.fetchone()[0]
print(f"Purged pre-Nov 16 attribution. Active post_exit_attribution count: {active_attr_count} (Expected: 12)")
assert active_attr_count == 12, f"Expected 12 active attribution rows, found {active_attr_count}"

conn.commit()
c.execute("VACUUM")
conn.close()
print("Committed and VACUUM completed on SQLite database.")

print("\n--- Step 3: Synchronize live_ledger.csv ---")
conn = sqlite3.connect(SQLITE_PATH)
df_clean = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY entry_time ASC", conn)
conn.close()

df_clean.to_csv(CSV_PATH, index=False)
print(f"Synchronized {CSV_PATH} with {len(df_clean)} rows.")

print("\n--- Step 4: Synchronize daily_performance_ledger.csv ---")
conn = sqlite3.connect(SQLITE_PATH)
c = conn.cursor()
c.execute("""
    SELECT 
        substr(entry_time, 1, 10) as dt,
        count(*) as total_trades,
        sum(case when net_pnl > 0 then 1 else 0 end) as wins,
        sum(case when net_pnl <= 0 then 1 else 0 end) as losses,
        round(sum(net_pnl), 2) as net_pnl
    FROM paper_trades
    GROUP BY dt
    ORDER BY dt ASC
""")
daily_rows = c.fetchall()
conn.close()

daily_data = []
for r in daily_rows:
    dt, tot, w, l, net = r
    wr = (w / tot * 100) if tot > 0 else 0.0
    daily_data.append({
        "Date": dt,
        "Total_Trades": tot,
        "Wins": w,
        "Losses": l,
        "Win_Rate_Pct": round(wr, 1),
        "Net_PnL": net
    })

df_daily = pd.DataFrame(daily_data)
df_daily.to_csv(DAILY_CSV_PATH, index=False)
print(f"Synchronized {DAILY_CSV_PATH} with {len(df_daily)} daily rows:")
print(df_daily)

print("\n=== Clean-Room Verification Completed Successfully ===")
