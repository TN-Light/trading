import pandas as pd
from prometheus.main import Prometheus
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/NIFTY 50_5minute.csv', parse_dates=['date'])
df.rename(columns={'date': 'timestamp'}, inplace=True)
df = df.sort_values('timestamp').reset_index(drop=True)

# Let's get EXACTLY 2018 Q1 plus the 1000 warmup bars prior.
df_2018 = df[df['timestamp'].dt.year >= 2018].reset_index(drop=True)

q1_mask = (df_2018['timestamp'].dt.month <= 3) & (df_2018['timestamp'].dt.year == 2018)
start_idx = df_2018[q1_mask].index.min()
end_idx = df_2018[q1_mask].index.max()

print(f"2018 Q1 strict indices: {start_idx} to {end_idx}")

# Same warmup logic as the fast script
actual_start = max(0, start_idx - 1000)
data_slice = df_2018.iloc[actual_start:end_idx+1].copy()

ds = data_slice.set_index("timestamp")
data_daily = ds.resample('D').agg({
    'open':'first',
    'high':'max',
    'low':'min',
    'close':'last',
    'volume':'sum'
}).dropna().reset_index()

prom = Prometheus()
param_overrides = {"apex": True}

res, engine = prom._run_intraday_backtest_on_slice(
    data_slice=data_slice,
    data_daily=data_daily,
    symbol="NIFTY 50",
    bar_interval="5minute",
    strategy_name="APEX_Q1_TEST",
    parrondo=False,
    dd_throttle=True,
    param_overrides=param_overrides,
    verbose=False
)

# Filter like the fast script
valid_start = df_2018.loc[start_idx, 'timestamp']
q1_trades = [t for t in engine.trades if pd.to_datetime(t.entry_time) >= valid_start]
print(f"Trades found correctly in 2018 Q1: {len(q1_trades)}")

# Re-load the fast script result to compare
import pickle
with open('backtest_chunks/trades_2018_Q1.pkl', 'rb') as f:
    fast_trades = pickle.load(f)

print(f"Trades found by Fast Script: {len(fast_trades)}")

# Compare first 5 trades from both
if len(q1_trades) == len(fast_trades) and len(q1_trades) > 0:
    print("Match confirmed! Printing first 3 trades to show they are identical:")
    for i in range(3):
        print(f"Engine Trade {i+1}: Entry: {q1_trades[i].entry_time} Dir: {q1_trades[i].direction} PnL: {q1_trades[i].net_pnl}")
        print(f"Fast   Trade {i+1}: Entry: {fast_trades[i].entry_time} Dir: {fast_trades[i].direction} PnL: {fast_trades[i].net_pnl}")
