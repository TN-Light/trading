import pickle; import os; import pandas as pd; import numpy as np;
from prometheus.backtest.engine import BacktestEngine, BacktestResult, BacktestTrade

f_list = [f for f in os.listdir('backtest_chunks') if f.endswith('.pkl')]
all_trades = []
for f in f_list:
    all_trades.extend(pickle.load(open(os.path.join('backtest_chunks', f), 'rb')))

# sort explicitly by entry_time
all_trades.sort(key=lambda t: pd.to_datetime(t.entry_time))

engine = BacktestEngine(initial_capital=15000)
engine.trades = all_trades

# rebuild equity curve
cap = 15000
curve = [cap]
for t in all_trades:
    cap += t.net_pnl
    curve.append(cap)
    
engine.equity_curve = curve

start_dt = str(all_trades[0].entry_time) if all_trades else '2018-01-01'
end_dt = str(all_trades[-1].exit_time) if all_trades else '2026-03-27'

df_mock = pd.DataFrame({'timestamp': [start_dt, end_dt], 'close': [10000, 20000]})
res = engine._generate_result('8_YEAR_APEX', df_mock, cap) if hasattr(engine, '_generate_result') else engine._calculate_metrics('8_YEAR_APEX', df_mock, cap)

print(res.summary())

# Run 20,000 MC iterations
print("\n--- Monte Carlo Simulation (20,000 runs) ---")
mc = engine.monte_carlo_simulation(res, num_simulations=20000)
print(f"  Probability of profit: {mc.get('prob_profit', 0):.1f}%")
print(f"  Median final capital: Rs {mc.get('median_final_capital', 0):,.0f}")
print(f"  5th percentile (worst case): Rs {mc.get('p5_final_capital', 0):,.0f}")
print(f"  95th percentile (best case): Rs {mc.get('p95_final_capital', 0):,.0f}")
print(f"  Median max drawdown: {mc.get('median_max_drawdown', 0):.1f}%")

