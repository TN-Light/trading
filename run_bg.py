
import subprocess
with open('final_30d_tvi.txt', 'w', encoding='utf-8') as f:
    subprocess.run(['python', 'prometheus/main.py', 'backtest', '--symbol', 'NIFTY 50', '--days', '30', '--intraday', '--interval', '5', '--apex'], stdout=f, stderr=subprocess.STDOUT)
