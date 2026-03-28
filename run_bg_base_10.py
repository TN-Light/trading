
import subprocess
with open('fast_base.txt', 'w', encoding='utf-8') as f:
    subprocess.run(['python', '-u', 'prometheus/main.py', 'backtest', '--symbol', 'NIFTY 50', '--days', '10', '--intraday', '--interval', '5', '--apex'], stdout=f, stderr=subprocess.STDOUT)
