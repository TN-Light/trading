@echo off
setlocal
python run_regression_guard.py --base-ref HEAD~1 --target-ref WORKTREE --symbols "NIFTY 50" "NIFTY BANK" "SENSEX" --days 1825 3650 5475 --data-source yfinance --fetch-retries 2 %*
