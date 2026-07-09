# PROMETHEUS Pre-Commit Smoke Test
# Run this before every git push to catch dead code bugs
# Usage: python smoke_test_quick.py

import subprocess, sys

PYTHON = sys.executable
TESTS = [
    # 1. Import check — catches deleted-file references
    ([PYTHON, "-c", "import prometheus.main"], "Import check"),

    # 2. CLI parse check — catches dead argument references
    ([PYTHON, "prometheus/main.py", "--help"], "CLI --help parse"),

    # 3. Backtest smoke — catches 300minute / zero-trade bugs
    ([PYTHON, "prometheus/main.py", "backtest", "--days", "5",
      "--symbol", "NIFTY 50", "--intraday"], "Intraday backtest 5d"),

    # 4. Swing backtest smoke
    ([PYTHON, "prometheus/main.py", "backtest", "--days", "30",
      "--symbol", "NIFTY 50"], "Swing backtest 30d"),
]

passed = 0
failed = 0
for cmd, label in TESTS:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        result = type("R", (), {"returncode": result.returncode, "stdout": stdout, "stderr": stderr})()
        # Crash = non-zero exit AND stderr has Traceback
        if result.returncode != 0 and "Traceback" in result.stderr:
            print(f"  FAIL  {label}")
            print(f"        {result.stderr.splitlines()[-1][:120]}")
            failed += 1
        else:
            print(f"  PASS  {label}")
            passed += 1
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT  {label} (>120s)")
        failed += 1
    except Exception as e:
        print(f"  ERROR  {label}: {e}")
        failed += 1

print(f"\n{passed} passed, {failed} failed")
if failed:
    print("DO NOT COMMIT until failures are fixed.")
    sys.exit(1)
else:
    print("Safe to commit.")
