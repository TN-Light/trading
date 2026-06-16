#!/bin/bash
# ============================================================
# Prometheus Trading Bot — Startup Script
# Works on Linux, macOS, Termux (Android), WSL
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/bot.log"
PID_FILE="$SCRIPT_DIR/.bot.pid"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${RED}Bot is already running (PID $OLD_PID)${NC}"
        echo "Stop it first: bash stop_bot.sh  OR  pkill -f 'prometheus/main.py'"
        exit 1
    fi
fi

# Detect environment
if [ -d "/data/data/com.termux" ]; then
    echo -e "${GREEN}📱 Termux detected — acquiring wake-lock${NC}"
    termux-wake-lock 2>/dev/null
    PYTHON="python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

echo -e "${GREEN}🚀 Starting Prometheus Trading Bot${NC}"
echo "   Python: $PYTHON"
echo "   Log:    $LOG_FILE"
echo ""

# Start in background
nohup $PYTHON prometheus/main.py paper --data-source auto --fetch-retries 2 > "$LOG_FILE" 2>&1 &
BOT_PID=$!
echo "$BOT_PID" > "$PID_FILE"

# Wait a moment and verify it started
sleep 3
if kill -0 "$BOT_PID" 2>/dev/null; then
    echo -e "${GREEN}✅ Bot started successfully (PID $BOT_PID)${NC}"
    echo ""
    echo "Commands:"
    echo "  View logs:  tail -f $LOG_FILE"
    echo "  Stop bot:   pkill -f 'prometheus/main.py'"
    echo "  Check:      ps aux | grep prometheus"
else
    echo -e "${RED}❌ Bot failed to start. Check logs:${NC}"
    tail -20 "$LOG_FILE"
    exit 1
fi
