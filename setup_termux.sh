#!/bin/bash
# ============================================================
# Prometheus — Termux (Android) Setup Script
# Run this ONCE after cloning the repo
# ============================================================

set -e

echo "📱 Prometheus Termux Setup"
echo "========================="

# Step 1: System packages
echo ""
echo "📦 Installing system packages..."
pkg update -y
pkg upgrade -y
pkg install python git build-essential libffi openssl rust binutils -y

# Step 2: Install numpy/pandas from Termux repo (pre-built, no compilation)
echo ""
echo "📦 Installing numpy & pandas (pre-built for ARM)..."
pip install numpy pandas --prefer-binary --no-build-isolation 2>/dev/null || {
    echo "⚠️  pip binary install failed, trying Termux packages..."
    pkg install python-numpy python-pandas -y 2>/dev/null || {
        echo "⚠️  Termux packages not available, building from source (slow)..."
        MATHLIB=m pip install numpy
        pip install pandas
    }
}

# Step 3: Install remaining dependencies
echo ""
echo "📦 Installing Python packages..."
pip install scipy PyYAML rich loguru pytz python-dateutil requests websocket-client pyotp

echo ""
echo "📦 Installing trading packages..."
pip install yfinance smartapi-python kiteconnect

echo ""
echo "📦 Installing technical analysis..."
pip install ta pandas-ta

echo ""
echo "📦 Installing telegram & database..."
pip install python-telegram-bot sqlalchemy

# Step 4: Verify
echo ""
echo "✅ Verifying installation..."
python -c "
import pandas, numpy, scipy, yaml, rich, loguru
import requests, websocket, pyotp, pytz
print('Core packages: OK')
try:
    from SmartApi import SmartConnect
    print('Angel One API: OK')
except: print('Angel One API: FAILED (check smartapi-python)')
try:
    import telegram
    print('Telegram: OK')
except: print('Telegram: FAILED')
print()
print('✅ Setup complete! Now:')
print('  1. Edit credentials: nano prometheus/config/credentials.yaml')
print('  2. Start bot: bash start_bot.sh')
"
