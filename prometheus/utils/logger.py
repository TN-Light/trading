# ============================================================================
# PROMETHEUS — Utility: Structured Logging
# ============================================================================
"""
Structured logging with separate channels for trades, signals, and system events.
Uses loguru for clean, colored output.
"""

import os
import sys
from datetime import datetime
from loguru import logger

# Remove default logger
logger.remove()

# Create logs directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging(level: str = "INFO"):
    """Initialize all logging channels."""
    logger.remove()

    # Console output — colored, concise
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}",
        colorize=True,
    )

    # Main system log
    logger.add(
        os.path.join(LOG_DIR, "prometheus.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        rotation="50 MB",
        retention="30 days",
        compression="gz",
    )

    # Trade-specific log
    logger.add(
        os.path.join(LOG_DIR, "trades.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        filter=lambda record: "trade" in record["extra"],
        rotation="10 MB",
        retention="90 days",
    )

    # Signal-specific log
    logger.add(
        os.path.join(LOG_DIR, "signals.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        filter=lambda record: "signal" in record["extra"],
        rotation="10 MB",
        retention="90 days",
    )

    return logger


# Pre-bound loggers for specific domains
trade_logger = logger.bind(trade=True)
signal_logger = logger.bind(signal=True)


def log_trade(action: str, symbol: str, details: dict):
    """Log a trade event with structured data."""
    trade_logger.info(f"TRADE | {action} | {symbol} | {details}")


def log_signal(signal_type: str, symbol: str, strength: float, details: dict):
    """Log a signal event."""
    signal_logger.info(f"SIGNAL | {signal_type} | {symbol} | strength={strength:.2f} | {details}")


def log_risk(event: str, details: dict):
    """Log a risk management event."""
    logger.warning(f"RISK | {event} | {details}")
