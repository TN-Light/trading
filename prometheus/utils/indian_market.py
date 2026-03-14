# ============================================================================
# PROMETHEUS — Utility: Indian Market Constants & Helpers
# ============================================================================
"""
Indian market specific constants, trading calendar, lot sizes, and helpers.
All SEBI/NSE specific rules encoded here.
"""

from datetime import datetime, time, date, timedelta
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Trading Hours
# ---------------------------------------------------------------------------
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)

# ---------------------------------------------------------------------------
# Lot Sizes (updated periodically by NSE — keep current)
# ---------------------------------------------------------------------------
LOT_SIZES = {
    "NIFTY 50": 25,
    "NIFTY BANK": 15,
    "NIFTY FIN SERVICE": 25,
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "SENSEX": 10,
    "NIFTY IT": 25,
    "NIFTY MIDCAP SELECT": 50,
    "NIFTY NEXT 50": 25,
    # Stock F&O lot sizes — add as needed
    "RELIANCE": 250,
    "TCS": 150,
    "INFY": 300,
    "HDFCBANK": 550,
    "ICICIBANK": 700,
    "SBIN": 750,
    "TATAMOTORS": 575,
    "ITC": 1600,
    "AXISBANK": 600,
    "BAJFINANCE": 125,
}

# ---------------------------------------------------------------------------
# Strike Price Intervals
# ---------------------------------------------------------------------------
STRIKE_INTERVALS = {
    "NIFTY 50": 50,
    "NIFTY BANK": 100,
    "NIFTY FIN SERVICE": 50,
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "SENSEX": 100,
    "NIFTY IT": 50,
    "NIFTY MIDCAP SELECT": 25,
    "NIFTY NEXT 50": 50,
}

# ---------------------------------------------------------------------------
# Weekly Expiry Days
# ---------------------------------------------------------------------------
WEEKLY_EXPIRY_DAYS = {
    "NIFTY 50": "Thursday",       # NIFTY weekly on Thursday
    "NIFTY BANK": "Wednesday",    # BANKNIFTY weekly on Wednesday
    "NIFTY FIN SERVICE": "Tuesday",  # FINNIFTY weekly on Tuesday
    "NIFTY": "Thursday",
    "BANKNIFTY": "Wednesday",
    "FINNIFTY": "Tuesday",
}

# ---------------------------------------------------------------------------
# NSE Holidays 2025-2026 (update yearly)
# ---------------------------------------------------------------------------
NSE_HOLIDAYS_2025 = [
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr
    date(2025, 4, 10),   # Shri Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 6, 7),    # Bakri Id
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Janmashtami
    date(2025, 10, 1),   # Mahatma Gandhi Jayanti
    date(2025, 10, 2),   # Dussehra
    date(2025, 10, 21),  # Diwali (Laxmi Puja)
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
]

NSE_HOLIDAYS_2026 = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Mahashivratri
    date(2026, 3, 3),    # Holi
    date(2026, 3, 20),   # Id-Ul-Fitr
    date(2026, 3, 30),   # Shri Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 27),   # Bakri Id
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 25),   # Muharram
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 19),  # Dussehra
    date(2026, 11, 8),   # Diwali
    date(2026, 11, 9),   # Diwali Balipratipada
    date(2026, 12, 25),  # Christmas
]

ALL_HOLIDAYS = set(NSE_HOLIDAYS_2025 + NSE_HOLIDAYS_2026)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Check if Indian market is currently open."""
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)

    # Weekend check
    if dt.weekday() >= 5:
        return False

    # Holiday check
    if dt.date() in ALL_HOLIDAYS:
        return False

    # Time check
    current_time = dt.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_trading_day(d: Optional[date] = None) -> bool:
    """Check if a given date is a trading day."""
    if d is None:
        d = datetime.now(IST).date()
    if d.weekday() >= 5:
        return False
    return d not in ALL_HOLIDAYS


def next_trading_day(d: Optional[date] = None) -> date:
    """Get the next trading day from a given date."""
    if d is None:
        d = datetime.now(IST).date()
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d = d + timedelta(days=1)
    return d


def get_expiry_date(symbol: str, from_date: Optional[date] = None) -> date:
    """Get the next weekly expiry date for an index."""
    if from_date is None:
        from_date = datetime.now(IST).date()

    expiry_day_name = WEEKLY_EXPIRY_DAYS.get(symbol, "Thursday")
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4
    }
    target_weekday = day_map[expiry_day_name]

    days_ahead = target_weekday - from_date.weekday()
    if days_ahead < 0:
        days_ahead += 7

    expiry = from_date + timedelta(days=days_ahead)

    # If expiry falls on holiday, move to previous trading day
    while expiry in ALL_HOLIDAYS or expiry.weekday() >= 5:
        expiry = expiry - timedelta(days=1)

    return expiry


def days_to_expiry(symbol: str, from_date: Optional[date] = None) -> int:
    """Calculate trading days remaining to next expiry."""
    if from_date is None:
        from_date = datetime.now(IST).date()
    expiry = get_expiry_date(symbol, from_date)

    trading_days = 0
    current = from_date
    while current < expiry:
        current += timedelta(days=1)
        if is_trading_day(current):
            trading_days += 1

    return trading_days


def get_lot_size(symbol: str) -> int:
    """Get lot size for a symbol."""
    return LOT_SIZES.get(symbol, 1)


def get_strike_interval(symbol: str) -> int:
    """Get strike price interval for an index."""
    return STRIKE_INTERVALS.get(symbol, 50)


def get_atm_strike(spot_price: float, symbol: str) -> float:
    """Get the At-The-Money strike price."""
    interval = get_strike_interval(symbol)
    return round(spot_price / interval) * interval


def minutes_to_close(dt: Optional[datetime] = None) -> int:
    """Get minutes remaining to market close."""
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)

    close_dt = dt.replace(hour=15, minute=30, second=0, microsecond=0)
    if dt >= close_dt:
        return 0

    delta = close_dt - dt
    return int(delta.total_seconds() / 60)


def get_monthly_expiry(year: int, month: int) -> date:
    """Get the last Thursday of a month (monthly F&O expiry)."""
    # Start from end of month
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)

    last_day = next_month_first - timedelta(days=1)

    # Find last Thursday
    while last_day.weekday() != 3:  # 3 = Thursday
        last_day -= timedelta(days=1)

    # If holiday, move to previous trading day
    while last_day in ALL_HOLIDAYS:
        last_day -= timedelta(days=1)

    return last_day
