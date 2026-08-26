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
# Trading Hours (updated for SEBI 2025 CAS + F&O extension)
# ---------------------------------------------------------------------------
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)       # Kept for backward compat; cash equities
CASH_CLOSE = time(15, 15)         # F&O stocks regular trading (CAS begins)
FNO_CLOSE = time(15, 40)          # F&O derivatives extended session
PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)         # Random close between 9:07-9:08; 9:08 is conservative

# ---------------------------------------------------------------------------
# Lot Sizes (updated periodically by NSE — keep current)
# ---------------------------------------------------------------------------
LOT_SIZES = {
    "NIFTY 50": 65,              # Verified Aug 2026 (NSE/SEBI periodic review)
    "NIFTY BANK": 30,
    "NIFTY FIN SERVICE": 60,
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "SENSEX": 20,

    "NIFTY MIDCAP SELECT": 120,
    "NIFTY NEXT 50": 25,
    # Stock F&O lot sizes — add as needed
    "RELIANCE": 250,
    "TCS": 150,
    "INFY": 300,
    "HDFCBANK": 650,
    "ICICIBANK": 700,
    "SBIN": 750,
    "TATAMOTORS": 1600,            # TMPV (Tata Motors PV) — verified Aug 2026
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

    "NIFTY MIDCAP SELECT": 25,
    "NIFTY NEXT 50": 50,
}

# ---------------------------------------------------------------------------
# Weekly Expiry Days
# ---------------------------------------------------------------------------
WEEKLY_EXPIRY_DAYS = {
    "NIFTY 50": "Thursday",       # Legacy default; date-aware override applied below
    "NIFTY BANK": "Wednesday",    # Legacy default; date-aware override applied below
    "NIFTY FIN SERVICE": "Tuesday",  # Legacy/default
    "SENSEX": "Friday",           # BSE SENSEX weekly options expire on Friday
    "BANKEX": "Monday",           # BSE BANKEX weekly options expire on Monday
    "NIFTY": "Thursday",
    "BANKNIFTY": "Wednesday",
    "FINNIFTY": "Tuesday",
}

# NSE moved index and stock F&O expiries to Tuesday effective Sep 1, 2025.
NSE_TUESDAY_EXPIRY_CUTOVER = date(2025, 9, 1)

# BSE derivatives keep Friday/Monday expiry under the split schedule.
BSE_FRIDAY_EXPIRY_SYMBOLS = {
    "SENSEX",
    "BSX",
}

# ---------------------------------------------------------------------------
# NSE Holidays 2025-2027 (update yearly)
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
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali (Laxmi Puja)
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
]

NSE_HOLIDAYS_2026 = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 4),    # Holi
    date(2026, 3, 20),   # Eid-Ul-Fitr
    date(2026, 3, 27),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 27),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali Balipratipada
    date(2026, 11, 10),  # Diwali Balipratipada (if Tue)
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
]

SPECIAL_TRADING_DAYS = {
    date(2026, 2, 1),    # Union Budget Day
    date(2026, 11, 8),   # Diwali Muhurat Trading
}

NSE_HOLIDAYS_2027 = [
    date(2027, 1, 26),   # Republic Day
    date(2027, 3, 11),   # Maha Shivaratri
    date(2027, 3, 22),   # Holi
    date(2027, 3, 26),   # Good Friday
    date(2027, 4, 14),   # Dr. Ambedkar Jayanti
    date(2027, 5, 1),    # Maharashtra Day
    date(2027, 5, 13),   # Buddha Purnima
    date(2027, 6, 17),   # Eid-ul-Adha (Bakri Id)
    date(2027, 7, 16),   # Muharram
    date(2027, 8, 15),   # Independence Day
    date(2027, 9, 4),    # Janmashtami
    date(2027, 10, 2),   # Mahatma Gandhi Jayanti
    date(2027, 10, 8),   # Dussehra
    date(2027, 10, 28),  # Diwali (Laxmi Puja)
    date(2027, 10, 29),  # Diwali Balipratipada
    date(2027, 11, 19),  # Guru Nanak Jayanti
    date(2027, 12, 25),  # Christmas
]

ALL_HOLIDAYS = set(NSE_HOLIDAYS_2025 + NSE_HOLIDAYS_2026 + NSE_HOLIDAYS_2027)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Check if Indian market is currently open.

    Uses FNO_CLOSE (15:40) as the upper bound so derivative trades are
    not blocked during the SEBI-extended F&O session (15:30–15:40).
    """
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)

    if dt.date() in SPECIAL_TRADING_DAYS:
        if dt.date() == date(2026, 11, 8):
            return time(18, 15) <= dt.time() <= time(19, 15)
        return MARKET_OPEN <= dt.time() <= FNO_CLOSE

    # Weekend check
    if dt.weekday() >= 5:
        return False

    # Holiday check
    if dt.date() in ALL_HOLIDAYS:
        return False

    # Time check — upper bound is F&O extended close (15:40)
    current_time = dt.time()
    return MARKET_OPEN <= current_time <= FNO_CLOSE


def is_trading_day(d: Optional[date] = None) -> bool:
    """Check if a given date is a trading day."""
    if d is None:
        d = datetime.now(IST).date()
    if d in SPECIAL_TRADING_DAYS:
        return True
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
    """Get the next expiry date for an index (handles weekly/monthly transitions)."""
    if from_date is None:
        from_date = datetime.now(IST).date()

    normalized = _normalize_symbol_alias(symbol)
    expiry_day_name = _resolve_weekly_expiry_day_name(normalized, from_date)

    # 2026 Regulatory constraint: Only NIFTY and SENSEX have weekly expiries. 
    # All others are monthly.
    if from_date.year >= 2026 and normalized not in ["NIFTY 50", "NIFTY", "SENSEX", "BSX"]:
        monthly_exp = get_monthly_expiry(from_date.year, from_date.month, weekday_name=expiry_day_name)
        if from_date > monthly_exp:
            if from_date.month == 12:
                monthly_exp = get_monthly_expiry(from_date.year + 1, 1, weekday_name=expiry_day_name)
            else:
                monthly_exp = get_monthly_expiry(from_date.year, from_date.month + 1, weekday_name=expiry_day_name)
        return monthly_exp

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
    while expiry in ALL_HOLIDAYS or (expiry.weekday() >= 5 and expiry not in SPECIAL_TRADING_DAYS):
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


def minutes_to_close(dt: Optional[datetime] = None, fno: bool = True) -> int:
    """Get minutes remaining to market close.

    Args:
        fno: If True, use F&O derivatives close (15:40).
             If False, use cash/equity close (15:30).
    """
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)

    close_time = FNO_CLOSE if fno else MARKET_CLOSE
    close_dt = dt.replace(hour=close_time.hour, minute=close_time.minute,
                          second=0, microsecond=0)
    if dt >= close_dt:
        return 0

    delta = close_dt - dt
    return int(delta.total_seconds() / 60)


def get_monthly_expiry(year: int, month: int, weekday_name: str = "Thursday") -> date:
    """Get the last weekday expiry of a month (holiday-adjusted)."""
    # Start from end of month
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)

    last_day = next_month_first - timedelta(days=1)

    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
    }
    target_weekday = day_map.get(weekday_name, 3)

    # Find last requested expiry weekday
    while last_day.weekday() != target_weekday:
        last_day -= timedelta(days=1)

    # If holiday, move to previous trading day
    while last_day in ALL_HOLIDAYS:
        last_day -= timedelta(days=1)

    return last_day


def _normalize_symbol_alias(symbol: str) -> str:
    """Normalize symbol aliases to canonical index names used in this module."""
    s = (symbol or "").strip().upper()
    alias_map = {
        "NIFTY": "NIFTY 50",
        "NIFTY50": "NIFTY 50",
        "BANKNIFTY": "NIFTY BANK",
        "FINNIFTY": "NIFTY FIN SERVICE",
        "BSX": "SENSEX",
    }
    return alias_map.get(s, symbol)


def _is_bse_derivative_symbol(symbol: str) -> bool:
    """Identify symbols that follow BSE expiry schedule."""
    normalized = _normalize_symbol_alias(symbol)
    return normalized in BSE_FRIDAY_EXPIRY_SYMBOLS or normalized == "BANKEX"


def _resolve_weekly_expiry_day_name(symbol: str, on_date: Optional[date] = None) -> str:
    """Resolve symbol expiry weekday with historical schedule transitions."""
    if on_date is None:
        on_date = datetime.now(IST).date()

    normalized = _normalize_symbol_alias(symbol)

    if normalized in BSE_FRIDAY_EXPIRY_SYMBOLS:
        return "Friday"
    if normalized == "BANKEX":
        return "Monday"

    # Historical BANKNIFTY schedule compatibility used by backtests.
    if normalized == "NIFTY BANK" and on_date.year < 2023:
        return "Thursday"

    # NSE Tuesday standardization from Sep 1, 2025 for NSE index and stock derivatives.
    if on_date >= NSE_TUESDAY_EXPIRY_CUTOVER:
        return "Tuesday"

    return WEEKLY_EXPIRY_DAYS.get(normalized, "Thursday")


def is_weekly_expiry_day(symbol: str, check_date: Optional[date] = None) -> bool:
    """Return True when check_date is the active weekly expiry for the symbol."""
    if check_date is None:
        check_date = datetime.now(IST).date()
    if not is_trading_day(check_date):
        return False

    normalized = _normalize_symbol_alias(symbol)
    try:
        return get_expiry_date(normalized, from_date=check_date) == check_date
    except Exception:
        return False


def is_monthly_expiry_day(check_date: Optional[date] = None, symbol: str = "NIFTY 50") -> bool:
    """Return True when check_date is the symbol's monthly expiry day."""
    if check_date is None:
        check_date = datetime.now(IST).date()
    if not is_trading_day(check_date):
        return False

    try:
        expiry_weekday = _resolve_weekly_expiry_day_name(symbol, check_date)
        return get_monthly_expiry(check_date.year, check_date.month, weekday_name=expiry_weekday) == check_date
    except Exception:
        return False


def is_expiry_thursday_session(symbol: str, check_date: Optional[date] = None) -> bool:
    """Legacy-named helper used by APEX; returns True on the symbol's active weekly expiry session."""
    if check_date is None:
        check_date = datetime.now(IST).date()

    normalized = _normalize_symbol_alias(symbol)
    return is_weekly_expiry_day(normalized, check_date)


def is_monthly_expiry_session(symbol: str, check_date: Optional[date] = None) -> bool:
    """Return True when check_date is the symbol's monthly expiry session."""
    if check_date is None:
        check_date = datetime.now(IST).date()

    normalized = _normalize_symbol_alias(symbol)
    return is_monthly_expiry_day(check_date, symbol=normalized)
