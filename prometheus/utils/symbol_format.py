"""
Centralized instrument symbol formatting utilities.

The Kite / NSE API tradingsymbol is a strict continuous uppercase string with a
2-digit year code, no spaces, ending in CE/PE. Note that "NIFTY 50" the index
is always just ``NIFTY`` in trade symbols (the "50" is purely a display name
and is never part of any instrument tradingsymbol on Kite or the NSE feed).

Two distinct surface formats are required:

1. ``api_tradingsymbol`` — what gets sent to broker APIs (Kite Connect,
   Angel One SmartAPI), stored in DB rows, and used as the key for live quote
   subscription / LTP lookup. Example: ``NIFTY26JUL24500CE`` (monthly) or
   ``NIFTY2672324500CE`` (weekly).

2. ``human_search_name`` — what gets pasted into the Kite mobile app's search
   bar and shown in Telegram alerts. The Kite app strips the 2-digit year and
   inserts spaces; for weekly contracts it uses an ordinal day suffix:

       Monthly : ``NIFTY JUL 24500 CE``
       Weekly  : ``NIFTY 23rd JUL 24500 CE``

The monthly vs weekly disambiguation is needed because the Kite app shows both
contracts in the same search results — the year-stripped display name loses
information unless we add the day for weeklies.

References:
- Kite Connect 3 instruments CSV: tradingsymbol is uppercase continuous
  e.g. ``NIFTY159500CE`` (year code included, expiry recorded in a separate
  column). See https://kite.trade/docs/connect/v3/market-quotes/
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Index display-name -> underlying mapping
# ---------------------------------------------------------------------------
# Note: "NIFTY 50" → "NIFTY" because the NSE/Kite tradingsymbol never includes
# the "50". This is purely a display-name vs trade-name normalization.
INDEX_TO_UNDERLYING = {
    "NIFTY 50": "NIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "NIFTY FIN SERVICE": "FINNIFTY",
    "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
    "NIFTY FIN NIFTY": "FINNIFTY",
    "SENSEX": "SENSEX",
    "BANKEX": "BANKEX",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "FINNIFTY": "FINNIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTY": "NIFTY",
}


def resolve_underlying(symbol: str) -> str:
    """Resolve the underlying contract prefix for any input symbol.

    Accepts display names ("NIFTY 50", "NIFTY BANK"), pure underlyings
    ("NIFTY", "BANKNIFTY"), or stock tickers ("ICICIBANK"). Returns the
    uppercase underlying used in the NSE tradingsymbol.

    Examples:
        resolve_underlying("NIFTY 50")          -> "NIFTY"
        resolve_underlying("NIFTY BANK")         -> "BANKNIFTY"
        resolve_underlying("NIFTY FIN SERVICE")  -> "FINNIFTY"
        resolve_underlying("ICICIBANK")          -> "ICICIBANK"
        resolve_underlying("SENSEX")             -> "SENSEX"
    """
    if not symbol:
        return ""

    sym = symbol.strip()
    # Exact-match shortcut for known display names
    if sym in INDEX_TO_UNDERLYING:
        return INDEX_TO_UNDERLYING[sym]

    su = sym.upper().replace(" ", "")

    # Index family heuristics — order matters (longest/most-specific first)
    if "MIDCAP" in su and "NIFTY" in su:
        return "MIDCPNIFTY"
    if "BANK" in su and "NIFTY" in su:
        return "BANKNIFTY"
    if "FIN" in su and "NIFTY" in su:
        return "FINNIFTY"
    if "SENSEX" in su:
        return "SENSEX"
    if "BANKEX" in su:
        return "BANKEX"
    if "NIFTY" in su:
        return "NIFTY"

    # Stock F&O — return the uppercased ticker with no spaces
    return su


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _parse_expiry(expiry_in) -> Optional[date]:
    """Coerce an expiry value (str, date, datetime) to a ``date`` object."""
    if expiry_in is None:
        return None
    if isinstance(expiry_in, datetime):
        return expiry_in.date()
    if isinstance(expiry_in, date):
        return expiry_in
    if isinstance(expiry_in, str):
        s = expiry_in.strip()
        # Allow "2026-07-23 00:00:00" or "2026-07-23"
        if not s or s.upper() == "WEEKLY":
            return None
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _is_monthly_expiry(d: date) -> bool:
    """True if ``d`` is the last occurrence of its weekday in its month.

    This is the standard NSE/BSE convention: a monthly options contract expires
    on the LAST occurrence of its weekday-of-the-month (Tuesday for NSE post
    Sep 2025, Thursday for BSE/SENSEX); everything else is treated as a weekly
    contract.
    """
    return (d + timedelta(days=7)).month != d.month


_MONTH_CODES = {
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
    7: "7", 8: "8", 9: "9", 10: "O", 11: "N", 12: "D",
}


def _ordinal_suffix(day: int) -> str:
    """Return the English ordinal suffix for ``day`` (1 -> 'st', 22 -> 'nd', ...)."""
    if 10 <= (day % 100) <= 20:
        return "th"
    last = day % 10
    if last == 1:
        return "st"
    if last == 2:
        return "nd"
    if last == 3:
        return "rd"
    return "th"


def _format_strike(strike) -> str:
    """Format a strike price for inclusion in a tradingsymbol.

    Whole numbers lose the decimal: ``24500.0 -> "24500"``; fractional
    strikes keep their value: ``73.5 -> "73.5"``.
    """
    try:
        s = float(strike)
    except (TypeError, ValueError):
        return str(strike) if strike is not None else ""
    if s == int(s):
        return str(int(s))
    return str(s)


# ---------------------------------------------------------------------------
# API tradingsymbol — what the broker receives
# ---------------------------------------------------------------------------
def api_tradingsymbol(
    symbol: str,
    expiry,
    strike,
    option_type: str,
) -> str:
    """Build the strict continuous uppercase tradingsymbol for broker APIs.

    Format:
        Monthly : ``{UNDERLYING}{YY}{MON}{STRIKE}{CE|PE}``   e.g. NIFTY26JUL24500CE
        Weekly  : ``{UNDERLYING}{YY}{M}{DD}{STRIKE}{CE|PE}``  e.g. NIFTY2672324500CE

    Args:
        symbol: display symbol ("NIFTY 50", "NIFTY BANK", "ICICIBANK", ...).
        expiry: expiry as ``str`` ("YYYY-MM-DD"), ``date``, or ``datetime``.
        strike: strike price.
        option_type: "CE" or "PE".
    """
    underlying = resolve_underlying(symbol)
    opt = (option_type or "").upper()
    if opt not in ("CE", "PE"):
        return ""

    d = _parse_expiry(expiry)
    if d is None or not underlying:
        return ""

    strike_str = _format_strike(strike)
    yy = d.strftime("%y")

    if _is_monthly_expiry(d):
        mon = d.strftime("%b").upper()  # JUL, AUG, ...
        return f"{underlying}{yy}{mon}{strike_str}{opt}"

    m_code = _MONTH_CODES.get(d.month, str(d.month))
    dd = d.strftime("%d")
    return f"{underlying}{yy}{m_code}{dd}{strike_str}{opt}"


# ---------------------------------------------------------------------------
# Human search name — for Kite search bar / Telegram alerts
# ---------------------------------------------------------------------------
def human_search_name(
    symbol: str,
    expiry,
    strike,
    option_type: str,
) -> str:
    """Build a Kite-app-friendly search name (year stripped, spaces inserted).

    Format:
        Monthly : ``NIFTY JUL 24500 CE``
        Weekly  : ``NIFTY 23rd JUL 24500 CE``   (ordinal day suffix on weekly)

    The Kite mobile app shows contract names without the 2-digit year code to
    save space. Monthly contracts are uniquely identified by month + strike,
    but weekly contracts collide across years/months on month+strike alone, so
    we add the ordinal day-of-month (e.g. ``23rd``) to disambiguate.

    Args:
        symbol: display symbol.
        expiry: expiry as ``str``/``date``/``datetime``.
        strike: strike price.
        option_type: "CE" or "PE".
    """
    underlying = resolve_underlying(symbol)
    opt = (option_type or "").upper()
    if opt not in ("CE", "PE"):
        return ""

    d = _parse_expiry(expiry)
    if d is None or not underlying:
        return ""

    strike_str = _format_strike(strike)
    mon = d.strftime("%b").upper()

    if _is_monthly_expiry(d):
        return f"{underlying} {mon} {strike_str} {opt}"

    day = d.day
    suffix = _ordinal_suffix(day)
    return f"{underlying} {day}{suffix} {mon} {strike_str} {opt}"


# ---------------------------------------------------------------------------
# Reverse parser — recover structure from an API tradingsymbol
# ---------------------------------------------------------------------------
def parse_api_tradingsymbol(tradingsymbol: str) -> Optional[dict]:
    """Reverse-parse a Kite/NSE tradingsymbol into its components.

    Returns a dict with keys: underlying, expiry (date), strike, option_type.
    Returns ``None`` if the symbol can't be parsed.

    Examples:
        parse_api_tradingsymbol("NIFTY26JUL24500CE")
            -> {underlying="NIFTY", expiry=date(2026,7,30), strike=24500.0,
                option_type="CE", is_monthly=True}
        parse_api_tradingsymbol("NIFTY2672324500CE")
            -> {underlying="NIFTY", expiry=date(2026,7,23), strike=24500.0,
                option_type="CE", is_monthly=False}
        parse_api_tradingsymbol("BANKNIFTY2661857600CE")
            -> {underlying="BANKNIFTY", expiry=date(2026,6,18), ...}
    """
    ts = (tradingsymbol or "").strip().upper()
    if not ts:
        return None

    # Strip option type
    if ts.endswith("CE"):
        opt_type = "CE"
        body = ts[:-2]
    elif ts.endswith("PE"):
        opt_type = "PE"
        body = ts[:-2]
    else:
        return None

    # Identify underlying by matching longest known prefix first.
    # Underlying is uppercase letters only (no digits); after stripping it,
    # the remaining string is "{date_code}{strike}" with no separator. We then
    # find the strike by walking from the right consuming digits (and one "."),
    # bounded left by the date code (which always starts with 2 digits = year).
    underlying = None
    for name in sorted(INDEX_TO_UNDERLYING.values(), key=len, reverse=True):
        if body.startswith(name) and name.isalpha():
            underlying = name
            break
    if not underlying:
        # Stock option: underlying is the alphabetic prefix only
        i = 0
        while i < len(body) and body[i].isalpha():
            i += 1
        underlying = body[:i]
        if not underlying:
            return None
        # Stock underlying can contain a "."? No — NSE stock symbols are
        # uppercase letters only. If we hit a digit at position 0 there's
        # no underlying to parse.
        if not underlying.isalpha():
            return None

    rest = body[len(underlying):]  # "{date_code}{strike}"
    # The date code is fixed-length 5 chars (YY + M + DD for weekly or
    # YY + 3-letter-month for monthly). Strike is everything after that.
    if len(rest) < 5:
        return None
    date_code = rest[:5]
    strike_str = rest[5:]
    try:
        strike = float(strike_str)
    except ValueError:
        return None
    if strike <= 0 or not strike_str:
        return None

    yy = int(date_code[:2])
    # Try monthly: chars [2:5] are a 3-letter month abbreviation
    mon_str = date_code[2:5]
    month_map_str = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                      "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    if mon_str in month_map_str and mon_str.isalpha():
        month = month_map_str[mon_str]
        # NOTE: monthly-format tradingsymbols encode the month but NOT the
        # specific expiry day-of-month. NSE restructured its derivatives
        # framework effective September 1, 2025 — all NSE index & stock F&O
        # contracts now expire on the LAST TUESDAY of the month. BSE
        # derivatives (SENSEX, BANKEX) keep the LAST THURSDAY schedule.
        # We delegate to ``indian_market.get_monthly_expiry`` which already has
        # this date-aware logic baked in (see _resolve_weekly_expiry_day_name
        # in indian_market.py). Callers needing the exact expiry for fresh
        # trades should pass it via the upstream signal — this round-trip is
        # only used to display names for stale DB rows where the original
        # expiry date isn't separately available.
        try:
            from prometheus.utils.indian_market import (
                get_monthly_expiry, _resolve_weekly_expiry_day_name,
            )
            weekday_name = _resolve_weekly_expiry_day_name(underlying)
            d = get_monthly_expiry(2000 + yy, month, weekday_name=weekday_name)
        except Exception:
            # Fallback: last Tuesday (NSE convention post Sep-2025)
            from calendar import monthrange
            last_day = monthrange(2000 + yy, month)[1]
            d = date(2000 + yy, month, last_day)
            while d.weekday() != 1:  # Tuesday
                d -= timedelta(days=1)
        return {
            "underlying": underlying,
            "expiry": d,
            "expiry_year_month": (2000 + yy, month),
            "strike": strike,
            "option_type": opt_type,
            "is_monthly": True,
        }

    # Try weekly: YY M DD format (5 chars)
    try:
        m_char = date_code[2]
        dd = int(date_code[3:5])
        m_map = {
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
            "7": 7, "8": 8, "9": 9, "O": 10, "N": 11, "D": 12,
        }
        month = m_map.get(m_char)
        if month is None or not date_code[3:5].isdigit():
            return None
        d = date(2000 + yy, month, dd)
        return {
            "underlying": underlying,
            "expiry": d,
            "strike": strike,
            "option_type": opt_type,
            "is_monthly": False,
        }
    except (ValueError, KeyError, IndexError):
        return None


def human_search_name_from_api_symbol(tradingsymbol: str) -> str:
    """Convert an API tradingsymbol directly to its Kite-app search name.

    Convenience wrapper that parses the tradingsymbol and rebuilds the display
    form. Useful when we already have an API symbol (e.g. from a DB row) but
    need to show the user a paste-friendly string.
    """
    parsed = parse_api_tradingsymbol(tradingsymbol)
    if not parsed:
        # Could be a display-name placeholder like "NIFTY 50 24200 CE" —
        # return as-is so the user still sees something readable.
        return tradingsymbol
    return human_search_name(
        parsed["underlying"],
        parsed["expiry"],
        parsed["strike"],
        parsed["option_type"],
    )
