# ============================================================================
# PROMETHEUS — Utility: Options Mathematics
# ============================================================================
"""
Black-Scholes pricing, Greeks calculation, IV computation.
All math needed for options analysis — no external dependencies beyond scipy/numpy.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple, Optional
from enum import Enum


class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"


def black_scholes_price(
    S: float,       # Spot price
    K: float,       # Strike price
    T: float,       # Time to expiry in years
    r: float,       # Risk-free rate (annualized)
    sigma: float,   # Implied volatility (annualized)
    option_type: OptionType = OptionType.CALL
) -> float:
    """
    Black-Scholes option pricing.
    Returns theoretical option price.
    """
    if T <= 0:
        # At expiry — intrinsic value only
        if option_type == OptionType.CALL:
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == OptionType.CALL:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return max(price, 0)


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL
) -> dict:
    """
    Calculate all Greeks for an option.

    Returns dict with: delta, gamma, theta, vega, rho
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
        itm = intrinsic > 0
        return {
            "delta": (1.0 if itm else 0.0) if option_type == OptionType.CALL else (-1.0 if itm else 0.0),
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    # Delta
    if option_type == OptionType.CALL:
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # Theta (per day — divide annual by 365)
    if option_type == OptionType.CALL:
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    # Vega (per 1% change in IV)
    vega = S * sqrt_T * pdf_d1 / 100

    # Rho (per 1% change in rate)
    if option_type == OptionType.CALL:
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "rho": round(rho, 4),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = OptionType.CALL,
    precision: float = 1e-6,
    max_vol: float = 5.0
) -> Optional[float]:
    """
    Calculate implied volatility using Brent's method.
    Returns IV as a decimal (e.g., 0.25 for 25%).
    Returns None if IV cannot be computed (e.g., deep ITM with no time value).
    """
    if T <= 0 or market_price <= 0:
        return None

    # Check bounds
    intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
    if market_price < intrinsic:
        return None

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 0.001, max_vol, xtol=precision)
        return round(iv, 6)
    except (ValueError, RuntimeError):
        return None


def iv_percentile(iv_series: np.ndarray, current_iv: float) -> float:
    """
    Calculate IV Percentile — what % of past IV readings are below current IV.
    Standard lookback: 252 trading days (1 year).
    """
    if len(iv_series) == 0:
        return 50.0
    count_below = np.sum(iv_series < current_iv)
    return round((count_below / len(iv_series)) * 100, 2)


def iv_rank(iv_series: np.ndarray, current_iv: float) -> float:
    """
    Calculate IV Rank — where current IV sits between 1-year high and low.
    Formula: (current - min) / (max - min) * 100
    """
    if len(iv_series) == 0:
        return 50.0
    iv_min = np.min(iv_series)
    iv_max = np.max(iv_series)
    if iv_max == iv_min:
        return 50.0
    return round((current_iv - iv_min) / (iv_max - iv_min) * 100, 2)


def max_pain(
    strikes: np.ndarray,
    call_oi: np.ndarray,
    put_oi: np.ndarray,
    spot: float
) -> float:
    """
    Calculate Max Pain — the strike price where total option buyer losses are maximized.
    This is where the market tends to gravitate toward on expiry.
    """
    total_pain = np.zeros(len(strikes))

    for i, strike in enumerate(strikes):
        # Call holders' pain at this expiry price
        call_pain = np.sum(call_oi * np.maximum(strikes - strike, 0))
        # Put holders' pain at this expiry price
        put_pain = np.sum(put_oi * np.maximum(strike - strikes, 0))
        total_pain[i] = call_pain + put_pain

    # Max pain = strike where total buyer pain is maximum
    return strikes[np.argmax(total_pain)]


def pcr_ratio(put_oi_total: float, call_oi_total: float) -> float:
    """
    Put-Call Ratio based on Open Interest.
    PCR > 1.2 = oversold / bullish (contrarian)
    PCR < 0.8 = overbought / bearish (contrarian)
    PCR 0.8-1.2 = neutral
    """
    if call_oi_total == 0:
        return 0.0
    return round(put_oi_total / call_oi_total, 4)


def calculate_payoff(
    spot_range: np.ndarray,
    positions: list,
) -> np.ndarray:
    """
    Calculate strategy payoff across a range of spot prices.

    positions: list of dicts, each with:
        - type: 'CE' or 'PE'
        - strike: float
        - action: 'buy' or 'sell'
        - premium: float (per unit)
        - lots: int
        - lot_size: int
    """
    total_payoff = np.zeros(len(spot_range))

    for pos in positions:
        strike = pos["strike"]
        premium = pos["premium"]
        lots = pos["lots"]
        lot_size = pos["lot_size"]
        quantity = lots * lot_size
        multiplier = 1 if pos["action"] == "buy" else -1

        if pos["type"] == "CE":
            intrinsic = np.maximum(spot_range - strike, 0)
        else:
            intrinsic = np.maximum(strike - spot_range, 0)

        payoff = multiplier * (intrinsic - premium) * quantity
        total_payoff += payoff

    return total_payoff


def get_implied_vol_at_strike(
    spot: float,
    strike: float,
    atm_sigma: float,
    skew_slope: float = -0.10,
) -> float:
    """
    Estimate implied volatility at a given strike using linear skew model.

    Volatility smile approximation:
        sigma(K) = atm_sigma - skew_slope * (K/S - 1)

    For ATM (K ≈ S), returns atm_sigma unchanged.
    For OTM puts (K < S), skew adds vol (smile effect).
    For OTM calls (K > S), skew slightly reduces vol.

    Args:
        spot: Current spot price
        strike: Option strike price
        atm_sigma: ATM implied volatility (annualized decimal, e.g. 0.20)
        skew_slope: Skew slope (negative = typical equity skew).
                    Default -0.10 calibrated for Indian indices.

    Returns:
        Adjusted implied volatility at the given strike.
    """
    if spot <= 0:
        return atm_sigma

    moneyness = (strike / spot) - 1.0  # 0 at ATM, negative for OTM puts, positive for OTM calls
    sigma_adj = atm_sigma - skew_slope * moneyness

    # Clamp to reasonable range
    sigma_adj = max(sigma_adj, 0.05)
    sigma_adj = min(sigma_adj, 1.0)

    return sigma_adj


def breakeven_points(
    spot_range: np.ndarray,
    payoff: np.ndarray
) -> list:
    """Find breakeven points where payoff crosses zero."""
    breakevens = []
    for i in range(1, len(payoff)):
        if payoff[i - 1] * payoff[i] < 0:
            # Linear interpolation
            x = spot_range[i - 1] + (0 - payoff[i - 1]) * (
                spot_range[i] - spot_range[i - 1]
            ) / (payoff[i] - payoff[i - 1])
            breakevens.append(round(x, 2))
    return breakevens
