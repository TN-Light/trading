# ============================================================================
# PROMETHEUS — Signal Engine: Technical Analysis (Institutional Grade)
# ============================================================================
"""
Technical indicators prioritized by institutional utility:
  Tier 1 (0.80-0.85): Volume Profile, VWAP, OI Analysis, Liquidity Sweeps
  Tier 2 (0.60-0.70): FVG/Imbalance detection, Fibonacci OTE
  Tier 3 (0.20-0.40): RSI divergence, MACD (secondary only)

No indicator is used standalone — all feed into the signal fusion model.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field


@dataclass
class TechnicalSignal:
    """A single technical signal with metadata."""
    name: str
    direction: str       # "bullish", "bearish", "neutral"
    strength: float      # 0 to 1
    timeframe: str
    price_level: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Tier 1: Institutional-Grade Indicators
# ---------------------------------------------------------------------------

def calculate_vwap(df: pd.DataFrame, anchor: str = "day") -> pd.DataFrame:
    """
    Volume Weighted Average Price — institutional benchmark.

    VWAP is the most important intraday reference level because institutional
    algorithms use it as execution benchmark. Price above VWAP = bullish,
    below = bearish. Standard deviation bands act as dynamic S/R.
    """
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()

    df["vwap"] = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)

    # VWAP standard deviation bands
    df["vwap_sq"] = ((typical_price ** 2) * df["volume"]).cumsum() / cumulative_vol.replace(0, np.nan)
    df["vwap_std"] = np.sqrt(np.maximum(df["vwap_sq"] - df["vwap"] ** 2, 0))
    df["vwap_upper_1"] = df["vwap"] + df["vwap_std"]
    df["vwap_lower_1"] = df["vwap"] - df["vwap_std"]
    df["vwap_upper_2"] = df["vwap"] + 2 * df["vwap_std"]
    df["vwap_lower_2"] = df["vwap"] - 2 * df["vwap_std"]

    return df


def calculate_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session-anchored VWAP — resets at market open each trading day.

    For intraday bars, VWAP must reset at session start because institutional
    algorithms benchmark against the session VWAP, not a multi-day cumulative.
    Returns the same columns as calculate_vwap(): vwap, vwap_upper/lower_1/2.
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["_date"] = ts.dt.date

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["_tp_vol"] = typical_price * df["volume"]
    df["_tp2_vol"] = (typical_price ** 2) * df["volume"]

    # Cumulative sums within each trading session (day)
    df["_cum_tp_vol"] = df.groupby("_date")["_tp_vol"].cumsum()
    df["_cum_tp2_vol"] = df.groupby("_date")["_tp2_vol"].cumsum()
    df["_cum_vol"] = df.groupby("_date")["volume"].cumsum()

    safe_vol = df["_cum_vol"].replace(0, np.nan)
    df["vwap"] = df["_cum_tp_vol"] / safe_vol
    vwap_sq = df["_cum_tp2_vol"] / safe_vol
    df["vwap_std"] = np.sqrt(np.maximum(vwap_sq - df["vwap"] ** 2, 0))

    df["vwap_upper_1"] = df["vwap"] + df["vwap_std"]
    df["vwap_lower_1"] = df["vwap"] - df["vwap_std"]
    df["vwap_upper_2"] = df["vwap"] + 2 * df["vwap_std"]
    df["vwap_lower_2"] = df["vwap"] - 2 * df["vwap_std"]

    df.drop(columns=["_date", "_tp_vol", "_tp2_vol", "_cum_tp_vol", "_cum_tp2_vol", "_cum_vol"], inplace=True)
    return df


def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    lookback: int = 20
) -> Dict:
    """
    Volume Profile — shows where the most volume traded at each price level.

    Key levels:
    - POC (Point of Control): Price with highest volume — strongest S/R
    - Value Area High/Low: 70% of volume traded within this range
    - HVN (High Volume Node): Price levels with heavy trading — S/R zones
    - LVN (Low Volume Node): Price levels with light trading — price travels fast through these
    """
    recent = df.tail(lookback)
    if recent.empty:
        return {}

    price_min = recent["low"].min()
    price_max = recent["high"].max()
    bin_size = (price_max - price_min) / num_bins

    if bin_size == 0:
        return {}

    # Create price bins
    bins = np.linspace(price_min, price_max, num_bins + 1)
    volume_at_price = np.zeros(num_bins)

    for _, row in recent.iterrows():
        # Distribute volume across the candle's range
        low_bin = int((row["low"] - price_min) / bin_size)
        high_bin = int((row["high"] - price_min) / bin_size)
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))

        if low_bin == high_bin:
            volume_at_price[low_bin] += row["volume"]
        else:
            bins_in_range = high_bin - low_bin + 1
            vol_per_bin = row["volume"] / bins_in_range
            for b in range(low_bin, high_bin + 1):
                volume_at_price[b] += vol_per_bin

    # Point of Control
    poc_idx = np.argmax(volume_at_price)
    poc_price = bins[poc_idx] + bin_size / 2

    # Value Area (70% of total volume)
    total_vol = volume_at_price.sum()
    target_vol = total_vol * 0.70
    sorted_indices = np.argsort(volume_at_price)[::-1]
    cumulative = 0
    va_bins = []
    for idx in sorted_indices:
        cumulative += volume_at_price[idx]
        va_bins.append(idx)
        if cumulative >= target_vol:
            break

    va_high = bins[max(va_bins)] + bin_size
    va_low = bins[min(va_bins)]

    # High/Low Volume Nodes
    mean_vol = volume_at_price.mean()
    hvn = [bins[i] + bin_size / 2 for i in range(num_bins) if volume_at_price[i] > mean_vol * 1.5]
    lvn = [bins[i] + bin_size / 2 for i in range(num_bins) if 0 < volume_at_price[i] < mean_vol * 0.5]

    return {
        "poc": round(poc_price, 2),
        "value_area_high": round(va_high, 2),
        "value_area_low": round(va_low, 2),
        "hvn": [round(p, 2) for p in hvn],
        "lvn": [round(p, 2) for p in lvn],
        "bin_prices": (bins[:-1] + bin_size / 2).tolist(),
        "bin_volumes": volume_at_price.tolist(),
    }


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold_pct: float = 0.002
) -> List[Dict]:
    """
    Detect liquidity sweeps — when price briefly breaks a swing high/low
    then reverses. This is institutional stop-hunting behavior.

    A sweep occurs when:
    1. Price breaks above a recent swing high (sweeps sell stops)
    2. Closes back below it in the same or next candle
    3. This traps breakout buyers and fuels the reversal
    """
    sweeps = []
    if len(df) < lookback + 5:
        return sweeps

    recent = df.tail(lookback + 5).reset_index(drop=True)

    # Find swing highs and lows
    swing_highs = []
    swing_lows = []

    for i in range(2, len(recent) - 2):
        if recent["high"].iloc[i] > recent["high"].iloc[i-1] and \
           recent["high"].iloc[i] > recent["high"].iloc[i-2] and \
           recent["high"].iloc[i] > recent["high"].iloc[i+1] and \
           recent["high"].iloc[i] > recent["high"].iloc[i+2]:
            swing_highs.append((i, recent["high"].iloc[i]))

        if recent["low"].iloc[i] < recent["low"].iloc[i-1] and \
           recent["low"].iloc[i] < recent["low"].iloc[i-2] and \
           recent["low"].iloc[i] < recent["low"].iloc[i+1] and \
           recent["low"].iloc[i] < recent["low"].iloc[i+2]:
            swing_lows.append((i, recent["low"].iloc[i]))

    # Check for sweeps of swing highs (bearish signal)
    for idx, swing_high in swing_highs:
        for j in range(idx + 1, min(idx + 6, len(recent))):
            if recent["high"].iloc[j] > swing_high:
                # Price swept above swing high
                if recent["close"].iloc[j] < swing_high:
                    # Closed back below — bearish sweep
                    sweeps.append({
                        "type": "bearish_sweep",
                        "level": swing_high,
                        "sweep_candle_idx": j,
                        "timestamp": recent["timestamp"].iloc[j] if "timestamp" in recent.columns else j,
                        "strength": min(abs(recent["high"].iloc[j] - swing_high) / swing_high / threshold_pct, 1.0)
                    })
                break

    # Check for sweeps of swing lows (bullish signal)
    for idx, swing_low in swing_lows:
        for j in range(idx + 1, min(idx + 6, len(recent))):
            if recent["low"].iloc[j] < swing_low:
                if recent["close"].iloc[j] > swing_low:
                    # Closed back above — bullish sweep
                    sweeps.append({
                        "type": "bullish_sweep",
                        "level": swing_low,
                        "sweep_candle_idx": j,
                        "timestamp": recent["timestamp"].iloc[j] if "timestamp" in recent.columns else j,
                        "strength": min(abs(swing_low - recent["low"].iloc[j]) / swing_low / threshold_pct, 1.0)
                    })
                break

    return sweeps


# ---------------------------------------------------------------------------
# Tier 2: Structural Analysis (SMC-Derived, Quantified)
# ---------------------------------------------------------------------------

def detect_fair_value_gaps(
    df: pd.DataFrame,
    min_gap_pct: float = 0.001
) -> List[Dict]:
    """
    Fair Value Gap (FVG) detection — price imbalances that tend to get filled.

    An FVG exists when:
    - Bullish FVG: candle[i-1].high < candle[i+1].low (gap up imbalance)
    - Bearish FVG: candle[i-1].low > candle[i+1].high (gap down imbalance)

    These are zones where aggressive buying/selling created an imbalance
    that price often revisits (fills) before continuing.
    """
    fvgs = []
    if len(df) < 3:
        return fvgs

    for i in range(1, len(df) - 1):
        prev_high = df["high"].iloc[i - 1]
        next_low = df["low"].iloc[i + 1]
        prev_low = df["low"].iloc[i - 1]
        next_high = df["high"].iloc[i + 1]
        current_close = df["close"].iloc[i]

        # Bullish FVG: gap between prev candle high and next candle low
        if next_low > prev_high:
            gap_size = next_low - prev_high
            gap_pct = gap_size / current_close
            if gap_pct >= min_gap_pct:
                fvgs.append({
                    "type": "bullish_fvg",
                    "top": next_low,
                    "bottom": prev_high,
                    "midpoint": (next_low + prev_high) / 2,
                    "gap_size": gap_size,
                    "gap_pct": round(gap_pct * 100, 3),
                    "candle_idx": i,
                    "timestamp": df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
                    "filled": False,
                })

        # Bearish FVG
        if next_high < prev_low:
            gap_size = prev_low - next_high
            gap_pct = gap_size / current_close
            if gap_pct >= min_gap_pct:
                fvgs.append({
                    "type": "bearish_fvg",
                    "top": prev_low,
                    "bottom": next_high,
                    "midpoint": (prev_low + next_high) / 2,
                    "gap_size": gap_size,
                    "gap_pct": round(gap_pct * 100, 3),
                    "candle_idx": i,
                    "timestamp": df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
                    "filled": False,
                })

    # Check if FVGs have been filled by subsequent price action
    for fvg in fvgs:
        fill_start = fvg["candle_idx"] + 2
        for j in range(fill_start, len(df)):
            if fvg["type"] == "bullish_fvg" and df["low"].iloc[j] <= fvg["bottom"]:
                fvg["filled"] = True
                break
            elif fvg["type"] == "bearish_fvg" and df["high"].iloc[j] >= fvg["top"]:
                fvg["filled"] = True
                break

    return fvgs


def fibonacci_ote_levels(
    swing_high: float,
    swing_low: float,
    direction: str = "bullish"
) -> Dict[str, float]:
    """
    Fibonacci retracement levels with OTE (Optimal Trade Entry) zone.

    The 0.618-0.786 zone (OTE) is where institutional algorithms tend to
    enter on pullbacks. This has statistical backing from market microstructure
    research — mean reversion tends to occur at these levels due to
    institutional order clustering.
    """
    diff = swing_high - swing_low

    if direction == "bullish":
        # Retracement from high to low (buying on pullback)
        levels = {
            "0.236": swing_high - 0.236 * diff,
            "0.382": swing_high - 0.382 * diff,
            "0.500": swing_high - 0.500 * diff,
            "0.618": swing_high - 0.618 * diff,  # OTE Start
            "0.705": swing_high - 0.705 * diff,  # OTE Sweet Spot
            "0.786": swing_high - 0.786 * diff,  # OTE End
            "swing_low": swing_low,
            "swing_high": swing_high,
            "ote_zone_top": swing_high - 0.618 * diff,
            "ote_zone_bottom": swing_high - 0.786 * diff,
        }
    else:
        # Retracement from low to high (selling on rally)
        levels = {
            "0.236": swing_low + 0.236 * diff,
            "0.382": swing_low + 0.382 * diff,
            "0.500": swing_low + 0.500 * diff,
            "0.618": swing_low + 0.618 * diff,
            "0.705": swing_low + 0.705 * diff,
            "0.786": swing_low + 0.786 * diff,
            "swing_low": swing_low,
            "swing_high": swing_high,
            "ote_zone_top": swing_low + 0.786 * diff,
            "ote_zone_bottom": swing_low + 0.618 * diff,
        }

    return {k: round(v, 2) for k, v in levels.items()}


# ---------------------------------------------------------------------------
# Tier 3: Classical Indicators (Secondary Confluence Only)
# ---------------------------------------------------------------------------

def calculate_rsi(data, period: int = 14) -> pd.Series:
    """RSI — used ONLY for divergence detection, never standalone."""
    # Accept either a Series (close prices) or DataFrame (extract close)
    if isinstance(data, pd.DataFrame):
        series = data["close"]
    else:
        series = data
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = 30
) -> Optional[Dict]:
    """
    RSI Divergence — the ONE context where RSI has genuine statistical edge.

    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high

    This works because it signals momentum exhaustion before price reversal.
    """
    if len(df) < lookback + rsi_period:
        return None

    df = df.copy()
    df["rsi"] = calculate_rsi(df["close"], rsi_period)
    recent = df.tail(lookback).reset_index(drop=True)

    # Find swing lows in price and RSI
    price_lows = []
    rsi_lows = []
    price_highs = []
    rsi_highs = []

    for i in range(2, len(recent) - 2):
        # Swing lows
        if recent["low"].iloc[i] < recent["low"].iloc[i-1] and \
           recent["low"].iloc[i] < recent["low"].iloc[i+1]:
            price_lows.append((i, recent["low"].iloc[i]))
            rsi_lows.append((i, recent["rsi"].iloc[i]))

        # Swing highs
        if recent["high"].iloc[i] > recent["high"].iloc[i-1] and \
           recent["high"].iloc[i] > recent["high"].iloc[i+1]:
            price_highs.append((i, recent["high"].iloc[i]))
            rsi_highs.append((i, recent["rsi"].iloc[i]))

    # Bullish divergence: lower low in price, higher low in RSI
    if len(price_lows) >= 2:
        p1, p2 = price_lows[-2], price_lows[-1]
        r1, r2 = rsi_lows[-2], rsi_lows[-1]
        if p2[1] < p1[1] and r2[1] > r1[1]:
            return {
                "type": "bullish_divergence",
                "direction": "bullish",
                "strength": min(abs(r2[1] - r1[1]) / 20, 1.0),
                "price_level": p2[1],
                "rsi_at_low": r2[1],
            }

    # Bearish divergence: higher high in price, lower high in RSI
    if len(price_highs) >= 2:
        p1, p2 = price_highs[-2], price_highs[-1]
        r1, r2 = rsi_highs[-2], rsi_highs[-1]
        if p2[1] > p1[1] and r2[1] < r1[1]:
            return {
                "type": "bearish_divergence",
                "direction": "bearish",
                "strength": min(abs(r1[1] - r2[1]) / 20, 1.0),
                "price_level": p2[1],
                "rsi_at_high": r2[1],
            }

    return None


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — used for dynamic stop-loss calculation."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Supertrend indicator — useful as a trailing stop mechanism.
    Cleaner than moving averages for trend following.
    """
    df = df.copy()
    atr = calculate_atr(df, period)

    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = max(lower_band.iloc[i],
                                      supertrend.iloc[i-1] if direction.iloc[i-1] == 1 else lower_band.iloc[i])
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i],
                                      supertrend.iloc[i-1] if direction.iloc[i-1] == -1 else upper_band.iloc[i])

    df["supertrend"] = supertrend
    df["supertrend_direction"] = direction  # 1 = bullish, -1 = bearish

    return df
