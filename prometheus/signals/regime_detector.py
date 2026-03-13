# ============================================================================
# PROMETHEUS — Signal Engine: Market Regime Detector
# ============================================================================
"""
Classifies market into regimes to select the right strategy module.

Based on the AMD (Accumulation-Manipulation-Distribution) cycle from Wyckoff,
quantified with measurable metrics instead of subjective chart reading.

Regimes:
  ACCUMULATION → Low volatility, range-bound, OI building → Mean reversion strategies
  MARKUP       → Trending up, expanding volatility → Trend following (buy calls)
  DISTRIBUTION → High volatility, range-bound at top → Hedged positions
  MARKDOWN     → Trending down, expanding volatility → Trend following (buy puts)
  VOLATILE     → VIX spike, news-driven → Volatility strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime classification with confidence metrics."""
    regime: MarketRegime
    confidence: float          # 0 to 1
    volatility_regime: str     # "low", "medium", "high", "extreme"
    trend_strength: float      # -1 (strong down) to +1 (strong up)
    mean_reversion_score: float  # 0 to 1 — higher = more mean-reverting
    recommended_strategies: list  # which strategy modules to activate
    details: str = ""


class RegimeDetector:
    """
    Multi-factor regime detection combining:
    1. Volatility analysis (realized vs implied, VIX level)
    2. Trend analysis (ADX, price vs moving averages)
    3. Volume analysis (volume expansion/contraction)
    4. Statistical tests (Hurst exponent for trend vs mean-reversion)
    """

    # VIX thresholds for Indian market (India VIX specific)
    VIX_LOW = 12
    VIX_MEDIUM = 18
    VIX_HIGH = 24
    VIX_EXTREME = 30

    def __init__(self, **kwargs):
        """
        Initialize with optional tunable parameters for sensitivity sweeps.

        Tunable parameters (kwargs):
            trend_strength_strong (default 0.4): Threshold for MARKUP/MARKDOWN detection
            trend_strength_sideways (default 0.3): Threshold for sideways classification
            vol_expanding_mult (default 1.2): Multiplier for vol_5 vs vol_20 (volatility expansion)
            volume_profile_mult (default 1.3): Multiplier for volume profile expansion check
            hurst_accumulation (default 0.45): Hurst threshold for ACCUMULATION regime
            trend_strength_weak (default 0.15): Threshold for weak bullish/bearish default
            sideways_direction (default 0.2): Threshold for sideways direction detection
        """
        self.trend_strength_strong = kwargs.get("trend_strength_strong", 0.4)
        self.trend_strength_sideways = kwargs.get("trend_strength_sideways", 0.3)
        self.vol_expanding_mult = kwargs.get("vol_expanding_mult", 1.2)
        self.volume_profile_mult = kwargs.get("volume_profile_mult", 1.3)
        self.hurst_accumulation = kwargs.get("hurst_accumulation", 0.45)
        self.trend_strength_weak = kwargs.get("trend_strength_weak", 0.15)
        self.sideways_direction = kwargs.get("sideways_direction", 0.2)

    def detect(
        self,
        df: pd.DataFrame,
        vix: Optional[float] = None,
        oi_metrics: Optional[Dict] = None
    ) -> RegimeState:
        """
        Detect current market regime from price data and supplementary signals.

        Args:
            df: OHLCV DataFrame (minimum 50 bars)
            vix: Current India VIX value
            oi_metrics: OI analysis metrics from OIAnalyzer
        """
        if len(df) < 50:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                volatility_regime="unknown",
                trend_strength=0.0,
                mean_reversion_score=0.5,
                recommended_strategies=["trend"],
                details="Insufficient data for regime detection"
            )

        # 1. Volatility Analysis
        vol_regime, realized_vol = self._analyze_volatility(df, vix)

        # 2. Trend Analysis
        trend_strength, trend_direction = self._analyze_trend(df)

        # 3. Mean Reversion Score (Hurst Exponent)
        hurst = self._calculate_hurst(df["close"].values)
        mean_reversion_score = max(0, 1 - hurst * 2) if hurst < 0.5 else 0

        # 4. Volume Pattern
        volume_expanding = self._analyze_volume_pattern(df)

        # 5. Classify Regime
        regime, confidence = self._classify_regime(
            vol_regime, trend_strength, trend_direction,
            hurst, volume_expanding, oi_metrics
        )

        # 6. Strategy Recommendations
        strategies = self._recommend_strategies(regime, vol_regime)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            volatility_regime=vol_regime,
            trend_strength=trend_strength,
            mean_reversion_score=mean_reversion_score,
            recommended_strategies=strategies,
            details=self._build_details(
                regime, vol_regime, trend_strength, hurst, realized_vol, vix
            )
        )

    def _analyze_volatility(
        self,
        df: pd.DataFrame,
        vix: Optional[float] = None
    ) -> Tuple[str, float]:
        """Classify volatility regime using realized vol and VIX."""
        # Realized volatility (20-day annualized)
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        realized_vol_20 = returns.tail(20).std() * np.sqrt(252) * 100

        # Also check short-term vol vs long-term vol
        realized_vol_5 = returns.tail(5).std() * np.sqrt(252) * 100
        vol_expanding = realized_vol_5 > realized_vol_20 * self.vol_expanding_mult

        # Use VIX if available, otherwise use realized vol
        reference_vol = vix if vix else realized_vol_20

        if reference_vol < self.VIX_LOW:
            regime = "low"
        elif reference_vol < self.VIX_MEDIUM:
            regime = "medium"
        elif reference_vol < self.VIX_HIGH:
            regime = "high"
        else:
            regime = "extreme"

        return regime, realized_vol_20

    def _analyze_trend(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Quantify trend strength using ADX-like calculation
        combined with price position relative to moving averages.

        Returns:
            trend_strength: -1 to +1
            direction: "up", "down", "sideways"
        """
        close = df["close"]

        # Price vs EMAs
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean()

        current_close = close.iloc[-1]
        current_ema20 = ema_20.iloc[-1]
        current_ema50 = ema_50.iloc[-1]

        # EMA alignment score
        ema_score = 0.0
        if current_close > current_ema20 > current_ema50:
            ema_score = 0.7  # Bullish alignment
        elif current_close < current_ema20 < current_ema50:
            ema_score = -0.7  # Bearish alignment
        elif current_close > current_ema20:
            ema_score = 0.3
        elif current_close < current_ema20:
            ema_score = -0.3

        # ADX-inspired directional strength
        high = df["high"]
        low = df["low"]
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        atr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1).rolling(14).mean()

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, np.nan))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(14).mean()

        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
        current_plus_di = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0
        current_minus_di = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0

        # ADX strength (0 to 1)
        adx_strength = min(current_adx / 50, 1.0)

        # Combine EMA and ADX
        if current_plus_di > current_minus_di:
            trend_strength = (ema_score + adx_strength) / 2
        else:
            trend_strength = (ema_score - adx_strength) / 2

        trend_strength = max(-1, min(1, trend_strength))

        if abs(trend_strength) < self.sideways_direction:
            direction = "sideways"
        elif trend_strength > 0:
            direction = "up"
        else:
            direction = "down"

        return trend_strength, direction

    def _calculate_hurst(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """
        Hurst Exponent — distinguishes trending from mean-reverting markets.

        H > 0.5: Trending (momentum works)
        H = 0.5: Random walk (no edge from trend/mean-reversion)
        H < 0.5: Mean-reverting (mean reversion works)
        """
        if len(prices) < max_lag * 3:
            return 0.5

        lags = range(2, max_lag + 1)
        rs_values = []

        for lag in lags:
            # Rescaled range calculation
            returns = np.diff(np.log(prices[-lag * 10:]))
            if len(returns) < lag:
                continue

            chunks = [returns[i:i + lag] for i in range(0, len(returns) - lag + 1, lag)]
            rs_for_lag = []

            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean_ret = np.mean(chunk)
                cumulative = np.cumsum(chunk - mean_ret)
                R = np.max(cumulative) - np.min(cumulative)
                S = np.std(chunk, ddof=1)
                if S > 0:
                    rs_for_lag.append(R / S)

            if rs_for_lag:
                rs_values.append((np.log(lag), np.log(np.mean(rs_for_lag))))

        if len(rs_values) < 3:
            return 0.5

        log_lags = [v[0] for v in rs_values]
        log_rs = [v[1] for v in rs_values]

        # Linear regression to get Hurst exponent
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]

        return max(0, min(1, hurst))

    def _analyze_volume_pattern(self, df: pd.DataFrame) -> bool:
        """Check if volume is expanding or contracting."""
        if "volume" not in df.columns:
            return False

        vol_5 = df["volume"].tail(5).mean()
        vol_20 = df["volume"].tail(20).mean()

        return vol_5 > vol_20 * self.volume_profile_mult  # Volume expanding

    def _classify_regime(
        self,
        vol_regime: str,
        trend_strength: float,
        trend_direction: str,
        hurst: float,
        volume_expanding: bool,
        oi_metrics: Optional[Dict] = None
    ) -> Tuple[MarketRegime, float]:
        """Classify market regime based on all factors."""

        # Extreme volatility overrides everything
        if vol_regime == "extreme":
            return MarketRegime.VOLATILE, 0.85

        # Strong uptrend
        if trend_strength > self.trend_strength_strong and hurst > 0.5:
            confidence = min(trend_strength + (hurst - 0.5), 1.0)
            return MarketRegime.MARKUP, confidence

        # Strong downtrend
        if trend_strength < -self.trend_strength_strong and hurst > 0.5:
            confidence = min(abs(trend_strength) + (hurst - 0.5), 1.0)
            return MarketRegime.MARKDOWN, confidence

        # Low volatility + sideways + mean-reverting
        if vol_regime in ("low", "medium") and abs(trend_strength) < self.trend_strength_sideways and hurst < self.hurst_accumulation:
            confidence = (1 - abs(trend_strength)) * (1 - hurst)
            return MarketRegime.ACCUMULATION, min(confidence, 0.8)

        # High volatility + sideways at top
        if vol_regime == "high" and abs(trend_strength) < self.trend_strength_sideways:
            return MarketRegime.DISTRIBUTION, 0.6

        # High volatility + trending
        if vol_regime == "high" and abs(trend_strength) > self.trend_strength_sideways:
            return MarketRegime.VOLATILE, 0.7

        # Default: use trend direction with lower confidence
        if trend_strength > self.trend_strength_weak:
            return MarketRegime.MARKUP, abs(trend_strength)
        elif trend_strength < -self.trend_strength_weak:
            return MarketRegime.MARKDOWN, abs(trend_strength)
        else:
            return MarketRegime.ACCUMULATION, 0.4

    def _recommend_strategies(
        self,
        regime: MarketRegime,
        vol_regime: str
    ) -> list:
        """Map regime to recommended strategy modules."""
        strategy_map = {
            MarketRegime.ACCUMULATION: ["mean_reversion", "expiry"],
            MarketRegime.MARKUP: ["trend"],
            MarketRegime.DISTRIBUTION: ["mean_reversion", "expiry"],
            MarketRegime.MARKDOWN: ["trend"],
            MarketRegime.VOLATILE: ["volatility"],
            MarketRegime.UNKNOWN: ["trend"],
        }
        return strategy_map.get(regime, ["trend"])

    def detect_fast(
        self,
        df: pd.DataFrame,
        recheck_every: int = 5,
    ) -> RegimeState:
        """
        Per-bar regime detection with caching for performance.

        Designed for bar-by-bar calling inside a backtest signal generator.
        Re-detects regime every `recheck_every` bars (default 5) and returns
        the cached result in between.

        Args:
            df: OHLCV DataFrame (full data_so_far — will use last 100 bars)
            recheck_every: Re-detect regime every N calls
        """
        self._cache_bar_count = getattr(self, '_cache_bar_count', 0) + 1
        if hasattr(self, '_cached_regime') and self._cache_bar_count < recheck_every:
            return self._cached_regime

        self._cache_bar_count = 0
        # Use last 100 bars for detection (50 minimum + buffer for rolling calcs)
        window = df.tail(100) if len(df) > 100 else df
        result = self.detect(window)
        self._cached_regime = result
        return result

    def reset_cache(self):
        """Reset the per-bar detection cache. Call before a new backtest run."""
        if hasattr(self, '_cached_regime'):
            del self._cached_regime
        self._cache_bar_count = 0

    def _build_details(
        self,
        regime: MarketRegime,
        vol_regime: str,
        trend: float,
        hurst: float,
        realized_vol: float,
        vix: Optional[float]
    ) -> str:
        """Build human-readable regime description."""
        parts = [
            f"Regime: {regime.value.upper()}",
            f"Volatility: {vol_regime} (realized={realized_vol:.1f}%)",
            f"Trend: {'UP' if trend > 0.15 else 'DOWN' if trend < -0.15 else 'SIDEWAYS'} ({trend:+.2f})",
            f"Hurst: {hurst:.2f} ({'trending' if hurst > 0.5 else 'mean-reverting'})",
        ]
        if vix:
            parts.append(f"India VIX: {vix:.1f}")
        return " | ".join(parts)
