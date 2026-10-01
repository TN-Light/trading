"""
Prometheus Target & Stop-Loss Calibrator.

Dynamically calibrates option profit targets and stop-losses based on:
  1. Signal Tier (Tier S, A, B, C, D)
  2. Higher Timeframe (HTF) Trend Alignment (BULLISH / BEARISH / NEUTRAL)
  3. Net Gamma Exposure (GEX / ZGL regime: LONG_GAMMA vs SHORT_GAMMA)
  4. Instrument Microstructure Noise Floors & Structural ORB Retest Levels

Market Microstructure Rationale:
  In a LONG_GAMMA regime (net_gex > 0), market makers/dealers hedge by selling as
  the underlying rises and buying as it drops. This institutional flow acts as a
  volatility damper, capping directional expansion. When combined with Tier C
  setups (lacking 1H trend alignment or low confluence), projecting multi-ATR
  continuation leads to premature reversals into the dealer hedging wall.
  Therefore, Tier C setups in positive gamma or counter-trend conditions are
  dynamically compressed into achievable scalp boundaries, and the dynamic retest
  expansion (sl_pts * 1.2) is strictly bypassed to prevent target inflation.

  Conversely, Tier S ("Perfect Storm") and Tier B ("Golden Setup" / "Institutional
  Trend Day") setups possess full 1H trend alignment and institutional volume,
  retaining full uncompressed multi-ATR targets and retest expansion.
"""

from typing import NamedTuple, Dict, Any, Optional, Tuple


class TargetCalibrationResult(NamedTuple):
    """Container for target and stop-loss calibration outputs."""
    target_gain_pts: float
    sl_pts: float
    tgt_price: float
    sl_price: float
    is_compressed: bool

    @property
    def breakeven_trigger_pts(self) -> float:
        """Breakeven trigger point (50% of target gain)."""
        return round(self.target_gain_pts * 0.50, 1)

    def __getitem__(self, item):
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
            raise KeyError(item)
        return tuple.__getitem__(self, item)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_gain_pts": self.target_gain_pts,
            "sl_pts": self.sl_pts,
            "tgt_price": self.tgt_price,
            "sl_price": self.sl_price,
            "is_compressed": self.is_compressed,
            "breakeven_trigger_pts": self.breakeven_trigger_pts,
        }


def calculate_structural_sl(
    symbol: str,
    spot_price: float,
    spot_atr: float,
    orb_high: Optional[float] = None,
    orb_low: Optional[float] = None,
    direction: str = "bullish",
    noise_floor: Optional[float] = None,
    atm_delta: float = 0.50,
) -> Tuple[float, Optional[float]]:
    """
    Calculate structural stop-loss in option points anchored to the ORB level with retest buffer.

    Returns:
        (structural_sl_pts, spot_sl_level)
    """
    sym_u = symbol.upper()
    if noise_floor is None:
        if "BANK" in sym_u:
            noise_floor = 35.0
        elif "SENSEX" in sym_u or "BSX" in sym_u:
            noise_floor = 30.0
        else:
            noise_floor = 10.0

    # Instrument Retest Buffer:
    # Minimum buffer below/above breakout line to absorb normal retest wicks
    if "BANK" in sym_u:
        retest_buffer = max(25.0, 0.30 * spot_atr)
    elif "SENSEX" in sym_u or "BSX" in sym_u:
        retest_buffer = max(35.0, 0.30 * spot_atr)
    elif "NIFTY" in sym_u:
        retest_buffer = max(12.0, 0.30 * spot_atr)
    else:
        retest_buffer = max(10.0, 0.30 * spot_atr)

    structural_sl_pts = noise_floor
    spot_sl_level = None
    if direction == "bullish" and orb_high and orb_high > 0:
        spot_sl_level = round(orb_high - retest_buffer, 2)
        spot_risk = max(spot_price - spot_sl_level, spot_atr * 0.5)
        structural_sl_pts = max(noise_floor, round(atm_delta * spot_risk, 1))
    elif direction == "bearish" and orb_low and orb_low > 0:
        spot_sl_level = round(orb_low + retest_buffer, 2)
        spot_risk = max(spot_sl_level - spot_price, spot_atr * 0.5)
        structural_sl_pts = max(noise_floor, round(atm_delta * spot_risk, 1))

    return structural_sl_pts, spot_sl_level


def calibrate_target_and_sl(
    symbol: str,
    spot_price: float,
    spot_atr: float = 0.0,
    opt_ltp: float = 0.0,
    edge_score: float = 5.0,
    tier: Optional[str] = "C",
    is_htf_aligned: bool = False,
    gamma_regime: Optional[str] = "NEUTRAL",
    net_gex: float = 0.0,
    structural_sl_pts: float = 0.0,
    is_low_vix: bool = False,
    noise_floor: Optional[float] = None,
    min_target: Optional[float] = None,
    orb_high: Optional[float] = None,
    orb_low: Optional[float] = None,
    direction: str = "bullish",
    atm_delta: float = 0.50,
) -> TargetCalibrationResult:
    """
    Calibrate option target gain and stop-loss boundaries.

    Interface Contract:
      calibrate_target_and_sl(symbol, spot_price, spot_atr, opt_ltp, edge_score, tier, is_htf_aligned, gamma_regime, net_gex, structural_sl_pts)
      Returns: (target_gain_pts: float, sl_pts: float, tgt_price: float, sl_price: float, is_compressed: bool)

      Contract Rules:
        - If tier == "C" and (not is_htf_aligned or gamma_regime == "LONG_GAMMA" or net_gex > 0):
          - Bank Nifty: 20.0 <= target_gain_pts <= 30.0
          - Sensex: 22.0 <= target_gain_pts <= 32.0
          - Nifty 50: 8.0 <= target_gain_pts <= 14.0
          - Retest expansion (sl_pts * 1.2) is strictly bypassed.
          - is_compressed = True
        - If tier in ("S", "B"):
          - Uncompressed multi-ATR target calculation with full retest expansion retained.
          - is_compressed = False
    """
    sym_u = symbol.upper()
    if spot_atr <= 0:
        if "BANK" in sym_u:
            spot_atr = 88.0
        elif "SENSEX" in sym_u or "BSX" in sym_u:
            spot_atr = 98.0
        elif "NIFTY" in sym_u:
            spot_atr = 25.0
        else:
            spot_atr = max(10.0, spot_price * 0.0035)

    eom = atm_delta * spot_atr  # Expected Option Move per 15M bar

    # Dynamic Quality Multiplier based on Signal Conviction Score
    if edge_score >= 8.0:
        quality_mult = 1.15
    elif edge_score >= 6.5:
        quality_mult = 0.85
    else:
        quality_mult = 0.65

    if is_low_vix:
        quality_mult *= 0.80

    # Default Instrument Noise Floors & Min Targets
    if "BANK" in sym_u:
        default_noise_floor = 35.0
        default_min_target = 25.0
    elif "SENSEX" in sym_u or "BSX" in sym_u:
        default_noise_floor = 30.0
        default_min_target = 22.0
    else:
        default_noise_floor = 10.0
        default_min_target = 8.0

    actual_noise_floor = noise_floor if noise_floor is not None else default_noise_floor
    actual_min_target = min_target if min_target is not None else default_min_target

    # Determine structural SL if not provided explicitly
    if structural_sl_pts <= 0 and (orb_high or orb_low):
        structural_sl_pts, _ = calculate_structural_sl(
            symbol=symbol,
            spot_price=spot_price,
            spot_atr=spot_atr,
            orb_high=orb_high,
            orb_low=orb_low,
            direction=direction,
            noise_floor=actual_noise_floor,
            atm_delta=atm_delta,
        )

    # ── Regime & Tier Aware Target Compression Logic ──
    tier_clean = tier.upper().replace("TIER", "").strip().strip("_") if tier else "C"
    is_tier_c = (tier_clean == "C")

    regime_str = str(gamma_regime).upper() if gamma_regime else "NEUTRAL"
    is_positive_gamma = (net_gex > 0) or ("LONG" in regime_str)
    is_counter_trend = not is_htf_aligned

    should_compress = is_tier_c and (is_counter_trend or is_positive_gamma)

    if should_compress:
        is_compressed = True
        if "BANK" in sym_u:
            comp_min, comp_max = 20.0, 30.0
            comp_scale = 0.55
        elif "SENSEX" in sym_u or "BSX" in sym_u:
            comp_min, comp_max = 22.0, 32.0
            comp_scale = 0.55
        elif "NIFTY" in sym_u:
            comp_min, comp_max = 8.0, 14.0
            comp_scale = 0.75
        else:
            comp_min, comp_max = 8.0, 14.0
            comp_scale = 0.55

        raw_target = round(eom * comp_scale, 1)
        target_gain_pts = min(comp_max, max(comp_min, raw_target))
        if opt_ltp > 0 and round(opt_ltp * 0.45, 1) >= comp_min:
            target_gain_pts = min(target_gain_pts, round(opt_ltp * 0.45, 1))
        target_gain_pts = min(comp_max, max(comp_min, target_gain_pts))
    else:
        is_compressed = False
        target_gain_pts = round(max(actual_min_target, eom * quality_mult), 1)
        if opt_ltp > 0:
            target_gain_pts = min(target_gain_pts, round(opt_ltp * 0.45, 1))

    # Base Stop Loss: use structural distance, noise floor, or EOM-based, whichever is larger
    sl_pts = max(structural_sl_pts, actual_noise_floor, round(0.55 * eom, 1))

    # Retest Breathing Room Protection:
    # If structural retest SL requires more room than target_gain_pts * 1.2,
    # dynamically scale target_gain_pts UP for uncompressed Tier S / Tier B runners.
    # CRITICALLY: Strictly BYPASS dynamic retest expansion for compressed Tier C signals!
    if not is_compressed:
        if sl_pts > round(target_gain_pts * 1.2, 1):
            target_gain_pts = max(target_gain_pts, round(sl_pts * 1.2, 1))
            if opt_ltp > 0:
                target_gain_pts = min(target_gain_pts, round(opt_ltp * 0.50, 1))

    # Hard Safety Ceiling: never risk > 35% total option premium
    if opt_ltp > 0:
        sl_pts = min(sl_pts, round(opt_ltp * 0.35, 1))

    if opt_ltp > 0:
        tgt_price = round(opt_ltp + target_gain_pts, 2)
        sl_price = round(max(1.0, opt_ltp - sl_pts), 2)
    else:
        tgt_price = round(target_gain_pts, 2)
        sl_price = round(max(0.0, -sl_pts), 2)

    return TargetCalibrationResult(
        target_gain_pts=target_gain_pts,
        sl_pts=sl_pts,
        tgt_price=tgt_price,
        sl_price=sl_price,
        is_compressed=is_compressed,
    )
