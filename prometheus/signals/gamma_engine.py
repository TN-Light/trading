"""Institutional Gamma Exposure (GEX) and Zero Gamma Level (ZGL) Engine.

Computes dealer gamma profile across the active option chain:
  - Strike-level Black-Scholes Gamma (Γ)
  - Aggregate Call GEX and Put GEX
  - Net GEX (dealer delta-hedging pressure per 1% spot move)
  - Zero Gamma Level (ZGL / Gamma Flip Point where market transitions between
    mean-reverting long-gamma regime and trending/breakout short-gamma regime)

Deployed strictly as passive background telemetry (Option A) — logs and tracks
gamma profile metrics without gating live capital until empirically verified.
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np
import pandas as pd
from loguru import logger

from prometheus.utils.indian_market import get_lot_size, get_strike_interval


def _standard_normal_pdf(x: float) -> float:
    """Standard normal probability density function: phi(x) = (1/sqrt(2pi)) * exp(-0.5 * x^2)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def calculate_black_scholes_gamma(
    spot: float,
    strike: float,
    dte: float,
    sigma: float,
    r: float = 0.07,
) -> float:
    """Calculate Black-Scholes option Gamma: d(Delta)/d(Spot) = phi(d1) / (S * sigma * sqrt(T)).

    Gamma is identical for European calls and puts.
    """
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return 0.0

    T = max(dte, 0.25) / 365.0
    vol_sqrt_t = sigma * math.sqrt(T)
    if vol_sqrt_t <= 1e-9:
        return 0.0

    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    phi_d1 = _standard_normal_pdf(d1)
    gamma = phi_d1 / (spot * vol_sqrt_t)
    return gamma


class GammaEngine:
    """Calculates Net GEX and Zero Gamma Level (ZGL) from options chain."""

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate

    def calculate_gex(
        self,
        chain_df: pd.DataFrame,
        spot_price: float,
        symbol: str,
        dte: float = 2.0,
        default_iv: float = 0.15,
    ) -> Dict[str, Any]:
        """Compute Net Gamma Exposure (GEX) and Zero Gamma Level (ZGL).

        Args:
            chain_df: Option chain DataFrame with columns:
                      'strike_price', 'option_type' (CE/PE), 'open_interest',
                      optional 'implied_volatility' or 'iv'.
            spot_price: Current underlying index spot level.
            symbol: Index symbol ("NIFTY 50", "NIFTY BANK", "SENSEX").
            dte: Days to expiry for the near contract.
            default_iv: Fallback annualized implied volatility.

        Returns:
            Dictionary with:
              - net_gex: Raw net rupee gamma per 1% move
              - net_gex_cr: Net GEX in Crores (INR)
              - call_gex_cr: Call GEX in Crores
              - put_gex_cr: Put GEX in Crores
              - zgl: Zero Gamma Level (spot price where GEX flips)
              - gamma_regime: "LONG_GAMMA" (volatility dampening / mean reversion) or
                              "SHORT_GAMMA" (volatility acceleration / breakout)
        """
        empty_result = {
            "net_gex": 0.0,
            "net_gex_cr": 0.0,
            "call_gex_cr": 0.0,
            "put_gex_cr": 0.0,
            "zgl": round(spot_price, 2) if spot_price > 0 else 0.0,
            "gamma_regime": "NEUTRAL",
        }

        if chain_df is None or chain_df.empty or spot_price <= 0:
            return empty_result

        try:
            lot_size = get_lot_size(symbol)
            if lot_size <= 0:
                lot_size = 15 if "BANK" in symbol else 50
        except Exception:
            lot_size = 50

        # Standardize column names
        df = chain_df.copy()
        col_map = {c.lower(): c for c in df.columns}

        strike_col = col_map.get("strike_price") or col_map.get("strike")
        type_col = col_map.get("option_type") or col_map.get("type") or col_map.get("instrument_type")
        oi_col = col_map.get("open_interest") or col_map.get("oi")
        iv_col = col_map.get("implied_volatility") or col_map.get("iv")

        if not strike_col or not type_col or not oi_col:
            return empty_result

        strikes_data: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            try:
                stk = float(row[strike_col])
                otype = str(row[type_col]).upper().strip()
                oi = float(row[oi_col] or 0.0)
                if oi <= 0 or stk <= 0:
                    continue

                # Filter to relevant strike envelope (within ±10% of spot)
                if abs(stk - spot_price) / spot_price > 0.12:
                    continue

                iv = float(row[iv_col]) if iv_col and pd.notna(row.get(iv_col)) else default_iv
                if iv > 1.0:  # e.g. given as percentage like 14.5%
                    iv = iv / 100.0
                if iv <= 0.02 or iv > 1.50:
                    iv = default_iv

                strikes_data.append({
                    "strike": stk,
                    "type": "CE" if "CE" in otype or "CALL" in otype else "PE",
                    "oi": oi,
                    "iv": iv,
                })
            except Exception:
                continue

        if not strikes_data:
            return empty_result

        # Compute GEX at current spot
        call_gex_tot = 0.0
        put_gex_tot = 0.0

        for item in strikes_data:
            gamma = calculate_black_scholes_gamma(
                spot=spot_price,
                strike=item["strike"],
                dte=dte,
                sigma=item["iv"],
                r=self.r,
            )
            # Dollar/Rupee Gamma per 1% spot move:
            # GEX = Gamma * OI * LotSize * Spot * 0.01
            dollar_gamma = gamma * item["oi"] * lot_size * spot_price * 0.01

            if item["type"] == "CE":
                call_gex_tot += dollar_gamma
            else:
                # Dealers short puts to retail / long hedgers -> dealer put gamma is negative
                put_gex_tot -= dollar_gamma

        net_gex = call_gex_tot + put_gex_tot
        net_gex_cr = round(net_gex / 1e7, 2)
        call_gex_cr = round(call_gex_tot / 1e7, 2)
        put_gex_cr = round(put_gex_tot / 1e7, 2)

        # ── ZERO GAMMA LEVEL (ZGL) SOLVER ──
        # Scan hypothetical spot prices around current spot to find where Net GEX crosses zero
        try:
            interval = get_strike_interval(symbol)
        except Exception:
            interval = 50.0

        min_scan = spot_price * 0.92
        max_scan = spot_price * 1.08
        steps = 40
        grid = np.linspace(min_scan, max_scan, steps)

        grid_gex: List[Tuple[float, float]] = []
        for s_eval in grid:
            cg = 0.0
            pg = 0.0
            for item in strikes_data:
                g = calculate_black_scholes_gamma(
                    spot=s_eval,
                    strike=item["strike"],
                    dte=dte,
                    sigma=item["iv"],
                    r=self.r,
                )
                dg = g * item["oi"] * lot_size * s_eval * 0.01
                if item["type"] == "CE":
                    cg += dg
                else:
                    pg -= dg
            grid_gex.append((float(s_eval), float(cg + pg)))

        # Find zero crossing
        zgl = spot_price
        for i in range(len(grid_gex) - 1):
            s1, g1 = grid_gex[i]
            s2, g2 = grid_gex[i + 1]
            if (g1 <= 0 and g2 >= 0) or (g1 >= 0 and g2 <= 0):
                # Linear interpolation for zero crossing
                if abs(g2 - g1) > 1e-6:
                    zgl = s1 - g1 * (s2 - s1) / (g2 - g1)
                else:
                    zgl = (s1 + s2) / 2.0
                break

        regime = "LONG_GAMMA" if net_gex >= 0 else "SHORT_GAMMA"

        return {
            "net_gex": round(net_gex, 2),
            "net_gex_cr": net_gex_cr,
            "call_gex_cr": call_gex_cr,
            "put_gex_cr": put_gex_cr,
            "zgl": round(zgl, 2),
            "gamma_regime": regime,
        }
