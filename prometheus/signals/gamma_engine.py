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
    if (
        spot is None or strike is None or dte is None or sigma is None
        or math.isnan(spot) or math.isnan(strike) or math.isnan(dte) or math.isnan(sigma)
        or math.isinf(spot) or math.isinf(strike) or math.isinf(dte) or math.isinf(sigma)
        or spot <= 0 or strike <= 0 or sigma <= 0
    ):
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
        symbol: str = "NIFTY 50",
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
            "zgl": None,
            "has_zgl": False,
            "gamma_regime": "NEUTRAL",
        }

        if (
            chain_df is None
            or chain_df.empty
            or spot_price is None
            or math.isnan(spot_price)
            or math.isinf(spot_price)
            or spot_price <= 0
        ):
            return empty_result

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
                stk_raw = row[strike_col]
                oi_raw = row[oi_col]
                if pd.isna(stk_raw) or pd.isna(oi_raw):
                    continue
                stk = float(stk_raw)
                oi = float(oi_raw)
                if math.isnan(stk) or math.isinf(stk) or math.isnan(oi) or math.isinf(oi):
                    continue
                if oi <= 0 or stk <= 0:
                    continue

                otype = str(row[type_col]).upper().strip()

                # Filter to relevant strike envelope (within ±10% of spot)
                if abs(stk - spot_price) / spot_price > 0.12:
                    continue

                iv = default_iv
                if iv_col and pd.notna(row.get(iv_col)):
                    try:
                        iv_val = float(row[iv_col])
                        if not math.isnan(iv_val) and not math.isinf(iv_val):
                            iv = iv_val
                    except Exception:
                        iv = default_iv

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

        def _compute_net_gex(s: float) -> float:
            cg = 0.0
            pg = 0.0
            factor = (s ** 2) * 0.01
            for itm in strikes_data:
                g = calculate_black_scholes_gamma(
                    spot=s,
                    strike=itm["strike"],
                    dte=dte,
                    sigma=itm["iv"],
                    r=self.r,
                )
                val = itm["oi"] * g * factor
                if itm["type"] == "CE":
                    cg += val
                else:
                    pg -= val
            return cg + pg

        # Compute GEX at current spot:
        # Rupee Notional Gamma per 1% spot move: GEX_INR = Q * Gamma * S^2 * 0.01
        # Angel One SmartAPI opnInterest (Q) is already in total underlying shares.
        call_gex_tot = 0.0
        put_gex_tot = 0.0
        s_factor = (spot_price ** 2) * 0.01

        for item in strikes_data:
            gamma = calculate_black_scholes_gamma(
                spot=spot_price,
                strike=item["strike"],
                dte=dte,
                sigma=item["iv"],
                r=self.r,
            )
            rupee_gamma = item["oi"] * gamma * s_factor

            if item["type"] == "CE":
                call_gex_tot += rupee_gamma
            else:
                # Dealers short puts to retail / long hedgers -> dealer put gamma is negative
                put_gex_tot -= rupee_gamma

        net_gex = call_gex_tot + put_gex_tot
        net_gex_cr = round(net_gex / 1e7, 2)
        call_gex_cr = round(call_gex_tot / 1e7, 2)
        put_gex_cr = round(put_gex_tot / 1e7, 2)

        # ── ZERO GAMMA LEVEL (ZGL) SOLVER ──
        # Check if the chain is unipolar (must have both calls and puts to cross zero)
        has_calls = any(itm["type"] == "CE" for itm in strikes_data)
        has_puts = any(itm["type"] == "PE" for itm in strikes_data)

        found_crossing = False
        zgl_val = None

        if has_calls and has_puts:
            # High-resolution scanning grid around ATM to find where Net GEX crosses zero
            min_scan = spot_price * 0.90
            max_scan = spot_price * 1.10
            coarse_grid = np.linspace(min_scan, max_scan, 50)
            fine_grid = np.linspace(spot_price * 0.96, spot_price * 1.04, 60)
            grid = np.unique(np.sort(np.concatenate([coarse_grid, fine_grid])))

            grid_gex: List[Tuple[float, float]] = []
            for s_eval in grid:
                grid_gex.append((float(s_eval), _compute_net_gex(float(s_eval))))

            peak_abs_gex = max((abs(g) for _, g in grid_gex), default=0.0)
            noise_thresh = max(1.0, 1e-4 * peak_abs_gex)

            for i in range(len(grid_gex) - 1):
                s1, g1 = grid_gex[i]
                s2, g2 = grid_gex[i + 1]
                # True zero crossing requires opposite signs and significant amplitude above underflow noise
                if (g1 < 0 and g2 > 0) or (g1 > 0 and g2 < 0):
                    if max(abs(g1), abs(g2)) < noise_thresh:
                        continue
                    # Bracket found: refine root via bisection
                    low, high = s1, s2
                    flow, fhigh = g1, g2
                    root = (s1 + s2) / 2.0
                    for _ in range(12):
                        mid = 0.5 * (low + high)
                        fmid = _compute_net_gex(mid)
                        if abs(fmid) < 1e-4:
                            root = mid
                            break
                        if (flow < 0 and fmid > 0) or (flow > 0 and fmid < 0):
                            high, fhigh = mid, fmid
                        else:
                            low, flow = mid, fmid
                        root = mid
                    zgl_val = round(root, 2)
                    found_crossing = True
                    break
                elif abs(g1) <= 1e-4 and i > 0:
                    _, g_prev = grid_gex[i - 1]
                    if ((g_prev < 0 and g2 > 0) or (g_prev > 0 and g2 < 0)) and max(abs(g_prev), abs(g2)) >= noise_thresh:
                        zgl_val = round(s1, 2)
                        found_crossing = True
                        break

        has_zgl = found_crossing
        zgl = zgl_val if found_crossing else None
        regime = "LONG_GAMMA" if net_gex >= 0 else "SHORT_GAMMA"

        return {
            "net_gex": round(net_gex, 2),
            "net_gex_cr": net_gex_cr,
            "call_gex_cr": call_gex_cr,
            "put_gex_cr": put_gex_cr,
            "zgl": zgl,
            "has_zgl": has_zgl,
            "gamma_regime": regime,
        }
