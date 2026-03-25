from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _to_timestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def derive_pillar_rows(
    trades: List[Dict],
    profile_name: str,
    symbol: str,
    days: int,
    phoenix_window_minutes: int = 35,
) -> List[Dict]:
    """Build per-trade pillar attribution rows from backtest trade dictionaries."""
    rows: List[Dict] = []
    if not trades:
        return rows

    atr_values = [_safe_float(t.get("atr_at_entry", 0.0), 0.0) for t in trades]
    atr_nonzero = [x for x in atr_values if x > 0]
    atr_p40 = float(pd.Series(atr_nonzero).quantile(0.40)) if atr_nonzero else 0.0

    for t in trades:
        ts = _to_timestamp(str(t.get("entry_time", "")))
        hour = int(ts.hour) if not pd.isna(ts) else -1
        minute = int(ts.minute) if not pd.isna(ts) else -1

        liqsweep = bool(t.get("signal_liqsweep", False))
        vol_surge = bool(t.get("signal_vol_surge", False))
        vol_confirm = bool(t.get("signal_vol_confirm", False))
        vp_gravity = bool(t.get("signal_vp", False))

        atr_entry = _safe_float(t.get("atr_at_entry", 0.0), 0.0)

        p1_trap = liqsweep and (not vol_surge)
        p2_coil = vol_confirm and ((atr_p40 <= 0.0) or (atr_entry <= atr_p40))
        p3_gravity = vp_gravity
        p4_time_ok = not (hour == 12)

        row = {
            "profile": profile_name,
            "symbol": symbol,
            "days": days,
            "entry_time": str(t.get("entry_time", "")),
            "exit_time": str(t.get("exit_time", "")),
            "direction": str(t.get("direction", "")),
            "strategy": str(t.get("strategy", "")),
            "net_pnl": _safe_float(t.get("net_pnl", 0.0), 0.0),
            "is_win": 1 if _safe_float(t.get("net_pnl", 0.0), 0.0) > 0 else 0,
            "exit_reason": str(t.get("exit_reason", "")),
            "entry_hour": hour,
            "entry_minute": minute,
            "pillar_1_phantom_liquidity_trap": int(p1_trap),
            "pillar_2_volatility_coil": int(p2_coil),
            "pillar_3_volume_profile_gravity": int(p3_gravity),
            "pillar_4_time_segmentation_ok": int(p4_time_ok),
            "pillar_5_phoenix_recovery": 0,
            "phoenix_recovered_prev_loss": 0,
        }
        row["pillar_count"] = (
            row["pillar_1_phantom_liquidity_trap"]
            + row["pillar_2_volatility_coil"]
            + row["pillar_3_volume_profile_gravity"]
            + row["pillar_4_time_segmentation_ok"]
        )
        rows.append(row)

    indexed: List[Tuple[int, Dict]] = list(enumerate(rows))
    indexed.sort(key=lambda x: _to_timestamp(x[1]["entry_time"]))

    for idx, loss_row in indexed:
        if loss_row["is_win"] == 1:
            continue
        loss_ts = _to_timestamp(loss_row["entry_time"])
        if pd.isna(loss_ts):
            continue

        for win_idx, win_row in indexed:
            if win_idx <= idx:
                continue
            if win_row["is_win"] == 0:
                continue
            if win_row["symbol"] != loss_row["symbol"]:
                continue
            if win_row["direction"] == loss_row["direction"]:
                continue

            win_ts = _to_timestamp(win_row["entry_time"])
            if pd.isna(win_ts):
                continue
            dt_min = (win_ts - loss_ts).total_seconds() / 60.0
            if 0 <= dt_min <= phoenix_window_minutes:
                win_row["pillar_5_phoenix_recovery"] = 1
                loss_row["phoenix_recovered_prev_loss"] = 1
                break

    return rows


def summarize_pillars(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    pillars = [
        "pillar_1_phantom_liquidity_trap",
        "pillar_2_volatility_coil",
        "pillar_3_volume_profile_gravity",
        "pillar_4_time_segmentation_ok",
        "pillar_5_phoenix_recovery",
    ]

    for p in pillars:
        active = [r for r in rows if int(r.get(p, 0)) == 1]
        if not active:
            out[p] = {"active_trades": 0, "win_rate_pct": 0.0, "net_pnl": 0.0}
            continue
        wins = sum(int(r.get("is_win", 0)) for r in active)
        net = sum(_safe_float(r.get("net_pnl", 0.0), 0.0) for r in active)
        out[p] = {
            "active_trades": float(len(active)),
            "win_rate_pct": float(100.0 * wins / len(active)),
            "net_pnl": float(net),
        }

    losses = [r for r in rows if int(r.get("is_win", 0)) == 0]
    recovered_losses = [r for r in losses if int(r.get("phoenix_recovered_prev_loss", 0)) == 1]
    out["phoenix_recovery_stats"] = {
        "loss_count": float(len(losses)),
        "recovered_loss_count": float(len(recovered_losses)),
        "recovered_loss_pct": float(100.0 * len(recovered_losses) / len(losses)) if losses else 0.0,
    }

    return out
