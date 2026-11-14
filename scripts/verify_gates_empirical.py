import pandas as pd
import numpy as np

def run_analysis(csv_path, name="NIFTY 50"):
    print(f"\n=======================================================")
    print(f"ANALYSIS FOR: {name} ({csv_path})")
    print(f"=======================================================")
    df = pd.read_csv(csv_path)
    col_map = {c.lower(): c for c in df.columns}
    ts_col = col_map.get("timestamp") or col_map.get("date") or col_map.get("datetime") or df.columns[0]
    df["ts"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("ts").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[col_map[c]].astype(float)

    # 14-period ATR
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs())
    )
    df["atr"] = tr.rolling(14).mean()
    df["date"] = df["ts"].dt.date

    # Session ORB (first 15m bar of each day) & Session Expanding Typical-Price VWAP proxy
    orb_high = df.groupby("date")["high"].transform("first")
    df["orb_high"] = orb_high
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = df.groupby("date")["tp"].transform(lambda s: s.expanding().mean())

    # Features
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["clv_bull"] = (df["close"] - df["low"]) / rng
    df["bar_atr_mult"] = rng / df["atr"]
    df["orb_ext_atr"] = (df["close"] - df["orb_high"]) / df["atr"]
    df["vwap_ext_atr"] = (df["close"] - df["vwap"]).abs() / df["atr"]

    # Path-dependent 4-bar (1-hour) evaluation:
    # Does trade hit +1.0x ATR target BEFORE hitting -0.75x ATR stop-loss?
    wins_path, wins_naive = [], []
    highs, lows, closes, atrs, dates = df["high"].values, df["low"].values, df["close"].values, df["atr"].values, df["date"].values
    n = len(df)

    for i in range(n - 4):
        if np.isnan(atrs[i]) or atrs[i] <= 0 or dates[i] != dates[i + 4]:
            wins_path.append(np.nan)
            wins_naive.append(np.nan)
            continue
        entry = closes[i]
        sl = entry - 0.75 * atrs[i]
        tgt = entry + 1.00 * atrs[i]
        outcome = 0  # 1 = target hit first, 0 = SL hit first or failed to clear friction
        for k in range(1, 5):
            if lows[i + k] <= sl:
                outcome = 0
                break
            if highs[i + k] >= tgt:
                outcome = 1
                break
        else:
            # Neither hit: win only if 1h net gain > 0.25x ATR (clears theta + statutory fees)
            outcome = 1 if (closes[i + 4] - entry) > 0.25 * atrs[i] else 0
        wins_path.append(outcome)
        wins_naive.append(1 if closes[i + 4] > entry else 0)

    # Pad the last 4 bars
    wins_path.extend([np.nan] * 4)
    wins_naive.extend([np.nan] * 4)

    df["win_path"] = wins_path
    df["win_naive"] = wins_naive

    brk = df[(df["close"] > df["orb_high"]) & df["win_path"].notna()].copy()

    print("=== 1. CLV / WICK REJECTION GATE (ORB Breakouts) ===")
    for label, mask in [
        ("Pin-Bar / Weak Close (CLV < 0.65)", brk["clv_bull"] < 0.65),
        ("Strong Close (CLV >= 0.65)", brk["clv_bull"] >= 0.65),
    ]:
        sub = brk[mask]
        print(f"{label:38s} | N={len(sub):6d} | Naive Spot WR={sub['win_naive'].mean()*100:5.1f}% | True Option Path WR={sub['win_path'].mean()*100:5.1f}%")

    print("\n=== 2. ORB EXTENSION BUCKETS (Properly Binned) ===")
    for lo, hi in [(0.0, 0.15), (0.15, 1.20), (1.20, 1.80), (1.80, 3.00), (3.00, 99.0)]:
        sub = brk[(brk["orb_ext_atr"] > lo) & (brk["orb_ext_atr"] <= hi) & (brk["clv_bull"] >= 0.65)]
        print(f"ORB Ext {lo:4.2f}x - {hi:5.2f}x ATR | N={len(sub):6d} | Naive Spot WR={sub['win_naive'].mean()*100:5.1f}% | True Option Path WR={sub['win_path'].mean()*100:5.1f}%")

    print("\n=== 3. SINGLE-BAR EXPANSION (> 1.85x ATR) BY CONTEXT ===")
    for label, mask in [
        ("Bar > 1.85x ATR + Weak Close (CLV < 0.70)", (brk["bar_atr_mult"] > 1.85) & (brk["clv_bull"] < 0.70)),
        ("Bar > 1.85x ATR + Near VWAP (Ignition)", (brk["bar_atr_mult"] > 1.85) & (brk["clv_bull"] >= 0.70) & (brk["vwap_ext_atr"] <= 1.50)),
        ("Bar > 1.85x ATR + Far From VWAP (Blow-off)", (brk["bar_atr_mult"] > 1.85) & (brk["clv_bull"] >= 0.70) & (brk["vwap_ext_atr"] > 1.50)),
    ]:
        sub = brk[mask]
        print(f"{label:44s} | N={len(sub):5d} | Naive Spot WR={sub['win_naive'].mean()*100:5.1f}% | True Option Path WR={sub['win_path'].mean()*100:5.1f}%")

    # 5. ORB High Sweeps (High > ORB High, Close <= ORB High)
    sweeps = df[(df["high"] > df["orb_high"]) & (df["close"] <= df["orb_high"]) & df["win_path"].notna()]
    print("\n=== 5. ORB HIGH SWEEPS (Fakeout Pin Bars: High > ORB, Close <= ORB) ===")
    pin_bars = sweeps[sweeps["clv_bull"] < 0.35]
    print(f"All Sweeps                         | N={len(sweeps):5d} | Naive Spot WR={sweeps['win_naive'].mean()*100:5.1f}% | True Option Path WR={sweeps['win_path'].mean()*100:5.1f}%")
    print(f"Pin-Bar Sweeps (CLV < 0.35)        | N={len(pin_bars):5d} | Naive Spot WR={pin_bars['win_naive'].mean()*100:5.1f}% | True Option Path WR={pin_bars['win_path'].mean()*100:5.1f}%")
    df["is_brk"] = df["close"] > df["orb_high"]
    first_brk = df[df["is_brk"]].groupby("date").first().reset_index()
    first_brk = first_brk[first_brk["win_path"].notna()]
    print("\n=== 4. FIRST ORB BREAKOUT OF THE DAY (True Initial Ignition) ===")
    for label, mask in [
        ("First Breakout Weak Close (CLV < 0.65)", first_brk["clv_bull"] < 0.65),
        ("First Breakout Strong Close (CLV >= 0.65)", first_brk["clv_bull"] >= 0.65),
        ("First Breakout Ignition (CLV >= 0.70 & VWAP <= 1.5x)", (first_brk["clv_bull"] >= 0.70) & (first_brk["vwap_ext_atr"] <= 1.50)),
        ("First Breakout Exhaustion (CLV < 0.65 or VWAP > 1.8x)", (first_brk["clv_bull"] < 0.65) | (first_brk["vwap_ext_atr"] > 1.80)),
    ]:
        sub = first_brk[mask]
        print(f"{label:52s} | N={len(sub):5d} | Naive Spot WR={sub['win_naive'].mean()*100:5.1f}% | True Option Path WR={sub['win_path'].mean()*100:5.1f}%")

if __name__ == "__main__":
    run_analysis("dataset/NIFTY 50_15minute.csv", "NIFTY 50")
    # Also run on BANK NIFTY
    import os
    bn_path = "dataset/NIFTY BANK_15minute.csv"
    if os.path.exists(bn_path):
        run_analysis(bn_path, "NIFTY BANK")
