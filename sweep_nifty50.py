"""
Resumable NIFTY 50 Parrondo Sweep — saves each combo incrementally to CSV.
If interrupted, re-run and it skips already-completed combos.
"""
import sys
import os
import csv
import itertools
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CSV_FILE = "parrondo_sweep_NIFTY 50_5475days.csv"
SYMBOL = "NIFTY 50"
DAYS = 5475

PARAM_GRID = {
    "trend_strength_strong": [0.35, 0.40, 0.45],
    "trend_strength_sideways": [0.25, 0.30, 0.35],
    "vol_expanding_mult": [1.15, 1.20, 1.25],
    "hurst_accumulation": [0.40, 0.45, 0.50],
    "mr_min_score": [2.0, 2.5, 3.0],
}

FIELDNAMES = [
    "combo", "trend_strength_strong", "trend_strength_sideways",
    "vol_expanding_mult", "hurst_accumulation", "mr_min_score",
    "pf", "sharpe", "wr", "dd", "return", "trades", "cagr", "alpha", "calmar",
]


def load_completed():
    """Load already-completed combo labels from CSV."""
    if not os.path.exists(CSV_FILE):
        return set()
    done = set()
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["combo"])
    return done


def append_result(row: dict):
    """Append a single result row to CSV (creates header if new file)."""
    exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    from prometheus.main import Prometheus

    # Generate all 243 combos
    combos = list(itertools.product(
        PARAM_GRID["trend_strength_strong"],
        PARAM_GRID["trend_strength_sideways"],
        PARAM_GRID["vol_expanding_mult"],
        PARAM_GRID["hurst_accumulation"],
        PARAM_GRID["mr_min_score"],
    ))

    # Check what's already done
    completed = load_completed()
    remaining = []
    for strong, sideways, vol_exp, hurst, mr_score in combos:
        label = (f"TS{strong:.2f}_SS{sideways:.2f}_VE{vol_exp:.2f}_"
                 f"HA{hurst:.2f}_MR{mr_score:.1f}")
        if label not in completed:
            remaining.append((label, strong, sideways, vol_exp, hurst, mr_score))

    print(f"\n{'='*70}")
    print(f"  RESUMABLE PARRONDO SWEEP — {SYMBOL}")
    print(f"  Total combos: {len(combos)} | Already done: {len(completed)} | Remaining: {len(remaining)}")
    print(f"{'='*70}\n")

    if not remaining:
        print("  All 243 combos already completed! Loading results...")
        show_results()
        return

    # Init Prometheus and fetch data ONCE
    prometheus = Prometheus()
    print(f"  Fetching {DAYS} days of daily data for {SYMBOL}...")
    data_all = prometheus.data.fetch_historical(SYMBOL, days=DAYS, interval="day", force_refresh=True)

    if data_all.empty or len(data_all) < 100:
        print(f"  ERROR: Insufficient data: {len(data_all)} bars")
        return

    print(f"  Data: {len(data_all)} bars "
          f"({str(data_all['timestamp'].iloc[0])[:10]} to {str(data_all['timestamp'].iloc[-1])[:10]})")
    print(f"\n  Starting sweep of {len(remaining)} remaining combos...\n")

    start_time = time.time()
    for idx, (label, strong, sideways, vol_exp, hurst, mr_score) in enumerate(remaining):
        combo_start = time.time()

        regime_override = {
            "trend_strength_strong": strong,
            "trend_strength_sideways": sideways,
            "vol_expanding_mult": vol_exp,
            "hurst_accumulation": hurst,
        }

        try:
            result, _ = prometheus._run_backtest_on_slice(
                data_all, SYMBOL, label,
                param_overrides={
                    "regime_overrides": regime_override,
                    "mr_min_score": mr_score,
                },
                verbose=False,
                parrondo=True,
            )

            row = {
                "combo": label,
                "trend_strength_strong": strong,
                "trend_strength_sideways": sideways,
                "vol_expanding_mult": vol_exp,
                "hurst_accumulation": hurst,
                "mr_min_score": mr_score,
                "pf": round(result.profit_factor, 3),
                "sharpe": round(result.sharpe_ratio, 3),
                "wr": round(result.win_rate / 100 if result.win_rate > 1 else result.win_rate, 4),
                "dd": round(result.max_drawdown_pct, 2),
                "return": round(result.total_return_pct, 2),
                "trades": result.total_trades,
                "cagr": round(result.annualized_return_pct, 2),
                "alpha": round(getattr(result, 'alpha_pct', 0), 2),
                "calmar": round(getattr(result, 'calmar_ratio', 0), 3),
            }
        except Exception as e:
            print(f"  [{idx+1}/{len(remaining)}] {label} FAILED: {e}")
            row = {
                "combo": label,
                "trend_strength_strong": strong,
                "trend_strength_sideways": sideways,
                "vol_expanding_mult": vol_exp,
                "hurst_accumulation": hurst,
                "mr_min_score": mr_score,
                "pf": 0, "sharpe": 0, "wr": 0, "dd": 100,
                "return": 0, "trades": 0, "cagr": 0, "alpha": 0, "calmar": 0,
            }

        # Save immediately — survives crash
        append_result(row)

        elapsed = time.time() - combo_start
        total_elapsed = time.time() - start_time
        avg_per_combo = total_elapsed / (idx + 1)
        eta = avg_per_combo * (len(remaining) - idx - 1)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(remaining)}] {label}  "
                  f"PF={row['pf']:.2f} Sharpe={row['sharpe']:.2f} "
                  f"WR={row['wr']*100:.0f}% DD={row['dd']:.1f}%  "
                  f"({elapsed:.1f}s, ETA {eta/60:.0f}min)")

    print(f"\n  {'='*70}")
    print(f"  SWEEP COMPLETE — {len(combos)} total combos saved to {CSV_FILE}")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"  {'='*70}\n")

    show_results()


def show_results():
    """Print filtered results from completed CSV."""
    if not os.path.exists(CSV_FILE):
        print("  No results file found.")
        return

    results = []
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                k: float(row[k]) if k not in ("combo",) else row[k]
                for k in row
            })

    # Filter: WR >= 52% AND DD <= 55% AND PF >= 1.4
    filtered = [
        r for r in results
        if r['wr'] >= 0.52 and r['dd'] <= 55.0 and r['pf'] >= 1.4
    ]
    filtered.sort(key=lambda x: -x['sharpe'])

    print(f"\n  {len(filtered)}/{len(results)} combos passed filters (WR>=52%, DD<=55%, PF>=1.4)")
    print(f"\n  TOP 20 (by Sharpe):")
    print(f"  {'Rank':<5} {'Combo':<40} {'PF':<6} {'Sharpe':<7} {'WR':<6} {'DD':<7} {'CAGR':<7}")
    print(f"  {'─'*80}")

    for rank, r in enumerate(filtered[:20], 1):
        print(f"  {rank:<5} {r['combo'][:39]:<40} {r['pf']:>5.2f} {r['sharpe']:>6.2f} "
              f"{r['wr']*100:>5.0f}% {r['dd']:>6.1f}% {r['cagr']:>6.1f}%")

    if filtered:
        best = filtered[0]
        print(f"\n  BEST COMBO:")
        print(f"    trend_strength_strong:   {best['trend_strength_strong']:.2f}")
        print(f"    trend_strength_sideways: {best['trend_strength_sideways']:.2f}")
        print(f"    vol_expanding_mult:      {best['vol_expanding_mult']:.2f}")
        print(f"    hurst_accumulation:      {best['hurst_accumulation']:.2f}")
        print(f"    mr_min_score:            {best['mr_min_score']:.1f}")
        print(f"    PF={best['pf']:.2f}, Sharpe={best['sharpe']:.2f}, "
              f"WR={best['wr']*100:.0f}%, DD={best['dd']:.1f}%, CAGR={best['cagr']:.1f}%")


if __name__ == "__main__":
    main()
