from pathlib import Path
from datetime import datetime
import argparse
import os
from multiprocessing import Pool, cpu_count
import pandas as pd

# Prevent Telegram connection attempts during analysis
os.environ["PROM_TELEGRAM_BOT_TOKEN"] = ""
os.environ["PROM_TELEGRAM_CHAT_ID"] = ""

from prometheus.main import Prometheus
from prometheus.backtest.engine import BacktestEngine
from prometheus.signals.technical import calculate_vwap

DEFAULT_SYMBOLS = ["NIFTY 50", "NIFTY BANK"]


def parse_dt(ts):
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        try:
            return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def summarize_trades(trades, start_dt, end_dt):
    pnls = []
    for t in trades or []:
        dt = parse_dt(getattr(t, "entry_time", None))
        if dt and start_dt <= dt <= end_dt:
            pnls.append(float(getattr(t, "net_pnl", 0.0) or 0.0))

    total = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wins_pnl = sum(wins)
    losses_pnl = abs(sum(losses))
    net = sum(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    win_rate = (len(wins) / total * 100.0) if total else 0.0
    profit_factor = (wins_pnl / losses_pnl) if losses_pnl > 0 else (float("inf") if wins else 0.0)
    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "wins_pnl": wins_pnl,
        "losses_pnl": losses_pnl,
        "win_rate": win_rate,
        "net_pnl": net,
        "avg_profit": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def resample_ohlcv(data, rule):
    if data is None or data.empty:
        return pd.DataFrame()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return (
        data.sort_values("timestamp")
        .set_index("timestamp")
        .resample(rule, origin="start_day")
        .agg(agg)
        .dropna()
        .reset_index()
    )


def compute_hourly_bias_map(data_hourly):
    hourly_bias_map = {}
    if data_hourly is None or data_hourly.empty or len(data_hourly) < 10:
        return hourly_bias_map

    hourly_vwap = calculate_vwap(data_hourly.copy())
    for i in range(10, len(data_hourly)):
        chunk = data_hourly.iloc[max(0, i - 20):i + 1]
        date_key = str(chunk["timestamp"].iloc[-1])[:10]

        recent_h = chunk.tail(5)
        highs = recent_h["high"].values
        lows = recent_h["low"].values

        hh_count = sum(1 for j in range(1, len(highs)) if highs[j] > highs[j - 1])
        hl_count = sum(1 for j in range(1, len(lows)) if lows[j] > lows[j - 1])
        lh_count = sum(1 for j in range(1, len(highs)) if highs[j] < highs[j - 1])
        ll_count = sum(1 for j in range(1, len(lows)) if lows[j] < lows[j - 1])

        close_val = recent_h["close"].iloc[-1]
        if "vwap" in hourly_vwap.columns and i < len(hourly_vwap):
            vwap_val = hourly_vwap["vwap"].iloc[i]
        else:
            vwap_val = close_val

        bull_points = hh_count + hl_count + (1 if close_val > vwap_val else 0)
        bear_points = lh_count + ll_count + (1 if close_val < vwap_val else 0)

        if bull_points >= 4 and bull_points > bear_points + 1:
            hourly_bias_map[date_key] = "bullish"
        elif bear_points >= 4 and bear_points > bull_points + 1:
            hourly_bias_map[date_key] = "bearish"
        else:
            hourly_bias_map[date_key] = "neutral"

    return hourly_bias_map


def _read_ohlcv_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=["date"],
        usecols=["date", "open", "high", "low", "close", "volume"],
    )
    df.rename(columns={"date": "timestamp"}, inplace=True)
    return df


def load_15m_data(prom, symbol, csv_dir: Path):
    csv_15m = csv_dir / f"{symbol}_15minute.csv"
    if csv_15m.exists():
        df = _read_ohlcv_csv(csv_15m)
        df = prom.data._clean_ohlcv(df, source="csv", interval="15minute")
        return df, "csv_15m"

    csv_5m = csv_dir / f"{symbol}_5minute.csv"
    if csv_5m.exists():
        df = _read_ohlcv_csv(csv_5m)
        df = prom.data._clean_ohlcv(df, source="csv", interval="5minute")
        df = resample_ohlcv(df, "15min")
        df = prom.data._clean_ohlcv(df, source="csv", interval="15minute")
        return df, "csv_5m_resample"

    df = prom.data.fetch_historical(symbol, days=365, interval="15minute", force_refresh=False)
    if df is None or df.empty:
        return pd.DataFrame(), "auto"
    df = prom.data._clean_ohlcv(df, source="auto", interval="15minute")
    return df, "auto"


def compute_monthly_volatility(data_15m):
    df = data_15m.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["month"] = df["timestamp"].dt.to_period("M")
    vol = df.groupby("month")["ret"].std().dropna()
    return vol


def select_top_months(vol_series, top_n):
    ranked = vol_series.sort_values(ascending=False)
    return ranked.head(top_n)


def compute_vol_series_map(prom, symbols, csv_dir: Path):
    vol_map = {}
    for sym in symbols:
        data_15m, _ = load_15m_data(prom, sym, csv_dir)
        if data_15m is None or data_15m.empty:
            continue
        vol_series = compute_monthly_volatility(data_15m)
        if not vol_series.empty:
            vol_map[sym] = vol_series
    return vol_map


def select_crisis_months(vol_map, top_n, scope, primary_symbol):
    if not vol_map:
        return []

    scope = (scope or "primary").lower()
    if scope == "combined":
        combined = pd.concat(vol_map, axis=1).mean(axis=1, skipna=True).dropna()
        series = select_top_months(combined, top_n)
    elif scope == "union":
        union_map = {}
        for series in vol_map.values():
            top_series = select_top_months(series, top_n)
            for period, vol in top_series.items():
                union_map[period] = max(float(vol), union_map.get(period, float(vol)))
        if not union_map:
            return []
        series = pd.Series(union_map).sort_values(ascending=False)
    else:
        primary = primary_symbol if primary_symbol in vol_map else next(iter(vol_map))
        series = select_top_months(vol_map[primary], top_n)

    return month_windows(series)


def build_cost_cfg(base_cfg, slippage_override):
    cfg = dict(base_cfg or {})
    if slippage_override is not None:
        cfg["slippage_pct"] = float(slippage_override)
    return cfg


def build_scenarios(stress_slippage):
    scenarios = [("base", None)]
    if stress_slippage is not None and stress_slippage > 0:
        scenarios.append((f"slip_{stress_slippage:.2f}", stress_slippage))
    return scenarios


def run_swing_15m(prom, symbol, data_primary, cost_cfg, scenario_label):
    data_hourly = resample_ohlcv(data_primary, "60min")
    data_daily = resample_ohlcv(data_primary, "1D")

    regime_state = prom.regime_detector.detect(data_daily) if len(data_daily) >= 50 else None
    hourly_bias_map = compute_hourly_bias_map(data_hourly)

    capital = prom.initial_capital
    capital_tracker = {"capital": capital, "peak": capital}

    param_overrides_dict = {"mr_min_score": 2.5}

    signal_gen = prom._make_signal_generator(
        regime_state=regime_state,
        hourly_bias_map=hourly_bias_map,
        capital=capital,
        primary_interval="15minute",
        symbol=symbol,
        param_overrides=param_overrides_dict,
        parrondo=False,
        capital_tracker=capital_tracker,
    )

    max_pos = 1 if prom.initial_capital < 30000 else 2
    engine = BacktestEngine(
        initial_capital=prom.initial_capital,
        cost_config=cost_cfg,
        entry_timing=False,
        entry_pullback_atr=0.3,
        entry_max_wait_bars=2,
        capital_tracker=capital_tracker,
        max_positions=max_pos,
        vol_target=0.0,
        dd_throttle=True,
        equity_curve_filter=False,
        half_capacity_mode=False,
        half_capacity_alpha=0.5,
        equity_ma_window=50,
        equity_ma_sizing=False,
        equity_ma_band=0.05,
        dsq_filter=False,
        dsq_lookback=20,
        dsq_baseline_window=252,
        dsq_soft=0.25,
        dsq_hard=0.60,
        dsq_min_scalar=0.25,
    )

    result = engine.run(
        data=data_primary,
        signal_generator=signal_gen,
        strategy_name=f"swing15m_{scenario_label}_{symbol.replace(' ', '_')}",
        warmup_bars=30,
    )

    return engine, result


def month_windows(months):
    windows = []
    for period, vol in months.items():
        start_dt = period.start_time
        end_dt = period.end_time
        label = str(period)
        windows.append({"label": label, "start": start_dt, "end": end_dt, "vol": float(vol)})
    return windows


def run_symbol_job(job):
    symbol, csv_dir, config_path, stress_slippage, months = job

    try:
        from prometheus.utils.logger import logger
        if hasattr(logger, "remove"):
            logger.remove()
    except Exception:
        pass

    prom_loader = Prometheus(config_path=config_path, mode_override="paper")
    prom_loader.data.configure_historical_fetch(source="auto", retries=2)

    data_15m, source = load_15m_data(prom_loader, symbol, Path(csv_dir))
    if data_15m is None or data_15m.empty:
        rows = []
        for win in months:
            rows.append({
                "scenario": "base",
                "period": win["label"],
                "symbol": symbol,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "wins_pnl": 0.0,
                "losses_pnl": 0.0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "note": f"no data ({source})",
            })
        return rows

    base_costs = prom_loader.config.get("backtest", {}).get("costs", {})
    scenarios = build_scenarios(stress_slippage)

    rows = []
    for label, slip in scenarios:
        prom = Prometheus(config_path=config_path, mode_override="paper")
        prom.data.configure_historical_fetch(source="auto", retries=2)

        cost_cfg = build_cost_cfg(base_costs, slip)
        engine, _ = run_swing_15m(prom, symbol, data_15m, cost_cfg, label)
        trades = getattr(engine, "trades", [])

        for win in months:
            stats = summarize_trades(trades, win["start"], win["end"])
            rows.append({
                "scenario": label,
                "period": win["label"],
                "symbol": symbol,
                **stats,
                "note": source,
            })

    return rows


def run_scenario_job(job):
    symbol, scenario_label, slip, csv_dir, config_path, months = job

    try:
        from prometheus.utils.logger import logger
        if hasattr(logger, "remove"):
            logger.remove()
    except Exception:
        pass

    prom_loader = Prometheus(config_path=config_path, mode_override="paper")
    prom_loader.data.configure_historical_fetch(source="auto", retries=2)

    data_15m, source = load_15m_data(prom_loader, symbol, Path(csv_dir))
    if data_15m is None or data_15m.empty:
        rows = []
        for win in months:
            rows.append({
                "scenario": scenario_label,
                "period": win["label"],
                "symbol": symbol,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "wins_pnl": 0.0,
                "losses_pnl": 0.0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "note": f"no data ({source})",
            })
        return rows

    base_costs = prom_loader.config.get("backtest", {}).get("costs", {})
    prom = Prometheus(config_path=config_path, mode_override="paper")
    prom.data.configure_historical_fetch(source="auto", retries=2)

    cost_cfg = build_cost_cfg(base_costs, slip)
    engine, _ = run_swing_15m(prom, symbol, data_15m, cost_cfg, scenario_label)
    trades = getattr(engine, "trades", [])

    rows = []
    for win in months:
        stats = summarize_trades(trades, win["start"], win["end"])
        rows.append({
            "scenario": scenario_label,
            "period": win["label"],
            "symbol": symbol,
            **stats,
            "note": source,
        })

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Crisis proxy months for swing-15m logic.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--csv-dir", default="dataset")
    parser.add_argument("--processes", type=int, default=0)
    parser.add_argument("--parallel-scenarios", action="store_true")
    parser.add_argument("--top-months", type=int, default=6)
    parser.add_argument("--vol-symbol", default="NIFTY 50")
    parser.add_argument("--vol-scope", choices=["primary", "combined", "union"], default="primary")
    parser.add_argument("--stress-slippage", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS
    csv_dir = args.csv_dir
    config_path = str(Path("prometheus/config/settings.yaml"))

    vol_symbol = args.vol_symbol if args.vol_symbol else symbols[0]
    vol_symbols = symbols if args.vol_scope != "primary" else [vol_symbol]
    prom = Prometheus(config_path=config_path, mode_override="paper")
    prom.data.configure_historical_fetch(source="auto", retries=2)
    vol_map = compute_vol_series_map(prom, vol_symbols, Path(csv_dir))
    if not vol_map:
        print("No data available to compute volatility months.")
        return

    months = select_crisis_months(vol_map, args.top_months, args.vol_scope, vol_symbol)
    if not months:
        print("No volatility months found for the selected scope.")
        return

    print("\nSelected crisis proxy months (highest volatility):")
    for win in months:
        print(f"- {win['label']} (vol={win['vol']:.6f})")

    scenarios = build_scenarios(args.stress_slippage)
    if args.parallel_scenarios:
        jobs = [
            (sym, label, slip, csv_dir, config_path, months)
            for sym in symbols
            for label, slip in scenarios
        ]
        worker = run_scenario_job
    else:
        jobs = [(sym, csv_dir, config_path, args.stress_slippage, months) for sym in symbols]
        worker = run_symbol_job

    if args.processes <= 0:
        processes = max(1, min(cpu_count() - 1, len(jobs)))
    else:
        processes = min(args.processes, len(jobs)) if jobs else 1

    if processes > 1:
        with Pool(processes) as pool:
            results = pool.map(worker, jobs)
    else:
        results = [worker(job) for job in jobs]

    rows = [row for chunk in results for row in chunk]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No results generated.")
        return

    df["win_rate"] = df["win_rate"].round(1)
    df["net_pnl"] = df["net_pnl"].round(2)
    df["avg_profit"] = df["avg_profit"].round(2)
    df["avg_loss"] = df["avg_loss"].round(2)
    df["profit_factor"] = df["profit_factor"].replace([float("inf")], 999.0).round(2)

    display_cols = [
        "scenario",
        "period",
        "symbol",
        "trades",
        "win_rate",
        "net_pnl",
        "avg_profit",
        "avg_loss",
        "profit_factor",
        "note",
    ]
    print("\nPer-symbol results (Crisis proxy months):")
    print(df[display_cols].to_string(index=False))

    summary = (
        df.groupby(["scenario", "period"], as_index=False)
        .agg({
            "trades": "sum",
            "wins": "sum",
            "losses": "sum",
            "wins_pnl": "sum",
            "losses_pnl": "sum",
            "net_pnl": "sum",
        })
    )

    summary["win_rate"] = summary.apply(
        lambda r: (r["wins"] / r["trades"] * 100.0) if r["trades"] else 0.0,
        axis=1,
    )
    summary["avg_profit"] = summary.apply(
        lambda r: (r["wins_pnl"] / r["wins"]) if r["wins"] else 0.0,
        axis=1,
    )
    summary["avg_loss"] = summary.apply(
        lambda r: (r["losses_pnl"] / r["losses"]) if r["losses"] else 0.0,
        axis=1,
    )
    summary["profit_factor"] = summary.apply(
        lambda r: (r["wins_pnl"] / r["losses_pnl"]) if r["losses_pnl"] else (float("inf") if r["wins"] else 0.0),
        axis=1,
    )

    summary = summary[[
        "scenario",
        "period",
        "trades",
        "win_rate",
        "net_pnl",
        "avg_profit",
        "avg_loss",
        "profit_factor",
    ]]

    summary["win_rate"] = summary["win_rate"].round(1)
    summary["net_pnl"] = summary["net_pnl"].round(2)
    summary["avg_profit"] = summary["avg_profit"].round(2)
    summary["avg_loss"] = summary["avg_loss"].round(2)
    summary["profit_factor"] = summary["profit_factor"].replace([float("inf")], 999.0).round(2)

    print("\nTotals by month (Crisis proxy):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
