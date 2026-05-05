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

PERIODS = [
    {"label": "5y", "years": 5},
    {"label": "10y", "years": 10},
    {"label": "crisis_2008", "start": datetime(2008, 1, 1), "end": datetime(2009, 3, 31)},
]


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


def extract_trade_pnls(trades, start_dt, end_dt):
    pnls = []
    for t in trades or []:
        dt = parse_dt(getattr(t, "entry_time", None))
        if dt and start_dt <= dt <= end_dt:
            pnls.append(float(getattr(t, "net_pnl", 0.0) or 0.0))
    return pnls


def summarize_pnls(pnls):
    total = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    net = sum(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    win_rate = (len(wins) / total * 100.0) if total else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    return {
        "trades": total,
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
        .resample(rule)
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


def run_swing_15m(prom, symbol, data_primary, data_hourly, data_daily):
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

    cost_cfg = prom.config.get("backtest", {}).get("costs", {})
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

    engine.run(
        data=data_primary,
        signal_generator=signal_gen,
        strategy_name=f"swing15m_{symbol.replace(' ', '_')}",
        warmup_bars=30,
    )

    return engine


def _read_ohlcv_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=["date"],
        usecols=["date", "open", "high", "low", "close", "volume"],
    )
    df.rename(columns={"date": "timestamp"}, inplace=True)
    return df


def load_15m_data(prom, symbol, days, csv_dir: Path):
    csv_path = csv_dir / f"{symbol}_15minute.csv"
    if csv_path.exists():
        df = _read_ohlcv_csv(csv_path)
        df = prom.data._clean_ohlcv(df, source="csv", interval="15minute")
        return df, "csv"

    df = prom.data.fetch_historical(symbol, days=days, interval="15minute", force_refresh=False)
    if df is None or df.empty:
        return pd.DataFrame(), "auto"
    df = prom.data._clean_ohlcv(df, source="auto", interval="15minute")
    return df, "auto"


def resolve_period_window(data, period):
    min_dt = data["timestamp"].min()
    max_dt = data["timestamp"].max()

    if "years" in period:
        end_dt = max_dt
        start_dt = end_dt - pd.DateOffset(years=int(period["years"]))
    else:
        start_dt = period["start"]
        end_dt = period["end"]

    if end_dt < min_dt or start_dt > max_dt:
        return None, None

    return max(start_dt, min_dt), min(end_dt, max_dt)


def run_period_job(job):
    symbol, period, config_path, csv_dir, max_days = job
    try:
        from prometheus.utils.logger import logger
        if hasattr(logger, "remove"):
            logger.remove()
    except Exception:
        pass

    prom = Prometheus(config_path=config_path, mode_override="paper")
    prom.data.configure_historical_fetch(source="auto", retries=2)

    data_15m, source = load_15m_data(prom, symbol, days=max_days, csv_dir=Path(csv_dir))
    if data_15m is None or data_15m.empty:
        return {
            "period": period["label"],
            "symbol": symbol,
            "trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "note": f"no data ({source})",
            "pnls": [],
        }

    start_dt, end_dt = resolve_period_window(data_15m, period)
    if start_dt is None or end_dt is None:
        return {
            "period": period["label"],
            "symbol": symbol,
            "trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "note": "no data",
            "pnls": [],
        }

    data_slice = data_15m[(data_15m["timestamp"] >= start_dt) & (data_15m["timestamp"] <= end_dt)].copy()
    if data_slice.empty:
        return {
            "period": period["label"],
            "symbol": symbol,
            "trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "note": "no data",
            "pnls": [],
        }

    data_hourly = resample_ohlcv(data_slice, "60min")
    data_daily = resample_ohlcv(data_slice, "1D")

    engine = run_swing_15m(prom, symbol, data_slice, data_hourly, data_daily)
    pnls = extract_trade_pnls(getattr(engine, "trades", []), start_dt, end_dt)
    stats = summarize_pnls(pnls)
    return {
        "period": period["label"],
        "symbol": symbol,
        **stats,
        "note": source,
        "pnls": pnls,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Swing logic on 15-minute data (long periods).")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--csv-dir", default="dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS
    csv_dir = args.csv_dir

    max_years = max(p.get("years", 0) for p in PERIODS)
    max_days = max_years * 365 if max_years else 365

    jobs = []
    for sym in symbols:
        for period in PERIODS:
            jobs.append((sym, period, str(Path("prometheus/config/settings.yaml")), csv_dir, max_days))

    rows = []
    period_pnls = {p["label"]: [] for p in PERIODS}

    processes = args.processes
    if processes <= 0:
        processes = max(1, min(cpu_count() - 1, 6))
    processes = min(processes, len(jobs)) if jobs else 1

    if processes > 1:
        with Pool(processes) as pool:
            results = pool.map(run_period_job, jobs)
    else:
        results = [run_period_job(job) for job in jobs]

    for res in results:
        rows.append({
            "period": res["period"],
            "symbol": res["symbol"],
            "trades": res["trades"],
            "win_rate": res["win_rate"],
            "net_pnl": res["net_pnl"],
            "avg_profit": res["avg_profit"],
            "avg_loss": res["avg_loss"],
            "profit_factor": res["profit_factor"],
            "note": res["note"],
        })
        period_pnls[res["period"]].extend(res.get("pnls", []))

    df = pd.DataFrame(rows)
    if not df.empty:
        df["win_rate"] = df["win_rate"].round(1)
        df["net_pnl"] = df["net_pnl"].round(2)
        df["avg_profit"] = df["avg_profit"].round(2)
        df["avg_loss"] = df["avg_loss"].round(2)
        df["profit_factor"] = df["profit_factor"].replace([float("inf")], 999.0).round(2)

    print("\nPer-symbol results (Swing on 15m):")
    print(df.to_string(index=False))

    summary_rows = []
    for label, pnls in period_pnls.items():
        stats = summarize_pnls(pnls)
        summary_rows.append({"period": label, **stats})

    df_sum = pd.DataFrame(summary_rows)
    if not df_sum.empty:
        df_sum["win_rate"] = df_sum["win_rate"].round(1)
        df_sum["net_pnl"] = df_sum["net_pnl"].round(2)
        df_sum["avg_profit"] = df_sum["avg_profit"].round(2)
        df_sum["avg_loss"] = df_sum["avg_loss"].round(2)
        df_sum["profit_factor"] = df_sum["profit_factor"].replace([float("inf")], 999.0).round(2)

    print("\nTotals by period (Swing on 15m):")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
