import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
from datetime import datetime
import os
import json
import re
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

os.environ["PROM_TELEGRAM_BOT_TOKEN"] = ""
os.environ["PROM_TELEGRAM_CHAT_ID"] = ""

from prometheus.main import Prometheus
from prometheus.backtest.engine import BacktestEngine
from prometheus.signals.technical import calculate_vwap

pd.options.mode.chained_assignment = None

RUN_STATE_VERSION = 2
CHECKPOINT_INTERVAL_SECONDS = 300

SUMMARY_PERIODS = [
    {"label": "full"},
    {"label": "10y", "years": 10},
    {"label": "5y", "years": 5},
    {"label": "train_2015_2019", "start": datetime(2015, 1, 1), "end": datetime(2019, 12, 31)},
    {"label": "test_2020_plus", "start": datetime(2020, 1, 1)},
]

CRISIS_PERIODS = [
    {"label": "2020-03", "start": datetime(2020, 3, 1), "end": datetime(2020, 3, 31)},
    {"label": "2020-04", "start": datetime(2020, 4, 1), "end": datetime(2020, 4, 30)},
    {"label": "2020-05", "start": datetime(2020, 5, 1), "end": datetime(2020, 5, 31)},
    {"label": "2020-06", "start": datetime(2020, 6, 1), "end": datetime(2020, 6, 30)},
    {"label": "2022-02", "start": datetime(2022, 2, 1), "end": datetime(2022, 2, 28)},
    {"label": "2024-06", "start": datetime(2024, 6, 1), "end": datetime(2024, 6, 30)},
]

ALL_PERIODS = SUMMARY_PERIODS + CRISIS_PERIODS
STRICT_HOLDOUT_LABELS = {"train_2015_2019", "test_2020_plus"} | {period["label"] for period in CRISIS_PERIODS}
ISOLATED_PRE_ROLL_DAYS = 180

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

def resample_ohlcv(data, rule):
    if data is None or data.empty: return pd.DataFrame()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return data.sort_values("timestamp").set_index("timestamp").resample(rule, origin="start_day").agg(agg).dropna().reset_index()

def parse_dt(ts):
    if ts is None: return None
    try: return datetime.fromisoformat(str(ts))
    except Exception:
        try: return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
        except Exception: return None


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return slug or "run"


def build_data_signature(data, source, symbol):
    if data is None or data.empty:
        return {
            "symbol": symbol,
            "source": source,
            "data_len": 0,
            "first_timestamp": "",
            "last_timestamp": "",
        }

    first_timestamp = str(pd.to_datetime(data["timestamp"].iloc[0])) if "timestamp" in data.columns else ""
    last_timestamp = str(pd.to_datetime(data["timestamp"].iloc[-1])) if "timestamp" in data.columns else ""
    return {
        "symbol": symbol,
        "source": source,
        "data_len": int(len(data)),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def load_resume_state(state_file: Path, expected_signature: dict):
    if not state_file.exists():
        return {"version": RUN_STATE_VERSION, "data_signature": expected_signature, "completed_scenarios": {}}

    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {"version": RUN_STATE_VERSION, "data_signature": expected_signature, "completed_scenarios": {}}

    if payload.get("version") != RUN_STATE_VERSION or payload.get("data_signature") != expected_signature:
        return {"version": RUN_STATE_VERSION, "data_signature": expected_signature, "completed_scenarios": {}}

    completed = payload.get("completed_scenarios") or {}
    if not isinstance(completed, dict):
        completed = {}

    return {
        "version": RUN_STATE_VERSION,
        "data_signature": expected_signature,
        "completed_scenarios": completed,
    }


def save_resume_state(state_file: Path, data_signature: dict, completed_scenarios: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": RUN_STATE_VERSION,
        "data_signature": data_signature,
        "completed_scenarios": completed_scenarios,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    temp_file = state_file.with_suffix(state_file.suffix + ".tmp")
    temp_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temp_file, state_file)


def write_metrics_csv(metrics_file: Path, rows):
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    temp_file = metrics_file.with_suffix(metrics_file.suffix + ".tmp")
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, metrics_file)


def slice_period_context(data, start_dt, end_dt, pre_roll_days: int = ISOLATED_PRE_ROLL_DAYS):
    if data is None or data.empty:
        return pd.DataFrame()

    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    context_start = start_dt - pd.Timedelta(days=int(pre_roll_days))
    return data[(data["timestamp"] >= context_start) & (data["timestamp"] <= end_dt)].copy()


def wrap_signal_generator_with_cutoff(signal_generator, cutoff_dt):
    cutoff_dt = pd.to_datetime(cutoff_dt) if cutoff_dt is not None else None

    def wrapped(data_so_far):
        signal = signal_generator(data_so_far)
        if signal is None or cutoff_dt is None or data_so_far is None or data_so_far.empty:
            return signal

        last_timestamp = pd.to_datetime(data_so_far["timestamp"].iloc[-1])
        if last_timestamp < cutoff_dt:
            return None
        return signal

    return wrapped


def summarize_trade_groups(trade_df, group_col, bucket_col):
    if trade_df is None or trade_df.empty or group_col not in trade_df.columns:
        return pd.DataFrame(columns=[bucket_col, "trades", "wins", "losses", "win_rate", "net_pnl", "avg_profit", "avg_loss", "profit_factor", "avg_hold_min", "avg_trade_pnl"])

    rows = []
    for bucket_value, group in trade_df.groupby(group_col, dropna=False):
        pnls = pd.to_numeric(group.get("net_pnl"), errors="coerce").fillna(0.0)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        wins_pnl = float(wins.sum())
        losses_pnl = float(abs(losses.sum()))
        trade_count = int(len(group))
        win_count = int((pnls > 0).sum())
        loss_count = int((pnls < 0).sum())
        profit_factor = wins_pnl / losses_pnl if losses_pnl > 0 else (float("inf") if win_count else 0.0)
        rows.append({
            bucket_col: bucket_value,
            "trades": trade_count,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round((win_count / trade_count * 100.0) if trade_count else 0.0, 1),
            "net_pnl": round(float(pnls.sum()), 2),
            "avg_profit": round(float(wins.mean()) if len(wins) else 0.0, 2),
            "avg_loss": round(float(abs(losses.mean())) if len(losses) else 0.0, 2),
            "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf"),
            "avg_hold_min": round(float(pd.to_numeric(group.get("hold_min"), errors="coerce").mean()) if "hold_min" in group.columns else 0.0, 1),
            "avg_trade_pnl": round(float(pnls.mean()) if trade_count else 0.0, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["net_pnl", "trades"], ascending=[False, False]).reset_index(drop=True)
    return df


def build_trade_frame(result, data_primary, prom):
    trades = getattr(result, "trades", []) or []
    if not trades:
        return pd.DataFrame()

    data_daily = resample_ohlcv(data_primary, "1D")
    condition_cache = {}
    rows = []

    for trade in trades:
        entry_time = parse_dt(trade.get("entry_time") or trade.get("entry"))
        exit_time = parse_dt(trade.get("exit_time") or trade.get("exit"))
        if entry_time is None:
            continue

        entry_ts = pd.to_datetime(entry_time)
        day_key = str(entry_ts.normalize())[:10]
        if day_key not in condition_cache:
            daily_slice = data_daily[data_daily["timestamp"] <= entry_ts]
            if len(daily_slice) >= 50:
                regime_state = prom.regime_detector.detect(daily_slice)
                market_regime = getattr(getattr(regime_state, "regime", None), "value", "unknown")
                volatility_regime = getattr(regime_state, "volatility_regime", "unknown")
            else:
                market_regime = "unknown"
                volatility_regime = "unknown"
            condition_cache[day_key] = (market_regime, volatility_regime)

        market_regime, volatility_regime = condition_cache[day_key]
        net_pnl = float(trade.get("net_pnl", trade.get("pnl", 0.0)) or 0.0)
        hold_min = 0.0
        if entry_time is not None and exit_time is not None:
            hold_min = max((exit_time - entry_time).total_seconds() / 60.0, 0.0)

        rows.append({
            "entry_time": entry_ts.isoformat(sep=" "),
            "exit_time": exit_time.isoformat(sep=" ") if exit_time is not None else "",
            "symbol": trade.get("symbol", ""),
            "direction": trade.get("direction", ""),
            "strategy": trade.get("strategy", ""),
            "net_pnl": round(net_pnl, 2),
            "gross_pnl": round(float(trade.get("gross_pnl", 0.0) or 0.0), 2),
            "costs": round(float(trade.get("costs", 0.0) or 0.0), 2),
            "entry_price": round(float(trade.get("entry_price", 0.0) or 0.0), 2),
            "exit_price": round(float(trade.get("exit_price", 0.0) or 0.0), 2),
            "quantity": int(float(trade.get("quantity", 0) or 0)),
            "exit_reason": trade.get("exit_reason", ""),
            "entry_type": trade.get("entry_type", ""),
            "signal_liqsweep": bool(trade.get("signal_liqsweep", False)),
            "signal_fvg": bool(trade.get("signal_fvg", False)),
            "signal_vp": bool(trade.get("signal_vp", False)),
            "signal_ote": bool(trade.get("signal_ote", False)),
            "signal_rsi_div": bool(trade.get("signal_rsi_div", False)),
            "signal_vol_surge": bool(trade.get("signal_vol_surge", False)),
            "signal_vol_confirm": bool(trade.get("signal_vol_confirm", False)),
            "signal_vwap": bool(trade.get("signal_vwap", False)),
            "signal_bias": bool(trade.get("signal_bias", False)),
            "bull_score": round(float(trade.get("bull_score", 0.0) or 0.0), 2),
            "bear_score": round(float(trade.get("bear_score", 0.0) or 0.0), 2),
            "atr_at_entry": round(float(trade.get("atr_at_entry", 0.0) or 0.0), 2),
            "regime_at_entry": trade.get("regime_at_entry", "unknown"),
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "option_expiry_date": trade.get("option_expiry_date", ""),
            "entry_year": entry_ts.year,
            "entry_month": entry_ts.strftime("%Y-%m"),
            "hold_min": round(float(hold_min), 1),
            "win": net_pnl > 0,
        })

    return pd.DataFrame(rows)


def write_trade_artifacts(report_dir, run_day, symbol, scenario_label, trade_df):
    if report_dir is None or run_day is None or trade_df is None or trade_df.empty:
        return {}

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = slugify(symbol)
    safe_scenario = slugify(scenario_label)

    artifacts = {}

    def save_frame(frame, filename):
        file_path = report_dir / filename
        temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
        frame.to_csv(temp_file, index=False)
        os.replace(temp_file, file_path)
        return file_path

    artifacts["trades"] = save_frame(trade_df, f"swing_15m_trades_{run_day}_{safe_symbol}_{safe_scenario}.csv")
    regime_summary = summarize_trade_groups(trade_df, "regime_at_entry", "regime_at_entry")
    artifacts["regime_summary"] = save_frame(regime_summary, f"swing_15m_regime_summary_{run_day}_{safe_symbol}_{safe_scenario}.csv")
    volatility_summary = summarize_trade_groups(trade_df, "volatility_regime", "volatility_regime")
    artifacts["volatility_summary"] = save_frame(volatility_summary, f"swing_15m_volatility_summary_{run_day}_{safe_symbol}_{safe_scenario}.csv")
    yearly_summary = summarize_trade_groups(trade_df, "entry_year", "entry_year")
    artifacts["yearly_summary"] = save_frame(yearly_summary, f"swing_15m_yearly_summary_{run_day}_{safe_symbol}_{safe_scenario}.csv")
    monthly_summary = summarize_trade_groups(trade_df, "entry_month", "entry_month")
    artifacts["monthly_summary"] = save_frame(monthly_summary, f"swing_15m_monthly_summary_{run_day}_{safe_symbol}_{safe_scenario}.csv")
    return artifacts

def load_15m_data(prom, symbol, csv_dir: Path):
    csv_15m = csv_dir / f"{symbol}_15minute.csv"
    if csv_15m.exists():
        df = pd.read_csv(csv_15m, parse_dates=["date"], usecols=["date", "open", "high", "low", "close", "volume"])
        df.rename(columns={"date": "timestamp"}, inplace=True)
        return prom.data._clean_ohlcv(df, source="csv", interval="15minute"), "csv_15m"
    
    # Fallback to auto
    df = prom.data.fetch_historical(symbol, days=3650, interval="15minute", force_refresh=False)
    if df is not None and not df.empty:
        return prom.data._clean_ohlcv(df, source="auto", interval="15minute"), "auto"
        
    return pd.DataFrame(), "none"


def build_cost_cfg(base_cfg, slippage_override):
    cfg = dict(base_cfg or {})
    if slippage_override is not None:
        cfg["slippage_pct"] = float(slippage_override)
    return cfg


def resolve_period_window(data, period):
    min_dt = data["timestamp"].min()
    max_dt = data["timestamp"].max()

    if period.get("label") == "full":
        start_dt, end_dt = min_dt, max_dt
    elif "years" in period:
        end_dt = max_dt
        start_dt = end_dt - pd.DateOffset(years=int(period["years"]))
    else:
        start_dt = period.get("start", min_dt)
        end_dt = period.get("end", max_dt)

    if end_dt < min_dt or start_dt > max_dt:
        return None, None

    return max(start_dt, min_dt), min(end_dt, max_dt)

def run_swing_15m_engine(prom, symbol, data_primary, slip_override, checkpoint_path=None, checkpoint_interval_seconds=CHECKPOINT_INTERVAL_SECONDS, trade_start_dt=None):
    data_hourly = resample_ohlcv(data_primary, "60min")
    data_daily = resample_ohlcv(data_primary, "1D")
    regime_state = prom.regime_detector.detect(data_daily) if len(data_daily) >= 50 else None
    hourly_bias_map = compute_hourly_bias_map(data_hourly)
    capital = prom.initial_capital
    capital_tracker = {"capital": capital, "peak": capital}
    signal_gen = prom._make_signal_generator(
        regime_state=regime_state, hourly_bias_map=hourly_bias_map, capital=capital,
        primary_interval="15minute", symbol=symbol, param_overrides={"mr_min_score": 2.5},
        parrondo=False, capital_tracker=capital_tracker
    )
    if trade_start_dt is not None:
        signal_gen = wrap_signal_generator_with_cutoff(signal_gen, trade_start_dt)
    cost_cfg = dict(prom.config.get("backtest", {}).get("costs", {}))
    if slip_override is not None:
        cost_cfg["slippage_pct"] = float(slip_override)
    engine = BacktestEngine(
        initial_capital=prom.initial_capital, cost_config=cost_cfg, entry_timing=False, entry_pullback_atr=0.3,
        entry_max_wait_bars=2, capital_tracker=capital_tracker, max_positions=1 if prom.initial_capital < 30000 else 2,
        vol_target=0.0, dd_throttle=True, equity_curve_filter=False, half_capacity_mode=False
    )
    result = engine.run(
        data=data_primary,
        signal_generator=signal_gen,
        strategy_name=f"swing_{symbol}",
        warmup_bars=30,
        checkpoint_path=checkpoint_path,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
    )
    return engine, result


def build_equity_series(result, data_primary, warmup_bars: int = 30) -> pd.Series:
    equity_curve = np.asarray(getattr(result, "equity_curve", []), dtype=float)
    if len(equity_curve) <= 1 or data_primary is None or data_primary.empty:
        return pd.Series(dtype=float)

    trading_equity = equity_curve[1:]
    available_bars = max(len(data_primary) - warmup_bars, 0)
    series_len = min(len(trading_equity), available_bars)
    if series_len <= 0:
        return pd.Series(dtype=float)

    timestamps = pd.to_datetime(
        data_primary["timestamp"].iloc[warmup_bars:warmup_bars + series_len]
    ).to_list()
    return pd.Series(trading_equity[:series_len], index=pd.DatetimeIndex(timestamps))


def calculate_period_metrics(engine, result, data_primary, equity_series: pd.Series, start_dt, end_dt, period_label: str, scenario_label: str, source: str, evaluation_mode: str = "full_run_slice"):
    period_data = data_primary[(data_primary["timestamp"] >= start_dt) & (data_primary["timestamp"] <= end_dt)]
    window_equity = equity_series[(equity_series.index >= start_dt) & (equity_series.index <= end_dt)]

    slice_trades = []
    for trade in getattr(result, "trades", []) or []:
        trade_dt = parse_dt(trade.get("entry_time") or trade.get("entry"))
        if trade_dt and start_dt <= trade_dt <= end_dt:
            slice_trades.append(trade)

    pnls = np.array([float(t.get("net_pnl", t.get("pnl", 0.0)) or 0.0) for t in slice_trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    trade_count = int(len(pnls))
    win_rate = float((len(wins) / trade_count * 100.0) if trade_count else 0.0)
    net_pnl = float(pnls.sum())
    avg_profit = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0)
    avg_trade_pnl = float(pnls.mean()) if trade_count else 0.0

    if len(window_equity) >= 2:
        initial_cap = float(window_equity.iloc[0])
        final_cap = float(window_equity.iloc[-1])
        total_return_pct = ((final_cap / initial_cap) - 1.0) * 100.0 if initial_cap > 0 else 0.0

        period_days = max((pd.to_datetime(end_dt) - pd.to_datetime(start_dt)).days, 1)
        years = period_days / 365.25
        cagr = (((final_cap / initial_cap) ** (1.0 / years)) - 1.0) * 100.0 if (initial_cap > 0 and final_cap > 0 and years > 0) else 0.0

        peak = window_equity.cummax()
        drawdown_pct = ((peak - window_equity) / peak.replace(0, np.nan) * 100.0).fillna(0.0)
        max_dd = float(drawdown_pct.max())

        returns = window_equity.pct_change().dropna()
        returns = returns[returns != 0]
        if len(returns) > 0 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
            downside = returns[returns < 0]
            downside_std = float(downside.std()) if len(downside) > 0 else float(returns.std())
            sortino = float(returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

        if len(returns) > 2 and np.isfinite(sharpe):
            psr_pct, min_trl = engine._compute_psr_and_min_trl(returns.to_numpy(), sharpe, benchmark_sharpe=0.0, confidence=0.95)
        else:
            psr_pct, min_trl = 0.0, 0

        dd_start = None
        max_dd_duration_days = 0
        for ts, eq_val, peak_val in zip(window_equity.index, window_equity.to_numpy(dtype=float), peak.to_numpy(dtype=float)):
            if eq_val < peak_val:
                if dd_start is None:
                    dd_start = ts
            elif dd_start is not None:
                max_dd_duration_days = max(max_dd_duration_days, max((ts - dd_start).days, 0))
                dd_start = None
        if dd_start is not None:
            max_dd_duration_days = max(max_dd_duration_days, max((window_equity.index[-1] - dd_start).days, 0))

        if len(period_data) > 1:
            bh_return_pct = ((period_data["close"].iloc[-1] / period_data["close"].iloc[0]) - 1.0) * 100.0
            bh_cagr = (((1.0 + bh_return_pct / 100.0) ** (1.0 / years)) - 1.0) * 100.0 if years > 0 else 0.0
            alpha_pct = cagr - bh_cagr
        else:
            bh_return_pct = 0.0
            bh_cagr = 0.0
            alpha_pct = 0.0

        hold_durations = []
        for trade in slice_trades:
            try:
                entry_time = pd.to_datetime(trade.get("entry_time") or trade.get("entry"))
                exit_time = pd.to_datetime(trade.get("exit_time") or trade.get("exit"))
                hold_durations.append((exit_time - entry_time).total_seconds() / 60.0)
            except Exception:
                hold_durations.append(0.0)
        avg_hold_min = float(np.mean(hold_durations)) if hold_durations else 0.0
    else:
        initial_cap = float(window_equity.iloc[0]) if len(window_equity) else 0.0
        final_cap = float(window_equity.iloc[-1]) if len(window_equity) else 0.0
        total_return_pct = 0.0
        cagr = 0.0
        max_dd = 0.0
        sharpe = 0.0
        sortino = 0.0
        calmar = 0.0
        psr_pct = 0.0
        min_trl = 0
        max_dd_duration_days = 0
        bh_return_pct = 0.0
        bh_cagr = 0.0
        alpha_pct = 0.0
        avg_hold_min = 0.0

    return {
        "scenario": scenario_label,
        "period": period_label,
        "trades": trade_count,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round(win_rate, 1),
        "net_pnl": round(net_pnl, 2),
        "avg_profit": round(avg_profit, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf"),
        "total_return_pct": round(total_return_pct, 2),
        "cagr": round(cagr, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "alpha_pct": round(alpha_pct, 2),
        "psr_pct": round(psr_pct, 1),
        "min_trl": int(min_trl),
        "max_dd_duration_days": int(max_dd_duration_days),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "avg_hold_min": round(avg_hold_min, 1),
        "bh_return_pct": round(bh_return_pct, 2),
        "bh_cagr": round(bh_cagr, 2),
        "initial_capital": round(initial_cap, 2),
        "final_capital": round(final_cap, 2),
        "evaluation_mode": evaluation_mode,
        "note": source,
    }


def compute_scenario_rows(prom, data_15m, source, symbol, scenario_label, slip, checkpoint_path=None, checkpoint_dir=None, report_dir=None, run_day=None):
    if data_15m is None or data_15m.empty:
        return [{
            "scenario": scenario_label,
            "period": period["label"],
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_return_pct": 0.0,
            "cagr": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "alpha_pct": 0.0,
            "psr_pct": 0.0,
            "min_trl": 0,
            "max_dd_duration_days": 0,
            "avg_trade_pnl": 0.0,
            "avg_hold_min": 0.0,
            "bh_return_pct": 0.0,
            "bh_cagr": 0.0,
            "initial_capital": 0.0,
            "final_capital": 0.0,
            "evaluation_mode": "no_data",
            "note": f"no data ({source})",
        } for period in ALL_PERIODS]

    engine, result = run_swing_15m_engine(prom, symbol, data_15m, slip, checkpoint_path=checkpoint_path)
    equity_series = build_equity_series(result, data_15m, warmup_bars=30)
    trade_frame = build_trade_frame(result, data_15m, prom)
    trade_artifacts = write_trade_artifacts(report_dir, run_day, symbol, scenario_label, trade_frame)

    rows = []
    for period in ALL_PERIODS:
        start_dt, end_dt = resolve_period_window(data_15m, period)
        if start_dt is None or end_dt is None:
            rows.append({
                "scenario": scenario_label,
                "period": period["label"],
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "cagr": 0.0,
                "max_dd_pct": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "alpha_pct": 0.0,
                "psr_pct": 0.0,
                "min_trl": 0,
                "max_dd_duration_days": 0,
                "avg_trade_pnl": 0.0,
                "avg_hold_min": 0.0,
                "bh_return_pct": 0.0,
                "bh_cagr": 0.0,
                "initial_capital": 0.0,
                "final_capital": 0.0,
                "note": "no data",
            })
            continue

        if period["label"] in STRICT_HOLDOUT_LABELS:
            isolated_data = slice_period_context(data_15m, start_dt, end_dt, pre_roll_days=ISOLATED_PRE_ROLL_DAYS)
            if isolated_data is None or isolated_data.empty:
                rows.append({
                    "scenario": scenario_label,
                    "period": period["label"],
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "net_pnl": 0.0,
                    "avg_profit": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                    "total_return_pct": 0.0,
                    "cagr": 0.0,
                    "max_dd_pct": 0.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "calmar": 0.0,
                    "alpha_pct": 0.0,
                    "psr_pct": 0.0,
                    "min_trl": 0,
                    "max_dd_duration_days": 0,
                    "avg_trade_pnl": 0.0,
                    "avg_hold_min": 0.0,
                    "bh_return_pct": 0.0,
                    "bh_cagr": 0.0,
                    "initial_capital": 0.0,
                    "final_capital": 0.0,
                    "evaluation_mode": "no_data",
                    "note": f"{source}|isolated|no data",
                })
                continue

            isolated_checkpoint = None
            if checkpoint_dir is not None:
                isolated_checkpoint = Path(checkpoint_dir) / f"{slugify(symbol)}_{scenario_label}_{period['label']}.pkl"
            isolated_engine, isolated_result = run_swing_15m_engine(
                prom,
                symbol,
                isolated_data,
                slip,
                checkpoint_path=isolated_checkpoint,
                trade_start_dt=pd.to_datetime(start_dt),
            )
            isolated_equity_series = build_equity_series(isolated_result, isolated_data, warmup_bars=30)
            rows.append(calculate_period_metrics(
                engine=isolated_engine,
                result=isolated_result,
                data_primary=isolated_data,
                equity_series=isolated_equity_series,
                start_dt=pd.to_datetime(start_dt),
                end_dt=pd.to_datetime(end_dt),
                period_label=period["label"],
                scenario_label=scenario_label,
                source=f"{source}|isolated",
                evaluation_mode="isolated_holdout",
            ))
        else:
            rows.append(calculate_period_metrics(
                engine=engine,
                result=result,
                data_primary=data_15m,
                equity_series=equity_series,
                start_dt=pd.to_datetime(start_dt),
                end_dt=pd.to_datetime(end_dt),
                period_label=period["label"],
                scenario_label=scenario_label,
                source=f"{source}|full_run_slice",
                evaluation_mode="full_run_slice",
            ))

    return rows


def run_all():
    console = Console()
    symbol = "NIFTY 50"

    console.print(f"[cyan]Loading 15m data for {symbol}...[/cyan]")
    prom = Prometheus(config_path=str(Path("prometheus/config/settings.yaml")), mode_override="paper")
    prom.data.configure_historical_fetch(source="auto", retries=2)

    data_15m, source = load_15m_data(prom, symbol, Path("dataset"))
    if data_15m is None or data_15m.empty:
        console.print("[red]No dataset found![/red]")
        return
    console.print(
        f"Loaded {len(data_15m)} bars ({data_15m['timestamp'].min()} to {data_15m['timestamp'].max()}) from {source}"
    )

    output_path = Path("reports")
    output_path.mkdir(parents=True, exist_ok=True)
    run_day = datetime.now().strftime("%Y-%m-%d")
    metrics_file = output_path / f"swing_15m_metrics_{run_day}.csv"
    state_file = output_path / f"swing_15m_metrics_{run_day}.state.json"
    checkpoint_dir = output_path / "swing_15m_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("base", None),
        ("slip_0.30", 0.30),
    ]

    data_signature = build_data_signature(data_15m, source, symbol)
    resume_state = load_resume_state(state_file, data_signature)
    completed_scenarios = dict(resume_state.get("completed_scenarios", {}))

    rows = []
    for scenario_label, slip in scenarios:
        if scenario_label in completed_scenarios:
            console.print(f"[green]Resuming with completed scenario {scenario_label} from saved state.[/green]")
            rows.extend(completed_scenarios[scenario_label])

    for scenario_label, slip in scenarios:
        if scenario_label in completed_scenarios:
            continue

        console.print(f"[yellow]Running scenario {scenario_label}...[/yellow]")
        checkpoint_path = checkpoint_dir / f"{slugify(symbol)}_{scenario_label}.pkl"
        scenario_rows = compute_scenario_rows(
            prom,
            data_15m,
            source,
            symbol,
            scenario_label,
            slip,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            report_dir=output_path,
            run_day=run_day,
        )
        console.print(
            f"[green]Trade-level artifacts refreshed for {scenario_label} under {output_path}[/green]"
        )
        completed_scenarios[scenario_label] = scenario_rows
        rows.extend(scenario_rows)
        save_resume_state(state_file, data_signature, completed_scenarios)
        write_metrics_csv(metrics_file, rows)
        console.print(f"[green]Saved progress to {metrics_file}[/green]")

    if not rows:
        console.print("No results generated.")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        console.print("No results generated.")
        return

    for col in ["win_rate", "net_pnl", "avg_profit", "avg_loss", "profit_factor", "total_return_pct", "cagr", "max_dd_pct", "sharpe", "sortino", "calmar", "alpha_pct", "psr_pct", "avg_trade_pnl", "avg_hold_min", "bh_return_pct", "bh_cagr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    write_metrics_csv(metrics_file, df.to_dict(orient="records"))
    console.print(f"[green]Saved metrics to {metrics_file}[/green]")
    try:
        state_file.unlink()
    except FileNotFoundError:
        pass

    for checkpoint_file in checkpoint_dir.glob(f"{slugify(symbol)}_*.pkl"):
        try:
            checkpoint_file.unlink()
        except FileNotFoundError:
            pass

    summary_labels = [period["label"] for period in SUMMARY_PERIODS]
    crisis_labels = [period["label"] for period in CRISIS_PERIODS]

    display_cols = [
        "scenario", "period", "evaluation_mode", "trades", "win_rate", "net_pnl", "total_return_pct",
        "cagr", "max_dd_pct", "sharpe", "sortino", "calmar", "alpha_pct",
        "profit_factor", "avg_trade_pnl", "avg_hold_min", "psr_pct", "min_trl",
    ]

    print("\nPeriod summary (full / 10y / 5y / train / test):")
    print(
        df[df["period"].isin(summary_labels)][display_cols]
        .sort_values(["period", "scenario"])
        .round(2)
        .to_string(index=False)
    )

    print("\nSelected crisis months:")
    print(
        df[df["period"].isin(crisis_labels)][display_cols]
        .sort_values(["period", "scenario"])
        .round(2)
        .to_string(index=False)
    )

if __name__ == "__main__":
    run_all()
