from pathlib import Path
from datetime import datetime
import os
import pandas as pd

# Prevent Telegram connection attempts during analysis
os.environ["PROM_TELEGRAM_BOT_TOKEN"] = ""
os.environ["PROM_TELEGRAM_CHAT_ID"] = ""

from prometheus.main import Prometheus


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


def filter_trades(trades, start_dt, end_dt):
    filtered = []
    for t in trades or []:
        dt = parse_dt(getattr(t, "entry_time", None))
        if dt and start_dt <= dt <= end_dt:
            filtered.append(t)
    return filtered


def summarize(trades):
    total = len(trades)
    pnls = [float(getattr(t, "net_pnl", 0.0) or 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    net = sum(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    win_rate = (len(wins) / total * 100.0) if total else 0.0
    return {
        "trades": total,
        "win_rate": win_rate,
        "net_pnl": net,
        "avg_profit": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    prom = Prometheus(config_path=str(Path("prometheus/config/settings.yaml")), mode_override="paper")
    prom.data.configure_historical_fetch(source="auto", retries=2)

    symbols = ["SENSEX", "NIFTY IT", "NIFTY FIN SERVICE", "NIFTY BANK", "NIFTY 50"]
    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 30, 23, 59, 59)

    rows = []
    mode_trades = {"swing": [], "intraday": []}

    for sym in symbols:
        # Swing: run on a larger daily window, then filter April trades
        data_daily = prom.data.fetch_historical(sym, days=200, interval="day", force_refresh=False)
        if data_daily is None or data_daily.empty:
            rows.append({"mode": "swing", "symbol": sym, "trades": 0, "win_rate": 0.0,
                         "net_pnl": 0.0, "avg_profit": 0.0, "avg_loss": 0.0, "note": "no data"})
        else:
            _, engine_s = prom._run_backtest_on_slice(
                data_slice=data_daily,
                symbol=sym,
                strategy_name="swing_200d",
                param_overrides=None,
                verbose=False,
                parrondo=False,
                entry_timing=False,
                entry_pullback_atr=0.3,
                entry_max_wait_bars=2,
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
            trades = filter_trades(getattr(engine_s, "trades", []), start, end)
            mode_trades["swing"].extend(trades)
            stats = summarize(trades)
            rows.append({"mode": "swing", "symbol": sym, **stats})

        # Intraday: 15-minute backtest for last 30 days, filter April trades
        data_intra = prom.data.fetch_historical(sym, days=30, interval="15minute", force_refresh=False)
        data_daily = prom.data.fetch_historical(sym, days=180, interval="day", force_refresh=False)
        if data_intra is None or data_intra.empty:
            rows.append({"mode": "intraday", "symbol": sym, "trades": 0, "win_rate": 0.0,
                         "net_pnl": 0.0, "avg_profit": 0.0, "avg_loss": 0.0, "note": "no data"})
        else:
            _, engine_i = prom._run_intraday_backtest_on_slice(
                data_slice=data_intra,
                data_daily=data_daily,
                symbol=sym,
                bar_interval="15minute",
                strategy_name="intraday_15m",
                parrondo=False,
                dd_throttle=True,
                vol_target=0.0,
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
                param_overrides=None,
                verbose=False,
            )
            trades = filter_trades(getattr(engine_i, "trades", []), start, end)
            mode_trades["intraday"].extend(trades)
            stats = summarize(trades)
            rows.append({"mode": "intraday", "symbol": sym, **stats})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["win_rate"] = df["win_rate"].round(1)
        df["net_pnl"] = df["net_pnl"].round(2)
        df["avg_profit"] = df["avg_profit"].round(2)
        df["avg_loss"] = df["avg_loss"].round(2)

    print("\nPer-symbol results (Apr 2026):")
    print(df.to_string(index=False))

    summary_rows = []
    for mode, trades in mode_trades.items():
        stats = summarize(trades)
        summary_rows.append({"mode": mode, **stats})

    df_sum = pd.DataFrame(summary_rows)
    if not df_sum.empty:
        df_sum["win_rate"] = df_sum["win_rate"].round(1)
        df_sum["net_pnl"] = df_sum["net_pnl"].round(2)
        df_sum["avg_profit"] = df_sum["avg_profit"].round(2)
        df_sum["avg_loss"] = df_sum["avg_loss"].round(2)

    print("\nTotals by mode (Apr 2026):")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
