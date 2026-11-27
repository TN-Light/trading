import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from prometheus.strategies.credit_spread import CreditSpreadStrategy
from prometheus.signals.tier_classifier import classify_signal_tier


class MockOptionChain:
    def get_real_premium(self, symbol, strike, opt_str, expiry="", spot_price=0.0):
        # Short strike yields Rs 28, long hedge yields Rs 10 -> Net credit Rs 18 (>= 10% of 100 width)
        dist = abs(strike - spot_price)
        ltp = 28.0 if dist <= 300 else 10.0
        return {
            "ltp": ltp,
            "tradingsymbol": f"NIFTY{int(strike)}{opt_str}",
            "oi": 100000,
        }

    def get_option_chain(self, symbol, spot_price=0.0, expiry_date=""):
        return pd.DataFrame()


def _create_sideways_15m_df(spot=25000.0):
    base = datetime(2026, 11, 25, 9, 15)
    rows = []
    for i in range(20):
        ts = base + timedelta(minutes=15 * i)
        # Oscillating slightly around spot
        offset = 15.0 if i % 2 == 0 else -15.0
        c = spot + offset
        rows.append({
            "timestamp": ts,
            "open": spot,
            "high": c + 20.0,
            "low": c - 20.0,
            "close": c,
            "volume": 0,
        })
    return pd.DataFrame(rows)


def _create_neutral_1h_df(spot=25000.0):
    base = datetime(2026, 11, 24, 9, 15)
    rows = []
    # Oscillate so Close < EMA20 while EMA20 > EMA50 (neither bull nor bear alignment)
    for i in range(15):
        ts = base + timedelta(hours=i)
        c = spot + (40.0 if i < 12 else -10.0)
        rows.append({
            "timestamp": ts,
            "open": spot,
            "high": c + 20.0,
            "low": c - 20.0,
            "close": c,
            "volume": 0,
        })
    return pd.DataFrame(rows)


def test_neutral_1h_sideways_generates_spread_with_require_macro_trend():
    strat = CreditSpreadStrategy(
        require_macro_trend=True,
        require_long_gamma=False,
        min_credit_pct=0.10,
        max_days_to_expiry=3,
    )
    df_15m = _create_sideways_15m_df(25000.0)
    df_1h_neutral = _create_neutral_1h_df(25000.0)
    chain = MockOptionChain()
    today_d = df_15m.iloc[-1]["timestamp"].date()

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_d):
        sig = strat.evaluate_spread(
            df_15m,
            symbol="NIFTY 50",
            option_chain=chain,
            df_1h=df_1h_neutral,
            net_gex=0.0,
        )
    assert sig is not None, "Neutral 1H sideways market must generate a valid credit spread!"
    assert sig["net_credit"] >= 10.0


def test_neutral_gamma_accepted_short_gamma_rejected():
    strat = CreditSpreadStrategy(
        require_macro_trend=True,
        require_long_gamma=False,
        short_gamma_ban_threshold=-0.5,
        min_credit_pct=0.10,
    )
    df_15m = _create_sideways_15m_df(25000.0)
    df_1h_neutral = _create_neutral_1h_df(25000.0)
    chain = MockOptionChain()
    today_d = df_15m.iloc[-1]["timestamp"].date()

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_d):
        # Neutral Gamma (net_gex = 0.0) -> Accepted
        sig_neutral = strat.evaluate_spread(
            df_15m, symbol="NIFTY 50", option_chain=chain, df_1h=df_1h_neutral, net_gex=0.0
        )
        assert sig_neutral is not None

        # Short Gamma (net_gex = -5.0) -> Strictly Banned
        sig_short_gex = strat.evaluate_spread(
            df_15m, symbol="NIFTY 50", option_chain=chain, df_1h=df_1h_neutral, net_gex=-5.0
        )
        assert sig_short_gex is None


def test_1_to_3_dte_spread_routed_strictly_to_tier_c_paper():
    strat = CreditSpreadStrategy(
        require_macro_trend=True,
        require_long_gamma=False,
        min_credit_pct=0.10,
        max_days_to_expiry=3,
    )
    df_15m = _create_sideways_15m_df(25000.0)
    df_1h_neutral = _create_neutral_1h_df(25000.0)
    chain = MockOptionChain()
    today_d = df_15m.iloc[-1]["timestamp"].date()
    expiry_2dte = today_d + timedelta(days=2)

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=expiry_2dte):
        sig = strat.evaluate_spread(
            df_15m, symbol="NIFTY 50", option_chain=chain, df_1h=df_1h_neutral, net_gex=0.0
        )
    assert sig is not None
    assert sig["days_to_expiry"] == 2

    # Classify tier -> MUST be Tier C Paper Only (Live Capital protected strictly at 0-DTE)
    sig["bar_timestamp"] = df_15m.iloc[-1]["timestamp"]
    sig["score"] = 9.0
    sig["otm_sigma"] = 2.0
    tier_res = classify_signal_tier(sig)
    assert tier_res["tier"] == "C"
    assert tier_res["is_live_eligible"] is False


def test_post_1415_allows_credit_spread_but_blocks_option_buying():
    """Verify allow_option_buying=False skips PA option buying while still evaluating CreditSpreadStrategy."""
    from unittest.mock import MagicMock
    from prometheus.main import Prometheus

    p = Prometheus()
    p.data = MagicMock()
    df_15m = _create_sideways_15m_df(25000.0)
    df_1h = _create_neutral_1h_df(25000.0)
    p.data.fetch_intraday.return_value = df_15m
    p.data.fetch_historical.return_value = df_1h
    p.data.fetch_options_chain.return_value = pd.DataFrame()
    p.data.get_vix.return_value = 15.0
    p.data.angelone_options = MockOptionChain()

    p._pa_momentum_scanner = MagicMock()
    p._credit_spread_strategy = MagicMock()
    p._credit_spread_strategy.evaluate_spread.return_value = {
        "strategy_type": "credit_spread",
        "spread_type": "BULL_PUT_SPREAD",
        "action": "SELL_PUT_SPREAD",
        "net_credit": 18.0,
        "tradingsymbol": "NIFTY25000PE",
        "expiry": df_15m.iloc[-1]["timestamp"].date().isoformat(),
        "bar_timestamp": df_15m.iloc[-1]["timestamp"],
        "otm_sigma": 2.0,
        "signal_score": 9.0,
    }

    sig = p._get_intraday_signal_for_execution(
        "NIFTY 50", "15minute", False, allow_option_buying=False
    )

    p._pa_momentum_scanner.evaluate_bar.assert_not_called()
    p._credit_spread_strategy.evaluate_spread.assert_called_once()
    assert sig is not None
    assert sig["strategy_type"] == "credit_spread"
