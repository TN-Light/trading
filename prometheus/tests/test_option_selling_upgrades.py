# ============================================================================
# PROMETHEUS — Test Suite: Option Selling & Credit Spread Quantitative Upgrades
# ============================================================================
"""
Tests for the Option Selling upgrades:
1. Strictly 0-DTE / 1-DTE Expiry Afternoons (12:30 PM – 15:00 PM IST) & DTE >= 2 ban.
2. Dealer Long Gamma Confirmation (net_gex >= +1.5) & Short Gamma Ban (net_gex < -0.5).
3. Rigorous Macro Trend Co-Alignment (Bear Call = 1H Bear + Spot < VWAP; Bull Put = 1H Bull + Spot > VWAP).
4. Directional Symbol Lockout (No Martingale Ladders on stopped out symbol-direction pairs).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime, timedelta
from unittest.mock import patch

from prometheus.strategies.credit_spread import CreditSpreadStrategy


def create_mock_df(
    current_time_str="13:00",
    base_date=date(2026, 9, 22),
    spot=24000.0,
    trend="sideways",
    num_bars=30,
):
    """Create synthetic 15-minute bars ending at current_time_str on base_date."""
    h, m = map(int, current_time_str.split(":"))
    end_dt = datetime.combine(base_date, dtime(h, m))
    
    timestamps = [end_dt - timedelta(minutes=15 * (num_bars - 1 - i)) for i in range(num_bars)]
    
    # Generate prices with tiny variations around spot
    if trend == "bullish":
        # Consistently rising: Close > VWAP, EMA9 > EMA21
        prices = [spot - (num_bars - 1 - i) * 6.0 for i in range(num_bars)]
    elif trend == "bearish":
        # Consistently falling: Close < VWAP, EMA9 < EMA21
        prices = [spot + (num_bars - 1 - i) * 6.0 for i in range(num_bars)]
    else:
        prices = [spot + (i % 3 - 1) * 2.0 for i in range(num_bars)]
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + 4.0 for p in prices],
        "low": [p - 4.0 for p in prices],
        "close": prices,
        "volume": [1000] * num_bars,
    })
    return df


def create_mock_1h_df(
    base_date=date(2026, 9, 22),
    spot=24000.0,
    trend="bearish",
    num_bars=10,
):
    """Create synthetic 1-Hour candles for macro trend alignment."""
    end_dt = datetime.combine(base_date, dtime(13, 0))
    timestamps = [end_dt - timedelta(hours=num_bars - 1 - i) for i in range(num_bars)]
    
    if trend == "bullish":
        # Close > EMA20 > EMA50
        prices = [spot - (num_bars - 1 - i) * 40.0 for i in range(num_bars)]
    elif trend == "bearish":
        # Close < EMA20 < EMA50
        prices = [spot + (num_bars - 1 - i) * 40.0 for i in range(num_bars)]
    else:
        prices = [spot] * num_bars
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + 10.0 for p in prices],
        "low": [p - 10.0 for p in prices],
        "close": prices,
        "volume": [5000] * num_bars,
    })
    return df


class MockConstantChain:
    """Mock option chain returning realistic bids/asks."""
    def __init__(self, short_ltp: float = 35.0, long_ltp: float = 12.0):
        self.short_ltp = short_ltp
        self.long_ltp = long_ltp
        self._calls = 0

    def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
        self._calls += 1
        ltp = self.short_ltp if (self._calls % 2 == 1) else self.long_ltp
        return {
            "ltp": ltp,
            "ask": ltp + 0.5,
            "bid": ltp - 0.5,
            "tradingsymbol": f"{symbol}{int(strike)}{option_type}",
            "oi": 100000,
            "volume": 5000,
        }

    def get_option_ltp(self, tradingsymbol):
        return self.short_ltp

    def get_option_chain(self, symbol, spot_price=None, expiry_date=None):
        return pd.DataFrame()


def test_time_window_restriction():
    """Verify strictly 12:30 PM to 15:00 PM IST afternoon trading window."""
    strat = CreditSpreadStrategy(
        start_time=dtime(12, 30),
        end_time=dtime(15, 0),
        require_macro_trend=False,
        require_long_gamma=False,
    )
    chain = MockConstantChain()

    # Morning 10:30 AM bar should be skipped
    df_morning = create_mock_df(current_time_str="10:30")
    sig_morning = strat.evaluate_spread(df_morning, symbol="NIFTY 50", option_chain=chain)
    assert sig_morning is None, "Morning scan before 12:30 PM must be skipped!"

    # 12:15 PM bar should be skipped
    df_1215 = create_mock_df(current_time_str="12:15")
    sig_1215 = strat.evaluate_spread(df_1215, symbol="NIFTY 50", option_chain=chain)
    assert sig_1215 is None, "Scan before 12:30 PM must be skipped!"

    # 13:15 PM afternoon bar should be allowed
    df_afternoon = create_mock_df(current_time_str="13:15")
    base_date = df_afternoon.iloc[-1]["timestamp"].date()
    with patch("prometheus.strategies.credit_spread.get_expiry_date") as mock_exp:
        mock_exp.return_value = base_date
        sig_afternoon = strat.evaluate_spread(df_afternoon, symbol="NIFTY 50", option_chain=chain)
        assert sig_afternoon is not None, "Afternoon scan (13:15 PM) should be evaluated!"


def test_dte_restriction_ban_gte_2():
    """Verify banning DTE >= 2 across all indices (strictly 0-DTE or 1-DTE)."""
    strat = CreditSpreadStrategy(
        max_days_to_expiry=1,
        max_days_to_expiry_monthly=1,
        require_macro_trend=False,
        require_long_gamma=False,
    )
    chain = MockConstantChain()
    df = create_mock_df(current_time_str="13:00")
    base_date = df.iloc[-1]["timestamp"].date()

    with patch("prometheus.strategies.credit_spread.get_expiry_date") as mock_exp:
        # 1. DTE = 3 -> Skipped
        mock_exp.return_value = base_date + timedelta(days=3)
        sig = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain)
        assert sig is None, "DTE >= 2 (DTE=3) must be strictly skipped!"

        # 2. DTE = 2 -> Skipped
        mock_exp.return_value = base_date + timedelta(days=2)
        sig = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain)
        assert sig is None, "DTE >= 2 (DTE=2) must be strictly skipped!"

        # 3. 0-DTE -> Accepted
        mock_exp.return_value = base_date
        sig_0dte = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain)
        assert sig_0dte is not None, "0-DTE must be accepted!"

        # 4. 1-DTE -> Accepted
        mock_exp.return_value = base_date + timedelta(days=1)
        sig_1dte = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain)
        assert sig_1dte is not None, "1-DTE must be accepted!"


def test_dealer_gamma_regime_gates():
    """Verify Short Gamma ban (net_gex < -0.5) and Long Gamma confirmation (net_gex >= 1.5)."""
    strat = CreditSpreadStrategy(
        require_long_gamma=True,
        min_net_gex=1.5,
        short_gamma_ban_threshold=-0.5,
        require_macro_trend=False,
    )
    chain = MockConstantChain()
    df = create_mock_df(current_time_str="13:30")
    base_date = df.iloc[-1]["timestamp"].date()

    with patch("prometheus.strategies.credit_spread.get_expiry_date") as mock_exp:
        mock_exp.return_value = base_date
        # Short Gamma (net_gex = -0.8 < -0.5) -> BANNED
        sig_short_gamma = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain, net_gex=-0.8)
        assert sig_short_gamma is None, "Dealer Short Gamma (< -0.5) must strictly ban option selling!"

        # Sub-threshold GEX (net_gex = +0.8 < +1.5) -> REJECTED (Long Gamma confirmation required)
        sig_sub_gex = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain, net_gex=0.8)
        assert sig_sub_gex is None, "net_gex below +1.5 threshold must be rejected when require_long_gamma=True!"

        # Missing GEX telemetry -> REJECTED
        sig_no_gex = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain, net_gex=None)
        assert sig_no_gex is None, "Missing net_gex telemetry must be rejected when require_long_gamma=True!"

        # Confirmed Long Gamma (net_gex = +2.4 >= 1.5) -> ACCEPTED
        sig_long_gamma = strat.evaluate_spread(df, symbol="NIFTY 50", option_chain=chain, net_gex=2.4)
        assert sig_long_gamma is not None, "Confirmed Dealer Long Gamma (net_gex >= 1.5) must be accepted!"


def test_macro_trend_co_alignment():
    """Verify 1H macro trend co-alignment with spot vs VWAP."""
    strat = CreditSpreadStrategy(
        require_macro_trend=True,
        require_long_gamma=False,
    )
    chain = MockConstantChain()
    
    # 1. Bearish Macro Trend (1H Bear + Spot < VWAP) -> Generates BEAR_CALL_SPREAD
    df_bear = create_mock_df(current_time_str="13:30", spot=24000.0, trend="bearish")
    base_date = df_bear.iloc[-1]["timestamp"].date()
    df_1h_bear = create_mock_1h_df(base_date=base_date, spot=24000.0, trend="bearish")
    
    with patch("prometheus.strategies.credit_spread.get_expiry_date") as mock_exp:
        mock_exp.return_value = base_date
        sig_bear = strat.evaluate_spread(
            df_bear, symbol="NIFTY 50", option_chain=chain, df_1h=df_1h_bear
        )
        assert sig_bear is not None, "Bearish Macro + Spot < VWAP should generate a spread!"
        assert sig_bear["spread_type"] == "BEAR_CALL_SPREAD", "Should strictly generate BEAR_CALL_SPREAD!"

        # 2. Bullish Macro Trend (1H Bull + Spot > VWAP) -> Generates BULL_PUT_SPREAD
        df_bull = create_mock_df(current_time_str="13:30", spot=24000.0, trend="bullish")
        df_1h_bull = create_mock_1h_df(base_date=base_date, spot=24000.0, trend="bullish")
        sig_bull = strat.evaluate_spread(
            df_bull, symbol="NIFTY 50", option_chain=chain, df_1h=df_1h_bull
        )
        assert sig_bull is not None, "Bullish Macro + Spot > VWAP should generate a spread!"
        assert sig_bull["spread_type"] == "BULL_PUT_SPREAD", "Should strictly generate BULL_PUT_SPREAD!"

        # 3. Macro Trend Conflict: 1H is Bullish, but spot is below VWAP -> REJECTED
        df_conflict = create_mock_df(current_time_str="13:30", spot=24000.0, trend="bearish")
        sig_conflict = strat.evaluate_spread(
            df_conflict, symbol="NIFTY 50", option_chain=chain, df_1h=df_1h_bull
        )
        assert sig_conflict is None, "Conflicting macro trend must be rejected!"


def test_directional_symbol_lockout():
    """Verify directional symbol lockout blocks repeat stop-outs on same (symbol, direction)."""
    strat = CreditSpreadStrategy(
        require_macro_trend=False,
        require_long_gamma=False,
    )
    chain = MockConstantChain()
    df_bear = create_mock_df(current_time_str="13:30", spot=24000.0, trend="bearish")
    base_date = df_bear.iloc[-1]["timestamp"].date()

    with patch("prometheus.strategies.credit_spread.get_expiry_date") as mock_exp:
        mock_exp.return_value = base_date
        # Without lockout, generates BEAR_CALL_SPREAD
        sig_allowed = strat.evaluate_spread(df_bear, symbol="SENSEX", option_chain=chain)
        assert sig_allowed is not None
        assert sig_allowed["spread_type"] == "BEAR_CALL_SPREAD"

        # With (SENSEX, BEAR_CALL_SPREAD) in stopped_out_directions, entry must be rejected
        stopped_set = {("SENSEX", "BEAR_CALL_SPREAD")}
        sig_blocked = strat.evaluate_spread(
            df_bear, symbol="SENSEX", option_chain=chain, stopped_out_directions=stopped_set
        )
        assert sig_blocked is None, "Directional symbol lockout must block repeat entry on (SENSEX, BEAR_CALL_SPREAD)!"

        # Another symbol (NIFTY 50) is not affected
        sig_nifty = strat.evaluate_spread(
            df_bear, symbol="NIFTY 50", option_chain=chain, stopped_out_directions=stopped_set
        )
        assert sig_nifty is not None, "NIFTY 50 should not be blocked by SENSEX lockout!"
