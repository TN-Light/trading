"""
Unit tests for Barbell Option Selling Relaxed Gate Upgrades:
1. Dynamic Day-Range Gate & 2-Hour Rolling Consolidation Check
2. DTE Decoupling for Monthly-Only Indices (Bank Nifty & FinNifty)
3. Dynamic 0-DTE Strike Placement (Session High/Low + 1 strike, min 2 OTM)
4. 0-DTE 5% Credit Hurdle Acceptance (Terminal Expiry Theta Harvesting)
5. Full Settings Injection from settings.yaml into main.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch

from prometheus.strategies.credit_spread import CreditSpreadStrategy
from prometheus.utils.indian_market import get_atm_strike, get_strike_interval


def _generate_synthetic_df(
    start_dt: datetime,
    bars_count: int = 25,
    base_price: float = 24000.0,
    open_gap_high: float = None,
    open_gap_low: float = None,
    tight_last_8: bool = False,
    expanding_last_8: bool = False,
    trend_bias: str = "bearish",  # "bearish" -> Bear Call, "bullish" -> Bull Put
) -> pd.DataFrame:
    """Generate realistic 15M candles for testing credit spread gates."""
    timestamps = [start_dt + timedelta(minutes=15 * i) for i in range(bars_count)]
    rows = []
    curr = base_price

    for i, ts in enumerate(timestamps):
        if i == 0 and open_gap_high is not None and open_gap_low is not None:
            # First bar has large morning gap
            o = base_price
            h = open_gap_high
            l = open_gap_low
            c = base_price + 20.0
        elif tight_last_8 and i >= (bars_count - 8):
            # Consolidate in tight 10 pt range
            o = curr
            h = o + 5.0
            l = o - 5.0
            c = o + (1.0 if i % 2 == 0 else -1.0)
        elif expanding_last_8 and i >= (bars_count - 8):
            # Expanding volatility in last 8 bars
            o = curr
            h = o + 40.0
            l = o - 40.0
            c = o + 25.0
        else:
            o = curr
            h = o + 10.0
            l = o - 10.0
            c = o + (2.0 if trend_bias == "bullish" else -2.0)

        rows.append({
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 100000,
        })
        curr = c

    return pd.DataFrame(rows)


class MockConstantChain:
    """Mock option chain returning distinct premiums for short and long hedge legs."""
    def __init__(self, short_ltp: float = 40.0, long_ltp: float = 15.0):
        self.short_ltp = short_ltp
        self.long_ltp = long_ltp
        self._calls = 0

    def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
        self._calls += 1
        # In evaluate_spread, sq is queried first (calls % 2 == 1), then lq (calls % 2 == 0)
        ltp = self.short_ltp if (self._calls % 2 == 1) else self.long_ltp
        return {
            "ltp": ltp,
            "tradingsymbol": f"{symbol}{int(strike)}{option_type}",
        }

    def get_option_ltp(self, tradingsymbol):
        return self.short_ltp

    def get_option_chain(self, symbol, spot_price=None, expiry_date=None):
        return pd.DataFrame()


# ============================================================================
# 1. Rolling Consolidation Override on Morning Gaps
# ============================================================================

def test_credit_spread_rolling_consolidation_override_permits_trade():
    """Verify wide morning gap exceeding max_range_pct is overridden when 2-hour range <= 0.8% spot."""
    strategy = CreditSpreadStrategy(max_range_pct=0.018)  # 1.8% threshold
    # Spot 24000: 1.8% = 432 pts threshold
    # Create opening bar with 550 pt range (exceeds 432 pts threshold)
    # But last 8 bars consolidate tightly (range ~ 15 pts << 24000 * 0.008 = 192 pts)
    start_dt = datetime(2026, 8, 25, 9, 15)  # Tuesday (0-DTE or near)
    df = _generate_synthetic_df(
        start_dt=start_dt,
        bars_count=18,
        base_price=24000.0,
        open_gap_high=24350.0,
        open_gap_low=23800.0,  # 550 pt day range
        tight_last_8=True,
    )

    chain = MockConstantChain(short_ltp=35.0, long_ltp=12.0)

    # Evaluate spread
    sig = strategy.evaluate_spread(
        df,
        symbol="NIFTY 50",
        capital=100000,
        option_chain=chain,
    )

    # Should NOT be None because rolling consolidation override rescued it
    assert sig is not None, "Consolidation override should allow trade despite wide morning gap"
    assert sig["strategy_type"] == "credit_spread"
    assert sig["net_credit"] == 23.0


def test_credit_spread_rolling_consolidation_rejected_when_recent_range_wide():
    """Verify that if day range is wide AND last 8 bars are also wide/expanding, trade is rejected."""
    strategy = CreditSpreadStrategy(max_range_pct=0.018)
    start_dt = datetime(2026, 8, 25, 9, 15)
    df = _generate_synthetic_df(
        start_dt=start_dt,
        bars_count=18,
        base_price=24000.0,
        open_gap_high=24350.0,
        open_gap_low=23800.0,  # 550 pt range
        expanding_last_8=True,  # 2-hour range expands > 2x ATR and > 0.8%
    )

    chain = MockConstantChain(short_ltp=35.0, long_ltp=12.0)

    sig = strategy.evaluate_spread(
        df,
        symbol="NIFTY 50",
        capital=100000,
        option_chain=chain,
    )

    assert sig is None, "Expanding recent range must reject trade as strong trend day"


def test_credit_spread_rolling_consolidation_insufficient_bars_rejected():
    """Verify that if day range is wide and fewer than 8 bars elapsed today, override cannot trigger."""
    strategy = CreditSpreadStrategy(max_range_pct=0.018)
    # 5 bars today (market opened at 9:15, now 10:30). Wide morning gap.
    start_dt = datetime(2026, 8, 25, 9, 15)
    df = _generate_synthetic_df(
        start_dt=start_dt,
        bars_count=5,
        base_price=24000.0,
        open_gap_high=24350.0,
        open_gap_low=23800.0,  # 550 pt day range
        tight_last_8=False,
    )
    # Ensure full df has at least 15 bars total so length check passes, by adding yesterday's bars
    yest_dt = datetime(2026, 8, 24, 13, 0)
    yest_df = _generate_synthetic_df(start_dt=yest_dt, bars_count=12, base_price=24000.0)
    full_df = pd.concat([yest_df, df], ignore_index=True)

    chain = MockConstantChain(short_ltp=35.0, long_ltp=12.0)
    sig = strategy.evaluate_spread(full_df, symbol="NIFTY 50", capital=100000, option_chain=chain)
    assert sig is None, "With only 5 bars today, 2-hour consolidation cannot be confirmed; must reject wide day range"


def test_credit_spread_rolling_consolidation_multi_day_df_no_cross_day_contamination():
    """Verify rolling consolidation correctly isolates today's 8 bars without cross-day contamination."""
    strategy = CreditSpreadStrategy(max_range_pct=0.018)
    # Yesterday: 15 bars trading around 23500
    yest_dt = datetime(2026, 8, 24, 11, 45)
    yest_df = _generate_synthetic_df(start_dt=yest_dt, bars_count=15, base_price=23500.0)

    # Today: 10 bars (9:15 to 11:45). Bar 1 has huge morning gap from 23800 to 24350.
    # Subsequent 9 bars consolidate tightly around 24000 (range 10 pts)
    today_dt = datetime(2026, 8, 25, 9, 15)
    today_df = _generate_synthetic_df(
        start_dt=today_dt,
        bars_count=10,
        base_price=24000.0,
        open_gap_high=24350.0,
        open_gap_low=23800.0,
        tight_last_8=True,
    )

    full_df = pd.concat([yest_df, today_df], ignore_index=True)
    chain = MockConstantChain(short_ltp=35.0, long_ltp=12.0)
    sig = strategy.evaluate_spread(full_df, symbol="NIFTY 50", capital=100000, option_chain=chain)

    assert sig is not None, "2-hour consolidation on today's session must override wide morning gap on multi-day df"
    assert sig["strategy_type"] == "credit_spread"


# ============================================================================
# 2. DTE Decoupling for Monthly-Only Indices (Bank Nifty & FinNifty)
# ============================================================================

def test_credit_spread_dte_decoupling_banknifty_and_finnifty():
    """Verify Bank Nifty and FinNifty allow up to 14 DTE while Nifty 50 enforces 3 DTE."""
    strategy = CreditSpreadStrategy(
        max_days_to_expiry=3,
        max_days_to_expiry_monthly=14,
    )

    test_date = date(2026, 9, 1)
    df = _generate_synthetic_df(
        start_dt=datetime(2026, 9, 1, 10, 0),
        bars_count=16,
        base_price=50000.0,
    )

    # Test Bank Nifty with 7 DTE (accepted: <= 14 DTE)
    chain_bn = MockConstantChain(short_ltp=70.0, long_ltp=20.0)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=7)):
        sig_bn_7 = strategy.evaluate_spread(df, symbol="NIFTY BANK", capital=100000, option_chain=chain_bn)
        assert sig_bn_7 is not None, "Bank Nifty with 7 DTE must be accepted under 14 DTE monthly policy"

    # Test Bank Nifty with 15 DTE (rejected: > 14 DTE)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=15)):
        sig_bn_15 = strategy.evaluate_spread(df, symbol="NIFTY BANK", capital=100000, option_chain=chain_bn)
        assert sig_bn_15 is None, "Bank Nifty with 15 DTE must be rejected (> 14 DTE)"

    # Test FinNifty with 10 DTE (accepted: <= 14 DTE)
    df_fin = _generate_synthetic_df(
        start_dt=datetime(2026, 9, 1, 10, 0),
        bars_count=16,
        base_price=23000.0,
    )
    chain_fin = MockConstantChain(short_ltp=40.0, long_ltp=15.0)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=10)):
        sig_fin_10 = strategy.evaluate_spread(df_fin, symbol="NIFTY FIN SERVICE", capital=100000, option_chain=chain_fin)
        assert sig_fin_10 is not None, "FinNifty with 10 DTE must be accepted under 14 DTE monthly policy"

    # Test Nifty 50 with 7 DTE (rejected: > 3 DTE)
    df_nifty = _generate_synthetic_df(
        start_dt=datetime(2026, 9, 1, 10, 0),
        bars_count=16,
        base_price=24000.0,
    )
    chain_nifty = MockConstantChain(short_ltp=40.0, long_ltp=15.0)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=7)):
        sig_nifty_7 = strategy.evaluate_spread(df_nifty, symbol="NIFTY 50", capital=100000, option_chain=chain_nifty)
        assert sig_nifty_7 is None, "Nifty 50 with 7 DTE must be rejected (> 3 DTE limit)"

    # Test Nifty 50 with 2 DTE (accepted: <= 3 DTE)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=2)):
        sig_nifty_2 = strategy.evaluate_spread(df_nifty, symbol="NIFTY 50", capital=100000, option_chain=chain_nifty)
        assert sig_nifty_2 is not None, "Nifty 50 with 2 DTE must be accepted (<= 3 DTE limit)"


def test_credit_spread_dte_decoupling_with_none_max_days_to_expiry():
    """Verify max_days_to_expiry_monthly works even when max_days_to_expiry is None."""
    strategy = CreditSpreadStrategy(
        max_days_to_expiry=None,
        max_days_to_expiry_monthly=14,
    )
    test_date = date(2026, 9, 1)
    df = _generate_synthetic_df(
        start_dt=datetime(2026, 9, 1, 10, 0),
        bars_count=16,
        base_price=50000.0,
    )
    chain_bn = MockConstantChain(short_ltp=70.0, long_ltp=20.0)

    # Bank Nifty with 15 DTE must be rejected (> 14 DTE) even when max_days_to_expiry is None
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=15)):
        sig_bn_15 = strategy.evaluate_spread(df, symbol="NIFTY BANK", capital=100000, option_chain=chain_bn)
        assert sig_bn_15 is None, "Bank Nifty with 15 DTE must be rejected (> 14 DTE) when max_days_to_expiry is None"

    # Bank Nifty with 7 DTE must be accepted (<= 14 DTE)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=7)):
        sig_bn_7 = strategy.evaluate_spread(df, symbol="NIFTY BANK", capital=100000, option_chain=chain_bn)
        assert sig_bn_7 is not None, "Bank Nifty with 7 DTE must be accepted (<= 14 DTE)"


def test_credit_spread_dte_decoupling_symbol_aliases():
    """Verify BANKNIFTY and FINNIFTY aliases properly decouple to max_days_to_expiry_monthly."""
    strategy = CreditSpreadStrategy(
        max_days_to_expiry=3,
        max_days_to_expiry_monthly=14,
    )
    test_date = date(2026, 9, 1)
    df = _generate_synthetic_df(start_dt=datetime(2026, 9, 1, 10, 0), bars_count=16, base_price=50000.0)
    chain_bn = MockConstantChain(short_ltp=70.0, long_ltp=20.0)  # net 50.0 >= 45.0 (15% of 300 pt width)
    chain_fin = MockConstantChain(short_ltp=50.0, long_ltp=15.0)  # net 35.0 >= 22.5 (15% of 150 pt width)

    # BANKNIFTY with 8 DTE (accepted: <= 14 DTE, even though > 3 DTE)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=8)):
        sig_bn = strategy.evaluate_spread(df, symbol="BANKNIFTY", capital=100000, option_chain=chain_bn)
        assert sig_bn is not None, "BANKNIFTY alias must be accepted under monthly DTE limit"

    # FINNIFTY with 8 DTE (accepted: <= 14 DTE, even though > 3 DTE)
    df_fin = _generate_synthetic_df(start_dt=datetime(2026, 9, 1, 10, 0), bars_count=16, base_price=23000.0)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=test_date + timedelta(days=8)):
        sig_fin = strategy.evaluate_spread(df_fin, symbol="FINNIFTY", capital=100000, option_chain=chain_fin)
        assert sig_fin is not None, "FINNIFTY alias must be accepted under monthly DTE limit"


# ============================================================================
# 3. Dynamic 0-DTE Strike Placement (Session High/Low + 1 strike, min 2 OTM)
# ============================================================================

def test_credit_spread_0dte_strike_placement_bear_call():
    """Verify 0-DTE Bear Call short strike is max(get_atm_strike(today_high + interval), atm_strike + 2*interval)."""
    strategy = CreditSpreadStrategy()
    today_dt = date(2026, 9, 22)  # Tuesday 0-DTE

    df = _generate_synthetic_df(
        start_dt=datetime(2026, 9, 22, 10, 0),
        bars_count=16,
        base_price=24000.0,
    )
    # Ensure today_bars has high of 24120.0 and close is 23980 (bearish below vwap)
    df.loc[df.index[-5], "high"] = 24120.0
    df.loc[df.index[-1], "close"] = 23980.0
    df.loc[df.index[-1], "open"] = 23985.0

    chain = MockConstantChain(short_ltp=30.0, long_ltp=10.0)

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=chain)

    assert sig is not None, "0-DTE Bear Call spread should evaluate cleanly"
    assert sig["spread_type"] == "BEAR_CALL_SPREAD"

    today_bars = df[df["timestamp"].dt.date == today_dt]
    today_high = float(today_bars["high"].max())
    close = float(df.iloc[-1]["close"])
    atm_strike = get_atm_strike(close, "NIFTY 50")
    interval = get_strike_interval("NIFTY 50")

    expected_strike = max(
        strategy.get_atm_strike(today_high + interval, "NIFTY 50"),
        atm_strike + (2 * interval),
    )
    assert sig["short_strike"] == expected_strike, f"Expected {expected_strike}, got {sig['short_strike']}"
    # Verify hedge leg is hedge_otm_steps (3) further OTM
    assert sig["long_strike"] == sig["short_strike"] + (3 * interval)


def test_credit_spread_0dte_strike_placement_bull_put():
    """Verify 0-DTE Bull Put short strike is min(get_atm_strike(today_low - interval), atm_strike - 2*interval)."""
    strategy = CreditSpreadStrategy()
    today_dt = date(2026, 9, 22)  # Tuesday 0-DTE

    # Create bullish context: close > vwap
    df = _generate_synthetic_df(
        start_dt=datetime(2026, 9, 22, 10, 0),
        bars_count=16,
        base_price=24000.0,
        trend_bias="bullish",
    )
    df.loc[df.index[-5], "low"] = 23880.0
    df.loc[df.index[-1], "close"] = 24040.0
    df.loc[df.index[-1], "open"] = 24035.0

    chain = MockConstantChain(short_ltp=30.0, long_ltp=10.0)

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=chain)

    assert sig is not None, "0-DTE Bull Put spread should evaluate cleanly"
    assert sig["spread_type"] == "BULL_PUT_SPREAD"

    today_bars = df[df["timestamp"].dt.date == today_dt]
    today_low = float(today_bars["low"].min())
    close = float(df.iloc[-1]["close"])
    atm_strike = get_atm_strike(close, "NIFTY 50")
    interval = get_strike_interval("NIFTY 50")

    expected_strike = min(
        strategy.get_atm_strike(today_low - interval, "NIFTY 50"),
        atm_strike - (2 * interval),
    )
    assert sig["short_strike"] == expected_strike, f"Expected {expected_strike}, got {sig['short_strike']}"
    assert sig["long_strike"] == sig["short_strike"] - (3 * interval)


def test_credit_spread_0dte_strike_placement_anchoring_above_high_when_gap_ran():
    """Verify that when session high is significantly elevated, 0-DTE Bear Call anchors above session high."""
    strategy = CreditSpreadStrategy()
    today_dt = date(2026, 9, 22)
    df = _generate_synthetic_df(start_dt=datetime(2026, 9, 22, 10, 0), bars_count=16, base_price=24000.0)

    # Morning high reached 24280, now pulled back to 24020
    df.loc[df.index[2], "high"] = 24280.0
    df.loc[df.index[-1], "close"] = 24020.0
    df.loc[df.index[-1], "open"] = 24025.0

    chain = MockConstantChain(short_ltp=30.0, long_ltp=10.0)
    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=chain)

    assert sig is not None
    # 24280 + 50 = 24330 -> get_atm_strike(24330) = 24350
    # atm_strike = 24000, atm_strike + 2*50 = 24100
    # short_strike should be max(24350, 24100) = 24350!
    assert sig["short_strike"] == 24350.0, f"Expected 24350.0 anchored above session high, got {sig['short_strike']}"


# ============================================================================
# 4. 0-DTE 5% Credit Hurdle Acceptance
# ============================================================================

def test_credit_spread_0dte_5pct_credit_hurdle_accepted():
    """Verify on 0-DTE, net credit >= 5% of strike width (₹7.50 on 150-pt width) is accepted."""
    strategy = CreditSpreadStrategy()
    today_dt = date(2026, 9, 22)
    df = _generate_synthetic_df(start_dt=datetime(2026, 9, 22, 10, 0), bars_count=16, base_price=24000.0)

    # Strike width = 150 pts (3 * 50).
    # 5% hurdle = 7.50 pts.
    # Test net_credit = 8.00 pts (short=18.0, long=10.0 -> net=8.00 >= 7.50):
    # Under old 8% hurdle (12.0 pts), 8.00 would be rejected.
    # Under new 5% hurdle, 8.00 must be accepted!
    class ThinCreditChain:
        def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
            if strike > 24000:
                return {"ltp": 10.0 if strike > 24100 else 18.0, "tradingsymbol": f"NIFTY{int(strike)}CE"}
            else:
                return {"ltp": 10.0 if strike < 23900 else 18.0, "tradingsymbol": f"NIFTY{int(strike)}PE"}

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=ThinCreditChain())

    assert sig is not None, "8.00 pt credit on 150 pt width (5.33%) must be accepted under 5% 0-DTE hurdle"
    assert sig["net_credit"] == 8.00


def test_credit_spread_0dte_sub_5pct_credit_hurdle_rejected():
    """Verify on 0-DTE, net credit < 5% of strike width (e.g. ₹6.00 on 150-pt width) is rejected."""
    strategy = CreditSpreadStrategy()
    today_dt = date(2026, 9, 22)
    df = _generate_synthetic_df(start_dt=datetime(2026, 9, 22, 10, 0), bars_count=16, base_price=24000.0)

    # 4% credit: net credit = 6.00 pts (short=16.0, long=10.0 -> net=6.00 < 7.50)
    class Sub5PctChain:
        def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
            if strike > 24000:
                return {"ltp": 10.0 if strike > 24100 else 16.0, "tradingsymbol": f"NIFTY{int(strike)}CE"}
            else:
                return {"ltp": 10.0 if strike < 23900 else 16.0, "tradingsymbol": f"NIFTY{int(strike)}PE"}

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=Sub5PctChain())

    assert sig is None, "6.00 pt credit on 150 pt width (4.0%) must be rejected as below 5% hurdle"


def test_credit_spread_0dte_exact_boundary_hurdle():
    """Verify exact 5.0% boundary hurdle behavior (₹7.50 on 150-pt width accepted, ₹7.49 rejected)."""
    strategy = CreditSpreadStrategy()
    today_dt = date(2026, 9, 22)
    df = _generate_synthetic_df(start_dt=datetime(2026, 9, 22, 10, 0), bars_count=16, base_price=24000.0)

    # 1. Test exactly 7.50 pts credit (short=17.50, long=10.00 -> net=7.50 == 150 * 0.05)
    class Exact750Chain:
        def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
            if strike > 24000:
                return {"ltp": 10.0 if strike > 24100 else 17.50, "tradingsymbol": f"NIFTY{int(strike)}CE"}
            return {"ltp": 10.0 if strike < 23900 else 17.50, "tradingsymbol": f"NIFTY{int(strike)}PE"}

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig_exact = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=Exact750Chain())
    assert sig_exact is not None, "Exactly 7.50 pts credit (5.0%) must be accepted on 150 pt width"
    assert sig_exact["net_credit"] == 7.50

    # 2. Test 7.49 pts credit (short=17.49, long=10.00 -> net=7.49 < 7.50)
    class Sub750Chain:
        def get_real_premium(self, symbol, strike, option_type, expiry=None, spot_price=None):
            if strike > 24000:
                return {"ltp": 10.0 if strike > 24100 else 17.49, "tradingsymbol": f"NIFTY{int(strike)}CE"}
            return {"ltp": 10.0 if strike < 23900 else 17.49, "tradingsymbol": f"NIFTY{int(strike)}PE"}

    with patch("prometheus.strategies.credit_spread.get_expiry_date", return_value=today_dt):
        sig_sub = strategy.evaluate_spread(df, symbol="NIFTY 50", capital=100000, option_chain=Sub750Chain())
    assert sig_sub is None, "7.49 pts credit (4.99%) must be rejected as below 5.0% hurdle"


# ============================================================================
# 5. Full Settings Injection in main.py
# ============================================================================

def test_credit_spread_settings_injection():
    """Verify main.py instantiates CreditSpreadStrategy using full settings from settings.yaml."""
    from prometheus.config import get

    cs_cfg = get("intraday.credit_spread", {})
    assert cs_cfg.get("enabled") is True
    assert cs_cfg.get("strike_otm_steps") == 1
    assert cs_cfg.get("hedge_otm_steps") == 3
    assert cs_cfg.get("target_decay_pct") == 0.70
    assert cs_cfg.get("breakeven_decay_pct") == 0.50
    assert cs_cfg.get("max_loss_multiplier") == 1.5
    assert cs_cfg.get("min_credit_pct") == 0.15
    assert cs_cfg.get("max_days_to_expiry") == 1
    assert cs_cfg.get("max_days_to_expiry_monthly") == 1
    assert cs_cfg.get("max_range_pct") == 0.018

    # Simulate instantiation code from main.py
    strategy = CreditSpreadStrategy(
        strike_otm_steps=int(cs_cfg.get("strike_otm_steps", 1)),
        hedge_otm_steps=int(cs_cfg.get("hedge_otm_steps", 3)),
        target_decay_pct=float(cs_cfg.get("target_decay_pct", 0.70)),
        breakeven_decay_pct=float(cs_cfg.get("breakeven_decay_pct", 0.50)),
        max_loss_multiplier=float(cs_cfg.get("max_loss_multiplier", 1.5)),
        min_credit_pct=float(cs_cfg.get("min_credit_pct", 0.15)),
        max_days_to_expiry=int(cs_cfg.get("max_days_to_expiry", 1)),
        max_days_to_expiry_monthly=int(cs_cfg.get("max_days_to_expiry_monthly", 1)),
        max_range_pct=float(cs_cfg.get("max_range_pct", 0.018)),
    )

    assert strategy.strike_otm_steps == 1
    assert strategy.hedge_otm_steps == 3
    assert strategy.target_decay_pct == 0.70
    assert strategy.breakeven_decay_pct == 0.50
    assert strategy.max_loss_multiplier == 1.5
    assert strategy.min_credit_pct == 0.15
    assert strategy.max_days_to_expiry == 1
    assert strategy.max_days_to_expiry_monthly == 1
    assert strategy.max_range_pct == 0.018


def test_main_py_actual_instantiation_from_settings():
    """Verify Prometheus creates _credit_spread_strategy with settings.yaml parameters via _get_intraday_signal_for_execution."""
    from prometheus.main import Prometheus

    p = Prometheus()
    if hasattr(p, "_credit_spread_strategy"):
        delattr(p, "_credit_spread_strategy")

    # Mock data methods so _get_intraday_signal_for_execution runs safely
    p.data = MagicMock()
    p.data.get_vix = MagicMock(return_value=14.0)
    p.data.fetch_intraday = MagicMock(return_value=None)
    p._pa_momentum_scanner = MagicMock()
    p._pa_momentum_scanner.evaluate_bar.return_value = None

    # Call _get_intraday_signal_for_execution to trigger strategy instantiation
    p._get_intraday_signal_for_execution("NIFTY 50", "15minute", False)

    assert hasattr(p, "_credit_spread_strategy"), "Prometheus must instantiate _credit_spread_strategy"
    strategy = p._credit_spread_strategy

    assert strategy.strike_otm_steps == 1
    assert strategy.hedge_otm_steps == 3
    assert strategy.target_decay_pct == 0.70
    assert strategy.breakeven_decay_pct == 0.50
    assert strategy.max_loss_multiplier == 1.5
    assert strategy.min_credit_pct == 0.15
    assert strategy.max_days_to_expiry == 1
    assert strategy.max_days_to_expiry_monthly == 1
    assert strategy.max_range_pct == 0.018
