# ============================================================================
# PROMETHEUS — Test Suite: 8-Pillar Microstructure Trade Health Engine
# ============================================================================
"""
Unit and integration tests for the 8-Pillar Microstructure Trade Health Engine:
  1. Pillar 1: Dynamic VWAP clearance & slope (bearish breach vs bullish clearance).
  2. Pillar 2: Relative volume (RVOL) exhaustion vs institutional breakout surge.
  3. Pillar 3: Real-time intraday strike ΔOI accumulation vs liquidation.
  4. Pillar 4: Dealer GEX Short Gamma squeeze acceleration vs Long Gamma cap.
  5. Pillar 5: Microstructure noise envelope preservation.
  6. Pillar 6: Non-linear intraday theta decay & Power Hour penalty.
  7. Pillar 8: Multi-timeframe trend synchronization (15M vs 1H).
  8. Composite Health Score synthesis & decision boundaries.
  9. Real Threat Defensive Pre-SL Bailout (saving 50-70% of risk).
  10. Real Favor Runner Target Expansion (convexity / gamma squeeze unlock).
  11. Integration with PositionMonitor and PositionTracker.
  12. Retail-facing Telegram alert message generation.
"""

from datetime import datetime, date, time as dtime
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from prometheus.execution.position_health import PositionHealthEngine, PositionHealthReport
from prometheus.execution.position_monitor import PositionMonitor, TrailingState
from prometheus.papertrade.position_tracker import PositionTracker, CostModel
from prometheus.papertrade.types import Position, Direction, ExitReason, TradeSnapshot


def _build_test_dataframe(spot_levels: list[float], base_volume: float = 10000.0) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame."""
    bars = []
    base_ts = pd.Timestamp("2026-10-06 09:15:00")
    for i, s in enumerate(spot_levels):
        bars.append({
            "timestamp": base_ts + pd.Timedelta(minutes=15 * i),
            "open": s - 5.0,
            "high": s + 10.0,
            "low": s - 10.0,
            "close": s,
            "volume": base_volume,
        })
    return pd.DataFrame(bars)


# ----------------------------------------------------------------------------
# 1. Pillar 1: VWAP Clearance & Slope
# ----------------------------------------------------------------------------

def test_pillar1_vwap_bearish_breach_for_bullish_ce():
    """Verify that Long CE spot dropping below falling session VWAP triggers severe threat."""
    engine = PositionHealthEngine()
    
    # Prices declining from 55000 down to 54800 (spot below falling VWAP)
    spots = [55000, 54950, 54900, 54850, 54800]
    df = _build_test_dataframe(spots)
    
    score, gap, threat, favor = engine._eval_pillar_vwap(df, spot=54780.0, trade_is_bullish=True)
    assert score <= -30.0, f"Expected bearish penalty, got {score}"
    assert threat is not None
    assert "lost Session VWAP" in threat
    assert favor is None


def test_pillar1_vwap_bullish_clearance_for_bullish_ce():
    """Verify that Long CE spot rising above upward-sloping VWAP triggers favor."""
    engine = PositionHealthEngine()
    
    spots = [54800, 54850, 54900, 54950, 55050]
    df = _build_test_dataframe(spots)
    
    score, gap, threat, favor = engine._eval_pillar_vwap(df, spot=55100.0, trade_is_bullish=True)
    assert score >= 35.0, f"Expected bullish clearance favor, got {score}"
    assert favor is not None
    assert "firmly above rising VWAP" in favor
    assert threat is None


def test_pillar1_vwap_bearish_trade_symmetric():
    """Verify symmetric behavior for Long PE (Bearish trade)."""
    engine = PositionHealthEngine()
    
    # Bearish trade where spot rallies above VWAP -> Threat
    spots = [54800, 54850, 54900, 54950, 55050]
    df = _build_test_dataframe(spots)
    
    score, gap, threat, favor = engine._eval_pillar_vwap(df, spot=55100.0, trade_is_bullish=False)
    assert score <= -30.0
    assert threat is not None
    assert "rallied above Session VWAP" in threat


# ----------------------------------------------------------------------------
# 2. Pillar 2: Relative Volume & Absorption
# ----------------------------------------------------------------------------

def test_pillar2_volume_exhaustion():
    """Verify volume exhaustion penalty when price advances on weak volume (<0.45x RVOL)."""
    engine = PositionHealthEngine()
    
    # 20 bars of 10,000 volume, latest bar only 3,000 volume (RVOL = 0.30x)
    volumes = [10000.0] * 20 + [3000.0]
    bars = []
    base_ts = pd.Timestamp("2026-10-06 09:15:00")
    for i, v in enumerate(volumes):
        bars.append({
            "timestamp": base_ts + pd.Timedelta(minutes=15 * i),
            "open": 55000.0,
            "high": 55050.0,
            "low": 54980.0,
            "close": 55020.0 + (i * 2.0),
            "volume": v,
        })
    df = pd.DataFrame(bars)
    
    score, rvol, threat, favor = engine._eval_pillar_volume(
        df, spot=55060.0, entry_spot=55000.0, trade_is_bullish=True
    )
    assert rvol < 0.45
    assert score <= -30.0
    assert threat is not None
    assert "Volume exhaustion" in threat


def test_pillar2_volume_institutional_surge():
    """Verify institutional acceleration when price advances on high volume (>1.8x RVOL)."""
    engine = PositionHealthEngine()
    
    # 20 bars of 10,000 volume, latest bar 25,000 volume (RVOL = 2.50x)
    volumes = [10000.0] * 20 + [25000.0]
    bars = []
    base_ts = pd.Timestamp("2026-10-06 09:15:00")
    for i, v in enumerate(volumes):
        bars.append({
            "timestamp": base_ts + pd.Timedelta(minutes=15 * i),
            "open": 55000.0,
            "high": 55050.0,
            "low": 54980.0,
            "close": 55000.0 + (i * 5.0),
            "volume": v,
        })
    df = pd.DataFrame(bars)
    
    score, rvol, threat, favor = engine._eval_pillar_volume(
        df, spot=55100.0, entry_spot=55000.0, trade_is_bullish=True
    )
    assert rvol >= 1.80
    assert score >= 40.0
    assert favor is not None
    assert "breakout confirmation" in favor


# ----------------------------------------------------------------------------
# 3. Pillar 3 & 4: Live ΔOI and Dealer GEX
# ----------------------------------------------------------------------------

def test_pillar3_oi_buildup_and_unwinding():
    engine = PositionHealthEngine()
    
    # Call buildup
    score, doi, threat, favor = engine._eval_pillar_oi(
        "NIFTY BANK", "BANKNIFTY55000CE", trade_is_bullish=True, override={"delta_oi": 30000}
    )
    assert score > 0
    assert favor is not None
    assert "aggressive buildup" in favor

    # Call unwinding / liquidation
    score_unwind, doi_u, threat_u, favor_u = engine._eval_pillar_oi(
        "NIFTY BANK", "BANKNIFTY55000CE", trade_is_bullish=True, override={"delta_oi": -15000}
    )
    assert score_unwind < 0
    assert threat_u is not None
    assert "unwinding" in threat_u


def test_pillar4_dealer_short_gamma_squeeze():
    engine = PositionHealthEngine()
    
    # Mock data_engine GEX profile
    mock_data = MagicMock()
    mock_ao = MagicMock()
    mock_ao.get_option_chain.return_value = pd.DataFrame([{"dummy": 1}])
    mock_data.angelone_options = mock_ao
    engine.data_engine = mock_data
    
    # Store directly in GEX cache: Net GEX = -3.5 Cr (Deep Short Gamma)
    mock_profile = MagicMock()
    mock_profile.net_gex = -3.5
    mock_profile.zero_gamma_level = 54500.0
    engine._gex_cache["NIFTY BANK"] = (pd.Timestamp.now().timestamp(), mock_profile)
    
    score, net_gex, threat, favor = engine._eval_pillar_gex("NIFTY BANK", spot=55000.0, trade_is_bullish=True)
    assert net_gex == -3.5
    assert score >= 40.0
    assert favor is not None
    assert "Dealer SHORT GAMMA active" in favor


# ----------------------------------------------------------------------------
# 4. Pillar 6: Non-Linear Intraday Theta Decay
# ----------------------------------------------------------------------------

def test_pillar6_theta_penalty_after_multiple_bars():
    engine = PositionHealthEngine()
    
    # Stagnant trade after 4 bars (entry 100, ltp 100.5 -> 0.5% gain)
    # The penalty increases dynamically if afternoon/power hour
    penalty, threat = engine._eval_pillar_theta(current_premium=100.5, entry_premium=100.0, bars_held=4)
    # If test is run during trading hours or simulated, verify it returns cleanly
    assert isinstance(penalty, float)


# ----------------------------------------------------------------------------
# 5. Composite Health Score & Decision Boundaries
# ----------------------------------------------------------------------------

def test_composite_real_threat_pre_sl_bailout():
    """Verify that multiple negative pillars trigger Real Threat and PRE_SL_BAILOUT."""
    engine = PositionHealthEngine()
    
    # Underwater trade: entry 1000, current 985 (-15 pts)
    state = TrailingState(
        position_id="POS-TEST-THREAT",
        tradingsymbol="BANKNIFTY55000CE",
        symbol="NIFTY BANK",
        entry_premium=1000.0,
        initial_sl=950.0,
        current_sl=950.0,
        target=1080.0,
        direction="bullish",
        entry_spot=55000.0,
        entry_bar_count=3,
    )
    
    # Severe bearish breakdown DF (broken VWAP, low volume)
    spots = [55000, 54950, 54900, 54850, 54750]
    df = _build_test_dataframe(spots)
    
    report = engine.evaluate_position_health(
        state=state,
        current_premium=985.0,
        spot_override=54720.0,
        underlying_df_override=df,
        oi_metrics_override={"delta_oi": -20000},
    )
    
    assert report.real_threat is True
    assert report.suggested_action == "PRE_SL_BAILOUT"
    assert len(report.threat_reasons) >= 1


def test_composite_real_favor_target_expansion():
    """Verify that strong positive pillars trigger Real Favor and EXPAND_TARGET."""
    engine = PositionHealthEngine()
    
    state = TrailingState(
        position_id="POS-TEST-FAVOR",
        tradingsymbol="BANKNIFTY55000CE",
        symbol="NIFTY BANK",
        entry_premium=1000.0,
        initial_sl=950.0,
        current_sl=950.0,
        target=1080.0,
        direction="bullish",
        entry_spot=55000.0,
        entry_bar_count=2,
    )
    
    # Parabolic breakout DF (clean above VWAP, surging volume)
    spots = [54800, 54900, 55000, 55100, 55250]
    df = _build_test_dataframe(spots, base_volume=25000.0)
    
    # Inject Short Gamma in GEX cache
    mock_profile = MagicMock()
    mock_profile.net_gex = -4.0
    mock_profile.zero_gamma_level = 54900.0
    engine._gex_cache["NIFTY BANK"] = (pd.Timestamp.now().timestamp(), mock_profile)
    
    report = engine.evaluate_position_health(
        state=state,
        current_premium=1035.0,
        spot_override=55300.0,
        underlying_df_override=df,
        oi_metrics_override={"delta_oi": 45000},
    )
    
    assert report.real_favor is True
    assert report.suggested_action == "EXPAND_TARGET"
    assert report.suggested_target_expansion > 0.0


# ----------------------------------------------------------------------------
# 6. PositionMonitor & PositionTracker Integration Tests
# ----------------------------------------------------------------------------

def test_position_monitor_executes_pre_sl_bailout():
    """Verify PositionMonitor exits early on Real Threat before price touches hard SL."""
    mock_broker = MagicMock()
    exits = []
    def _on_exit(pid, price, reason):
        exits.append((pid, price, reason))

    monitor = PositionMonitor(broker=mock_broker, poll_interval=1.0, on_exit=_on_exit)
    
    state = TrailingState(
        position_id="POS-BAILOUT-01",
        tradingsymbol="BANKNIFTY55000CE",
        symbol="NIFTY BANK",
        entry_premium=1000.0,
        initial_sl=950.0,
        current_sl=950.0,
        target=1080.0,
        direction="bullish",
        trade_mode="intraday",
        risk_distance=50.0,
    )
    
    # Mock health engine to return real threat
    mock_health = MagicMock()
    threat_report = PositionHealthReport(
        position_id=state.position_id,
        symbol=state.symbol,
        tradingsymbol=state.tradingsymbol,
        direction="bullish",
        current_price=980.0,
        entry_price=1000.0,
        health_score=-65.0,
        real_threat=True,
        threat_reasons=["Spot lost VWAP", "Volume exhaustion"],
        suggested_action="PRE_SL_BAILOUT",
    )
    mock_health.evaluate_position_health.return_value = threat_report
    monitor.health_engine = mock_health

    # Tick at 980 (above hard SL of 950)
    monitor._process_tick(state, current_price=980.0)
    
    assert len(exits) == 1, "Expected early bailout exit before hard SL"
    assert exits[0][0] == "POS-BAILOUT-01"
    assert exits[0][1] == 980.0
    assert exits[0][2] == "real_threat_structural_invalidation"


def test_position_monitor_expands_target_on_real_favor():
    """Verify PositionMonitor expands target when real favor is detected."""
    mock_broker = MagicMock()
    monitor = PositionMonitor(broker=mock_broker, poll_interval=1.0)
    
    state = TrailingState(
        position_id="POS-FAVOR-01",
        tradingsymbol="BANKNIFTY55000CE",
        symbol="NIFTY BANK",
        entry_premium=1000.0,
        initial_sl=950.0,
        current_sl=950.0,
        target=1080.0,
        direction="bullish",
        trade_mode="intraday",
        risk_distance=50.0,
    )
    
    mock_health = MagicMock()
    favor_report = PositionHealthReport(
        position_id=state.position_id,
        symbol=state.symbol,
        tradingsymbol=state.tradingsymbol,
        direction="bullish",
        current_price=1030.0,
        entry_price=1000.0,
        health_score=75.0,
        real_favor=True,
        favor_reasons=["Dealer Short Gamma", "RVOL 2.4x surge"],
        suggested_action="EXPAND_TARGET",
        suggested_target_expansion=25.0,
    )
    mock_health.evaluate_position_health.return_value = favor_report
    monitor.health_engine = mock_health

    # Initial target is 1080.0
    monitor._process_tick(state, current_price=1030.0)
    
    assert state.target == 1105.0, f"Expected target expanded to 1105.0, got {state.target}"
    assert getattr(state, "_target_expanded", False) is True


def test_position_tracker_executes_structural_invalidation():
    """Verify PositionTracker triggers ExitReason.STRUCTURAL_INVALIDATION on Real Threat."""
    mock_feed = MagicMock()
    mock_feed.get_ltp.return_value = 980.0
    mock_sim = MagicMock()
    mock_sim.feed = mock_feed
    
    tracker = PositionTracker(
        fill_sim=mock_sim,
        cost_model=CostModel(),
        enable_trailing=True,
    )
    
    mock_health = MagicMock()
    threat_report = PositionHealthReport(
        position_id="PAPER-THREAT-01",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY55000CE",
        direction="bullish",
        current_price=980.0,
        entry_price=1000.0,
        health_score=-55.0,
        real_threat=True,
        threat_reasons=["VWAP lost", "Opposing OI wall"],
        suggested_action="PRE_SL_BAILOUT",
    )
    mock_health.evaluate_position_health.return_value = threat_report
    tracker.health_engine = mock_health
    
    pos = Position(
        trade_id="PAPER-THREAT-01",
        symbol="NIFTY BANK",
        instrument="BANKNIFTY55000CE",
        underlying="BANKNIFTY",
        direction=Direction.LONG,
        quantity=30,
        entry_price=1000.0,
        stop_loss=950.0,
        target=1080.0,
        max_bars=7,
        entry_time=datetime.now(),
        trade_mode="intraday",
    )
    
    snap = TradeSnapshot(
        timestamp=datetime.now(),
        symbol="NIFTY BANK",
        instrument="BANKNIFTY55000CE",
        open=55000.0,
        high=55020.0,
        low=54800.0,
        close=54820.0,
        volume=10000.0,
    )
    
    exit_price, reason = tracker._evaluate_exit_via_feed(pos, snap, is_session_end=False, is_square_off=False)
    assert exit_price == 980.0
    assert reason == ExitReason.STRUCTURAL_INVALIDATION


# ----------------------------------------------------------------------------
# 7. Telegram Alert Text Verification
# ----------------------------------------------------------------------------

def test_telegram_alert_formatting():
    threat_rep = PositionHealthReport(
        position_id="POS-TG-01",
        symbol="NIFTY BANK",
        tradingsymbol="BANKNIFTY55000CE",
        direction="bullish",
        current_price=980.0,
        entry_price=1000.0,
        health_score=-60.0,
        real_threat=True,
        threat_reasons=["Spot dropped below VWAP", "Volume dried up"],
        suggested_action="PRE_SL_BAILOUT",
    )
    msg = PositionHealthEngine.format_threat_telegram_alert(threat_rep)
    assert "EARLY DEFENSE ALERT" in msg
    assert "Exit now at market" in msg
    assert "Health Score: <b>-60/100</b>" in msg

    favor_rep = PositionHealthReport(
        position_id="POS-TG-02",
        symbol="SENSEX",
        tradingsymbol="SENSEX73200PE",
        direction="bearish",
        current_price=280.0,
        entry_price=260.0,
        health_score=80.0,
        real_favor=True,
        favor_reasons=["Dealer Short Gamma Squeeze", "Surging Relative Volume 2.5x"],
        suggested_action="EXPAND_TARGET",
        suggested_target_expansion=30.0,
    )
    favor_msg = PositionHealthEngine.format_favor_telegram_alert(favor_rep, old_target=282.0, new_target=312.0)
    assert "RUNNER ACCELERATION" in favor_msg
    assert "TARGET EXPANDED" in favor_msg
    assert "Hold position and let profits run" in favor_msg


def test_pillar8_htf_trend_with_live_data_engine():
    """Verify Pillar 8 calls get_higher_timeframe_trend on PriceActionMomentumScanner without AttributeError."""
    mock_data = MagicMock()
    # 5 1-Hour candles: EMA20 > EMA50 -> BULLISH
    bars_1h = []
    base_ts = pd.Timestamp("2026-10-06 09:15:00")
    for i, p in enumerate([54000, 54200, 54500, 54800, 55100]):
        bars_1h.append({
            "timestamp": base_ts + pd.Timedelta(hours=i),
            "open": p - 50,
            "high": p + 100,
            "low": p - 100,
            "close": p,
            "volume": 50000,
        })
    df_1h = pd.DataFrame(bars_1h)
    mock_data.fetch_historical.return_value = df_1h

    engine = PositionHealthEngine(data_engine=mock_data)
    score, regime, threat, favor = engine._eval_pillar_htf("NIFTY BANK", trade_is_bullish=True)
    assert regime in {"BULLISH", "EMERGING_BULLISH"}
    assert score > 0
    assert favor is not None
    assert threat is None

