"""
Unit tests for Audit Batch 10 fixes:
1. Precision tolerance in PaperTradeEngine for single-lot risk ceiling (NIFTY 13.1 vs 13.08 pts).
2. Intraday trade counter bookkeeping for paper-tracked Rank #1 live-eligible positions.
3. Post-market status transition to standby mode.
"""

from unittest.mock import MagicMock
from prometheus.papertrade.engine import PaperTradeEngine
from prometheus.papertrade.types import Direction


def test_papertrade_engine_precision_tolerance_for_nifty_risk_ceiling():
    """Verify that a valid NIFTY signal with 13.10 pt SL distance (within 850 budget rounding)
    is NOT rejected by the single-lot risk ceiling (13.08 pts threshold + 0.05 tolerance)."""
    mock_feed = MagicMock()
    mock_feed.get_quote.return_value = (161.15, 161.0, 161.3)
    mock_feed.get_ltp.return_value = 161.15

    engine = PaperTradeEngine(feed=mock_feed)
    engine.enforce_risk_ceiling = True

    sig_nifty = MagicMock()
    sig_nifty.symbol = "NIFTY 50"
    sig_nifty.instrument = "NIFTY17NOV2625250CE"
    sig_nifty.direction = Direction.LONG
    sig_nifty.entry_price_hint = 161.15
    sig_nifty.stop_loss = 148.05  # sl_dist = 161.15 - 148.05 = 13.10 pts
    sig_nifty.target = 187.35
    sig_nifty.max_bars = 20
    sig_nifty.strategy = "Golden_Setup"
    sig_nifty.signal_score = 9.0
    sig_nifty.tier = "B"
    sig_nifty.net_gex = None
    sig_nifty.gamma_regime = "SHORT_GAMMA"
    sig_nifty.bar_timestamp = None

    engine.fill_sim = MagicMock()
    engine.fill_sim.simulate_fill.return_value = MagicMock(fill_price=161.15, source="quote")

    pos = engine.process_new_signal(sig_nifty)
    assert pos is not None, "Valid NIFTY signal with 13.10 pt SL was spuriously rejected!"


def test_papertrade_engine_strictly_rejects_truly_excessive_sl():
    """Verify that an SL distance that genuinely exceeds 850 budget (e.g. 15.0 pts on NIFTY) is rejected."""
    mock_feed = MagicMock()
    mock_feed.get_quote.return_value = (161.15, 161.0, 161.3)
    mock_feed.get_ltp.return_value = 161.15

    engine = PaperTradeEngine(feed=mock_feed)
    engine.enforce_risk_ceiling = True

    sig_excessive = MagicMock()
    sig_excessive.symbol = "NIFTY 50"
    sig_excessive.instrument = "NIFTY17NOV2625250CE"
    sig_excessive.direction = Direction.LONG
    sig_excessive.entry_price_hint = 161.15
    sig_excessive.stop_loss = 145.0  # sl_dist = 16.15 pts = Rs 1,049.75 > Rs 850
    sig_excessive.target = 187.35
    sig_excessive.max_bars = 20
    sig_excessive.strategy = "Golden_Setup"
    sig_excessive.signal_score = 9.0
    sig_excessive.tier = "B"
    sig_excessive.net_gex = None
    sig_excessive.gamma_regime = "SHORT_GAMMA"
    sig_excessive.bar_timestamp = None

    engine.fill_sim = MagicMock()
    engine.fill_sim.simulate_fill.return_value = MagicMock(fill_price=161.15, source="quote")

    pos = engine.process_new_signal(sig_excessive)
    assert pos is None, "Excessive SL signal should have been rejected!"
