"""Tests for OI Cache module."""

import unittest
from unittest.mock import MagicMock, patch
from prometheus.pipeline.oi_cache import OICache, OICacheEntry, INDEX_SYMBOLS


class TestOICacheEntry(unittest.TestCase):
    """Test OICacheEntry dataclass."""
    
    def test_fresh_entry(self):
        from datetime import datetime
        entry = OICacheEntry(symbol="NIFTY 50", fetch_time=datetime.now())
        self.assertTrue(entry.is_fresh)
        self.assertLess(entry.age_seconds, 1)
    
    def test_stale_entry(self):
        entry = OICacheEntry(symbol="NIFTY 50", fetch_time=None)
        self.assertFalse(entry.is_fresh)
        self.assertEqual(entry.age_seconds, float('inf'))
    
    def test_defaults(self):
        entry = OICacheEntry(symbol="NIFTY 50")
        self.assertEqual(entry.sentiment_score, 0.0)
        self.assertEqual(entry.sentiment_direction, "neutral")
        self.assertEqual(entry.oi_signals, [])
        self.assertEqual(entry.oi_metrics, {})


class TestOICacheInit(unittest.TestCase):
    """Test OI cache initialization."""
    
    def test_filters_to_index_symbols_only(self):
        mock_data = MagicMock()
        mock_analyzer = MagicMock()
        symbols = ["NIFTY 50", "NIFTY BANK", "HDFCBANK", "RELIANCE", "SBIN"]
        
        cache = OICache(mock_data, mock_analyzer, symbols=symbols)
        
        # Should only keep index symbols
        self.assertIn("NIFTY 50", cache._symbols)
        self.assertIn("NIFTY BANK", cache._symbols)
        self.assertNotIn("HDFCBANK", cache._symbols)
        self.assertNotIn("RELIANCE", cache._symbols)
        self.assertNotIn("SBIN", cache._symbols)
    
    def test_no_symbols_uses_defaults(self):
        mock_data = MagicMock()
        mock_analyzer = MagicMock()
        
        cache = OICache(mock_data, mock_analyzer, symbols=None)
        self.assertEqual(set(cache._symbols), INDEX_SYMBOLS)
    
    def test_empty_on_init(self):
        cache = OICache(MagicMock(), MagicMock(), symbols=["NIFTY 50"])
        self.assertEqual(cache.symbols_cached, 0)
        self.assertIsNone(cache.get("NIFTY 50"))


class TestOICacheConfluence(unittest.TestCase):
    """Test get_oi_confluence helper."""
    
    def test_no_data_returns_default(self):
        cache = OICache(MagicMock(), MagicMock(), symbols=["NIFTY 50"])
        result = cache.get_oi_confluence("NIFTY 50", "bullish")
        
        self.assertTrue(result["agrees"])
        self.assertEqual(result["score_boost"], 0.0)
        self.assertTrue(result["stale"])
    
    def test_agrees_with_bullish_signal(self):
        from datetime import datetime
        cache = OICache(MagicMock(), MagicMock(), symbols=["NIFTY 50"])
        
        entry = OICacheEntry(
            symbol="NIFTY 50",
            sentiment_direction="bullish",
            sentiment_score=0.5,
            oi_metrics={"pcr": {"oi": 1.35}, "oi_sentiment": {"score": 0.5, "direction": "bullish"}},
            fetch_time=datetime.now(),
        )
        cache._cache["NIFTY 50"] = entry
        
        result = cache.get_oi_confluence("NIFTY 50", "bullish")
        self.assertTrue(result["agrees"])
        self.assertGreater(result["score_boost"], 0)
        self.assertFalse(result["stale"])
    
    def test_contradicts_signal(self):
        from datetime import datetime
        cache = OICache(MagicMock(), MagicMock(), symbols=["NIFTY 50"])
        
        entry = OICacheEntry(
            symbol="NIFTY 50",
            sentiment_direction="bearish",
            sentiment_score=-0.4,
            oi_metrics={"pcr": {"oi": 0.6}},
            fetch_time=datetime.now(),
        )
        cache._cache["NIFTY 50"] = entry
        
        result = cache.get_oi_confluence("NIFTY 50", "bullish")
        self.assertFalse(result["agrees"])
        self.assertLess(result["score_boost"], 0)  # Penalty
    
    def test_neutral_gives_no_boost(self):
        from datetime import datetime
        cache = OICache(MagicMock(), MagicMock(), symbols=["NIFTY 50"])
        
        entry = OICacheEntry(
            symbol="NIFTY 50",
            sentiment_direction="neutral",
            sentiment_score=0.05,
            oi_metrics={},
            fetch_time=datetime.now(),
        )
        cache._cache["NIFTY 50"] = entry
        
        result = cache.get_oi_confluence("NIFTY 50", "bullish")
        self.assertTrue(result["agrees"])
        self.assertEqual(result["score_boost"], 0.0)


class TestOICacheThread(unittest.TestCase):
    """Test thread lifecycle."""
    
    def test_start_stop(self):
        cache = OICache(MagicMock(), MagicMock(), symbols=["NIFTY 50"])
        # Don't actually start (would try API calls), just verify state
        self.assertFalse(cache.is_running)
    
    def test_start_with_no_symbols(self):
        cache = OICache(MagicMock(), MagicMock(), symbols=["HDFCBANK"])
        cache.start()  # Should not start thread (no index symbols)
        self.assertFalse(cache.is_running)


if __name__ == "__main__":
    unittest.main()
