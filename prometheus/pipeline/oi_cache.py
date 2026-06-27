"""OI Cache: Background thread that fetches option chain data every 2 minutes.

Architecture:
  - Daemon thread fetches option chains for index symbols via Angel One
  - Runs OIAnalyzer on each chain, stores results in thread-safe dict
  - Scanner reads from cache during scan cycle → 0ms latency
  - Only fetches for index symbols (stocks lack liquid F&O chains)

API cost per refresh cycle:
  - searchScrip: 0 calls (cached daily by AngelOneOptionChain)
  - getMarketData: 1 batch call per symbol (20 contracts → 1 batch of 50)
  - Total: ~5 API calls every 2 minutes = negligible rate impact
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from prometheus.utils.logger import logger


# Index symbols that have liquid F&O chains worth analyzing
INDEX_SYMBOLS: Set[str] = {
    "NIFTY 50",
    "NIFTY BANK",
    "SENSEX",
    "NIFTY FIN SERVICE",
    "NIFTY MIDCAP SELECT",
    "NIFTY IT",
}


@dataclass
class OICacheEntry:
    """Cached OI analysis result for one symbol."""
    symbol: str
    oi_result: Dict = field(default_factory=dict)    # from OIAnalyzer.analyze()
    oi_signals: list = field(default_factory=list)    # list of OISignal
    oi_metrics: Dict = field(default_factory=dict)    # pcr, max_pain, etc.
    sentiment_score: float = 0.0                       # -1 (bearish) to +1 (bullish)
    sentiment_direction: str = "neutral"
    spot_price: float = 0.0
    chain_size: int = 0
    fetch_time: Optional[datetime] = None
    error: str = ""
    
    @property
    def age_seconds(self) -> float:
        """How old this cache entry is."""
        if not self.fetch_time:
            return float('inf')
        return (datetime.now() - self.fetch_time).total_seconds()
    
    @property
    def is_fresh(self) -> bool:
        """Cache entry is less than 5 minutes old."""
        return self.age_seconds < 300


class OICache:
    """Thread-safe OI data cache with background refresh.
    
    Usage:
        cache = OICache(prometheus.data, prometheus.oi_analyzer)
        cache.start()
        
        # In scan cycle:
        entry = cache.get("NIFTY 50")
        if entry and entry.is_fresh:
            oi_signals = entry.oi_signals
            sentiment = entry.sentiment_direction
    """
    
    def __init__(
        self,
        data_engine,
        oi_analyzer,
        symbols: Optional[List[str]] = None,
        refresh_interval_seconds: int = 120,
    ):
        self._data = data_engine
        self._oi_analyzer = oi_analyzer
        
        # Only cache index symbols (filter out stocks)
        if symbols:
            self._symbols = [s for s in symbols if s in INDEX_SYMBOLS]
        else:
            self._symbols = list(INDEX_SYMBOLS)
        
        self._refresh_interval = refresh_interval_seconds
        self._cache: Dict[str, OICacheEntry] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def get(self, symbol: str) -> Optional[OICacheEntry]:
        """Get cached OI data for a symbol (thread-safe read)."""
        with self._lock:
            return self._cache.get(symbol)
    
    def get_all(self) -> Dict[str, OICacheEntry]:
        """Get all cached entries (thread-safe copy)."""
        with self._lock:
            return dict(self._cache)
    
    @property
    def symbols_cached(self) -> int:
        """Number of symbols with cached data."""
        with self._lock:
            return len(self._cache)
    
    @property
    def is_running(self) -> bool:
        """Whether the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()
    
    # ------------------------------------------------------------------
    # Background Refresh
    # ------------------------------------------------------------------
    
    def start(self):
        """Start the background refresh thread."""
        if self.is_running:
            return
        
        if not self._symbols:
            logger.info("OICache: no index symbols to cache, skipping start")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name='oi-cache',
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"OICache: started background thread — "
            f"{len(self._symbols)} symbols, "
            f"refresh every {self._refresh_interval}s"
        )
    
    def stop(self):
        """Stop the background refresh thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            logger.info("OICache: background thread stopped")
    
    def _refresh_loop(self):
        """Background loop: fetch OI data for all symbols."""
        # Initial fetch immediately
        self._refresh_all()
        
        while not self._stop_event.is_set():
            self._stop_event.wait(self._refresh_interval)
            if self._stop_event.is_set():
                break
            
            # Only refresh during market hours (9:15 - 15:30)
            now = datetime.now()
            t = now.time()
            from datetime import time as dtime
            if t < dtime(9, 15) or t >= dtime(15, 35):
                continue
            
            self._refresh_all()
    
    def _refresh_all(self):
        """Fetch OI data for all tracked symbols."""
        t0 = time.monotonic()
        success_count = 0
        
        for symbol in self._symbols:
            if self._stop_event.is_set():
                break
            
            entry = self._refresh_symbol(symbol)
            if entry and not entry.error:
                success_count += 1
        
        elapsed = time.monotonic() - t0
        logger.info(
            f"OICache: refreshed {success_count}/{len(self._symbols)} symbols "
            f"in {elapsed:.1f}s"
        )
    
    def _refresh_symbol(self, symbol: str) -> OICacheEntry:
        """Fetch and analyze OI for one symbol."""
        entry = OICacheEntry(symbol=symbol, fetch_time=datetime.now())
        
        try:
            # Get spot price
            spot = 0.0
            if hasattr(self._data, 'get_spot_price'):
                spot = self._data.get_spot_price(symbol) or 0
            if spot <= 0:
                # Fallback: get from latest historical data
                try:
                    df = self._data.fetch_historical(symbol, days=2, interval='day')
                    if df is not None and not df.empty:
                        spot = float(df['close'].iloc[-1])
                except Exception:
                    pass
            
            if spot <= 0:
                entry.error = "No spot price available"
                logger.debug(f"OICache: {symbol} — no spot price, skipping")
                with self._lock:
                    self._cache[symbol] = entry
                return entry
            
            entry.spot_price = spot
            
            # Fetch option chain
            chain_df = None
            angelone_options = getattr(self._data, 'angelone_options', None)
            if angelone_options:
                try:
                    from prometheus.data.angelone_options import AngelOneOptionChain
                    if isinstance(angelone_options, AngelOneOptionChain):
                        chain_df = angelone_options.get_option_chain(
                            symbol, spot_price=spot, strikes_around_atm=10,
                        )
                except Exception as e:
                    logger.debug(f"OICache: Angel One chain failed for {symbol}: {e}")
            
            # Fallback to engine's fetch_options_chain
            if (chain_df is None or chain_df.empty) and hasattr(self._data, 'fetch_options_chain'):
                try:
                    chain_df = self._data.fetch_options_chain(symbol)
                except Exception as e:
                    logger.debug(f"OICache: engine chain failed for {symbol}: {e}")
            
            if chain_df is None or chain_df.empty:
                entry.error = "Empty option chain"
                logger.debug(f"OICache: {symbol} — empty chain")
                with self._lock:
                    self._cache[symbol] = entry
                return entry
            
            entry.chain_size = len(chain_df)
            
            # Run OI Analyzer
            oi_result = self._oi_analyzer.analyze(chain_df, spot)
            
            if isinstance(oi_result, dict):
                entry.oi_result = oi_result
                entry.oi_signals = oi_result.get("signals", [])
                entry.oi_metrics = oi_result.get("metrics", {})
                
                # Extract sentiment
                sentiment = entry.oi_metrics.get("oi_sentiment", {})
                if isinstance(sentiment, dict):
                    entry.sentiment_score = float(sentiment.get("score", 0))
                    entry.sentiment_direction = sentiment.get("direction", "neutral")
            
            logger.info(
                f"OICache: {symbol} — {entry.chain_size} contracts, "
                f"sentiment={entry.sentiment_direction} "
                f"(score={entry.sentiment_score:+.3f}), "
                f"PCR={entry.oi_metrics.get('pcr', {}).get('oi', 0):.2f}"
            )
            
        except Exception as e:
            entry.error = str(e)
            logger.warning(f"OICache: {symbol} refresh error: {e}")
        
        with self._lock:
            self._cache[symbol] = entry
        
        return entry
    
    # ------------------------------------------------------------------
    # Signal Integration Helpers
    # ------------------------------------------------------------------
    
    def get_oi_confluence(self, symbol: str, direction: str) -> Dict:
        """Get OI confluence data for a signal direction.
        
        Returns:
            {
                "agrees": bool,        # OI sentiment agrees with signal direction
                "score_boost": float,   # How much to boost/penalize confidence (-0.2 to +0.2)
                "pcr": float,          # Current PCR value
                "support": float,      # OI-based support level
                "resistance": float,   # OI-based resistance level
                "signals": list,       # Relevant OI signals
                "summary": str,        # Human-readable summary
                "stale": bool,         # Whether the data is older than 5 min
            }
        """
        entry = self.get(symbol)
        
        if not entry or entry.error or not entry.is_fresh:
            return {
                "agrees": True,  # Default: don't block on no data
                "score_boost": 0.0,
                "pcr": 0.0,
                "support": 0.0,
                "resistance": 0.0,
                "signals": [],
                "summary": "No OI data available" if not entry else f"OI data stale ({entry.age_seconds:.0f}s old)",
                "stale": True,
            }
        
        oi_direction = entry.sentiment_direction
        oi_score = entry.sentiment_score
        
        # Does OI agree with signal?
        agrees = (
            (direction == "bullish" and oi_direction in ("bullish", "neutral")) or
            (direction == "bearish" and oi_direction in ("bearish", "neutral"))
        )
        
        # Score boost: OI confirms → boost. OI contradicts → penalize.
        if oi_direction == "neutral" or abs(oi_score) < 0.1:
            score_boost = 0.0
        elif agrees:
            score_boost = min(abs(oi_score) * 0.3, 0.15)  # Max +15% confidence
        else:
            score_boost = -min(abs(oi_score) * 0.3, 0.15)  # Max -15% penalty
        
        # Extract levels
        pcr = entry.oi_metrics.get("pcr", {}).get("oi", 0)
        support = entry.oi_metrics.get("strongest_support", 0)
        resistance = entry.oi_metrics.get("strongest_resistance", 0)
        
        # Build summary
        signal_details = [s.details for s in entry.oi_signals if hasattr(s, 'details')]
        summary_parts = []
        if pcr:
            summary_parts.append(f"PCR={pcr:.2f}")
        summary_parts.append(f"OI sentiment={oi_direction} ({oi_score:+.3f})")
        if agrees:
            summary_parts.append("✓ confirms signal")
        else:
            summary_parts.append("✗ contradicts signal")
        
        return {
            "agrees": agrees,
            "score_boost": round(score_boost, 3),
            "pcr": pcr,
            "support": support,
            "resistance": resistance,
            "signals": signal_details[:3],  # Top 3 signals
            "summary": " | ".join(summary_parts),
            "stale": False,
        }
