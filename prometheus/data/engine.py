# ============================================================================
# PROMETHEUS — Data Engine: Market Data Fetcher
# ============================================================================
"""
Multi-source data fetcher for Indian market data.
Primary: Zerodha Kite Connect
Fallback: yfinance (for historical data when Kite unavailable)
NSE direct: For options chain and OI data
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
import time
import requests
import json

from prometheus.utils.logger import logger
from prometheus.utils.indian_market import IST, is_trading_day


class KiteDataFeed:
    """Zerodha Kite Connect data feed handler."""

    # Kite instrument token mapping for indices
    INDEX_TOKENS = {
        "NIFTY 50": 256265,
        "NIFTY BANK": 260105,
        "NIFTY FIN SERVICE": 257801,
        "INDIA VIX": 264969,
    }

    def __init__(self, api_key: str = "", access_token: str = ""):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = None
        self._instruments_cache = None

        if api_key and access_token:
            self._init_kite()

    def _init_kite(self):
        """Initialize Kite Connect client."""
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.info("Kite Connect initialized successfully")
        except ImportError:
            logger.warning("kiteconnect not installed. Run: pip install kiteconnect")
        except Exception as e:
            logger.error(f"Kite Connect init failed: {e}")

    def is_connected(self) -> bool:
        """Check if Kite is connected and authenticated."""
        if self.kite is None:
            return False
        try:
            self.kite.profile()
            return True
        except Exception:
            return False

    def get_historical_data(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str = "day"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Kite.

        interval: minute, 3minute, 5minute, 10minute, 15minute, 30minute,
                  60minute, day, week, month
        """
        if not self.kite:
            logger.warning("Kite not connected. Using fallback data source.")
            return pd.DataFrame()

        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                oi=True
            )
            df = pd.DataFrame(data)
            if not df.empty:
                df.rename(columns={"date": "timestamp"}, inplace=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.error(f"Historical data fetch failed: {e}")
            return pd.DataFrame()

    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Get last traded prices for multiple instruments."""
        if not self.kite:
            return {}
        try:
            instruments = [f"NSE:{s}" for s in symbols]
            data = self.kite.ltp(instruments)
            return {
                s.replace("NSE:", ""): v["last_price"]
                for s, v in data.items()
            }
        except Exception as e:
            logger.error(f"LTP fetch failed: {e}")
            return {}

    def get_quote(self, symbols: List[str]) -> dict:
        """Get full quote with OHLC, volume, OI for instruments."""
        if not self.kite:
            return {}
        try:
            instruments = [f"NSE:{s}" for s in symbols]
            return self.kite.quote(instruments)
        except Exception as e:
            logger.error(f"Quote fetch failed: {e}")
            return {}

    def get_instruments(self, exchange: str = "NFO") -> pd.DataFrame:
        """Get instrument list for options/futures discovery."""
        if not self.kite:
            return pd.DataFrame()
        try:
            if self._instruments_cache is None:
                instruments = self.kite.instruments(exchange)
                self._instruments_cache = pd.DataFrame(instruments)
            return self._instruments_cache
        except Exception as e:
            logger.error(f"Instruments fetch failed: {e}")
            return pd.DataFrame()


class YFinanceFallback:
    """Fallback data source using yfinance (free, no API key needed)."""

    # yfinance ticker mapping for Indian instruments
    TICKER_MAP = {
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK",
        "SENSEX": "^BSESN",
        "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS",
        "NIFTY IT": "^CNXIT",
        "NIFTY MIDCAP SELECT": "NIFTY_MIDCAP_SELECT.NS",
        "NIFTY NEXT 50": "^NSMIDCP",
        "INDIA VIX": "^INDIAVIX",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "TATAMOTORS": "TATAMOTORS.NS",
        "ITC": "ITC.NS",
        "AXISBANK": "AXISBANK.NS",
        "BAJFINANCE": "BAJFINANCE.NS",
        # US indices (backtest only)
        "SPX": "^GSPC",
        "NASDAQ": "^IXIC",
        "DOW": "^DJI",
        # Forex (backtest only)
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
    }

    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data via yfinance.

        interval mapping:
            1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
        """
        try:
            import yfinance as yf
            ticker = self.TICKER_MAP.get(symbol, f"{symbol}.NS")
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

            if data.empty:
                return pd.DataFrame()

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data.reset_index()
            col_map = {"Date": "timestamp", "Datetime": "timestamp",
                       "Open": "open", "High": "high", "Low": "low",
                       "Close": "close", "Volume": "volume"}
            df.rename(columns=col_map, inplace=True)

            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df[required_cols]

        except ImportError:
            logger.warning("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"yfinance fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/last price via yfinance."""
        try:
            import yfinance as yf
            ticker = self.TICKER_MAP.get(symbol, f"{symbol}.NS")
            data = yf.Ticker(ticker)
            return data.info.get("regularMarketPrice") or data.info.get("previousClose")
        except Exception as e:
            logger.error(f"Current price fetch failed for {symbol}: {e}")
            return None


class NSEDirectFeed:
    """
    Direct NSE website data fetch for options chain and OI data.
    Uses NSE's public JSON endpoints.
    """

    BASE_URL = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._cookies_set = False

    def _set_cookies(self):
        """Visit main page first to get cookies (NSE requires this)."""
        if self._cookies_set:
            return
        try:
            self.session.get(self.BASE_URL, timeout=10)
            self._cookies_set = True
        except Exception:
            pass

    def get_options_chain(self, symbol: str = "NIFTY") -> Optional[dict]:
        """
        Fetch live options chain from NSE.
        Returns raw JSON with all strikes, OI, IV, volume, etc.
        """
        self._set_cookies()
        symbol_map = {
            "NIFTY 50": "NIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "NIFTY FIN SERVICE": "FINNIFTY",
        }
        nse_symbol = symbol_map.get(symbol, symbol)

        url = f"{self.BASE_URL}/api/option-chain-indices?symbol={nse_symbol}"
        if nse_symbol not in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
            url = f"{self.BASE_URL}/api/option-chain-equities?symbol={nse_symbol}"

        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"NSE options chain: HTTP {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"NSE options chain fetch failed: {e}")
            return None

    def parse_options_chain(self, raw_data: dict) -> pd.DataFrame:
        """Parse raw NSE options chain JSON into a clean DataFrame."""
        if not raw_data or "records" not in raw_data:
            return pd.DataFrame()

        records = raw_data["records"]["data"]
        parsed = []

        for record in records:
            strike = record.get("strikePrice", 0)
            expiry = record.get("expiryDate", "")

            # Call data
            ce = record.get("CE", {})
            if ce:
                parsed.append({
                    "strike": strike,
                    "expiry": expiry,
                    "option_type": "CE",
                    "ltp": ce.get("lastPrice", 0),
                    "bid": ce.get("bidprice", 0),
                    "ask": ce.get("askPrice", 0),
                    "volume": ce.get("totalTradedVolume", 0),
                    "oi": ce.get("openInterest", 0),
                    "oi_change": ce.get("changeinOpenInterest", 0),
                    "iv": ce.get("impliedVolatility", 0),
                    "underlying": ce.get("underlyingValue", 0),
                })

            # Put data
            pe = record.get("PE", {})
            if pe:
                parsed.append({
                    "strike": strike,
                    "expiry": expiry,
                    "option_type": "PE",
                    "ltp": pe.get("lastPrice", 0),
                    "bid": pe.get("bidprice", 0),
                    "ask": pe.get("askPrice", 0),
                    "volume": pe.get("totalTradedVolume", 0),
                    "oi": pe.get("openInterest", 0),
                    "oi_change": pe.get("changeinOpenInterest", 0),
                    "iv": pe.get("impliedVolatility", 0),
                    "underlying": pe.get("underlyingValue", 0),
                })

        df = pd.DataFrame(parsed)
        if not df.empty:
            df["timestamp"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        return df

    def get_india_vix(self) -> Optional[float]:
        """Get current India VIX."""
        self._set_cookies()
        try:
            url = f"{self.BASE_URL}/api/allIndices"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for idx in data.get("data", []):
                    if "VIX" in idx.get("index", "").upper():
                        return idx.get("last", None)
            return None
        except Exception as e:
            logger.error(f"India VIX fetch failed: {e}")
            return None

    def get_fii_dii_data(self) -> Optional[dict]:
        """Get FII/DII daily activity data."""
        self._set_cookies()
        try:
            url = f"{self.BASE_URL}/api/fiidiiActivity/WEB"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"FII/DII data fetch failed: {e}")
            return None


class DataEngine:
    """
    Unified data engine that combines all data sources.
    Auto-selects best available source.
    """

    def __init__(self, kite_api_key: str = "", kite_access_token: str = ""):
        self.kite = KiteDataFeed(kite_api_key, kite_access_token)
        self.yf = YFinanceFallback()
        self.nse = NSEDirectFeed()

        from prometheus.data.store import DataStore
        self.store = DataStore()

        logger.info("Data Engine initialized")

    def fetch_historical(
        self,
        symbol: str,
        days: int = 365,
        interval: str = "day",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical data — tries Kite first, then yfinance fallback.
        Caches in SQLite.
        """
        end_date = datetime.now(IST).strftime("%Y-%m-%d")
        start_date = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")

        # Check cache first
        if not force_refresh:
            cached = self.store.get_ohlcv(symbol, interval, start=start_date, end=end_date)
            if not cached.empty and len(cached) > (days * 0.5):
                logger.debug(f"Using cached data for {symbol} ({len(cached)} rows)")
                return cached

        # Try Kite first
        if self.kite.is_connected():
            token = KiteDataFeed.INDEX_TOKENS.get(symbol)
            if token:
                df = self.kite.get_historical_data(token, start_date, end_date, interval)
                if not df.empty:
                    df["symbol"] = symbol
                    self.store.save_ohlcv(df, symbol, interval)
                    return df

        # Fallback to yfinance
        yf_interval_map = {
            "day": "1d", "week": "1wk", "month": "1mo",
            "60minute": "1h", "15minute": "15m", "5minute": "5m",
            "minute": "1m",
        }
        yf_interval = yf_interval_map.get(interval, "1d")
        df = self.yf.get_historical_data(symbol, start_date, end_date, yf_interval)

        if not df.empty:
            self.store.save_ohlcv(df, symbol, interval)
            logger.info(f"Fetched {len(df)} rows for {symbol} via yfinance")

        return df

    def fetch_options_chain(self, symbol: str = "NIFTY 50") -> pd.DataFrame:
        """Fetch and parse options chain data."""
        raw = self.nse.get_options_chain(symbol)
        if raw is None:
            return pd.DataFrame()

        df = self.nse.parse_options_chain(raw)
        if not df.empty:
            df["symbol"] = symbol
            self.store.save_options_chain(df)

        return df

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price from best available source."""
        # Try Kite first
        if self.kite.is_connected():
            prices = self.kite.get_ltp([symbol])
            if symbol in prices:
                return prices[symbol]

        # yfinance fallback
        return self.yf.get_current_price(symbol)

    def get_vix(self) -> Optional[float]:
        """Get current India VIX."""
        vix = self.nse.get_india_vix()
        if vix:
            return vix
        return self.yf.get_current_price("INDIA VIX")
