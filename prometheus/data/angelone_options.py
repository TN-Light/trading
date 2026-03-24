# ============================================================================
# PROMETHEUS — Angel One Option Chain Fetcher
# ============================================================================
"""
Fetches live option chain data via Angel One SmartAPI.
Replaces NSE website scraping with authenticated, reliable API calls.
Also provides real premium lookups for paper trading.

SmartAPI endpoints used:
- searchScrip("NFO", query) -> discover option contracts + symbol tokens
- getMarketData("FULL", {"NFO": [tokens]}) -> LTP, OI, bid/ask, volume
- optionGreek(params) -> server-computed delta, gamma, theta, vega, IV
"""

import time
import re
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional

from prometheus.utils.logger import logger


class AngelOneOptionChain:
    """Fetches live option chain and real premiums via Angel One SmartAPI."""

    # Map Prometheus symbol names to NFO underlying names
    UNDERLYING_MAP = {
        "NIFTY 50": "NIFTY",
        "NIFTY BANK": "BANKNIFTY",
        "NIFTY FIN SERVICE": "FINNIFTY",
        "SENSEX": "SENSEX",
        "NIFTY IT": "NIFTY IT",
        "NIFTY MIDCAP SELECT": "NIFTY MIDCAP SELECT",
    }

    def __init__(self, fetcher):
        """
        Reuses existing AngelOneFetcher's SmartConnect session.

        Args:
            fetcher: AngelOneFetcher instance (already has _obj, _ensure_connected)
        """
        self._fetcher = fetcher
        self._token_cache: Dict[str, List[Dict]] = {}  # keyed by "NIFTY"
        self._cache_date: str = ""  # invalidate daily
        self._last_call: float = 0.0
        self._min_interval: float = 0.35  # ~3 req/sec
        self._disabled_until: float = 0.0
        self._auth_cooldown_sec: int = 300

    def _is_temporarily_disabled(self) -> bool:
        return time.time() < self._disabled_until

    def _mark_auth_failure(self, reason: str):
        self._disabled_until = time.time() + self._auth_cooldown_sec
        logger.warning(
            f"Angel One option chain disabled for {self._auth_cooldown_sec}s due to auth failure: {reason}"
        )

    def _rate_limit(self):
        """Simple rate limiter for API calls."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _get_obj(self):
        """Get authenticated SmartConnect object."""
        if self._is_temporarily_disabled():
            return None
        if not self._fetcher._ensure_connected():
            return None
        return self._fetcher._obj

    # ------------------------------------------------------------------
    # Contract discovery
    # ------------------------------------------------------------------

    def discover_contracts(
        self,
        symbol: str,
        expiry_date: str = None,
        strikes_around_atm: int = 10,
        spot_price: float = None,
    ) -> List[Dict]:
        """
        Discover NFO option contracts via searchScrip.

        Args:
            symbol: e.g. "NIFTY 50"
            expiry_date: "YYYY-MM-DD" or None for nearest expiry
            strikes_around_atm: number of strikes above+below ATM
            spot_price: current spot price for ATM calculation

        Returns:
            List of {"tradingsymbol", "symboltoken", "strike", "option_type", "expiry"}
        """
        underlying = self.UNDERLYING_MAP.get(symbol, "NIFTY")
        today_str = date.today().isoformat()

        # Check daily cache
        cache_key = underlying
        if self._cache_date == today_str and cache_key in self._token_cache:
            contracts = self._token_cache[cache_key]
        else:
            obj = self._get_obj()
            if not obj:
                return []

            self._rate_limit()
            try:
                result = obj.searchScrip("NFO", underlying)
                if not result or not result.get("data"):
                    msg = str(result.get("message", "")) if isinstance(result, dict) else ""
                    code = str(result.get("errorCode", "")) if isinstance(result, dict) else ""
                    if code == "AG8001" or "invalid token" in msg.lower():
                        self._mark_auth_failure(msg or code or "Invalid Token")
                    logger.warning(f"searchScrip returned no data for {underlying}")
                    return []

                contracts = result["data"]
                self._token_cache[cache_key] = contracts
                self._cache_date = today_str
                logger.info(f"Angel One: discovered {len(contracts)} NFO contracts for {underlying}")
            except Exception as e:
                if "invalid token" in str(e).lower() or "AG8001" in str(e):
                    self._mark_auth_failure(str(e))
                logger.error(f"Angel One searchScrip error: {e}")
                return []

        # Filter contracts
        candidates = []
        for c in contracts:
            ts = c.get("tradingsymbol", "")
            token = c.get("symboltoken", "")

            # Parse tradingsymbol to extract strike/type/expiry
            parsed = self._parse_tradingsymbol(ts, underlying)
            if not parsed:
                continue

            # Filter by option type (skip futures)
            if parsed["option_type"] not in ("CE", "PE"):
                continue

            # Filter by strike range around ATM
            if spot_price and spot_price > 0:
                from prometheus.utils.indian_market import get_strike_interval
                interval = get_strike_interval(symbol)
                atm = round(spot_price / interval) * interval
                strike_range = strikes_around_atm * interval
                if abs(parsed["strike"] - atm) > strike_range:
                    continue

            candidates.append({
                "tradingsymbol": ts,
                "symboltoken": token,
                "strike": parsed["strike"],
                "option_type": parsed["option_type"],
                "expiry": parsed.get("expiry_str", ""),
            })

        # Primary: requested expiry only
        if expiry_date:
            filtered = [c for c in candidates if c.get("expiry") == expiry_date]
            if filtered:
                return filtered

            # Fallback: nearest available expiry to avoid empty chain due stale calendar mapping
            today_iso = date.today().isoformat()
            expiries = sorted({c.get("expiry", "") for c in candidates if c.get("expiry", "")})
            future_expiries = [e for e in expiries if e >= today_iso]
            chosen = future_expiries[0] if future_expiries else (expiries[0] if expiries else "")
            if chosen:
                fallback = [c for c in candidates if c.get("expiry") == chosen]
                if fallback:
                    logger.warning(
                        f"Angel One: requested expiry {expiry_date} unavailable for {symbol}; "
                        f"using nearest {chosen}"
                    )
                    return fallback

        return candidates

    def _parse_tradingsymbol(self, ts: str, underlying: str) -> Optional[Dict]:
        """
        Parse Angel One tradingsymbol to extract strike, option_type, expiry.
        E.g. "NIFTY25MAR22500CE" -> {"strike": 22500, "option_type": "CE", "expiry_str": "2025-03-27"}
        """
        try:
            # Pattern: UNDERLYING + DDMMMYY + STRIKE + CE/PE
            # or: UNDERLYING + DDMMMYYYY + STRIKE + CE/PE
            suffix = ts[len(underlying):]
            if not suffix:
                return None

            # Match option type at end
            if suffix.endswith("CE"):
                option_type = "CE"
                suffix = suffix[:-2]
            elif suffix.endswith("PE"):
                option_type = "PE"
                suffix = suffix[:-2]
            else:
                return None  # futures or unknown

            # Parse with explicit DDMMMYY pattern first (most common)
            # e.g. "07APR2623500" -> date="07APR26", strike="23500"
            m = re.match(r'^(\d{2}[A-Z]{3}\d{2})(\d+)$', suffix)
            if m:
                date_part, strike_str = m.groups()
                strike = float(strike_str)
                expiry_str = ""
                try:
                    dt = datetime.strptime(date_part, "%d%b%y")
                    expiry_str = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
                return {"strike": strike, "option_type": option_type, "expiry_str": expiry_str}

            # Fallback: DDMMMYYYY pattern (e.g. "07APR202623500")
            m = re.match(r'^(\d{2}[A-Z]{3}\d{4})(\d+)$', suffix)
            if m:
                date_part, strike_str = m.groups()
                strike = float(strike_str)
                expiry_str = ""
                try:
                    dt = datetime.strptime(date_part, "%d%b%Y")
                    expiry_str = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
                return {"strike": strike, "option_type": option_type, "expiry_str": expiry_str}

            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Market data (LTP, OI, bid/ask)
    # ------------------------------------------------------------------

    def fetch_market_data(self, contracts: List[Dict]) -> pd.DataFrame:
        """
        Batch-fetch market data for a list of contracts.

        Uses getMarketData("FULL", ...) in batches of 50.
        Returns DataFrame with: tradingsymbol, ltp, bid, ask, volume, oi, oi_change
        """
        obj = self._get_obj()
        if not obj or not contracts:
            return pd.DataFrame()

        tokens = [c["symboltoken"] for c in contracts]
        token_map = {c["symboltoken"]: c for c in contracts}
        results = []

        # Batch in groups of 50
        batch_size = 50
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            self._rate_limit()
            try:
                response = obj.getMarketData("FULL", {"NFO": batch})
                if response and response.get("status") and response.get("data"):
                    fetched = response["data"].get("fetched", [])
                    for item in fetched:
                        token = str(item.get("symbolToken", ""))
                        contract = token_map.get(token, {})
                        results.append({
                            "tradingsymbol": contract.get("tradingsymbol", ""),
                            "symboltoken": token,
                            "strike": contract.get("strike", 0),
                            "option_type": contract.get("option_type", ""),
                            "expiry": contract.get("expiry", ""),
                            "ltp": float(item.get("ltp", 0)),
                            "open": float(item.get("open", 0)),
                            "high": float(item.get("high", 0)),
                            "low": float(item.get("low", 0)),
                            "close": float(item.get("close", 0)),
                            "volume": int(item.get("tradeVolume", 0) or 0),
                            "oi": int(item.get("opnInterest", 0) or 0),
                            "oi_change": int(item.get("opnInterestChange", 0) or 0),
                            "bid": float(item.get("bestBidPrice", 0) or 0),
                            "ask": float(item.get("bestAskPrice", 0) or 0),
                            "underlying": float(item.get("ltp", 0)),
                        })
            except Exception as e:
                logger.warning(f"Angel One getMarketData error (batch {i}): {e}")

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def fetch_greeks(self, contracts: List[Dict]) -> Dict[str, Dict]:
        """
        Fetch option Greeks from Angel One for a list of contracts.

        Returns dict keyed by tradingsymbol: {delta, gamma, theta, vega, iv}
        """
        obj = self._get_obj()
        if not obj:
            return {}

        greeks_map = {}
        for c in contracts:
            self._rate_limit()
            try:
                params = {
                    "name": c.get("tradingsymbol", ""),
                    "expirydate": c.get("expiry", ""),
                    "strikeprice": str(c.get("strike", "")),
                    "optiontype": c.get("option_type", ""),
                }
                result = obj.optionGreek(params)
                if result and result.get("status") and result.get("data"):
                    data = result["data"]
                    greeks_map[c["tradingsymbol"]] = {
                        "delta": float(data.get("delta", 0) or 0),
                        "gamma": float(data.get("gamma", 0) or 0),
                        "theta": float(data.get("theta", 0) or 0),
                        "vega": float(data.get("vega", 0) or 0),
                        "iv": float(data.get("impliedVolatility", 0) or 0),
                    }
            except Exception as e:
                logger.debug(f"Angel One optionGreek error for {c.get('tradingsymbol')}: {e}")

        return greeks_map

    # ------------------------------------------------------------------
    # Master: get full option chain
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        symbol: str,
        spot_price: float = None,
        expiry_date: str = None,
        strikes_around_atm: int = 10,
        include_greeks: bool = False,
    ) -> pd.DataFrame:
        """
        Get full option chain with market data (and optionally Greeks).

        Returns DataFrame with columns matching OI Analyzer format:
        [option_type, strike, expiry, oi, oi_change, volume, iv, ltp, bid, ask, underlying]
        Plus optionally: delta, gamma, theta, vega
        """
        # Get nearest expiry if not specified
        if not expiry_date:
            from prometheus.utils.indian_market import get_expiry_date
            exp = get_expiry_date(symbol)
            expiry_date = exp.isoformat() if exp else None

        # Discover contracts
        contracts = self.discover_contracts(
            symbol, expiry_date, strikes_around_atm, spot_price
        )
        if not contracts:
            logger.warning(f"Angel One: no contracts found for {symbol}")
            return pd.DataFrame()

        # Fetch market data
        df = self.fetch_market_data(contracts)
        if df.empty:
            return df

        # Set underlying to spot price
        if spot_price:
            df["underlying"] = spot_price

        # Optionally fetch Greeks (expensive: 1 API call per contract)
        if include_greeks and len(contracts) <= 40:
            greeks = self.fetch_greeks(contracts)
            for col in ["delta", "gamma", "theta", "vega", "iv_greek"]:
                df[col] = 0.0
            for idx, row in df.iterrows():
                ts = row.get("tradingsymbol", "")
                if ts in greeks:
                    g = greeks[ts]
                    df.at[idx, "delta"] = g["delta"]
                    df.at[idx, "gamma"] = g["gamma"]
                    df.at[idx, "theta"] = g["theta"]
                    df.at[idx, "vega"] = g["vega"]
                    if g["iv"] > 0:
                        df.at[idx, "iv"] = g["iv"]

        logger.info(
            f"Angel One option chain: {symbol}, {len(df)} contracts, "
            f"spot={spot_price}, expiry={expiry_date}"
        )
        return df

    # ------------------------------------------------------------------
    # Single-contract lookup (for paper trading)
    # ------------------------------------------------------------------

    def get_real_premium(
        self,
        symbol: str,
        strike: float,
        option_type: str,
        expiry: str = None,
        spot_price: float = None,
    ) -> Optional[Dict]:
        """
        Get real premium for a single option contract.

        Returns: {"ltp", "bid", "ask", "spread", "delta", "gamma", "theta", "iv"}
        or None if unavailable.
        """
        # Find the contract token
        contracts = self.discover_contracts(
            symbol, expiry_date=expiry, strikes_around_atm=20, spot_price=spot_price
        )

        # Find exact match
        target = None
        for c in contracts:
            if c["strike"] == strike and c["option_type"] == option_type:
                target = c
                break

        if not target:
            # Try closest strike
            for c in contracts:
                if abs(c["strike"] - strike) < 1 and c["option_type"] == option_type:
                    target = c
                    break

        if not target:
            return None

        # Fetch market data for this one contract
        obj = self._get_obj()
        if not obj:
            return None

        self._rate_limit()
        try:
            result = obj.ltpData("NFO", target["tradingsymbol"], target["symboltoken"])
            if result and result.get("status") and result.get("data"):
                data = result["data"]
                ltp = float(data.get("ltp", 0) or 0)
                if ltp <= 0:
                    return None

                premium = {
                    "ltp": ltp,
                    "bid": 0.0,
                    "ask": 0.0,
                    "spread": 0.0,
                    "tradingsymbol": target["tradingsymbol"],
                    "symboltoken": target["symboltoken"],
                }

                # Try to get bid/ask via full market data
                self._rate_limit()
                try:
                    full = obj.getMarketData("FULL", {"NFO": [target["symboltoken"]]})
                    if full and full.get("data", {}).get("fetched"):
                        f = full["data"]["fetched"][0]
                        premium["bid"] = float(f.get("bestBidPrice", 0) or 0)
                        premium["ask"] = float(f.get("bestAskPrice", 0) or 0)
                        premium["spread"] = premium["ask"] - premium["bid"]
                        premium["oi"] = int(f.get("opnInterest", 0) or 0)
                        premium["volume"] = int(f.get("tradeVolume", 0) or 0)
                except Exception:
                    pass

                # Try Greeks
                self._rate_limit()
                try:
                    g_result = obj.optionGreek({
                        "name": target["tradingsymbol"],
                        "expirydate": expiry or target.get("expiry", ""),
                        "strikeprice": str(strike),
                        "optiontype": option_type,
                    })
                    if g_result and g_result.get("data"):
                        gd = g_result["data"]
                        premium["delta"] = float(gd.get("delta", 0) or 0)
                        premium["gamma"] = float(gd.get("gamma", 0) or 0)
                        premium["theta"] = float(gd.get("theta", 0) or 0)
                        premium["vega"] = float(gd.get("vega", 0) or 0)
                        premium["iv"] = float(gd.get("impliedVolatility", 0) or 0)
                except Exception:
                    pass

                return premium
        except Exception as e:
            logger.debug(f"Angel One get_real_premium error: {e}")
            return None

    def get_ltp_by_token(self, tradingsymbol: str) -> Optional[float]:
        """Quick LTP lookup for position monitoring."""
        # Search all cached contracts for this tradingsymbol
        for contracts in self._token_cache.values():
            for c in contracts:
                if c.get("tradingsymbol") == tradingsymbol:
                    token = c.get("symboltoken", "")
                    if token:
                        obj = self._get_obj()
                        if not obj:
                            return None
                        self._rate_limit()
                        try:
                            result = obj.ltpData("NFO", tradingsymbol, token)
                            if result and result.get("data"):
                                return float(result["data"].get("ltp", 0) or 0)
                        except Exception:
                            pass
                    return None
        return None


# ============================================================================
# Factory
# ============================================================================

def create_angelone_option_chain() -> Optional[AngelOneOptionChain]:
    """Create from existing AngelOneFetcher. Returns None if credentials missing."""
    try:
        from prometheus.data.angelone_fetcher import create_angelone_fetcher
        fetcher = create_angelone_fetcher()
        if fetcher is None:
            return None
        return AngelOneOptionChain(fetcher)
    except Exception as e:
        logger.warning(f"Angel One option chain init failed: {e}")
        return None
