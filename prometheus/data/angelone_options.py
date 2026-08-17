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

    # Map Prometheus symbol names to NFO underlying names.
    # `searchScrip` queries by *tradingsymbol prefix*, not the long-form NSE name.
    # Earlier versions sent "NIFTY MIDCAP SELECT" verbatim, which returned no data —
    # Angel One's search requires the actual tradingsymbol root ("MIDCPNIFTY").
    # Similarly for FINNIFTY and NIFTY IT (sent as "NIFTY IT" before — also returned nothing).
    # SENSEX is the one exception: Angel One does index it under the long name.
    UNDERLYING_MAP = {
        "NIFTY 50": "NIFTY",
        "NIFTY BANK": "BANKNIFTY",
        "NIFTY FIN SERVICE": "FINNIFTY",
        "SENSEX": "SENSEX",
        "NIFTY IT": "NIFTYIT",
        "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
    }

    # Per-underlying options-exchange segment.
    #
    # Bug (2026-07-28 audit): all five Angel One API calls in this file
    # hardcoded ``"NFO"`` (NSE F&O) as the exchange segment. SENSEX
    # options don't trade on NSE — they trade on BSE F&O (segment code
    # "BFO"). ``searchScrip("NFO", "SENSEX")`` therefore correctly
    # returned empty (the BSE-listed contracts weren't there), and
    # every SENSEX signal generated in production was silently dropped
    # by ``main.py:_price_options`` at line 5493-5503 ("no live LTP
    # from Angel One — signal dropped"). On 2026-07-28 alone, 8 of 8
    # generated signals were SENSEX, and 100% of them were dropped —
    # so the operator got zero paper-trade entries for the entire
    # session despite a working signal engine and a logged-in Angel
    # One account.
    #
    # Fix: per-underlying exchange-segment lookup. ``_exchange_for``
    # returns "BFO" for SENSEX (and any future BSE-listed contract)
    # and "NFO" for everything else (NIFTY 50, NIFTY BANK, FINNIFTY,
    # NIFTY IT, NIFTY MIDCAP SELECT, NIFTY MIDCAP, etc.). The same
    # lookup is applied at every API call site that previously
    # hardcoded "NFO": ``searchScrip``, two ``getMarketData``, two
    # ``ltpData``.
    OPTIONS_EXCHANGE_SEGMENTS = {
        "NIFTY": "NFO",
        "BANKNIFTY": "NFO",
        "FINNIFTY": "NFO",
        "NIFTYIT": "NFO",
        "MIDCPNIFTY": "NFO",
        "SENSEX": "BFO",  # BSE F&O options
    }

    @classmethod
    def _exchange_for(cls, underlying: str) -> str:
        """Resolve the Angel One exchange segment (NFO vs BFO) for an
        underlying root name (``UNDERLYING_MAP`` value). Defaults to NFO
        for unknown underlyings so future NSE-listed contracts don't
        have to update the map."""
        return cls.OPTIONS_EXCHANGE_SEGMENTS.get(underlying, "NFO")

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
        """Disable the option-chain path AND propagate to the shared
        cooldown so the historical fetcher / VIX fetch respect it too.

        Bug (2026-08-17 audit): previously ``_mark_auth_failure`` only
        set ``_disabled_until`` (read by ``_get_obj``), but the historical
        fetcher's SmartAPIRateLimiter was never informed — so after an
        AG8001 the option-chain stayed silent for 5 min while the fetcher
        kept firing at 1.0s pace and either got its own AG8001 or
        competed for the same broken session.
        """
        self._disabled_until = time.time() + self._auth_cooldown_sec
        rl = getattr(self._fetcher, "_rate_limiter", None)
        if rl is not None:
            rl.set_cooldown(float(self._auth_cooldown_sec))
        logger.warning(
            f"Angel One option chain disabled for {self._auth_cooldown_sec}s due to auth failure: {reason}"
        )

    def _mark_rate_limited(self, result, where: str) -> bool:
        """Detect AB1021/AB1020/429 in an API response and propagate a
        20s global cooldown to the shared SmartAPIRateLimiter.

        Returns True iff the response looks rate-limited.  Callers MUST
        skip fallbacks when this returns True — re-trying within
        milliseconds is what triggered the SDK's 131-second urllib3 retry
        cascade (documented in commit fbd9ddd) before August 2026.

        Bug (2026-08-17 audit): ``discover_contracts`` only inspected
        responses for ``AG8001`` ("invalid token") and silently bailed
        on ``AB1021`` ("too many requests") without setting any
        cooldown. Every scan-cycle retry pattern burned through the
        retry budget until the Angel One endpoint itself cooled down.
        """
        if not result or not isinstance(result, dict):
            return False
        if result.get("status"):
            return False
        code = str(result.get("errorcode") or result.get("errorCode") or "")
        msg = str(result.get("message", "") or "")
        is_rl = (
            code in ("AB1021", "AB1020")
            or "too many requests" in msg.lower()
            or "exceeding access rate" in msg.lower()
            or "access denied" in msg.lower()
        )
        if is_rl:
            rl = getattr(self._fetcher, "_rate_limiter", None)
            if rl is not None:
                rl.set_cooldown(20.0)
            logger.warning(
                f"Angel One option-chain rate-limited in {where} "
                f"(code={code or 'unknown'}, msg={msg[:80]}); "
                f"propagated 20s global cooldown"
            )
        return is_rl

    def _rate_limit(self):
        """Gate every option-chain API call through the shared
        SmartAPIRateLimiter (1.0s minimum gap + global cooldown).

        Bug (2026-08-17 audit): the option-chain path had its own
        independent 0.35s pacer, separate from the historical fetcher's
        SmartAPIRateLimiter added in commit fbd9ddd.  With max_workers=3
        concurrent scan threads + 7 intraday instruments (Session 30
        expansion), the option-chain alone fired at ~3 calls/sec/worker
        = ~9 calls/sec across workers, vs Angel One's ~3 req/sec
        limit. The two limiters were totally uncoordinated — a 429 on
        searchScrip set no cooldown visible to ``fetch_historical``,
        and vice-versa.  Unifying the gate on the fetcher's existing
        ``SmartAPIRateLimiter`` means ALL Angel One callers
        (historical fetcher, VIX fetch, searchScrip, getMarketData,
        ltpData, optionGreek) respect a single global rate and
        cooldown.

        Backward-compat: if the fetcher doesn't expose
        ``_rate_limiter`` (e.g. legacy callers or test fixtures via
        ``__new__``), fall back to the original 0.35s pacer so unit
        tests like ``test_angelone_options_searchScrip_uses_resolved_segment_per_underlying``
        keep working unchanged.
        """
        rl = getattr(self._fetcher, "_rate_limiter", None)
        if rl is not None:
            rl.wait()
            return
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
                # Bug (2026-07-28 audit): use the per-underlying exchange
                # segment (NFO vs BFO) instead of hardcoded "NFO" — SENSEX
                # options trade on BFO, so searchScrip("NFO", "SENSEX")
                # returned empty and every SENSEX signal was dropped.
                seg = self._exchange_for(underlying)

                # Bug (2026-08-14 audit): Angel One tradingsymbols use the
                # DD-MON-YY format: NIFTY18AUG2624400CE (Aug 18 2026, 24400).
                # searchScrip does prefix-matching, so "NIFTY" matches both
                # NIFTY and NIFTYNXT50 (7120 contaminating results).
                # Fix: build a date-specific query from the expiry date to
                # narrow the search (e.g. "NIFTY18AUG" for Aug 18 expiry).
                # Falls back to bare underlying if no expiry is available.
                search_query = underlying
                if expiry_date:
                    try:
                        from datetime import datetime as _dt
                        exp_dt = _dt.strptime(expiry_date[:10], "%Y-%m-%d")
                        dd = f"{exp_dt.day:02d}"
                        mon = exp_dt.strftime("%b").upper()
                        search_query = f"{underlying}{dd}{mon}"
                    except Exception:
                        pass

                result = obj.searchScrip(seg, search_query)
                # Bug (2026-08-17 audit): if the FIRST searchScrip tripped
                # AB1021, do NOT immediately fire the fallback — that
                # would double the rate-limit window. The shared limiter
                # has now set a 20s global cooldown; bail this call so
                # other workers / symbols take the wait too.
                if self._mark_rate_limited(result, f"searchScrip('{seg}', '{search_query}')"):
                    return []

                if not result or not result.get("data"):
                    # Fallback ONLY when the first call returned no data
                    # due to a tradingsymbol format mismatch (e.g. stale
                    # daily-expiry calendar). Skip fallback for AB1021 —
                    # it would extend the rate-limit window.
                    if search_query != underlying:
                        result = obj.searchScrip(seg, underlying)
                        if self._mark_rate_limited(result, f"searchScrip_fallback('{seg}', '{underlying}')"):
                            return []

                if not result or not result.get("data"):
                    msg = str(result.get("message", "")) if isinstance(result, dict) else ""
                    code = str(result.get("errorCode", "")) if isinstance(result, dict) else ""
                    if code == "AG8001" or "invalid token" in msg.lower():
                        self._mark_auth_failure(msg or code or "Invalid Token")
                    logger.warning(f"searchScrip returned no data for {underlying} on {seg}")
                    return []

                raw_contracts = result["data"]

                # Post-filter: keep ONLY contracts whose tradingsymbol
                # starts with the exact underlying followed by a digit
                # (the day code). This rejects prefix collisions like
                # NIFTYNXT50 when searching for NIFTY.
                contracts = [
                    c for c in raw_contracts
                    if c.get("tradingsymbol", "").startswith(underlying)
                    and len(c.get("tradingsymbol", "")) > len(underlying)
                    and c["tradingsymbol"][len(underlying)].isdigit()
                ]

                self._token_cache[cache_key] = contracts
                self._cache_date = today_str
                logger.info(
                    f"Angel One: discovered {len(contracts)} {seg} contracts "
                    f"for {underlying} (raw={len(raw_contracts)}, "
                    f"query={search_query})"
                )
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

        Angel One searchScrip returns tradingsymbols in DD-MON-YY format:
            {SYMBOL}{DD}{MON}{YY}{STRIKE}{CE/PE}
            e.g. NIFTY18AUG2624400CE  -> Aug 18, 2026, strike 24400

        This is DIFFERENT from Kite's format (YY-MON-strike for monthly,
        YY-M-DD-strike for weekly). We try the Angel One format first
        (since this parser only processes searchScrip data), then fall
        back to the Kite formats for compatibility.

        Bug (2026-08-14 audit): the previous parser only handled Kite
        formats, so Angel One's `NIFTY18AUG2624400CE` was parsed as
        YY=18 MON=AUG strike=2624400 (absurd) -> filtered by ATM range
        -> get_real_premium returned None -> ALL NIFTY signals dropped
        since July 28 when the Live LTP Required gate was added.
        """
        try:
            from prometheus.utils.indian_market import (
                get_monthly_expiry, _resolve_weekly_expiry_day_name,
            )
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

            # ── Angel One format (primary): DD + 3-letter MON + YY + strike ──
            # e.g. "18AUG2624400" → DD=18, MON=AUG, YY=26, strike=24400
            # Plausibility guard: the same regex matches Kite-format symbols
            # (e.g. BANKNIFTY "26JUL56900" → DD=26, MON=JUL, YY=56, strike=900
            # which is year 2056, absurd). We accept the Angel One parse ONLY
            # if the derived year is within ±3 of now, day is 1-31, and strike
            # is > 100. Otherwise fall through to the Kite parser.
            m = re.match(r'^(\d{2})([A-Z]{3})(\d{2})(\d+)$', suffix)
            if m:
                dd_str, mon_str, yy_str, strike_str = m.groups()
                ao_year = 2000 + int(yy_str)
                ao_day = int(dd_str)
                ao_strike = float(strike_str)
                now_year = datetime.now().year
                if (1 <= ao_day <= 31
                        and abs(ao_year - now_year) <= 3
                        and ao_strike > 100):
                    expiry_str = ""
                    try:
                        month = datetime.strptime(mon_str, "%b").month
                        expiry_dt = date(ao_year, month, ao_day)
                        expiry_str = expiry_dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                    return {"strike": ao_strike, "option_type": option_type,
                            "expiry_str": expiry_str}

            # ── Kite monthly format (fallback): YY + 3-letter MON + strike ──
            # e.g. "26JUL56900" → YY=26, MON=JUL, strike=56900
            m = re.match(r'^(\d{2})([A-Z]{3})(\d+)$', suffix)
            if m:
                yy_str, mon_str, strike_str = m.groups()
                strike = float(strike_str)
                expiry_str = ""
                try:
                    year = 2000 + int(yy_str)
                    month = datetime.strptime(mon_str, "%b").month
                    expiry_weekday = _resolve_weekly_expiry_day_name(
                        underlying, on_date=date(year, month, 15),
                    )
                    monthly_dt = get_monthly_expiry(year, month, expiry_weekday)
                    expiry_str = monthly_dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                return {"strike": strike, "option_type": option_type,
                        "expiry_str": expiry_str}

            # ── Kite weekly format: YY + single-char M + DD + strike ──
            # M is 1-9 (Jan-Sep), O (Oct), N (Nov), D (Dec)
            month_map = {
                "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                "7": 7, "8": 8, "9": 9, "O": 10, "N": 11, "D": 12,
            }
            m = re.match(r'^(\d{2})([1-9OND])(\d{2})(\d+)$', suffix)
            if m:
                yy_str, m_char, dd_str, strike_str = m.groups()
                strike = float(strike_str)
                expiry_str = ""
                try:
                    year = 2000 + int(yy_str)
                    month = month_map[m_char]
                    day = int(dd_str)
                    expiry_dt = date(year, month, day)
                    expiry_str = expiry_dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                return {"strike": strike, "option_type": option_type,
                        "expiry_str": expiry_str}

            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Market data (LTP, OI, bid/ask)
    # ------------------------------------------------------------------

    def fetch_market_data(self, contracts: List[Dict], underlying: str = None) -> pd.DataFrame:
        """
        Batch-fetch market data for a list of contracts.

        Uses getMarketData("FULL", ...) in batches of 50.
        Returns DataFrame with: tradingsymbol, ltp, bid, ask, volume, oi, oi_change

        Bug (2026-07-28 audit): ``underlying`` resolves the exchange segment
        (NFO vs BFO). When omitted, falls back to "NFO" (legacy behavior) —
        but callers passing SENSEX contracts MUST set ``underlying="SENSEX"``
        so the request ships on BFO and not the wrong NSE F&O segment.
        """
        obj = self._get_obj()
        if not obj or not contracts:
            return pd.DataFrame()

        # Bug (2026-07-28 audit): resolve segment from underlying, default NFO.
        seg = self._exchange_for(underlying) if underlying else "NFO"

        tokens = [c["symboltoken"] for c in contracts]
        token_map = {c["symboltoken"]: c for c in contracts}
        results = []

        # Batch in groups of 50
        batch_size = 50
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            self._rate_limit()
            try:
                # Bug (2026-07-28 audit): route BSE-listed contracts
                # (SENSEX) on BFO instead of hardcoding "NFO".
                response = obj.getMarketData("FULL", {seg: batch})
                # Bug (2026-08-17 audit): detect AB1021 on getMarketData
                # — bail this batch and let the shared cooldown govern
                # subsequent fetches across all symbols.
                if self._mark_rate_limited(response, f"getMarketData({seg}, batch={i})"):
                    # Stop the loop; remaining batches will hit the
                    # shared cooldown automatically and clear in ~20s.
                    break
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
        # Bug (2026-07-28 audit): pass the underlying so the exchange
        # segment (NFO vs BFO) is routed correctly — SENSEX contracts
        # ship on BFO, others on NFO. Without this, every SENSEX option
        # chain fetch went to NFO and returned empty.
        underlying = self.UNDERLYING_MAP.get(symbol, "NIFTY")
        df = self.fetch_market_data(contracts, underlying=underlying)
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

        # If expiry is None, filter to the nearest available expiry to avoid matching longer-dated contracts
        if not expiry and contracts:
            expiries = sorted({c.get("expiry", "") for c in contracts if c.get("expiry", "")})
            today_iso = date.today().isoformat()
            future_expiries = [e for e in expiries if e >= today_iso]
            nearest_expiry = future_expiries[0] if future_expiries else (expiries[0] if expiries else None)
            if nearest_expiry:
                contracts = [c for c in contracts if c.get("expiry") == nearest_expiry]

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
            # Bug (2026-07-28 audit): resolve segment (NFO vs BFO) from
            # the symbol's underlying so SENSEX contracts route on BFO.
            underlying = self.UNDERLYING_MAP.get(symbol, "NIFTY")
            seg = self._exchange_for(underlying)
            result = obj.ltpData(seg, target["tradingsymbol"], target["symboltoken"])
            # Bug (2026-08-17 audit): AB1021 on ltpData previously passed
            # through silently as None — the next getMarketData / optionGreek
            # retried in the same call. Bail immediately on rate-limit so
            # the shared cooldown governs all pending Angel One callers.
            if self._mark_rate_limited(result, f"ltpData({seg}, {target['tradingsymbol']})"):
                return None
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
                    # Bug (2026-07-28 audit): same segment routing as the
                    # ltpData call above — SENSEX ships on BFO.
                    full = obj.getMarketData("FULL", {seg: [target["symboltoken"]]})
                    if not self._mark_rate_limited(full, f"getMarketData({seg}, {target['tradingsymbol']})"):
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
                    if not self._mark_rate_limited(g_result, f"optionGreek({target['tradingsymbol']})"):
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
        """Quick LTP lookup for position monitoring.

        Bug (2026-07-28 audit): the ``ltpData`` call below hardcoded
        ``"NFO"``. Since ``_token_cache`` is keyed by underlying root
        (e.g. ``"NIFTY"``, ``"SENSEX"``), iterating ``items()`` lets
        us resolve the correct exchange segment per cache bucket —
        so SENSEX contracts route on BFO and NIFTY routes on NFO.
        """
        # Search all cached contracts for this tradingsymbol
        for underlying, contracts in self._token_cache.items():
            for c in contracts:
                if c.get("tradingsymbol") == tradingsymbol:
                    token = c.get("symboltoken", "")
                    if token:
                        obj = self._get_obj()
                        if not obj:
                            return None
                        self._rate_limit()
                        try:
                            # Bug (2026-07-28 audit): route via the
                            # correct segment (NFO vs BFO) keyed by the
                            # underlying cache key.
                            seg = self._exchange_for(underlying)
                            result = obj.ltpData(seg, tradingsymbol, token)
                            # Bug (2026-08-17 audit): AB1021 on the
                            # position-monitor LTP poll previously
                            # silently returned None — but the next bar
                            # cycle re-fired immediately, extending the
                            # rate-limit window. Propagate cooldown so
                            # subsequent bars respect it too.
                            if self._mark_rate_limited(result, f"ltpData({seg}, {tradingsymbol})"):
                                return None
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
