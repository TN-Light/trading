"""
Fill simulator for the paper trading subsystem.

Responsible for translating an entry/exit decision into a realistic fill
price. Honors:

* Live LTP from Angel One when available (most realistic).
* Black-Scholes theoretical premium as a fallback.
* Caller-supplied price hint as a final safety net.

CRITICAL behavior — the 2026-07-17 paper bug:

    The PaperTrader.fill at MARKET price 0.0 caused a Rs 11,476 phantom loss
    when Angel One's quote subscription died at the 13:19 service restart.
    This simulator enforces the same rule the fix introduced in
    ``prometheus/execution/paper_trader.py``: **refuse to fill at 0.0**, fall
    back to a nonzero hint, and only REJECT if no usable price is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from prometheus.papertrade.types import Direction
from prometheus.utils.logger import logger


# Realistic options slippage — matches the backtest engine's 0.15% default.
DEFAULT_SLIPPAGE_BPS = 15  # 15 basis points = 0.15%


class PriceFeed(Protocol):
    """Minimal price-feed interface the simulator consumes.

    Any object exposing these methods will work (live broker, replay engine,
    SQLite-backed historical feed, etc.). The engine wires a concrete
    implementation in at construction time.
    """

    def get_ltp(self, instrument: str) -> float:
        """Return last-traded price for ``instrument`` or 0 if unavailable."""

    def get_quote(
        self, instrument: str
    ) -> Optional[tuple]:
        """Return ``(ltp, bid, ask)`` or ``None`` if no live quote is
        available. Bid/ask spreads are used when LTP alone is too crude
        (worst-case fills for buys at ask / sells at bid).
        """


@dataclass
class FillResult:
    fill_price: float
    source: str           # "live", "mid", "theoretical", "hint", "rejected"
    message: str = ""


class FillSimulator:
    """Converts entry/exit intent into a realistic fill price.

    Purely stateless — the engine owns the Snapshot the simulator is invoked
    against. Keeps a single tunable parameter (``slippage_bps``) for entire
    engine runs.

    Args:
        feed: implements ``PriceFeed`` (ltp / quote lookup).
        slippage_bps: basis points of slippage applied to LTP for MARKET fills.
            Default 15 bps = 0.15% (matches backtest engine).
        use_bid_ask: if True and a bid/ask quote is available, fill at the
            ask for BUY and the bid for SELL (worst-case, real-world behavior).
            Defaults True.
    """

    def __init__(
        self,
        feed: PriceFeed,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        use_bid_ask: bool = True,
    ):
        self.feed = feed
        self.slippage_bps = float(slippage_bps)
        self.use_bid_ask = bool(use_bid_ask)

    # ------------------------------------------------------------------
    def fill(
        self,
        instrument: str,
        direction: Direction,
        price_hint: float = 0.0,
        side: str = "BUY",   # "BUY" or "SELL"; for closing LONG we SELL
        theoretical_price: float = 0.0,
    ) -> FillResult:
        """Resolve a fill price for a market order.

        Args:
            instrument: API tradingsymbol.
            direction: LONG/SHORT of the underlying trade (only used for sign
                conventions if ``side`` alone is ambiguous).
            price_hint: explicit fallback price supplied by caller (the
                breakeven-SL or current-sl value computed by square-off).
                Used only when no live quote is available.
            side: ``"BUY"`` to acquire, ``"SELL"`` to dispose. For LONG trades
                entries are BUY and exits are SELL; for SHORT trades it's
                reversed. Specifying ``side`` explicitly avoids ambiguity.
            theoretical_price: Black-Scholes theoretical premium (already
                computed by signal_path); used as the final fallback if no
                quote and no hint.
        """
        # Priority 1: live bid/ask (worst-case real-world fill)
        if self.use_bid_ask:
            quote = None
            try:
                quote = self.feed.get_quote(instrument)
            except Exception as e:
                logger.debug(f"FillSimulator: get_quote failed for {instrument}: {e}")

            if quote:
                ltp, bid, ask = quote
                if side == "BUY" and ask > 0:
                    return FillResult(ask, "live_ask")
                if side == "SELL" and bid > 0:
                    return FillResult(bid, "live_bid")
                if ltp > 0:
                    # Bid/ask zero but LTP valid — use mid*(1±slippage)
                    slip = ltp * self.slippage_bps / 10000.0
                    fp = ltp + slip if side == "BUY" else ltp - slip
                    return FillResult(fp, "live_ltp_with_slippage")

        # Priority 2: LTP from the feed (with slippage)
        try:
            ltp = self.feed.get_ltp(instrument)
        except Exception as e:
            logger.debug(f"FillSimulator: get_ltp failed for {instrument}: {e}")
            ltp = 0.0

        if ltp > 0:
            slip = ltp * self.slippage_bps / 10000.0
            fp = ltp + slip if side == "BUY" else ltp - slip
            return FillResult(fp, "ltp_with_slippage")

        # Priority 3: explicit caller hint (e.g. breakeven SL precomputed)
        if price_hint > 0:
            logger.warning(
                f"FillSimulator: no live quote for {instrument}; using "
                f"caller hint Rs {price_hint:.2f} as fill price ({side} {direction.value})"
            )
            return FillResult(price_hint, "hint")

        # Priority 4: theoretical BS price (already computed upstream)
        if theoretical_price > 0:
            slip = theoretical_price * self.slippage_bps / 10000.0
            fp = theoretical_price + slip if side == "BUY" else theoretical_price - slip
            logger.warning(
                f"FillSimulator: no live quote for {instrument}; using "
                f"theoretical BS price Rs {theoretical_price:.2f} + {self.slippage_bps}bps "
                f"slippage = Rs {fp:.2f}"
            )
            return FillResult(fp, "theoretical")

        # Final: refuse to fill at 0.0. The 2026-07-17 paper bug filled at
        # 0.0 here, booking the entire premium as a phantom loss.
        logger.error(
            f"FillSimulator: refusing to fill {side} {instrument} {direction.value} "
            f"at 0.0 — no live quote, no caller hint, no theoretical price. "
            f"Rejecting the order instead of fabricating a price."
        )
        return FillResult(0.0, "rejected", message="no quote available, refusing to fill at 0.0")
