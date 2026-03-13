# ============================================================================
# PROMETHEUS — Execution: Zerodha Kite Connect Implementation
# ============================================================================
"""
Production Zerodha Kite Connect order execution.
Handles authentication, order placement, position tracking, and margin checks.
"""

from typing import Dict, List, Optional
from datetime import datetime
import time

from prometheus.execution.broker import (
    BrokerBase, Order, Position, OrderType, OrderSide,
    ProductType, OrderStatus
)
from prometheus.utils.logger import logger


class KiteExecutor(BrokerBase):
    """Zerodha Kite Connect broker implementation."""

    # Map our enums to Kite's string values
    SIDE_MAP = {
        OrderSide.BUY: "BUY",
        OrderSide.SELL: "SELL",
    }
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.SL: "SL",
        OrderType.SL_M: "SL-M",
    }
    PRODUCT_MAP = {
        ProductType.MIS: "MIS",
        ProductType.NRML: "NRML",
        ProductType.CNC: "CNC",
    }

    def __init__(self, api_key: str, api_secret: str, access_token: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None
        self._last_order_time = 0
        self._min_order_interval = 0.5  # seconds between orders (rate limit safety)

    def connect(self) -> bool:
        """Connect and authenticate with Kite."""
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)

            if self.access_token:
                self.kite.set_access_token(self.access_token)
            else:
                logger.error(
                    "No access token. Generate one at: "
                    f"https://kite.trade/connect/login?api_key={self.api_key}"
                )
                return False

            # Verify connection
            profile = self.kite.profile()
            logger.info(f"Connected to Kite as: {profile.get('user_name', 'Unknown')}")
            return True

        except ImportError:
            logger.error("kiteconnect not installed. Run: pip install kiteconnect")
            return False
        except Exception as e:
            logger.error(f"Kite connection failed: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if Kite is connected."""
        if self.kite is None:
            return False
        try:
            self.kite.profile()
            return True
        except Exception:
            return False

    def place_order(self, order: Order) -> Order:
        """Place an order on Kite."""
        if not self.kite:
            order.status = OrderStatus.REJECTED
            order.message = "Broker not connected"
            return order

        # Rate limit safety
        elapsed = time.time() - self._last_order_time
        if elapsed < self._min_order_interval:
            time.sleep(self._min_order_interval - elapsed)

        try:
            params = {
                "tradingsymbol": order.tradingsymbol,
                "exchange": order.exchange,
                "transaction_type": self.SIDE_MAP[order.side],
                "quantity": order.quantity,
                "order_type": self.ORDER_TYPE_MAP[order.order_type],
                "product": self.PRODUCT_MAP[order.product],
                "variety": "regular",
                "tag": order.tag[:20] if order.tag else "PROMETHEUS",
            }

            if order.order_type == OrderType.LIMIT:
                params["price"] = order.price
            if order.order_type in (OrderType.SL, OrderType.SL_M):
                params["trigger_price"] = order.trigger_price
                if order.order_type == OrderType.SL:
                    params["price"] = order.price

            order_id = self.kite.place_order(**params)
            self._last_order_time = time.time()

            order.order_id = str(order_id)
            order.status = OrderStatus.OPEN
            order.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logger.info(
                f"Order placed: {order.side.value} {order.quantity} "
                f"{order.tradingsymbol} @ {order.order_type.value} "
                f"(ID: {order.order_id})"
            )

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            logger.error(f"Order placement failed: {e}")

        return order

    def modify_order(self, order_id: str, **kwargs) -> Order:
        """Modify an existing order on Kite."""
        if not self.kite:
            return Order(order_id=order_id, status=OrderStatus.REJECTED,
                         message="Broker not connected")

        try:
            self.kite.modify_order(
                variety="regular",
                order_id=order_id,
                **kwargs
            )
            logger.info(f"Order modified: {order_id}")
            return self.get_order_status(order_id)
        except Exception as e:
            logger.error(f"Order modify failed: {e}")
            return Order(order_id=order_id, status=OrderStatus.REJECTED,
                         message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on Kite."""
        if not self.kite:
            return False
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Order cancel failed: {e}")
            return False

    def get_order_status(self, order_id: str) -> Order:
        """Get status of a specific order."""
        if not self.kite:
            return Order(order_id=order_id, status=OrderStatus.REJECTED)

        try:
            order_history = self.kite.order_history(order_id)
            if not order_history:
                return Order(order_id=order_id, status=OrderStatus.REJECTED)

            latest = order_history[-1]  # most recent update
            status_map = {
                "COMPLETE": OrderStatus.COMPLETE,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "OPEN": OrderStatus.OPEN,
                "TRIGGER PENDING": OrderStatus.TRIGGER_PENDING,
            }

            return Order(
                order_id=order_id,
                tradingsymbol=latest.get("tradingsymbol", ""),
                side=OrderSide.BUY if latest.get("transaction_type") == "BUY" else OrderSide.SELL,
                quantity=latest.get("quantity", 0),
                price=latest.get("price", 0),
                trigger_price=latest.get("trigger_price", 0),
                status=status_map.get(latest.get("status", ""), OrderStatus.PENDING),
                filled_quantity=latest.get("filled_quantity", 0),
                average_price=latest.get("average_price", 0),
                message=latest.get("status_message", ""),
            )
        except Exception as e:
            logger.error(f"Order status fetch failed: {e}")
            return Order(order_id=order_id, status=OrderStatus.REJECTED, message=str(e))

    def get_orders(self) -> List[Order]:
        """Get all today's orders."""
        if not self.kite:
            return []
        try:
            orders = self.kite.orders()
            return [self._parse_kite_order(o) for o in orders]
        except Exception as e:
            logger.error(f"Orders fetch failed: {e}")
            return []

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self.kite:
            return []
        try:
            positions = self.kite.positions()
            result = []
            for p in positions.get("net", []):
                if p.get("quantity", 0) != 0:
                    result.append(Position(
                        tradingsymbol=p.get("tradingsymbol", ""),
                        exchange=p.get("exchange", "NFO"),
                        quantity=p.get("quantity", 0),
                        average_price=p.get("average_price", 0),
                        last_price=p.get("last_price", 0),
                        pnl=p.get("pnl", 0),
                        unrealized_pnl=p.get("unrealised", 0),
                        value=p.get("value", 0),
                    ))
            return result
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return []

    def get_margins(self) -> Dict:
        """Get available margins."""
        if not self.kite:
            return {}
        try:
            margins = self.kite.margins()
            equity = margins.get("equity", {})
            return {
                "available_cash": equity.get("available", {}).get("cash", 0),
                "available_margin": equity.get("available", {}).get("live_balance", 0),
                "used_margin": equity.get("utilised", {}).get("debits", 0),
                "total_collateral": equity.get("available", {}).get("collateral", 0),
            }
        except Exception as e:
            logger.error(f"Margins fetch failed: {e}")
            return {}

    def get_ltp(self, tradingsymbol: str, exchange: str = "NFO") -> float:
        """Get last traded price for an instrument."""
        if not self.kite:
            return 0.0
        try:
            key = f"{exchange}:{tradingsymbol}"
            data = self.kite.ltp([key])
            return data.get(key, {}).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"LTP fetch failed for {tradingsymbol}: {e}")
            return 0.0

    def _parse_kite_order(self, o: dict) -> Order:
        """Parse Kite order dict into our Order dataclass."""
        status_map = {
            "COMPLETE": OrderStatus.COMPLETE,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "OPEN": OrderStatus.OPEN,
            "TRIGGER PENDING": OrderStatus.TRIGGER_PENDING,
        }
        return Order(
            order_id=str(o.get("order_id", "")),
            tradingsymbol=o.get("tradingsymbol", ""),
            exchange=o.get("exchange", "NFO"),
            side=OrderSide.BUY if o.get("transaction_type") == "BUY" else OrderSide.SELL,
            order_type=OrderType(o.get("order_type", "MARKET")),
            product=ProductType(o.get("product", "MIS")),
            quantity=o.get("quantity", 0),
            price=o.get("price", 0),
            trigger_price=o.get("trigger_price", 0),
            status=status_map.get(o.get("status", ""), OrderStatus.PENDING),
            filled_quantity=o.get("filled_quantity", 0),
            average_price=o.get("average_price", 0),
            timestamp=str(o.get("order_timestamp", "")),
            tag=o.get("tag", ""),
            message=o.get("status_message", ""),
        )


def generate_tradingsymbol(
    symbol: str,
    expiry_date: str,
    strike: float,
    option_type: str
) -> str:
    """
    Generate Kite-compatible tradingsymbol for F&O.

    Example: NIFTY2530622500CE
    Format: {SYMBOL}{YY}{M}{DD}{STRIKE}{CE/PE}

    Args:
        symbol: "NIFTY" or "BANKNIFTY"
        expiry_date: "2025-03-06" format
        strike: 22500
        option_type: "CE" or "PE"
    """
    from datetime import datetime as dt
    d = dt.strptime(expiry_date, "%Y-%m-%d")

    # Month encoding: Jan=1..Dec=O (Kite uses 1-9, O, N, D)
    month_map = {
        1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
        7: "7", 8: "8", 9: "9", 10: "O", 11: "N", 12: "D"
    }

    yy = d.strftime("%y")
    m = month_map[d.month]
    dd = d.strftime("%d")

    # Strike formatting — remove .0 for whole numbers
    strike_str = str(int(strike)) if strike == int(strike) else str(strike)

    return f"{symbol}{yy}{m}{dd}{strike_str}{option_type}"
