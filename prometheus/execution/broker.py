# ============================================================================
# PROMETHEUS — Execution: Broker Abstraction Layer
# ============================================================================
"""
Abstract broker interface. All broker-specific implementations
(Zerodha, Upstox, Dhan) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    MIS = "MIS"        # Intraday (Margin Intraday Square-off)
    NRML = "NRML"      # Overnight/carry forward
    CNC = "CNC"        # Cash and carry (delivery)


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGER_PENDING = "TRIGGER_PENDING"


@dataclass
class Order:
    """Represents a broker order."""
    order_id: str = ""
    symbol: str = ""
    exchange: str = "NFO"
    tradingsymbol: str = ""       # broker-specific symbol (e.g. "NIFTY2530622500CE")
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.MIS
    quantity: int = 0
    price: float = 0.0            # for LIMIT orders
    trigger_price: float = 0.0    # for SL/SL-M orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    timestamp: str = ""
    tag: str = ""                 # custom tag for tracking
    message: str = ""             # rejection/error message

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "tradingsymbol": self.tradingsymbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "product": self.product.value,
            "quantity": self.quantity,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "timestamp": self.timestamp,
            "tag": self.tag,
        }


@dataclass
class Position:
    """Represents an open position."""
    tradingsymbol: str = ""
    symbol: str = ""
    exchange: str = "NFO"
    product: ProductType = ProductType.MIS
    quantity: int = 0              # +ve for long, -ve for short
    average_price: float = 0.0
    last_price: float = 0.0
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    value: float = 0.0


class BrokerBase(ABC):
    """Abstract broker interface — all brokers implement this."""

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and connect to broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active."""
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Place an order. Returns updated Order with order_id and status."""
        pass

    @abstractmethod
    def modify_order(self, order_id: str, **kwargs) -> Order:
        """Modify an existing order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order."""
        pass

    @abstractmethod
    def get_orders(self) -> List[Order]:
        """Get all orders for today."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def get_margins(self) -> Dict:
        """Get available margins."""
        pass

    @abstractmethod
    def get_ltp(self, tradingsymbol: str, exchange: str = "NFO") -> float:
        """Get last traded price."""
        pass
