# ============================================================================
# PROMETHEUS — Execution: Paper Trading Simulator
# ============================================================================
"""
Paper trading engine that simulates real broker behavior without real money.
Maintains virtual portfolio, tracks P&L, and logs all trades.
Essential for testing strategies before going live.
"""

from typing import Dict, List, Optional
from datetime import datetime
import uuid
import copy

from prometheus.execution.broker import (
    BrokerBase, Order, Position, OrderType, OrderSide,
    ProductType, OrderStatus
)
from prometheus.data.store import DataStore
from prometheus.backtest.engine import ZerodhaCostModel
from prometheus.utils.logger import logger


class PaperTrader(BrokerBase):
    """
    Simulated broker for paper trading.

    Simulates:
    - Order placement with simulated fills (instant for MARKET, price-based for LIMIT)
    - Position tracking with real-time P&L
    - Margin tracking (virtual)
    - Stop-loss and target order management
    - Realistic Zerodha cost model (brokerage, STT, GST, stamp duty, etc.)
    """

    def __init__(self, initial_capital: float = 200000):
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.used_margin = 0.0
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.store = DataStore()
        self._connected = True
        self._price_feed: Dict[str, float] = {}  # simulated price feed
        self._real_premiums: Dict[str, Dict] = {}  # real bid/ask from Angel One
        self.cost_model = ZerodhaCostModel()  # Realistic Zerodha fees
        self.total_costs = 0.0  # Running total of all fees paid

        logger.info(f"Paper Trader initialized with Rs {initial_capital:,.0f} (realistic costs enabled)")

    def connect(self) -> bool:
        """Paper trader is always connected."""
        self._connected = True
        logger.info("Paper Trader connected (simulated)")
        return True

    def is_connected(self) -> bool:
        return self._connected

    def update_prices(self, prices: Dict[str, float]):
        """
        Update simulated price feed.
        Call this with real-time or historical prices to keep positions updated.
        """
        self._price_feed.update(prices)
        self._update_positions()
        self._check_pending_orders()

    def get_mid_premium(self, tradingsymbol: str) -> float:
        """Calculate the fair-value mid-price. Fallback to LTP if spread is zero."""
        real = self._real_premiums.get(tradingsymbol)
        if not real:
            return self.get_ltp(tradingsymbol)
            
        bid = real["bid"]
        ask = real["ask"]
        ltp = real["ltp"]
        
        # If feed provides consolidated single price or no valid spread
        if bid <= 0 or ask <= 0 or bid == ask:
            return ltp
            
        return (bid + ask) / 2.0

    def set_real_premium(self, tradingsymbol: str, ltp: float, bid: float = 0, ask: float = 0):
        """
        Feed real option premium from Angel One for accurate paper execution.
        Uses bid for sells, ask for buys (worst-case fills like real market).
        """
        self._real_premiums[tradingsymbol] = {
            "ltp": ltp,
            "bid": bid if bid > 0 else ltp,
            "ask": ask if ask > 0 else ltp,
        }
        
        # Diagnostic print for the first few quotes of a symbol
        if not hasattr(self, "_diagnostic_print_counts"):
            self._diagnostic_print_counts = {}
        
        count = self._diagnostic_print_counts.get(tradingsymbol, 0)
        if count < 3:
            quote = self._real_premiums[tradingsymbol]
            mid = self.get_mid_premium(tradingsymbol)
            logger.info(f"DIAGNOSTIC [{tradingsymbol}] | Bid: {quote['bid']:.2f} | Ask: {quote['ask']:.2f} | LTP: {quote['ltp']:.2f} | Mid: {mid:.2f}")
            self._diagnostic_print_counts[tradingsymbol] = count + 1
            
        # Also update the generic price feed
        self._price_feed[tradingsymbol] = ltp

    def get_real_premium(self, tradingsymbol: str) -> Optional[Dict]:
        """Get real premium data if available."""
        return self._real_premiums.get(tradingsymbol)

    def place_order(self, order: Order) -> Order:
        """Simulate order placement."""
        order.order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        order.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Validate margin
        estimated_cost = self._estimate_cost(order)
        if estimated_cost > self.available_cash:
            order.status = OrderStatus.REJECTED
            order.message = (
                f"Insufficient margin. Required: Rs {estimated_cost:,.0f}, "
                f"Available: Rs {self.available_cash:,.0f}"
            )
            logger.warning(f"Paper order rejected: {order.message}")
            self.orders[order.order_id] = order
            return order

        if order.order_type == OrderType.MARKET:
            # Use real premium if available, else simulated price
            real = self._real_premiums.get(order.tradingsymbol)
            if real:
                # Real bid/ask → fill at actual spread (more realistic than %)
                if order.side == OrderSide.BUY:
                    fill_price = real["ask"]  # buy at ask
                else:
                    fill_price = real["bid"]  # sell at bid
                logger.debug(f"Real premium fill: {order.tradingsymbol} {order.side.value} @ {fill_price:.2f}")
            else:
                current_price = self._price_feed.get(order.tradingsymbol, order.price)
                if current_price <= 0:
                    current_price = order.price

                # Simulate slippage (0.15% — realistic options slippage, matches backtest)
                slippage = current_price * 0.0015
                if order.side == OrderSide.BUY:
                    fill_price = current_price + slippage
                else:
                    fill_price = current_price - slippage

            order.average_price = round(fill_price, 2)
            order.filled_quantity = order.quantity
            order.status = OrderStatus.COMPLETE

            self._update_position_from_fill(order)
            self._log_fill(order)

        elif order.order_type == OrderType.LIMIT:
            order.status = OrderStatus.OPEN
            logger.info(f"Paper LIMIT order placed: {order.order_id} waiting for fill at {order.price}")

        elif order.order_type in (OrderType.SL, OrderType.SL_M):
            order.status = OrderStatus.TRIGGER_PENDING
            logger.info(
                f"Paper SL order placed: {order.order_id} "
                f"trigger={order.trigger_price}"
            )

        self.orders[order.order_id] = order
        return order

    def modify_order(self, order_id: str, **kwargs) -> Order:
        """Modify a pending order."""
        if order_id not in self.orders:
            return Order(order_id=order_id, status=OrderStatus.REJECTED,
                         message="Order not found")

        order = self.orders[order_id]
        if order.status in (OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return Order(order_id=order_id, status=OrderStatus.REJECTED,
                         message=f"Cannot modify {order.status.value} order")

        for key, value in kwargs.items():
            if hasattr(order, key):
                setattr(order, key, value)

        logger.info(f"Paper order modified: {order_id}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status in (OrderStatus.COMPLETE, OrderStatus.CANCELLED):
            return False

        order.status = OrderStatus.CANCELLED
        logger.info(f"Paper order cancelled: {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        return self.orders.get(
            order_id,
            Order(order_id=order_id, status=OrderStatus.REJECTED, message="Not found")
        )

    def get_orders(self) -> List[Order]:
        """Get all orders."""
        return list(self.orders.values())

    def get_positions(self) -> List[Position]:
        """Get open positions."""
        return [p for p in self.positions.values() if p.quantity != 0]

    def get_margins(self) -> Dict:
        """Get virtual margins (includes realistic costs)."""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        realized = sum(t.get("net_pnl", 0) for t in self.trade_history)
        return {
            "available_cash": round(self.available_cash, 2),
            "available_margin": round(self.available_cash + total_unrealized, 2),
            "used_margin": round(self.used_margin, 2),
            "total_collateral": 0,
            "total_pnl": round(total_unrealized, 2),
            "equity": round(self.initial_capital + total_unrealized + realized, 2),
            "total_costs": round(self.total_costs, 2),
            "realized_pnl": round(realized, 2),
            "gross_pnl": round(sum(t.get("gross_pnl", 0) for t in self.trade_history), 2),
        }

    def get_ltp(self, tradingsymbol: str, exchange: str = "NFO") -> float:
        """Get last price from simulated feed."""
        return self._price_feed.get(tradingsymbol, 0.0)

    def get_portfolio_value(self) -> float:
        """Total portfolio value including unrealized P&L."""
        margins = self.get_margins()
        return margins.get("equity", self.initial_capital)

    def get_realized_pnl(self) -> float:
        """Sum of all realized P&L."""
        return sum(t.get("net_pnl", 0) for t in self.trade_history)

    def _update_position_from_fill(self, order: Order):
        """Update positions after an order fill."""
        key = order.tradingsymbol
        if key not in self.positions:
            self.positions[key] = Position(
                tradingsymbol=key,
                symbol=order.symbol,
                exchange=order.exchange,
                product=order.product,
            )

        pos = self.positions[key]
        fill_qty = order.filled_quantity
        fill_price = order.average_price

        if order.side == OrderSide.BUY:
            if pos.quantity >= 0:
                # Adding to long
                total_value = (pos.average_price * pos.quantity) + (fill_price * fill_qty)
                pos.quantity += fill_qty
                pos.average_price = total_value / pos.quantity if pos.quantity > 0 else 0
            else:
                # Closing short
                close_qty = min(fill_qty, abs(pos.quantity))
                realized_pnl = (pos.average_price - fill_price) * close_qty

                # Calculate realistic trading costs
                buy_value = fill_price * close_qty
                sell_value = pos.average_price * close_qty
                costs = self.cost_model.calculate_costs(buy_value, sell_value, "options")
                net_pnl = realized_pnl - costs["total"]
                self.total_costs += costs["total"]

                self.available_cash += net_pnl
                pos.quantity += fill_qty
                if pos.quantity > 0:
                    pos.average_price = fill_price

                self.trade_history.append({
                    "timestamp": order.timestamp,
                    "symbol": key,
                    "side": "COVER",
                    "quantity": fill_qty,
                    "price": fill_price,
                    "gross_pnl": round(realized_pnl, 2),
                    "costs": costs,
                    "net_pnl": round(net_pnl, 2),
                })

            # Deduct cost
            cost = fill_price * fill_qty
            self.available_cash -= cost
            self.used_margin += cost

        elif order.side == OrderSide.SELL:
            if pos.quantity <= 0:
                # Adding to short (option selling — margin required)
                total_value = (pos.average_price * abs(pos.quantity)) + (fill_price * fill_qty)
                pos.quantity -= fill_qty
                pos.average_price = total_value / abs(pos.quantity) if pos.quantity != 0 else 0
            else:
                # Closing long
                close_qty = min(fill_qty, pos.quantity)
                realized_pnl = (fill_price - pos.average_price) * close_qty

                # Calculate realistic trading costs
                buy_value = pos.average_price * close_qty
                sell_value = fill_price * close_qty
                costs = self.cost_model.calculate_costs(buy_value, sell_value, "options")
                net_pnl = realized_pnl - costs["total"]
                self.total_costs += costs["total"]

                self.available_cash += net_pnl + (pos.average_price * close_qty)
                self.used_margin -= pos.average_price * close_qty
                pos.quantity -= fill_qty
                if pos.quantity < 0:
                    pos.average_price = fill_price

                self.trade_history.append({
                    "timestamp": order.timestamp,
                    "symbol": key,
                    "side": "SELL",
                    "quantity": fill_qty,
                    "price": fill_price,
                    "gross_pnl": round(realized_pnl, 2),
                    "costs": costs,
                    "net_pnl": round(net_pnl, 2),
                })

            # Credit premium for option selling
            if pos.quantity < 0:
                self.available_cash += fill_price * fill_qty

    def _update_positions(self):
        """Update unrealized P&L using current prices."""
        for key, pos in self.positions.items():
            if pos.quantity == 0:
                continue

            # Use mid-price for fair-value SL evaluation and unrealized PnL, not bid/ask spreads
            if key in self._real_premiums:
                current_price = self.get_mid_premium(key)
            else:
                current_price = self._price_feed.get(key, pos.last_price)
                
            if current_price <= 0:
                continue

            pos.last_price = current_price
            if pos.quantity > 0:
                pos.unrealized_pnl = (current_price - pos.average_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.average_price - current_price) * abs(pos.quantity)
            pos.pnl = pos.unrealized_pnl

    def _check_pending_orders(self):
        """Check if any pending LIMIT or SL orders should be filled."""
        for order_id, order in self.orders.items():
            if order.status == OrderStatus.OPEN and order.order_type == OrderType.LIMIT:
                current = self._price_feed.get(order.tradingsymbol, 0)
                if current <= 0:
                    continue

                # LIMIT BUY fills when price drops to limit
                if order.side == OrderSide.BUY and current <= order.price:
                    order.average_price = order.price
                    order.filled_quantity = order.quantity
                    order.status = OrderStatus.COMPLETE
                    self._update_position_from_fill(order)
                    self._log_fill(order)

                # LIMIT SELL fills when price rises to limit
                elif order.side == OrderSide.SELL and current >= order.price:
                    order.average_price = order.price
                    order.filled_quantity = order.quantity
                    order.status = OrderStatus.COMPLETE
                    self._update_position_from_fill(order)
                    self._log_fill(order)

            elif order.status == OrderStatus.TRIGGER_PENDING:
                current = self._price_feed.get(order.tradingsymbol, 0)
                if current <= 0:
                    continue

                triggered = False
                if order.side == OrderSide.SELL and current <= order.trigger_price:
                    triggered = True  # SL for long position
                elif order.side == OrderSide.BUY and current >= order.trigger_price:
                    triggered = True  # SL for short position

                if triggered:
                    if order.order_type == OrderType.SL_M:
                        order.average_price = current
                    else:
                        order.average_price = order.price
                    order.filled_quantity = order.quantity
                    order.status = OrderStatus.COMPLETE
                    self._update_position_from_fill(order)
                    self._log_fill(order)
                    logger.info(f"Paper SL triggered: {order_id} at {current}")

    def _estimate_cost(self, order: Order) -> float:
        """Estimate margin/cost for an order."""
        if order.side == OrderSide.BUY:
            price = order.price if order.price > 0 else self._price_feed.get(order.tradingsymbol, 100)
            return price * order.quantity
        else:
            # Option selling requires higher margin — estimate 3x premium
            price = order.price if order.price > 0 else self._price_feed.get(order.tradingsymbol, 100)
            return price * order.quantity * 3

    def _log_fill(self, order: Order):
        """Log a filled order."""
        logger.info(
            f"[PAPER] {order.side.value} {order.filled_quantity} "
            f"{order.tradingsymbol} @ Rs {order.average_price:.2f} "
            f"(ID: {order.order_id})"
        )

        # Log to database
        self.store.log_trade({
            "symbol": order.tradingsymbol,
            "instrument": order.tradingsymbol,
            "action": order.side.value,
            "quantity": order.filled_quantity,
            "price": order.average_price,
            "order_type": order.order_type.value,
            "strategy": "paper",
            "signal_strength": 0,
            "stop_loss": 0,
            "target": 0,
        })
