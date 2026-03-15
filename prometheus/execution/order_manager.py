# ============================================================================
# PROMETHEUS — Execution: Order Manager
# ============================================================================
"""
Manages the full lifecycle of orders — from signal to execution to exit.
Bridges strategy signals with broker execution.
Handles multi-leg orders for spreads and straddles.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from prometheus.execution.broker import (
    BrokerBase, Order, OrderType, OrderSide, ProductType, OrderStatus
)
from prometheus.execution.kite_executor import generate_tradingsymbol
from prometheus.risk.manager import RiskManager
from prometheus.data.store import DataStore
from prometheus.utils.indian_market import get_lot_size, get_expiry_date
from prometheus.utils.logger import logger


@dataclass
class ManagedPosition:
    """A position being managed by the order manager."""
    position_id: str
    symbol: str
    strategy: str
    direction: str
    entry_orders: List[Order]
    exit_orders: List[Order]
    stop_loss: float
    target: float
    trailing_stop: float
    entry_time: str
    status: str = "open"      # open, partial, closed
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    # Live trailing stop support
    tradingsymbol: str = ""
    entry_premium: float = 0.0
    sl_order_id: str = ""
    max_bars: int = 7
    breakeven_ratio: float = 0.6


class OrderManager:
    """
    Orchestrates order execution with risk checks.

    Flow:
    1. Receive signal (FusedSignal or TradeSetup)
    2. Run risk pre-check
    3. Build order(s) based on strategy type
    4. Submit to broker
    5. Monitor fills
    6. Place protective orders (SL, target)
    7. Track position until closed
    """

    def __init__(
        self,
        broker: BrokerBase,
        risk_manager: RiskManager,
        mode: str = "paper"
    ):
        self.broker = broker
        self.risk = risk_manager
        self.mode = mode
        self.store = DataStore()
        self.managed_positions: Dict[str, ManagedPosition] = {}
        self._position_counter = 0

    def execute_signal(
        self,
        signal: Dict,
        confirm: bool = True
    ) -> Optional[ManagedPosition]:
        """
        Execute a trading signal through the full pipeline.

        Args:
            signal: Dict with keys: symbol, direction, action, entry_price,
                    stop_loss, target, confidence, strategy, reasoning
            confirm: If True (semi-auto mode), wait for user confirmation

        Returns:
            ManagedPosition if executed, None if rejected
        """
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        target = signal.get("target", 0)
        confidence = signal.get("confidence", 0)
        strategy = signal.get("strategy", "")
        action = signal.get("action", "HOLD")

        if action == "HOLD":
            return None

        # Step 1: Risk pre-check
        risk_check = self.risk.pre_trade_check({
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "stop_loss": sl,
            "quantity": 0,
            "cost": entry * get_lot_size(symbol),
        })

        if not risk_check.approved:
            logger.warning(
                f"Trade REJECTED by risk manager: {risk_check.reason} "
                f"(violations: {risk_check.violations})"
            )
            return None

        # Step 2: Position sizing via risk-based calculation
        lot_size = get_lot_size(symbol)
        sizing = self.risk.calculate_position_size(entry, sl, lot_size)
        quantity = sizing.get("quantity", lot_size)
        if quantity <= 0:
            quantity = lot_size  # minimum 1 lot

        # Step 3: Build orders based on action type
        if action in ("BUY_CE", "BUY_PE"):
            return self._execute_directional_options(
                signal, quantity, risk_check
            )
        elif action == "BUY_STRADDLE":
            return self._execute_straddle(signal, quantity)
        elif action == "BUY_STRANGLE":
            return self._execute_strangle(signal, quantity)
        else:
            logger.warning(f"Unknown action: {action}")
            return None

    def _execute_directional_options(
        self,
        signal: Dict,
        quantity: int,
        risk_check
    ) -> Optional[ManagedPosition]:
        """Execute a directional options buying trade."""
        symbol = signal["symbol"]
        action = signal["action"]
        option_type = "CE" if "CE" in action else "PE"
        lot_size = get_lot_size(symbol)

        # Use enriched signal data if available (from refine_with_strategy),
        # otherwise fall back to ATM calculation
        if signal.get("strike") and signal.get("instrument"):
            strike = signal["strike"]
            tradingsymbol = signal["instrument"]
            estimated_premium = signal.get("entry_price", 0)
            lots = signal.get("lots", max(1, quantity // lot_size))
            total_qty = lots * lot_size
        else:
            from prometheus.utils.indian_market import get_atm_strike
            spot = signal.get("spot_price", signal.get("entry_price", 0))
            strike = get_atm_strike(spot, symbol)
            expiry = get_expiry_date(symbol)
            lots = max(1, quantity // lot_size)
            total_qty = lots * lot_size

            underlying = symbol.replace(" ", "").replace("NIFTY50", "NIFTY").replace("NIFTYBANK", "BANKNIFTY")
            if "BANK" in symbol.upper():
                underlying = "BANKNIFTY"
            elif "FIN" in symbol.upper():
                underlying = "FINNIFTY"
            else:
                underlying = "NIFTY"

            expiry_str = expiry.strftime("%Y-%m-%d") if hasattr(expiry, 'strftime') else str(expiry)
            tradingsymbol = generate_tradingsymbol(underlying, expiry_str, strike, option_type)
            estimated_premium = signal.get("entry_price", 0)

        # Build entry order (set price for PaperTrader reference)
        entry_order = Order(
            symbol=symbol,
            tradingsymbol=tradingsymbol,
            exchange="NFO",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            product=ProductType.MIS,  # Intraday
            quantity=total_qty,
            price=estimated_premium,  # reference price for paper fills
            tag=f"P-{signal.get('strategy', 'TREND')[:5]}",
        )

        # Place entry order
        entry_order = self.broker.place_order(entry_order)

        if entry_order.status == OrderStatus.REJECTED:
            logger.error(f"Entry order rejected: {entry_order.message}")
            return None

        # Create managed position
        position_id = self._next_position_id()
        managed = ManagedPosition(
            position_id=position_id,
            symbol=symbol,
            strategy=signal.get("strategy", "trend"),
            direction=signal.get("direction", ""),
            entry_orders=[entry_order],
            exit_orders=[],
            stop_loss=signal.get("stop_loss", 0),
            target=signal.get("target", 0),
            trailing_stop=0,
            entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.managed_positions[position_id] = managed

        # Place stop-loss order
        if managed.stop_loss > 0 and entry_order.status == OrderStatus.COMPLETE:
            sl_order = self._place_stop_loss(
                tradingsymbol, total_qty, managed.stop_loss, option_type,
                entry_order.average_price
            )
            if sl_order:
                managed.exit_orders.append(sl_order)

        # Populate live trailing stop fields
        managed.tradingsymbol = tradingsymbol
        managed.entry_premium = entry_order.average_price
        if managed.exit_orders:
            managed.sl_order_id = managed.exit_orders[0].order_id

        # Record entry in risk manager (fixes position/trade tracking)
        self.risk.record_trade_entry({
            "symbol": symbol,
            "instrument": tradingsymbol,
            "direction": signal.get("direction", ""),
            "entry_price": entry_order.average_price,
            "quantity": total_qty,
            "cost": entry_order.average_price * total_qty,
        })

        # Log to database
        self.store.log_trade({
            "symbol": tradingsymbol,
            "instrument": tradingsymbol,
            "action": "BUY",
            "quantity": total_qty,
            "price": entry_order.average_price,
            "order_type": "MARKET",
            "strategy": signal.get("strategy", "trend"),
            "signal_strength": signal.get("confidence", 0),
            "stop_loss": managed.stop_loss,
            "target": managed.target,
        })

        logger.info(
            f"Position opened: {position_id} | "
            f"{lots}L {tradingsymbol} @ {entry_order.average_price:.2f} | "
            f"SL: {managed.stop_loss:.2f} | Target: {managed.target:.2f}"
        )

        return managed

    def _execute_straddle(self, signal: Dict, quantity: int) -> Optional[ManagedPosition]:
        """Execute a straddle (buy ATM CE + ATM PE)."""
        legs = signal.get("legs", [])
        if len(legs) < 2:
            return None

        position_id = self._next_position_id()
        entry_orders = []

        for leg in legs:
            order = Order(
                symbol=signal["symbol"],
                tradingsymbol=leg.get("tradingsymbol", ""),
                exchange="NFO",
                side=OrderSide.BUY if leg.get("action") == "BUY" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                product=ProductType.MIS,
                quantity=leg.get("lots", 1) * leg.get("lot_size", 25),
                tag=f"P-VOL-{position_id[-4:]}",
            )
            filled = self.broker.place_order(order)
            entry_orders.append(filled)

            if filled.status == OrderStatus.REJECTED:
                # Cancel already-placed legs
                for prev in entry_orders[:-1]:
                    if prev.status in (OrderStatus.OPEN, OrderStatus.COMPLETE):
                        self._close_order(prev)
                logger.error(f"Straddle leg rejected: {filled.message}")
                return None

        managed = ManagedPosition(
            position_id=position_id,
            symbol=signal["symbol"],
            strategy="volatility",
            direction="neutral",
            entry_orders=entry_orders,
            exit_orders=[],
            stop_loss=0,  # Straddle SL is based on total premium paid
            target=0,
            trailing_stop=0,
            entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        total_cost = sum(o.average_price * o.filled_quantity for o in entry_orders if o.status == OrderStatus.COMPLETE)
        managed.stop_loss = total_cost * 0.5  # exit if 50% of premium lost

        self.managed_positions[position_id] = managed
        logger.info(f"Straddle opened: {position_id} | Total cost: Rs {total_cost:.0f}")
        return managed

    def _execute_strangle(self, signal: Dict, quantity: int) -> Optional[ManagedPosition]:
        """Execute a strangle (buy OTM CE + OTM PE)."""
        # Same structure as straddle, different strikes
        return self._execute_straddle(signal, quantity)

    def close_position(
        self,
        position_id: str,
        reason: str = "manual"
    ) -> Optional[float]:
        """Close a managed position and return realized P&L."""
        if position_id not in self.managed_positions:
            logger.warning(f"Position not found: {position_id}")
            return None

        managed = self.managed_positions[position_id]
        if managed.status == "closed":
            return managed.realized_pnl

        total_pnl = 0.0

        # Cancel any pending exit orders
        for exit_order in managed.exit_orders:
            if exit_order.status in (OrderStatus.OPEN, OrderStatus.TRIGGER_PENDING):
                self.broker.cancel_order(exit_order.order_id)

        # Close all legs
        for entry_order in managed.entry_orders:
            if entry_order.status != OrderStatus.COMPLETE:
                continue

            close_side = OrderSide.SELL if entry_order.side == OrderSide.BUY else OrderSide.BUY
            close_order = Order(
                symbol=managed.symbol,
                tradingsymbol=entry_order.tradingsymbol,
                exchange=entry_order.exchange,
                side=close_side,
                order_type=OrderType.MARKET,
                product=entry_order.product,
                quantity=entry_order.filled_quantity,
                tag=f"P-EXIT-{reason[:5]}",
            )

            filled = self.broker.place_order(close_order)
            if filled.status == OrderStatus.COMPLETE:
                if entry_order.side == OrderSide.BUY:
                    pnl = (filled.average_price - entry_order.average_price) * entry_order.filled_quantity
                else:
                    pnl = (entry_order.average_price - filled.average_price) * entry_order.filled_quantity
                total_pnl += pnl

        managed.realized_pnl = round(total_pnl, 2)
        managed.status = "closed"

        # Record in risk manager (pass trade info so _open_positions gets cleaned)
        self.risk.record_trade_result(total_pnl, trade={
            "instrument": managed.tradingsymbol,
            "symbol": managed.symbol,
        })

        logger.info(
            f"Position closed: {position_id} | "
            f"P&L: Rs {total_pnl:+,.0f} | Reason: {reason}"
        )

        return total_pnl

    def close_all_positions(self, reason: str = "system") -> float:
        """Close all open positions. Returns total P&L."""
        total = 0.0
        for pid, managed in self.managed_positions.items():
            if managed.status != "closed":
                pnl = self.close_position(pid, reason)
                if pnl is not None:
                    total += pnl
        return total

    def create_trailing_state(self, position_id: str):
        """Build a TrailingState from a ManagedPosition for the PositionMonitor."""
        from prometheus.execution.position_monitor import TrailingState
        managed = self.managed_positions.get(position_id)
        if not managed or managed.status == "closed":
            return None
        return TrailingState(
            position_id=position_id,
            tradingsymbol=managed.tradingsymbol,
            symbol=managed.symbol,
            entry_premium=managed.entry_premium,
            initial_sl=managed.stop_loss,
            current_sl=managed.stop_loss,
            target=managed.target,
            direction=managed.direction,
            strategy=managed.strategy,
            entry_time=managed.entry_time,
            sl_order_id=managed.sl_order_id,
            max_bars=managed.max_bars,
            breakeven_ratio=managed.breakeven_ratio,
        )

    def _place_stop_loss(
        self,
        tradingsymbol: str,
        quantity: int,
        sl_index_price: float,
        option_type: str,
        entry_premium: float
    ) -> Optional[Order]:
        """Place a stop-loss order for an options position."""
        # For options buying: SL is based on premium, not index level
        # Estimate option premium at SL level using rough delta
        delta = 0.5  # ATM assumption
        index_sl_distance = abs(sl_index_price - entry_premium / delta)  # rough mapping
        premium_sl = max(entry_premium * 0.50, entry_premium - index_sl_distance * delta)

        sl_order = Order(
            tradingsymbol=tradingsymbol,
            exchange="NFO",
            side=OrderSide.SELL,
            order_type=OrderType.SL_M,
            product=ProductType.MIS,
            quantity=quantity,
            trigger_price=round(premium_sl, 2),
            tag="P-SL",
        )

        filled = self.broker.place_order(sl_order)
        if filled.status == OrderStatus.REJECTED:
            logger.warning(f"SL order rejected: {filled.message}")
            return None
        return filled

    def _close_order(self, order: Order):
        """Close/reverse an existing filled order."""
        close_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        close = Order(
            tradingsymbol=order.tradingsymbol,
            exchange=order.exchange,
            side=close_side,
            order_type=OrderType.MARKET,
            product=order.product,
            quantity=order.filled_quantity,
            tag="P-CANCEL",
        )
        self.broker.place_order(close)

    def _next_position_id(self) -> str:
        """Generate a unique position ID."""
        self._position_counter += 1
        return f"POS-{datetime.now().strftime('%Y%m%d')}-{self._position_counter:04d}"
