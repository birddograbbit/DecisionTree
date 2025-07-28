"""
Order Management System for sophisticated trading strategies.

This module handles complex order types including limit orders, trailing stops,
and multi-bar order persistence used by momentum strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order with all necessary attributes.
    """
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    trail_amount: Optional[float] = None  # For trailing stops
    trail_percent: Optional[float] = None  # For trailing stops
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    persistence_bars: int = 1
    bars_active: int = 0
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    parent_order_id: Optional[str] = None  # For stop loss/take profit orders
    metadata: Dict = field(default_factory=dict)


class OrderManagementSystem:
    """
    Manages order lifecycle including creation, execution, and risk management.
    
    This system handles:
    - Limit order execution with realistic fill logic
    - Multi-bar order persistence
    - Trailing stop management
    - Position tracking
    - Risk management (stop loss, take profit)
    """
    
    def __init__(self, allow_same_bar_exit: bool = False):
        """
        Initialize the order management system.
        
        Parameters:
        -----------
        allow_same_bar_exit : bool
            Whether to allow positions to be entered and exited on the same bar
        """
        self.allow_same_bar_exit = allow_same_bar_exit
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.order_counter = 0
        self.current_bar_entries: set = set()  # Track entries on current bar
        
    def create_order(self, symbol: str, side: str, quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_amount: Optional[float] = None,
                    trail_percent: Optional[float] = None,
                    persistence_bars: int = 1,
                    parent_order_id: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> Order:
        """
        Create a new order.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        side : str
            'buy' or 'sell'
        quantity : float
            Order quantity
        order_type : OrderType
            Type of order
        price : float, optional
            Limit price
        stop_price : float, optional
            Stop price
        trail_amount : float, optional
            Trailing stop amount
        trail_percent : float, optional
            Trailing stop percentage
        persistence_bars : int
            Number of bars to keep order active
        parent_order_id : str, optional
            ID of parent order (for bracket orders)
        metadata : dict, optional
            Additional order metadata
            
        Returns:
        --------
        Order
            Created order object
        """
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            persistence_bars=persistence_bars,
            parent_order_id=parent_order_id,
            metadata=metadata or {}
        )
        
        self.active_orders[order_id] = order
        logger.debug(f"Created {order_type.value} order {order_id}: "
                    f"{side} {quantity} {symbol} @ {price}")
        
        return order
    
    def process_bar(self, symbol: str, bar_data: pd.Series) -> List[Order]:
        """
        Process orders for a new bar of data.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        bar_data : pd.Series
            OHLCV data for the bar
            
        Returns:
        --------
        List[Order]
            List of filled orders
        """
        filled_orders = []
        expired_orders = []
        
        # Reset current bar entries
        self.current_bar_entries.clear()
        
        # Process each active order
        for order_id, order in self.active_orders.items():
            if order.symbol != symbol:
                continue
            
            # Increment bars active
            order.bars_active += 1
            
            # Check for expiration
            if order.bars_active > order.persistence_bars:
                order.status = OrderStatus.EXPIRED
                expired_orders.append(order_id)
                logger.debug(f"Order {order_id} expired after {order.bars_active} bars")
                continue
            
            # Check for fill based on order type
            if self._check_order_fill(order, bar_data):
                filled_orders.append(order)
                self.current_bar_entries.add(symbol)
        
        # Remove expired and filled orders
        for order_id in expired_orders:
            del self.active_orders[order_id]
        
        for order in filled_orders:
            self._execute_order(order, bar_data)
        
        # Update trailing stops
        self._update_trailing_stops(symbol, bar_data)
        
        return filled_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Parameters:
        -----------
        order_id : str
            Order ID to cancel
            
        Returns:
        --------
        bool
            True if order was cancelled
        """
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            logger.debug(f"Cancelled order {order_id}")
            return True
        return False
    
    def create_bracket_order(self, symbol: str, side: str, quantity: float,
                           entry_price: Optional[float] = None,
                           stop_loss_price: Optional[float] = None,
                           take_profit_price: Optional[float] = None,
                           use_trailing_stop: bool = False,
                           trail_amount: Optional[float] = None) -> Tuple[Order, Order, Order]:
        """
        Create a bracket order (entry + stop loss + take profit).
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        side : str
            'buy' or 'sell'
        quantity : float
            Order quantity
        entry_price : float, optional
            Entry limit price (None for market order)
        stop_loss_price : float, optional
            Stop loss price
        take_profit_price : float, optional
            Take profit price
        use_trailing_stop : bool
            Whether to use trailing stop
        trail_amount : float, optional
            Trailing stop amount
            
        Returns:
        --------
        Tuple[Order, Order, Order]
            (entry_order, stop_loss_order, take_profit_order)
        """
        # Create entry order
        entry_order = self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT if entry_price else OrderType.MARKET,
            price=entry_price
        )
        
        # Create stop loss order
        stop_order = None
        if stop_loss_price:
            stop_side = 'sell' if side == 'buy' else 'buy'
            stop_order = self.create_order(
                symbol=symbol,
                side=stop_side,
                quantity=quantity,
                order_type=OrderType.TRAILING_STOP if use_trailing_stop else OrderType.STOP,
                stop_price=stop_loss_price,
                trail_amount=trail_amount,
                parent_order_id=entry_order.order_id
            )
        
        # Create take profit order
        tp_order = None
        if take_profit_price:
            tp_side = 'sell' if side == 'buy' else 'buy'
            tp_order = self.create_order(
                symbol=symbol,
                side=tp_side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=take_profit_price,
                parent_order_id=entry_order.order_id
            )
        
        return entry_order, stop_order, tp_order
    
    def get_position(self, symbol: str) -> float:
        """
        Get current position for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
            
        Returns:
        --------
        float
            Current position (positive for long, negative for short)
        """
        return self.positions.get(symbol, 0.0)
    
    def _check_order_fill(self, order: Order, bar_data: pd.Series) -> bool:
        """
        Check if an order should be filled based on bar data.
        
        Parameters:
        -----------
        order : Order
            Order to check
        bar_data : pd.Series
            OHLCV data
            
        Returns:
        --------
        bool
            True if order should be filled
        """
        # Check same-bar exit restriction
        if not self.allow_same_bar_exit and order.symbol in self.current_bar_entries:
            if self.get_position(order.symbol) != 0:
                return False
        
        # Market orders always fill
        if order.order_type == OrderType.MARKET:
            return True
        
        # Limit orders
        if order.order_type == OrderType.LIMIT:
            if order.side == 'buy':
                # Buy limit fills if low <= limit price
                return bar_data['low'] <= order.price
            else:
                # Sell limit fills if high >= limit price
                return bar_data['high'] >= order.price
        
        # Stop orders
        if order.order_type == OrderType.STOP:
            if order.side == 'buy':
                # Buy stop fills if high >= stop price
                return bar_data['high'] >= order.stop_price
            else:
                # Sell stop fills if low <= stop price
                return bar_data['low'] <= order.stop_price
        
        return False
    
    def _execute_order(self, order: Order, bar_data: pd.Series) -> None:
        """
        Execute a filled order.
        
        Parameters:
        -----------
        order : Order
            Order to execute
        bar_data : pd.Series
            OHLCV data
        """
        # Determine fill price
        if order.order_type == OrderType.MARKET:
            fill_price = bar_data['open']
        elif order.order_type == OrderType.LIMIT:
            fill_price = order.price
        elif order.order_type == OrderType.STOP:
            # Stop orders fill at stop price or worse
            if order.side == 'buy':
                fill_price = max(order.stop_price, bar_data['open'])
            else:
                fill_price = min(order.stop_price, bar_data['open'])
        else:
            fill_price = bar_data['close']
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        
        # Update position
        if order.side == 'buy':
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        else:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - order.quantity
        
        # Move to filled orders
        del self.active_orders[order.order_id]
        self.filled_orders.append(order)
        
        logger.info(f"Filled {order.order_type.value} order {order.order_id}: "
                   f"{order.side} {order.quantity} {order.symbol} @ {fill_price:.2f}")
        
        # Cancel related orders if this was a bracket order
        if order.parent_order_id:
            self._cancel_related_orders(order.parent_order_id)
    
    def _update_trailing_stops(self, symbol: str, bar_data: pd.Series) -> None:
        """
        Update trailing stop orders based on price movement.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        bar_data : pd.Series
            OHLCV data
        """
        for order in self.active_orders.values():
            if (order.symbol == symbol and 
                order.order_type == OrderType.TRAILING_STOP):
                
                position = self.get_position(symbol)
                
                if position > 0 and order.side == 'sell':
                    # Long position - update stop if price increased
                    if order.trail_amount:
                        new_stop = bar_data['high'] - order.trail_amount
                    else:
                        new_stop = bar_data['high'] * (1 - order.trail_percent)
                    
                    if new_stop > order.stop_price:
                        order.stop_price = new_stop
                        logger.debug(f"Updated trailing stop {order.order_id} to {new_stop:.2f}")
                
                elif position < 0 and order.side == 'buy':
                    # Short position - update stop if price decreased
                    if order.trail_amount:
                        new_stop = bar_data['low'] + order.trail_amount
                    else:
                        new_stop = bar_data['low'] * (1 + order.trail_percent)
                    
                    if new_stop < order.stop_price:
                        order.stop_price = new_stop
                        logger.debug(f"Updated trailing stop {order.order_id} to {new_stop:.2f}")
    
    def _cancel_related_orders(self, parent_order_id: str) -> None:
        """
        Cancel all orders related to a parent order.
        
        Parameters:
        -----------
        parent_order_id : str
            Parent order ID
        """
        to_cancel = []
        for order_id, order in self.active_orders.items():
            if order.parent_order_id == parent_order_id:
                to_cancel.append(order_id)
        
        for order_id in to_cancel:
            self.cancel_order(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get list of active orders.
        
        Parameters:
        -----------
        symbol : str, optional
            Filter by symbol
            
        Returns:
        --------
        List[Order]
            List of active orders
        """
        if symbol:
            return [o for o in self.active_orders.values() if o.symbol == symbol]
        return list(self.active_orders.values())
    
    def get_order_summary(self) -> Dict[str, any]:
        """
        Get summary of order management system state.
        
        Returns:
        --------
        Dict[str, any]
            Summary statistics
        """
        return {
            'active_orders': len(self.active_orders),
            'filled_orders': len(self.filled_orders),
            'positions': dict(self.positions),
            'order_types': {
                ot.value: sum(1 for o in self.active_orders.values() 
                            if o.order_type == ot)
                for ot in OrderType
            }
        }