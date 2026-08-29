"""
Execution handler module for simulating order execution.
"""

import logging
from abc import ABC, abstractmethod
from collections import deque

from data import DataHandler
from event import Event, FillEvent, OrderEvent

logger = logging.getLogger(__name__)


class ExecutionHandler(ABC):
    """
    Abstract base class for execution handlers.
    Provides an interface for executing orders and generating fill events.
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> None:
        """
        Takes an OrderEvent and executes it, producing a FillEvent
        that gets placed onto the events queue.
        """
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution handler that converts all order events into
    fill events with simulated slippage and commission.
    """

    def __init__(
        self,
        events: deque[Event],
        data_handler: DataHandler,
        fixed_commission: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        """
        Initialises the handler, saving the events queue and data handler.

        Args:
            events: The shared event queue.
            data_handler: Supplies the latest bar used as the execution price.
            fixed_commission: Flat commission charged per fill.
            slippage_pct: Fraction the fill price moves against the order, e.g.
                0.0005 for 5 bps. LONG fills pay more and SHORT fills receive
                less; EXIT fills are modelled without slippage.
        """
        self.events = events
        self.data_handler = data_handler
        self.fixed_commission = fixed_commission
        self.slippage_pct = slippage_pct

    def execute_order(self, event: OrderEvent) -> None:
        """
        Converts OrderEvent to FillEvent.
        """
        latest_bar = self.data_handler.get_latest_bar(event.symbol)

        # If no bar data is available, we cannot execute the order in this simulation.
        if latest_bar is None:
            logger.warning(
                f"No price data available for {event.symbol}. Cannot execute order."
            )
            return

        base_price = latest_bar.close
        direction = event.direction

        # Apply the configured slippage to the base price.
        # LONG: pay more (+slippage)
        # SHORT: receive less (-slippage)
        # EXIT: For simplicity, assume worst-case execution if we don't know the exact side
        # In a real system, EXIT would check current position to determine if it's a buy or sell.
        slippage_pct = self.slippage_pct

        if direction == "LONG":
            fill_price = base_price * (1 + slippage_pct)
        elif direction == "SHORT":
            fill_price = base_price * (1 - slippage_pct)
        elif direction == "EXIT":
            fill_price = (
                base_price  # No slippage for EXIT orders in this simplified model
            )
        else:
            fill_price = base_price

        slippage_value = abs(fill_price - base_price)

        # Create FillEvent
        fill_event = FillEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            quantity=event.quantity,
            direction=event.direction,
            fill_price=fill_price,
            commission=self.fixed_commission,
            slippage=slippage_value,
        )

        # Log the fill
        logger.info(
            f"FILLED {fill_event.timestamp} {fill_event.direction} {fill_event.quantity} {fill_event.symbol} "
            f"@ {fill_event.fill_price:.4f} (comm: {fill_event.commission}, slippage: {fill_event.slippage:.4f})"
        )

        # Put the FillEvent onto the queue
        self.events.append(fill_event)
