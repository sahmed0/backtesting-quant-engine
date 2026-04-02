"""
Execution handler module for simulating order execution.
"""

from abc import ABC, abstractmethod
from queue import Queue
from datetime import datetime, timezone
import logging
from typing import Any

from event import OrderEvent, FillEvent
from data import DataHandler

# Configure basic logging for the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExecutionHandler(ABC):
    """
    Abstract base class for execution handlers.
    Provides an interface for executing orders and generating fill events.
    """

    @abstractmethod
    def executeOrder(self, event: OrderEvent) -> None:
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

    def __init__(self, eventsQueue: Queue, dataHandler: DataHandler, fixed_commission: float = 0.001):
        """
        Initialises the handler, saving the events queue and data handler.
        """
        self.eventsQueue = eventsQueue
        self.dataHandler = dataHandler
        self.fixed_commission = fixed_commission

    def executeOrder(self, event: OrderEvent) -> None:
        """
        Converts OrderEvent to FillEvent.
        """
        if event.type != 'ORDER':
            return

        latest_bar = self.dataHandler.getLatestBar(event.symbol)
        
        # If no bar data is available, we cannot execute the order realistically in this simulation.
        if not latest_bar or 'close' not in latest_bar:
            logger.warning(f"No price data available for {event.symbol}. Cannot execute order.")
            return

        base_price = float(latest_bar['close'])
        direction = event.direction

        # Calculate slippage (0.05%)
        # LONG: pay more (+0.05%)
        # SHORT: receive less (-0.05%)
        # EXIT: For simplicity, assume worst-case execution if we don't know the exact side
        # In a real system, EXIT would check current position to determine if it's a buy or sell.
        slippage_pct = 0.0005
        
        if direction == 'LONG':
            fill_price = base_price * (1 + slippage_pct)
        elif direction == 'SHORT':
            fill_price = base_price * (1 - slippage_pct)
        elif direction == 'EXIT':
            # We charge the slippage amount directly against the base price.
            # To simulate a penalty, we can just use the worse price based on the side of the trade,
            # but without position info, we'll arbitrarily charge 0.05% against the expected return.
            # Here we'll just log it and apply a neutral price, as EXIT direction might be handled by the portfolio.
            # Wait, actually let's just make it neutral if we don't know, or + slippage to be safe.
            fill_price = base_price # or perhaps apply a flat penalty. We'll use base_price for now.
        else:
            fill_price = base_price
            
        slippage_value = abs(fill_price - base_price)

        # Create FillEvent
        fill_event = FillEvent(
            symbol=event.symbol,
            timestamp=datetime.now(timezone.utc),
            quantity=event.quantity,
            direction=event.direction,
            fillPrice=fill_price,
            commission=self.fixed_commission,
            slippage=slippage_value
        )

        # Log the fill
        logger.info(
            f"FILLED {fill_event.direction} {fill_event.quantity} {fill_event.symbol} "
            f"@ {fill_event.fillPrice:.4f} (comm: {fill_event.commission}, slippage: {fill_event.slippage:.4f})"
        )

        # Put the FillEvent onto the queue
        self.eventsQueue.put(fill_event)
