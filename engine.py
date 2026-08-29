from collections import deque

from data import DataHandler
from event import Event, FillEvent, MarketEvent, OrderEvent, SignalEvent
from execution import ExecutionHandler
from portfolio import Portfolio
from strategy import Strategy


class Backtest:
    """
    Encapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
        events: deque[Event],
    ):
        """
        Initialises the backtest.

        Args:
            data_handler: The MarketDataHandler instance.
            strategy: The Strategy object.
            portfolio: The Portfolio object.
            execution_handler: The ExecutionHandler object.
            events: The Event Queue object.
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = events

    async def run(self):
        """
        Executes the backtest logic.
        """
        while self.data_handler.continue_backtest:
            self.data_handler.update_bars()

            while self.events:
                event = self.events.popleft()

                match event:
                    case MarketEvent():
                        self.strategy.calculate_signals(event)
                        self.portfolio.update_timeindex(event)
                    case SignalEvent():
                        self.portfolio.update_signal(event)
                    case OrderEvent():
                        self.execution_handler.execute_order(event)
                    case FillEvent():
                        self.portfolio.update_fill(event)
