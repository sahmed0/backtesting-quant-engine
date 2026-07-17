import queue
import time
from typing import Any


class Backtest:
    """
    Encapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(
        self,
        data_handler: Any,
        strategy: Any,
        portfolio: Any,
        execution_handler: Any,
        events: queue.Queue,
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

            while True:
                try:
                    event = self.events.get(block=False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == "MARKET":
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)
                        elif event.type == "SIGNAL":
                            self.portfolio.update_signal(event)
                        elif event.type == "ORDER":
                            self.execution_handler.execute_order(event)
                        elif event.type == "FILL":
                            self.portfolio.update_fill(event)

            time.sleep(0)
