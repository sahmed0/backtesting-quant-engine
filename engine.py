import logging
from collections import deque

from data import DataHandler
from event import (
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderFailedEvent,
    SignalEvent,
)
from execution import ExecutionHandler
from portfolio import Portfolio
from strategy import Strategy

logger = logging.getLogger(__name__)


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

    async def run(self) -> None:
        """
        Executes the backtest logic.

        The per-bar sequence is load-bearing and must not be reordered:

          1. Fill orders queued on the previous bar, at THIS bar's open.
          2. Mark to market at this bar's close, with those fills reflected.
          3. Evaluate signals off this bar's close; orders queue for next bar.

        Because fills always happen at the open, before signals are evaluated at
        the close, a strategy can never transact at a price it used to decide.
        """
        while True:
            market = self.data_handler.update_bars()
            if market is None:
                break

            self.execution_handler.on_market(market)
            self._drain()

            self.portfolio.update_timeindex(market)

            self.strategy.calculate_signals(market)
            self._drain()

        # The data ended. Anything still pending never gets a fill.
        self.execution_handler.cancel_pending()
        self._drain()

    def _drain(self) -> None:
        """
        Dispatches every queued event until the queue is empty.

        Handlers may queue further events while draining (a signal begets an
        order begets a fill), which is why this drains to empty rather than
        iterating a snapshot.
        """
        while self.events:
            event = self.events.popleft()

            match event:
                case SignalEvent():
                    self.portfolio.update_signal(event)
                case OrderEvent():
                    self.execution_handler.execute_order(event)
                case FillEvent():
                    self.portfolio.update_fill(event)
                case OrderFailedEvent():
                    logger.info(
                        f"ORDER FAILED {event.timestamp} {event.direction} "
                        f"{event.quantity} {event.symbol} (reason: {event.reason})"
                    )
                case MarketEvent():
                    # Market data reaches the engine as update_bars()'s return
                    # value, never through the queue. One arriving here means a
                    # producer regressed to queueing bars.
                    raise RuntimeError(
                        "MarketEvent must not be queued; update_bars() returns it"
                    )
