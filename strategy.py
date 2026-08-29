"""
Strategy module for the backtesting engine.
"""

from abc import ABC, abstractmethod
from collections import deque
from queue import Queue
from typing import Literal

from event import MarketEvent, SignalEvent


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    def __init__(self, events: Queue, allow_short: bool = False):
        """
        Initialises the strategy with the events queue.

        Args:
            events: The shared event queue.
            allow_short: When False (the default) the strategy is long-only and
                never emits SHORT signals. When True the strategy may open short
                positions.
        """
        self.events = events
        self.allow_short = allow_short

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Calculates trading signals based on the provided market event.
        """
        pass


class SimpleMovingAverageStrategy(Strategy):
    """
    A simple moving average crossover strategy.
    Emits LONG signals when the short MA crosses above the long MA.
    Emits SHORT signals when the short MA crosses below the long MA.
    On a reversal the existing position is flattened with an EXIT before the
    opposite position is opened, since the portfolio sizes each order at a
    fixed quantity and cannot flip a position in a single order.
    """

    def __init__(
        self,
        events: Queue,
        short_window: int,
        long_window: int,
        allow_short: bool = False,
    ):
        """
        Initialises the strategy with short and long moving average windows.
        """
        super().__init__(events, allow_short)
        self.short_window = short_window
        self.long_window = long_window

        # Maps symbol to a deque of its most recent closing prices
        self.prices: dict[str, deque[float]] = {}

        # Maps symbol to its current position state ('LONG', 'SHORT', or None
        # when flat)
        self.positions: dict[str, Literal["LONG", "SHORT"] | None] = {}

    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Calculates and emits SMA crossover signals.
        """
        symbol = event.symbol
        close_price = event.close

        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=self.long_window)
            self.positions[symbol] = None

        self.prices[symbol].append(close_price)

        # Wait for the warm-up period to complete
        if len(self.prices[symbol]) < self.long_window:
            return

        prices_list = list(self.prices[symbol])
        short_ma = sum(prices_list[-self.short_window :]) / self.short_window
        long_ma = sum(prices_list) / self.long_window

        current_position = self.positions[symbol]

        if short_ma > long_ma and current_position != "LONG":
            # Cover any open short before going long.
            if current_position == "SHORT":
                self.events.put(SignalEvent(symbol, event.timestamp, "EXIT"))
            self.events.put(SignalEvent(symbol, event.timestamp, "LONG"))
            self.positions[symbol] = "LONG"

        elif short_ma < long_ma:
            if self.allow_short and current_position != "SHORT":
                # Close any open long before going short.
                if current_position == "LONG":
                    self.events.put(SignalEvent(symbol, event.timestamp, "EXIT"))
                self.events.put(SignalEvent(symbol, event.timestamp, "SHORT"))
                self.positions[symbol] = "SHORT"
            elif not self.allow_short and current_position == "LONG":
                # Long-only: simply flatten the existing long.
                self.events.put(SignalEvent(symbol, event.timestamp, "EXIT"))
                self.positions[symbol] = None
