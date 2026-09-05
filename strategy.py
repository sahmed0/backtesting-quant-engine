"""
Strategy module for the backtesting engine.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Literal

from event import Event, FillEvent, MarketEvent, OrderFailedEvent, SignalEvent


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    A strategy tracks two distinct kinds of position state, mirroring a real
    order management system's lifecycle (new -> acked -> filled / rejected):

      - ``intent``:   what the strategy has *asked* for. Set the moment a signal
                      is emitted, before any fill exists. Trading decisions key
                      off ``intent`` so that a strategy does not re-emit the same
                      signal during the one-bar gap between placing an order and
                      it filling at the next open.
      - ``position``: fill-truth. Updated only when a fill actually lands
                      (``on_fill``), so it reflects what the broker really did.

    The lifecycle for a symbol: flat (``intent``/``position`` both ``None``) ->
    a signal sets ``intent`` -> the fill at the next open confirms ``position``.
    If instead the order is rejected, ``on_order_failed`` reverts ``intent`` back
    to fill-truth so the lost signal can fire again later. Without this, a
    rejected order would leave the strategy believing it holds a position it
    never got, suppressing every subsequent signal for that symbol.
    """

    def __init__(self, events: deque[Event] | None = None, allow_short: bool = False):
        """
        Initialises the strategy with the events queue.

        Args:
            events: The shared event queue. Signals are appended to it for the
                engine to route. When omitted a private queue is created, which
                lets a strategy be exercised directly (e.g. in unit tests)
                without an engine; callers then read the signals off it.
            allow_short: When False (the default) the strategy is long-only and
                never emits SHORT signals. When True the strategy may open short
                positions.
        """
        self.events: deque[Event] = events if events is not None else deque()
        self.allow_short = allow_short

        # What we've asked for (drives signal suppression) vs. fill-truth.
        self.intent: dict[str, Literal["LONG", "SHORT"] | None] = {}
        self.position: dict[str, Literal["LONG", "SHORT"] | None] = {}

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Calculates trading signals based on the provided market event.
        """
        pass

    def on_fill(self, event: FillEvent) -> None:
        """
        Records fill-truth when an order actually fills.

        Only ``position`` is updated here; ``intent`` is deliberately left
        untouched. During a flatten-before-reverse pair (an EXIT immediately
        followed by an opposite entry, both pending at once), the EXIT fill must
        not clobber the still-pending entry intent.
        """
        if event.direction == "EXIT":
            self.position[event.symbol] = None
        else:  # LONG / SHORT - mypy narrows the Literal in this branch
            self.position[event.symbol] = event.direction

    def on_order_failed(self, event: OrderFailedEvent) -> None:
        """
        Reverts intent to fill-truth when an order dies.

        The order the strategy asked for never happened, so ``intent`` is rolled
        back to whatever the strategy actually holds. The signal that produced
        the failed order is thus free to fire again on a later qualifying bar.
        """
        self.intent[event.symbol] = self.position.get(event.symbol)


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
        events: deque[Event],
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

    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Calculates and emits SMA crossover signals.

        Decisions key off ``intent`` (what we've already asked for), not
        ``position`` (fill-truth): between placing an order and its fill at the
        next open the strategy must not re-emit the same crossover signal.
        """
        symbol = event.symbol
        close_price = event.close

        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=self.long_window)
            self.intent[symbol] = None

        self.prices[symbol].append(close_price)

        # Wait for the warm-up period to complete
        if len(self.prices[symbol]) < self.long_window:
            return

        prices_list = list(self.prices[symbol])
        short_ma = sum(prices_list[-self.short_window :]) / self.short_window
        long_ma = sum(prices_list) / self.long_window

        current_position = self.intent.get(symbol)

        if short_ma > long_ma and current_position != "LONG":
            # Cover any open short before going long.
            if current_position == "SHORT":
                self.events.append(SignalEvent(symbol, event.timestamp, "EXIT"))
            self.events.append(SignalEvent(symbol, event.timestamp, "LONG"))
            self.intent[symbol] = "LONG"

        elif short_ma < long_ma:
            if self.allow_short and current_position != "SHORT":
                # Close any open long before going short.
                if current_position == "LONG":
                    self.events.append(SignalEvent(symbol, event.timestamp, "EXIT"))
                self.events.append(SignalEvent(symbol, event.timestamp, "SHORT"))
                self.intent[symbol] = "SHORT"
            elif not self.allow_short and current_position == "LONG":
                # Long-only: simply flatten the existing long.
                self.events.append(SignalEvent(symbol, event.timestamp, "EXIT"))
                self.intent[symbol] = None
