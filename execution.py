"""
Execution handler module for simulating order execution.
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Literal

from data import DataHandler
from event import (
    Event,
    FailReason,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderFailedEvent,
)
from portfolio import Portfolio

logger = logging.getLogger(__name__)

FillTiming = Literal["next_open", "same_close"]


class ExecutionHandler(ABC):
    """
    Abstract base class for execution handlers.
    Provides an interface for executing orders and generating fill events.
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> None:
        """
        Takes an OrderEvent and accepts it for execution. Whether that produces
        a fill immediately or on a later bar is the handler's business.
        """
        pass

    @abstractmethod
    def on_market(self, event: MarketEvent) -> None:
        """
        Called by the engine at the top of every bar, before the bar is marked
        to market and before signals are evaluated. Handlers fill whatever they
        accepted on earlier bars.
        """
        pass

    @abstractmethod
    def cancel_pending(self) -> None:
        """
        Called once by the engine after the final bar. Any order still waiting
        for a fill will never get one, and dies here.
        """
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution handler that converts order events into fill events
    with simulated slippage and commission.

    Market orders queue on the bar they are placed and fill at the *next* bar's
    open. This is what keeps the engine honest: a signal computed from bar t's
    close cannot transact at bar t's close, because at the moment the decision
    is made that price is the last thing known, not the next thing tradeable.
    """

    def __init__(
        self,
        events: deque[Event],
        data_handler: DataHandler,
        portfolio: Portfolio,
        commission_per_share: float = 0.005,
        min_commission: float = 1.00,
        slippage_pct: float = 0.0005,
        fill_timing: FillTiming = "next_open",
    ):
        """
        Initialises the handler.

        Args:
            events: The shared event queue.
            data_handler: Supplies the latest bar, used as the execution price
                in "same_close" mode only.
            portfolio: Consulted at fill time for affordability.
            commission_per_share: Dollars charged per share filled. The
                actual commission is max(commission_per_share × qty,
                min_commission) in total dollars per fill.
            min_commission: Floor on the per-fill commission in dollars.
            slippage_pct: Fraction the fill price moves against the order, e.g.
                0.0005 for 5 bps. Applied by trade *side*: a BUY pays more
                (open × (1 + s)), a SELL receives less (open × (1 − s)). EXITs
                are BUYs or SELLs like any other fill, so they carry slippage too.
            fill_timing: "next_open" (the honest default) queues orders to fill
                at the next bar's open. "same_close" fills immediately at the
                latest close, reproducing the look-ahead this engine used to
                have. It exists solely so the fill-timing impact script can
                measure what that look-ahead was worth, and must not be exposed
                in the UI.
        """
        self.events = events
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_pct = slippage_pct
        self.fill_timing: FillTiming = fill_timing

        self._pending: list[OrderEvent] = []
        # Orders that never filled because the data ran out.
        self.dropped_orders = 0

    def execute_order(self, event: OrderEvent) -> None:
        """
        Accepts an order. In "next_open" mode this only queues it; the fill
        happens on the next bar. In "same_close" mode it fills at once.
        """
        if self.fill_timing == "next_open":
            self._pending.append(event)
            return

        latest_bar = self.data_handler.get_latest_bar(event.symbol)
        if latest_bar is None:
            self._fail(event, event.timestamp, "NO_PRICE")
            return
        self._try_fill(event, latest_bar.close, latest_bar.timestamp)

    def on_market(self, event: MarketEvent) -> None:
        """
        Fills every order accepted on previous bars at this bar's open, in the
        order they arrived.
        """
        if self.fill_timing != "next_open":
            return

        pending, self._pending = self._pending, []
        for order in pending:
            if order.symbol != event.symbol:
                # Single-symbol engine, so this cannot happen; failing loudly
                # beats filling an order at another instrument's price.
                raise RuntimeError(
                    f"Pending order for {order.symbol} but bar is for {event.symbol}"
                )
            self._try_fill(order, event.open, event.timestamp)

    def cancel_pending(self) -> None:
        """
        Kills orders still waiting when the data ends. They are never filled at
        the last close and the position is never force-flattened - either would
        invent a trade the strategy could not have made.
        """
        for order in self._pending:
            self._fail(order, order.timestamp, "END_OF_DATA")
            logger.info(
                f"DROPPED {order.timestamp} {order.direction} {order.quantity} "
                f"{order.symbol} (reason: END_OF_DATA)"
            )
        self.dropped_orders += len(self._pending)
        self._pending = []

    def _try_fill(
        self, order: OrderEvent, base_price: float, timestamp: datetime
    ) -> None:
        """
        Prices an order off base_price, asks the portfolio whether it is
        affordable, and emits either a FillEvent or an OrderFailedEvent.

        Shared by both fill timings so the cost maths and the affordability
        check cannot drift apart between them.
        """
        direction = order.direction
        side = order.side

        # Apply the configured slippage by trade side:
        #   BUY  pays more:     base × (1 + slippage_pct)
        #   SELL receives less: base × (1 − slippage_pct)
        # This holds for EXITs too - an EXIT is a BUY (cover) or a SELL (close).
        if side == "BUY":
            fill_price = base_price * (1 + self.slippage_pct)
        else:  # SELL
            fill_price = base_price * (1 - self.slippage_pct)

        # Commission is total dollars per fill; slippage is total dollars, for
        # reporting only - it is already embedded in fill_price above.
        commission = max(
            self.commission_per_share * order.quantity, self.min_commission
        )
        slippage_value = abs(fill_price - base_price) * order.quantity

        can_fill, reason = self.portfolio.can_execute(
            order, fill_price, commission, slippage_value
        )
        if not can_fill:
            assert reason is not None
            self._fail(order, timestamp, reason)
            logger.info(
                f"REJECTED {timestamp} {direction} {order.quantity} {order.symbol} "
                f"@ {fill_price:.4f} (reason: {reason})"
            )
            return

        fill_event = FillEvent(
            symbol=order.symbol,
            timestamp=timestamp,
            quantity=order.quantity,
            direction=direction,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_value,
            side=side,
        )

        logger.info(
            f"FILLED {fill_event.timestamp} {fill_event.direction} {fill_event.quantity} {fill_event.symbol} "
            f"@ {fill_event.fill_price:.4f} (comm: {fill_event.commission}, slippage: {fill_event.slippage:.4f})"
        )

        self.events.append(fill_event)

    def _fail(self, order: OrderEvent, timestamp: datetime, reason: FailReason) -> None:
        """
        Queues an OrderFailedEvent for a dead order.
        """
        self.events.append(
            OrderFailedEvent(
                symbol=order.symbol,
                timestamp=timestamp,
                direction=order.direction,
                quantity=order.quantity,
                reason=reason,
            )
        )
