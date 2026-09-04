from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class MarketEvent:
    """
    Handles the receipt of new market data updates.

    Carries the fully parsed bar. The data handler builds this once at the CSV
    boundary; no consumer re-parses it.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class SignalEvent:
    """
    Handles the receipt of a new trading signal.
    """

    symbol: str
    timestamp: datetime
    direction: Literal["LONG", "SHORT", "EXIT"]


@dataclass(frozen=True)
class OrderEvent:
    """
    Handles the receipt of a new order to be sent to an execution system.

    `direction` is the strategy's intent (LONG/SHORT/EXIT); `side` is what the
    broker actually does with the shares (BUY/SELL). They differ on exits: an
    EXIT that closes a long is a SELL, an EXIT that covers a short is a BUY.
    Slippage and cash flows are driven by `side`, not `direction`.
    """

    symbol: str
    timestamp: datetime
    quantity: float
    direction: Literal["LONG", "SHORT", "EXIT"]
    order_type: Literal["MARKET", "LIMIT"]
    side: Literal["BUY", "SELL"]


@dataclass(frozen=True)
class FillEvent:
    """
    Encapsulates the notion of a filled order, as returned from a brokerage.

    `timestamp` is the bar the order *filled* on, which is one bar later than
    the OrderEvent's timestamp: orders are placed off bar t's close and fill at
    bar t+1's open.

    `direction` is the trade intent (LONG/SHORT/EXIT); `side` is BUY/SELL.
    """

    symbol: str
    timestamp: datetime
    quantity: float
    direction: Literal["LONG", "SHORT", "EXIT"]
    fill_price: float
    commission: float
    slippage: float
    side: Literal["BUY", "SELL"]


# Why an order died. Every one of these reaches the strategy as an
# OrderFailedEvent, so a rejected order is never a silent drop.
FailReason = Literal[
    "SIZER_DECLINED",
    "NO_POSITION",
    "INSUFFICIENT_CASH",
    "NO_PRICE",
    "END_OF_DATA",
]


@dataclass(frozen=True)
class OrderFailedEvent:
    """
    Emitted whenever an order dies instead of filling.

    An order can die at signal time (the sizer declined, or there is nothing to
    exit) or at fill time (unaffordable, no price, or the data ran out while it
    was still pending).
    """

    symbol: str
    timestamp: datetime
    direction: Literal["LONG", "SHORT", "EXIT"]
    quantity: float
    reason: FailReason


# Every event that can travel through the engine's queue. The engine dispatches
# on the concrete type, so this union is exhaustive by construction.
#
# MarketEvent is a member for historical reasons only: the data handler now
# returns bars directly to the engine rather than queueing them, so a
# MarketEvent reaching the queue is a bug (the engine's _drain raises on one).
Event = MarketEvent | SignalEvent | OrderEvent | FillEvent | OrderFailedEvent
