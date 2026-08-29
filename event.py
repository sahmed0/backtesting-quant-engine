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
    """

    symbol: str
    timestamp: datetime
    quantity: float
    direction: Literal["LONG", "SHORT", "EXIT"]
    order_type: Literal["MARKET", "LIMIT"]


@dataclass(frozen=True)
class FillEvent:
    """
    Encapsulates the notion of a filled order, as returned from a brokerage.
    """

    symbol: str
    timestamp: datetime
    quantity: float
    direction: Literal["LONG", "SHORT", "EXIT"]
    fill_price: float
    commission: float
    slippage: float


# Every event that can travel through the engine's queue.
Event = MarketEvent | SignalEvent | OrderEvent | FillEvent
