from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class MarketEvent:
    """
    Handles the receipt of new market data updates.
    """
    symbol: str
    timestamp: datetime
    close: float
    high: float
    low: float
    volume: float
    type: str = 'MARKET'

@dataclass
class SignalEvent:
    """
    Handles the receipt of a new trading signal.
    """
    symbol: str
    timestamp: datetime
    direction: Literal['LONG', 'SHORT', 'EXIT']
    type: str = 'SIGNAL'

@dataclass
class OrderEvent:
    """
    Handles the receipt of a new order to be sent to an execution system.
    """
    symbol: str
    timestamp: datetime
    quantity: float
    direction: Literal['LONG', 'SHORT', 'EXIT']
    orderType: Literal['MARKET', 'LIMIT']
    type: str = 'ORDER'

@dataclass
class FillEvent:
    """
    Encapsulates the notion of a filled order, as returned from a brokerage.
    """
    symbol: str
    timestamp: datetime
    quantity: float
    direction: Literal['LONG', 'SHORT', 'EXIT']
    fillPrice: float
    commission: float
    slippage: float
    type: str = 'FILL'
