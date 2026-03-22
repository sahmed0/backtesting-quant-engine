"""
Strategy module for the backtesting engine.
"""

from abc import ABC, abstractmethod
from collections import deque
from queue import Queue
from typing import Dict, Literal, Optional
from event import MarketEvent, SignalEvent

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    def __init__(self, eventsQueue: Queue):
        """
        Initialises the strategy with the events queue.
        """
        self.eventsQueue = eventsQueue

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
    Emits EXIT signals when the short MA crosses below the long MA.
    """

    def __init__(self, eventsQueue: Queue, short_window: int, long_window: int):
        """
        Initialises the strategy with short and long moving average windows.
        """
        super().__init__(eventsQueue)
        self.short_window = short_window
        self.long_window = long_window
        
        # Map symbol to a deque of its most recent closing prices
        self.prices: Dict[str, deque[float]] = {}
        
        # Map symbol to its current position state ('LONG' or None)
        self.positions: Dict[str, Optional[Literal['LONG', 'SHORT', 'EXIT']]] = {}

    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Calculates and emits SMA crossover signals.
        """
        if event.type != 'MARKET':
            return
            
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
        short_ma = sum(prices_list[-self.short_window:]) / self.short_window
        long_ma = sum(prices_list) / self.long_window
        
        current_position = self.positions[symbol]
        
        if short_ma > long_ma and current_position != 'LONG':
            self.eventsQueue.put(SignalEvent(symbol, event.timestamp, 'LONG'))
            self.positions[symbol] = 'LONG'
            
        elif short_ma < long_ma and current_position == 'LONG':
            self.eventsQueue.put(SignalEvent(symbol, event.timestamp, 'EXIT'))
            self.positions[symbol] = 'EXIT'
