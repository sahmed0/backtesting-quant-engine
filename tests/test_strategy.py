import unittest
from queue import Queue
from datetime import datetime
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from event import MarketEvent, SignalEvent
from strategy import SimpleMovingAverageStrategy

class TestSimpleMovingAverageStrategy(unittest.TestCase):
    def setUp(self):
        self.eventsQueue = Queue()
        self.strategy = SimpleMovingAverageStrategy(self.eventsQueue, short_window=2, long_window=4)
        self.symbol = "AAPL"
        
    def _create_market_event(self, price: float) -> MarketEvent:
        return MarketEvent(
            symbol=self.symbol,
            timestamp=datetime.now(),
            close=price,
            high=price,
            low=price,
            volume=100
        )

    def test_warm_up_period(self):
        # Insert 3 events, less than long_window (4)
        for price in [10.0, 11.0, 12.0]:
            self.strategy.calculate_signals(self._create_market_event(price))
            
        self.assertTrue(self.eventsQueue.empty(), "Should not emit signals during warm-up period")

    def test_crossover_long(self):
        # Prices: 10, 10, 10, 10 -> MAs: short=10, long=10
        for price in [10.0, 10.0, 10.0, 10.0]:
            self.strategy.calculate_signals(self._create_market_event(price))
        self.assertTrue(self.eventsQueue.empty())
            
        # Price: 12 -> prices: [10, 10, 10, 12]
        # short_ma (last 2) = (10 + 12)/2 = 11
        # long_ma (last 4) = (10 + 10 + 10 + 12)/4 = 10.5
        # 11 > 10.5, emit LONG
        self.strategy.calculate_signals(self._create_market_event(12.0))
        
        self.assertFalse(self.eventsQueue.empty())
        event = self.eventsQueue.get()
        self.assertIsInstance(event, SignalEvent)
        self.assertEqual(event.direction, 'LONG')
        self.assertEqual(event.symbol, self.symbol)
        
    def test_crossover_exit(self):
        # Trigger LONG first
        for price in [10.0, 10.0, 10.0, 10.0, 12.0]:
            self.strategy.calculate_signals(self._create_market_event(price))
            
        # Empty the queue
        while not self.eventsQueue.empty():
            self.eventsQueue.get()
            
        # Prices are now: [10, 10, 10, 12] (maxlen 4) -> [10, 10, 12, 8]
        # Price: 8 -> short_ma = (12 + 8)/2 = 10
        # long_ma = (10 + 10 + 12 + 8)/4 = 10
        # Price: 6 -> [10, 12, 8, 6] -> short_ma = 7, long_ma = 9
        # 7 < 9, emit EXIT
        self.strategy.calculate_signals(self._create_market_event(8.0))
        self.assertTrue(self.eventsQueue.empty()) # MAs are equal, no change
        
        self.strategy.calculate_signals(self._create_market_event(6.0))
        self.assertFalse(self.eventsQueue.empty())
        event = self.eventsQueue.get()
        self.assertEqual(event.direction, 'EXIT')

if __name__ == '__main__':
    unittest.main()
