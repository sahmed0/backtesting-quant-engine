import unittest
from collections import deque
from datetime import UTC, datetime

from data import DataHandler
from event import FillEvent, MarketEvent, OrderEvent
from execution import SimulatedExecutionHandler


class MockDataHandler(DataHandler):
    """
    Mock data handler for testing purposes.
    """

    def __init__(self, price: float):
        self.price = price

    def get_latest_bar(self, symbol: str) -> MarketEvent | None:
        return MarketEvent(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            open=self.price,
            high=self.price,
            low=self.price,
            close=self.price,
            volume=0.0,
        )

    def update_bars(self) -> None:
        pass


class TestSimulatedExecutionHandler(unittest.TestCase):
    def setUp(self):
        self.events = deque()
        # Mock price of 100.0 for easy calculation
        self.data_handler = MockDataHandler(price=100.0)
        self.execution_handler = SimulatedExecutionHandler(
            self.events, self.data_handler, fixed_commission=0.001
        )

    def test_execute_order_long(self):
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(UTC),
            quantity=10,
            direction="LONG",
            order_type="MARKET",
        )

        self.execution_handler.execute_order(order)

        # Check that one event was added to the queue
        self.assertEqual(len(self.events), 1)

        fill_event = self.events.popleft()
        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.symbol, "AAPL")
        self.assertEqual(fill_event.direction, "LONG")
        self.assertEqual(fill_event.quantity, 10)
        self.assertEqual(fill_event.commission, 0.001)

        # 0.05% slippage on 100.0 is 0.05
        # LONG trades should fill at a higher price (worse)
        self.assertAlmostEqual(fill_event.fill_price, 100.05, places=4)
        self.assertAlmostEqual(fill_event.slippage, 0.05, places=4)

    def test_execute_order_short(self):
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(UTC),
            quantity=10,
            direction="SHORT",
            order_type="MARKET",
        )

        self.execution_handler.execute_order(order)

        fill_event = self.events.popleft()

        # 0.05% slippage on 100.0 is 0.05
        # SHORT trades should fill at a lower price (worse)
        self.assertAlmostEqual(fill_event.fill_price, 99.95, places=4)
        self.assertAlmostEqual(fill_event.slippage, 0.05, places=4)

    def test_execute_order_exit(self):
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(UTC),
            quantity=10,
            direction="EXIT",
            order_type="MARKET",
        )

        self.execution_handler.execute_order(order)

        fill_event = self.events.popleft()

        # EXIT trades default to neutral execution in this simulation
        self.assertAlmostEqual(fill_event.fill_price, 100.0, places=4)
        self.assertAlmostEqual(fill_event.slippage, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
