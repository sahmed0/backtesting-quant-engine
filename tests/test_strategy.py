import unittest
from collections import deque
from datetime import datetime

from event import (
    Event,
    FillEvent,
    MarketEvent,
    OrderFailedEvent,
    SignalEvent,
)
from strategy import SimpleMovingAverageStrategy


class TestSimpleMovingAverageStrategy(unittest.TestCase):
    def setUp(self):
        self.events: deque[Event] = deque()
        # Long-only by default; allow_short defaults to False.
        self.strategy = SimpleMovingAverageStrategy(
            self.events, short_window=2, long_window=4
        )
        # A separate instance with shorting enabled for the long/short tests.
        self.short_strategy = SimpleMovingAverageStrategy(
            self.events, short_window=2, long_window=4, allow_short=True
        )
        self.symbol = "AAPL"

    def _create_market_event(self, price: float) -> MarketEvent:
        return MarketEvent(
            symbol=self.symbol,
            timestamp=datetime.now(),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=100,
        )

    def test_warm_up_period(self):
        # Insert 3 events, less than long_window (4)
        for price in [10.0, 11.0, 12.0]:
            self.strategy.calculate_signals(self._create_market_event(price))

        self.assertEqual(
            len(self.events), 0, "Should not emit signals during warm-up period"
        )

    def test_crossover_long(self):
        # Prices: 10, 10, 10, 10 -> MAs: short=10, long=10
        for price in [10.0, 10.0, 10.0, 10.0]:
            self.strategy.calculate_signals(self._create_market_event(price))
        self.assertEqual(len(self.events), 0)

        # Price: 12 -> prices: [10, 10, 10, 12]
        # short_ma (last 2) = (10 + 12)/2 = 11
        # long_ma (last 4) = (10 + 10 + 10 + 12)/4 = 10.5
        # 11 > 10.5, emit LONG
        self.strategy.calculate_signals(self._create_market_event(12.0))

        self.assertGreater(len(self.events), 0)
        event = self.events.popleft()
        assert isinstance(event, SignalEvent)
        self.assertEqual(event.direction, "LONG")
        self.assertEqual(event.symbol, self.symbol)

    def test_long_only_exits_on_downcross(self):
        # Default strategy is long-only: a downward cross flattens to flat and
        # never opens a short.
        for price in [10.0, 10.0, 10.0, 10.0, 12.0]:
            self.strategy.calculate_signals(self._create_market_event(price))

        self.events.clear()

        # Price: 6 -> [10, 12, 8, 6] -> short_ma < long_ma -> EXIT only
        self.strategy.calculate_signals(self._create_market_event(8.0))
        self.strategy.calculate_signals(self._create_market_event(6.0))

        self.assertGreater(len(self.events), 0)
        event = self.events.popleft()
        assert isinstance(event, SignalEvent)
        self.assertEqual(event.direction, "EXIT")
        self.assertEqual(len(self.events), 0, "Long-only strategy must not emit SHORT")

    def test_crossover_flip_long_to_short(self):
        # Trigger LONG first
        for price in [10.0, 10.0, 10.0, 10.0, 12.0]:
            self.short_strategy.calculate_signals(self._create_market_event(price))

        # Empty the queue
        self.events.clear()

        # Prices are now: [10, 10, 10, 12] (maxlen 4) -> [10, 10, 12, 8]
        # Price: 8 -> short_ma = (12 + 8)/2 = 10
        # long_ma = (10 + 10 + 12 + 8)/4 = 10
        # Price: 6 -> [10, 12, 8, 6] -> short_ma = 7, long_ma = 9
        # 7 < 9, flip from LONG to SHORT: emit EXIT then SHORT
        self.short_strategy.calculate_signals(self._create_market_event(8.0))
        self.assertEqual(len(self.events), 0)  # MAs are equal, no change

        self.short_strategy.calculate_signals(self._create_market_event(6.0))
        self.assertGreater(len(self.events), 0)

        exit_event = self.events.popleft()
        assert isinstance(exit_event, SignalEvent)
        self.assertEqual(exit_event.direction, "EXIT")

        short_event = self.events.popleft()
        assert isinstance(short_event, SignalEvent)
        self.assertEqual(short_event.direction, "SHORT")
        self.assertEqual(short_event.symbol, self.symbol)

        self.assertEqual(len(self.events), 0)

    def test_crossover_short_from_flat(self):
        # First fully-formed window already has short_ma < long_ma, so the
        # strategy opens a short directly without a preceding EXIT.
        for price in [10.0, 10.0, 10.0, 8.0]:
            self.short_strategy.calculate_signals(self._create_market_event(price))

        # Prices: [10, 10, 10, 8] -> short_ma = 9, long_ma = 9.5 -> SHORT
        self.assertGreater(len(self.events), 0)
        event = self.events.popleft()
        assert isinstance(event, SignalEvent)
        self.assertEqual(event.direction, "SHORT")
        self.assertEqual(len(self.events), 0)

    def test_crossover_flip_short_to_long(self):
        # Open a short from flat first.
        for price in [10.0, 10.0, 10.0, 8.0]:
            self.short_strategy.calculate_signals(self._create_market_event(price))

        self.events.clear()

        # Prices: [10, 10, 8] -> add 14 -> [10, 10, 8, 14]
        # short_ma = (8 + 14)/2 = 11, long_ma = (10 + 10 + 8 + 14)/4 = 10.5
        # 11 > 10.5, flip from SHORT to LONG: emit EXIT then LONG
        self.short_strategy.calculate_signals(self._create_market_event(14.0))
        self.assertGreater(len(self.events), 0)

        exit_event = self.events.popleft()
        assert isinstance(exit_event, SignalEvent)
        self.assertEqual(exit_event.direction, "EXIT")

        long_event = self.events.popleft()
        assert isinstance(long_event, SignalEvent)
        self.assertEqual(long_event.direction, "LONG")
        self.assertEqual(len(self.events), 0)

    def test_rejected_order_reverts_intent_and_signal_refires(self):
        """
        The canonical desync scenario, now fixed.

        A LONG signal is emitted (intent -> LONG) but its order is rejected
        (e.g. the sizer declined). on_order_failed must revert intent to
        fill-truth (None), so that a later still-qualifying crossover fires the
        LONG again instead of being silently suppressed.
        """
        # Warm up to a fully-formed window with equal MAs (no signal yet).
        for price in [10.0, 10.0, 10.0, 10.0]:
            self.strategy.calculate_signals(self._create_market_event(price))
        self.assertEqual(len(self.events), 0)

        # First LONG crossover: [10, 10, 10, 12] -> short 11 > long 10.5.
        self.strategy.calculate_signals(self._create_market_event(12.0))
        first = self.events.popleft()
        assert isinstance(first, SignalEvent)
        self.assertEqual(first.direction, "LONG")
        self.assertEqual(self.strategy.intent[self.symbol], "LONG")

        # The order dies at fill time. Intent must roll back to fill-truth.
        self.strategy.on_order_failed(
            OrderFailedEvent(self.symbol, datetime.now(), "LONG", 10, "SIZER_DECLINED")
        )
        self.assertIsNone(self.strategy.intent[self.symbol])

        # A later bar still has short_ma > long_ma ([10, 10, 12, 12]): because
        # intent reverted, the LONG fires again - previously lost forever.
        self.strategy.calculate_signals(self._create_market_event(12.0))
        second = self.events.popleft()
        assert isinstance(second, SignalEvent)
        self.assertEqual(second.direction, "LONG")
        self.assertEqual(self.strategy.intent[self.symbol], "LONG")

    def test_exit_fill_preserves_pending_entry_intent(self):
        """
        Flatten-before-reverse: when a flip emits EXIT then the opposite entry,
        both orders are pending at once. The EXIT fill landing first must update
        fill-truth (position -> None) without clobbering the still-pending SHORT
        intent.
        """
        # Open a long and confirm the fill (position -> LONG).
        for price in [10.0, 10.0, 10.0, 10.0, 12.0]:
            self.short_strategy.calculate_signals(self._create_market_event(price))
        self.short_strategy.on_fill(
            FillEvent(self.symbol, datetime.now(), 10, "LONG", 12.0, 1.0, 0.0, "BUY")
        )
        self.assertEqual(self.short_strategy.position[self.symbol], "LONG")
        self.events.clear()

        # Flip LONG -> SHORT: [10, 10, 12, 8] equal MAs (no change), then
        # [10, 12, 8, 6] short 7 < long 9 -> EXIT then SHORT, intent -> SHORT.
        self.short_strategy.calculate_signals(self._create_market_event(8.0))
        self.short_strategy.calculate_signals(self._create_market_event(6.0))
        self.assertEqual(self.short_strategy.intent[self.symbol], "SHORT")

        # The EXIT fills first (FIFO). Position flattens but intent stays SHORT.
        self.short_strategy.on_fill(
            FillEvent(self.symbol, datetime.now(), 10, "EXIT", 6.0, 1.0, 0.0, "SELL")
        )
        self.assertIsNone(self.short_strategy.position[self.symbol])
        self.assertEqual(
            self.short_strategy.intent[self.symbol],
            "SHORT",
            "EXIT fill must not clear the pending SHORT intent",
        )

        # Then the SHORT fills: position confirms SHORT, intent unchanged.
        self.short_strategy.on_fill(
            FillEvent(self.symbol, datetime.now(), 10, "SHORT", 6.0, 1.0, 0.0, "SELL")
        )
        self.assertEqual(self.short_strategy.position[self.symbol], "SHORT")
        self.assertEqual(self.short_strategy.intent[self.symbol], "SHORT")


if __name__ == "__main__":
    unittest.main()
