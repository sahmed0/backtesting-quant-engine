"""
Tests for the data handlers.
"""

import os
import shutil
import unittest
from collections import deque

from data import CSVDataHandler
from event import Event, MarketEvent


class TestCSVDataHandler(unittest.TestCase):
    """
    Test suite for the streaming CSV data handler.
    """

    def setUp(self):
        """
        Sets up the test environment by creating dummy CSV data.
        """
        self.csv_dir = "test_data_tmp"
        os.makedirs(self.csv_dir, exist_ok=True)
        with open(os.path.join(self.csv_dir, "AAPL.csv"), "w") as f:
            f.write("timestamp,open,high,low,close,volume\n")
            f.write("2023-01-01T10:00:00,149.5,151.0,149.0,150.0,1000\n")
            f.write("2023-01-01T10:01:00,150.0,151.5,150.0,150.5,1500\n")

        self.events: deque[Event] = deque()
        self.handler = CSVDataHandler(self.events, self.csv_dir, ["AAPL"])

    def tearDown(self):
        """
        Cleans up the test environment.
        """
        if os.path.exists(self.csv_dir):
            shutil.rmtree(self.csv_dir)

    def test_update_bars(self):
        """
        Tests that update_bars returns each parsed bar in turn and None once the
        data is exhausted. The bar is returned rather than queued: it is the
        engine's loop condition.
        """
        # First bar
        event1 = self.handler.update_bars()
        assert isinstance(event1, MarketEvent)
        self.assertEqual(event1.symbol, "AAPL")
        self.assertEqual(event1.close, 150.0)
        self.assertEqual(event1.open, 149.5)

        # get_latest_bar returns the parsed MarketEvent itself: the CSV is
        # parsed exactly once, at the boundary, so consumers read floats
        # directly rather than casting strings on use.
        latest = self.handler.get_latest_bar("AAPL")
        assert isinstance(latest, MarketEvent)
        self.assertEqual(latest.close, 150.0)

        # Second bar
        event2 = self.handler.update_bars()
        assert isinstance(event2, MarketEvent)
        self.assertEqual(event2.close, 150.5)

        # No more data
        self.assertIsNone(self.handler.update_bars())

        # The handler no longer queues anything.
        self.assertEqual(len(self.events), 0)

    def test_get_latest_bar_before_first_update(self):
        """
        Before any bar has been read there is no bar to return -- None, not an
        empty dict, so callers can branch on a real absence.
        """
        self.assertIsNone(self.handler.get_latest_bar("AAPL"))

    def test_rejects_multiple_symbols(self):
        """
        The engine is single-symbol.
        """
        with self.assertRaises(ValueError) as ctx:
            CSVDataHandler(deque(), self.csv_dir, ["AAPL", "MSFT"])
        self.assertIn("exactly one symbol", str(ctx.exception))

    def test_rejects_empty_symbol_list(self):
        """
        Zero symbols is as unsupported as two.
        """
        with self.assertRaises(ValueError):
            CSVDataHandler(deque(), self.csv_dir, [])

    def test_rejects_csv_missing_required_columns(self):
        """
        A CSV without an `open` column cannot supply a complete bar. Orders fill
        at the next bar's open, so this must fail loudly at construction rather
        than as a KeyError midway through a run.
        """
        with open(os.path.join(self.csv_dir, "NOOPEN.csv"), "w") as f:
            f.write("timestamp,high,low,close,volume\n")
            f.write("2023-01-01T10:00:00,151.0,149.0,150.0,1000\n")

        with self.assertRaises(ValueError) as ctx:
            CSVDataHandler(deque(), self.csv_dir, ["NOOPEN"])

        message = str(ctx.exception)
        self.assertIn("NOOPEN.csv", message)
        self.assertIn("open", message)


if __name__ == "__main__":
    unittest.main()
