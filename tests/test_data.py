"""
Tests for the data handlers.
"""

import os
import shutil
import unittest
from collections import deque

from data import CSVDataHandler
from event import MarketEvent


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

        self.events = deque()
        self.handler = CSVDataHandler(self.events, self.csv_dir, ["AAPL"])

    def tearDown(self):
        """
        Cleans up the test environment.
        """
        if os.path.exists(self.csv_dir):
            shutil.rmtree(self.csv_dir)

    def test_update_bars(self):
        """
        Tests whether update_bars correctly pushes MarketEvents to the queue
        and updates the continue_backtest flag when data is exhausted.
        """
        self.assertTrue(self.handler.continue_backtest)

        # First bar
        self.handler.update_bars()
        self.assertEqual(len(self.events), 1)
        event1 = self.events.popleft()
        self.assertEqual(event1.symbol, "AAPL")
        self.assertEqual(event1.open, 149.5)
        self.assertEqual(event1.close, 150.0)

        # get_latest_bar returns the parsed MarketEvent itself: the CSV is
        # parsed exactly once, at the boundary, so consumers read floats
        # directly rather than casting strings on use.
        latest = self.handler.get_latest_bar("AAPL")
        self.assertIsInstance(latest, MarketEvent)
        self.assertEqual(latest.close, 150.0)

        # Second bar
        self.handler.update_bars()
        self.assertEqual(len(self.events), 1)
        event2 = self.events.popleft()
        self.assertEqual(event2.close, 150.5)

        # No more data
        self.handler.update_bars()
        self.assertFalse(self.handler.continue_backtest)


if __name__ == "__main__":
    unittest.main()
