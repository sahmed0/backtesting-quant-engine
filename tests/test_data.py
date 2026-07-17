"""
Tests for the data handlers.
"""

import os
import shutil
import unittest
from queue import Queue

from data import CSVDataHandler


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
            f.write("timestamp,close,high,low,volume\n")
            f.write("2023-01-01T10:00:00,150.0,151.0,149.0,1000\n")
            f.write("2023-01-01T10:01:00,150.5,151.5,150.0,1500\n")

        self.events = Queue()
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
        self.assertEqual(self.events.qsize(), 1)
        event1 = self.events.get()
        self.assertEqual(event1.symbol, "AAPL")
        self.assertEqual(event1.close, 150.0)

        # get_latest_bar returns the raw CSV row; values are strings that the
        # rest of the system casts to float on use.
        latest = self.handler.get_latest_bar("AAPL")
        self.assertEqual(float(latest["close"]), 150.0)

        # Second bar
        self.handler.update_bars()
        self.assertEqual(self.events.qsize(), 1)
        event2 = self.events.get()
        self.assertEqual(event2.close, 150.5)

        # No more data
        self.handler.update_bars()
        self.assertFalse(self.handler.continue_backtest)


if __name__ == "__main__":
    unittest.main()
