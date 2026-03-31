"""
Tests for the data handlers.
"""

import unittest
from queue import Queue
import os
import shutil
from datetime import datetime
from data import PolarsCSVDataHandler

class TestPolarsCSVDataHandler(unittest.TestCase):
    """
    Test suite for the Polars-based CSV data handler.
    """

    def setUp(self):
        """
        Sets up the test environment by creating dummy CSV data.
        """
        self.csvDir = "test_data_tmp"
        os.makedirs(self.csvDir, exist_ok=True)
        with open(os.path.join(self.csvDir, "AAPL.csv"), "w") as f:
            f.write("timestamp,close,high,low,volume\n")
            f.write("2023-01-01T10:00:00,150.0,151.0,149.0,1000\n")
            f.write("2023-01-01T10:01:00,150.5,151.5,150.0,1500\n")
        
        self.eventsQueue = Queue()
        self.handler = PolarsCSVDataHandler(self.eventsQueue, self.csvDir, ["AAPL"])

    def tearDown(self):
        """
        Cleans up the test environment.
        """
        if os.path.exists(self.csvDir):
            shutil.rmtree(self.csvDir)

    def testUpdateBars(self):
        """
        Tests whether updateBars correctly pushes MarketEvents to the queue
        and updates the shouldContinueBacktest flag when data is exhausted.
        """
        self.assertTrue(self.handler.shouldContinueBacktest)
        
        # First bar
        self.handler.updateBars()
        self.assertEqual(self.eventsQueue.qsize(), 1)
        event1 = self.eventsQueue.get()
        self.assertEqual(event1.symbol, "AAPL")
        self.assertEqual(event1.close, 150.0)
        
        latest = self.handler.getLatestBar("AAPL")
        self.assertEqual(latest['close'], 150.0)

        # Second bar
        self.handler.updateBars()
        self.assertEqual(self.eventsQueue.qsize(), 1)
        event2 = self.eventsQueue.get()
        self.assertEqual(event2.close, 150.5)

        # No more data
        self.handler.updateBars()
        self.assertFalse(self.handler.shouldContinueBacktest)

if __name__ == "__main__":
    unittest.main()
