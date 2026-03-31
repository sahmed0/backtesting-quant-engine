"""
Tests for the portfolio module.
"""

import unittest
from datetime import datetime
from queue import Queue
import polars as pl
from portfolio import Portfolio
from event import MarketEvent, SignalEvent, FillEvent

class TestPortfolio(unittest.TestCase):
    def test_portfolio_initialisation(self):
        """
        Tests that the portfolio initialises correctly.
        """
        events_queue = Queue()
        portfolio = Portfolio(events_queue=events_queue, initial_capital=50000.0)
        self.assertEqual(portfolio.initial_capital, 50000.0)
        self.assertEqual(portfolio.current_cash, 50000.0)
        self.assertEqual(portfolio.current_positions, {})
        self.assertEqual(portfolio.all_positions, [])
        self.assertEqual(portfolio.current_holdings, {})
        self.assertEqual(portfolio.all_holdings, [])
        self.assertEqual(portfolio.current_prices, {})

    def test_update_timeindex(self):
        """
        Tests that updating the timeindex correctly calculates holdings and equity.
        """
        events_queue = Queue()
        portfolio = Portfolio(events_queue=events_queue, initial_capital=100000.0)
        
        # Simulate an existing position
        portfolio.current_positions['AAPL'] = 10.0
        portfolio.current_cash = 90000.0  # Spent 10000 on AAPL initially
        
        # New market event for AAPL at price 1500
        event_time = datetime(2023, 1, 1, 10, 0, 0)
        event = MarketEvent(
            symbol='AAPL',
            timestamp=event_time,
            close=1500.0,
            high=1510.0,
            low=1490.0,
            volume=1000000.0
        )
        
        portfolio.update_timeindex(event)
        
        self.assertEqual(portfolio.current_prices['AAPL'], 1500.0)
        self.assertEqual(portfolio.current_holdings['AAPL'], 15000.0)
        self.assertEqual(portfolio.current_holdings['cash'], 90000.0)
        self.assertEqual(portfolio.current_holdings['total'], 105000.0)
        self.assertEqual(portfolio.current_holdings['timestamp'], event_time)
        
        self.assertEqual(len(portfolio.all_holdings), 1)
        self.assertEqual(portfolio.all_holdings[0]['total'], 105000.0)

    def test_update_signal(self):
        """
        Tests that SignalEvents enqueue valid OrderEvents based on rules.
        """
        events_queue = Queue()
        portfolio = Portfolio(events_queue=events_queue, initial_capital=100000.0)
        portfolio.current_prices['AAPL'] = 150.0 # Setup a price
        
        # Test valid LONG
        signal1 = SignalEvent('AAPL', datetime.now(), 'LONG')
        portfolio.update_signal(signal1)
        self.assertEqual(events_queue.qsize(), 1)
        order1 = events_queue.get()
        self.assertEqual(order1.type, 'ORDER')
        self.assertEqual(order1.direction, 'LONG')
        self.assertEqual(order1.quantity, 100.0)
        
        # Test invalid LONG (not enough cash)
        portfolio.current_cash = 100.0
        signal2 = SignalEvent('AAPL', datetime.now(), 'LONG')
        portfolio.update_signal(signal2)
        self.assertEqual(events_queue.qsize(), 0) # Should not queue
        
        # Test valid EXIT
        portfolio.current_positions['AAPL'] = 50.0
        signal3 = SignalEvent('AAPL', datetime.now(), 'EXIT')
        portfolio.update_signal(signal3)
        self.assertEqual(events_queue.qsize(), 1)
        order3 = events_queue.get()
        self.assertEqual(order3.direction, 'EXIT')
        self.assertEqual(order3.quantity, 50.0)
        
        # Test invalid EXIT (no shares)
        portfolio.current_positions['AAPL'] = 0.0
        signal4 = SignalEvent('AAPL', datetime.now(), 'EXIT')
        portfolio.update_signal(signal4)
        self.assertEqual(events_queue.qsize(), 0)

    def test_update_fill(self):
        """
        Tests updating positions and cash based on a fill.
        """
        events_queue = Queue()
        portfolio = Portfolio(events_queue=events_queue, initial_capital=100000.0)
        
        # Test LONG fill
        fill1 = FillEvent('AAPL', datetime.now(), 10.0, 'LONG', 150.0, 5.0, 1.0)
        portfolio.update_fill(fill1)
        self.assertEqual(portfolio.current_positions['AAPL'], 10.0)
        # Cost = 10*150 = 1500. Total = 1500+5+1 = 1506. Cash = 100000 - 1506 = 98494
        self.assertEqual(portfolio.current_cash, 98494.0)
        
        # Test EXIT fill
        fill2 = FillEvent('AAPL', datetime.now(), 5.0, 'EXIT', 200.0, 5.0, 1.0)
        portfolio.update_fill(fill2)
        self.assertEqual(portfolio.current_positions['AAPL'], 5.0)
        # Revenue = 5*200 = 1000. Cash increases by 1000 - 5 - 1 = 994. Cash = 98494 + 994 = 99488
        self.assertEqual(portfolio.current_cash, 99488.0)
        
    def test_generate_equity_curve(self):
        """
        Tests generating the equity curve DataFrame.
        """
        events_queue = Queue()
        portfolio = Portfolio(events_queue=events_queue, initial_capital=100000.0)
        
        # Empty curve
        df_empty = portfolio.generate_equity_curve()
        self.assertEqual(len(df_empty), 0)
        
        # With data
        t1 = datetime(2023, 1, 1)
        portfolio.all_holdings.append({'timestamp': t1, 'cash': 100000.0, 'total': 100000.0})
        t2 = datetime(2023, 1, 2)
        portfolio.all_holdings.append({'timestamp': t2, 'cash': 90000.0, 'AAPL': 15000.0, 'total': 105000.0})
        
        df = portfolio.generate_equity_curve()
        self.assertEqual(len(df), 2)
        self.assertListEqual(df.columns, ['timestamp', 'total'])
        self.assertEqual(df['total'][1], 105000.0)

if __name__ == '__main__':
    unittest.main()
