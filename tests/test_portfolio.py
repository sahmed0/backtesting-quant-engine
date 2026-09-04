"""
Tests for the portfolio module.
"""

import unittest
from collections import deque
from datetime import datetime

from event import (
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderFailedEvent,
    SignalEvent,
)
from portfolio import Portfolio
from position_sizing import FixedSizer


class TestPortfolio(unittest.TestCase):
    def test_portfolio_initialisation(self):
        """
        Tests that the portfolio initialises correctly.
        """
        events = deque()
        portfolio = Portfolio(events=events, initial_capital=50000.0)
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
        events = deque()
        portfolio = Portfolio(events=events, initial_capital=100000.0)

        # Simulate an existing position
        portfolio.current_positions["AAPL"] = 10.0
        portfolio.current_cash = 90000.0  # Spent 10000 on AAPL initially

        # New market event for AAPL at price 1500
        event_time = datetime(2023, 1, 1, 10, 0, 0)
        event = MarketEvent(
            symbol="AAPL",
            timestamp=event_time,
            open=1495.0,
            high=1510.0,
            low=1490.0,
            close=1500.0,
            volume=1000000.0,
        )

        portfolio.update_timeindex(event)

        self.assertEqual(portfolio.current_prices["AAPL"], 1500.0)
        self.assertEqual(portfolio.current_holdings["AAPL"], 15000.0)
        self.assertEqual(portfolio.current_holdings["cash"], 90000.0)
        self.assertEqual(portfolio.current_holdings["total"], 105000.0)
        # Timestamps are stored as epoch floats for JSON/chart serialisation.
        self.assertEqual(
            portfolio.current_holdings["timestamp"], event_time.timestamp()
        )

        self.assertEqual(len(portfolio.all_holdings), 1)
        self.assertEqual(portfolio.all_holdings[0]["total"], 105000.0)

    def test_update_signal(self):
        """
        Tests that SignalEvents enqueue valid OrderEvents based on rules.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=100000.0)
        portfolio.current_prices["AAPL"] = 150.0  # Setup a price

        # Test valid LONG
        signal1 = SignalEvent("AAPL", datetime.now(), "LONG")
        portfolio.update_signal(signal1)
        self.assertEqual(len(events), 1)
        order1 = events.popleft()
        assert isinstance(order1, OrderEvent)
        self.assertEqual(order1.direction, "LONG")
        self.assertEqual(order1.side, "BUY")  # LONG entry is a BUY
        self.assertEqual(order1.quantity, 100.0)

        # A LONG that cash cannot cover is STILL ordered: affordability is no
        # longer judged here, because the order will fill at the next bar's
        # open, not at the price visible now. can_execute rejects it at fill
        # time instead.
        portfolio.current_cash = 100.0
        signal2 = SignalEvent("AAPL", datetime.now(), "LONG")
        portfolio.update_signal(signal2)
        self.assertEqual(len(events), 1)
        order2 = events.popleft()
        assert isinstance(order2, OrderEvent)
        self.assertEqual(order2.direction, "LONG")

        # Test valid EXIT of a long: closing a long is a SELL.
        portfolio.current_positions["AAPL"] = 50.0
        signal3 = SignalEvent("AAPL", datetime.now(), "EXIT")
        portfolio.update_signal(signal3)
        self.assertEqual(len(events), 1)
        order3 = events.popleft()
        assert isinstance(order3, OrderEvent)
        self.assertEqual(order3.direction, "EXIT")
        self.assertEqual(order3.side, "SELL")
        self.assertEqual(order3.quantity, 50.0)

        # An EXIT covering a short is a BUY.
        portfolio.current_positions["AAPL"] = -30.0
        signal3b = SignalEvent("AAPL", datetime.now(), "EXIT")
        portfolio.update_signal(signal3b)
        self.assertEqual(len(events), 1)
        order3b = events.popleft()
        assert isinstance(order3b, OrderEvent)
        self.assertEqual(order3b.side, "BUY")
        self.assertEqual(order3b.quantity, 30.0)

        # An EXIT with nothing to exit dies loudly rather than silently.
        portfolio.current_positions["AAPL"] = 0.0
        signal4 = SignalEvent("AAPL", datetime.now(), "EXIT")
        portfolio.update_signal(signal4)
        self.assertEqual(len(events), 1)
        failed = events.popleft()
        assert isinstance(failed, OrderFailedEvent)
        self.assertEqual(failed.reason, "NO_POSITION")

    def test_update_signal_sizer_declined(self):
        """
        A sizer returning 0 must not silently swallow the signal.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(
            events=events, initial_capital=100000.0, sizer=FixedSizer(0.0)
        )
        portfolio.current_prices["AAPL"] = 150.0

        portfolio.update_signal(SignalEvent("AAPL", datetime.now(), "LONG"))

        self.assertEqual(len(events), 1)
        failed = events.popleft()
        assert isinstance(failed, OrderFailedEvent)
        self.assertEqual(failed.reason, "SIZER_DECLINED")

    def test_update_signal_no_price(self):
        """
        An entry signal for a symbol with no price yet cannot be sized honestly.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=100000.0)
        # current_prices has no AAPL entry, so it defaults to 0.0.

        portfolio.update_signal(SignalEvent("AAPL", datetime.now(), "LONG"))

        self.assertEqual(len(events), 1)
        failed = events.popleft()
        assert isinstance(failed, OrderFailedEvent)
        self.assertEqual(failed.reason, "NO_PRICE")

    def test_can_execute(self):
        """
        Fill-time affordability. The BUY arm mirrors exactly what update_fill
        charges: fill_price * qty + commission. Slippage is already inside
        fill_price, so the slippage argument is ignored here.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=1000.0)
        order = OrderEvent("AAPL", datetime.now(), 10.0, "LONG", "MARKET", "BUY")

        # Costs 99.0 * 10 + 1.0 = 991.0, and cash is 1000.0. Slippage arg ignored.
        self.assertEqual(portfolio.can_execute(order, 99.0, 1.0, 0.5), (True, None))

        # Costs 100.0 * 10 + 1.0 = 1001.0 > 1000.0.
        self.assertEqual(
            portfolio.can_execute(order, 100.0, 1.0, 0.5),
            (False, "INSUFFICIENT_CASH"),
        )

        # Exactly affordable: 99.9 * 10 + 1.0 = 1000.0. Boundary is >=.
        self.assertEqual(portfolio.can_execute(order, 99.9, 1.0, 0.5), (True, None))

        # A SHORT is a SELL, which brings cash in rather than consuming it, so
        # it is allowed even with the cash that rejected the LONG.
        short = OrderEvent("AAPL", datetime.now(), 10.0, "SHORT", "MARKET", "SELL")
        self.assertEqual(portfolio.can_execute(short, 100.0, 1.0, 0.5), (True, None))

    def test_update_fill(self):
        """
        Tests updating positions and cash based on a fill. Cash moves by side,
        and slippage is never charged again - it is already in fill_price.
        """
        events = deque()
        portfolio = Portfolio(events=events, initial_capital=100000.0)

        # BUY fill (LONG entry): pay fill_cost + commission only.
        fill1 = FillEvent("AAPL", datetime.now(), 10.0, "LONG", 150.0, 5.0, 1.0, "BUY")
        portfolio.update_fill(fill1)
        self.assertEqual(portfolio.current_positions["AAPL"], 10.0)
        # Cost = 10*150 = 1500. + commission 5 = 1505. Cash = 100000 - 1505 = 98495.
        # The slippage of 1.0 is NOT charged.
        self.assertEqual(portfolio.current_cash, 98495.0)

        # SELL fill (EXIT closing a long): receive fill_cost - commission.
        fill2 = FillEvent("AAPL", datetime.now(), 5.0, "EXIT", 200.0, 5.0, 1.0, "SELL")
        portfolio.update_fill(fill2)
        self.assertEqual(portfolio.current_positions["AAPL"], 5.0)
        # Revenue = 5*200 = 1000. - commission 5 = 995. Cash = 98495 + 995 = 99490.
        self.assertEqual(portfolio.current_cash, 99490.0)

    def test_buy_fill_does_not_double_charge_slippage(self):
        """
        Regression for the removed double-count: a BUY fill must change cash by
        exactly -(fill_price * qty + commission), with slippage untouched.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=100000.0)

        fill = FillEvent(
            "AAPL", datetime.now(), 10.0, "LONG", 105.0525, 1.0, 0.525, "BUY"
        )
        portfolio.update_fill(fill)

        # Exactly fill_price*qty + commission = 1050.525 + 1.0 = 1051.525.
        self.assertAlmostEqual(
            portfolio.current_cash, 100000.0 - (105.0525 * 10.0 + 1.0), places=9
        )

    def test_generate_equity_curve(self):
        """
        Tests generating the equity curve DataFrame.
        """
        events = deque()
        portfolio = Portfolio(events=events, initial_capital=100000.0)

        # Empty curve
        df_empty = portfolio.generate_equity_curve()
        self.assertEqual(len(df_empty), 0)

        # With data
        t1 = datetime(2023, 1, 1)
        portfolio.all_holdings.append(
            {"timestamp": t1, "cash": 100000.0, "total": 100000.0}
        )
        t2 = datetime(2023, 1, 2)
        portfolio.all_holdings.append(
            {"timestamp": t2, "cash": 90000.0, "AAPL": 15000.0, "total": 105000.0}
        )

        df = portfolio.generate_equity_curve()
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["timestamp", "total"])
        self.assertEqual(df["total"][1], 105000.0)


if __name__ == "__main__":
    unittest.main()
