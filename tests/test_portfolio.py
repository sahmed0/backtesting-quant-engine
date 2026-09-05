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
        events: deque[Event] = deque()
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
        events: deque[Event] = deque()
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

    def test_can_execute_short_margin(self):
        """
        A short is checked at fill time against 50% of notional + commission
        (Reg-T-style 150% collateral). An EXIT is always allowed - it never
        needs new cash.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=1000.0)
        short = OrderEvent("AAPL", datetime.now(), 10.0, "SHORT", "MARKET", "SELL")

        # Notional 100*10 = 1000; required margin = 0.5*1000 + 1 = 501 <= 1000.
        self.assertEqual(portfolio.can_execute(short, 100.0, 1.0, 0.5), (True, None))

        # Exact boundary: cash 501 == 0.5*1000 + 1. Boundary is >=.
        portfolio.current_cash = 501.0
        self.assertEqual(portfolio.can_execute(short, 100.0, 1.0, 0.5), (True, None))

        # Cash below the margin requirement is rejected.
        portfolio.current_cash = 500.0
        self.assertEqual(
            portfolio.can_execute(short, 100.0, 1.0, 0.5),
            (False, "INSUFFICIENT_MARGIN"),
        )

        # An EXIT is unconditionally allowed, even with no cash: covering a short
        # draws on the proceeds ledger and closing a long only brings cash in.
        portfolio.current_cash = 0.0
        exit_long = OrderEvent("AAPL", datetime.now(), 10.0, "EXIT", "MARKET", "SELL")
        exit_short = OrderEvent("AAPL", datetime.now(), 10.0, "EXIT", "MARKET", "BUY")
        self.assertEqual(
            portfolio.can_execute(exit_long, 100.0, 1.0, 0.5), (True, None)
        )
        self.assertEqual(
            portfolio.can_execute(exit_short, 100.0, 1.0, 0.5), (True, None)
        )

    def test_update_fill(self):
        """
        Tests updating positions and cash based on a fill. Cash moves by side,
        and slippage is never charged again - it is already in fill_price.
        """
        events: deque[Event] = deque()
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

    def test_short_entry_segregates_proceeds(self):
        """
        A short entry's proceeds go to the segregated ledger, not spendable
        cash. Only the commission comes out of cash, and total equity is
        unchanged by the entry apart from that commission.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=10000.0)
        portfolio.current_prices["AAPL"] = 100.0  # mark at the entry price

        # SHORT entry (SELL) of 10 @ 100: proceeds 1000 to the ledger, cash
        # loses only the commission, position goes to -10.
        fill = FillEvent("AAPL", datetime.now(), 10.0, "SHORT", 100.0, 1.0, 0.5, "SELL")
        portfolio.update_fill(fill)

        self.assertEqual(portfolio.current_positions["AAPL"], -10.0)
        self.assertEqual(portfolio.short_proceeds["AAPL"], 1000.0)
        self.assertEqual(portfolio.current_cash, 9999.0)  # 10000 - commission
        # Equity = cash 9999 + proceeds 1000 + (-10 * 100) = 9999. The proceeds
        # net exactly against the negative market value: no free money.
        self.assertEqual(portfolio.total_equity(), 9999.0)

    def test_short_cover_draws_down_ledger(self):
        """
        Covering a short (BUY) returns the segregated proceeds, pays for the
        cover and the commission, and zeroes both the position and the ledger.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=10000.0)

        # Open the short first: cash 9999, proceeds 1000, pos -10.
        portfolio.update_fill(
            FillEvent("AAPL", datetime.now(), 10.0, "SHORT", 100.0, 1.0, 0.5, "SELL")
        )
        # Cover 10 @ 90 (price fell, so the short profits).
        portfolio.update_fill(
            FillEvent("AAPL", datetime.now(), 10.0, "EXIT", 90.0, 1.0, 0.5, "BUY")
        )

        self.assertEqual(portfolio.current_positions["AAPL"], 0.0)
        self.assertEqual(portfolio.short_proceeds["AAPL"], 0.0)
        # cash = 9999 + proceeds 1000 - cover 900 - commission 1 = 10098. The
        # short made 100 gross (sold @100, covered @90) minus two commissions.
        self.assertEqual(portfolio.current_cash, 10098.0)

    def test_partial_short_cover_is_an_engine_bug(self):
        """
        The strategy always flattens the whole short before reversing, so a
        cover for anything but the full position is an engine bug, not a user
        error - update_fill must fail loudly.
        """
        events: deque[Event] = deque()
        portfolio = Portfolio(events=events, initial_capital=10000.0)
        portfolio.update_fill(
            FillEvent("AAPL", datetime.now(), 10.0, "SHORT", 100.0, 1.0, 0.5, "SELL")
        )
        with self.assertRaises(ValueError):
            portfolio.update_fill(
                FillEvent("AAPL", datetime.now(), 4.0, "EXIT", 90.0, 1.0, 0.5, "BUY")
            )

    def test_short_proceeds_do_not_lever_the_sizer(self):
        """
        A PercentEquitySizer sizes off total_equity(). Opening a
        short must not inflate equity, so the quantity sized for a later LONG is
        identical to the no-short baseline.
        """
        from position_sizing import PercentEquitySizer

        sizer = PercentEquitySizer(0.1)

        # Baseline: no short open, equity == initial capital, price 50.
        baseline = Portfolio(events=deque(), initial_capital=10000.0, sizer=sizer)
        baseline.current_prices["AAPL"] = 50.0
        baseline_qty = sizer.size("AAPL", "LONG", 50.0, baseline)

        # With a short open first: proceeds are segregated, so equity is
        # unchanged (bar the commission), and the LONG sizes the same.
        with_short = Portfolio(events=deque(), initial_capital=10000.0, sizer=sizer)
        with_short.current_prices["MSFT"] = 100.0
        with_short.update_fill(
            FillEvent("MSFT", datetime.now(), 10.0, "SHORT", 100.0, 0.0, 0.5, "SELL")
        )
        with_short.current_prices["AAPL"] = 50.0
        short_qty = sizer.size("AAPL", "LONG", 50.0, with_short)

        # Commission was 0, so equity is exactly equal and the quantities match.
        self.assertEqual(with_short.total_equity(), 10000.0)
        self.assertEqual(short_qty, baseline_qty)

    def test_generate_equity_curve(self):
        """
        Tests generating the equity curve DataFrame.
        """
        events: deque[Event] = deque()
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
