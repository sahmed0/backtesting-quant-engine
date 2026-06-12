"""
Tests for position sizing strategies.
"""

import math
import unittest
from queue import Queue

from portfolio import Portfolio
from position_sizing import (
    FixedSizer,
    PercentEquitySizer,
    VolatilityTargetSizer,
    ATRStopSizer,
    FractionalKellySizer,
)


def make_trade(symbol, direction, quantity, price, commission=0.0, slippage=0.0):
    """Builds a trade record matching the shape Portfolio.update_fill records."""
    return {
        "timestamp": 0.0,
        "symbol": symbol,
        "direction": direction,
        "quantity": quantity,
        "price": price,
        "commission": commission,
        "slippage": slippage,
    }


class TestFixedSizer(unittest.TestCase):
    def test_returns_constant_quantity(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = FixedSizer(100.0)
        self.assertEqual(sizer.size("AAPL", "LONG", 150.0, portfolio), 100.0)
        # Independent of price and direction.
        self.assertEqual(sizer.size("AAPL", "SHORT", 999.0, portfolio), 100.0)

    def test_is_default_sizer(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        self.assertIsInstance(portfolio.sizer, FixedSizer)


class TestPercentEquitySizer(unittest.TestCase):
    def test_allocates_fraction_of_equity(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = PercentEquitySizer(fraction=0.1)
        # 10% of 100,000 = 10,000 of exposure at $200 => 50 units.
        self.assertEqual(sizer.size("AAPL", "LONG", 200.0, portfolio), 50.0)

    def test_sizes_off_current_equity_including_positions(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        # Hold 100 shares now worth $500 each, cash drawn down accordingly.
        portfolio.current_positions["AAPL"] = 100.0
        portfolio.current_prices["AAPL"] = 500.0
        portfolio.current_cash = 50000.0
        # Equity = 50,000 cash + 50,000 position = 100,000. 10% / $100 = 100 units.
        self.assertEqual(sizer_size(portfolio, 0.1, "MSFT", 100.0), 100.0)

    def test_zero_price_returns_zero(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = PercentEquitySizer(fraction=0.1)
        self.assertEqual(sizer.size("AAPL", "LONG", 0.0, portfolio), 0.0)

    def test_rejects_non_positive_fraction(self):
        with self.assertRaises(ValueError):
            PercentEquitySizer(fraction=0.0)


def sizer_size(portfolio, fraction, symbol, price):
    return PercentEquitySizer(fraction=fraction).size(symbol, "LONG", price, portfolio)


class TestVolatilityTargetSizer(unittest.TestCase):
    def test_returns_zero_before_enough_history(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = VolatilityTargetSizer(target_volatility=0.15, lookback=5)
        # Only 3 prices fed; need lookback + 1 = 6.
        for price in (100.0, 101.0, 102.0):
            sizer.update_market("AAPL", price)
        self.assertEqual(sizer.size("AAPL", "LONG", 102.0, portfolio), 0.0)

    def test_flat_series_returns_zero(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = VolatilityTargetSizer(target_volatility=0.15, lookback=5)
        for _ in range(6):
            sizer.update_market("AAPL", 100.0)  # zero volatility
        self.assertEqual(sizer.size("AAPL", "LONG", 100.0, portfolio), 0.0)

    def test_lower_volatility_gives_larger_position(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)

        calm = VolatilityTargetSizer(
            target_volatility=0.15, lookback=5, max_leverage=100.0
        )
        choppy = VolatilityTargetSizer(
            target_volatility=0.15, lookback=5, max_leverage=100.0
        )

        for p in (100.0, 100.1, 100.2, 100.1, 100.2, 100.3):
            calm.update_market("AAPL", p)
        for p in (100.0, 105.0, 95.0, 108.0, 92.0, 110.0):
            choppy.update_market("AAPL", p)

        calm_qty = calm.size("AAPL", "LONG", 100.3, portfolio)
        choppy_qty = choppy.size("AAPL", "LONG", 110.0, portfolio)
        self.assertGreater(calm_qty, choppy_qty)

    def test_respects_max_leverage_cap(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        # Very calm series wants huge leverage; cap holds it to max_leverage.
        sizer = VolatilityTargetSizer(
            target_volatility=10.0, lookback=5, max_leverage=1.0
        )
        for p in (100.0, 100.01, 100.02, 100.01, 100.02, 100.03):
            sizer.update_market("AAPL", p)
        qty = sizer.size("AAPL", "LONG", 100.03, portfolio)
        # Capped at 1x equity: 100,000 / 100.03.
        self.assertTrue(math.isclose(qty, 100000.0 / 100.03, rel_tol=1e-9))

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            VolatilityTargetSizer(target_volatility=0.0)
        with self.assertRaises(ValueError):
            VolatilityTargetSizer(lookback=1)


class TestATRStopSizer(unittest.TestCase):
    def _feed(self, sizer, symbol, bars):
        for high, low, close in bars:
            sizer.update_market(symbol, close, high, low)

    def test_returns_zero_before_enough_history(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = ATRStopSizer(risk_fraction=0.02, atr_period=3)
        # Need atr_period + 1 = 4 bars; feed only 3.
        self._feed(sizer, "AAPL", [(101, 99, 100), (102, 100, 101), (103, 101, 102)])
        self.assertEqual(sizer.size("AAPL", "LONG", 102.0, portfolio), 0.0)

    def test_risks_fixed_fraction_of_equity(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = ATRStopSizer(
            risk_fraction=0.02, atr_period=3, atr_multiple=2.0, max_leverage=100.0
        )
        # Each bar has a constant True Range of 2.0 (high-low=2, no gaps), so
        # ATR = 2.0 and stop_distance = 2 * 2 = 4.
        self._feed(
            sizer,
            "AAPL",
            [
                (101, 99, 100),
                (101, 99, 100),
                (101, 99, 100),
                (101, 99, 100),
            ],
        )
        # risk capital = 2% of 100,000 = 2,000; qty = 2,000 / 4 = 500.
        self.assertTrue(
            math.isclose(sizer.size("AAPL", "LONG", 100.0, portfolio), 500.0)
        )

    def test_wider_atr_gives_smaller_position(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        tight = ATRStopSizer(risk_fraction=0.02, atr_period=3, max_leverage=100.0)
        wide = ATRStopSizer(risk_fraction=0.02, atr_period=3, max_leverage=100.0)
        self._feed(tight, "AAPL", [(100.5, 99.5, 100)] * 4)  # TR = 1
        self._feed(wide, "AAPL", [(105, 95, 100)] * 4)  # TR = 10
        tight_qty = tight.size("AAPL", "LONG", 100.0, portfolio)
        wide_qty = wide.size("AAPL", "LONG", 100.0, portfolio)
        self.assertGreater(tight_qty, wide_qty)

    def test_respects_max_leverage_cap(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        # Tiny ATR would demand a huge position; cap holds it to 1x equity.
        sizer = ATRStopSizer(
            risk_fraction=0.02, atr_period=3, atr_multiple=2.0, max_leverage=1.0
        )
        self._feed(sizer, "AAPL", [(100.001, 99.999, 100)] * 4)
        qty = sizer.size("AAPL", "LONG", 100.0, portfolio)
        self.assertTrue(math.isclose(qty, 100000.0 / 100.0))

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            ATRStopSizer(risk_fraction=0.0)
        with self.assertRaises(ValueError):
            ATRStopSizer(atr_multiple=0.0)


class TestFractionalKellySizer(unittest.TestCase):
    def test_uses_base_fraction_before_min_trades(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        sizer = FractionalKellySizer(min_trades=10, base_fraction=0.02)
        # No trade history yet -> base fraction: 2% of 100,000 / $100 = 20 units.
        self.assertTrue(
            math.isclose(sizer.size("AAPL", "LONG", 100.0, portfolio), 20.0)
        )

    def test_kelly_fraction_from_realised_trades(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        # Build a history: 60% win rate, wins +10%, losses -5% => R = 2.
        # f* = 0.6 - 0.4/2 = 0.4; half Kelly => 0.2 of equity.
        portfolio.trades = []
        for _ in range(6):
            portfolio.trades.append(make_trade("AAPL", "LONG", 10, 100.0))
            portfolio.trades.append(make_trade("AAPL", "EXIT", 10, 110.0))
        for _ in range(4):
            portfolio.trades.append(make_trade("AAPL", "LONG", 10, 100.0))
            portfolio.trades.append(make_trade("AAPL", "EXIT", 10, 95.0))

        sizer = FractionalKellySizer(kelly_fraction=0.5, min_trades=10)
        # 0.2 of 100,000 = 20,000 at $100 => 200 units.
        self.assertTrue(
            math.isclose(sizer.size("AAPL", "LONG", 100.0, portfolio), 200.0)
        )

    def test_no_edge_returns_zero(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        # All losers -> no edge -> no position.
        portfolio.trades = []
        for _ in range(10):
            portfolio.trades.append(make_trade("AAPL", "LONG", 10, 100.0))
            portfolio.trades.append(make_trade("AAPL", "EXIT", 10, 95.0))
        sizer = FractionalKellySizer(kelly_fraction=0.5, min_trades=10)
        self.assertEqual(sizer.size("AAPL", "LONG", 100.0, portfolio), 0.0)

    def test_short_trades_scored_correctly(self):
        portfolio = Portfolio(events_queue=Queue(), initial_capital=100000.0)
        # Short entered at 100, covered at 90 -> a winning short (+10%).
        portfolio.trades = [
            make_trade("AAPL", "SHORT", 10, 100.0),
            make_trade("AAPL", "EXIT", 10, 90.0),
        ]
        returns = FractionalKellySizer._completed_returns(portfolio.trades, "AAPL")
        self.assertEqual(len(returns), 1)
        self.assertGreater(returns[0], 0.0)

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            FractionalKellySizer(kelly_fraction=0.0)
        with self.assertRaises(ValueError):
            FractionalKellySizer(min_trades=0)


if __name__ == "__main__":
    unittest.main()
