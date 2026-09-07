"""
Golden full-stack end-to-end test.

Runs the real engine over the on-disk tests/fixtures/GOLD.csv through the real
CSVDataHandler - data -> strategy -> portfolio -> execution -> performance - and
locks every value to the penny: the three fills (bar, direction, side, quantity,
fill price, commission, slippage), the dropped end-of-data order, the final cash,
the full 13-point equity curve, and the summary total return / max drawdown /
trade count / win rate.

Every number here was hand-computed independently of the engine; if the engine
disagrees, the engine is wrong first. Sharpe and CAGR are deliberately not
asserted - they are annualisation-sensitive and covered by the performance unit
tests.
"""

import asyncio
from collections import deque
from datetime import UTC, datetime

import pytest

import performance
from data import CSVDataHandler
from engine import Backtest
from event import Event
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import FixedSizer
from strategy import SimpleMovingAverageStrategy

# Trades ledger, in fill order. Each row is
# (direction, side, quantity, fill_price, commission, slippage, fill_bar_day).
# fill_bar_day is the day-of-month of the bar the fill lands on (bar 5 =>
# 2024-01-06, bar 9 => 2024-01-10, bar 11 => 2024-01-12).
EXPECTED_FILLS = [
    ("LONG", "BUY", 10.0, 105.0525, 1.00, 0.525, 6),
    ("EXIT", "SELL", 10.0, 99.95, 1.00, 0.50, 10),
    ("LONG", "BUY", 10.0, 114.057, 1.00, 0.57, 12),
]

# Equity curve, one point per bar close (13 bars).
EXPECTED_EQUITY_CURVE = [
    10000.0,
    10000.0,
    10000.0,
    10000.0,
    10000.0,
    10008.475,
    10028.475,
    10048.475,
    9958.475,
    9946.975,
    9946.975,
    9955.405,
    9705.405,
]


def _fill_bar_timestamp(day: int) -> float:
    """Unix seconds of the GOLD bar dated 2024-01-<day> (UTC), as stored."""
    return datetime(2024, 1, day, tzinfo=UTC).timestamp()


def test_golden_run_matches_the_hand_traced_ledger():
    events: deque[Event] = deque()
    data_handler = CSVDataHandler(events, "tests/fixtures", ["GOLD"])
    strategy = SimpleMovingAverageStrategy(events, short_window=2, long_window=4)
    portfolio = Portfolio(events, initial_capital=10_000.0, sizer=FixedSizer(10.0))
    execution_handler = SimulatedExecutionHandler(
        events,
        data_handler,
        portfolio,
        commission_per_share=0.005,
        min_commission=1.00,
        slippage_pct=0.0005,
    )
    backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)
    asyncio.run(backtest.run())

    # --- Fills, exact, in order ---------------------------------------------
    assert len(portfolio.trades) == len(EXPECTED_FILLS)
    for trade, expected in zip(portfolio.trades, EXPECTED_FILLS, strict=True):
        direction, side, qty, price, commission, slippage, bar_day = expected
        assert trade["symbol"] == "GOLD"
        assert trade["direction"] == direction
        assert trade["side"] == side
        assert trade["quantity"] == pytest.approx(qty, rel=1e-12)
        # The portfolio records the fill price under "price".
        assert trade["price"] == pytest.approx(price, rel=1e-12)
        assert trade["commission"] == pytest.approx(commission, rel=1e-12)
        assert trade["slippage"] == pytest.approx(slippage, rel=1e-12)
        # Each fill lands on the bar *after* the signal (next-open fill).
        assert trade["timestamp"] == pytest.approx(_fill_bar_timestamp(bar_day))

    # The bar-12 EXIT signal pends and is dropped when the data ends.
    assert execution_handler.dropped_orders == 1

    # --- Cash and equity curve --------------------------------------------
    assert portfolio.current_cash == pytest.approx(8805.405, rel=1e-12)

    equity_curve = portfolio.generate_equity_curve()["total"].to_list()
    assert len(equity_curve) == len(EXPECTED_EQUITY_CURVE)
    for actual_equity, expected_equity in zip(
        equity_curve, EXPECTED_EQUITY_CURVE, strict=True
    ):
        assert actual_equity == pytest.approx(expected_equity, rel=1e-12)

    # --- Summary stats -------------------------------------------------------
    stats = performance.create_summary_stats(portfolio)
    assert stats["total_return"] == pytest.approx(-0.0294595, rel=1e-6)
    # Peak equity is bar 7 (10048.475); the trough is the final bar (9705.405).
    assert stats["max_drawdown"] == pytest.approx(343.07 / 10048.475, rel=1e-9)
    # Only one round trip completes; the bar-11 LONG never closes (dropped EXIT).
    assert stats["num_trades"] == 1
    # That single round trip nets -53.025, so no trip is profitable.
    assert stats["win_rate"] == 0.0
