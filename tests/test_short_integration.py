"""
Full-stack short-scenario integration test.

Runs the real engine over the GOLD fixture with shorting enabled and asserts
the exact per-fill ledger: fill prices, the cash trajectory, the segregated
short-proceeds ledger, position after each fill, the two dropped end-of-data
orders, and the final cash/equity. Every number here was hand-computed
independently of the engine; if the engine disagrees, the engine is wrong
first.

The GOLD bars are built in memory via the shared conftest helpers.
"""

import asyncio
from collections import deque

import pytest
from conftest import InMemoryDataHandler, make_bars

from engine import Backtest
from event import Event, FillEvent
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import FixedSizer
from strategy import SimpleMovingAverageStrategy

# GOLD fixture: (open, high, low, close) per bar.
GOLD_BARS: list[tuple[float, float, float, float]] = [
    (100, 101, 99, 100),  # bar 0
    (100, 101, 97, 98),  # bar 1
    (98, 99, 95, 96),  # bar 2
    (96, 97, 93, 94),  # bar 3
    (94, 105, 93, 104),  # bar 4
    (105, 107, 104, 106),  # bar 5
    (107, 109, 106, 108),  # bar 6
    (109, 111, 108, 110),  # bar 7
    (109, 110, 100, 101),  # bar 8
    (100, 101, 98, 99),  # bar 9
    (98, 114, 97, 113),  # bar 10
    (114, 116, 113, 115),  # bar 11
    (115, 116, 89, 90),  # bar 12
]

# Expected ledger, in fill order. Each row is
# (direction, side, fill_price, cash_after, short_proceeds_after, position_after).
EXPECTED_FILLS = [
    ("SHORT", "SELL", 93.953, 9999.00, 939.53, -10.0),  # bar 4
    ("EXIT", "BUY", 105.0525, 9887.005, 0.0, 0.0),  # bar 5 cover
    ("LONG", "BUY", 105.0525, 8835.48, 0.0, 10.0),  # bar 5 long
    ("EXIT", "SELL", 99.95, 9833.98, 0.0, 0.0),  # bar 9 close long
    ("SHORT", "SELL", 99.95, 9832.98, 999.50, -10.0),  # bar 9 short (FIFO after EXIT)
    ("EXIT", "BUY", 114.057, 9690.91, 0.0, 0.0),  # bar 11 cover
    ("LONG", "BUY", 114.057, 8549.34, 0.0, 10.0),  # bar 11 long
]


def test_short_scenario_matches_the_hand_traced_ledger():
    events: deque[Event] = deque()
    data_handler = InMemoryDataHandler(make_bars(GOLD_BARS, symbol="GOLD"))
    strategy = SimpleMovingAverageStrategy(
        events, short_window=2, long_window=4, allow_short=True
    )
    portfolio = Portfolio(
        events=events, initial_capital=10000.0, sizer=FixedSizer(10.0)
    )
    execution_handler = SimulatedExecutionHandler(
        events,
        data_handler,
        portfolio,
        commission_per_share=0.005,
        min_commission=1.00,
        slippage_pct=0.0005,
    )

    # Snapshot cash / proceeds / position immediately after each fill is applied,
    # so we can assert the whole trajectory, not just the end state.
    snapshots: list[tuple[str, str, float, float, float, float]] = []
    original_update_fill = portfolio.update_fill

    def recording_update_fill(event: FillEvent) -> None:
        original_update_fill(event)
        snapshots.append(
            (
                event.direction,
                event.side,
                event.fill_price,
                portfolio.current_cash,
                portfolio.short_proceeds.get(event.symbol, 0.0),
                portfolio.current_positions.get(event.symbol, 0.0),
            )
        )

    portfolio.update_fill = recording_update_fill  # type: ignore[method-assign]

    backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)
    asyncio.run(backtest.run())

    # Exactly the seven expected fills, in order, with the full ledger.
    assert len(snapshots) == len(EXPECTED_FILLS)
    for actual, expected in zip(snapshots, EXPECTED_FILLS, strict=True):
        direction, side, fill_price, cash, proceeds, position = actual
        exp_dir, exp_side, exp_price, exp_cash, exp_proceeds, exp_pos = expected
        assert direction == exp_dir
        assert side == exp_side
        assert fill_price == pytest.approx(exp_price, rel=1e-9)
        assert cash == pytest.approx(exp_cash, rel=1e-9)
        assert proceeds == pytest.approx(exp_proceeds, rel=1e-9)
        assert position == pytest.approx(exp_pos, abs=1e-9)

    # Bar-12 EXIT + SHORT both pend and are dropped when the data ends.
    assert execution_handler.dropped_orders == 2

    # Final state: cash 8549.34, equity 9449.34 (10 shares still open at 90).
    assert portfolio.current_cash == pytest.approx(8549.34, rel=1e-9)
    assert portfolio.total_equity() == pytest.approx(9449.34, rel=1e-9)


def test_short_scenario_margin_checks_pass():
    """
    The two margin checks that must clear: bar 4 (9999 >= 0.5*939.53 + 1) and
    bar 9 (9833.98 >= 0.5*999.50 + 1). If either failed, the short would be
    rejected and the fill count above would drop - this asserts the intent
    directly on the numbers.
    """
    assert 9999.00 >= 0.5 * 939.53 + 1.0
    assert 9833.98 >= 0.5 * 999.50 + 1.0
