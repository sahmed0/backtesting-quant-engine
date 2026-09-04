"""
Tests for the backtest engine loop.

These drive the real components over a handful of hand-written bars rather than
mocks, because the property under test - that a signal on bar t fills at bar
t+1's open - lives in the interaction between them, which mocks cannot show.
"""

import asyncio
from collections import deque

import pytest
from conftest import InMemoryDataHandler, make_bars

from engine import Backtest
from event import Event, MarketEvent, SignalEvent
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import FixedSizer
from strategy import Strategy

SLIPPAGE = 0.0005


class SignalOnBarStrategy(Strategy):
    """Emits one LONG signal on a chosen bar index, then nothing."""

    def __init__(self, events: deque[Event], signal_on: int = 1):
        super().__init__(events)
        self.signal_on = signal_on
        self.seen = 0
        self.closes_at_signal: list[float] = []

    def calculate_signals(self, event: MarketEvent) -> None:
        if self.seen == self.signal_on:
            self.closes_at_signal.append(event.close)
            self.events.append(
                SignalEvent(
                    symbol=event.symbol, timestamp=event.timestamp, direction="LONG"
                )
            )
        self.seen += 1


@pytest.fixture
def bars():
    # (open, high, low, close). Opens deliberately differ from the prior close
    # so a fill price identifies which bar & field it came from.
    return make_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (102.0, 103.0, 101.0, 102.0),
            (110.0, 112.0, 109.0, 111.0),
            (120.0, 121.0, 119.0, 120.0),
            (130.0, 131.0, 129.0, 130.0),
        ]
    )


def _build(bars, signal_on=1, capital=100_000.0):
    events: deque[Event] = deque()
    data_handler = InMemoryDataHandler(bars)
    strategy = SignalOnBarStrategy(events, signal_on=signal_on)
    portfolio = Portfolio(events, initial_capital=capital, sizer=FixedSizer(10.0))
    execution = SimulatedExecutionHandler(
        events,
        data_handler,
        portfolio,
        commission_per_share=0.0,
        min_commission=0.0,
        slippage_pct=SLIPPAGE,
    )
    backtest = Backtest(data_handler, strategy, portfolio, execution, events)
    return backtest, portfolio, execution, strategy


def test_engine_initialization(bars):
    events: deque[Event] = deque()
    data_handler = InMemoryDataHandler(bars)
    portfolio = Portfolio(events)
    strategy = SignalOnBarStrategy(events)
    execution = SimulatedExecutionHandler(events, data_handler, portfolio)

    backtest = Backtest(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution,
        events=events,
    )

    assert backtest.data_handler is data_handler
    assert backtest.strategy is strategy
    assert backtest.portfolio is portfolio
    assert backtest.execution_handler is execution
    assert backtest.events is events


def test_signal_on_bar_t_fills_at_bar_t_plus_1_open(bars):
    """The headline invariant: a signal on bar t fills at bar t+1's open."""
    backtest, portfolio, _, strategy = _build(bars, signal_on=1)
    asyncio.run(backtest.run())

    assert len(portfolio.trades) == 1
    trade = portfolio.trades[0]

    # The signal was evaluated off bar 1's close of 102...
    assert strategy.closes_at_signal == [102.0]
    # ...and filled at bar 2's OPEN of 110, not bar 1's close of 102 and not
    # bar 2's close of 111: 110 * (1 + 0.0005) = 110.055
    assert trade["price"] == pytest.approx(110.055)
    assert trade["timestamp"] == bars[2].timestamp.timestamp()


def test_signal_on_final_bar_is_dropped_not_filled(bars):
    backtest, portfolio, execution, _ = _build(bars, signal_on=len(bars) - 1)
    asyncio.run(backtest.run())

    assert portfolio.trades == []
    assert execution.dropped_orders == 1
    assert portfolio.current_positions.get("TEST", 0.0) == 0.0


def test_equity_curve_has_one_point_per_bar(bars):
    backtest, portfolio, _, _ = _build(bars, signal_on=1)
    asyncio.run(backtest.run())

    assert len(portfolio.all_holdings) == len(bars)


def test_mark_to_market_reflects_the_fill_on_its_own_bar(bars):
    """
    update_timeindex runs after on_market, so the bar the fill lands on already
    values the position.
    """
    backtest, portfolio, _, _ = _build(bars, signal_on=1)
    asyncio.run(backtest.run())

    # Fill is on bar 2; holdings[2] must already show 10 shares at bar 2's
    # close of 111 = 1110.0
    assert portfolio.all_holdings[2]["TEST"] == pytest.approx(1110.0)
    assert portfolio.all_holdings[1]["TEST"] == pytest.approx(0.0)


def test_progress_callback_reports_bars(bars):
    backtest, _, _, _ = _build(bars, signal_on=1)
    seen: list[int] = []

    asyncio.run(backtest.run(progress_cb=seen.append, yield_every=2))

    # 5 bars, yielding every 2 -> callbacks after bars 2 and 4.
    assert seen == [2, 4]


def test_queued_market_event_is_rejected(bars):
    """Bars reach the engine as a return value; queueing one is a bug."""
    backtest, _, _, _ = _build(bars, signal_on=1)
    backtest.events.append(bars[0])

    with pytest.raises(RuntimeError, match="MarketEvent must not be queued"):
        asyncio.run(backtest.run())


def test_queue_is_fully_drained_at_end_of_run(bars):
    """
    SIGNAL -> ORDER -> FILL is walked within a drain, including the final drain
    after cancel_pending, so nothing is left stranded.
    """
    backtest, _, _, _ = _build(bars, signal_on=len(bars) - 1)
    asyncio.run(backtest.run())

    assert len(backtest.events) == 0
