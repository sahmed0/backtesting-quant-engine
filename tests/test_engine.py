"""
Tests for the backtest engine loop.

These drive the real components over a handful of hand-written bars rather than
mocks, because the property under test - that a signal on bar t fills at bar
t+1's open - lives in the interaction between them, which mocks cannot show.
"""

import asyncio
from collections import deque
from datetime import UTC, datetime

import pytest
from conftest import InMemoryDataHandler, make_bars

from engine import Backtest
from event import Event, MarketEvent, SignalEvent
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import FixedSizer
from strategy import SimpleMovingAverageStrategy, Strategy

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


def test_queue_is_fully_drained_at_end_of_run(bars):
    """
    SIGNAL -> ORDER -> FILL is walked within a drain, including the final drain
    after cancel_pending, so nothing is left stranded.
    """
    backtest, _, _, _ = _build(bars, signal_on=len(bars) - 1)
    asyncio.run(backtest.run())

    assert len(backtest.events) == 0


class WarmupDataHandler(InMemoryDataHandler):
    """Replays live bars after handing out a fixed list of warm-up bars."""

    def __init__(self, warmup: list[MarketEvent], live: list[MarketEvent]):
        super().__init__(live)
        self.warmup = warmup

    def warmup_bars(self) -> list[MarketEvent]:
        return list(self.warmup)


class RecordingStrategy(Strategy):
    def __init__(self, events: deque[Event]):
        super().__init__(events)
        self.primed: list[MarketEvent] = []
        self.signalled: list[MarketEvent] = []

    def prime(self, event: MarketEvent) -> None:
        self.primed.append(event)

    def calculate_signals(self, event: MarketEvent) -> None:
        self.signalled.append(event)


class RecordingSizer(FixedSizer):
    def __init__(self, quantity: float = 10.0):
        super().__init__(quantity)
        self.seen: list[float] = []

    def update_market(
        self,
        symbol: str,
        price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        self.seen.append(price)


def test_warmup_bars_prime_strategy_and_sizer_but_are_not_marked(bars):
    warmup = make_bars(
        [(90.0, 91.0, 89.0, 90.0), (91.0, 92.0, 90.0, 91.0), (92.0, 93.0, 91.0, 92.0)],
        start=datetime(2023, 12, 1, tzinfo=UTC),
    )
    events: deque[Event] = deque()
    data_handler = WarmupDataHandler(warmup, bars)
    strategy = RecordingStrategy(events)
    sizer = RecordingSizer()
    portfolio = Portfolio(events, initial_capital=100_000.0, sizer=sizer)
    execution = SimulatedExecutionHandler(events, data_handler, portfolio)
    asyncio.run(Backtest(data_handler, strategy, portfolio, execution, events).run())

    assert strategy.primed == warmup
    assert strategy.signalled == bars
    assert len(portfolio.all_holdings) == len(bars)
    assert sizer.seen == [b.close for b in warmup + bars]


def test_sma_with_warmup_can_signal_on_first_live_bar():
    primed_bars = make_bars([(10.0, 10.0, 10.0, 10.0), (10.0, 10.0, 10.0, 10.0)])
    live = make_bars(
        [(20.0, 20.0, 20.0, 20.0)], start=datetime(2024, 1, 3, tzinfo=UTC)
    )[0]

    events: deque[Event] = deque()
    strategy = SimpleMovingAverageStrategy(events, short_window=2, long_window=3)
    for bar in primed_bars:
        strategy.prime(bar)
    strategy.calculate_signals(live)
    assert list(events) == [SignalEvent("TEST", live.timestamp, "LONG")]

    cold_events: deque[Event] = deque()
    cold = SimpleMovingAverageStrategy(cold_events, short_window=2, long_window=3)
    cold.calculate_signals(live)
    assert len(cold_events) == 0
