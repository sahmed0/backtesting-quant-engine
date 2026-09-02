"""
Tests for the simulated execution handler.

The theme is timing: an order accepted on bar t must not fill on bar t, it must
fill at bar t+1's open, and if there is no bar t+1 it must not fill at all.
"""

from collections import deque

import pytest
from conftest import InMemoryDataHandler, make_bars

from event import Event, FillEvent, OrderEvent, OrderFailedEvent
from execution import SimulatedExecutionHandler
from portfolio import Portfolio

SLIPPAGE = 0.0005
COMMISSION = 0.001


@pytest.fixture
def bars():
    # (open, high, low, close) - bar 1 opens at 110, well clear of bar 0's
    # close of 100, so a fill at the wrong bar is unmistakable.
    return make_bars([(100.0, 101.0, 99.0, 100.0), (110.0, 112.0, 109.0, 111.0)])


@pytest.fixture
def events() -> deque[Event]:
    return deque()


@pytest.fixture
def portfolio(events):
    return Portfolio(events, initial_capital=100_000.0)


@pytest.fixture
def handler(events, bars, portfolio):
    return SimulatedExecutionHandler(
        events,
        InMemoryDataHandler(bars),
        portfolio,
        fixed_commission=COMMISSION,
        slippage_pct=SLIPPAGE,
    )


def _order(bars, direction="LONG", quantity=10.0, bar_index=0):
    return OrderEvent(
        symbol="TEST",
        timestamp=bars[bar_index].timestamp,
        quantity=quantity,
        direction=direction,
        order_type="MARKET",
    )


def test_order_does_not_fill_on_the_bar_it_is_placed(handler, events, bars):
    """The whole point of the change: no same-bar fills."""
    handler.execute_order(_order(bars))

    assert len(events) == 0


def test_order_fills_at_next_bar_open(handler, events, bars):
    handler.execute_order(_order(bars))
    handler.on_market(bars[1])

    assert len(events) == 1
    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    # Bar 1's open is 110, NOT bar 0's close of 100.
    # LONG pays up: 110 * (1 + 0.0005) = 110.055
    assert fill.fill_price == pytest.approx(110.055)
    assert fill.commission == pytest.approx(COMMISSION)
    # Stamped with the bar it filled on, not the bar it was placed on.
    assert fill.timestamp == bars[1].timestamp


def test_short_fill_receives_less_at_next_open(handler, events, bars):
    handler.execute_order(_order(bars, direction="SHORT"))
    handler.on_market(bars[1])

    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    # SHORT receives less: 110 * (1 - 0.0005) = 109.945
    assert fill.fill_price == pytest.approx(109.945)


def test_exit_fills_at_the_open_without_slippage(handler, events, bars, portfolio):
    """EXITs carry no slippage in the current cost model; they still fill at the open."""
    portfolio.current_positions["TEST"] = 10.0
    handler.execute_order(_order(bars, direction="EXIT"))
    handler.on_market(bars[1])

    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    assert fill.fill_price == pytest.approx(110.0)
    assert fill.slippage == pytest.approx(0.0)


def test_second_bar_does_not_refill_a_filled_order(handler, events, bars):
    """Pending orders are cleared once filled, not replayed on every bar."""
    handler.execute_order(_order(bars))
    handler.on_market(bars[1])
    events.clear()

    handler.on_market(bars[1])

    assert len(events) == 0


def test_orders_fill_fifo(handler, events, bars, portfolio):
    portfolio.current_positions["TEST"] = 10.0
    handler.execute_order(_order(bars, direction="EXIT"))
    handler.execute_order(_order(bars, direction="LONG"))
    handler.on_market(bars[1])

    fills = [e for e in events if isinstance(e, FillEvent)]
    assert [f.direction for f in fills] == ["EXIT", "LONG"]
