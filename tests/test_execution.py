"""
Tests for the simulated execution handler.

The theme is timing: an order accepted on bar t must not fill on bar t, it must
fill at bar t+1's open, and if there is no bar t+1 it must not fill at all.
"""

from collections import deque
from typing import Literal

import pytest
from conftest import InMemoryDataHandler, make_bars

from event import Event, FillEvent, OrderEvent, OrderFailedEvent
from execution import SimulatedExecutionHandler
from portfolio import Portfolio

Side = Literal["BUY", "SELL"]

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


def _side_for(direction: str) -> Side:
    # Default side by intent; EXIT defaults to SELL (closing a long), and a
    # short-cover test passes side="BUY" explicitly.
    mapping: dict[str, Side] = {"LONG": "BUY", "SHORT": "SELL", "EXIT": "SELL"}
    return mapping[direction]


def _order(
    bars, direction="LONG", quantity=10.0, bar_index=0, side: Side | None = None
):
    return OrderEvent(
        symbol="TEST",
        timestamp=bars[bar_index].timestamp,
        quantity=quantity,
        direction=direction,
        order_type="MARKET",
        side=side if side is not None else _side_for(direction),
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


def test_pending_order_is_dropped_at_end_of_data(handler, events, bars):
    handler.execute_order(_order(bars, bar_index=1))
    handler.cancel_pending()

    assert handler.dropped_orders == 1
    assert len(events) == 1
    failed = events.popleft()
    assert isinstance(failed, OrderFailedEvent)
    assert failed.reason == "END_OF_DATA"

    # It is dropped, not filled at the last close, and not force-flattened.
    handler.cancel_pending()
    assert handler.dropped_orders == 1


def test_fill_time_cash_rejection(events, bars, portfolio):
    """
    Cash is checked against the actual fill price, not the price that was on
    screen when the signal fired.
    """
    # 10 shares at bar 0's close of 100 would cost ~1000 and look affordable.
    # At bar 1's open of 110 the fill costs
    #   110.055 * 10 + 0.001 + 0.055 = 1100.606 > 1050
    portfolio.current_cash = 1050.0
    handler = SimulatedExecutionHandler(
        events,
        InMemoryDataHandler(bars),
        portfolio,
        fixed_commission=COMMISSION,
        slippage_pct=SLIPPAGE,
    )

    handler.execute_order(_order(bars))
    handler.on_market(bars[1])

    assert len(events) == 1
    failed = events.popleft()
    assert isinstance(failed, OrderFailedEvent)
    assert failed.reason == "INSUFFICIENT_CASH"
    assert failed.quantity == 10.0
    # Rejection is all-or-nothing: no partial fill, nothing charged.
    assert portfolio.current_positions.get("TEST", 0.0) == 0.0
    assert portfolio.current_cash == 1050.0


def test_orders_fill_fifo(handler, events, bars, portfolio):
    portfolio.current_positions["TEST"] = 10.0
    handler.execute_order(_order(bars, direction="EXIT"))
    handler.execute_order(_order(bars, direction="LONG"))
    handler.on_market(bars[1])

    fills = [e for e in events if isinstance(e, FillEvent)]
    assert [f.direction for f in fills] == ["EXIT", "LONG"]


def test_same_close_mode_fills_immediately_at_latest_close(events, bars, portfolio):
    """
    The look-ahead mode kept only for the fill-timing impact comparison. It still
    runs the affordability check, so the only difference from next_open is when
    the fill happens.
    """
    data_handler = InMemoryDataHandler(bars)
    handler = SimulatedExecutionHandler(
        events,
        data_handler,
        portfolio,
        fixed_commission=COMMISSION,
        slippage_pct=SLIPPAGE,
        fill_timing="same_close",
    )
    data_handler.update_bars()  # bar 0 is now the latest bar

    handler.execute_order(_order(bars))

    assert len(events) == 1
    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    # Fills at bar 0's own close of 100 - the look-ahead, reproduced.
    # BUY: 100 * (1 + 0.0005) = 100.05
    assert fill.fill_price == pytest.approx(100.05)


def test_same_close_mode_fails_with_no_price(events, bars, portfolio):
    handler = SimulatedExecutionHandler(
        events,
        InMemoryDataHandler(bars),
        portfolio,
        fill_timing="same_close",
    )
    # No bar has been read, so there is no price to fill against.
    handler.execute_order(_order(bars))

    failed = events.popleft()
    assert isinstance(failed, OrderFailedEvent)
    assert failed.reason == "NO_PRICE"
