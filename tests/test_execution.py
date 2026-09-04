"""
Tests for the simulated execution handler.

Two themes run through this file:
  - Timing: an order accepted on bar t must not fill on bar t, it must fill at
    bar t+1's open, and if there is no bar t+1 it must not fill at all.
  - Cost: slippage is applied by trade *side* and embedded in
    the fill price; commission is max(commission_per_share × qty, min_commission)
    in total dollars; slippage is reported in total dollars but never charged.
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
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00


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
        commission_per_share=COMMISSION_PER_SHARE,
        min_commission=MIN_COMMISSION,
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
    # BUY pays up: 110 * (1 + 0.0005) = 110.055
    assert fill.fill_price == pytest.approx(110.055)
    # slippage is total dollars: abs(110.055 - 110) * 10 = 0.55
    assert fill.slippage == pytest.approx(0.55)
    # commission = max(0.005 * 10, 1.00) = 1.00 (the floor bites at 10 shares)
    assert fill.commission == pytest.approx(1.00)
    assert fill.side == "BUY"
    # Stamped with the bar it filled on, not the bar it was placed on.
    assert fill.timestamp == bars[1].timestamp


def test_short_fill_receives_less_at_next_open(handler, events, bars):
    handler.execute_order(_order(bars, direction="SHORT"))
    handler.on_market(bars[1])

    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    # SELL receives less: 110 * (1 - 0.0005) = 109.945
    assert fill.fill_price == pytest.approx(109.945)
    assert fill.slippage == pytest.approx(0.55)  # abs(109.945 - 110) * 10
    assert fill.side == "SELL"


def test_exit_of_a_long_sells_and_takes_slippage(handler, events, bars, portfolio):
    """EXIT slippage by side: closing a long is a SELL, so it pays slippage."""
    portfolio.current_positions["TEST"] = 10.0
    handler.execute_order(_order(bars, direction="EXIT", side="SELL"))
    handler.on_market(bars[1])

    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    # 110 * (1 - 0.0005) = 109.945
    assert fill.fill_price == pytest.approx(109.945)
    assert fill.slippage == pytest.approx(0.55)
    assert fill.side == "SELL"


def test_exit_covering_a_short_buys_and_takes_slippage(
    handler, events, bars, portfolio
):
    """Covering a short is a BUY, so it pays slippage the other way."""
    portfolio.current_positions["TEST"] = -10.0
    handler.execute_order(_order(bars, direction="EXIT", side="BUY"))
    handler.on_market(bars[1])

    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    # 110 * (1 + 0.0005) = 110.055
    assert fill.fill_price == pytest.approx(110.055)
    assert fill.slippage == pytest.approx(0.55)
    assert fill.side == "BUY"


def test_commission_hits_the_minimum_on_small_fills(handler, events, bars):
    """10 shares * 0.005 = 0.05, floored to the 1.00 minimum."""
    handler.execute_order(_order(bars, quantity=10.0))
    handler.on_market(bars[1])

    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    assert fill.commission == pytest.approx(1.00)


def test_commission_scales_with_quantity_above_the_minimum(events, bars):
    """500 shares * 0.005 = 2.50, well above the 1.00 floor."""
    portfolio = Portfolio(events, initial_capital=10_000_000.0)
    handler = SimulatedExecutionHandler(
        events,
        InMemoryDataHandler(bars),
        portfolio,
        commission_per_share=COMMISSION_PER_SHARE,
        min_commission=MIN_COMMISSION,
        slippage_pct=SLIPPAGE,
    )
    handler.execute_order(_order(bars, quantity=500.0))
    handler.on_market(bars[1])

    fill = events.popleft()
    assert isinstance(fill, FillEvent)
    assert fill.commission == pytest.approx(2.50)


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
    screen when the signal fired. Slippage is already inside the fill price and
    is not added on top of the requirement.
    """
    # 10 shares at bar 0's close of 100 would cost ~1000 and look affordable.
    # At bar 1's open of 110 the BUY costs
    #   110.055 * 10 + 1.00 (commission) = 1101.55 > 1050
    portfolio.current_cash = 1050.0
    handler = SimulatedExecutionHandler(
        events,
        InMemoryDataHandler(bars),
        portfolio,
        commission_per_share=COMMISSION_PER_SHARE,
        min_commission=MIN_COMMISSION,
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
    handler.execute_order(_order(bars, direction="EXIT", side="SELL"))
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
        commission_per_share=COMMISSION_PER_SHARE,
        min_commission=MIN_COMMISSION,
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
