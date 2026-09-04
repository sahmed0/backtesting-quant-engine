"""
Tests for the event core.

Events are immutable value objects: once an event is on the queue, no consumer
may rewrite what another consumer will read.
"""

import dataclasses
from datetime import UTC, datetime

import pytest

from event import FillEvent, MarketEvent, OrderEvent, SignalEvent

TIMESTAMP = datetime(2024, 1, 1, tzinfo=UTC)


def make_market_event() -> MarketEvent:
    return MarketEvent(
        symbol="AAPL",
        timestamp=TIMESTAMP,
        open=99.5,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
    )


def test_market_event_is_frozen():
    event = make_market_event()

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.close = 200.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "event",
    [
        make_market_event(),
        SignalEvent(symbol="AAPL", timestamp=TIMESTAMP, direction="LONG"),
        OrderEvent(
            symbol="AAPL",
            timestamp=TIMESTAMP,
            quantity=10,
            direction="LONG",
            order_type="MARKET",
            side="BUY",
        ),
        FillEvent(
            symbol="AAPL",
            timestamp=TIMESTAMP,
            quantity=10,
            direction="LONG",
            fill_price=100.0,
            commission=1.0,
            slippage=0.05,
            side="BUY",
        ),
    ],
    ids=["market", "signal", "order", "fill"],
)
def test_every_event_is_frozen(event):
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.symbol = "MSFT"


def test_market_event_carries_the_parsed_bar():
    """The bar is parsed once, at the data boundary: floats, not strings."""
    event = make_market_event()

    assert event.open == 99.5
    assert event.high == 101.0
    assert event.low == 99.0
    assert event.close == 100.0
    assert event.volume == 1000.0
    assert event.timestamp == TIMESTAMP


def test_events_have_no_type_tag():
    """
    Dispatch is by concrete type, not a mutable string tag. A lingering `type`
    field would mean two competing sources of truth for what an event is.
    """
    for event_type in (MarketEvent, SignalEvent, OrderEvent, FillEvent):
        field_names = {f.name for f in dataclasses.fields(event_type)}
        assert "type" not in field_names
