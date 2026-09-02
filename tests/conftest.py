"""
Shared test fixtures.

`InMemoryDataHandler` lets a test drive the real engine over a handful of
hand-written bars, so assertions are about real component behaviour rather than
about mocks agreeing with mocks.
"""

from datetime import UTC, datetime, timedelta

from data import DataHandler
from event import MarketEvent


def make_bars(
    rows: list[tuple[float, float, float, float]],
    symbol: str = "TEST",
    start: datetime | None = None,
) -> list[MarketEvent]:
    """
    Builds a list of daily MarketEvents from (open, high, low, close) tuples,
    one bar per day starting at `start`.

    Args:
        rows: One (open, high, low, close) per bar.
        symbol: The symbol to stamp on every bar.
        start: Timestamp of the first bar; defaults to 2024-01-01 UTC.

    Returns:
        The bars, in chronological order.
    """
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=UTC)

    return [
        MarketEvent(
            symbol=symbol,
            timestamp=start + timedelta(days=i),
            open=o,
            high=h,
            low=low,
            close=c,
            volume=1000.0,
        )
        for i, (o, h, low, c) in enumerate(rows)
    ]


class InMemoryDataHandler(DataHandler):
    """
    Replays a fixed list of MarketEvents, mimicking CSVDataHandler's contract
    without touching the filesystem.
    """

    def __init__(self, bars: list[MarketEvent]):
        self.bars = bars
        self._index = 0
        self.latest_symbol_data: dict[str, MarketEvent] = {}

    def get_latest_bar(self, symbol: str) -> MarketEvent | None:
        return self.latest_symbol_data.get(symbol)

    def update_bars(self) -> MarketEvent | None:
        if self._index >= len(self.bars):
            return None
        bar = self.bars[self._index]
        self._index += 1
        self.latest_symbol_data[bar.symbol] = bar
        return bar
