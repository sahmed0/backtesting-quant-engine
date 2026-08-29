"""
Data handler module for the backtesting engine.
"""

import csv
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime

from event import Event, MarketEvent

# Columns every symbol CSV must provide. `open` is required because orders fill
# at the next bar's open; a file without it cannot be simulated honestly.
REQUIRED_COLUMNS = frozenset({"timestamp", "open", "high", "low", "close", "volume"})


class DataHandler(ABC):
    """
    Abstract base class for data handlers.
    Provides an interface for fetching historical or live market data.
    """

    continue_backtest: bool = True

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> MarketEvent | None:
        """
        Retrieves the most recent bar for a given symbol to prevent lookahead
        bias. Returns None before the first bar has been read.
        """
        pass

    @abstractmethod
    def update_bars(self) -> None:
        """
        Pushes the latest bar to the events queue to drive the event loop.
        """
        pass


class CSVDataHandler(DataHandler):
    """
    Streams bars for a single symbol from a CSV file.

    Each row is parsed exactly once, at construction of the MarketEvent, and
    every consumer reads the typed event rather than re-parsing strings.
    """

    def __init__(
        self,
        events: deque[Event],
        csv_dir: str,
        symbols: list[str],
        start_date: "datetime | None" = None,
        end_date: "datetime | None" = None,
    ):
        """
        Initialises the data handler with the event queue and symbol.

        Args:
            events: The shared event queue.
            csv_dir: Directory holding one ``<symbol>.csv`` per symbol.
            symbols: The symbol to stream, as a single-element list.
            start_date: When set, bars dated before this (tz-aware, UTC) are
                skipped. Used to restrict a run to an in-sample or out-of-sample
                window without copying the underlying CSV.
            end_date: When set, bars dated after this are skipped. The bound is
                inclusive on both ends.

        Raises:
            ValueError: If more or fewer than one symbol is given, or if a CSV
                is missing required columns.
        """
        if len(symbols) != 1:
            raise ValueError("CSVDataHandler supports exactly one symbol")

        self.events = events
        self.csv_dir = csv_dir
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

        self.continue_backtest: bool = True
        self.symbol_data: dict[str, Iterator[MarketEvent]] = {}
        self.latest_symbol_data: dict[str, MarketEvent | None] = {}

        self._load_data()

    def _load_data(self) -> None:
        """
        Prepares the data generators to stream rows without overwhelming memory immediately.
        """
        for symbol in self.symbols:
            file_path = os.path.join(self.csv_dir, f"{symbol}.csv")
            self._validate_header(file_path, symbol)

            # Create a generator function to keep the file open
            # only while we are actually reading from it.
            def make_bar_generator(path: str, sym: str) -> Iterator[MarketEvent]:
                with open(path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        timestamp = datetime.fromisoformat(row["timestamp"]).replace(
                            tzinfo=UTC
                        )

                        # Apply the optional in-/out-of-sample date filter.
                        if self.start_date is not None and timestamp < self.start_date:
                            continue
                        if self.end_date is not None and timestamp > self.end_date:
                            continue

                        yield MarketEvent(
                            symbol=sym,
                            timestamp=timestamp,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                        )

            # Create a streaming generator for each symbol
            self.symbol_data[symbol] = make_bar_generator(file_path, symbol)
            self.latest_symbol_data[symbol] = None

    def _validate_header(self, file_path: str, symbol: str) -> None:
        """
        Fails fast if the CSV cannot supply a complete bar. Checked eagerly at
        construction so the error surfaces before a run starts, rather than as a
        KeyError midway through the event loop.
        """
        with open(file_path, encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []

        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(
                f"{symbol}.csv is missing required columns: {sorted(missing)}"
            )

    def get_latest_bar(self, symbol: str) -> MarketEvent | None:
        """
        Returns the last fetched bar for the specified symbol, or None before
        the first bar has been read.
        """
        return self.latest_symbol_data.get(symbol)

    def update_bars(self) -> None:
        """
        Fetches the next row for all symbols and triggers a market event to drive the simulation.
        """
        for symbol in self.symbols:
            try:
                bar = next(self.symbol_data[symbol])
                self.latest_symbol_data[symbol] = bar
                self.events.append(bar)

            except StopIteration:
                self.continue_backtest = False
