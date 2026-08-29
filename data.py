"""
Data handler module for the backtesting engine.
"""

import csv
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from event import Event, MarketEvent


class DataHandler(ABC):
    """
    Abstract base class for data handlers.
    Provides an interface for fetching historical or live market data.
    """

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> dict[str, Any]:
        """
        Retrieves the most recent bar data for a given symbol to prevent lookahead bias.
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
    Data handler for high-performance CSV processing.
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
        Initialises the data handler with the event queue and symbols.

        Args:
            events: The shared event queue.
            csv_dir: Directory holding one ``<symbol>.csv`` per symbol.
            symbols: Symbols to stream.
            start_date: When set, bars dated before this (tz-aware, UTC) are
                skipped. Used to restrict a run to an in-sample or out-of-sample
                window without copying the underlying CSV.
            end_date: When set, bars dated after this are skipped. The bound is
                inclusive on both ends.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

        self.continue_backtest: bool = True
        self.symbol_data: dict[str, Iterator[dict[str, Any]]] = {}
        self.latest_symbol_data: dict[str, dict[str, Any]] = {}

        self._load_data()

    def _load_data(self) -> None:
        """
        Prepares the data generators to stream rows without overwhelming memory immediately.
        """
        for symbol in self.symbols:
            file_path = os.path.join(self.csv_dir, f"{symbol}.csv")

            # Create a generator function to keep the file open
            # only while we are actually reading from it.
            def make_row_generator(path):
                with open(path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Apply the optional in-/out-of-sample date filter. Only
                        # parse the timestamp when a bound is actually set.
                        if self.start_date is not None or self.end_date is not None:
                            ts = datetime.fromisoformat(row["timestamp"]).replace(
                                tzinfo=UTC
                            )
                            if self.start_date is not None and ts < self.start_date:
                                continue
                            if self.end_date is not None and ts > self.end_date:
                                continue
                        yield row

            # Create a streaming generator for each symbol
            self.symbol_data[symbol] = make_row_generator(file_path)
            self.latest_symbol_data[symbol] = {}

    def get_latest_bar(self, symbol: str) -> dict[str, Any]:
        """
        Returns the last fetched bar for the specified symbol.
        """
        return self.latest_symbol_data.get(symbol, {})

    def update_bars(self) -> None:
        """
        Fetches the next row for all symbols and triggers a market event to drive the simulation.
        """
        for symbol in self.symbols:
            try:
                row = next(self.symbol_data[symbol])
                self.latest_symbol_data[symbol] = row

                timestamp_raw = row["timestamp"]

                # Adapts to string or native datetime parsing
                if isinstance(timestamp_raw, str):
                    timestamp = datetime.fromisoformat(timestamp_raw).replace(
                        tzinfo=UTC
                    )
                else:
                    timestamp = timestamp_raw.replace(tzinfo=UTC)

                event = MarketEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    close=float(row["close"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    volume=float(row["volume"]),
                )
                self.events.append(event)

            except StopIteration:
                self.continue_backtest = False
