"""
Data handler module for the backtesting engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Any, List
from queue import Queue
from datetime import datetime, timezone
import os
import csv
# import polars as pl # deprecating in favor of streaming CSV processing with built-in csv module
from event import MarketEvent

class DataHandler(ABC):
    """
    Abstract base class for data handlers.
    Provides an interface for fetching historical or live market data.
    """

    @abstractmethod
    def getLatestBar(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieves the most recent bar data for a given symbol to prevent lookahead bias.
        """
        pass

    @abstractmethod
    def updateBars(self) -> None:
        """
        Pushes the latest bar to the events queue to drive the event loop.
        """
        pass

class CSVDataHandler(DataHandler):
    """
    Data handler for high-performance CSV processing.
    """

    def __init__(self, eventsQueue: Queue, csvDir: str, symbolList: List[str]):
        """
        Initialises the data handler with the event queue and symbols.
        """
        self.eventsQueue = eventsQueue
        self.csvDir = csvDir
        self.symbolList = symbolList
        
        self.shouldContinueBacktest: bool = True
        self.symbolData: Dict[str, Iterator[Dict[str, Any]]] = {}
        self.latestSymbolData: Dict[str, Dict[str, Any]] = {}

        self._loadData()

    """
    # POLARS VERSION - DEPRECATED IN FAVOR OF STREAMING CSV PROCESSING WITH BUILT-IN CSV MODULE
    def _loadData(self) -> None:
        # Prepares the data generators to stream rows without overwhelming memory immediately.
        for symbol in self.symbolList:
            filePath = os.path.join(self.csvDir, f"{symbol}.csv")
            
            # Using lazy loading for optimising query plan before materialisation
            # We then collect and convert to a row iterator for row-by-row streaming
            lazyDf = pl.scan_csv(filePath)
            df = lazyDf.collect()
            
            self.symbolData[symbol] = df.iter_rows(named=True)
            self.latestSymbolData[symbol] = {}
            """

    def _loadData(self) -> None:
        """
        Prepares the data generators to stream rows without overwhelming memory immediately.
        """
        for symbol in self.symbolList:
            filePath = os.path.join(self.csvDir, f"{symbol}.csv")
            
            # Create a generator function to keep the file open 
            # only while we are actually reading from it.
            def make_row_generator(path):
                with open(path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        yield row
            
            # Create a streaming generator for each symbol
            self.symbolData[symbol] = make_row_generator(filePath)
            self.latestSymbolData[symbol] = {}

    def getLatestBar(self, symbol: str) -> Dict[str, Any]:
        """
        Returns the last fetched bar for the specified symbol.
        """
        return self.latestSymbolData.get(symbol, {})

    def updateBars(self) -> None:
        """
        Fetches the next row for all symbols and triggers a market event to drive the simulation.
        """
        for symbol in self.symbolList:
            try:
                row = next(self.symbolData[symbol])
                self.latestSymbolData[symbol] = row
                
                timestampRaw = row['timestamp']
                
                # Adapts to string or native datetime parsing
                if isinstance(timestampRaw, str):
                    timestamp = datetime.fromisoformat(timestampRaw).replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestampRaw.replace(tzinfo=timezone.utc)
                
                event = MarketEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    close=float(row['close']),
                    high=float(row['high']),
                    low=float(row['low']),
                    volume=float(row['volume'])
                )
                self.eventsQueue.put(event)
                
            except StopIteration:
                self.shouldContinueBacktest = False
