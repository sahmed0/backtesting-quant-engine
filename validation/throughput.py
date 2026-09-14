"""
Measure the CLI backtest throughput in bars/second.

Measure how fast a full AAPL run processes bars on the CLI (not the browser).

Takes the median of three runs. "Bars" is the number of market bars
the engine marked to market, i.e. the length of the equity curve.

Usage:
  python validation/throughput.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from collections import deque

# The validation scripts live in a subdirectory, so the repo root is not on the
# import path when this file is run as `python validation/throughput.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import CSVDataHandler
from engine import Backtest
from event import Event
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import PercentEquitySizer
from strategy import SimpleMovingAverageStrategy

SYMBOL = "AAPL"
DATA_DIR = "data"
SHORT_WINDOW = 5
LONG_WINDOW = 20
INITIAL_CAPITAL = 100_000.0
N_RUNS = 3


async def _timed_run() -> tuple[int, float]:
    """Runs one full AAPL backtest, returns (bars processed, elapsed seconds)."""
    events: deque[Event] = deque()
    data_handler = CSVDataHandler(events, DATA_DIR, [SYMBOL])
    strategy = SimpleMovingAverageStrategy(
        events, short_window=SHORT_WINDOW, long_window=LONG_WINDOW
    )
    portfolio = Portfolio(
        events, initial_capital=INITIAL_CAPITAL, sizer=PercentEquitySizer(fraction=0.1)
    )
    execution_handler = SimulatedExecutionHandler(events, data_handler, portfolio)
    backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)

    start = time.perf_counter()
    await backtest.run()
    elapsed = time.perf_counter() - start

    bars = len(portfolio.generate_equity_curve())
    return bars, elapsed


def main() -> None:
    csv_path = os.path.join(DATA_DIR, f"{SYMBOL}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: data file not found at {csv_path}")
        sys.exit(1)

    rates = []
    bars = 0
    for i in range(1, N_RUNS + 1):
        bars, elapsed = asyncio.run(_timed_run())
        rate = bars / elapsed if elapsed > 0 else float("inf")
        rates.append(rate)
        print(
            f"Run {i}: {bars} bars in {elapsed * 1000:.1f} ms  -> {rate:,.0f} bars/sec"
        )

    median_rate = statistics.median(rates)
    print("-" * 48)
    print(
        f"{SYMBOL}: {bars} bars, median of {N_RUNS} runs "
        f"-> {median_rate:,.0f} bars/sec (CLI, CPython)"
    )


if __name__ == "__main__":
    main()
