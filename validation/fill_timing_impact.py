"""
Measure how much of the backtest's performance comes from fill timing.

Runs the same AAPL backtest three times, changing only when orders fill:

  * ``same_close`` - fills at the close the signal was computed from (mild look-ahead).
  * ``same_open``  - fills at the signal bar's own open, before the close that
    produced the signal was known (severe look-ahead).
  * ``next_open``  - fills at the next bar's open (the engine's default).

Everything else (data, strategy, position sizing, costs) is identical between
the runs, so any difference in results comes from fill timing alone.

Usage:
  python validation/fill_timing_impact.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import deque

# The validation scripts live in a subdirectory, so the repo root is not on the
# import path when this file is run as `python validation/fill_timing_impact.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import performance
from data import CSVDataHandler
from engine import Backtest
from event import Event
from execution import FillTiming, SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import PercentEquitySizer
from strategy import SimpleMovingAverageStrategy

SYMBOL = "AAPL"
DATA_DIR = "data"
SHORT_WINDOW = 5
LONG_WINDOW = 20
INITIAL_CAPITAL = 100_000.0

MODES: list[tuple[FillTiming, str, str]] = [
    ("same_close", "same close", "(mild)"),
    ("same_open", "same open", "(severe)"),
    ("next_open", "next open", "(honest)"),
]
WIDTHS = [10, 13, 13]


async def _run(fill_timing: FillTiming) -> dict:
    """Runs one AAPL backtest under the given fill timing, returns its stats."""
    events: deque[Event] = deque()
    data_handler = CSVDataHandler(DATA_DIR, [SYMBOL])
    strategy = SimpleMovingAverageStrategy(
        events, short_window=SHORT_WINDOW, long_window=LONG_WINDOW
    )
    portfolio = Portfolio(
        events, initial_capital=INITIAL_CAPITAL, sizer=PercentEquitySizer(fraction=0.1)
    )
    execution_handler = SimulatedExecutionHandler(
        events, data_handler, portfolio, fill_timing=fill_timing
    )
    backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)
    await backtest.run()
    return performance.create_summary_stats(portfolio)


def main() -> None:
    csv_path = os.path.join(DATA_DIR, f"{SYMBOL}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: data file not found at {csv_path}")
        sys.exit(1)

    results = []
    for mode, _, _ in MODES:
        stats = asyncio.run(_run(mode))
        if "error" in stats:
            print(f"Error in {mode} run: {stats['error']}")
            sys.exit(1)
        results.append(stats)

    print("=" * 72)
    print(
        f"Fill-timing impact - {SYMBOL}, SMA({SHORT_WINDOW},{LONG_WINDOW}), "
        f"PercentEquity(0.1)"
    )
    print("=" * 72)
    header = f"{'Metric':<18}"
    notes = " " * 18
    for (_, name, note), width in zip(MODES, WIDTHS, strict=True):
        header += f"{name:>{width}}"
        notes += " " * (width - len(name)) + f"{note:<{len(name)}}"
    print(header)
    print(notes.rstrip())
    print("-" * 72)

    def row(label: str, key: str, fmt: str) -> None:
        cells = (
            format(r[key], f">{w}{fmt}") for r, w in zip(results, WIDTHS, strict=True)
        )
        print(f"{label:<18}" + "".join(cells))

    row("Sharpe", "sharpe_ratio", ".2f")
    row("Total Return", "total_return", ".2%")
    row("Max Drawdown", "max_drawdown", ".2%")
    row("# Trades", "num_trades", "d")
    print("=" * 72)


if __name__ == "__main__":
    main()
