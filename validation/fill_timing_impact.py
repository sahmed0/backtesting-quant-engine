"""
Show how much of the backtest's returns disappear once trades fill honestly.

The engine's key rule is that a trade signaled on a bar's close can't also
fill at that same close - that would be looking into the future. Instead it
fills at the next bar's open. Letting trades fill on the signal bar's own
close was the biggest reason the old engine looked better than it should
have. This script measures that effect directly: it runs the same AAPL
backtest twice, changing only when fills happen, and shows how much of the
headline Sharpe ratio and return comes from that unrealistic timing.

  * ``same_close`` - fills right away, on the same close that triggered the
    signal. This is unrealistic (it's look-ahead bias) and only exists here
    for comparison.
  * ``next_open``  - fills at the next bar's open.

Everything else (data, strategy, position sizing, costs) is identical
between the two runs, so any difference in results comes from fill timing
alone.

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


async def _run(fill_timing: FillTiming) -> dict:
    """Runs one AAPL backtest under the given fill timing, returns its stats."""
    events: deque[Event] = deque()
    data_handler = CSVDataHandler(events, DATA_DIR, [SYMBOL])
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

    same_close = asyncio.run(_run("same_close"))
    next_open = asyncio.run(_run("next_open"))
    for name, stats in (("same_close", same_close), ("next_open", next_open)):
        if "error" in stats:
            print(f"Error in {name} run: {stats['error']}")
            sys.exit(1)

    print("=" * 72)
    print(
        f"Fill-timing impact - {SYMBOL}, SMA({SHORT_WINDOW},{LONG_WINDOW}), "
        f"PercentEquity(0.1)"
    )
    print("=" * 72)
    print("Same-bar fills overstate performance by transacting at a price the")
    print("strategy had already seen. next_open fills at the following bar's open.")
    print()

    header = f"{'Metric':<18}{'same_close':>14}{'next_open':>14}{'delta':>14}"
    print(header)
    print("-" * len(header))

    def row(label: str, sc: float, no: float, pct: bool) -> None:
        if pct:
            print(
                f"{label:<18}{sc * 100:>13.2f}%{no * 100:>13.2f}%"
                f"{(no - sc) * 100:>13.2f}%"
            )
        else:
            print(f"{label:<18}{sc:>14.2f}{no:>14.2f}{no - sc:>14.2f}")

    row("Sharpe", same_close["sharpe_ratio"], next_open["sharpe_ratio"], pct=False)
    row("Total Return", same_close["total_return"], next_open["total_return"], pct=True)
    row("Max Drawdown", same_close["max_drawdown"], next_open["max_drawdown"], pct=True)
    print(
        f"{'# Trades':<18}{same_close['num_trades']:>14d}"
        f"{next_open['num_trades']:>14d}"
        f"{next_open['num_trades'] - same_close['num_trades']:>14d}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
