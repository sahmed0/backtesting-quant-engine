"""
Demonstrating curve over-fitting with an in-sample / out-of-sample split.

A strategy's parameters can be tuned to look brilliant on one fixed slice of
history simply by fitting that slice's *noise*. This script makes that effect
visible and measurable:

1.  Split a symbol's price history chronologically into an in-sample (IS)
    period and a held-out out-of-sample (OOS) period.
2.  Grid-search the SimpleMovingAverageStrategy's (short_window, long_window)
    over the IS period and pick the best parameters by Sharpe ratio. This is
    the "optimisation" a naive researcher would do.
3.  Run those frozen IS-best parameters once on the OOS period.

The gap between the IS and OOS results is the over-fitting "tax". As a second,
sharper illustration the script also evaluates every parameter set on the OOS
period (something you could only do with hindsight) to show that the IS winner
is usually *not* the OOS winner -- the ranking does not survive contact with
unseen data.

Run:  python overfitting_demo.py [SYMBOL]   (default SYMBOL: AAPL)
"""

import csv
import os
import queue
import sys
import asyncio
import logging
import itertools
from datetime import datetime, timezone

# The execution handler logs every fill at INFO. Across a whole grid search
# that is thousands of lines of noise, so quiet it to WARNING for the demo.
logging.getLogger("execution").setLevel(logging.WARNING)

from data import CSVDataHandler
from strategy import SimpleMovingAverageStrategy
from portfolio import Portfolio
from position_sizing import PercentEquitySizer
from execution import SimulatedExecutionHandler
from engine import Backtest
import performance

# Parameter grid to search. Only pairs with short < long are valid crossovers.
SHORT_WINDOWS = [5, 10, 15, 20]
LONG_WINDOWS = [25, 50, 100, 200]

# Fraction of the (chronological) history reserved for in-sample optimisation.
IS_FRACTION = 0.70

DATA_DIR = "data"
INITIAL_CAPITAL = 100000.0


def read_timestamps(csv_path: str) -> list[datetime]:
    """Reads the (tz-aware, UTC) timestamp of every bar, in file order."""
    timestamps: list[datetime] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timestamps.append(
                datetime.fromisoformat(row["timestamp"]).replace(tzinfo=timezone.utc)
            )
    return timestamps


def split_dates(timestamps: list[datetime], fraction: float):
    """
    Splits the timeline so the first ``fraction`` of bars is in-sample and the
    remainder is out-of-sample, with no overlapping bar between the two.

    Returns (is_start, is_end, oos_start, oos_end).
    """
    split_idx = int(len(timestamps) * fraction)
    is_start, is_end = timestamps[0], timestamps[split_idx - 1]
    oos_start, oos_end = timestamps[split_idx], timestamps[-1]
    return is_start, is_end, oos_start, oos_end


async def _run_async(symbol, short_w, long_w, start, end) -> dict:
    """Runs a single backtest over [start, end] and returns its summary stats."""
    events: queue.Queue = queue.Queue()
    data_handler = CSVDataHandler(
        events, DATA_DIR, [symbol], start_date=start, end_date=end
    )
    strategy = SimpleMovingAverageStrategy(
        events, short_window=short_w, long_window=long_w
    )
    portfolio = Portfolio(
        events, initial_capital=INITIAL_CAPITAL, sizer=PercentEquitySizer(fraction=0.1)
    )
    execution = SimulatedExecutionHandler(events, data_handler)
    backtest = Backtest(data_handler, strategy, portfolio, execution, events)
    await backtest.run()
    return performance.create_summary_stats(portfolio)


def run_once(symbol, short_w, long_w, start, end) -> dict:
    """Synchronous wrapper around a single backtest run."""
    return asyncio.run(_run_async(symbol, short_w, long_w, start, end))


def param_grid():
    """Yields every valid (short_window, long_window) pair (short < long)."""
    for short_w, long_w in itertools.product(SHORT_WINDOWS, LONG_WINDOWS):
        if short_w < long_w:
            yield short_w, long_w


def evaluate_grid(symbol, start, end) -> dict:
    """
    Runs every parameter pair over [start, end].

    Returns {(short, long): stats}. Combinations that produce no usable equity
    curve (e.g. the long window never warms up on a short slice) are skipped.
    """
    results = {}
    for short_w, long_w in param_grid():
        stats = run_once(symbol, short_w, long_w, start, end)
        if "error" not in stats:
            results[(short_w, long_w)] = stats
    return results


def _fmt_stats(stats: dict) -> str:
    """One-line metric summary for a stats dict."""
    return (
        f"Sharpe {stats['sharpe_ratio']:6.2f}   "
        f"Return {stats['total_return'] * 100:7.2f}%   "
        f"MaxDD {stats['max_drawdown'] * 100:6.2f}%   "
        f"Trades {stats['num_trades']:>3}"
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The execution handler logs every fill at INFO. Across a whole grid search
    # that is thousands of lines of noise, so quiet it to WARNING for the demo.
    logging.getLogger("execution").setLevel(logging.WARNING)

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: data file not found at {csv_path}")
        sys.exit(1)

    timestamps = read_timestamps(csv_path)
    is_start, is_end, oos_start, oos_end = split_dates(timestamps, IS_FRACTION)

    def d(dt: datetime) -> str:
        return dt.date().isoformat()

    print("=" * 72)
    print(f"Curve over-fitting demonstration  --  {symbol}")
    print("=" * 72)
    print(
        f"In-sample (optimise here):  {d(is_start)} -> {d(is_end)}  "
        f"({int(len(timestamps) * IS_FRACTION)} bars)"
    )
    print(
        f"Out-of-sample (held out):   {d(oos_start)} -> {d(oos_end)}  "
        f"({len(timestamps) - int(len(timestamps) * IS_FRACTION)} bars)"
    )
    print()

    # --- 1. Optimise on the in-sample period -----------------------------
    is_results = evaluate_grid(symbol, is_start, is_end)
    if not is_results:
        print("No parameter combination produced a usable in-sample result.")
        sys.exit(1)

    is_ranked = sorted(
        is_results.items(), key=lambda kv: kv[1]["sharpe_ratio"], reverse=True
    )

    print("In-sample grid search, ranked by Sharpe (top 5):")
    print("-" * 72)
    for (short_w, long_w), stats in is_ranked[:5]:
        print(f"  SMA({short_w:>2},{long_w:>3})   {_fmt_stats(stats)}")
    print()

    best_params = is_ranked[0][0]
    best_is_stats = is_ranked[0][1]

    # --- 2. Evaluate every set on the held-out period --------------------
    oos_results = evaluate_grid(symbol, oos_start, oos_end)

    # The IS winner, now run on data it never saw.
    oos_for_is_best = oos_results.get(best_params)

    print("=" * 72)
    print("The over-fitting tax")
    print("=" * 72)
    short_w, long_w = best_params
    print(f"Best in-sample parameters:  SMA({short_w}, {long_w})")
    print(f"  In-sample:      {_fmt_stats(best_is_stats)}")
    if oos_for_is_best is not None:
        print(f"  Out-of-sample:  {_fmt_stats(oos_for_is_best)}")

        is_sharpe = best_is_stats["sharpe_ratio"]
        oos_sharpe = oos_for_is_best["sharpe_ratio"]
        is_ret = best_is_stats["total_return"]
        oos_ret = oos_for_is_best["total_return"]
        # Walk-forward efficiency: how much of the IS edge survived out of
        # sample. ~1.0 is robust; near 0 or negative means the edge was noise.
        wfe = (oos_ret / is_ret) if is_ret != 0 else float("nan")
        print()
        print(f"  Sharpe degradation:        {is_sharpe:.2f} -> {oos_sharpe:.2f}")
        print(
            f"  Return degradation:        {is_ret * 100:.2f}% -> {oos_ret * 100:.2f}%"
        )
        print(f"  Walk-forward efficiency:   {wfe:.2f}  (OOS return / IS return)")
    else:
        print("  Out-of-sample:  (no usable result for these parameters)")
    print()

    # --- 3. Hindsight: would the IS winner have won out of sample? --------
    if oos_results:
        oos_ranked = sorted(
            oos_results.items(), key=lambda kv: kv[1]["sharpe_ratio"], reverse=True
        )
        oos_order = [params for params, _ in oos_ranked]
        rank = oos_order.index(best_params) + 1 if best_params in oos_order else None

        hindsight_best = oos_ranked[0]
        print("=" * 72)
        print("Did the in-sample winner survive? (hindsight view)")
        print("=" * 72)
        if rank is not None:
            print(
                f"  In-sample best SMA({short_w},{long_w}) ranks #{rank} of "
                f"{len(oos_ranked)} out of sample."
            )
        hb_params, hb_stats = hindsight_best
        print(
            f"  Actual out-of-sample best: SMA({hb_params[0]},{hb_params[1]})   "
            f"{_fmt_stats(hb_stats)}"
        )
        print()
        print("  If the IS ranking were robust, the IS winner would rank near #1")
        print("  out of sample too. The further down it sits, the more the")
        print("  optimisation fit noise rather than a repeatable edge.")
    print("=" * 72)


if __name__ == "__main__":
    main()
