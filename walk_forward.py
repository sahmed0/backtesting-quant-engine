"""
Rolling walk-forward analysis.

A single in-sample/out-of-sample split (see overfitting_demo.py) gives just one
held-out number, and that number depends heavily on *where* you happen to cut.
Walk-forward generalises the idea: it slides an (optimise, then test) pair across
the whole history.

For each fold:
  1. Optimise the SimpleMovingAverageStrategy's (short, long) on an in-sample
     (IS) window.
  2. Freeze the winner and run it on the immediately-following out-of-sample
     (OOS) window.
  3. Keep only the OOS result.
Then slide forward by one OOS window and repeat.

Stitching every OOS segment together yields one continuous equity curve in which
*every* point was traded with parameters chosen only from prior data -- a far
more honest estimate of live performance than optimising over all history at
once. The script contrasts that stitched walk-forward result against the naive
"optimise on everything" number to show how much of the latter is illusion.

Run:  python walk_forward.py [SYMBOL]   (default SYMBOL: AAPL)
"""

import os
import queue
import sys
import asyncio

import numpy as np

from data import CSVDataHandler
from strategy import SimpleMovingAverageStrategy
from portfolio import Portfolio
from position_sizing import PercentEquitySizer
from execution import SimulatedExecutionHandler
from engine import Backtest
import performance

# Reuse the slicing-aware helpers and grid from the simple-split demo. Importing
# the module also quiets the execution handler's per-fill logging.
from overfitting_demo import (
    read_timestamps,
    param_grid,
    _fmt_stats,
    DATA_DIR,
    INITIAL_CAPITAL,
)

# Rolling-window geometry, measured in bars. With ~250 trading days per year on
# this daily data: IS ~= 4 years, OOS ~= 1 year, stepping one OOS window at a
# time so the OOS segments tile the timeline without overlap.
IS_WINDOW = 1000
OOS_WINDOW = 250
STEP = OOS_WINDOW


async def _run_async(symbol, short_w, long_w, start, end) -> Portfolio:
    """Runs one backtest over [start, end] and returns the populated portfolio."""
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
    return portfolio


def run_backtest(symbol, short_w, long_w, start, end):
    """Runs a backtest and returns (summary_stats, equity_dataframe)."""
    portfolio = asyncio.run(_run_async(symbol, short_w, long_w, start, end))
    return performance.create_summary_stats(
        portfolio
    ), portfolio.generate_equity_curve()


def optimise(symbol, start, end):
    """
    Grid-searches the parameter space over [start, end] and returns the best
    (params, stats) by Sharpe ratio, or None if nothing produced a usable curve.
    """
    best = None
    for short_w, long_w in param_grid():
        stats, _ = run_backtest(symbol, short_w, long_w, start, end)
        if "error" in stats:
            continue
        if best is None or stats["sharpe_ratio"] > best[1]["sharpe_ratio"]:
            best = ((short_w, long_w), stats)
    return best


def make_folds(n_bars: int):
    """
    Yields (is_start_idx, is_end_idx, oos_start_idx, oos_end_idx) bar indices for
    each rolling fold that fits entirely within ``n_bars``.
    """
    is_start = 0
    while is_start + IS_WINDOW + OOS_WINDOW <= n_bars:
        is_end = is_start + IS_WINDOW - 1
        oos_start = is_end + 1
        oos_end = oos_start + OOS_WINDOW - 1
        yield is_start, is_end, oos_start, oos_end
        is_start += STEP


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Every fold runs a full grid search; the execution handler's per-fill INFO
    # logging would bury the fold table, so quiet it to WARNING.
    logging.getLogger("execution").setLevel(logging.WARNING)

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: data file not found at {csv_path}")
        sys.exit(1)

    timestamps = read_timestamps(csv_path)
    folds = list(make_folds(len(timestamps)))
    if not folds:
        print(
            f"Not enough data for a single fold "
            f"(need >= {IS_WINDOW + OOS_WINDOW} bars, have {len(timestamps)})."
        )
        sys.exit(1)

    def d(idx: int) -> str:
        return timestamps[idx].date().isoformat()

    print("=" * 78)
    print(f"Rolling walk-forward analysis  --  {symbol}")
    print("=" * 78)
    print(
        f"Bars: {len(timestamps)}   IS window: {IS_WINDOW}   "
        f"OOS window: {OOS_WINDOW}   step: {STEP}   folds: {len(folds)}"
    )
    print()
    print(
        "Fold  IS period              OOS period             Params       "
        "IS Shp  OOS Shp"
    )
    print("-" * 78)

    oos_return_chunks = []  # per-bar OOS returns, in chronological order
    wfe_per_fold = []  # OOS return / IS return for each fold
    sharpe_pairs = []  # (IS Sharpe, OOS Sharpe) per fold, OOS usable only
    chosen_params = []  # winning params per fold (to gauge stability)

    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, start=1):
        best = optimise(symbol, timestamps[is_s], timestamps[is_e])
        if best is None:
            print(f"{i:>3}   {d(is_s)}->{d(is_e)}   (no usable in-sample result)")
            continue
        (short_w, long_w), is_stats = best

        oos_stats, oos_equity = run_backtest(
            symbol, short_w, long_w, timestamps[oos_s], timestamps[oos_e]
        )
        oos_sharpe = oos_stats.get("sharpe_ratio", float("nan"))

        print(
            f"{i:>3}   {d(is_s)}->{d(is_e)}   {d(oos_s)}->{d(oos_e)}   "
            f"SMA({short_w:>2},{long_w:>3})  {is_stats['sharpe_ratio']:6.2f}  "
            f"{oos_sharpe:6.2f}"
        )

        chosen_params.append((short_w, long_w))

        # Accumulate this fold's OOS per-bar returns for the stitched curve.
        if not oos_equity.empty and len(oos_equity) > 1:
            returns = oos_equity["total"].pct_change().dropna().to_numpy()
            oos_return_chunks.append(returns)

        is_ret = is_stats["total_return"]
        if "error" not in oos_stats:
            sharpe_pairs.append((is_stats["sharpe_ratio"], oos_sharpe))
            if is_ret != 0:
                wfe_per_fold.append(oos_stats["total_return"] / is_ret)

    if not oos_return_chunks:
        print("\nNo usable out-of-sample segments were produced.")
        sys.exit(1)

    # --- Stitch the OOS segments into one continuous equity curve ---------
    all_returns = np.concatenate(oos_return_chunks)
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + all_returns)
    equity = np.insert(equity, 0, INITIAL_CAPITAL)

    wf_sharpe = performance.calculate_sharpe_ratio(all_returns)
    wf_total_return = equity[-1] / INITIAL_CAPITAL - 1.0
    wf_maxdd = performance.calculate_drawdown(equity)
    avg_wfe = float(np.mean(wfe_per_fold)) if wfe_per_fold else float("nan")

    # Per-fold degradation: how the *same* parameters scored IS vs OOS. This is
    # direction-stable evidence of over-fitting, unlike a single naive number.
    mean_is_sharpe = (
        float(np.mean([p[0] for p in sharpe_pairs])) if sharpe_pairs else float("nan")
    )
    mean_oos_sharpe = (
        float(np.mean([p[1] for p in sharpe_pairs])) if sharpe_pairs else float("nan")
    )
    n_degraded = sum(1 for is_s, oos_s in sharpe_pairs if oos_s < is_s)
    n_distinct = len(set(chosen_params))

    print()
    print("=" * 78)
    print("Walk-forward verdict")
    print("=" * 78)
    print("Per-fold in-sample vs out-of-sample Sharpe (same parameters):")
    print(f"  Mean IS Sharpe:   {mean_is_sharpe:5.2f}")
    print(f"  Mean OOS Sharpe:  {mean_oos_sharpe:5.2f}")
    print(f"  OOS worse than IS in {n_degraded} of {len(sharpe_pairs)} folds.")
    print(
        f"  The optimiser picked {n_distinct} different parameter sets across "
        f"{len(chosen_params)} folds"
    )
    print("  -- an unstable 'best' is itself a sign the edge is being fitted to noise.")
    print()
    print("Stitched out-of-sample equity curve (the realistic, tradeable result):")
    print(
        f"  Sharpe {wf_sharpe:6.2f}   Return {wf_total_return * 100:7.2f}%   "
        f"MaxDD {wf_maxdd * 100:6.2f}%   over {len(all_returns)} OOS bars"
    )

    # Naive benchmark: optimise once over the entire history. Its headline is
    # in-sample (scored on the very data it was tuned on), so it is the number a
    # naive backtest would over-report.
    naive = optimise(symbol, timestamps[0], timestamps[-1])
    if naive is not None:
        (ns, nl), naive_stats = naive
        print(
            f"\nFor contrast, a single naive optimisation over all "
            f"{len(timestamps)} bars  SMA({ns},{nl}):"
        )
        print(f"  {_fmt_stats(naive_stats)}  <- in-sample, optimistic")

    print(
        f"\n  Average per-fold walk-forward efficiency: {avg_wfe:.2f}  "
        f"(OOS return / IS return)"
    )
    print("  ~1.0 means the in-sample edge carried forward; near 0 or negative")
    print("  means each fold's optimisation was largely fitting noise.")
    print("=" * 78)


if __name__ == "__main__":
    main()
