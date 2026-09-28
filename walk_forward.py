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
*every* point was traded with parameters chosen only from prior data, a far
more honest estimate of live performance than optimising over all history at
once. The script contrasts that stitched walk-forward result against the naive
"optimise on everything" number. Every run is first given the bars just before
its window to fill its indicators.

Run:  python walk_forward.py [SYMBOL]   (default SYMBOL: AAPL)
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime

import numpy as np

import performance
from overfitting_demo import (
    DATA_DIR,
    SIZING_FRACTION,
    WARMUP_START,
    _fmt_stats,
    cost_line,
    param_grid,
    read_bars_meta,
    run_once,
    run_sma,
)

# Rolling-window geometry, measured in bars. With ~250 trading days per year on
# this daily data: IS ~= 4 years, OOS ~= 1 year, stepping one OOS window at a
# time so the OOS segments tile the timeline without overlap.
IS_WINDOW = 1000
OOS_WINDOW = 250
STEP = OOS_WINDOW
FOLD_OFFSET = WARMUP_START


@dataclass(frozen=True)
class FoldResult:
    is_start: int  # bar indices
    is_end: int
    oos_start: int
    oos_end: int
    params: tuple[int, int]
    is_sharpe: float
    oos_sharpe: float
    is_cagr: float
    oos_cagr: float
    oos_ok: bool  # False when the OOS run returned an "error" stats dict


@dataclass(frozen=True)
class WalkForwardResult:
    offset: int
    folds: list[FoldResult]  # folds whose IS optimisation produced a result
    oos_returns: np.ndarray  # stitched per-bar OOS returns, chronological
    first_oos: int  # bar index of the first OOS bar of the first fold
    last_oos: int  # bar index of the last OOS bar of the last fold


def optimise(
    symbol: str, start: datetime, end: datetime
) -> tuple[tuple[int, int], dict] | None:
    """
    Grid-searches the parameter space over [start, end] and returns the best
    (params, stats) by Sharpe ratio, or None if nothing produced a usable curve.
    """
    best = None
    for short_w, long_w in param_grid():
        stats = run_once(symbol, short_w, long_w, start, end)
        if "error" in stats:
            continue
        if best is None or stats["sharpe_ratio"] > best[1]["sharpe_ratio"]:
            best = ((short_w, long_w), stats)
    return best


def make_folds(n_bars: int, offset: int) -> Iterator[tuple[int, int, int, int]]:
    """
    Yields (is_start_idx, is_end_idx, oos_start_idx, oos_end_idx) bar indices for
    each rolling fold that fits entirely within ``n_bars``, the first IS window
    starting at bar ``offset``.
    """
    is_start = offset
    while is_start + IS_WINDOW + OOS_WINDOW <= n_bars:
        is_end = is_start + IS_WINDOW - 1
        oos_start = is_end + 1
        oos_end = oos_start + OOS_WINDOW - 1
        yield is_start, is_end, oos_start, oos_end
        is_start += STEP


def run_walk_forward(
    symbol: str,
    timestamps: list[datetime],
    offset: int,
    on_fold: Callable[[FoldResult], None] | None = None,
) -> WalkForwardResult:
    """Optimises and tests every fold starting at ``offset`` and stitches the OOS returns."""
    folds: list[FoldResult] = []
    oos_return_chunks: list[np.ndarray] = []

    for is_s, is_e, oos_s, oos_e in make_folds(len(timestamps), offset):
        best = optimise(symbol, timestamps[is_s], timestamps[is_e])
        if best is None:
            continue
        params, is_stats = best

        portfolio = run_sma(symbol, *params, timestamps[oos_s], timestamps[oos_e])
        oos_stats = performance.create_summary_stats(portfolio)
        oos_ok = "error" not in oos_stats

        # Each fold starts flat with fresh capital, so the move from one fold's
        # last close to the next fold's first bar is not in the stitched curve.
        oos_equity = portfolio.generate_equity_curve()
        if not oos_equity.empty and len(oos_equity) > 1:
            returns = oos_equity["total"].pct_change().dropna().to_numpy()
            oos_return_chunks.append(returns)

        fold = FoldResult(
            is_start=is_s,
            is_end=is_e,
            oos_start=oos_s,
            oos_end=oos_e,
            params=params,
            is_sharpe=is_stats["sharpe_ratio"],
            oos_sharpe=oos_stats["sharpe_ratio"] if oos_ok else float("nan"),
            is_cagr=is_stats["cagr"],
            oos_cagr=oos_stats["cagr"] if oos_ok else float("nan"),
            oos_ok=oos_ok,
        )
        folds.append(fold)
        if on_fold is not None:
            on_fold(fold)

    if not oos_return_chunks:
        raise SystemExit("\nNo usable out-of-sample segments were produced.")

    return WalkForwardResult(
        offset=offset,
        folds=folds,
        oos_returns=np.concatenate(oos_return_chunks),
        first_oos=folds[0].oos_start,
        last_oos=folds[-1].oos_end,
    )


def _curve_stats(returns: np.ndarray, ppy: float) -> tuple[float, float, float]:
    """Sharpe, total return and MaxDD of the equity curve compounded from ``returns``."""
    equity = np.insert(np.cumprod(1.0 + returns), 0, 1.0)
    sharpe = performance.calculate_sharpe_ratio(returns, periods=ppy)
    return sharpe, float(equity[-1] - 1.0), performance.calculate_drawdown(equity)


def stitched_stats(result: WalkForwardResult, ppy: float) -> tuple[float, float, float]:
    """Sharpe, total return and MaxDD of the stitched OOS equity curve."""
    return _curve_stats(result.oos_returns, ppy)


def buy_and_hold_stats(
    folds: list[FoldResult], closes_all: np.ndarray, ppy: float
) -> tuple[float, float, float]:
    """
    Sharpe, total return and MaxDD of holding the asset over exactly the bars
    of each fold's OOS window, with the gaps between folds left out.
    """
    chunks = []
    for f in folds:
        closes = closes_all[f.oos_start : f.oos_end + 1]
        chunks.append(np.diff(closes) / closes[:-1])
    return _curve_stats(np.concatenate(chunks), ppy)


def efficiency(folds: list[FoldResult]) -> tuple[float, int, int]:
    """
    Median of OOS CAGR / IS CAGR over the folds whose IS CAGR is positive and
    whose OOS run succeeded. Returns (median, folds used, folds in total).
    """
    ratios = [f.oos_cagr / f.is_cagr for f in folds if f.oos_ok and f.is_cagr > 0]
    median = float(np.median(ratios)) if ratios else float("nan")
    return median, len(ratios), len(folds)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling walk-forward analysis.")
    parser.add_argument("symbol", nargs="?", default="AAPL")
    return parser.parse_args(argv)


def _fmt_curve(sharpe: float, total_return: float, max_dd: float) -> str:
    return (
        f"Sharpe {sharpe:6.2f}   Return {total_return * 100:7.2f}%   "
        f"MaxDD {max_dd * 100:6.2f}%"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Every fold runs a full grid search; the execution handler's per-fill INFO
    # logging and the engine's per-dropped-order INFO logging would bury the
    # fold table, so quiet both to WARNING.
    logging.getLogger("execution").setLevel(logging.WARNING)
    logging.getLogger("engine").setLevel(logging.WARNING)

    args = parse_args(sys.argv[1:])
    symbol = args.symbol
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: data file not found at {csv_path}")
        sys.exit(1)

    timestamps, closes = read_bars_meta(csv_path)
    n_folds = len(list(make_folds(len(timestamps), FOLD_OFFSET)))
    if n_folds == 0:
        print(
            f"Not enough data for a single fold "
            f"(need >= {FOLD_OFFSET + IS_WINDOW + OOS_WINDOW} bars, "
            f"have {len(timestamps)})."
        )
        sys.exit(1)

    def d(idx: int) -> str:
        return timestamps[idx].date().isoformat()

    print("=" * 78)
    print(f"Rolling walk-forward analysis  -  {symbol}")
    print("=" * 78)
    print(
        f"Bars: {len(timestamps)}   IS window: {IS_WINDOW}   "
        f"OOS window: {OOS_WINDOW}   step: {STEP}   "
        f"first IS bar: {FOLD_OFFSET}   folds: {n_folds}"
    )
    print(cost_line())
    print(
        "Every run is first given the bars just before its window "
        "(long window - 1 of them) to fill its moving averages."
    )
    print()
    print(
        "Fold  IS period              OOS period             Params       "
        "IS Shp  OOS Shp"
    )
    print("-" * 78)

    def print_fold(f: FoldResult) -> None:
        i = (f.is_start - FOLD_OFFSET) // STEP + 1
        short_w, long_w = f.params
        print(
            f"{i:>3}   {d(f.is_start)}->{d(f.is_end)}   "
            f"{d(f.oos_start)}->{d(f.oos_end)}   "
            f"SMA({short_w:>2},{long_w:>3})  {f.is_sharpe:6.2f}  "
            f"{f.oos_sharpe:6.2f}"
        )

    result = run_walk_forward(symbol, timestamps, FOLD_OFFSET, on_fold=print_fold)

    # Annualise from the full history's bar density rather than assuming 252.
    ppy = performance.infer_periods_per_year(
        np.array([t.timestamp() for t in timestamps])
    )
    wf_sharpe, wf_total_return, wf_maxdd = stitched_stats(result, ppy)

    # Per-fold degradation: how the same parameters scored IS vs OOS.
    usable = [f for f in result.folds if f.oos_ok]
    mean_is_sharpe = (
        float(np.mean([f.is_sharpe for f in usable])) if usable else float("nan")
    )
    mean_oos_sharpe = (
        float(np.mean([f.oos_sharpe for f in usable])) if usable else float("nan")
    )
    n_degraded = sum(1 for f in usable if f.oos_sharpe < f.is_sharpe)
    n_distinct = len({f.params for f in result.folds})
    wfe, wfe_used, wfe_total = efficiency(result.folds)

    print()
    print("=" * 78)
    print("Walk-forward verdict")
    print("=" * 78)
    print("Per-fold in-sample vs out-of-sample Sharpe (same parameters):")
    print(f"  Mean IS Sharpe:   {mean_is_sharpe:5.2f}")
    print(f"  Mean OOS Sharpe:  {mean_oos_sharpe:5.2f}")
    print(f"  OOS worse than IS in {n_degraded} of {len(usable)} folds.")
    print(
        f"  The optimiser picked {n_distinct} different parameter sets across "
        f"{len(result.folds)} folds."
    )
    print(
        f"  Median walk-forward efficiency: {wfe:.2f}  (annualised OOS return / "
        f"annualised IS return, {wfe_used} of {wfe_total} folds with positive "
        f"IS return)"
    )
    print()
    print(
        f"Stitched out-of-sample equity curve ({len(result.oos_returns)} bars, "
        f"{d(result.first_oos)} -> {d(result.last_oos)}):"
    )
    print(f"  {_fmt_curve(wf_sharpe, wf_total_return, wf_maxdd)}")

    # Stationary-bootstrap 95% CI on the stitched OOS Sharpe: how much of that
    # headline is sampling noise. 1000 resamples in the CLI.
    ci_lo, ci_hi = performance.sharpe_confidence_interval(
        result.oos_returns, ppy, n_resamples=1000
    )
    print(f"  Bootstrap 95% CI on OOS Sharpe: [{ci_lo:.2f}, {ci_hi:.2f}]")

    naive = optimise(symbol, timestamps[0], timestamps[-1])
    bh_sharpe, bh_return, bh_maxdd = buy_and_hold_stats(result.folds, closes, ppy)

    print()
    print("Same span, for comparison:")
    if naive is not None:
        (ns, nl), naive_stats = naive
        span_stats = run_once(
            symbol, ns, nl, timestamps[result.first_oos], timestamps[result.last_oos]
        )
        label = f"Naive SMA({ns},{nl}), chosen on all history"
        if "error" in span_stats:
            print(f"  {label:<45}(no usable result)")
        else:
            print(f"  {label:<45}{_fmt_stats(span_stats)}")
    label = f"Buy and hold {symbol} (100% invested)"
    print(f"  {label:<45}{_fmt_curve(bh_sharpe, bh_return, bh_maxdd)}")
    print(
        f"  Total returns are not comparable: the strategy puts at most "
        f"{SIZING_FRACTION:.0%} of equity in the stock."
    )

    if naive is not None:
        print()
        print(
            f"Naive optimisation over all {len(timestamps)} bars "
            f"(in-sample: scored on the data it was tuned on):"
        )
        print(f"  SMA({ns},{nl})   {_fmt_stats(naive_stats)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
