"""
Measure how often each mean-reversion check passes on 60-bar windows.

The OU strategy only opens a trade on a window that looks mean-reverting. This
script applies the candidate checks to two sets of windows and prints the pass
rates side by side:

  * random walks, which have no mean reversion, so any pass is a false alarm;
  * every rolling 60-bar window of AAPL's log close.

A useful check passes close to its nominal rate on random walks (5% for the 5%
Dickey-Fuller test). The sign check and the half-life rule pass most of them.

Usage:
  python validation/ou_gate_check.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections.abc import Callable

import numpy as np

# The validation scripts live in a subdirectory, so the repo root is not on the
# import path when this file is run as `python validation/ou_gate_check.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from strategies.ou_strategy import OUCalibration, calibrate_ou

WINDOW = 60
N_WALKS = 5000
STEP_SD = 0.02
CSV_PATH = os.path.join(ROOT, "data", "AAPL.csv")


def random_walk_windows() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        math.log(100) + np.cumsum(rng.normal(0, STEP_SD, WINDOW))
        for _ in range(N_WALKS)
    ]


def aapl_windows() -> list[np.ndarray]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        closes = [float(row["close"]) for row in csv.DictReader(f)]
    log_close = np.log(np.array(closes))
    return [log_close[i : i + WINDOW] for i in range(len(log_close) - WINDOW + 1)]


def pass_rate(
    windows: list[np.ndarray],
    level: float | None,
    check: Callable[[OUCalibration], bool],
) -> float:
    return float(np.mean([check(calibrate_ou(w, level)) for w in windows]))


def main() -> None:
    walks = random_walk_windows()
    aapl = aapl_windows()

    rows: list[tuple[str, float | None, Callable[[OUCalibration], bool]]] = [
        ("Sign only (0 < phi < 1)", None, lambda fit: fit.mean_reverting),
        # Shown because it is a commonly suggested check, and it still passes
        # most random walks.
        (
            "Half-life < 30 bars",
            None,
            lambda fit: fit.mean_reverting and fit.half_life < 30,
        ),
        ("Dickey-Fuller 10%", 0.10, lambda fit: fit.passed),
        ("Dickey-Fuller 5%", 0.05, lambda fit: fit.passed),
        ("Dickey-Fuller 1%", 0.01, lambda fit: fit.passed),
    ]

    title = f"Mean-reversion check pass rates on {WINDOW}-bar windows of log prices"
    print(title)
    print("-" * len(title))
    walks_head = f"Random walks ({N_WALKS})"
    aapl_head = f"AAPL ({len(aapl)} windows)"
    print(f"{'Check':<30} {walks_head:<21} {aapl_head}")
    for label, level, check in rows:
        walk_pct = f"{100 * pass_rate(walks, level, check):.1f}%"
        aapl_pct = f"{100 * pass_rate(aapl, level, check):.1f}%"
        print(f"{label:<30} {walk_pct:>9}{'':<13}{aapl_pct:>9}")


if __name__ == "__main__":
    main()
