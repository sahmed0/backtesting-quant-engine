"""
Unit tests for the walk-forward helpers. None of these run a backtest.
"""

import math

import numpy as np
import pytest

from walk_forward import (
    FoldResult,
    buy_and_hold_stats,
    efficiency,
    make_folds,
)


def make_fold(
    is_cagr=0.1, oos_cagr=0.1, oos_ok=True, oos_start=0, oos_end=0
) -> FoldResult:
    return FoldResult(
        is_start=0,
        is_end=0,
        oos_start=oos_start,
        oos_end=oos_end,
        params=(5, 25),
        is_sharpe=1.0,
        oos_sharpe=0.5,
        is_cagr=is_cagr,
        oos_cagr=oos_cagr,
        oos_ok=oos_ok,
    )


def test_make_folds_starts_at_offset_and_tiles():
    assert list(make_folds(1450 + 199, 199)) == [(199, 1198, 1199, 1448)]

    folds = list(make_folds(6599, 199))
    assert len(folds) == 21
    assert folds[0][0] == 199
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert nxt[2] == prev[3] + 1
    assert folds[-1][3] <= 6598


def test_efficiency_uses_median_of_positive_is_folds():
    folds = [
        make_fold(0.2, 0.1, True),
        make_fold(0.1, 0.1, True),
        make_fold(-0.1, 0.05, True),
        make_fold(0.3, 0.3, False),
    ]
    assert efficiency(folds) == (0.75, 2, 4)


def test_efficiency_nan_when_no_usable_folds():
    median, used, total = efficiency([make_fold(-0.1, 0.1), make_fold(0.2, 0.1, False)])
    assert math.isnan(median)
    assert (used, total) == (0, 2)


def test_buy_and_hold_uses_only_fold_bars():
    closes = np.array([10.0, 11.0, 12.0, 100.0, 13.0, 14.0])
    folds = [make_fold(oos_start=0, oos_end=2), make_fold(oos_start=4, oos_end=5)]

    _, total_return, max_dd = buy_and_hold_stats(folds, closes, ppy=252)

    assert total_return == pytest.approx(1.1 * 12 / 11 * 14 / 13 - 1, rel=1e-12)
    assert max_dd == 0.0
