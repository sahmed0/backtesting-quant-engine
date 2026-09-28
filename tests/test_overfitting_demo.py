"""
Unit tests for the overfitting demo's helpers. None of these run a backtest.
"""

from datetime import UTC, datetime, timedelta

from overfitting_demo import cost_line, split_dates


def test_cost_line_matches_execution_defaults():
    assert cost_line() == (
        "Sizing: 10% of equity per entry. Costs: 5.0 bps slippage, "
        "$0.005/share commission ($1.00 minimum)."
    )


def test_split_dates_starts_in_sample_at_warmup_start():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ts = [start + timedelta(days=i) for i in range(1000)]

    assert split_dates(ts, 0.70) == (ts[199], ts[699], ts[700], ts[999])
