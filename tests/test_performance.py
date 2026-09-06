"""
Unit tests for the performance module. Every public function is covered here.
"""

import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd

from performance import (
    calculate_alpha,
    calculate_cagr,
    calculate_drawdown,
    calculate_sharpe_ratio,
    calculate_trade_stats,
    completed_round_trips,
    create_summary_stats,
    infer_periods_per_year,
)
from position_sizing import FractionalKellySizer

DAY = 86400.0
BASE_TS = 1_600_000_000.0  # arbitrary fixed epoch for deterministic timestamps


def make_trade(symbol, direction, quantity, price, commission=0.0, timestamp=0.0):
    """Builds a trade record matching the shape Portfolio.update_fill records."""
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "direction": direction,
        "side": "BUY" if direction in ("LONG",) else "SELL",
        "quantity": quantity,
        "price": price,
        "commission": commission,
        "slippage": 0.0,
    }


# --- calculate_sharpe_ratio / calculate_drawdown (pre-existing) --------------


def test_calculate_sharpe_ratio():
    # Constant returns, std=0, should return 0.0
    returns = np.array([0.01, 0.01, 0.01, 0.01])
    assert calculate_sharpe_ratio(returns) == 0.0

    # Normal returns
    returns = np.array([0.01, 0.02, -0.01, 0.01, -0.02])
    # mean = 0.002
    # std (ddof=1) = approx 0.01643167
    # sharpe = (0.002 / 0.01643) * sqrt(252) = approx 1.932
    sharpe = calculate_sharpe_ratio(returns, periods=252)
    assert round(sharpe, 3) == 1.932


def test_calculate_drawdown():
    # Strictly increasing, 0% drawdown
    equity = np.array([100.0, 105.0, 110.0, 115.0])
    assert calculate_drawdown(equity) == 0.0

    # Simple peak-to-trough
    equity = np.array([100.0, 110.0, 88.0, 120.0])
    # Peak is 110, trough is 88. Drawdown = (110 - 88) / 110 = 0.2
    assert calculate_drawdown(equity) == 0.2

    # Drawdown doesn't reset until a new peak
    equity = np.array([100.0, 120.0, 96.0, 110.0, 60.0, 150.0])
    # Drawdowns from 120:
    # 96 -> (120-96)/120 = 24/120 = 0.2
    # 110 -> 10/120 = 0.083
    # 60 -> 60/120 = 0.5
    # Max is 0.5
    assert calculate_drawdown(equity) == 0.5


# --- infer_periods_per_year -------------------------------------------------


def test_infer_ppy_daily_equity_is_252():
    # Real daily equity data (weekdays minus market holidays) has ~252 trading
    # days per 365.25-day year; the canonical case the metric targets.
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "AAPL.csv")
    seconds = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seconds.append(datetime.fromisoformat(row["timestamp"]).timestamp())
    ppy = infer_periods_per_year(np.array(seconds))
    assert abs(ppy - 252) <= 3


def test_infer_ppy_seven_day_daily_is_365():
    # A bar every calendar day: 365.25 steps per Julian year, exactly.
    ts = np.array([BASE_TS + i * DAY for i in range(366)])
    assert abs(infer_periods_per_year(ts) - 365.25) < 1e-6


def test_infer_ppy_hourly_24_7():
    # A bar every hour, round the clock: 31_557_600 / 3600 = 8766 per year.
    ts = np.array([BASE_TS + i * 3600.0 for i in range(1000)])
    assert abs(infer_periods_per_year(ts) - 8766.0) < 1e-6


def test_infer_ppy_fallback_when_too_few_bars():
    assert infer_periods_per_year(np.array([])) == 252.0
    assert infer_periods_per_year(np.array([BASE_TS])) == 252.0


def test_infer_ppy_fallback_when_no_elapsed_time():
    # Two identical timestamps span zero seconds -> fall back to 252.
    assert infer_periods_per_year(np.array([BASE_TS, BASE_TS])) == 252.0


# --- calculate_cagr ----------------------------------------------------------


def test_calculate_cagr_two_years():
    # 100k -> 200k over exactly two Julian years: (2)^(1/2) - 1.
    ts = np.array([BASE_TS, BASE_TS + 2 * 365.25 * DAY])
    cagr = calculate_cagr(100_000.0, 200_000.0, ts)
    assert abs(cagr - (2.0**0.5 - 1.0)) < 1e-9


def test_calculate_cagr_guards():
    ts = np.array([BASE_TS, BASE_TS + 365.25 * DAY])
    assert calculate_cagr(0.0, 100.0, ts) == 0.0  # non-positive initial
    assert calculate_cagr(100.0, 0.0, ts) == 0.0  # non-positive final
    assert calculate_cagr(100.0, 200.0, np.array([BASE_TS])) == 0.0  # <2 points


# --- calculate_alpha ---------------------------------------------------------


def test_calculate_alpha_hand_computed():
    # Four aligned periods.
    s = np.array([0.02, 0.01, 0.03, -0.01])
    b = np.array([0.01, 0.005, 0.02, -0.005])
    # mean_s = 0.0125, mean_b = 0.0075.
    # cov(s, b) ddof=1 = 5.25e-4 / 3 = 1.75e-4.
    # var(b)   ddof=1 = 3.25e-4 / 3 = 1.0833...e-4.
    # beta = 1.75e-4 / 1.0833e-4 = 5.25/3.25 = 1.6153846.
    # alpha_per_period = 0.0125 - 1.6153846 * 0.0075 = 3.846154e-4.
    # annualised (x252) = 0.09692308.
    alpha = calculate_alpha(s, b, periods=252)
    assert abs(alpha - 0.09692308) < 1e-6


def test_calculate_alpha_guards():
    # Too few points, or a flat benchmark (zero variance), yield 0.
    assert calculate_alpha(np.array([0.01]), np.array([0.01])) == 0.0
    s = np.array([0.01, 0.02, 0.03])
    flat = np.array([0.01, 0.01, 0.01])
    assert calculate_alpha(s, flat) == 0.0


# --- completed_round_trips ---------------------------------------------------


def test_round_trips_long_win_and_loss():
    trades = [
        make_trade("AAPL", "LONG", 10, 100.0, commission=1.0),
        make_trade("AAPL", "EXIT", 10, 110.0, commission=1.0),
        make_trade("AAPL", "LONG", 10, 100.0),
        make_trade("AAPL", "EXIT", 10, 90.0),
    ]
    trips = completed_round_trips(trades)
    assert len(trips) == 2

    # Win: (110-100)*10 - 1 - 1 = 98; 98 / (100*10) = 0.098.
    assert trips[0]["direction"] == "LONG"
    assert abs(trips[0]["net_pnl"] - 98.0) < 1e-12
    assert abs(trips[0]["net_return"] - 0.098) < 1e-12
    # Loss: (90-100)*10 = -100; -100 / 1000 = -0.1.
    assert abs(trips[1]["net_pnl"] + 100.0) < 1e-12
    assert abs(trips[1]["net_return"] + 0.1) < 1e-12


def test_round_trips_short_win_sign_flip():
    trades = [
        make_trade("AAPL", "SHORT", 10, 100.0),
        make_trade("AAPL", "EXIT", 10, 90.0),
    ]
    trips = completed_round_trips(trades)
    assert len(trips) == 1
    # Short profits when price falls: gross flips to +100.
    assert trips[0]["direction"] == "SHORT"
    assert abs(trips[0]["net_pnl"] - 100.0) < 1e-12
    assert abs(trips[0]["net_return"] - 0.1) < 1e-12


def test_round_trips_unclosed_entry_ignored():
    trades = [
        make_trade("AAPL", "LONG", 10, 100.0),
        make_trade("AAPL", "EXIT", 10, 110.0),
        make_trade("AAPL", "LONG", 10, 105.0),  # still open at end of run
    ]
    trips = completed_round_trips(trades)
    assert len(trips) == 1
    assert trips[0]["exit_price"] == 110.0


def test_round_trips_interleaved_symbols_paired_independently():
    trades = [
        make_trade("AAPL", "LONG", 10, 100.0),
        make_trade("MSFT", "SHORT", 5, 200.0),
        make_trade("AAPL", "EXIT", 10, 110.0),  # closes AAPL long
        make_trade("MSFT", "EXIT", 5, 190.0),  # covers MSFT short
    ]
    trips = completed_round_trips(trades)
    by_symbol = {t["symbol"]: t for t in trips}
    assert set(by_symbol) == {"AAPL", "MSFT"}
    assert abs(by_symbol["AAPL"]["net_pnl"] - 100.0) < 1e-12
    # MSFT short: (190-200)*5 flipped -> +50.
    assert abs(by_symbol["MSFT"]["net_pnl"] - 50.0) < 1e-12


def test_round_trips_skips_zero_notional():
    trades = [
        make_trade("AAPL", "LONG", 10, 0.0),  # zero entry price -> skip
        make_trade("AAPL", "EXIT", 10, 110.0),
    ]
    assert completed_round_trips(trades) == []


# --- calculate_trade_stats ---------------------------------------------------


def test_calculate_trade_stats_count_and_duration():
    trades = [
        make_trade("AAPL", "LONG", 10, 100.0, timestamp=BASE_TS),
        make_trade("AAPL", "EXIT", 10, 110.0, timestamp=BASE_TS + 2 * DAY),
        make_trade("AAPL", "LONG", 10, 110.0, timestamp=BASE_TS + 3 * DAY),
        make_trade("AAPL", "EXIT", 10, 120.0, timestamp=BASE_TS + 7 * DAY),
    ]
    num_trades, avg_duration_days = calculate_trade_stats(trades)
    assert num_trades == 2
    # Durations: 2 days and 4 days -> average 3.0 days.
    assert abs(avg_duration_days - 3.0) < 1e-12


def test_calculate_trade_stats_empty():
    assert calculate_trade_stats([]) == (0, 0.0)


# --- Kelly delegation regression --------------------------------------------


def test_kelly_completed_returns_uses_shared_helper():
    # Same case as the position-sizing suite: commission-only cost, 0.098 net.
    trades = [
        make_trade("AAPL", "LONG", 10, 100.0, commission=1.0),
        make_trade("AAPL", "EXIT", 10, 110.0, commission=1.0),
    ]
    returns = FractionalKellySizer._completed_returns(trades, "AAPL")
    assert len(returns) == 1
    assert abs(returns[0] - 0.098) < 1e-9


# --- create_summary_stats end-to-end ----------------------------------------


class _StubPortfolio:
    """Minimal duck type: create_summary_stats only reads these three members."""

    def __init__(self, df, trades, initial_capital):
        self._df = df
        self.trades = trades
        self.initial_capital = initial_capital

    def generate_equity_curve(self):
        return self._df


def _daily_curve(totals, prices):
    n = len(totals)
    return pd.DataFrame(
        {
            "timestamp": [BASE_TS + i * DAY for i in range(n)],
            "total": totals,
            "price": prices,
        }
    )


def test_create_summary_stats_end_to_end():
    df = _daily_curve(
        totals=[100_000.0, 101_000.0, 100_500.0, 102_000.0],
        prices=[100.0, 101.0, 100.5, 102.0],
    )
    # Three completed round trips, two of them winners -> win rate 2/3.
    trades = [
        make_trade("AAPL", "LONG", 10, 100.0),
        make_trade("AAPL", "EXIT", 10, 110.0),  # win
        make_trade("AAPL", "LONG", 10, 110.0),
        make_trade("AAPL", "EXIT", 10, 120.0),  # win
        make_trade("AAPL", "LONG", 10, 120.0),
        make_trade("AAPL", "EXIT", 10, 115.0),  # loss
    ]
    # _StubPortfolio is a deliberate duck type: create_summary_stats reads only
    # three members of it.
    stats = create_summary_stats(_StubPortfolio(df, trades, 100_000.0))

    assert "error" not in stats
    assert abs(stats["total_return"] - 0.02) < 1e-12
    assert stats["num_trades"] == 3
    # Win Rate (trades): profitable trips / completed trips = 2 / 3.
    assert abs(stats["win_rate"] - 2.0 / 3.0) < 1e-12
    # Daily bars -> 365.25 periods/year (7-day density in this synthetic curve).
    assert abs(stats["periods_per_year"] - 365.25) < 1e-6
    # Benchmark-relative and risk-adjusted metrics are present and finite.
    for key in ("sharpe_ratio", "alpha", "information_ratio", "calmar_ratio", "cagr"):
        assert key in stats
        assert np.isfinite(stats[key])


def test_create_summary_stats_no_trades_gives_zero_win_rate():
    df = _daily_curve(
        totals=[100_000.0, 100_500.0, 101_000.0],
        prices=[100.0, 100.5, 101.0],
    )
    stats = create_summary_stats(_StubPortfolio(df, [], 100_000.0))
    assert stats["win_rate"] == 0.0
    assert stats["num_trades"] == 0


def test_create_summary_stats_empty_and_insufficient():
    empty = _StubPortfolio(pd.DataFrame(), [], 100_000.0)
    assert "error" in create_summary_stats(empty)

    one_row = _StubPortfolio(
        _daily_curve(totals=[100_000.0], prices=[100.0]), [], 100_000.0
    )
    assert "error" in create_summary_stats(one_row)
