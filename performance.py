"""
Performance metrics and summary statistics for trading portfolios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Imported for type hints only. Importing at runtime would be circular:
    # portfolio -> position_sizing -> performance -> portfolio.
    from portfolio import Portfolio

# One (Julian) year in seconds: 365.25 days. Used to translate an elapsed
# wall-clock span into a periods-per-year annualisation factor.
SECONDS_PER_YEAR = 31_557_600


def infer_periods_per_year(timestamps: np.ndarray) -> float:
    """
    Infers the number of bars per year from the actual bar timestamps, so
    annualisation matches the data's real density instead of assuming 252.

    ``timestamps`` are unix seconds. With ``n`` bars spanning ``elapsed``
    seconds there are ``n - 1`` inter-bar steps, so::

        periods_per_year = (n - 1) / (elapsed / SECONDS_PER_YEAR)

    Falls back to 252.0 when there are fewer than two bars or the span is
    non-positive. Never round the result inside further maths; round only for
    display.
    """
    n = len(timestamps)
    if n < 2:
        return 252.0

    elapsed = float(timestamps[-1] - timestamps[0])
    if elapsed <= 0:
        return 252.0

    return (n - 1) / (elapsed / SECONDS_PER_YEAR)


def calculate_sharpe_ratio(returns: np.ndarray, periods: float = 252) -> float:
    """
    Calculates the annualized Sharpe ratio of a returns stream based on a number of
    trading periods per year (e.g., 252 for daily data, or inferred from the data).
    """
    if len(returns) == 0:
        return 0.0

    stdev = np.std(returns, ddof=1)
    if stdev == 0.0 or np.isnan(stdev):
        return 0.0

    mean_return = np.mean(returns)
    sharpe = (mean_return / stdev) * np.sqrt(periods)
    return float(sharpe)


def calculate_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculates the maximum peak-to-trough decline as a percentage.
    """
    if len(equity_curve) == 0:
        return 0.0

    # Calculate the cumulative maximum peak
    high_water_mark = np.maximum.accumulate(equity_curve)

    # Calculate drawdowns from the high water mark
    # Suppress warnings for division by zero if high_water_mark has zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = (high_water_mark - equity_curve) / high_water_mark
        drawdowns[np.isnan(drawdowns)] = 0.0
        drawdowns[np.isinf(drawdowns)] = 0.0

    return float(np.max(drawdowns))


def calculate_cagr(
    initial_equity: float, final_equity: float, timestamps: np.ndarray
) -> float:
    """
    Calculates the Compound Annual Growth Rate from the equity at the start and
    end of the run, using the actual elapsed wall-clock time spanned by the
    (unix-second) timestamps.
    """
    if initial_equity <= 0 or final_equity <= 0 or len(timestamps) < 2:
        return 0.0

    seconds_per_year = 365.25 * 24 * 3600
    years = (timestamps[-1] - timestamps[0]) / seconds_per_year
    if years <= 0:
        return 0.0

    return float((final_equity / initial_equity) ** (1.0 / years) - 1.0)


def calculate_alpha(
    strategy_returns: np.ndarray, benchmark_returns: np.ndarray, periods: float = 252
) -> float:
    """
    Calculates the annualized Jensen's alpha: the strategy's average return in
    excess of what its market exposure (beta to the benchmark) would predict.
    """
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0

    benchmark_variance = np.var(benchmark_returns, ddof=1)
    if benchmark_variance == 0.0 or np.isnan(benchmark_variance):
        return 0.0

    beta = (
        np.cov(strategy_returns, benchmark_returns, ddof=1)[0, 1] / benchmark_variance
    )
    alpha_per_period = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
    # Annualize the per-period alpha to match the convention used elsewhere.
    return float(alpha_per_period * periods)


def completed_round_trips(trades: list[dict]) -> list[dict]:
    """
    Pairs each position entry (LONG/SHORT) with the EXIT that closes it, per
    symbol, and returns one dict per completed round trip.

    The portfolio always flattens a position before reversing, so within a
    symbol entries and exits strictly alternate; an entry still open at the end
    of the run has no closing EXIT and is dropped. This is the single source of
    round-trip pairing shared by the trade stats here and the Kelly sizer.

    Each round trip is::

        {
            "symbol", "direction",        # entry direction: "LONG" or "SHORT"
            "entry_price", "exit_price",
            "quantity", "entry_ts", "exit_ts",
            "net_pnl",     # gross P&L minus commission on both legs
            "net_return",  # net_pnl / (entry_price * quantity)
        }

    ``net_pnl`` is ``(exit_price - entry_price) * quantity`` for a long and the
    negative of that for a short, less the entry and exit commissions. Slippage
    is *not* subtracted: it is already embedded in the recorded fill prices,
    so charging it again would double-count it.
    """
    trips: list[dict] = []
    open_entries: dict[str, dict] = {}

    for trade in trades:
        symbol = trade["symbol"]
        direction = trade["direction"]

        if direction in ("LONG", "SHORT"):
            # Record the entry only while flat in this symbol; a second entry
            # without an intervening EXIT should not happen (flatten-before-
            # reverse), and if it did we keep the first.
            if symbol not in open_entries:
                open_entries[symbol] = trade
        elif direction == "EXIT":
            entry = open_entries.pop(symbol, None)
            if entry is None:
                continue
            entry_price = entry["price"]
            quantity = entry["quantity"]
            notional = entry_price * quantity
            if notional <= 0:
                continue
            gross = (trade["price"] - entry_price) * quantity
            if entry["direction"] == "SHORT":
                gross = -gross
            net_pnl = gross - entry["commission"] - trade["commission"]
            trips.append(
                {
                    "symbol": symbol,
                    "direction": entry["direction"],
                    "entry_price": entry_price,
                    "exit_price": trade["price"],
                    "quantity": quantity,
                    "entry_ts": entry["timestamp"],
                    "exit_ts": trade["timestamp"],
                    "net_pnl": net_pnl,
                    "net_return": net_pnl / notional,
                }
            )

    return trips


def calculate_trade_stats(trades: list[dict]) -> tuple[int, float]:
    """
    Returns (number_of_completed_round_trips, average_duration_in_days) from the
    shared round-trip pairing. A position still open at the end of the run is
    not counted, as it has no closing duration.
    """
    trips = completed_round_trips(trades)
    num_trades = len(trips)

    seconds_per_day = 24 * 3600
    durations = [t["exit_ts"] - t["entry_ts"] for t in trips]
    avg_duration_days = (
        (sum(durations) / len(durations) / seconds_per_day) if durations else 0.0
    )
    return num_trades, avg_duration_days


def create_summary_stats(portfolio: Portfolio) -> dict:
    """
    Returns a dictionary of: Total Return, Sharpe Ratio, Max Drawdown, Win Rate,
    CAGR, Alpha, Information Ratio, and the periods-per-year used to annualise.
    Alpha and IR are measured relative to a buy-and-hold benchmark of the traded
    asset.
    """
    df = portfolio.generate_equity_curve()

    if df.empty:
        return {"error": "Portfolio is empty. No performance stats to calculate."}

    equity_curve = df["total"].to_numpy()

    if len(equity_curve) < 2:
        return {
            "error": "Insufficient data points in portfolio to calculate performance stats."
        }

    initial_capital = portfolio.initial_capital
    final_equity = equity_curve[-1]

    # Total Return
    total_return = (final_equity / initial_capital) - 1.0

    # Calculate returns
    returns_series = df["total"].pct_change().dropna()
    returns = returns_series.to_numpy()

    # Annualisation factor inferred from the bar timestamps, not a fixed
    # 252, so the metrics are correct for any bar density (daily, hourly, 24/7).
    timestamps = df["timestamp"].to_numpy()
    periods_per_year = infer_periods_per_year(timestamps)

    # Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(returns, periods=periods_per_year)

    # Max Drawdown
    max_drawdown = calculate_drawdown(equity_curve)

    # Win Rate (trades): fraction of completed round trips that were profitable
    # net of commissions. The old per-bar hit rate is dropped.
    trips = completed_round_trips(portfolio.trades)
    if trips:
        wins = sum(1 for t in trips if t["net_pnl"] > 0)
        win_rate = wins / len(trips)
    else:
        win_rate = 0.0

    # CAGR over the actual elapsed time of the run.
    cagr = calculate_cagr(initial_capital, final_equity, timestamps)

    # Calmar ratio: annualized return per unit of worst peak-to-trough loss.
    calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0.0

    # Round-trip trade count and average holding period (from the same pairing).
    num_trades = len(trips)
    seconds_per_day = 24 * 3600
    durations = [t["exit_ts"] - t["entry_ts"] for t in trips]
    avg_trade_duration = (
        (sum(durations) / len(durations) / seconds_per_day) if durations else 0.0
    )

    # Benchmark-relative metrics (buy-and-hold of the underlying asset). These
    # require the per-bar asset price, which is only present when the equity
    # curve carries it.
    alpha = 0.0
    information_ratio = 0.0
    if "price" in df.columns and len(returns) > 0:
        prices = df["price"].to_numpy()
        # Per-period buy-and-hold returns, aligned to the strategy returns
        # (both start one bar in, so the lengths match).
        benchmark_returns = np.diff(prices) / prices[:-1]

        alpha = calculate_alpha(returns, benchmark_returns, periods=periods_per_year)
        # The information ratio is the Sharpe ratio of the active return stream
        # (strategy return minus benchmark return).
        information_ratio = calculate_sharpe_ratio(
            returns - benchmark_returns, periods=periods_per_year
        )

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "cagr": cagr,
        "alpha": alpha,
        "information_ratio": information_ratio,
        "calmar_ratio": calmar_ratio,
        "num_trades": num_trades,
        "avg_trade_duration": avg_trade_duration,
        "periods_per_year": periods_per_year,
    }
