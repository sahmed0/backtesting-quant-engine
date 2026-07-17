"""
Performance metrics and summary statistics for trading portfolios.
"""

import numpy as np

from portfolio import Portfolio


def calculate_sharpe_ratio(returns: np.ndarray, periods: int = 252) -> float:
    """
    Calculates the annualized Sharpe ratio of a returns stream based on a number of
    trading periods (e.g., 252 for daily data).
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
    strategy_returns: np.ndarray, benchmark_returns: np.ndarray, periods: int = 252
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


def calculate_trade_stats(trades: list) -> tuple[int, float]:
    """
    Pairs each position entry (LONG/SHORT) with the EXIT that closes it to count
    completed round-trip trades and their average duration.

    Returns (number_of_trades, average_duration_in_days). The portfolio always
    flattens a position before reversing, so per symbol entries and exits
    alternate; a position still open at the end of the run is not counted, as it
    has no closing duration.
    """
    num_trades = 0
    durations = []
    open_entry_time: dict[str, float] = {}

    for trade in trades:
        symbol = trade["symbol"]
        direction = trade["direction"]
        timestamp = trade["timestamp"]

        if direction in ("LONG", "SHORT"):
            # Mark the entry time only if we are currently flat in this symbol.
            if symbol not in open_entry_time:
                open_entry_time[symbol] = timestamp
        elif direction == "EXIT":
            if symbol in open_entry_time:
                durations.append(timestamp - open_entry_time.pop(symbol))
                num_trades += 1

    seconds_per_day = 24 * 3600
    avg_duration_days = (
        (sum(durations) / len(durations) / seconds_per_day) if durations else 0.0
    )
    return num_trades, avg_duration_days


def create_summary_stats(portfolio: Portfolio) -> dict:
    """
    Returns a dictionary of: Total Return, Sharpe Ratio, Max Drawdown, Win Rate,
    CAGR, Alpha, and Information Ratio. The last two are measured relative to a
    buy-and-hold benchmark of the traded asset.
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

    # Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(returns)

    # Max Drawdown
    max_drawdown = calculate_drawdown(equity_curve)

    # Win Rate (percentage of periods with positive return)
    if len(returns) > 0:
        win_periods = np.sum(returns > 0)
        win_rate = win_periods / len(returns)
    else:
        win_rate = 0.0

    # CAGR over the actual elapsed time of the run.
    timestamps = df["timestamp"].to_numpy()
    cagr = calculate_cagr(initial_capital, final_equity, timestamps)

    # Calmar ratio: annualized return per unit of worst peak-to-trough loss.
    calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0.0

    # Round-trip trade count and average holding period.
    num_trades, avg_trade_duration = calculate_trade_stats(portfolio.trades)

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

        alpha = calculate_alpha(returns, benchmark_returns)
        # The information ratio is the Sharpe ratio of the active return stream
        # (strategy return minus benchmark return).
        information_ratio = calculate_sharpe_ratio(returns - benchmark_returns)

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
    }
