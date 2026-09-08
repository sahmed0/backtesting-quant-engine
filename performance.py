"""
Performance metrics and summary statistics for trading portfolios.
"""

from __future__ import annotations

import math
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


# --- Deflated Sharpe & bootstrap CIs -----------------------------------------
#
# All Sharpe inputs to the Deflated Sharpe maths are PER-PERIOD (non-annualised)
# and ``kurt`` is raw kurtosis (Normal = 3). Everything here is stdlib + numpy so
# it runs inside Pyodide without pulling scipy into the browser payload.

# Euler-Mascheroni constant, used to weight the expected maximum of N iid
# Sharpe estimates (Bailey & Lopez de Prado 2014).
EULER_MASCHERONI = 0.5772156649015329


def normal_cdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """
    Standard-normal inverse CDF (quantile) by bisection on ``normal_cdf`` over
    [-10, 10], 80 iterations - deterministic and free of magic constants.
    Raises ``ValueError`` for ``p`` outside the open interval (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("normal_ppf requires 0 < p < 1")

    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """
    Expected maximum of ``n_trials`` iid Sharpe estimates whose variance is
    ``sr_variance`` (Bailey & Lopez de Prado 2014, Appendix D)::

        SR0 = sqrt(V) * [(1 - gamma) * Phi^-1(1 - 1/N)
                         + gamma * Phi^-1(1 - 1/(N*e))]

    with gamma the Euler-Mascheroni constant. Returns 0.0 when there are fewer
    than two trials or the variance is non-positive.
    """
    if n_trials < 2 or sr_variance <= 0.0:
        return 0.0

    n = float(n_trials)
    gamma = EULER_MASCHERONI
    term = (1.0 - gamma) * normal_ppf(1.0 - 1.0 / n) + gamma * normal_ppf(
        1.0 - 1.0 / (n * math.e)
    )
    return math.sqrt(sr_variance) * term


def deflated_sharpe_ratio(
    sr_observed: float,
    sr_variance: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt: float,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the probability the
    observed per-period Sharpe is genuinely positive after correcting for having
    selected the best of ``n_trials`` configurations.

    All Sharpe inputs are per-period (non-annualised); ``kurt`` is raw kurtosis
    (Normal = 3). With ``sr0 = expected_max_sharpe(sr_variance, n_trials)``::

        DSR = Phi( (sr_obs - sr0) * sqrt(n_obs - 1)
                   / sqrt(1 - skew*sr_obs + ((kurt - 1)/4) * sr_obs^2) )

    Returns 0.0 when the variance radicand is non-positive or ``n_obs < 2``.
    """
    if n_obs < 2:
        return 0.0

    sr0 = expected_max_sharpe(sr_variance, n_trials)
    radicand = 1.0 - skew * sr_observed + ((kurt - 1.0) / 4.0) * sr_observed**2
    if radicand <= 0.0:
        return 0.0

    z = (sr_observed - sr0) * math.sqrt(n_obs - 1) / math.sqrt(radicand)
    return normal_cdf(z)


def returns_moments(returns: np.ndarray) -> tuple[float, float]:
    """
    Population skewness ``mean((r-mu)^3)/sigma^3`` and raw kurtosis
    ``mean((r-mu)^4)/sigma^4`` (sigma the population std) of a returns stream,
    for feeding the Deflated Sharpe. Returns ``(0.0, 3.0)`` - the Normal
    reference - when the sample has fewer than two points or zero variance.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0, 3.0

    sigma = float(np.std(r))
    if sigma == 0.0:
        return 0.0, 3.0

    centered = r - np.mean(r)
    skew = float(np.mean(centered**3) / sigma**3)
    kurt = float(np.mean(centered**4) / sigma**4)
    return skew, kurt


def bootstrap_sharpe_samples(
    returns: np.ndarray,
    periods_per_year: float,
    n_resamples: int,
    rng: np.random.Generator,
    mean_block: int | None = None,
) -> list[float]:
    """
    Annualised Sharpe ratios of ``n_resamples`` stationary-bootstrap resamples of
    ``returns`` (Politis-Romano). Serial dependence is preserved by drawing
    variable-length blocks: the expected block length is ``L = mean_block or
    max(2, round(n ** (1/3)))``; each step continues the current block with
    probability ``1 - 1/L`` (wrapping ``(i+1) mod n``), otherwise jumps to a
    fresh uniform start. Each resample has the same length as ``returns``.

    The passed ``rng`` is advanced in place, so a caller can invoke this in
    batches (yielding to the event loop between calls) and get one continuous,
    reproducible stream of resamples.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    samples: list[float] = []
    if n < 2:
        return samples

    length = mean_block if mean_block is not None else max(2, round(n ** (1.0 / 3.0)))
    p_jump = 1.0 / length
    sqrt_ppy = math.sqrt(periods_per_year)

    for _ in range(n_resamples):
        idx = np.empty(n, dtype=np.int64)
        i = int(rng.integers(0, n))
        for k in range(n):
            idx[k] = i
            if rng.random() < p_jump:
                i = int(rng.integers(0, n))
            else:
                i = (i + 1) % n
        resample = r[idx]
        stdev = float(np.std(resample, ddof=1))
        if stdev == 0.0:
            samples.append(0.0)
        else:
            samples.append(float(np.mean(resample) / stdev * sqrt_ppy))
    return samples


def sharpe_confidence_interval(
    returns: np.ndarray,
    periods_per_year: float,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval (default 95%) for the annualised Sharpe ratio,
    via the stationary bootstrap. Deterministic for a given ``seed``. Returns
    ``(0.0, 0.0)`` when there are fewer than 10 returns.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 10:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    samples = bootstrap_sharpe_samples(r, periods_per_year, n_resamples, rng)
    if not samples:
        return 0.0, 0.0

    tail = (1.0 - ci) / 2.0 * 100.0
    lo = float(np.percentile(samples, tail))
    hi = float(np.percentile(samples, 100.0 - tail))
    return lo, hi
