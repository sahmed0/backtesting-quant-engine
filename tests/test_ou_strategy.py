import math
import warnings
from collections import deque
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from numpy.random import default_rng

from event import Event, MarketEvent, SignalEvent
from strategies.ou_strategy import (
    OrnsteinUhlenbeckStrategy,
    OUCalibration,
    calibrate_ou,
    dickey_fuller_critical_value,
    gate_summary,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def base_strategy():
    """Returns a strategy instance with a small window for fast testing."""
    return OrnsteinUhlenbeckStrategy(
        symbol="AAPL", window_size=10, entry_z=2.0, gate_level=None
    )


def fixed_fit(
    mu: float, sigma_eq: float, mean_reverting: bool = True, passed: bool = True
) -> OUCalibration:
    """A fit with the given equilibrium, for testing the signal logic alone."""
    return OUCalibration(
        mu, sigma_eq, 0.1, math.log(2) / 0.1, -3.0, mean_reverting, passed
    )


def create_market_event(price: float) -> MarketEvent:
    """Helper to quickly mock market data ticks."""
    return MarketEvent(
        symbol="AAPL",
        timestamp=T0,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=100,
    )


def feed(strategy, price):
    """
    Feeds one bar and returns the signal it emitted, or None.

    The strategy appends signals to its queue rather than returning them (it
    conforms to Strategy.calculate_signals -> None), so a bar that emits
    nothing simply leaves the queue untouched.
    """
    before = len(strategy.events)
    strategy.calculate_signals(create_market_event(price))
    if len(strategy.events) == before:
        return None
    return strategy.events[-1]


def test_strategy_warmup(base_strategy):
    """Ensure no signals are generated while the rolling window is filling."""
    # Feed it 9 prices (window size is 10)
    for i in range(9):
        assert feed(base_strategy, 100.0 + i) is None


def test_trending_rejection(base_strategy):
    """
    If a stock is purely trending up, the OU process should realize
    it is NOT mean-reverting (theta <= 0) and refuse to trade.

    The model fits log-prices, so the non-mean-reverting series is a
    *geometric* trend (constant log-returns => a straight line in log space).
    A linear price ramp is concave in log space and would read as mean-reverting.
    """
    # Geometric trend: 100, 110, 121, ... -> log-prices are linear -> theta ~ 0.
    for i in range(10):
        feed(base_strategy, 100.0 * 1.1**i)

    fit = base_strategy._calibrate()

    # A straight line in log space is not mean-reverting
    assert fit.mean_reverting is False
    assert fit.passed is False


def test_mean_reversion_signals(monkeypatch):
    """
    Feed the strategy a stable baseline, then artificially spike the price.
    With shorting enabled it should generate a SHORT signal expecting a return
    to the mean.
    """
    strategy = OrnsteinUhlenbeckStrategy(
        symbol="AAPL", window_size=10, entry_z=2.0, allow_short=True
    )
    monkeypatch.setattr(
        strategy, "_calibrate", lambda: fixed_fit(math.log(100.0), 0.01)
    )

    # 1. Establish a flat baseline at price = 100.0 (with tiny noise to avoid divide-by-zero)
    stable_prices = [100.1, 99.9, 100.2, 99.8, 100.1, 99.9, 100.0, 100.1, 99.9]

    for p in stable_prices:
        feed(strategy, p)

    # 2. Spike the price to 110.0 (This is a massive > 2.0 Z-score move)
    signal = feed(strategy, 110.0)

    # 3. Verify it wants to short the spike
    assert signal is not None
    assert isinstance(signal, SignalEvent)
    assert signal.direction == "SHORT"
    assert strategy.intent["AAPL"] == "SHORT"

    # 4. Crash the price back down to the mean (100.0)
    exit_signal = feed(strategy, 100.0)

    # 5. Verify it closed the trade
    assert exit_signal is not None
    assert exit_signal.direction == "EXIT"
    assert strategy.intent["AAPL"] is None


def test_long_only_ignores_spike(monkeypatch):
    """With shorting disabled (the default) an upward spike should not short."""
    strategy = OrnsteinUhlenbeckStrategy(symbol="AAPL", window_size=10, entry_z=2.0)
    monkeypatch.setattr(
        strategy, "_calibrate", lambda: fixed_fit(math.log(100.0), 0.01)
    )

    stable_prices = [100.1, 99.9, 100.2, 99.8, 100.1, 99.9, 100.0, 100.1, 99.9]
    for p in stable_prices:
        feed(strategy, p)

    assert feed(strategy, 110.0) is None
    assert strategy.intent.get("AAPL") is None


def test_signals_reach_a_shared_queue(monkeypatch):
    """
    When an events queue is supplied the engine sees the signal on it. This is
    the path the engine actually uses; the tests above rely on the private
    queue the base class creates when none is passed.
    """
    events: deque[Event] = deque()
    strategy = OrnsteinUhlenbeckStrategy(
        events, symbol="AAPL", window_size=10, entry_z=2.0, allow_short=True
    )
    monkeypatch.setattr(
        strategy, "_calibrate", lambda: fixed_fit(math.log(100.0), 0.01)
    )

    for p in [100.1, 99.9, 100.2, 99.8, 100.1, 99.9, 100.0, 100.1, 99.9]:
        strategy.calculate_signals(create_market_event(p))
    assert len(events) == 0

    strategy.calculate_signals(create_market_event(110.0))

    assert len(events) == 1
    signal = events[0]
    assert isinstance(signal, SignalEvent)
    assert signal.direction == "SHORT"


def test_ignores_other_symbols(base_strategy):
    """A bar for a different symbol must not enter the rolling window."""
    other = MarketEvent(
        symbol="MSFT",
        timestamp=T0,
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=100,
    )
    base_strategy.calculate_signals(other)

    assert len(base_strategy.prices) == 0
    assert len(base_strategy.events) == 0


def test_prime_fills_window_without_signals(base_strategy):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(9):
        price = 100.0 + i
        base_strategy.prime(
            MarketEvent(
                "AAPL", start + timedelta(days=i), price, price, price, price, 100
            )
        )

    assert base_strategy.warmup_period == 9
    assert len(base_strategy.prices) == 9
    assert len(base_strategy.events) == 0


def prime_flat(strategy: OrnsteinUhlenbeckStrategy, bars: int = 9) -> None:
    for i in range(bars):
        strategy.prime(
            MarketEvent("AAPL", T0 + timedelta(days=i), 100.0, 100.0, 100.0, 100.0, 100)
        )


def random_walks(rng: np.random.Generator, paths: int) -> list[np.ndarray]:
    return [math.log(100) + np.cumsum(rng.normal(0, 0.02, 60)) for _ in range(paths)]


def ar1_path(rng: np.random.Generator, phi: float, bars: int = 60) -> np.ndarray:
    mean = math.log(100)
    lp = np.empty(bars)
    lp[0] = mean
    for t in range(1, bars):
        lp[t] = mean + phi * (lp[t - 1] - mean) + rng.normal(0, 0.02)
    return lp


def test_exact_mapping_matches_closed_form():
    lp = ar1_path(default_rng(11), 0.8)
    x = lp[:-1]
    y = np.diff(lp)
    b, a = np.polyfit(x, y, 1)
    residuals = y - (a + b * x)
    sigma_eps = math.sqrt(np.sum(residuals**2) / (len(y) - 2))

    fit = calibrate_ou(lp, None)

    assert fit.mean_reverting is True
    assert fit.theta == pytest.approx(-math.log(1 + b), rel=1e-12)
    assert fit.mu == pytest.approx(-a / b, rel=1e-12)
    assert fit.sigma_eq == pytest.approx(
        sigma_eps / math.sqrt(1 - (1 + b) ** 2), rel=1e-12
    )
    assert fit.half_life == pytest.approx(math.log(2) / fit.theta, rel=1e-12)


def test_critical_values():
    assert dickey_fuller_critical_value(0.05, 59) == pytest.approx(-2.9119, abs=1e-4)
    assert dickey_fuller_critical_value(0.01, 10**9) == pytest.approx(
        -3.43035, abs=1e-4
    )
    with pytest.raises(ValueError):
        dickey_fuller_critical_value(0.2, 59)
    with pytest.raises(ValueError):
        dickey_fuller_critical_value(0.05, 2)


def test_gate_rejects_random_walks_at_nominal_rate():
    walks = random_walks(default_rng(12345), 2000)

    at_5 = np.mean([calibrate_ou(w, 0.05).passed for w in walks])
    sign_only = np.mean([calibrate_ou(w, None).passed for w in walks])

    assert 0.03 <= at_5 <= 0.08
    # OLS on a short window is biased towards mean reversion, so the sign check
    # alone passes most random walks.
    assert sign_only > 0.85


def test_gate_accepts_strong_mean_reversion():
    rng = default_rng(2024)
    paths = [ar1_path(rng, 0.5) for _ in range(500)]

    assert np.mean([calibrate_ou(p, 0.05).passed for p in paths]) > 0.9


def test_failed_gate_keeps_the_fit():
    walks = random_walks(default_rng(12345), 2000)
    window = next(
        w
        for w in walks
        if calibrate_ou(w, None).mean_reverting and not calibrate_ou(w, 0.05).passed
    )

    loose = calibrate_ou(window, None)
    gated = calibrate_ou(window, 0.05)

    assert gated.mean_reverting is True
    assert gated.passed is False
    assert gated.mu == loose.mu
    assert gated.sigma_eq == loose.sigma_eq


def test_degenerate_windows_fail_cleanly():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat = calibrate_ou(np.full(60, math.log(100)), 0.05)
        line = calibrate_ou(math.log(100) + 0.01 * np.arange(60), 0.05)

    assert flat.mean_reverting is False
    assert line.mean_reverting is False


def test_no_entry_when_gate_fails(monkeypatch, base_strategy):
    prime_flat(base_strategy)
    monkeypatch.setattr(
        base_strategy,
        "_calibrate",
        lambda: fixed_fit(math.log(100), 0.01, passed=False),
    )

    assert feed(base_strategy, 90.0) is None
    assert base_strategy.intent.get("AAPL") is None


def hold_long(strategy: OrnsteinUhlenbeckStrategy) -> None:
    prime_flat(strategy)
    strategy.intent["AAPL"] = "LONG"
    strategy.position["AAPL"] = "LONG"


def test_held_position_exits_at_exit_z_when_gate_fails(monkeypatch, base_strategy):
    hold_long(base_strategy)
    monkeypatch.setattr(
        base_strategy,
        "_calibrate",
        lambda: fixed_fit(math.log(100), 0.01, passed=False),
    )

    signal = feed(base_strategy, 101.0)

    assert len(base_strategy.events) == 1
    assert signal is not None
    assert signal.direction == "EXIT"
    assert base_strategy.intent["AAPL"] is None


def test_held_position_stays_open_below_exit_z_when_gate_fails(
    monkeypatch, base_strategy
):
    hold_long(base_strategy)
    monkeypatch.setattr(
        base_strategy,
        "_calibrate",
        lambda: fixed_fit(math.log(100), 0.01, passed=False),
    )

    assert feed(base_strategy, 95.0) is None
    assert base_strategy.intent["AAPL"] == "LONG"


def test_exit_when_window_stops_mean_reverting(monkeypatch, base_strategy):
    hold_long(base_strategy)
    monkeypatch.setattr(
        base_strategy,
        "_calibrate",
        lambda: fixed_fit(0.0, 0.0, mean_reverting=False, passed=False),
    )

    signal = feed(base_strategy, 97.0)

    assert len(base_strategy.events) == 1
    assert signal is not None
    assert signal.direction == "EXIT"
    assert base_strategy.intent["AAPL"] is None


def test_no_signal_when_flat_and_not_mean_reverting(monkeypatch, base_strategy):
    prime_flat(base_strategy)
    monkeypatch.setattr(
        base_strategy,
        "_calibrate",
        lambda: fixed_fit(0.0, 0.0, mean_reverting=False, passed=False),
    )

    assert feed(base_strategy, 90.0) is None
    assert base_strategy.intent.get("AAPL") is None


def test_window_counters(base_strategy):
    prices = 100 * np.exp(np.cumsum(default_rng(7).normal(0, 0.02, 12)))
    for price in prices:
        feed(base_strategy, float(price))

    assert base_strategy.windows_tested == 3
    assert 0 <= base_strategy.windows_passed <= 3


def test_invalid_gate_level_rejected():
    with pytest.raises(ValueError):
        OrnsteinUhlenbeckStrategy(symbol="AAPL", gate_level=0.2)


def test_gate_summary_strings():
    assert gate_summary(0.05, 0, 0, 60, False) == (
        "The mean-reversion check never ran: the data is shorter than the "
        "60-bar window."
    )
    assert gate_summary(0.05, 6540, 301, 60, True) == (
        "Mean-reversion check (5% level) passed on 4.6% of 6,540 windows. "
        "Trades only open while the latest window passes."
    )
    assert gate_summary(None, 200, 190, 60, False) == (
        "Mean-reversion check (sign check only) passed on 95.0% of 200 windows. "
        "Trades only open while the latest window passes. No trades were made."
    )
