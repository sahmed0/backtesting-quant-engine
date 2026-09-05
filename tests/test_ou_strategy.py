from collections import deque
from datetime import datetime

import pytest

from event import Event, MarketEvent, SignalEvent
from strategies.ou_strategy import OrnsteinUhlenbeckStrategy


@pytest.fixture
def base_strategy():
    """Returns a strategy instance with a small window for fast testing."""
    return OrnsteinUhlenbeckStrategy(symbol="AAPL", window_size=10, entry_z=2.0)


def create_market_event(price: float) -> MarketEvent:
    """Helper to quickly mock market data ticks."""
    return MarketEvent(
        symbol="AAPL",
        timestamp=datetime.now(),
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

    The model now fits log-prices, so the non-mean-reverting series is a
    *geometric* trend (constant log-returns => a straight line in log space).
    A linear price ramp is concave in log space and would read as mean-reverting.
    """
    # Geometric trend: 100, 110, 121, ... -> log-prices are linear -> theta ~ 0.
    for i in range(10):
        feed(base_strategy, 100.0 * 1.1**i)

    # Manually trigger the calibration
    mu, sigma, is_mean_reverting = base_strategy._calibrate_ou_parameters()

    # A straight line in log space is not mean-reverting
    assert is_mean_reverting is False


def test_mean_reversion_signals():
    """
    Feed the strategy a stable baseline, then artificially spike the price.
    With shorting enabled it should generate a SHORT signal expecting a return
    to the mean.
    """
    strategy = OrnsteinUhlenbeckStrategy(
        symbol="AAPL", window_size=10, entry_z=2.0, allow_short=True
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


def test_long_only_ignores_spike():
    """With shorting disabled (the default) an upward spike should not short."""
    strategy = OrnsteinUhlenbeckStrategy(symbol="AAPL", window_size=10, entry_z=2.0)

    stable_prices = [100.1, 99.9, 100.2, 99.8, 100.1, 99.9, 100.0, 100.1, 99.9]
    for p in stable_prices:
        feed(strategy, p)

    assert feed(strategy, 110.0) is None
    assert strategy.intent.get("AAPL") is None


def test_signals_reach_a_shared_queue():
    """
    When an events queue is supplied the engine sees the signal on it. This is
    the path the engine actually uses; the tests above rely on the private
    queue the base class creates when none is passed.
    """
    events: deque[Event] = deque()
    strategy = OrnsteinUhlenbeckStrategy(
        events, symbol="AAPL", window_size=10, entry_z=2.0, allow_short=True
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
        timestamp=datetime.now(),
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=100,
    )
    base_strategy.calculate_signals(other)

    assert len(base_strategy.prices) == 0
    assert len(base_strategy.events) == 0
