import pytest
import numpy as np
from datetime import datetime
from event import MarketEvent
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
        open=price, high=price, low=price, close=price, volume=100
    )

def test_strategy_warmup(base_strategy):
    """Ensure no signals are generated while the rolling window is filling."""
    # Feed it 9 prices (window size is 10)
    for i in range(9):
        event = create_market_event(100.0 + i)
        signal = base_strategy.calculate_signals(event)
        assert signal is None
        
def test_trending_rejection(base_strategy):
    """
    If a stock is purely trending up, the OU process should realize 
    it is NOT mean-reverting (theta <= 0) and refuse to trade.
    """
    # Create a strictly trending price series: 10, 20, 30, 40...
    for i in range(10):
        base_strategy.calculate_signals(create_market_event(float((i+1) * 10)))
        
    # Manually trigger the calibration
    mu, sigma, is_mean_reverting = base_strategy._calibrate_ou_parameters()
    
    # A straight upward line is not mean-reverting
    assert is_mean_reverting is False

def test_mean_reversion_signals(base_strategy):
    """
    Feed the strategy a stable baseline, then artificially spike the price.
    It should generate a SHORT signal expecting a return to the mean.
    """
    # 1. Establish a flat baseline at price = 100.0 (with tiny noise to avoid divide-by-zero)
    stable_prices = [100.1, 99.9, 100.2, 99.8, 100.1, 99.9, 100.0, 100.1, 99.9]
    
    for p in stable_prices:
        base_strategy.calculate_signals(create_market_event(p))
        
    # 2. Spike the price to 110.0 (This is a massive > 2.0 Z-score move)
    spike_event = create_market_event(110.0)
    signal = base_strategy.calculate_signals(spike_event)
    
    # 3. Verify it wants to short the spike
    assert signal is not None
    assert signal.type == 'SIGNAL'
    assert signal.direction == 'SHORT'
    assert base_strategy.invested == 'SHORT'
    
    # 4. Crash the price back down to the mean (100.0)
    mean_event = create_market_event(100.0)
    exit_signal = base_strategy.calculate_signals(mean_event)
    
    # 5. Verify it closed the trade
    assert exit_signal is not None
    assert exit_signal.direction == 'EXIT'
    assert base_strategy.invested is False
    