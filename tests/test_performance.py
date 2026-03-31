"""
Unit tests for the performance module.
"""

import numpy as np
import pytest
from performance import calculate_sharpe_ratio, calculate_drawdown

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
