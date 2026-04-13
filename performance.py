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
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdowns = (high_water_mark - equity_curve) / high_water_mark
        drawdowns[np.isnan(drawdowns)] = 0.0
        drawdowns[np.isinf(drawdowns)] = 0.0
    
    return float(np.max(drawdowns))

def create_summary_stats(portfolio: Portfolio) -> dict:
    """
    Returns a dictionary of: Total Return, Sharpe Ratio, Max Drawdown, and Win Rate.
    """
    df = portfolio.generate_equity_curve()
    
    if df.empty:
        return {"error": "Portfolio is empty. No performance stats to calculate."}
        
    equity_curve = df['total'].to_numpy()
    
    if len(equity_curve) < 2:
        return {"error": "Insufficient data points in portfolio to calculate performance stats."}
        
    initial_capital = portfolio.initial_capital
    final_equity = equity_curve[-1]
    
    # Total Return
    total_return = (final_equity / initial_capital) - 1.0
    
    # Calculate returns
    returns_series = df['total'].pct_change().dropna()
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
        
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }
