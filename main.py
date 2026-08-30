import asyncio
import logging
import os
import sys
from collections import deque

import performance
from data import CSVDataHandler
from engine import Backtest
from event import Event
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import (
    ATRStopSizer,
    FixedSizer,
    FractionalKellySizer,
    PercentEquitySizer,
    VolatilityTargetSizer,
)
from strategies.ou_strategy import OrnsteinUhlenbeckStrategy
from strategy import SimpleMovingAverageStrategy

# Strategy to run: 'sma' (moving average crossover) or 'ou' (Ornstein-Uhlenbeck)
STRATEGY = "sma"

# Position sizing: 'fixed', 'percent' (% of equity), 'vol' (volatility target),
# 'atr' (ATR-stop fixed risk), or 'kelly' (fractional Kelly).
SIZING = "percent"


def build_strategy(events, symbol):
    """Constructs the configured strategy instance."""
    if STRATEGY == "ou":
        return OrnsteinUhlenbeckStrategy(
            events, symbol, window_size=60, entry_z=2.0, exit_z=0.0
        )
    return SimpleMovingAverageStrategy(events, short_window=5, long_window=20)


def build_sizer():
    """Constructs the configured position sizer."""
    if SIZING == "percent":
        return PercentEquitySizer(fraction=0.1)
    if SIZING == "vol":
        return VolatilityTargetSizer(target_volatility=0.15, lookback=20)
    if SIZING == "atr":
        return ATRStopSizer(risk_fraction=0.02, atr_period=14, atr_multiple=2.0)
    if SIZING == "kelly":
        return FractionalKellySizer(kelly_fraction=0.5, min_trades=10)
    return FixedSizer(100.0)


async def main_async():
    # Library modules only ever call logger.info; the entry point decides that
    # those records reach the console.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Initialise Queue
    events: deque[Event] = deque()

    # Set CSV Path
    csv_path = "data/AAPL.csv"

    # Error Handling
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    # Prepare DataHandler Arguments
    csv_dir = os.path.dirname(csv_path)
    symbol = os.path.splitext(os.path.basename(csv_path))[0]

    # Initialise Components
    data_handler = CSVDataHandler(events, csv_dir, [symbol])
    strategy = build_strategy(events, symbol)
    portfolio = Portfolio(events, initial_capital=100000.0, sizer=build_sizer())
    execution_handler = SimulatedExecutionHandler(events, data_handler, portfolio)
    backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)

    # Run Backtest
    await backtest.run()

    # Performance Summary
    stats = performance.create_summary_stats(portfolio)
    if "error" in stats:
        print(stats["error"])
    else:
        print("-" * 40)
        print("Performance Summary")
        print("-" * 40)
        print(f"Total Return: {stats['total_return'] * 100:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {stats['max_drawdown'] * 100:.2f}%")
        print(f"Win Rate:     {stats['win_rate'] * 100:.2f}%")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main_async())
