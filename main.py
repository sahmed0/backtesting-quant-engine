import os
import sys
import queue
import asyncio

from data import CSVDataHandler
from strategy import SimpleMovingAverageStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
from engine import Backtest
import performance

async def main_async():
    # Initialise Queue
    events = queue.Queue()

    # Set CSV Path
    csv_path = 'data/sample_data.csv'

    # Error Handling
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    # Prepare DataHandler Arguments
    csv_dir = os.path.dirname(csv_path)
    symbol = os.path.splitext(os.path.basename(csv_path))[0]

    # Initialise Components
    data_handler = CSVDataHandler(events, csv_dir, [symbol])
    strategy = SimpleMovingAverageStrategy(events, short_window=5, long_window=20)
    portfolio = Portfolio(events, initial_capital=100000.0)
    execution_handler = SimulatedExecutionHandler(events, data_handler)
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
