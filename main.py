import os
import sys
import queue

from data import PolarsCSVDataHandler
from strategy import SimpleMovingAverageStrategy
from strategies.ou_strategy import OrnsteinUhlenbeckStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
from engine import Backtest
import performance

if __name__ == "__main__":
    # Initialise Queue
    events = queue.Queue()

    # Set CSV Path
    csv_path = 'data/GOOG.csv'

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    # Prepare DataHandler Arguments
    csv_dir = os.path.dirname(csv_path)
    symbol = os.path.splitext(os.path.basename(csv_path))[0]

    # Initialise Components
    data_handler = PolarsCSVDataHandler(events, csv_dir, [symbol])
    strategy = SimpleMovingAverageStrategy(events, short_window=5, long_window=20)
    # strategy = OrnsteinUhlenbeckStrategy(events, symbol, window_size=60, entry_z=2.0, exit_z=0.0)
    portfolio = Portfolio(events, initial_capital=100000.0)
    execution_handler = SimulatedExecutionHandler(events, data_handler)
    backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)

    # Run Backtest
    backtest.run()

    # Performance Summary
    performance.create_summary_stats(portfolio)
