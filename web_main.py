from pyscript import document, window
from pyodide.ffi import create_proxy
import asyncio
import queue
import os
import traceback

from data import CSVDataHandler
from strategy import SimpleMovingAverageStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
from engine import Backtest
import performance

async def run_backtest(event):
    status_el = document.getElementById("status")
    btn = document.getElementById("run-btn")
    file_input = document.getElementById("csv-upload")
    error_output = document.getElementById("error-output")
    
    error_output.innerText = ""
    status_el.innerText = "Reading data..."
    btn.disabled = True
    btn.innerText = "Running..."
    
    try:
        files = file_input.files
        if not files or files.length == 0:
            status_el.innerText = "No file selected."
            return
            
        file = files.item(0)
        # await file.text() handles the JS Promise
        text_content = await file.text()
        
        # Write to virtual file system
        os.makedirs('/data', exist_ok=True)
        # Extract symbol from filename (e.g. 'AAPL.csv' -> 'AAPL')
        symbol = os.path.splitext(file.name)[0]
        csv_path = f'/data/{symbol}.csv'
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
            
        status_el.innerText = f"Running backtest for {symbol}..."
        
        # Initialize backtest components
        events = queue.Queue()
        data_handler = CSVDataHandler(events, '/data', [symbol])
        strategy = SimpleMovingAverageStrategy(events, short_window=5, long_window=20)
        portfolio = Portfolio(events, initial_capital=100000.0)
        execution_handler = SimulatedExecutionHandler(events, data_handler)
        backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)
        
        # Await the execution of the async backtest
        await backtest.run()
        
        status_el.innerText = "Calculating performance..."
        
        # Get metrics
        stats = performance.create_summary_stats(portfolio)
        
        if "error" in stats:
            error_output.innerText = stats["error"]
            status_el.innerText = "Error calculating stats."
        else:
            document.getElementById("val-return").innerText = f"{stats['total_return'] * 100:.2f}%"
            document.getElementById("val-sharpe").innerText = f"{stats['sharpe_ratio']:.2f}"
            document.getElementById("val-drawdown").innerText = f"{stats['max_drawdown'] * 100:.2f}%"
            document.getElementById("val-winrate").innerText = f"{stats['win_rate'] * 100:.2f}%"
            status_el.innerText = "Backtest complete!"
            
    except Exception as e:
        error_output.innerText = f"Error: {str(e)}\n{traceback.format_exc()}"
        status_el.innerText = "An error occurred during execution."
    finally:
        btn.disabled = False
        btn.innerText = "Run Backtest"

    return True

def setup():
    btn = document.getElementById("run-btn")
    # Bind the run_backtest async function to the button click event
    click_proxy = create_proxy(run_backtest)
    btn.addEventListener("click", click_proxy)
    
    status_el = document.getElementById("status")
    status_el.innerText = "Ready. Please upload a CSV file."

# Initialise when script loads
setup()