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
import logging
import re

# Logging handler to push logs to the UI Table
class WebOrderBookHandler(logging.Handler):
    def __init__(self, table_body_id):
        super().__init__()
        self.table_body_id = table_body_id
        # Regex to parse specifically the "FILLED" message from execution.py
        self.pattern = re.compile(
            r"FILLED\s+(?P<time>.*?)\s+(?P<side>LONG|SHORT|EXIT)\s+(?P<qty>.*?)\s+(?P<symbol>.*?)\s+@\s+(?P<price>.*?)\s+\(comm:\s+(?P<comm>.*?),\s+slippage:\s+(?P<slip>.*?)\)"
        )

    def emit(self, record):
        msg = self.format(record)
        match = self.pattern.search(msg)
        
        if match:
            data = match.groupdict()
            self.add_row_to_table(data)

    def add_row_to_table(self, data):
        tbody = document.getElementById(self.table_body_id)
        if not tbody:
            return

        tr = document.createElement("tr")
        
        # Determine CSS class for side
        side_class = f"dir-{data['side'].lower()}"
        
        # Simple formatting for time - extract just the date if it's a long timestamp string
        # Assuming format like "2026-04-04 11:23:45.678000+00:00"
        time_display = data['time'].split('.')[0] if '.' in data['time'] else data['time']
        if ' ' in time_display:
            time_display = time_display.split(' ')[0] # Just the date not the HH:MM:SS time

        tr.innerHTML = f"""
            <td>{time_display}</td>
            <td class="{side_class}">{data['side']}</td>
            <td>{float(data['qty']):.0f}</td>
            <td>{data['symbol']}</td>
            <td>{float(data['price']):.2f}</td>
            <td>{float(data['comm']):.4f}</td>
            <td>{float(data['slip']):.4f}</td>
        """
        # Prepend new trades to the top of the table
        if tbody.firstChild:
            tbody.insertBefore(tr, tbody.firstChild)
        else:
            tbody.appendChild(tr)

# Initialise the handler but don't attach yet
ui_handler = WebOrderBookHandler("order-log-body")
ui_handler.setFormatter(logging.Formatter('%(message)s'))
from execution import logger as execution_logger
execution_logger.addHandler(ui_handler)
execution_logger.propagate = False # Prevent double logging to console

async def run_backtest(event):
    status_el = document.getElementById("status")
    btn = document.getElementById("run-btn")
    file_input = document.getElementById("csv-upload")
    error_output = document.getElementById("error-output")
    
    error_output.innerText = ""
    status_el.innerText = "Reading data..."
    btn.disabled = True
    btn.innerText = "Running..."
    
    # Clear previous logs
    document.getElementById("order-log-body").innerHTML = ""
    
    try:
        ticker_select = document.getElementById("ticker-select")
        
        files = file_input.files
        if files and files.length > 0:
            # Handle uploaded file (Priority)
            file = files.item(0)
            text_content = await file.text()
            
            # Write to virtual file system
            os.makedirs('/data', exist_ok=True)
            symbol = os.path.splitext(file.name)[0]
            csv_path = f'/data/{symbol}.csv'
            
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
                
            status_el.innerText = f"Running backtest for uploaded {symbol}..."
        elif ticker_select.value:
            # Handle pre-loaded ticker
            symbol = ticker_select.value
            status_el.innerText = f"Running backtest for pre-loaded {symbol}..."
        else:
            status_el.innerText = "No data source selected."
            btn.disabled = False
            btn.innerText = "Run Backtest"
            return
        
        # Initialise backtest components
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
            
            # Pass data to JS for charts
            import json
            df = portfolio.generate_equity_curve()
            if not df.empty and 'price' in df.columns:
                timestamps = df['timestamp'].tolist()
                equity = df['total'].tolist()
                prices = df['price'].tolist()
                trades = portfolio.trades
                
                window.updateCharts(
                    json.dumps(timestamps),
                    json.dumps(equity),
                    json.dumps(prices),
                    json.dumps(trades)
                )
                
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
    status_el.innerText = "Ready. Select a ticker or upload a CSV file."

# Initialise when script loads
setup()