import asyncio
import csv
import json
import logging
import os
import re
import traceback
from collections import deque
from datetime import UTC, datetime

import numpy as np
from pyodide.ffi import create_proxy
from pyscript import document, window

import performance
from data import CSVDataHandler
from engine import Backtest
from event import Event
from execution import SimulatedExecutionHandler
from execution import logger as execution_logger
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


def build_sizer(choice, periods_per_year):
    """Maps the UI position-sizing choice to a PositionSizer instance.

    ``periods_per_year`` is inferred from the loaded data and used by the
    volatility-target sizer to annualise its measured volatility; the other
    sizers ignore it.
    """
    if choice == "vol":
        return VolatilityTargetSizer(
            target_volatility=0.15, lookback=20, periods=periods_per_year
        )
    if choice == "atr":
        return ATRStopSizer(risk_fraction=0.02, atr_period=14, atr_multiple=2.0)
    if choice == "kelly":
        return FractionalKellySizer(kelly_fraction=0.5, min_trades=10)
    if choice == "fixed":
        return FixedSizer(100.0)
    # Default: percent-of-equity.
    return PercentEquitySizer(fraction=0.1)


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
        time_display = (
            data["time"].split(".")[0] if "." in data["time"] else data["time"]
        )
        if " " in time_display:
            time_display = time_display.split(" ")[
                0
            ]  # Just the date not the HH:MM:SS time

        tr.innerHTML = f"""
            <td>{time_display}</td>
            <td class="{side_class}">{data["side"]}</td>
            <td>{float(data["qty"]):.0f}</td>
            <td>{data["symbol"]}</td>
            <td>{float(data["price"]):.2f}</td>
            <td>{float(data["comm"]):.4f}</td>
            <td>{float(data["slip"]):.4f}</td>
        """
        # Prepend new trades to the top of the table
        if tbody.firstChild:
            tbody.insertBefore(tr, tbody.firstChild)
        else:
            tbody.appendChild(tr)


# Initialise the handler but don't attach yet
ui_handler = WebOrderBookHandler("order-log-body")
ui_handler.setFormatter(logging.Formatter("%(message)s"))
execution_logger.addHandler(ui_handler)
execution_logger.propagate = False  # Prevent double logging to console

# --- Overfitting Lab -------------------------------------------------------
# Parameter grid searched for the in-sample / out-of-sample heatmaps. The lab
# always analyses the moving-average crossover (two integer windows grid nicely)
# regardless of which strategy is selected for a normal backtest.
OF_SHORT_WINDOWS = [5, 10, 15, 20]
OF_LONG_WINDOWS = [25, 50, 100, 200]
OF_IS_FRACTION = 0.70


async def _ensure_symbol(file_input, ticker_select):
    """
    Resolves the active data source to a symbol whose CSV lives under /data,
    writing an uploaded file into the virtual filesystem if one is present.
    Returns the symbol, or None when nothing is selected.
    """
    files = file_input.files
    if files and files.length > 0:
        file = files.item(0)
        text_content = await file.text()
        os.makedirs("/data", exist_ok=True)
        symbol = os.path.splitext(file.name)[0]
        with open(f"/data/{symbol}.csv", "w", encoding="utf-8") as f:
            f.write(text_content)
        return symbol
    if ticker_select.value:
        return ticker_select.value
    return None


def _read_timestamps(csv_path):
    """Reads every bar's (tz-aware, UTC) timestamp from a symbol CSV, in order."""
    timestamps = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timestamps.append(
                datetime.fromisoformat(row["timestamp"]).replace(tzinfo=UTC)
            )
    return timestamps


def _infer_ppy(csv_path):
    """Infers periods-per-year from a symbol CSV's bar timestamps."""
    seconds = np.array([t.timestamp() for t in _read_timestamps(csv_path)])
    return performance.infer_periods_per_year(seconds)


async def _grid_sharpe(
    symbol,
    start,
    end,
    sizer_choice,
    initial_capital,
    commission_per_share,
    min_commission,
    slippage_pct,
    periods_per_year,
):
    """
    Runs the SMA parameter grid over [start, end] and returns a 2D list of
    Sharpe ratios indexed [short_idx][long_idx]; cells where short >= long, or
    that produce no usable curve, are None. Yields to the event loop between
    runs so the progress status can repaint, and returns (grid, runs_done).
    """
    grid = []
    runs_done = 0
    for short_w in OF_SHORT_WINDOWS:
        row = []
        for long_w in OF_LONG_WINDOWS:
            if short_w >= long_w:
                row.append(None)
                continue
            events: deque[Event] = deque()
            data_handler = CSVDataHandler(
                events, "/data", [symbol], start_date=start, end_date=end
            )
            strategy = SimpleMovingAverageStrategy(
                events, short_window=short_w, long_window=long_w
            )
            portfolio = Portfolio(
                events,
                initial_capital=initial_capital,
                sizer=build_sizer(sizer_choice, periods_per_year),
            )
            execution_handler = SimulatedExecutionHandler(
                events,
                data_handler,
                portfolio,
                commission_per_share=commission_per_share,
                min_commission=min_commission,
                slippage_pct=slippage_pct,
            )
            backtest = Backtest(
                data_handler, strategy, portfolio, execution_handler, events
            )
            await backtest.run()
            stats = performance.create_summary_stats(portfolio)
            row.append(None if "error" in stats else stats["sharpe_ratio"])
            runs_done += 1
            # Hand control back to the browser so the status text can update.
            await asyncio.sleep(0)
        grid.append(row)
    return grid, runs_done


def _rank_cells(grid):
    """Returns valid (sharpe, i, j) cells sorted best-first by Sharpe."""
    cells = []
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            if val is not None:
                cells.append((val, i, j))
    cells.sort(reverse=True)
    return cells


async def analyse_overfitting(event):
    status_el = document.getElementById("status")
    analyse_btn = document.getElementById("analyse-btn")
    run_btn = document.getElementById("run-btn")
    file_input = document.getElementById("csv-upload")
    error_output = document.getElementById("error-output")

    error_output.innerText = ""
    analyse_btn.disabled = True
    run_btn.disabled = True
    analyse_btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analysing...'

    # The grid runs dozens of backtests; mute per-fill logging so the Order Book
    # table isn't flooded with the analysis's intermediate fills.
    execution_logger.setLevel(logging.WARNING)

    try:
        ticker_select = document.getElementById("ticker-select")
        symbol = await _ensure_symbol(file_input, ticker_select)
        if symbol is None:
            status_el.innerText = "No data source selected."
            return

        csv_path = f"/data/{symbol}.csv"

        # Mirror the main backtest's costs and sizing so the grids are comparable.
        sizer_choice = document.getElementById("sizer-select").value
        initial_capital = float(document.getElementById("initial-capital").value)
        if initial_capital <= 0:
            raise ValueError("Initial capital must be greater than zero.")
        commission_per_share = float(
            document.getElementById("commission-per-share").value
        )
        min_commission = float(document.getElementById("min-commission").value)
        slippage_pct = float(document.getElementById("slippage").value) / 100.0
        if commission_per_share < 0 or min_commission < 0 or slippage_pct < 0:
            raise ValueError("Commission and slippage cannot be negative.")

        # Chronological 70/30 split, with no overlapping bar between the windows.
        timestamps = _read_timestamps(csv_path)
        n = len(timestamps)
        split_idx = int(n * OF_IS_FRACTION)
        if split_idx < 1 or split_idx >= n:
            raise ValueError("Not enough data to form an in-/out-of-sample split.")
        is_start, is_end = timestamps[0], timestamps[split_idx - 1]
        oos_start, oos_end = timestamps[split_idx], timestamps[-1]

        # Annualisation factor for the vol-target sizer, from the full history
        # so both windows are sized on the same footing.
        periods_per_year = performance.infer_periods_per_year(
            np.array([t.timestamp() for t in timestamps])
        )

        valid = sum(
            1
            for short_w in OF_SHORT_WINDOWS
            for long_w in OF_LONG_WINDOWS
            if short_w < long_w
        )
        total = 2 * valid

        status_el.innerText = f"Analysing... 0/{total}"
        is_grid, done_is = await _grid_sharpe(
            symbol,
            is_start,
            is_end,
            sizer_choice,
            initial_capital,
            commission_per_share,
            min_commission,
            slippage_pct,
            periods_per_year,
        )
        status_el.innerText = f"Analysing... {done_is}/{total}"
        oos_grid, _ = await _grid_sharpe(
            symbol,
            oos_start,
            oos_end,
            sizer_choice,
            initial_capital,
            commission_per_share,
            min_commission,
            slippage_pct,
            periods_per_year,
        )

        is_cells = _rank_cells(is_grid)
        oos_cells = _rank_cells(oos_grid)
        if not is_cells or not oos_cells:
            raise ValueError("No parameter combination produced a usable result.")

        # The in-sample winner (the cell a naive researcher would pick) and where
        # that exact cell lands in the out-of-sample ranking.
        _, is_best_i, is_best_j = is_cells[0]
        oos_order = [(i, j) for _, i, j in oos_cells]
        is_best_oos_rank = (
            oos_order.index((is_best_i, is_best_j)) + 1
            if (is_best_i, is_best_j) in oos_order
            else None
        )
        _, oos_best_i, oos_best_j = oos_cells[0]

        payload = {
            "symbol": symbol,
            "short_windows": OF_SHORT_WINDOWS,
            "long_windows": OF_LONG_WINDOWS,
            "is_sharpe": is_grid,
            "oos_sharpe": oos_grid,
            "is_best": [is_best_i, is_best_j],
            "oos_best": [oos_best_i, oos_best_j],
            "is_best_oos_rank": is_best_oos_rank,
            "num_cells": len(oos_order),
            "is_range": [
                is_start.date().isoformat(),
                is_end.date().isoformat(),
                split_idx,
            ],
            "oos_range": [
                oos_start.date().isoformat(),
                oos_end.date().isoformat(),
                n - split_idx,
            ],
            "is_best_params": [OF_SHORT_WINDOWS[is_best_i], OF_LONG_WINDOWS[is_best_j]],
            "oos_best_params": [
                OF_SHORT_WINDOWS[oos_best_i],
                OF_LONG_WINDOWS[oos_best_j],
            ],
            "is_best_is_sharpe": is_grid[is_best_i][is_best_j],
            "is_best_oos_sharpe": oos_grid[is_best_i][is_best_j],
            "oos_best_oos_sharpe": oos_grid[oos_best_i][oos_best_j],
        }
        window.updateHeatmaps(json.dumps(payload))
        status_el.innerHTML = '<i class="fa-solid fa-check text-success"></i>'

    except Exception as e:
        error_output.innerText = f"Error: {str(e)}\n{traceback.format_exc()}"
        status_el.innerText = "An error occurred during analysis."
    finally:
        # Restore live fill logging for normal backtests.
        execution_logger.setLevel(logging.INFO)
        analyse_btn.disabled = False
        run_btn.disabled = False
        analyse_btn.innerHTML = (
            '<i class="fa-solid fa-magnifying-glass-chart"></i> Analyse Overfitting'
        )

    return True


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
            os.makedirs("/data", exist_ok=True)
            symbol = os.path.splitext(file.name)[0]
            csv_path = f"/data/{symbol}.csv"

            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            status_el.innerText = f"Running backtest for uploaded {symbol}..."
        elif ticker_select.value:
            # Handle pre-loaded ticker
            symbol = ticker_select.value
            csv_path = f"/data/{symbol}.csv"
            status_el.innerText = f"Running backtest for pre-loaded {symbol}..."
        else:
            status_el.innerText = "No data source selected."
            btn.disabled = False
            btn.innerText = "Run Backtest"
            return

        # Initialise backtest components
        events: deque[Event] = deque()
        data_handler = CSVDataHandler(events, "/data", [symbol])

        # Select the strategy chosen in the UI
        strategy_choice = document.getElementById("strategy-select").value
        allow_short = document.getElementById("allow-short").checked
        if strategy_choice == "ou":
            window_size = int(document.getElementById("ou-window").value)
            entry_z = float(document.getElementById("ou-entry-z").value)
            exit_z = float(document.getElementById("ou-exit-z").value)
            if window_size < 10:
                raise ValueError("OU window must be at least 10 periods.")
            strategy = OrnsteinUhlenbeckStrategy(
                events,
                symbol,
                window_size=window_size,
                entry_z=entry_z,
                exit_z=exit_z,
                allow_short=allow_short,
            )
        else:
            short_window = int(document.getElementById("sma-short").value)
            long_window = int(document.getElementById("sma-long").value)
            if short_window >= long_window:
                raise ValueError("Short window must be smaller than the long window.")
            strategy = SimpleMovingAverageStrategy(
                events,
                short_window=short_window,
                long_window=long_window,
                allow_short=allow_short,
            )
        sizer_choice = document.getElementById("sizer-select").value

        initial_capital = float(document.getElementById("initial-capital").value)
        if initial_capital <= 0:
            raise ValueError("Initial capital must be greater than zero.")

        commission_per_share = float(
            document.getElementById("commission-per-share").value
        )
        min_commission = float(document.getElementById("min-commission").value)
        # Slippage is entered as a percentage in the UI; convert to a fraction.
        slippage_pct = float(document.getElementById("slippage").value) / 100.0
        if commission_per_share < 0 or min_commission < 0 or slippage_pct < 0:
            raise ValueError("Commission and slippage cannot be negative.")

        # Annualisation factor inferred from the loaded data: used by the
        # vol-target sizer and surfaced under the Sharpe tile.
        periods_per_year = _infer_ppy(csv_path)

        portfolio = Portfolio(
            events,
            initial_capital=initial_capital,
            sizer=build_sizer(sizer_choice, periods_per_year),
        )
        execution_handler = SimulatedExecutionHandler(
            events,
            data_handler,
            portfolio,
            commission_per_share=commission_per_share,
            min_commission=min_commission,
            slippage_pct=slippage_pct,
        )
        backtest = Backtest(
            data_handler, strategy, portfolio, execution_handler, events
        )

        # Await the execution of the async backtest. The engine yields to the
        # browser periodically and calls back with its bar count.
        def on_progress(bars: int) -> None:
            status_el.innerText = f"Running... {bars} bars"

        await backtest.run(progress_cb=on_progress)

        status_el.innerText = "Calculating performance..."

        # Get metrics
        stats = performance.create_summary_stats(portfolio)

        if "error" in stats:
            error_output.innerText = stats["error"]
            status_el.innerText = "Error calculating stats."
        else:
            document.getElementById(
                "val-return"
            ).innerText = f"{stats['total_return'] * 100:.2f}%"
            document.getElementById(
                "val-sharpe"
            ).innerText = f"{stats['sharpe_ratio']:.2f}"
            document.getElementById(
                "cap-sharpe"
            ).innerText = f"annualised @ {round(stats['periods_per_year'])} periods/yr"
            document.getElementById(
                "val-drawdown"
            ).innerText = f"{stats['max_drawdown'] * 100:.2f}%"
            document.getElementById(
                "val-winrate"
            ).innerText = f"{stats['win_rate'] * 100:.2f}%"
            document.getElementById(
                "val-cagr"
            ).innerText = f"{stats['cagr'] * 100:.2f}%"
            document.getElementById(
                "val-alpha"
            ).innerText = f"{stats['alpha'] * 100:.2f}%"
            document.getElementById(
                "val-inforatio"
            ).innerText = f"{stats['information_ratio']:.2f}"
            document.getElementById(
                "val-calmar"
            ).innerText = f"{stats['calmar_ratio']:.2f}"
            document.getElementById("val-trades").innerText = f"{stats['num_trades']}"
            document.getElementById(
                "val-duration"
            ).innerText = f"{stats['avg_trade_duration']:.1f} days"

            # Pass data to JS for charts
            df = portfolio.generate_equity_curve()
            if not df.empty and "price" in df.columns:
                timestamps = df["timestamp"].tolist()
                equity = df["total"].tolist()
                prices = df["price"].tolist()
                trades = portfolio.trades

                # Buy-and-hold benchmark: deploy the full initial capital into
                # the asset at the first bar and hold, valued at each bar's price.
                first_price = prices[0] if prices else 0.0
                if first_price > 0:
                    benchmark = [
                        portfolio.initial_capital * (p / first_price) for p in prices
                    ]
                else:
                    benchmark = []

                window.updateCharts(
                    json.dumps(timestamps),
                    json.dumps(equity),
                    json.dumps(prices),
                    json.dumps(trades),
                    json.dumps(benchmark),
                )

            status_el.innerText = "Backtest Complete"

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

    # Bind the separate Overfitting Lab analysis button.
    analyse_btn = document.getElementById("analyse-btn")
    analyse_proxy = create_proxy(analyse_overfitting)
    analyse_btn.addEventListener("click", analyse_proxy)

    status_el = document.getElementById("status")
    status_el.innerText = "Engine Ready"


# Initialise when script loads
setup()
