import asyncio
import csv
import json
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


def _set_icon(element, icon_class, text=""):
    """Sets an element's content to a Font Awesome ``<i>`` plus optional text,
    built from DOM nodes rather than a raw markup string (no injection path)."""
    icon = document.createElement("i")
    icon.className = icon_class
    if text:
        element.replaceChildren(icon, document.createTextNode(text))
    else:
        element.replaceChildren(icon)


def render_order_book(trades):
    """Populates the Order Book table directly from ``portfolio.trades`` after a
    run, newest fill first. Rows are built with ``createElement``/``textContent``
    only - no log scraping, no raw markup. ``trades`` dicts carry every column
    (see ``Portfolio.update_fill``)."""
    tbody = document.getElementById("order-log-body")
    if tbody is None:
        return
    tbody.replaceChildren()
    for trade in reversed(trades):
        direction = trade["direction"]
        date_str = datetime.fromtimestamp(trade["timestamp"], UTC).strftime("%Y-%m-%d")
        cells = [
            (date_str, None),
            (direction, f"dir-{direction.lower()}"),
            (f"{trade['quantity']:.0f}", None),
            (trade["symbol"], None),
            (f"{trade['price']:.2f}", None),
            (f"{trade['commission']:.4f}", None),
            (f"{trade['slippage']:.4f}", None),
        ]
        tr = document.createElement("tr")
        for value, css_class in cells:
            td = document.createElement("td")
            td.textContent = value
            if css_class:
                td.className = css_class
            tr.appendChild(td)
        tbody.appendChild(tr)


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
        symbol = re.sub(r"[^A-Za-z0-9._=-]", "_", os.path.splitext(file.name)[0])
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
    Runs the SMA parameter grid over [start, end] and returns
    ``(grid, moments, runs_done)``. ``grid`` is a 2D list of Sharpe ratios
    indexed [short_idx][long_idx]; ``moments`` is a parallel grid of
    ``(sr_period, n_obs, skew, kurt)`` tuples (per-period, non-annualised) for
    the Deflated Sharpe Ratio. Cells where short >= long, or that produce no
    usable curve, are None in both grids. Yields to the event loop between runs
    so the progress status can repaint.
    """
    grid = []
    moments = []
    runs_done = 0
    for short_w in OF_SHORT_WINDOWS:
        row = []
        mrow = []
        for long_w in OF_LONG_WINDOWS:
            if short_w >= long_w:
                row.append(None)
                mrow.append(None)
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
            if "error" in stats:
                row.append(None)
                mrow.append(None)
            else:
                row.append(stats["sharpe_ratio"])
                # Per-period Sharpe and higher moments of this run's per-bar
                # equity returns, for the Deflated Sharpe of the IS-best cell.
                df = portfolio.generate_equity_curve()
                returns = df["total"].pct_change().dropna().to_numpy()
                if len(returns) >= 2 and np.std(returns, ddof=1) > 0:
                    sr_period = float(np.mean(returns) / np.std(returns, ddof=1))
                    skew, kurt = performance.returns_moments(returns)
                    mrow.append((sr_period, len(returns), skew, kurt))
                else:
                    mrow.append(None)
            runs_done += 1
            # Hand control back to the browser so the status text can update.
            await asyncio.sleep(0)
        grid.append(row)
        moments.append(mrow)
    return grid, moments, runs_done


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
    _set_icon(analyse_btn, "fa-solid fa-spinner fa-spin", " Analysing...")

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
        is_grid, is_moments, done_is = await _grid_sharpe(
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
        oos_grid, _, _ = await _grid_sharpe(
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

        # Deflated Sharpe Ratio of the in-sample pick: correct the IS-best
        # Sharpe for having tried this many configurations. Variance of the
        # per-period Sharpes across all valid cells drives the deflation.
        sr_periods = [m[0] for m_row in is_moments for m in m_row if m is not None]
        n_trials = len(sr_periods)
        sr_variance = float(np.var(sr_periods, ddof=1)) if n_trials >= 2 else 0.0
        best_m = is_moments[is_best_i][is_best_j]
        if best_m is not None and n_trials >= 2:
            best_sr, best_n_obs, best_skew, best_kurt = best_m
            dsr = performance.deflated_sharpe_ratio(
                best_sr, sr_variance, n_trials, best_n_obs, best_skew, best_kurt
            )
        else:
            dsr = 0.0

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
            "dsr": dsr,
            "dsr_n_trials": n_trials,
        }
        window.updateHeatmaps(json.dumps(payload))
        _set_icon(status_el, "fa-solid fa-check text-success")

    except Exception as e:
        error_output.innerText = f"Error: {str(e)}\n{traceback.format_exc()}"
        status_el.innerText = "An error occurred during analysis."
    finally:
        analyse_btn.disabled = False
        run_btn.disabled = False
        _set_icon(
            analyse_btn, "fa-solid fa-magnifying-glass-chart", " Analyse Overfitting"
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

    # Clear previous logs and the bootstrap-CI caption from any prior run.
    document.getElementById("order-log-body").replaceChildren()
    document.getElementById("cap-sharpe-ci").innerText = ""

    try:
        ticker_select = document.getElementById("ticker-select")

        files = file_input.files
        if files and files.length > 0:
            # Handle uploaded file (Priority)
            file = files.item(0)
            text_content = await file.text()

            # Write to virtual file system. Sanitise the filename-derived symbol
            # so it can't escape /data or inject odd characters downstream.
            os.makedirs("/data", exist_ok=True)
            symbol = re.sub(r"[^A-Za-z0-9._=-]", "_", os.path.splitext(file.name)[0])
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
        # browser periodically and calls back with its bar count; the button
        # text is left alone because app.js keys its busy state off it.
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

            # Bootstrap 95% CI on the annualised Sharpe: 500 stationary-bootstrap
            # resamples in 10 batches of 50, yielding to the browser between
            # batches so the tab stays responsive. Needs a reasonable sample;
            # skipped (caption left empty) below 30 returns.
            returns = (
                portfolio.generate_equity_curve()["total"]
                .pct_change()
                .dropna()
                .to_numpy()
            )
            if len(returns) >= 30:
                status_el.innerText = "Bootstrapping Sharpe CI..."
                rng = np.random.default_rng(42)
                samples: list[float] = []
                for _ in range(10):
                    samples.extend(
                        performance.bootstrap_sharpe_samples(
                            returns, stats["periods_per_year"], 50, rng
                        )
                    )
                    await asyncio.sleep(0)
                ci_lo = float(np.percentile(samples, 2.5))
                ci_hi = float(np.percentile(samples, 97.5))
                document.getElementById(
                    "cap-sharpe-ci"
                ).innerText = f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}] (bootstrap)"

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

            # Build the Order Book straight from the recorded fills (newest first).
            render_order_book(portfolio.trades)

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
