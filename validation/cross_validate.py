"""
Cross-validate the engine against backtrader.

Independent verification. The engine's own golden test
(tests/test_golden.py) proves the engine matches a spec I wrote.
This script instead runs an identical strategy through *both* this engine
and backtrader (pinned 1.9.78.123) and diffs the results.
Testing agreement with an established third-party backtester on the spec
that motivated choosing it: market orders fill at the next bar's open
(backtrader's ``cheat_on_open=False``, our next-open fill timing).

Two runs on AAPL, SMA(5, 20), long-only, fixed 100 shares, 100,000 cash:
  * Run A (zero cost): the exact-match gate. Zero trade mismatches AND final-equity
    relative delta < 1e-4 (1bp) are required, or the script exits non-zero.
  * Run B (with costs): 0.005/share commission, 1.00 minimum, both engines. Slippage
    is 0 in both - backtrader applies no percentage slippage to market orders by
    default, so including it would compare our slipped fills against backtrader's
    un-slipped ones. Run B differences are reported, not asserted.

Usage:
  python validation/cross_validate.py        # run both, write validation/RESULTS.md
  python validation/cross_validate.py --ci    # assert Run A only, write nothing

Pyodide note: backtrader is a CLI-only developer dependency. It must never be
imported from the engine modules (data/engine/portfolio/execution/...), which run
unchanged inside the browser under Pyodide where backtrader is unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import UTC, date, datetime

# backtrader 1.9.78.123 ships docstrings with invalid escape sequences that raise
# SyntaxWarning on import under modern Python; they are harmless noise.
warnings.filterwarnings("ignore", category=SyntaxWarning)

# The validation scripts live in a subdirectory, so the repo root is not on the
# import path when this file is run as `python validation/cross_validate.py`
# (only validation/ would be). Add the root before importing the engine modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt

from data import CSVDataHandler
from engine import Backtest
from event import Event
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from position_sizing import FixedSizer
from strategy import SimpleMovingAverageStrategy

# --- Shared experiment parameters -----------------------------------------------
SYMBOL = "AAPL"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, f"{SYMBOL}.csv")
SHORT_WINDOW = 5
LONG_WINDOW = 20
QUANTITY = 100
INITIAL_CASH = 100_000.0

COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")


@dataclass(frozen=True)
class Trade:
    """A single fill, normalised so both engines produce comparable tuples.

    ``quantity`` is unsigned and the direction lives in ``side`` (backtrader
    records a negative size for sells; we record a positive quantity plus a side
    string).
    """

    day: date
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float

    def key(self) -> tuple[date, str, float, float]:
        return (self.day, self.side, self.quantity, round(self.price, 6))


# --- Our engine -----------------------------------------------------------------
def run_ours(
    commission_per_share: float,
    min_commission: float,
    slippage_pct: float,
) -> tuple[list[Trade], float, int]:
    """Runs the real stack and returns (trades, final_equity, dropped_orders)."""
    events: deque[Event] = deque()
    data_handler = CSVDataHandler(events, DATA_DIR, [SYMBOL])
    strategy = SimpleMovingAverageStrategy(
        events, short_window=SHORT_WINDOW, long_window=LONG_WINDOW
    )
    portfolio = Portfolio(
        events, initial_capital=INITIAL_CASH, sizer=FixedSizer(QUANTITY)
    )
    execution_handler = SimulatedExecutionHandler(
        events,
        data_handler,
        portfolio,
        commission_per_share=commission_per_share,
        min_commission=min_commission,
        slippage_pct=slippage_pct,
    )
    backtest = Backtest(data_handler, strategy, portfolio, execution_handler, events)
    asyncio.run(backtest.run())

    trades = [
        Trade(
            day=datetime.fromtimestamp(t["timestamp"], UTC).date(),
            side=t["side"],
            quantity=abs(t["quantity"]),
            price=t["price"],
        )
        for t in portfolio.trades
    ]
    return trades, portfolio.total_equity(), execution_handler.dropped_orders


# --- backtrader -----------------------------------------------------------------
class PerShareCommission(bt.CommInfoBase):
    """max(per_share * shares, minimum) total dollars per fill.

    backtrader's commissions are percentage-per-notional or fixed-per-order;
    neither expresses a per-share charge with a floor, so we override the hook
    it calls for every (pseudo) execution.
    """

    params = (
        ("per_share", COMMISSION_PER_SHARE),
        ("minimum", MIN_COMMISSION),
        ("stocklike", True),
    )

    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        return max(self.p.per_share * abs(size), self.p.minimum)


class StateSMA(bt.Strategy):
    """State-based SMA crossover mirroring SimpleMovingAverageStrategy exactly.

    We deliberately do NOT use ``bt.ind.CrossOver``: our rule is a level
    comparison with intent tracking (enter as soon as short > long and we are not
    already long), not an edge detector, so it enters immediately on the first
    evaluated bar if short > long. Both engines first evaluate at bar index
    ``long - 1`` (the SMA warm-up), which is what keeps the trade lists aligned.
    """

    params = (("short", SHORT_WINDOW), ("long", LONG_WINDOW), ("qty", QUANTITY))

    def __init__(self) -> None:
        self.sma_short = bt.ind.SMA(self.data.close, period=self.p.short)
        self.sma_long = bt.ind.SMA(self.data.close, period=self.p.long)
        self.intent: str | None = None
        self.fills: list[Trade] = []

    def next(self) -> None:
        if self.sma_short[0] > self.sma_long[0] and self.intent != "LONG":
            self.buy(size=self.p.qty)
            self.intent = "LONG"
        elif self.sma_short[0] < self.sma_long[0] and self.intent == "LONG":
            self.sell(size=self.p.qty)
            self.intent = None

    def notify_order(self, order: bt.Order) -> None:
        if order.status != order.Completed:
            return
        self.fills.append(
            Trade(
                day=bt.num2date(order.executed.dt).date(),
                side="BUY" if order.isbuy() else "SELL",
                quantity=abs(order.executed.size),
                price=order.executed.price,
            )
        )


def run_backtrader(with_costs: bool) -> tuple[list[Trade], float]:
    """Runs backtrader and returns (trades, final_equity)."""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    if with_costs:
        cerebro.broker.addcommissioninfo(PerShareCommission())

    data = bt.feeds.GenericCSVData(
        dataname=DATA_PATH,
        dtformat="%Y-%m-%d",
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        headers=True,
    )
    cerebro.adddata(data)
    cerebro.addstrategy(StateSMA)

    strat = cerebro.run()[0]
    return strat.fills, cerebro.broker.getvalue()


# --- Comparison -----------------------------------------------------------------
@dataclass
class Comparison:
    label: str
    ours: list[Trade]
    theirs: list[Trade]
    equity_ours: float
    equity_theirs: float

    @property
    def mismatches(self) -> list[tuple[int, Trade | None, Trade | None]]:
        """Aligned rows where the two engines disagree (or lengths differ)."""
        rows: list[tuple[int, Trade | None, Trade | None]] = []
        for i in range(max(len(self.ours), len(self.theirs))):
            o = self.ours[i] if i < len(self.ours) else None
            t = self.theirs[i] if i < len(self.theirs) else None
            ok = o is not None and t is not None and o.key() == t.key()
            if not ok:
                rows.append((i, o, t))
        return rows

    @property
    def equity_rel_delta(self) -> float:
        denom = abs(self.equity_theirs) if self.equity_theirs else 1.0
        return abs(self.equity_ours - self.equity_theirs) / denom

    @property
    def trades_match(self) -> bool:
        return len(self.mismatches) == 0


def _fmt_trade(trade: Trade | None) -> str:
    if trade is None:
        return "-"
    return f"{trade.day} {trade.side} {trade.quantity:g} @ {trade.price:.6f}"


def _trade_table(trades: list[Trade], head: int = 5, tail: int = 5) -> str:
    """Markdown table of trades, abbreviated to head+tail when long."""
    lines = ["| # | date | side | qty | price |", "|---|---|---|---|---|"]

    def row(i: int, tr: Trade) -> str:
        return f"| {i} | {tr.day} | {tr.side} | {tr.quantity:g} | {tr.price:.6f} |"

    if len(trades) <= head + tail:
        lines += [row(i, tr) for i, tr in enumerate(trades)]
    else:
        lines += [row(i, trades[i]) for i in range(head)]
        lines.append(f"| … | _{len(trades) - head - tail} rows omitted_ | | | |")
        lines += [row(i, trades[i]) for i in range(len(trades) - tail, len(trades))]
    return "\n".join(lines)


def _mismatch_table(comp: Comparison) -> str:
    if comp.trades_match:
        return "_No mismatches._"
    lines = ["| # | ours | backtrader |", "|---|---|---|"]
    for i, o, t in comp.mismatches:
        lines.append(f"| {i} | {_fmt_trade(o)} | {_fmt_trade(t)} |")
    return "\n".join(lines)


def _print_comparison(comp: Comparison) -> None:
    status = "MATCH" if comp.trades_match else f"{len(comp.mismatches)} MISMATCH(ES)"
    print(f"\n=== {comp.label} ===")
    print(f"  trades: ours={len(comp.ours)} backtrader={len(comp.theirs)} -> {status}")
    print(
        f"  final equity: ours={comp.equity_ours:.6f} "
        f"backtrader={comp.equity_theirs:.6f} "
        f"(rel delta {comp.equity_rel_delta:.2e})"
    )
    for i, o, t in comp.mismatches[:10]:
        print(f"    [{i}] ours={_fmt_trade(o)} | bt={_fmt_trade(t)}")


# --- RESULTS.md -----------------------------------------------------------------
def _results_markdown(run_a: Comparison, run_b: Comparison, dropped_a: int) -> str:
    a_pass = run_a.trades_match and run_a.equity_rel_delta < 1e-4

    def analysis(comp: Comparison, gated: bool, dropped: int | None) -> str:
        paras = []
        if comp.trades_match:
            paras.append(
                f"All {len(comp.ours)} fills match on date, side, quantity and price "
                "to 6 decimal places. A signal on bar t's close fills at bar t+1's "
                "open in both engines, so the fill dates line up bar for bar."
            )
        else:
            paras.append(
                f"{len(comp.mismatches)} of {max(len(comp.ours), len(comp.theirs))} "
                "trades differ. They are listed in the mismatch table above."
            )
        if dropped == 0:
            paras.append(
                "No orders were dropped at the end of the data. The last signal is "
                "an exit that fills on the final bar, so neither engine is left with "
                "a pending order. If one were left, our engine would drop it and "
                "count it in `dropped_orders`, and backtrader would also leave it "
                "unfilled, so the two would still agree."
            )
        elif dropped:
            paras.append(
                f"Our engine dropped {dropped} unfilled order(s) at the end of the "
                "data and counted them in `dropped_orders`. backtrader also leaves "
                "these orders unfilled, so neither engine reports them as fills."
            )
        rel = comp.equity_rel_delta
        if rel < 1e-4:
            paras.append(f"Final equity matches to a relative difference of {rel:.2e}.")
        elif gated:
            paras.append(
                f"Final equity differs by a relative {rel:.2e}, which is outside "
                "the 0.01% tolerance."
            )
        else:
            paras.append(
                f"Final equity differs by a relative {rel:.2e}. Run B is not gated, "
                "so this is reported only."
            )
        return "\n\n".join(paras)

    return f"""# Cross-validation: engine vs backtrader

Generated by `validation/cross_validate.py`. Re-run the script to update this file rather than editing it by hand.

The same strategy runs through this engine and backtrader {bt.__version__}, and every fill and the final equity are compared. The strategy is a state-based SMA({SHORT_WINDOW}, {LONG_WINDOW}) crossover on {SYMBOL}, long-only, trading a fixed {QUANTITY} shares from {INITIAL_CASH:,.0f} starting cash. Both engines fill market orders at the next bar's open.

## Verdict

Run A has no costs and is the pass/fail gate. It passes when every trade matches and final equity agrees to within 0.01%. Result: {"PASS" if a_pass else "FAIL"} ({len(run_a.ours)} trades, equity difference {run_a.equity_rel_delta:.2e}).

Run B adds commission and is reported but not gated. Slippage stays at 0 in both runs because backtrader applies none to market orders by default, so including ours would create a difference that isn't real.

## Run A: no costs

Our engine made {len(run_a.ours)} trades and backtrader made {len(run_a.theirs)}. Final equity is {run_a.equity_ours:.6f} for ours and {run_a.equity_theirs:.6f} for backtrader.

### Trades (ours)

{_trade_table(run_a.ours)}

### Mismatches

{_mismatch_table(run_a)}

### Analysis

{analysis(run_a, True, dropped_a)}

## Run B: with commission

Commission is `max({COMMISSION_PER_SHARE} * shares, {MIN_COMMISSION})` dollars per fill in both engines: `SimulatedExecutionHandler` on our side, and a custom `bt.CommInfoBase._getcommission` on backtrader's.

Our engine made {len(run_b.ours)} trades and backtrader made {len(run_b.theirs)}. Final equity is {run_b.equity_ours:.6f} for ours and {run_b.equity_theirs:.6f} for backtrader.

### Trades (ours)

{_trade_table(run_b.ours)}

### Mismatches

{_mismatch_table(run_b)}

### Analysis

{analysis(run_b, False, None)}
"""


# --- Entry point ----------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-validate against backtrader.")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Assert Run A only and write nothing (for the CI gate).",
    )
    args = parser.parse_args(argv)

    ours_a, equity_ours_a, dropped_a = run_ours(0.0, 0.0, 0.0)
    theirs_a, equity_theirs_a = run_backtrader(with_costs=False)
    run_a = Comparison(
        "Run A (zero cost)", ours_a, theirs_a, equity_ours_a, equity_theirs_a
    )
    _print_comparison(run_a)

    a_ok = run_a.trades_match and run_a.equity_rel_delta < 1e-4

    if args.ci:
        if a_ok:
            print("\nRun A gate: PASS")
            return 0
        print("\nRun A gate: FAIL", file=sys.stderr)
        return 1

    ours_b, equity_ours_b, _ = run_ours(COMMISSION_PER_SHARE, MIN_COMMISSION, 0.0)
    theirs_b, equity_theirs_b = run_backtrader(with_costs=True)
    run_b = Comparison(
        "Run B (with costs)", ours_b, theirs_b, equity_ours_b, equity_theirs_b
    )
    _print_comparison(run_b)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(_results_markdown(run_a, run_b, dropped_a))
    print(f"\nWrote {RESULTS_PATH}")

    return 0 if a_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
