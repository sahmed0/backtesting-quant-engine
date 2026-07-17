"""
Portfolio module for the backtesting engine.
"""

from queue import Queue
from typing import Any

import pandas as pd


from event import FillEvent, MarketEvent, OrderEvent, SignalEvent
from position_sizing import FixedSizer, PositionSizer


class Portfolio:
    """
    Tracks positions, holdings, and calculates total equity over time.
    """

    def __init__(
        self,
        events: Queue,
        initial_capital: float = 100000.0,
        sizer: PositionSizer | None = None,
    ):
        """
        Initialises the portfolio with a starting capital and a position sizer.

        Args:
            events: The shared event queue.
            initial_capital: Starting cash.
            sizer: Strategy for converting signals into order quantities. When
                omitted, defaults to a fixed 100-unit size to preserve the
                engine's original behaviour.
        """
        self.events = events
        self.initial_capital = initial_capital
        self.sizer = sizer if sizer is not None else FixedSizer(100.0)
        self.current_cash = initial_capital

        self.trades: list[dict[str, Any]] = []

        # symbol: quantity
        self.current_positions: dict[str, float] = {}
        # List of historical positions snapshots
        self.all_positions: list[dict[str, Any]] = []

        # symbol: market_value
        self.current_holdings: dict[str, float] = {}
        # List of historical holdings snapshots
        self.all_holdings: list[dict[str, Any]] = []

        # symbol: current_price
        self.current_prices: dict[str, float] = {}

    def update_timeindex(self, event: MarketEvent) -> None:
        """
        Updates the portfolio holdings based on a new market event,
        and re-calculates total equity.
        """
        symbol = event.symbol
        price = event.close
        timestamp = event.timestamp

        self.current_prices[symbol] = price

        # Feed the bar to the sizer so volatility/price-history based sizers can
        # maintain their estimates. Signals alone are too sparse for this.
        self.sizer.update_market(symbol, price, event.high, event.low)

        # Ensure symbol exists in positions
        if symbol not in self.current_positions:
            self.current_positions[symbol] = 0.0

        quantity = self.current_positions[symbol]
        market_value = quantity * price

        self.current_holdings[symbol] = market_value
        # 'price' is stored purely as chart metadata for the latest bar; it is
        # not a position value and must be excluded from the equity total.
        self.current_holdings["price"] = price

        # Calculate total equity
        total_market_value = sum(
            value
            for key, value in self.current_holdings.items()
            if key not in ("cash", "total", "timestamp", "price")
        )
        total_equity = self.current_cash + total_market_value

        # Update current holdings with cash, total, and timestamp
        self.current_holdings["cash"] = self.current_cash
        self.current_holdings["total"] = total_equity
        self.current_holdings["timestamp"] = (
            timestamp.timestamp()
        )  # convert datetime to float

        # Append snapshots to history
        pos_snapshot = self.current_positions.copy()
        pos_snapshot["timestamp"] = timestamp.timestamp()  # convert datetime to float
        self.all_positions.append(pos_snapshot)

        self.all_holdings.append(self.current_holdings.copy())

    def total_equity(self) -> float:
        """
        Returns current total equity (cash plus the marked-to-market value of
        all open positions). Computed on demand so position sizers can size off
        equity even before the first holdings snapshot is recorded.
        """
        market_value = sum(
            qty * self.current_prices.get(symbol, 0.0)
            for symbol, qty in self.current_positions.items()
        )
        return self.current_cash + market_value

    def update_signal(self, event: SignalEvent) -> None:
        """
        Acts on a SignalEvent to generate new orders based on the portfolio logic.
        """
        symbol = event.symbol
        direction = event.direction
        timestamp = event.timestamp

        current_price = self.current_prices.get(symbol, 0.0)

        if direction == "EXIT":
            # Flatten whatever position exists, long or short. Exits are sized
            # from the held quantity, not the sizer.
            current_qty = self.current_positions.get(symbol, 0.0)
            if current_qty != 0:
                order = OrderEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    quantity=abs(current_qty),
                    direction="EXIT",
                    order_type="MARKET",
                )
                self.events.put(order)
            return

        # Size new entries via the configured position sizer. A non-positive
        # quantity means the sizer declined to trade (e.g. insufficient history).
        order_quantity = self.sizer.size(symbol, direction, current_price, self)
        if order_quantity <= 0:
            return

        if direction == "LONG":
            estimated_cost = order_quantity * current_price
            if estimated_cost > 0 and self.current_cash >= estimated_cost:
                order = OrderEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    quantity=order_quantity,
                    direction="LONG",
                    order_type="MARKET",
                )
                self.events.put(order)

        elif direction == "SHORT":
            # Opening a short sells shares we don't hold, which generates cash,
            # so there is no up-front cash requirement to check here.
            if current_price > 0:
                order = OrderEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    quantity=order_quantity,
                    direction="SHORT",
                    order_type="MARKET",
                )
                self.events.put(order)

    def update_fill(self, event: FillEvent) -> None:
        """
        Updates portfolio current positions and cash from a FillEvent.
        """
        symbol = event.symbol
        quantity = event.quantity
        direction = event.direction
        fill_price = event.fill_price
        commission = event.commission
        slippage = event.slippage

        fill_cost = fill_price * quantity
        total_cost = fill_cost + commission + slippage

        if symbol not in self.current_positions:
            self.current_positions[symbol] = 0.0

        if direction == "LONG":
            self.current_positions[symbol] += quantity
            self.current_cash -= total_cost
        elif direction == "SHORT":
            # Opening/adding to a short: sell shares and receive the proceeds
            # net of transaction costs.
            self.current_positions[symbol] -= quantity
            self.current_cash += fill_cost - commission - slippage
        elif direction == "EXIT":
            # Closing a position: sell to close a long, or buy to cover a short.
            if self.current_positions[symbol] >= 0:
                self.current_positions[symbol] -= quantity
                self.current_cash += fill_cost - commission - slippage
            else:
                self.current_positions[symbol] += quantity
                self.current_cash -= total_cost

        self.trades.append(
            {
                "timestamp": event.timestamp.timestamp(),
                "symbol": symbol,
                "direction": direction,
                "quantity": quantity,
                "price": fill_price,
                "commission": commission,
                "slippage": slippage,
            }
        )

    def generate_equity_curve(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame of the total equity over time.
        """
        if not self.all_holdings:
            return pd.DataFrame()

        df = pd.DataFrame(self.all_holdings)

        # In Pandas, filter columns by passing a list to the indexer
        if "price" in df.columns:
            return df[["timestamp", "total", "price"]]
        return df[["timestamp", "total"]]
