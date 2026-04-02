"""
Portfolio module for the backtesting engine.
"""

from typing import Dict, List, Any
from queue import Queue
import pandas as pd
# import polars as pl # deprecating in favor of Pandas to work with PyScript
from event import MarketEvent, SignalEvent, OrderEvent, FillEvent

class Portfolio:
    """
    Tracks positions, holdings, and calculates total equity over time.
    """

    def __init__(self, events_queue: Queue, initial_capital: float = 100000.0):
        """
        Initialises the portfolio with a starting capital.
        """
        self.events_queue = events_queue
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        
        # symbol: quantity
        self.current_positions: Dict[str, float] = {}
        # List of historical positions snapshots
        self.all_positions: List[Dict[str, Any]] = []
        
        # symbol: market_value
        self.current_holdings: Dict[str, float] = {}
        # List of historical holdings snapshots
        self.all_holdings: List[Dict[str, Any]] = []
        
        # symbol: current_price
        self.current_prices: Dict[str, float] = {}

    def update_timeindex(self, event: MarketEvent) -> None:
        """
        Updates the portfolio holdings based on a new market event,
        and re-calculates total equity.
        """
        symbol = event.symbol
        price = event.close
        timestamp = event.timestamp
        
        self.current_prices[symbol] = price
        
        # Ensure symbol exists in positions
        if symbol not in self.current_positions:
            self.current_positions[symbol] = 0.0
            
        quantity = self.current_positions[symbol]
        market_value = quantity * price
        
        self.current_holdings[symbol] = market_value
        
        # Calculate total equity
        total_market_value = sum(
            value for key, value in self.current_holdings.items() 
            if key not in ('cash', 'total', 'timestamp')
        )
        total_equity = self.current_cash + total_market_value
        
        # Update current holdings with cash, total, and timestamp
        self.current_holdings['cash'] = self.current_cash
        self.current_holdings['total'] = total_equity
        self.current_holdings['timestamp'] = timestamp.timestamp() # convert datetime to float
        
        # Append snapshots to history
        pos_snapshot = self.current_positions.copy()
        pos_snapshot['timestamp'] = timestamp.timestamp() # convert datetime to float
        self.all_positions.append(pos_snapshot)
        
        self.all_holdings.append(self.current_holdings.copy())

    def update_signal(self, event: SignalEvent) -> None:
        """
        Acts on a SignalEvent to generate new orders based on the portfolio logic.
        """
        symbol = event.symbol
        direction = event.direction
        timestamp = event.timestamp
        
        # Naive position sizing: fixed quantity
        # In a real system, you'd calculate this based on risk.
        order_quantity = 100.0 
        
        current_price = self.current_prices.get(symbol, 0.0)
        
        if direction == 'LONG':
            estimated_cost = order_quantity * current_price
            if estimated_cost > 0 and self.current_cash >= estimated_cost:
                order = OrderEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    quantity=order_quantity,
                    direction='LONG',
                    orderType='MARKET'
                )
                self.events_queue.put(order)
                
        elif direction == 'EXIT':
            current_qty = self.current_positions.get(symbol, 0.0)
            if current_qty > 0:
                order = OrderEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    quantity=current_qty,
                    direction='EXIT',
                    orderType='MARKET'
                )
                self.events_queue.put(order)

    def update_fill(self, event: FillEvent) -> None:
        """
        Updates portfolio current positions and cash from a FillEvent.
        """
        symbol = event.symbol
        quantity = event.quantity
        direction = event.direction
        fill_price = event.fillPrice
        commission = event.commission
        slippage = event.slippage
        
        fill_cost = fill_price * quantity
        total_cost = fill_cost + commission + slippage
        
        if symbol not in self.current_positions:
            self.current_positions[symbol] = 0.0
            
        if direction == 'LONG':
            self.current_positions[symbol] += quantity
            self.current_cash -= total_cost
        elif direction in ('SHORT', 'EXIT'):
            self.current_positions[symbol] -= quantity
            # Assuming EXIT implies selling an existing long position.
            # Cash increases by the fill cost, minus transaction costs
            self.current_cash += (fill_cost - commission - slippage)

    """
    # POLARS VERSION - DEPRECATED IN FAVOR OF PANDAS DATEFRAME FOR BETTER COMPATIBILITY WITH PYSCRIPT
    def generate_equity_curve(self) -> pl.DataFrame:
        # Returns a Polars DataFrame of the total equity over time.
        if not self.all_holdings:
            return pl.DataFrame()
            
        df = pl.DataFrame(self.all_holdings)
        return df.select(['timestamp', 'total'])
        """

    def generate_equity_curve(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame of the total equity over time.
        """
        if not self.all_holdings:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.all_holdings)
        
        # In Pandas, filter columns by passing a list to the indexer
        return df[['timestamp', 'total']]