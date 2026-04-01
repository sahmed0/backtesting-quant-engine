from multiprocessing import Queue

import numpy as np
from collections import deque
from event import MarketEvent, SignalEvent
from strategy import Strategy # Assuming Abstract Base Class is defined

class OrnsteinUhlenbeckStrategy(Strategy):
    """
    Dynamically estimates the parameters of an Ornstein-Uhlenbeck 
    process using a rolling window of prices to generate mean-reversion signals.
    """
    
    def __init__(self, eventsQueue: Queue, symbol: str, window_size: int = 60, entry_z: float = 2.0, exit_z: float = 0.0):
        """
        Args:
            symbol: The ticker symbol being traded.
            window_size: Number of periods to use for OLS calibration.
            entry_z: The Z-score threshold to enter a trade.
            exit_z: The Z-score threshold to exit a trade (usually 0, the mean).
        """
        super().__init__(eventsQueue)
        self.symbol = symbol
        self.window_size = window_size
        self.entry_z = entry_z
        self.exit_z = exit_z
        
        # State tracking
        self.prices = deque(maxlen=window_size)
        self.invested = False  # 'LONG', 'SHORT', or False
        
    def _calibrate_ou_parameters(self) -> tuple[float, float, float]:
        """
        Maps the OU process to an AutoRegressive AR(1) model: x_t - x_{t-1} = a + b*x_{t-1} + error
        Returns the dynamic mean, standard deviation, and a valid flag.
        """
        # Convert deque to numpy array for vector math
        P = np.array(self.prices)
        
        # x is lagged prices (t-1), y is price differences (t)
        x = P[:-1]
        y = np.diff(P)
        
        # Perform Linear Regression (OLS) -> y = mx + c
        # np.polyfit returns [slope (b), intercept (a)]
        b, a = np.polyfit(x, y, 1)
        
        # Calculate OU Parameters (assuming dt = 1)
        theta = -b
        
        # If theta is negative, the series is diverging (not mean reverting)
        if theta <= 0:
            return 0.0, 0.0, False
            
        mu = a / theta
        
        # Calculate the equilibrium standard deviation
        residuals = y - (a + b * x)
        sigma = np.std(residuals)
        
        # Equilibrium variance of the OU process is sigma^2 / 2*theta
        sigma_eq = sigma / np.sqrt(2 * theta)
        
        return mu, sigma_eq, True

    def calculate_signals(self, event: MarketEvent) -> SignalEvent | None:
        """
        Processes new market data and emits signals if thresholds are breached.
        """
        # Ensure the event is for our symbol
        if event.symbol != self.symbol:
            return None
            
        # Update our rolling window
        current_price = event.close
        self.prices.append(current_price)
        
        # Wait until the window is fully populated
        if len(self.prices) < self.window_size:
            return None
            
        # 1. Calibrate the SDE
        mu, sigma_eq, is_mean_reverting = self._calibrate_ou_parameters()
        
        if not is_mean_reverting or sigma_eq == 0:
            return None # Process is wandering, do not trade
            
        # 2. Calculate current Z-Score relative to the dynamic OU equilibrium
        z_score = (current_price - mu) / sigma_eq
        
        # 3. Generate Trading Logic
        signal = None
        
        if not self.invested:
            # Price is too high -> Expect reversion down -> SHORT
            if z_score > self.entry_z:
                signal = SignalEvent(self.symbol, event.timestamp, 'SHORT')
                self.invested = 'SHORT'
                
            # Price is too low -> Expect reversion up -> LONG
            elif z_score < -self.entry_z:
                signal = SignalEvent(self.symbol, event.timestamp, 'LONG')
                self.invested = 'LONG'
                
        else: # We are already in a trade, look for exit conditions
            if self.invested == 'LONG' and z_score >= self.exit_z:
                signal = SignalEvent(self.symbol, event.timestamp, 'EXIT')
                self.invested = False
                
            elif self.invested == 'SHORT' and z_score <= self.exit_z:
                signal = SignalEvent(self.symbol, event.timestamp, 'EXIT')
                self.invested = False
                
        return signal