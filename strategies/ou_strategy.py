import math
from collections import deque

import numpy as np

from event import Event, MarketEvent, SignalEvent
from strategy import Strategy  # Assuming Abstract Base Class is defined


class OrnsteinUhlenbeckStrategy(Strategy):
    """
    Dynamically estimates the parameters of an Ornstein-Uhlenbeck
    process using a rolling window of prices to generate mean-reversion signals.

    The OU process is fit on **log-prices** (``ln P``), so the calibrated mean
    ``mu`` and equilibrium standard deviation ``sigma_eq`` live in log-price
    space and the resulting z-score is scale-invariant: a 1% move contributes
    the same z regardless of the absolute price level.
    """

    def __init__(
        self,
        events: deque[Event] | None = None,
        symbol: str = "",
        window_size: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        allow_short: bool = False,
    ):
        """
        Args:
            events: The shared event queue. Generated signals are appended
                to it for the engine. When omitted the base class creates a
                private queue, so the strategy can be exercised directly (e.g.
                in unit tests) by reading the signals back off ``self.events``.
            symbol: The ticker symbol being traded.
            window_size: Number of periods to use for OLS calibration.
            entry_z: The Z-score threshold to enter a trade.
            exit_z: The Z-score threshold to exit a trade (usually 0, the mean).
            allow_short: When False (the default) the strategy is long-only and
                only enters when price is below equilibrium. When True it also
                shorts when price is above equilibrium.
        """
        super().__init__(events, allow_short)
        self.symbol = symbol
        self.window_size = window_size
        self.entry_z = entry_z
        self.exit_z = exit_z

        # Rolling window of log-prices used for calibration. Position state
        # (intent / position) is inherited from Strategy and keyed by symbol.
        self.prices: deque[float] = deque(maxlen=window_size)

    def _calibrate_ou_parameters(self) -> tuple[float, float, float]:
        """
        Maps the OU process to an AutoRegressive AR(1) model: x_t - x_{t-1} = a + b*x_{t-1} + error
        Returns the dynamic mean, standard deviation, and a valid flag.
        """
        # Convert deque to numpy array for vector math
        price_series = np.array(self.prices)

        # x is lagged prices (t-1), y is price differences (t)
        x = price_series[:-1]
        y = np.diff(price_series)

        # Perform Linear Regression (OLS) -> y = mx + c
        # np.polyfit returns [slope (b), intercept (a)]
        b, a = np.polyfit(x, y, 1)

        # Calculate OU Parameters (assuming dt = 1)
        theta = -b

        # If theta is non-positive the series is diverging or a pure random
        # walk (not mean reverting). A small epsilon guards against floating
        # point noise from making a flat/trending series look mean-reverting.
        if theta <= 1e-8:
            return 0.0, 0.0, False

        mu = a / theta

        # Calculate the equilibrium standard deviation
        residuals = y - (a + b * x)
        sigma = np.std(residuals, ddof=1)

        # Equilibrium variance of the OU process is sigma^2 / 2*theta
        sigma_eq = sigma / np.sqrt(2 * theta)

        return mu, sigma_eq, True

    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Processes new market data and emits signals if thresholds are breached.

        Signals are appended to ``self.events``; callers exercising the strategy
        without an engine read them back off that queue.
        """
        # Ignore events for other symbols
        if event.symbol != self.symbol:
            return

        # Update our rolling window with the log-price. A non-positive price has
        # no logarithm, so skip the bar rather than corrupt the window.
        current_price = event.close
        if current_price <= 0:
            return
        log_price = math.log(current_price)
        self.prices.append(log_price)

        # Wait until the window is fully populated
        if len(self.prices) < self.window_size:
            return

        # 1. Calibrate the SDE (on log-prices)
        mu, sigma_eq, is_mean_reverting = self._calibrate_ou_parameters()

        if not is_mean_reverting or sigma_eq == 0:
            return  # Process is wandering, do not trade

        # 2. Calculate current Z-Score relative to the dynamic OU equilibrium.
        # mu / sigma_eq are in log-price space, so the current price must be too.
        z_score = (log_price - mu) / sigma_eq

        # 3. Generate Trading Logic. Decisions key off intent (what we've asked
        # for), not fill-truth, so a signal is not re-emitted while its order is
        # still pending its next-open fill.
        signal: SignalEvent | None = None
        current_intent = self.intent.get(self.symbol)

        if current_intent is None:
            # Price is too high -> Expect reversion down -> SHORT
            if z_score > self.entry_z and self.allow_short:
                signal = SignalEvent(self.symbol, event.timestamp, "SHORT")
                self.intent[self.symbol] = "SHORT"

            # Price is too low -> Expect reversion up -> LONG
            elif z_score < -self.entry_z:
                signal = SignalEvent(self.symbol, event.timestamp, "LONG")
                self.intent[self.symbol] = "LONG"

        else:  # We are already in a trade, look for exit conditions
            if current_intent == "LONG" and z_score >= self.exit_z:
                signal = SignalEvent(self.symbol, event.timestamp, "EXIT")
                self.intent[self.symbol] = None

            elif current_intent == "SHORT" and z_score <= self.exit_z:
                signal = SignalEvent(self.symbol, event.timestamp, "EXIT")
                self.intent[self.symbol] = None

        # Push the signal onto the shared event queue so the engine's event
        # loop can route it to the portfolio.
        if signal is not None:
            self.events.append(signal)
