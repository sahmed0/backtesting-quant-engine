import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from event import Event, MarketEvent, SignalEvent
from strategy import Strategy

# MacKinnon (2010) response-surface coefficients for the Dickey-Fuller tau
# statistic with a constant and no trend: crit = b0 + b1/T + b2/T^2 + b3/T^3.
MACKINNON_TAU_C: dict[float, tuple[float, float, float, float]] = {
    0.01: (-3.43035, -6.5393, -16.786, -79.433),
    0.05: (-2.86154, -2.8903, -4.234, -40.040),
    0.10: (-2.56677, -1.5384, -2.809, 0.0),
}

GATE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)

GATE_LABELS: dict[float, str] = {
    0.01: "1% level",
    0.05: "5% level",
    0.10: "10% level",
}


def dickey_fuller_critical_value(level: float, n_obs: int) -> float:
    """Critical value of the Dickey-Fuller t-statistic at `level` for a regression on `n_obs` observations."""
    if level not in MACKINNON_TAU_C:
        raise ValueError(f"No Dickey-Fuller critical values for level {level}.")
    if n_obs < 3:
        raise ValueError("The Dickey-Fuller test needs at least 3 observations.")
    b0, b1, b2, b3 = MACKINNON_TAU_C[level]
    return b0 + b1 / n_obs + b2 / n_obs**2 + b3 / n_obs**3


@dataclass(frozen=True)
class OUCalibration:
    mu: float  # equilibrium log price
    sigma_eq: float  # equilibrium standard deviation of the log price
    theta: float  # mean-reversion speed per bar
    half_life: float  # ln 2 / theta, in bars
    t_stat: float  # Dickey-Fuller t-statistic of b
    mean_reverting: bool  # 0 < phi < 1, so mu, sigma_eq, theta and half_life are valid
    passed: bool  # mean_reverting and the Dickey-Fuller test (if any) passed: a trade may be opened


def _failed(t_stat: float = math.nan) -> OUCalibration:
    return OUCalibration(0.0, 0.0, 0.0, math.inf, t_stat, False, False)


def calibrate_ou(log_prices: np.ndarray, gate_level: float | None) -> OUCalibration:
    """
    Fits dx_t = a + b * x_{t-1} + e_t by OLS on a window of log prices and maps
    it exactly to an OU process: phi = 1 + b, theta = -ln(phi), mu = -a / b,
    sigma_eq = sigma_eps / sqrt(1 - phi^2).

    The window passes when 0 < phi < 1 and, if `gate_level` is set, the
    Dickey-Fuller t-statistic of b is below the critical value at that level.
    """
    x = log_prices[:-1]
    y = np.diff(log_prices)
    n = len(y)
    if n < 3:
        return _failed()

    sxx = float(np.sum((x - np.mean(x)) ** 2))
    if sxx == 0:
        return _failed()

    b, a = np.polyfit(x, y, 1)
    residuals = y - (a + b * x)
    sigma_eps = math.sqrt(float(np.sum(residuals**2)) / (n - 2))
    if sigma_eps == 0:
        return _failed()

    t_stat = float(b / (sigma_eps / math.sqrt(sxx)))
    phi = 1 + b
    if not (b < -1e-8 and phi > 0):
        return _failed(t_stat)

    theta = -math.log(phi)
    mu = -a / b
    sigma_eq = sigma_eps / math.sqrt(1 - phi**2)
    passed = gate_level is None or t_stat < dickey_fuller_critical_value(gate_level, n)
    return OUCalibration(
        float(mu),
        float(sigma_eq),
        float(theta),
        math.log(2) / theta,
        t_stat,
        True,
        passed,
    )


def gate_summary(
    gate_level: float | None,
    windows_tested: int,
    windows_passed: int,
    window_size: int,
    traded: bool,
) -> str:
    """One-line description of how often the mean-reversion check passed."""
    if windows_tested == 0:
        return (
            "The mean-reversion check never ran: the data is shorter than the "
            f"{window_size}-bar window."
        )
    label = "sign check only" if gate_level is None else GATE_LABELS[gate_level]
    pct = 100 * windows_passed / windows_tested
    text = (
        f"Mean-reversion check ({label}) passed on {pct:.1f}% of "
        f"{windows_tested:,} windows. Trades only open while the latest window passes."
    )
    if not traded:
        text += " No trades were made."
    return text


class OrnsteinUhlenbeckStrategy(Strategy):
    """
    Mean-reversion strategy that fits an Ornstein-Uhlenbeck process to a rolling
    window of prices and trades the z-score of the latest price.

    The fit is an AR(1) regression on log prices, so the z-score is scale-invariant.
    The regression maps to the OU parameters exactly (phi = 1 + b, theta = -ln phi),
    not by the Euler approximation.
    A trade opens only when the window passes a Dickey-Fuller test that rejects a
    random walk at the chosen level.
    A held position exits when its z-score reaches the exit level, or at once when
    the window shows no mean reversion at all.
    """

    def __init__(
        self,
        events: deque[Event] | None = None,
        symbol: str = "",
        window_size: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        allow_short: bool = False,
        gate_level: float | None = 0.05,
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
            gate_level: Dickey-Fuller significance level a window must pass
                before a trade is opened (0.01, 0.05 or 0.10), or None to
                require only 0 < phi < 1. A held position is not closed just
                because the test fails.
        """
        super().__init__(events, allow_short)
        if gate_level is not None and gate_level not in GATE_LEVELS:
            raise ValueError(
                f"gate_level must be one of {GATE_LEVELS} or None, got {gate_level}."
            )
        self.symbol = symbol
        self.window_size = window_size
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.gate_level = gate_level
        self.windows_tested = 0
        self.windows_passed = 0

        # Rolling window of log-prices used for calibration. Position state
        # (intent / position) is inherited from Strategy and keyed by symbol.
        self.prices: deque[float] = deque(maxlen=window_size)

    @property
    def warmup_period(self) -> int:
        return self.window_size - 1

    def prime(self, event: MarketEvent) -> None:
        if event.symbol == self.symbol and event.close > 0:
            self.prices.append(math.log(event.close))

    def _calibrate(self) -> OUCalibration:
        """Fits the OU process to the current window of log prices."""
        return calibrate_ou(np.array(self.prices), self.gate_level)

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

        fit = self._calibrate()
        self.windows_tested += 1
        if fit.passed:
            self.windows_passed += 1

        # Decisions key off intent (what we've asked for), not fill-truth, so a
        # signal is not re-emitted while its order is still pending its
        # next-open fill.
        signal: SignalEvent | None = None
        current_intent = self.intent.get(self.symbol)

        # The test decides when to open a trade. A held position exits on its
        # z-score, using this window's fit even if the window fails the test.
        if current_intent in ("LONG", "SHORT"):
            if not fit.mean_reverting:
                # The window no longer shows any mean reversion, so there is no
                # equilibrium to exit at.
                signal = SignalEvent(self.symbol, event.timestamp, "EXIT")
                self.intent[self.symbol] = None
            else:
                z_score = (log_price - fit.mu) / fit.sigma_eq
                if current_intent == "LONG" and z_score >= self.exit_z:
                    signal = SignalEvent(self.symbol, event.timestamp, "EXIT")
                    self.intent[self.symbol] = None

                elif current_intent == "SHORT" and z_score <= self.exit_z:
                    signal = SignalEvent(self.symbol, event.timestamp, "EXIT")
                    self.intent[self.symbol] = None

        elif fit.passed:
            # mu and sigma_eq are in log-price space, so the current price must
            # be too.
            z_score = (log_price - fit.mu) / fit.sigma_eq

            # Price is too high -> Expect reversion down -> SHORT
            if z_score > self.entry_z and self.allow_short:
                signal = SignalEvent(self.symbol, event.timestamp, "SHORT")
                self.intent[self.symbol] = "SHORT"

            # Price is too low -> Expect reversion up -> LONG
            elif z_score < -self.entry_z:
                signal = SignalEvent(self.symbol, event.timestamp, "LONG")
                self.intent[self.symbol] = "LONG"

        # Push the signal onto the shared event queue so the engine's event
        # loop can route it to the portfolio.
        if signal is not None:
            self.events.append(signal)
