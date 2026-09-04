"""
Position sizing strategies for the backtesting engine.

A :class:`PositionSizer` turns a trade decision (symbol, direction, price) into
a concrete order quantity. The portfolio holds one sizer and consults it for
every entry order, which decouples *how much to trade* from *when to trade*
(the strategy's job) and from bookkeeping (the portfolio's job).

Sizers only size new entries (LONG/SHORT). Exits always flatten the existing
position, so the portfolio sizes those itself from the held quantity.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    # Imported for type hints only; importing at runtime would be circular.
    from portfolio import Portfolio


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def size(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        price: float,
        portfolio: "Portfolio",
    ) -> float:
        """
        Returns the quantity to trade for a new entry. A non-positive return
        means "do not trade" (e.g. not enough history, or zero volatility).
        """
        raise NotImplementedError

    def update_market(
        self,
        symbol: str,
        price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        """
        Hook called by the portfolio on every market bar. The default is a
        no-op; sizers that need a price history (e.g. volatility targeting) or
        a high/low range (e.g. ATR) override this to accumulate it. Signals
        alone are too sparse, so the data has to come from the bar stream.
        """
        return None


class FixedSizer(PositionSizer):
    """
    Trades a constant quantity regardless of price or equity. This reproduces
    the engine's original behaviour and is the default so existing runs are
    unchanged.
    """

    def __init__(self, quantity: float = 100.0):
        self.quantity = quantity

    def size(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        price: float,
        portfolio: "Portfolio",
    ) -> float:
        return self.quantity


class PercentEquitySizer(PositionSizer):
    """
    Allocates a fixed fraction of current total equity to each position:

        quantity = (fraction * equity) / price

    Because it sizes off *current* equity it compounds naturally: position
    sizes grow as the account grows and shrink during drawdowns.
    """

    def __init__(self, fraction: float = 0.1):
        """
        Args:
            fraction: Share of total equity to deploy per position, e.g. 0.1
                allocates 10% of equity. Values > 1.0 imply leverage.
        """
        if fraction <= 0:
            raise ValueError("fraction must be positive")
        self.fraction = fraction

    def size(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        price: float,
        portfolio: "Portfolio",
    ) -> float:
        if price <= 0:
            return 0.0
        target_value = self.fraction * portfolio.total_equity()
        return target_value / price


class VolatilityTargetSizer(PositionSizer):
    """
    Sizes each position so its expected risk contribution is constant,
    targeting an annualised portfolio volatility:

        leverage  = min(target_volatility / asset_volatility, max_leverage)
        quantity  = (leverage * equity) / price

    where ``asset_volatility`` is the annualised standard deviation of recent
    returns. Calm assets get larger positions and turbulent ones get smaller
    positions, which keeps risk steady across regimes. This pairs well with the
    Ornstein-Uhlenbeck strategy, which already trades on a volatility-scaled
    z-score.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        lookback: int = 20,
        periods: int = 252,
        max_leverage: float = 1.0,
    ):
        """
        Args:
            target_volatility: Desired annualised volatility of the position,
                e.g. 0.15 for 15%.
            lookback: Number of returns used to estimate volatility.
            periods: Periods per year used to annualise (252 for daily bars).
            max_leverage: Cap on equity exposure, so a very low measured
                volatility cannot demand an unbounded position.
        """
        if target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.periods = periods
        self.max_leverage = max_leverage
        # Per-symbol rolling price window. We keep lookback + 1 prices because
        # n returns require n + 1 prices.
        self._prices: dict[str, deque[float]] = {}

    def update_market(
        self,
        symbol: str,
        price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        buf = self._prices.get(symbol)
        if buf is None:
            buf = deque(maxlen=self.lookback + 1)
            self._prices[symbol] = buf
        buf.append(price)

    def size(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        price: float,
        portfolio: "Portfolio",
    ) -> float:
        if price <= 0:
            return 0.0

        buf = self._prices.get(symbol)
        if buf is None or len(buf) < self.lookback + 1:
            return 0.0  # not enough history to estimate volatility yet

        prices = np.array(buf)
        returns = np.diff(prices) / prices[:-1]
        period_vol = np.std(returns, ddof=1)
        if period_vol == 0.0 or np.isnan(period_vol):
            return 0.0  # flat series: no estimable risk, so don't trade

        annual_vol = period_vol * np.sqrt(self.periods)
        leverage = min(self.target_volatility / annual_vol, self.max_leverage)
        target_value = leverage * portfolio.total_equity()
        return target_value / price


class ATRStopSizer(PositionSizer):
    """
    Fixed-risk sizing against an ATR-based stop (the "2% rule" / Van Tharp).

    Each trade risks a fixed fraction of equity, where the loss taken if the
    stop is hit is the position size times the stop distance:

        stop_distance = atr_multiple * ATR
        quantity      = (risk_fraction * equity) / stop_distance

    so a wider (more volatile) stop yields a smaller position and vice versa,
    holding the dollar risk per trade roughly constant. The position is also
    capped at ``max_leverage`` times equity so a very tight stop cannot demand
    an oversized position.

    ATR is the simple average of the True Range over ``atr_period`` bars, where
    True Range = max(high-low, |high-prev_close|, |low-prev_close|). This needs
    the bar high/low, which the portfolio supplies via ``update_market``.
    """

    def __init__(
        self,
        risk_fraction: float = 0.02,
        atr_period: int = 14,
        atr_multiple: float = 2.0,
        max_leverage: float = 1.0,
    ):
        """
        Args:
            risk_fraction: Fraction of equity to risk per trade, e.g. 0.02 for
                the classic 2% rule.
            atr_period: Number of bars averaged for the ATR.
            atr_multiple: Stop distance as a multiple of ATR.
            max_leverage: Cap on equity exposure for a single position.
        """
        if not 0 < risk_fraction <= 1:
            raise ValueError("risk_fraction must be in (0, 1]")
        if atr_period < 1:
            raise ValueError("atr_period must be at least 1")
        if atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive")
        self.risk_fraction = risk_fraction
        self.atr_period = atr_period
        self.atr_multiple = atr_multiple
        self.max_leverage = max_leverage
        # Per-symbol rolling window of (high, low, close). atr_period True
        # Ranges need atr_period + 1 bars (each TR references the prior close).
        self._bars: dict[str, deque[tuple[float, float, float]]] = {}

    def update_market(
        self,
        symbol: str,
        price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        # Fall back to close when a feed lacks high/low; True Range then
        # degenerates to the close-to-close move, which is still usable.
        bar_high = high if high is not None else price
        bar_low = low if low is not None else price
        buf = self._bars.get(symbol)
        if buf is None:
            buf = deque(maxlen=self.atr_period + 1)
            self._bars[symbol] = buf
        buf.append((bar_high, bar_low, price))

    def _atr(self, symbol: str) -> float | None:
        buf = self._bars.get(symbol)
        if buf is None or len(buf) < self.atr_period + 1:
            return None
        bars = list(buf)
        true_ranges = []
        for i in range(1, len(bars)):
            high, low, _ = bars[i]
            prev_close = bars[i - 1][2]
            true_range = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(true_range)
        return sum(true_ranges) / len(true_ranges)

    def size(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        price: float,
        portfolio: "Portfolio",
    ) -> float:
        if price <= 0:
            return 0.0

        atr = self._atr(symbol)
        if atr is None or atr <= 0:
            return 0.0  # not enough history, or a flat range

        stop_distance = self.atr_multiple * atr
        equity = portfolio.total_equity()
        quantity = (self.risk_fraction * equity) / stop_distance

        max_quantity = (self.max_leverage * equity) / price
        return min(quantity, max_quantity)


class FractionalKellySizer(PositionSizer):
    """
    Sizes by a fraction of the Kelly criterion, estimated from the realised
    win rate and payoff ratio of the strategy's own completed round trips:

        f* = W - (1 - W) / R
        quantity = (kelly_fraction * f* * equity) / price

    where W is the win probability and R the ratio of average win to average
    loss. Full Kelly maximises long-run growth but is famously volatile, so a
    fraction (``kelly_fraction``, e.g. 0.5 for "half Kelly") is used in
    practice. A non-positive f* means no measured edge, so no trade.

    Estimates are unreliable until enough trades exist, so until ``min_trades``
    completed round trips accumulate the sizer falls back to ``base_fraction``
    of equity. Outcomes are read from ``portfolio.trades`` per symbol; the
    portfolio always flattens before reversing, so entries and exits alternate
    and can be paired in sequence.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        min_trades: int = 10,
        base_fraction: float = 0.02,
        max_fraction: float = 1.0,
    ):
        """
        Args:
            kelly_fraction: Multiple of full Kelly to bet (0.5 = half Kelly).
            min_trades: Completed round trips required before trusting the
                Kelly estimate; below this, ``base_fraction`` is used.
            base_fraction: Fraction of equity used during the warm-up period.
            max_fraction: Upper bound on the equity fraction deployed, guarding
                against an overconfident estimate.
        """
        if not 0 < kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if min_trades < 1:
            raise ValueError("min_trades must be at least 1")
        if not 0 <= base_fraction <= 1:
            raise ValueError("base_fraction must be in [0, 1]")
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
        self.base_fraction = base_fraction
        self.max_fraction = max_fraction

    @staticmethod
    def _completed_returns(trades: list, symbol: str) -> list[float]:
        """
        Pairs each entry with the EXIT that closes it and returns the realised
        return of each round trip as a fraction of the entry notional, net of
        commission on both legs. Slippage is not subtracted: it is already
        embedded in the fill prices, so charging it here would count it twice.
        """
        returns: list[float] = []
        open_entry: dict | None = None

        for trade in trades:
            if trade["symbol"] != symbol:
                continue
            direction = trade["direction"]
            if direction in ("LONG", "SHORT"):
                if open_entry is None:
                    open_entry = trade
            elif direction == "EXIT" and open_entry is not None:
                entry_price = open_entry["price"]
                quantity = open_entry["quantity"]
                notional = entry_price * quantity
                if notional <= 0:
                    open_entry = None
                    continue
                gross = (trade["price"] - entry_price) * quantity
                if open_entry["direction"] == "SHORT":
                    gross = -gross
                costs = open_entry["commission"] + trade["commission"]
                returns.append((gross - costs) / notional)
                open_entry = None

        return returns

    def _kelly_fraction(self, returns: list[float]) -> float:
        wins = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]

        if not wins:
            return 0.0  # no winners: no edge to bet on
        win_prob = len(wins) / len(returns)
        if not losses:
            # No losses observed; payoff ratio is unbounded and f* -> W.
            return win_prob

        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        payoff_ratio = avg_win / avg_loss
        return win_prob - (1.0 - win_prob) / payoff_ratio

    def size(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        price: float,
        portfolio: "Portfolio",
    ) -> float:
        if price <= 0:
            return 0.0

        returns = self._completed_returns(portfolio.trades, symbol)
        if len(returns) < self.min_trades:
            fraction = self.base_fraction
        else:
            fraction = self.kelly_fraction * self._kelly_fraction(returns)

        fraction = max(0.0, min(fraction, self.max_fraction))
        if fraction <= 0:
            return 0.0
        return (fraction * portfolio.total_equity()) / price
