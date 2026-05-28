"""Volatility-based strategies and GARCH model."""
from __future__ import annotations
import numpy as np
import pandas as pd


class VolTargeting:
    """
    Volatility-targeting strategy.
    Scales position size so annualised vol ≈ target_vol.
    Signal = size multiplier (not ±1).
    Requires a base directional signal to combine with.
    """
    def __init__(self, target_vol: float = 0.10, window: int = 21, max_leverage: float = 2.0):
        self.target_vol  = target_vol
        self.window      = window
        self.max_leverage = max_leverage

    def size(self, prices: pd.Series) -> pd.Series:
        realized_vol = prices.pct_change().rolling(self.window).std() * np.sqrt(252)
        leverage = (self.target_vol / realized_vol).clip(upper=self.max_leverage)
        return leverage.fillna(1.0)

    def signals(self, prices: pd.Series) -> pd.Series:
        """Stand-alone: long with vol-scaled position."""
        return self.size(prices)

    def __repr__(self):
        return f"VolTargeting(target={self.target_vol:.0%}, w={self.window})"


class GARCHVol:
    """
    Simplified GARCH(1,1) volatility forecaster.
    omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Typical parameters (S&P 500):
        omega = 1e-6, alpha = 0.09, beta = 0.90
    """
    def __init__(self, omega: float = 1e-6, alpha: float = 0.09, beta: float = 0.90):
        self.omega = omega
        self.alpha = alpha
        self.beta  = beta

    def fit(self, prices: pd.Series) -> pd.Series:
        """Return conditional variance series (sigma^2 per day)."""
        returns = prices.pct_change().dropna().values
        n       = len(returns)
        var     = np.zeros(n)
        var[0]  = np.var(returns)

        for t in range(1, n):
            var[t] = (self.omega
                      + self.alpha * returns[t - 1] ** 2
                      + self.beta  * var[t - 1])

        vol = pd.Series(np.sqrt(var) * np.sqrt(252),
                        index=prices.index[1:], name="garch_vol_ann")
        return vol

    def signals(self, prices: pd.Series, base_signal: pd.Series = None) -> pd.Series:
        """
        Signal: scale a base directional signal by inverse GARCH vol
        (vol-targeting via GARCH forecast).
        """
        garch_vol = self.fit(prices)
        if base_signal is None:
            base_signal = pd.Series(1.0, index=garch_vol.index)
        base_signal = base_signal.reindex(garch_vol.index).fillna(0)
        target_vol  = 0.15
        leverage    = (target_vol / garch_vol).clip(upper=2.0)
        return (base_signal * leverage).fillna(0)

    def __repr__(self):
        return f"GARCH(ω={self.omega:.0e}, α={self.alpha}, β={self.beta})"
