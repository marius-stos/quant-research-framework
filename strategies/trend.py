"""Trend-following strategies."""
import numpy as np
import pandas as pd


class SMACrossover:
    """
    Simple Moving Average Crossover.
    Long when fast SMA > slow SMA, short otherwise.
    """
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def signals(self, prices: pd.Series) -> pd.Series:
        sma_f = prices.rolling(self.fast).mean()
        sma_s = prices.rolling(self.slow).mean()
        sig = np.sign(sma_f - sma_s)
        return sig.fillna(0)

    def __repr__(self):
        return f"SMACrossover(fast={self.fast}, slow={self.slow})"


class MACDStrategy:
    """
    MACD crossover: signal = sign(MACD line - signal line).
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def signals(self, prices: pd.Series) -> pd.Series:
        ema_f  = prices.ewm(span=self.fast,   adjust=False).mean()
        ema_s  = prices.ewm(span=self.slow,   adjust=False).mean()
        macd   = ema_f - ema_s
        sig_l  = macd.ewm(span=self.signal, adjust=False).mean()
        return np.sign(macd - sig_l).fillna(0)

    def __repr__(self):
        return f"MACD({self.fast},{self.slow},{self.signal})"


class BreakoutStrategy:
    """
    Donchian channel breakout.
    Long on new N-day high, short on new N-day low.
    """
    def __init__(self, window: int = 20):
        self.window = window

    def signals(self, prices: pd.Series) -> pd.Series:
        high = prices.rolling(self.window).max().shift(1)
        low  = prices.rolling(self.window).min().shift(1)
        sig  = pd.Series(0.0, index=prices.index)
        sig[prices > high] =  1.0
        sig[prices < low]  = -1.0
        return sig.fillna(0)

    def __repr__(self):
        return f"Breakout(window={self.window})"
