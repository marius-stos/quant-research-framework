"""Mean-reversion strategies."""
import numpy as np
import pandas as pd


class BollingerBands:
    """
    Bollinger Bands mean reversion.
    Long below lower band, short above upper band, flat inside.
    """
    def __init__(self, window: int = 20, n_std: float = 2.0):
        self.window = window
        self.n_std  = n_std

    def signals(self, prices: pd.Series) -> pd.Series:
        mid   = prices.rolling(self.window).mean()
        std   = prices.rolling(self.window).std()
        upper = mid + self.n_std * std
        lower = mid - self.n_std * std
        sig   = pd.Series(0.0, index=prices.index)
        sig[prices < lower] =  1.0   # oversold → long
        sig[prices > upper] = -1.0   # overbought → short
        return sig.fillna(0)

    def __repr__(self):
        return f"BollingerBands(w={self.window}, n={self.n_std})"


class RSIStrategy:
    """
    RSI-based mean reversion.
    Long when RSI < oversold, short when RSI > overbought.
    """
    def __init__(self, window: int = 14, oversold: float = 30, overbought: float = 70):
        self.window     = window
        self.oversold   = oversold
        self.overbought = overbought

    def _rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(span=self.window, adjust=False).mean()
        avg_l = loss.ewm(span=self.window, adjust=False).mean()
        rs    = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def signals(self, prices: pd.Series) -> pd.Series:
        rsi = self._rsi(prices)
        sig = pd.Series(0.0, index=prices.index)
        sig[rsi < self.oversold]   =  1.0
        sig[rsi > self.overbought] = -1.0
        return sig.fillna(0)

    def __repr__(self):
        return f"RSI(w={self.window}, os={self.oversold}, ob={self.overbought})"


class ZScoreStrategy:
    """
    Rolling Z-score mean reversion.
    Long below -threshold, short above +threshold.
    """
    def __init__(self, window: int = 20, threshold: float = 1.5):
        self.window    = window
        self.threshold = threshold

    def signals(self, prices: pd.Series) -> pd.Series:
        mu    = prices.rolling(self.window).mean()
        sigma = prices.rolling(self.window).std()
        z     = (prices - mu) / sigma.replace(0, np.nan)
        sig   = pd.Series(0.0, index=prices.index)
        sig[z < -self.threshold] =  1.0
        sig[z >  self.threshold] = -1.0
        return sig.fillna(0)

    def __repr__(self):
        return f"ZScore(w={self.window}, thr={self.threshold})"
