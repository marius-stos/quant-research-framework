"""Momentum strategies."""
import numpy as np
import pandas as pd


class JegadeeshTitman:
    """
    JT momentum: rank on past J-month return, skip last K months.
    Single-asset version: long if past return > 0, short otherwise.
    """
    def __init__(self, formation: int = 252, skip: int = 21):
        self.formation = formation
        self.skip      = skip

    def signals(self, prices: pd.Series) -> pd.Series:
        past_ret = prices.shift(self.skip).pct_change(self.formation)
        return np.sign(past_ret).fillna(0)

    def __repr__(self):
        return f"JT(form={self.formation}, skip={self.skip})"


class DualMomentum:
    """
    Antonacci Dual Momentum (single asset version).
    Absolute momentum: long only if 12m return > risk-free rate.
    """
    def __init__(self, lookback: int = 252, rf_annual: float = 0.02):
        self.lookback   = lookback
        self.rf_annual  = rf_annual
        self.rf_period  = rf_annual / 252 * lookback

    def signals(self, prices: pd.Series) -> pd.Series:
        mom = prices.pct_change(self.lookback)
        sig = pd.Series(0.0, index=prices.index)
        sig[mom > self.rf_period] = 1.0    # positive absolute momentum → long
        return sig.fillna(0)

    def __repr__(self):
        return f"DualMomentum(lb={self.lookback})"
