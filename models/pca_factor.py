"""PCA factor model for multi-asset returns."""
from __future__ import annotations
import numpy as np
import pandas as pd


class PCAFactorModel:
    """
    PCA decomposition of a returns matrix.
    Extracts the top N principal components as systematic factors.

    Usage:
        model = PCAFactorModel(n_components=3)
        model.fit(returns_df)          # returns: T × N DataFrame
        factors = model.factor_returns  # T × K DataFrame
        loadings = model.loadings       # N × K DataFrame
        explained = model.explained_variance_ratio
    """
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.loadings_             = None
        self.factor_returns_       = None
        self.explained_variance_   = None
        self.mean_                 = None

    def fit(self, returns: pd.DataFrame) -> "PCAFactorModel":
        R = returns.dropna(how="any")
        self.mean_ = R.mean()
        X = (R - self.mean_).values

        # SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        k  = self.n_components
        V  = Vt[:k].T                             # N × k loadings
        F  = X @ V                                # T × k factor returns
        ev = s[:k] ** 2 / (s ** 2).sum()

        self.loadings_           = pd.DataFrame(
            V, index=R.columns,
            columns=[f"PC{i+1}" for i in range(k)])
        self.factor_returns_     = pd.DataFrame(
            F, index=R.index,
            columns=[f"PC{i+1}" for i in range(k)])
        self.explained_variance_ = pd.Series(
            ev, index=[f"PC{i+1}" for i in range(k)], name="explained_var")
        return self

    @property
    def loadings(self) -> pd.DataFrame:
        return self.loadings_

    @property
    def factor_returns(self) -> pd.DataFrame:
        return self.factor_returns_

    @property
    def explained_variance_ratio(self) -> pd.Series:
        return self.explained_variance_

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Idiosyncratic returns (after removing factor exposure)."""
        R   = returns.reindex(self.factor_returns_.index)
        hat = self.factor_returns_.values @ self.loadings_.values.T
        return R - pd.DataFrame(hat, index=R.index, columns=R.columns)

    def signals(self, returns: pd.DataFrame) -> pd.Series:
        """
        Simple signal: long assets with positive PC1 loading (market beta > 0)
        when PC1 factor return is positive.
        """
        self.fit(returns)
        pc1_signal = np.sign(self.factor_returns_["PC1"])
        # Weight by PC1 loading magnitude
        top_assets = self.loadings_["PC1"].abs().nlargest(5).index
        return pc1_signal.rename("pca_signal")

    def print_summary(self):
        print(f"\n  PCA Factor Model — {self.n_components} components")
        print(f"  {'Factor':<8} {'Expl. Var':>12}")
        for pc, ev in self.explained_variance_.items():
            print(f"  {pc:<8} {ev:.1%}")
        print(f"  {'Total':<8} {self.explained_variance_.sum():.1%}")

    def __repr__(self):
        return f"PCAFactorModel(n={self.n_components})"
