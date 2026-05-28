"""
Regime detection models.
- HMMRegime: Hidden Markov Model (2-state: bull/bear)
- KalmanTrend: Kalman filter for adaptive trend estimation
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── HMM (Gaussian emissions, 2 states) ────────────────────────────────────────

class HMMRegime:
    """
    2-state Gaussian HMM fitted via Baum-Welch (EM).
    States: 0 = low-vol (bull), 1 = high-vol (bear).

    Outputs:
        state_probs: DataFrame of P(state=k | observations)
        viterbi_path: most likely state sequence
    """
    def __init__(self, n_states: int = 2, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter   = n_iter
        # Initial parameters (overwritten by fit)
        self.pi     = np.ones(n_states) / n_states
        self.A      = np.full((n_states, n_states), 1 / n_states)
        self.mu     = None
        self.sigma  = None

    def fit(self, returns: pd.Series) -> "HMMRegime":
        obs = returns.dropna().values
        T   = len(obs)
        K   = self.n_states

        # Initialise: sort by return magnitude
        kmeans_labels = (np.abs(obs) > np.abs(obs).median()).astype(int)
        self.mu    = np.array([obs[kmeans_labels == k].mean() for k in range(K)])
        self.sigma = np.array([obs[kmeans_labels == k].std() + 1e-6 for k in range(K)])
        self.A     = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.pi    = np.array([0.5, 0.5])

        for _ in range(self.n_iter):
            # E-step: forward-backward
            alpha = self._forward(obs)
            beta  = self._backward(obs)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] *
                                       self._emission(obs[t + 1], j) * beta[t + 1, j])
                xi[t] /= xi[t].sum() + 1e-300

            # M-step
            self.pi = gamma[0]
            self.A  = xi.sum(axis=0) / xi.sum(axis=(0, 2), keepdims=True).T + 1e-300
            self.A /= self.A.sum(axis=1, keepdims=True)
            for k in range(K):
                g = gamma[:, k]
                self.mu[k]    = (g * obs).sum() / g.sum()
                self.sigma[k] = np.sqrt((g * (obs - self.mu[k]) ** 2).sum() / g.sum()) + 1e-6

        self._obs  = obs
        self._index = returns.dropna().index
        return self

    def _emission(self, x, k):
        return (1 / (self.sigma[k] * np.sqrt(2 * np.pi)) *
                np.exp(-0.5 * ((x - self.mu[k]) / self.sigma[k]) ** 2))

    def _forward(self, obs):
        T, K  = len(obs), self.n_states
        alpha = np.zeros((T, K))
        alpha[0] = self.pi * np.array([self._emission(obs[0], k) for k in range(K)])
        alpha[0] /= alpha[0].sum() + 1e-300
        for t in range(1, T):
            for j in range(K):
                alpha[t, j] = self._emission(obs[t], j) * (alpha[t - 1] @ self.A[:, j])
            alpha[t] /= alpha[t].sum() + 1e-300
        return alpha

    def _backward(self, obs):
        T, K = len(obs), self.n_states
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            for i in range(K):
                beta[t, i] = sum(self.A[i, j] * self._emission(obs[t + 1], j) * beta[t + 1, j]
                                 for j in range(K))
            beta[t] /= beta[t].sum() + 1e-300
        return beta

    def predict_proba(self) -> pd.DataFrame:
        alpha = self._forward(self._obs)
        beta  = self._backward(self._obs)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        # Ensure state 0 = low-vol (bull) by checking mu order
        if self.mu[0] < self.mu[1]:
            bull_state = 0
        else:
            bull_state = 1
        df = pd.DataFrame(gamma, index=self._index,
                          columns=[f"P(state={k})" for k in range(self.n_states)])
        df["regime"] = np.where(gamma[:, bull_state] > 0.5, "Bull", "Bear")
        return df

    def signals(self, returns: pd.Series) -> pd.Series:
        """Long in bull regime, flat in bear."""
        self.fit(returns)
        proba = self.predict_proba()
        return (proba["regime"] == "Bull").astype(float).rename("hmm_signal")

    def __repr__(self):
        return f"HMMRegime(states={self.n_states})"


# ── Kalman filter trend ────────────────────────────────────────────────────────

class KalmanTrend:
    """
    Kalman filter for adaptive trend estimation.
    State: [level, trend]. Observation: price.
    Signal: long if filtered trend > 0, short otherwise.
    """
    def __init__(self, obs_noise: float = 1.0, process_noise: float = 0.01):
        self.obs_noise     = obs_noise
        self.process_noise = process_noise

    def filter(self, prices: pd.Series) -> pd.DataFrame:
        obs = prices.values
        T   = len(obs)

        # State transition: [level, trend]
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = self.process_noise * np.eye(2)
        R = np.array([[self.obs_noise]])

        x = np.array([obs[0], 0.0])
        P = np.eye(2) * 10

        levels = np.zeros(T)
        trends = np.zeros(T)

        for t in range(T):
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            y = obs[t] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K.flatten() * y.flatten()[0]
            P = (np.eye(2) - K @ H) @ P_pred

            levels[t] = x[0]
            trends[t] = x[1]

        return pd.DataFrame({"level": levels, "trend": trends}, index=prices.index)

    def signals(self, prices: pd.Series) -> pd.Series:
        filtered = self.filter(prices)
        return np.sign(filtered["trend"]).rename("kalman_signal")

    def __repr__(self):
        return f"KalmanTrend(obs_noise={self.obs_noise}, proc_noise={self.process_noise})"
