"""
Quant Research Framework — Multi-Strategy Simulation
=====================================================
Runs all 15 strategies on synthetic and real data, computes performance metrics,
and launches the interactive dashboard.

Usage:
    python3 run_simulation.py [--tickers SPY QQQ GLD] [--years 5] [--dashboard]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import run_backtest, BacktestConfig
from strategies.trend import SMACrossover, MACDStrategy, BreakoutStrategy
from strategies.mean_reversion import BollingerBands, RSIStrategy, ZScoreStrategy
from strategies.momentum import JegadeeshTitman, DualMomentum
from strategies.volatility import VolTargeting, GARCHVol
from models.regime import HMMRegime, KalmanTrend
from models.pca_factor import PCAFactorModel

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ── Data loading ───────────────────────────────────────────────────────────────

def load_prices(tickers: list[str], years: int = 5) -> pd.DataFrame:
    if HAS_YF and tickers:
        print(f"  Downloading {tickers} ({years}y)…")
        raw = yf.download(tickers, period=f"{years}y",
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        return raw.dropna()
    else:
        # Synthetic GBM data
        print("  Generating synthetic GBM data…")
        np.random.seed(42)
        T = years * 252
        n = len(tickers) if tickers else 3
        dt = 1 / 252
        mus    = [0.08, 0.06, 0.04][:n]
        sigmas = [0.18, 0.15, 0.12][:n]
        prices = {}
        for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
            r  = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(T)
            prices[f"Asset{i+1}"] = 100 * np.exp(np.cumsum(r))
        idx = pd.date_range("2019-01-01", periods=T, freq="B")
        return pd.DataFrame(prices, index=idx[:T])


# ── Strategy registry ──────────────────────────────────────────────────────────

def get_strategies() -> dict:
    """Returns all 15 strategies, each with a signals(prices) method."""
    return {
        # Trend-following (3)
        "SMA(20,50)":          SMACrossover(fast=20, slow=50),
        "SMA(50,200)":         SMACrossover(fast=50, slow=200),
        "MACD(12,26,9)":       MACDStrategy(),
        "Breakout(20)":        BreakoutStrategy(window=20),
        # Mean-reversion (4)
        "Bollinger(20,2)":     BollingerBands(window=20, n_std=2.0),
        "RSI(14)":             RSIStrategy(window=14),
        "ZScore(20,1.5)":      ZScoreStrategy(window=20, threshold=1.5),
        # Momentum (2)
        "JT(252,21)":          JegadeeshTitman(formation=252, skip=21),
        "DualMom(252)":        DualMomentum(lookback=252),
        # Volatility (2)
        "VolTarget(10%)":      VolTargeting(target_vol=0.10),
        "GARCH(0.09,0.90)":    GARCHVol(alpha=0.09, beta=0.90),
        # ML / Signal models (4)
        "HMM(2-state)":        HMMRegime(n_states=2),
        "Kalman-Trend":        KalmanTrend(obs_noise=1.0, process_noise=0.01),
        # Ensemble
        "Ensemble(trend)":     None,    # computed below
        "Ensemble(all)":       None,    # computed below
    }


# ── Run all strategies ─────────────────────────────────────────────────────────

def run_all(prices: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """
    Run all strategies on the first asset in prices.
    Returns dict of {strategy_name: BacktestResult}.
    """
    asset   = prices.iloc[:, 0]
    returns = asset.pct_change().dropna()
    results = {}
    strategies = get_strategies()

    print(f"\n  Running {len(strategies)} strategies on {asset.name}…\n")

    individual_signals = {}

    for name, strat in strategies.items():
        if strat is None:
            continue
        try:
            # HMM and Kalman operate on returns, others on prices
            if isinstance(strat, (HMMRegime, KalmanTrend)):
                sig = strat.signals(returns)
                sig = sig.reindex(asset.index).fillna(0)
            elif isinstance(strat, GARCHVol):
                sig = strat.signals(asset)
            else:
                sig = strat.signals(asset)

            result = run_backtest(asset, sig, cfg)
            results[name] = result
            individual_signals[name] = sig
            m = result.metrics
            print(f"  {name:<22} CAGR={m['CAGR']:+5.1f}%  "
                  f"Sharpe={m['Sharpe']:5.2f}  "
                  f"MaxDD={m['Max Drawdown']:+6.1f}%  "
                  f"WinRate={m['Win Rate']:.0f}%")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")

    # Ensemble signals
    trend_names = ["SMA(20,50)", "SMA(50,200)", "MACD(12,26,9)", "Breakout(20)"]
    all_names   = list(individual_signals.keys())

    for ens_name, members in [("Ensemble(trend)", trend_names),
                                ("Ensemble(all)",  all_names)]:
        valid = [individual_signals[n] for n in members if n in individual_signals]
        if valid:
            sig  = pd.concat(valid, axis=1).mean(axis=1)
            sig  = np.sign(sig)
            res  = run_backtest(asset, sig, cfg)
            results[ens_name] = res
            m    = res.metrics
            print(f"  {ens_name:<22} CAGR={m['CAGR']:+5.1f}%  "
                  f"Sharpe={m['Sharpe']:5.2f}  "
                  f"MaxDD={m['Max Drawdown']:+6.1f}%  "
                  f"WinRate={m['Win Rate']:.0f}%")

    # PCA on multi-asset (if multiple assets)
    if prices.shape[1] > 1:
        print("\n  PCA Factor Analysis:")
        pca = PCAFactorModel(n_components=3)
        pca.fit(prices.pct_change().dropna())
        pca.print_summary()

    return results


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res.metrics
        rows.append({
            "Strategy":      name,
            "CAGR (%)":      m["CAGR"],
            "Sharpe":        m["Sharpe"],
            "Sortino":       m["Sortino"],
            "MaxDD (%)":     m["Max Drawdown"],
            "Calmar":        m["Calmar"],
            "WinRate (%)":   m["Win Rate"],
            "Trades":        m["N Trades"],
        })
    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    print(f"\n{'='*80}")
    print(f"  STRATEGY RANKING — sorted by Sharpe")
    print(f"{'='*80}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-strategy quant research simulation")
    parser.add_argument("--tickers",   nargs="+", default=["SPY", "QQQ", "GLD"],
                        help="Yahoo Finance tickers")
    parser.add_argument("--years",     type=int,   default=5,
                        help="Backtest horizon in years")
    parser.add_argument("--capital",   type=float, default=100_000)
    parser.add_argument("--commission",type=float, default=5.0,
                        help="Commission in bps (one-way)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch Dash dashboard after simulation")
    args = parser.parse_args()

    print("=" * 65)
    print("  Quantitative Research Framework")
    print(f"  Tickers: {args.tickers}  |  Horizon: {args.years}y  |  Capital: ${args.capital:,.0f}")
    print("=" * 65)

    prices = load_prices(args.tickers, years=args.years)
    cfg    = BacktestConfig(
        initial_capital=args.capital,
        commission_bps=args.commission,
    )

    results = run_all(prices, cfg)
    summary = print_summary(results)

    if args.dashboard:
        from dashboard.app import launch
        launch(prices, results, summary)
    else:
        print("\n  Add --dashboard to launch the interactive Dash dashboard.")


if __name__ == "__main__":
    main()
