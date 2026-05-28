# Quantitative Research Framework

Modular Python framework for systematic strategy design, backtesting, and ML-driven signal generation. Supports 15+ strategies across trend-following, mean-reversion, momentum, and volatility regimes, with an interactive Dash dashboard displaying 50+ performance metrics.

---

## Architecture

```
quant_research_framework/
├── strategies/              # 15+ systematic strategies
│   ├── trend.py             # SMA crossover, MACD, Donchian breakout
│   ├── mean_reversion.py    # Bollinger bands, RSI, Z-score
│   ├── momentum.py          # Jegadeesh-Titman, Dual Momentum (Antonacci)
│   └── volatility.py        # Vol-targeting, GARCH(1,1)
├── models/                  # ML regime & factor models
│   ├── regime.py            # HMM (2-state bull/bear), Kalman filter trend
│   └── pca_factor.py        # PCA factor decomposition
├── backtest/
│   └── engine.py            # Vectorised backtesting engine
├── dashboard/
│   └── app.py               # Dash dashboard — 50+ metrics
└── run_simulation.py        # Entry point — runs all strategies
```

---

## Strategies (15)

| Category | Strategies |
|----------|-----------|
| **Trend** | SMA(20,50), SMA(50,200), MACD(12,26,9), Breakout(20) |
| **Mean-Reversion** | Bollinger(20,2), RSI(14), Z-Score(20,1.5) |
| **Momentum** | Jegadeesh-Titman(252,21), Dual Momentum (Antonacci) |
| **Volatility** | Vol-Targeting(10%), GARCH(0.09,0.90) |
| **ML Models** | HMM(2-state), Kalman-Trend |
| **Ensemble** | Ensemble(trend), Ensemble(all) |

---

## ML Models

- **HMM**: 2-state Gaussian Hidden Markov Model (Baum-Welch EM). Identifies bull/bear regimes from return dynamics.
- **Kalman Filter**: State-space model (level + trend). Adaptive trend estimation robust to noise.
- **GARCH(1,1)**: Conditional volatility forecasting for dynamic position sizing.
- **PCA**: Factor decomposition of multi-asset returns. Extracts systematic risk factors.

---

## Backtest Engine

- Vectorised execution (pandas/NumPy), 1-bar signal delay
- Transaction costs: commission + slippage in bps
- Metrics: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor

---

## Dashboard Panels (50+ metrics)

- Equity curves (all strategies overlaid)
- Drawdown profiles
- Strategy ranking by Sharpe
- Rolling Sharpe (63-day)
- Return distributions
- Monthly P&L heatmap
- Strategy return correlations
- Trade duration vs PnL scatter
- Full metrics table (sortable)

---

## Setup

```bash
pip install numpy pandas scipy plotly dash dash-bootstrap-components yfinance
```

**Run simulation:**
```bash
python3 run_simulation.py --tickers SPY QQQ GLD --years 5
```

**With dashboard:**
```bash
python3 run_simulation.py --tickers SPY QQQ GLD --years 5 --dashboard
# → http://127.0.0.1:8052
```
