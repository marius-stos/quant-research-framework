"""
Vectorised Backtesting Engine
==============================
Runs any strategy that exposes a `signals(prices) -> pd.Series` method.
Handles transaction costs, slippage, position sizing, and performance metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_bps:  float = 5.0        # one-way, basis points
    slippage_bps:    float = 3.0        # one-way, basis points
    risk_free_rate:  float = 0.02       # annual
    position_size:   float = 1.0        # fraction of capital
    allow_short:     bool  = True


@dataclass
class BacktestResult:
    equity:         pd.Series       # portfolio value over time
    returns:        pd.Series       # daily returns
    positions:      pd.Series       # position at each step
    trades:         pd.DataFrame    # trade log
    metrics:        dict = field(default_factory=dict)

    def __post_init__(self):
        self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> dict:
        r   = self.returns.dropna()
        ann = 252
        rf  = 0.02 / ann

        total_return  = (self.equity.iloc[-1] / self.equity.iloc[0]) - 1
        n_years       = len(r) / ann
        cagr          = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
        vol_ann       = r.std() * np.sqrt(ann)
        excess        = r - rf
        sharpe        = excess.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else 0
        sortino_denom = r[r < 0].std() * np.sqrt(ann)
        sortino       = excess.mean() * ann / sortino_denom if sortino_denom > 0 else 0
        drawdown      = (self.equity / self.equity.cummax()) - 1
        max_dd        = drawdown.min()
        calmar        = cagr / abs(max_dd) if max_dd != 0 else 0
        n_trades      = len(self.trades)
        win_rate      = (self.trades["pnl"] > 0).mean() if n_trades > 0 else 0
        profit_factor = (
            self.trades.loc[self.trades["pnl"] > 0, "pnl"].sum() /
            abs(self.trades.loc[self.trades["pnl"] < 0, "pnl"].sum())
            if (self.trades["pnl"] < 0).any() else np.inf
        )

        return {
            "Total Return":    round(total_return * 100, 2),
            "CAGR":            round(cagr * 100, 2),
            "Volatility":      round(vol_ann * 100, 2),
            "Sharpe":          round(sharpe, 3),
            "Sortino":         round(sortino, 3),
            "Max Drawdown":    round(max_dd * 100, 2),
            "Calmar":          round(calmar, 3),
            "Win Rate":        round(win_rate * 100, 1),
            "Profit Factor":   round(profit_factor, 3),
            "N Trades":        n_trades,
        }

    def print_metrics(self):
        print(f"\n{'─'*40}")
        for k, v in self.metrics.items():
            unit = "%" if "Return" in k or "CAGR" in k or "Vol" in k or "DD" in k or "Rate" in k else ""
            print(f"  {k:<20} {v}{unit}")
        print(f"{'─'*40}")


def run_backtest(
    prices:  pd.Series,
    signals: pd.Series,
    cfg:     BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """
    Vectorised single-asset backtest.

    signals: +1 (long), -1 (short), 0 (flat). Aligned to prices index.
    Positions enter at next open (1-bar delay).
    """
    prices  = prices.dropna()
    signals = signals.reindex(prices.index).fillna(0)

    # 1-bar execution delay
    positions = signals.shift(1).fillna(0)
    if not cfg.allow_short:
        positions = positions.clip(lower=0)

    # Daily returns of the instrument
    instr_ret = prices.pct_change()

    # Transaction cost on position changes
    pos_change  = positions.diff().abs()
    total_bps   = (cfg.commission_bps + cfg.slippage_bps) / 10_000
    trade_costs = pos_change * total_bps

    # Strategy return
    strat_ret = positions * instr_ret - trade_costs
    strat_ret.iloc[0] = 0

    # Equity curve
    equity  = cfg.initial_capital * (1 + strat_ret).cumprod()

    # Trade log: detect entries/exits
    trades = _build_trade_log(prices, positions, instr_ret, total_bps)

    return BacktestResult(equity=equity, returns=strat_ret,
                          positions=positions, trades=trades)


def _build_trade_log(prices, positions, instr_ret, cost_rate) -> pd.DataFrame:
    trades = []
    in_trade  = False
    entry_px  = 0.0
    entry_dt  = None
    direction = 0

    prev_pos = 0.0
    for dt, pos in positions.items():
        if not in_trade and pos != 0:
            in_trade  = True
            entry_px  = prices.loc[dt]
            entry_dt  = dt
            direction = int(np.sign(pos))
        elif in_trade and (pos == 0 or np.sign(pos) != direction):
            exit_px = prices.loc[dt]
            raw_ret = direction * (exit_px - entry_px) / entry_px
            pnl     = raw_ret - 2 * cost_rate   # entry + exit costs
            trades.append({
                "entry_date":  entry_dt,
                "exit_date":   dt,
                "direction":   "LONG" if direction == 1 else "SHORT",
                "entry_price": round(entry_px, 4),
                "exit_price":  round(exit_px, 4),
                "return_pct":  round(raw_ret * 100, 3),
                "pnl":         round(pnl * 100, 3),
            })
            in_trade  = False
            if pos != 0:
                in_trade  = True
                entry_px  = prices.loc[dt]
                entry_dt  = dt
                direction = int(np.sign(pos))
        prev_pos = pos

    return pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "direction",
                 "entry_price", "exit_price", "return_pct", "pnl"])
