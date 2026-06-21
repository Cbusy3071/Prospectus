"""Vectorized weekly-rebalance backtest engine.

Convention: a weight decided at date t (using information already public by
t) is applied to the return realized over the *next* period, never the period
ending at t itself -- that return already happened before the decision could
act on it. Transaction costs are charged at the date a position is entered,
modeled as an immediate drag rather than smeared into the next period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRANSACTION_COST_BPS


@dataclass
class BacktestResult:
    returns: pd.Series      # net per-period portfolio return
    nav: pd.Series          # cumulative growth of $1
    turnover: pd.Series     # sum(|delta weight|) at each rebalance
    held_weights: pd.DataFrame  # weights actually in effect at each return date


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    tx_cost_bps: float = TRANSACTION_COST_BPS,
    cash_return: float = 0.0,
) -> BacktestResult:
    """Run the backtest.

    `weights` is indexed by rebalance date (a subset of `returns.index`),
    columns are tickers, values are target portfolio weights (need not sum to
    1; the shortfall is treated as cash). `returns` is the full per-period
    return grid, same columns, every date the strategy could realize a return.
    """
    common_cols = returns.columns
    weights = weights.reindex(columns=common_cols, fill_value=0.0)

    held = weights.reindex(returns.index).ffill().fillna(0.0)
    applied = held.shift(1).fillna(0.0)

    asset_component = (applied * returns).sum(axis=1)
    invested_fraction = applied.sum(axis=1)
    cash_component = (1.0 - invested_fraction) * cash_return
    gross_return = asset_component + cash_component

    turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (tx_cost_bps / 10_000.0)

    net_return = gross_return - cost
    nav = (1.0 + net_return).cumprod()

    return BacktestResult(returns=net_return, nav=nav, turnover=turnover, held_weights=held)


def equal_weight_benchmark(returns: pd.DataFrame, tickers: list[str] | None = None) -> pd.Series:
    """Static equal-weight buy-and-hold benchmark over the same universe."""
    cols = tickers or list(returns.columns)
    available = returns[cols]
    valid_count = available.notna().sum(axis=1).replace(0, np.nan)
    return available.fillna(0.0).sum(axis=1) / valid_count
