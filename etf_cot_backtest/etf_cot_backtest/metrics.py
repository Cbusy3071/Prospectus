"""Performance statistics for comparing the strategy against benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(returns: pd.Series, periods_per_year: int = 52, rf: float = 0.0) -> dict:
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "win_rate": float("nan"),
            "n_periods": 0,
        }

    nav = (1.0 + returns).cumprod()
    total_return = nav.iloc[-1] - 1.0
    years = n / periods_per_year
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else float("nan")

    ann_vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
    ann_mean = returns.mean() * periods_per_year
    sharpe = (ann_mean - rf) / ann_vol if ann_vol > 0 else float("nan")

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else float("nan")

    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_periods": n,
    }


def compare_strategies(returns_by_name: dict[str, pd.Series], periods_per_year: int = 52, rf: float = 0.0) -> pd.DataFrame:
    rows = {
        name: performance_summary(series, periods_per_year=periods_per_year, rf=rf)
        for name, series in returns_by_name.items()
    }
    return pd.DataFrame(rows).T
