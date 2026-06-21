"""Turn the commercial-positioning signal into rebalance-date portfolio weights."""

from __future__ import annotations

import pandas as pd

from .config import PUBLICATION_LAG_DAYS


def apply_publication_lag(signal_df: pd.DataFrame, lag_days: int = PUBLICATION_LAG_DAYS) -> pd.DataFrame:
    """Shift each COT report's as-of date forward to when it was actually public.

    CFTC reports Tuesday's positioning but does not release it until the
    following Friday. Without this shift, a backtest would be trading on
    information before it existed (look-ahead bias).
    """
    out = signal_df.copy()
    out["effective_date"] = out["date"] + pd.Timedelta(days=lag_days)
    return out


def pivot_signal(signal_df: pd.DataFrame, value_col: str = "signal") -> pd.DataFrame:
    """Long [effective_date, etf, signal] -> wide [effective_date index, etf columns]."""
    return signal_df.pivot(index="effective_date", columns="etf", values=value_col).sort_index()


def align_to_rebalance_dates(signal_wide: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each rebalance date, look up the most recent signal already public by then.

    Equivalent to an as-of (backward) join per column: no peeking at a signal
    whose effective_date is after the rebalance date.
    """
    combined_index = signal_wide.index.union(rebalance_dates).sort_values()
    reindexed = signal_wide.reindex(combined_index).ffill()
    return reindexed.loc[rebalance_dates]


def signal_to_weights(signal_wide: pd.DataFrame, tradeable_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long-only weights: invest in proportion to positive commercial z-scores.

    Markets with a non-positive signal (commercials at/below their own
    historical norm) get zero weight. If every market is non-positive at a
    given rebalance, the whole portfolio sits in cash for that period.
    `tradeable_mask` (same shape, bool) can exclude ETFs not yet listed.
    """
    clipped = signal_wide.clip(lower=0).fillna(0.0)
    if tradeable_mask is not None:
        clipped = clipped.where(tradeable_mask.reindex_like(clipped).fillna(False), 0.0)

    row_sum = clipped.sum(axis=1)
    weights = clipped.div(row_sum.where(row_sum != 0, 1.0), axis=0)
    weights = weights.where(row_sum != 0, 0.0)
    return weights
