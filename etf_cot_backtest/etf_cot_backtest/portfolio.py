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


def subsample_rebalance_dates(
    weekly_index: pd.DatetimeIndex, every_n_weeks: int = 4
) -> pd.DatetimeIndex:
    """Keep every Nth weekly date so the portfolio rebalances less often.

    The signal still updates weekly; we just act on it every ``every_n_weeks``
    weeks (~monthly at the default 4) and hold in between, which is the simplest
    lever for cutting turnover. ``every_n_weeks <= 1`` keeps the weekly grid.
    """
    if every_n_weeks <= 1:
        return weekly_index
    return weekly_index[::every_n_weeks]


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


def signal_to_tilt_weights(
    signal_wide: pd.DataFrame, tradeable_mask: pd.DataFrame | None = None, tilt_strength: float = 1.0
) -> pd.DataFrame:
    """Always-fully-invested weights: equal-weight base, tilted by signal rank.

    `signal_to_weights` goes to cash whenever a market's signal is non-positive,
    which couples the signal to market-timing/cash-drag effects on top of
    whatever sector-selection value it has. This variant never holds cash: every
    tradeable market starts at `1/n` and the signal only redistributes weight
    among them (highest-ranked gets the most, lowest-ranked the least), so a
    flat or negative signal everywhere still leaves the portfolio fully invested
    at equal weight. A market with no signal yet (NaN, e.g. still in its
    z-score warm-up) is treated as rank-neutral -- it keeps its `1/n` base with
    no tilt.

    `tilt_strength=1.0` is the most extreme tilt for which weights stay
    non-negative (the lowest-ranked tradeable market in a window goes to ~0,
    the highest-ranked to ~`2/n`); `0.0` collapses to equal weight.
    """
    if tradeable_mask is None:
        tradeable_mask = pd.DataFrame(True, index=signal_wide.index, columns=signal_wide.columns)
    tradeable_mask = tradeable_mask.reindex_like(signal_wide).fillna(False)

    n = tradeable_mask.sum(axis=1)
    base = tradeable_mask.div(n.where(n != 0, 1.0), axis=0)

    # Rank only within the subset that's both tradeable and has a usable
    # (non-NaN) signal; ranking ignores NaNs by construction, so a market
    # outside that subset simply never gets a tilt away from its `1/n` base.
    valid = tradeable_mask & signal_wide.notna()
    valid_n = valid.sum(axis=1)
    ranked = signal_wide.where(valid).rank(axis=1, method="average")
    centered = ranked.sub((valid_n + 1) / 2.0, axis=0)

    k = tilt_strength * 2.0 / (n * (valid_n - 1)).where(valid_n > 1, 1.0)
    tilt = centered.mul(k, axis=0).where(valid, 0.0).fillna(0.0)

    weights = (base + tilt).where(tradeable_mask, 0.0)
    return weights.clip(lower=0.0)
