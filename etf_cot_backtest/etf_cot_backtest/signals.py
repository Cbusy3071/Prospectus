"""'Follow the commercials' positioning signal.

Commercial traders (hedgers: producers, merchants, end-users) are commonly
read as the COT report's "smart money" -- the theory being that they have
better information about their own market than speculators do. This module
turns their net positioning into a cross-sectionally comparable signal:

1. Net commercial position, scaled by open interest, so markets of very
   different sizes are comparable: net_pct_oi = (long - short) / open_interest.
2. A rolling z-score of that ratio against the market's own history, so a
   market trading at "fairly long for itself" is comparable across assets
   regardless of each market's typical positioning bias (e.g. commercials are
   structurally net short gold but net long the dollar index on average).

A positive z-score means commercials are more net-long (or less net-short)
than their recent norm -- read as bullish for that market.
"""

from __future__ import annotations

import pandas as pd

from .config import LOOKBACK_WEEKS, MIN_LOOKBACK_WEEKS


def net_pct_open_interest(cot_df: pd.DataFrame) -> pd.DataFrame:
    """Add a net_pct_oi column: (commercial_long - commercial_short) / open_interest."""
    out = cot_df.copy()
    out["net_pct_oi"] = (out["commercial_long"] - out["commercial_short"]) / out["open_interest"]
    return out


def compute_commercial_zscore(
    cot_df: pd.DataFrame,
    lookback: int = LOOKBACK_WEEKS,
    min_periods: int = MIN_LOOKBACK_WEEKS,
) -> pd.DataFrame:
    """Rolling z-score of net commercial positioning (% of OI), per ETF.

    Input: long-format frame with columns [date, etf, open_interest,
    commercial_long, commercial_short], one row per market per report date.

    Output: long-format frame with columns [date, etf, net_pct_oi, signal],
    where `signal` is NaN until `min_periods` observations are available.
    """
    df = net_pct_open_interest(cot_df).sort_values(["etf", "date"])

    def _zscore(series: pd.Series) -> pd.Series:
        roll = series.rolling(window=lookback, min_periods=min_periods)
        return (series - roll.mean()) / roll.std(ddof=0)

    df["signal"] = df.groupby("etf")["net_pct_oi"].transform(_zscore)
    return df[["date", "etf", "net_pct_oi", "signal"]].reset_index(drop=True)
