"""Commercial-positioning signals from the CFTC COT report.

Two signals live here, both built from commercial (hedger) net positioning
scaled by open interest -- net_pct_oi = (long - short) / open_interest -- so
markets of very different sizes are comparable:

1. ``compute_commercial_zscore`` -- the *level* "follow the commercials"
   signal. A rolling z-score of net_pct_oi against the market's own history;
   positive means commercials are more net-long than their norm, read
   (classically) as bullish. This is the textbook commodity-style read.

2. ``compute_commercial_shift_signal`` -- a *change* (shift) signal, optionally
   inverted. For equity sectors the commercials behave more like the side that
   leans against trends than "smart money", so a backtest of (1) underperforms
   buy-and-hold. This signal instead z-scores the *change* in commercial
   positioning over a multi-week window and, when ``invert=True``, flips the
   sign: a rising commercial net-long is read as bearish, i.e. we side with
   the speculative/momentum flow the commercials are leaning against. The
   change signal is whippier than the level, so it is EMA-smoothed to keep
   turnover down.

3. ``compute_commercial_meanrev_signal`` -- a mean-reversion read on the
   *level*. Rather than tracking the level continuously like (1), it only acts
   once positioning is at an extreme of its own history (a z-score deadband),
   betting the extreme reverts: a sector where commercials are crowded net-long
   is treated as overbought (no position / sell out), and one where they are at
   an extreme net-short deviation is treated as a buy. ``invert=True`` is what
   makes "crowded long = sell, extreme short = buy"; ``invert=False`` flips it
   back to the classic follow-the-commercials direction.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    INVERT_SIGNAL,
    LOOKBACK_WEEKS,
    MEANREV_THRESHOLD,
    MIN_LOOKBACK_WEEKS,
    SHIFT_WEEKS,
    SIGNAL_SMOOTH_SPAN,
)


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


def compute_commercial_shift_signal(
    cot_df: pd.DataFrame,
    shift_weeks: int = SHIFT_WEEKS,
    lookback: int = LOOKBACK_WEEKS,
    min_periods: int = MIN_LOOKBACK_WEEKS,
    smooth_span: int = SIGNAL_SMOOTH_SPAN,
    invert: bool = INVERT_SIGNAL,
) -> pd.DataFrame:
    """Signal from the *change* in commercial net positioning, optionally inverted.

    Input: long-format frame with columns [date, etf, open_interest,
    commercial_long, commercial_short], one row per market per report date.

    Steps, per ETF:
    1. ``shift`` = change in net_pct_oi over ``shift_weeks`` reports -- are
       commercials adding to or cutting their net-long lately?
    2. z-score that change against the ETF's own rolling history so a "big move
       for this sector" is comparable across sectors.
    3. if ``invert``, negate it -- fade the commercials (side with the spec /
       momentum flow they lean against), which is the read that holds up for
       equity sectors.
    4. EMA-smooth over ``smooth_span`` weeks, because a change signal whipsaws
       far more than a level signal and we want to keep turnover down.

    Output: long-format frame with columns [date, etf, net_pct_oi, shift,
    signal], where `signal` is NaN until enough history is available.
    """
    df = net_pct_open_interest(cot_df).sort_values(["etf", "date"]).reset_index(drop=True)

    df["shift"] = df.groupby("etf")["net_pct_oi"].diff(shift_weeks)

    def _zscore(series: pd.Series) -> pd.Series:
        roll = series.rolling(window=lookback, min_periods=min_periods)
        return (series - roll.mean()) / roll.std(ddof=0)

    z = df.groupby("etf")["shift"].transform(_zscore)
    if invert:
        z = -z
    df["signal_raw"] = z

    if smooth_span and smooth_span > 1:
        df["signal"] = df.groupby("etf")["signal_raw"].transform(
            lambda s: s.ewm(span=smooth_span, min_periods=1).mean()
        )
    else:
        df["signal"] = df["signal_raw"]

    return df[["date", "etf", "net_pct_oi", "shift", "signal"]].reset_index(drop=True)


def compute_commercial_meanrev_signal(
    cot_df: pd.DataFrame,
    lookback: int = LOOKBACK_WEEKS,
    min_periods: int = MIN_LOOKBACK_WEEKS,
    threshold: float = MEANREV_THRESHOLD,
    invert: bool = INVERT_SIGNAL,
) -> pd.DataFrame:
    """Mean-reversion signal on the *level* of commercial positioning.

    Fades positioning extremes instead of tracking the level continuously: bet
    that a crowded position reverts. Per ETF:

    1. ``level_z`` = rolling z-score of net_pct_oi (same as
       ``compute_commercial_zscore``) -- how stretched is positioning vs this
       sector's own history.
    2. ``inv`` = the fade direction. With ``invert=True`` (default) it is
       ``-level_z``: commercials crowded net-long (high z) reads bearish,
       commercials at an extreme net-short deviation (very negative z) reads
       bullish. ``invert=False`` keeps the classic follow direction.
    3. ``signal`` = ``inv - threshold``. This bakes in a **deadband**: paired
       with ``signal_to_weights`` (which keeps only the positive part), a sector
       earns weight only once ``inv`` clears ``threshold`` -- i.e. positioning
       is at least ``threshold`` z-scores into the favourable tail. In the
       normal zone, and on the overcrowded tail, the signal is non-positive so
       the sector sits in cash ("sell out when it overcrowds"); weight goes only
       to sectors at the buy extreme, scaled by how far past the threshold they
       are.

    Output: long-format frame [date, etf, net_pct_oi, level_z, signal], with
    NaN until ``min_periods`` history is available.
    """
    df = net_pct_open_interest(cot_df).sort_values(["etf", "date"]).reset_index(drop=True)

    def _zscore(series: pd.Series) -> pd.Series:
        roll = series.rolling(window=lookback, min_periods=min_periods)
        return (series - roll.mean()) / roll.std(ddof=0)

    z = df.groupby("etf")["net_pct_oi"].transform(_zscore)
    df["level_z"] = z

    inv = -z if invert else z
    df["signal"] = inv - threshold

    return df[["date", "etf", "net_pct_oi", "level_z", "signal"]].reset_index(drop=True)
