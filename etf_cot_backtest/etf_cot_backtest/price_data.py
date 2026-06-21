"""Fetch ETF price history and resample to weekly returns.

Like cftc_data.py, the live fetch here depends on outbound network access
(Yahoo Finance) that this sandbox's network policy blocks. Run it from an
environment with normal network access; tests exercise the resampling logic
against local fixtures instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)


def fetch_adjusted_close(
    tickers: list[str], start: str, end: str | None = None, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """Download daily adjusted close prices for `tickers` via yfinance.

    Returns a wide DataFrame indexed by date, one column per ticker.
    """
    cache_path = Path(cache_dir) / "adjusted_close.parquet" if cache_dir else None
    if cache_path and cache_path.exists():
        LOG.info("Loading cached price data from %s", cache_path)
        return pd.read_parquet(cache_path)

    import yfinance as yf

    LOG.info("Downloading adjusted close for %s from %s to %s", tickers, start, end)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # single-ticker download collapses the column index
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)

    return prices


def to_weekly_returns(daily_prices: pd.DataFrame, weekday: str = "W-FRI") -> pd.DataFrame:
    """Resample daily adjusted close to weekly (last obs per week) simple returns."""
    weekly_prices = daily_prices.resample(weekday).last()
    return weekly_prices.pct_change()
