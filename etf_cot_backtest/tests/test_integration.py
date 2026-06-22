"""End-to-end pipeline smoke test against synthetic data (no network).

Exercises the exact sequence cli.py wires together -- signal construction,
publication lag, weight construction, backtest, metrics -- so a wiring bug
between modules would show up here even though cli.py itself needs live
network access to cftc.gov / Yahoo Finance and can't run in this sandbox.
"""

import numpy as np
import pandas as pd

from etf_cot_backtest import backtest, portfolio, signals
from etf_cot_backtest.config import UNIVERSE
from etf_cot_backtest.metrics import compare_strategies


def _synthetic_cot(etfs, n_weeks=260, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n_weeks, freq="7D")  # weekly Tuesdays
    rows = []
    for etf in etfs:
        open_interest = 100_000 + rng.integers(-1000, 1000, n_weeks).cumsum() + 50_000
        net = rng.normal(loc=0, scale=open_interest * 0.05)
        comm_long = (open_interest / 2 + net / 2).clip(min=0)
        comm_short = (open_interest / 2 - net / 2).clip(min=0)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "etf": etf,
                    "open_interest": open_interest,
                    "commercial_long": comm_long,
                    "commercial_short": comm_short,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _synthetic_returns(etfs, n_weeks=260, seed=11, listing_offsets=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-05", periods=n_weeks, freq="7D")  # weekly Fridays
    listing_offsets = listing_offsets or {}
    data = {}
    for etf in etfs:
        r = rng.normal(loc=0.001, scale=0.02, size=n_weeks)
        offset = listing_offsets.get(etf, 0)
        if offset:
            r[:offset] = np.nan
        data[etf] = r
    return pd.DataFrame(data, index=dates)


def test_full_pipeline_runs_and_produces_sane_output():
    etfs = [m.etf for m in UNIVERSE]
    cot = _synthetic_cot(etfs)
    # pretend XLC (youngest ETF in the real universe, inception June 2018) only started trading partway through
    returns = _synthetic_returns(etfs, listing_offsets={"XLC": 80})

    signal_long = signals.compute_commercial_zscore(cot, lookback=52, min_periods=26)
    signal_long = portfolio.apply_publication_lag(signal_long, lag_days=3)
    signal_wide = portfolio.pivot_signal(signal_long)

    aligned = portfolio.align_to_rebalance_dates(signal_wide, returns.index)
    tradeable = returns.notna()
    weights = portfolio.signal_to_weights(aligned, tradeable_mask=tradeable)

    # No weight on an asset before it's tradeable.
    assert (weights.loc[returns["XLC"].isna(), "XLC"] == 0).all()
    # Weights at every rebalance date are non-negative and sum to at most 1.
    assert (weights >= 0).all().all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()

    result = backtest.run_backtest(weights, returns, tx_cost_bps=5.0)
    assert result.nav.notna().all()
    assert (result.nav > 0).all()

    benchmark = backtest.equal_weight_benchmark(returns)
    summary = compare_strategies({"strategy": result.returns, "benchmark": benchmark})

    assert set(summary.index) == {"strategy", "benchmark"}
    assert summary.loc["strategy", "n_periods"] == len(returns)
    assert np.isfinite(summary.loc["strategy", "cagr"])
