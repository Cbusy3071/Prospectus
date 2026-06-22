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


def test_shift_signal_monthly_rebalance_pipeline():
    # Mirrors cli.py's new path: fade-the-shift signal aligned to a ~monthly
    # rebalance grid, weights held between rebalances by the backtest engine.
    etfs = [m.etf for m in UNIVERSE]
    cot = _synthetic_cot(etfs)
    returns = _synthetic_returns(etfs, listing_offsets={"XLC": 80})

    shift = signals.compute_commercial_shift_signal(
        cot, shift_weeks=13, lookback=52, min_periods=26, smooth_span=4, invert=True
    )
    shift = portfolio.apply_publication_lag(shift, lag_days=3)
    shift_wide = portfolio.pivot_signal(shift)

    rebalance_dates = portfolio.subsample_rebalance_dates(returns.index, every_n_weeks=4)
    assert len(rebalance_dates) < len(returns.index)

    aligned = portfolio.align_to_rebalance_dates(shift_wide, rebalance_dates)
    tradeable = returns.loc[rebalance_dates].notna()
    weights = portfolio.signal_to_weights(aligned, tradeable_mask=tradeable)

    assert (weights >= 0).all().all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (weights.loc[returns["XLC"].loc[rebalance_dates].isna(), "XLC"] == 0).all()

    monthly = backtest.run_backtest(weights, returns, tx_cost_bps=5.0)
    weekly_weights = portfolio.signal_to_weights(
        portfolio.align_to_rebalance_dates(shift_wide, returns.index),
        tradeable_mask=returns.notna(),
    )
    weekly = backtest.run_backtest(weekly_weights, returns, tx_cost_bps=5.0)

    assert monthly.nav.notna().all() and (monthly.nav > 0).all()
    # Rebalancing ~monthly instead of weekly should not trade more often.
    assert monthly.turnover.sum() <= weekly.turnover.sum() + 1e-9


def test_shift_signal_tilt_pipeline_is_always_fully_invested():
    # Same shift signal as the long-only path, but via signal_to_tilt_weights:
    # never goes to cash, isolating the signal's value from market-timing drag.
    etfs = [m.etf for m in UNIVERSE]
    cot = _synthetic_cot(etfs)
    returns = _synthetic_returns(etfs, listing_offsets={"XLC": 80})

    shift = signals.compute_commercial_shift_signal(
        cot, shift_weeks=13, lookback=52, min_periods=26, smooth_span=4, invert=True
    )
    shift = portfolio.apply_publication_lag(shift, lag_days=3)
    shift_wide = portfolio.pivot_signal(shift)

    rebalance_dates = portfolio.subsample_rebalance_dates(returns.index, every_n_weeks=4)
    aligned = portfolio.align_to_rebalance_dates(shift_wide, rebalance_dates)
    tradeable = returns.loc[rebalance_dates].notna()
    weights = portfolio.signal_to_tilt_weights(aligned, tradeable_mask=tradeable, tilt_strength=1.0)

    assert (weights >= -1e-9).all().all()
    # Fully invested at every rebalance, unlike the long-only/cash variant.
    assert weights.sum(axis=1).round(6).eq(1.0).all()
    assert (weights.loc[returns["XLC"].loc[rebalance_dates].isna(), "XLC"] == 0).all()

    result = backtest.run_backtest(weights, returns, tx_cost_bps=5.0)
    assert result.nav.notna().all() and (result.nav > 0).all()


def test_meanrev_signal_pipeline_only_trades_at_extremes():
    # Mirrors cli.py's mean-reversion path: fade level extremes, long-only with
    # a z-score deadband, so most rebalances hold few/no positions.
    etfs = [m.etf for m in UNIVERSE]
    cot = _synthetic_cot(etfs)
    returns = _synthetic_returns(etfs, listing_offsets={"XLC": 80})

    mr = signals.compute_commercial_meanrev_signal(
        cot, lookback=52, min_periods=26, threshold=1.0, invert=True
    )
    mr = portfolio.apply_publication_lag(mr, lag_days=3)
    mr_wide = portfolio.pivot_signal(mr)

    rebalance_dates = portfolio.subsample_rebalance_dates(returns.index, every_n_weeks=4)
    aligned = portfolio.align_to_rebalance_dates(mr_wide, rebalance_dates)
    tradeable = returns.loc[rebalance_dates].notna()
    weights = portfolio.signal_to_weights(aligned, tradeable_mask=tradeable)

    assert (weights >= 0).all().all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (weights.loc[returns["XLC"].loc[rebalance_dates].isna(), "XLC"] == 0).all()
    # A z-score deadband should leave most names at zero weight on most dates,
    # so total invested exposure is well below the always-invested 1.0 average.
    assert weights.sum(axis=1).mean() < 1.0

    result = backtest.run_backtest(weights, returns, tx_cost_bps=5.0)
    assert result.nav.notna().all() and (result.nav > 0).all()
