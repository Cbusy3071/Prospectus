import math

import pandas as pd
import pytest

from etf_cot_backtest.portfolio import (
    align_to_rebalance_dates,
    apply_publication_lag,
    pivot_signal,
    signal_to_weights,
)


def test_apply_publication_lag_shifts_dates():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-07"]), "etf": ["A"], "signal": [1.0]})
    out = apply_publication_lag(df, lag_days=3)
    assert out["effective_date"].iloc[0] == pd.Timestamp("2020-01-10")


def test_pivot_signal():
    df = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(["2020-01-10", "2020-01-10", "2020-01-17"]),
            "etf": ["A", "B", "A"],
            "signal": [1.0, -1.0, 2.0],
        }
    )
    wide = pivot_signal(df)
    assert wide.loc[pd.Timestamp("2020-01-10"), "A"] == 1.0
    assert wide.loc[pd.Timestamp("2020-01-10"), "B"] == -1.0
    assert math.isnan(wide.loc[pd.Timestamp("2020-01-17"), "B"])


def test_align_to_rebalance_dates_uses_most_recent_public_signal():
    signal_wide = pd.DataFrame(
        {"A": [1.0, 2.0]},
        index=pd.to_datetime(["2020-01-10", "2020-01-17"]),
    )
    rebalance_dates = pd.to_datetime(["2020-01-08", "2020-01-12", "2020-01-20"])
    aligned = align_to_rebalance_dates(signal_wide, rebalance_dates)

    assert math.isnan(aligned.loc[pd.Timestamp("2020-01-08"), "A"])  # before any signal existed
    assert aligned.loc[pd.Timestamp("2020-01-12"), "A"] == 1.0  # only the Jan-10 signal is public yet
    assert aligned.loc[pd.Timestamp("2020-01-20"), "A"] == 2.0  # Jan-17 signal now public


def test_signal_to_weights_long_only_normalizes_positive_signals():
    signal = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [-2.0]})
    weights = signal_to_weights(signal)
    assert weights.loc[0, "A"] == pytest.approx(0.25)
    assert weights.loc[0, "B"] == pytest.approx(0.75)
    assert weights.loc[0, "C"] == 0.0
    assert weights.loc[0].sum() == pytest.approx(1.0)


def test_signal_to_weights_all_negative_means_all_cash():
    signal = pd.DataFrame({"A": [-1.0], "B": [-0.5]})
    weights = signal_to_weights(signal)
    assert weights.loc[0].sum() == 0.0


def test_signal_to_weights_respects_tradeable_mask():
    signal = pd.DataFrame({"A": [1.0], "B": [3.0]})
    tradeable = pd.DataFrame({"A": [True], "B": [False]})  # B not yet listed
    weights = signal_to_weights(signal, tradeable_mask=tradeable)
    assert weights.loc[0, "A"] == pytest.approx(1.0)
    assert weights.loc[0, "B"] == 0.0
