import math

import pandas as pd
import pytest

from etf_cot_backtest.signals import (
    compute_commercial_meanrev_signal,
    compute_commercial_shift_signal,
    compute_commercial_zscore,
    net_pct_open_interest,
)


def _cot_rows(etf, net_pct_oi_sequence, start="2020-01-07"):
    dates = pd.date_range(start, periods=len(net_pct_oi_sequence), freq="7D")
    # open_interest fixed at 100, commercial_short fixed at 0, so
    # (commercial_long - commercial_short) / open_interest == the requested ratio.
    return pd.DataFrame(
        {
            "date": dates,
            "etf": etf,
            "open_interest": 100,
            "commercial_long": [v * 100 for v in net_pct_oi_sequence],
            "commercial_short": 0,
        }
    )


def test_net_pct_open_interest():
    df = _cot_rows("A", [0.1, 0.2, -0.3])
    out = net_pct_open_interest(df)
    assert out["net_pct_oi"].tolist() == [0.1, 0.2, -0.3]


def test_zscore_matches_hand_calculation():
    # Last 4 values [0.1, 0.1, 0.1, 0.5] -> mean 0.2, std (ddof=0) sqrt(0.03),
    # z = 0.3 / sqrt(0.03) = sqrt(3).
    df = _cot_rows("A", [0.1, 0.1, 0.1, 0.1, 0.5])
    out = compute_commercial_zscore(df, lookback=4, min_periods=4)

    row = out[out["etf"] == "A"].reset_index(drop=True)
    assert row.loc[0:2, "signal"].isna().all()  # fewer than min_periods observations
    assert math.isnan(row.loc[3, "signal"])  # zero variance window -> 0/0
    assert row.loc[4, "signal"] == pytest.approx(math.sqrt(3))


def test_zscore_is_independent_per_etf():
    df = pd.concat(
        [
            _cot_rows("A", [0.1, 0.1, 0.1, 0.1, 0.5]),
            _cot_rows("B", [0.2, 0.2, 0.2, 0.2, 0.2]),
        ],
        ignore_index=True,
    )
    out = compute_commercial_zscore(df, lookback=4, min_periods=4)

    a_signal = out[out["etf"] == "A"].sort_values("date")["signal"].iloc[-1]
    b_signal = out[out["etf"] == "B"].sort_values("date")["signal"].iloc[-1]

    assert a_signal == pytest.approx(math.sqrt(3))
    assert math.isnan(b_signal)  # constant series -> zero variance -> undefined z-score


def test_shift_signal_measures_change_not_level():
    # Steadily rising net_pct_oi: the *level* is high at the end, but the
    # *shift* (change) is constant, so a shift signal keys off the slope.
    df = _cot_rows("A", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    out = compute_commercial_shift_signal(
        df, shift_weeks=1, lookback=3, min_periods=2, smooth_span=0, invert=False
    )
    row = out[out["etf"] == "A"].reset_index(drop=True)
    # shift over 1 week is a constant +0.1 once it exists
    assert row["shift"].dropna().round(6).eq(0.1).all()


def test_shift_signal_inversion_flips_sign():
    seq = [0.0, 0.1, 0.05, 0.2, 0.1, 0.3, 0.15, 0.4]
    common = dict(shift_weeks=1, lookback=4, min_periods=2, smooth_span=0)
    plain = compute_commercial_shift_signal(_cot_rows("A", seq), invert=False, **common)
    inverted = compute_commercial_shift_signal(_cot_rows("A", seq), invert=True, **common)

    p = plain["signal"].reset_index(drop=True)
    i = inverted["signal"].reset_index(drop=True)
    both_defined = p.notna() & i.notna()
    assert both_defined.any()
    import numpy as np
    assert np.allclose(i[both_defined].to_numpy(), -p[both_defined].to_numpy())


def test_shift_signal_smoothing_reduces_volatility():
    rng_seq = [0.0, 0.3, -0.2, 0.4, -0.3, 0.5, -0.1, 0.6, 0.0, 0.7, -0.2, 0.8]
    raw = compute_commercial_shift_signal(
        _cot_rows("A", rng_seq), shift_weeks=1, lookback=4, min_periods=2, smooth_span=0, invert=False
    )
    smoothed = compute_commercial_shift_signal(
        _cot_rows("A", rng_seq), shift_weeks=1, lookback=4, min_periods=2, smooth_span=4, invert=False
    )
    assert smoothed["signal"].dropna().diff().abs().mean() < raw["signal"].dropna().diff().abs().mean()


def test_meanrev_level_z_matches_plain_zscore():
    # The mean-reversion signal's level_z is the same rolling z-score as the
    # standalone level signal -- it just gets faded + thresholded afterwards.
    df = _cot_rows("A", [0.1, 0.1, 0.1, 0.1, 0.5])
    mr = compute_commercial_meanrev_signal(df, lookback=4, min_periods=4)
    plain = compute_commercial_zscore(df, lookback=4, min_periods=4)
    assert mr["level_z"].iloc[-1] == pytest.approx(plain["signal"].iloc[-1])
    assert mr["level_z"].iloc[-1] == pytest.approx(math.sqrt(3))


def test_meanrev_crowded_long_sells_extreme_short_buys():
    # invert=True: commercials crowded net-long -> non-positive signal (cash),
    # commercials at an extreme net-short deviation -> positive signal (buy).
    crowded_long = compute_commercial_meanrev_signal(
        _cot_rows("A", [0.1, 0.1, 0.1, 0.1, 0.5]), lookback=4, min_periods=4, threshold=1.0, invert=True
    )
    extreme_short = compute_commercial_meanrev_signal(
        _cot_rows("A", [0.5, 0.5, 0.5, 0.5, 0.1]), lookback=4, min_periods=4, threshold=1.0, invert=True
    )
    assert crowded_long["signal"].iloc[-1] < 0  # sell out when it overcrowds
    assert extreme_short["signal"].iloc[-1] > 0  # buy the high net-short deviation


def test_meanrev_threshold_widens_deadband():
    # Raising the threshold by d lowers the signal by exactly d everywhere, so a
    # reading that was a (small) buy can be pushed back into the no-trade zone.
    seq = [0.5, 0.5, 0.5, 0.5, 0.1]
    low = compute_commercial_meanrev_signal(_cot_rows("A", seq), lookback=4, min_periods=4, threshold=1.0, invert=True)
    high = compute_commercial_meanrev_signal(_cot_rows("A", seq), lookback=4, min_periods=4, threshold=2.0, invert=True)
    assert (low["signal"].iloc[-1] - high["signal"].iloc[-1]) == pytest.approx(1.0)
    assert low["signal"].iloc[-1] > 0 and high["signal"].iloc[-1] < 0
