import math

import pandas as pd
import pytest

from etf_cot_backtest.signals import compute_commercial_zscore, net_pct_open_interest


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
