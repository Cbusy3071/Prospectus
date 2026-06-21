import math

import pandas as pd
import pytest

from etf_cot_backtest.backtest import equal_weight_benchmark, run_backtest


def test_run_backtest_matches_hand_calculation():
    dates = pd.to_datetime(["2020-01-07", "2020-01-14", "2020-01-21", "2020-01-28"])
    returns = pd.DataFrame(
        {
            "X": [0.00, 0.10, 0.00, 0.02],
            "Y": [0.00, -0.05, 0.00, 0.02],
        },
        index=dates,
    )
    weights = pd.DataFrame({"X": [0.6], "Y": [0.4]}, index=[dates[0]])

    result = run_backtest(weights, returns, tx_cost_bps=10.0, cash_return=0.0)

    expected_returns = [-0.001, 0.04, 0.0, 0.02]
    for actual, expected in zip(result.returns.tolist(), expected_returns):
        assert actual == pytest.approx(expected, abs=1e-9)

    expected_turnover = [1.0, 0.0, 0.0, 0.0]
    assert result.turnover.tolist() == pytest.approx(expected_turnover, abs=1e-9)

    expected_nav_final = 0.999 * 1.04 * 1.0 * 1.02
    assert result.nav.iloc[-1] == pytest.approx(expected_nav_final, rel=1e-9)


def test_run_backtest_charges_cost_on_rebalance_only():
    dates = pd.to_datetime(["2020-01-07", "2020-01-14"])
    returns = pd.DataFrame({"X": [0.0, 0.0]}, index=dates)
    weights = pd.DataFrame({"X": [1.0, 1.0]}, index=dates)  # same weight both periods -> no 2nd trade

    result = run_backtest(weights, returns, tx_cost_bps=50.0)
    assert result.turnover.tolist() == [1.0, 0.0]
    assert result.returns.iloc[1] == pytest.approx(0.0)


def test_uninvested_fraction_earns_cash_return():
    dates = pd.to_datetime(["2020-01-07", "2020-01-14"])
    returns = pd.DataFrame({"X": [0.0, 0.10]}, index=dates)
    weights = pd.DataFrame({"X": [0.5]}, index=[dates[0]])

    result = run_backtest(weights, returns, tx_cost_bps=0.0, cash_return=0.02)
    # half invested at 10% + half cash at 2% = 0.06
    assert result.returns.iloc[1] == pytest.approx(0.06)


def test_equal_weight_benchmark_ignores_unlisted_assets():
    dates = pd.to_datetime(["2020-01-07", "2020-01-14"])
    returns = pd.DataFrame(
        {
            "OLD": [0.10, 0.10],
            "NEW": [math.nan, 0.20],  # not listed on the first date
        },
        index=dates,
    )
    bench = equal_weight_benchmark(returns)
    assert bench.iloc[0] == pytest.approx(0.10)  # only OLD existed
    assert bench.iloc[1] == pytest.approx(0.15)  # average of 0.10 and 0.20
