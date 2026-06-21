import math

import pandas as pd
import pytest

from etf_cot_backtest.metrics import compare_strategies, performance_summary


def test_constant_positive_returns():
    returns = pd.Series([0.01] * 52)
    summary = performance_summary(returns, periods_per_year=52)

    expected_total = 1.01**52 - 1
    assert summary["total_return"] == pytest.approx(expected_total)
    assert summary["cagr"] == pytest.approx(expected_total)  # exactly 1 year of data
    assert summary["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(summary["sharpe"])  # zero vol -> undefined Sharpe
    assert summary["max_drawdown"] == pytest.approx(0.0)
    assert math.isnan(summary["calmar"])  # zero drawdown -> undefined Calmar
    assert summary["win_rate"] == 1.0
    assert summary["n_periods"] == 52


def test_drawdown_and_cagr_with_a_loss():
    returns = pd.Series([0.10, -0.20, 0.10])
    summary = performance_summary(returns, periods_per_year=52)

    nav_final = 1.10 * 0.80 * 1.10
    expected_cagr = nav_final ** (52 / 3) - 1
    assert summary["total_return"] == pytest.approx(nav_final - 1)
    assert summary["cagr"] == pytest.approx(expected_cagr)
    assert summary["max_drawdown"] == pytest.approx(-0.20)
    assert summary["win_rate"] == pytest.approx(2 / 3)


def test_empty_series_returns_nans_not_errors():
    summary = performance_summary(pd.Series([], dtype=float))
    assert summary["n_periods"] == 0
    assert math.isnan(summary["cagr"])


def test_compare_strategies_builds_one_row_per_series():
    a = pd.Series([0.01, 0.02])
    b = pd.Series([0.00, 0.00])
    table = compare_strategies({"a": a, "b": b}, periods_per_year=52)

    assert set(table.index) == {"a", "b"}
    assert table.loc["a", "total_return"] == pytest.approx(performance_summary(a)["total_return"])
