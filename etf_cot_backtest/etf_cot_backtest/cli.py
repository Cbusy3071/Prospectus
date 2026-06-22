"""End-to-end runner: fetch data, build the signal, backtest, report results.

Example:
    python -m etf_cot_backtest.cli --start 2011-01-01 --cache-dir ./data_cache --output-dir ./output

This needs outbound network access to cftc.gov and Yahoo Finance, which is
not available in every environment (e.g. this is blocked in sandboxed Claude
Code web sessions) -- run it somewhere with normal internet access.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import backtest, cftc_data, portfolio, price_data, signals
from .config import (
    BENCHMARK_TICKER,
    ETF_TICKERS,
    INVERT_SIGNAL,
    LOOKBACK_WEEKS,
    MIN_LOOKBACK_WEEKS,
    PUBLICATION_LAG_DAYS,
    REBALANCE_WEEKS,
    SHIFT_WEEKS,
    SIGNAL_SMOOTH_SPAN,
    TRANSACTION_COST_BPS,
)
from .metrics import compare_strategies

LOG = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2011-01-01", help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD); defaults to today")
    p.add_argument("--cache-dir", default="./data_cache")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--lookback-weeks", type=int, default=LOOKBACK_WEEKS)
    p.add_argument("--min-lookback-weeks", type=int, default=MIN_LOOKBACK_WEEKS)
    p.add_argument("--shift-weeks", type=int, default=SHIFT_WEEKS,
                   help="Window over which to measure the change in commercial positioning")
    p.add_argument("--smooth-span", type=int, default=SIGNAL_SMOOTH_SPAN,
                   help="EMA span (weeks) used to damp whipsaw in the change signal")
    p.add_argument("--rebalance-weeks", type=int, default=REBALANCE_WEEKS,
                   help="Rebalance every Nth weekly print (~monthly at 4) to cut turnover")
    p.add_argument("--no-invert", dest="invert", action="store_false", default=INVERT_SIGNAL,
                   help="Follow (instead of fade) the commercial shift")
    p.add_argument("--lag-days", type=int, default=PUBLICATION_LAG_DAYS)
    p.add_argument("--tx-cost-bps", type=float, default=TRANSACTION_COST_BPS)
    p.add_argument("--cash-return", type=float, default=0.0, help="Per-period return assumed on uninvested cash")
    return p.parse_args(argv)


def _annualized_turnover(turnover: pd.Series, periods_per_year: int = 52) -> float:
    """Average one-way turnover per year (sum of |weight changes| / number of years)."""
    n_years = len(turnover) / periods_per_year
    return float(turnover.sum() / n_years) if n_years else float("nan")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_year = pd.Timestamp(args.start).year
    LOG.info("Fetching CFTC Legacy COT report from %s onward", start_year)
    cot_raw = cftc_data.fetch_legacy_cot(start_year=start_year, cache_dir=cache_dir)

    tickers = list(ETF_TICKERS)
    if BENCHMARK_TICKER not in tickers:
        tickers = tickers + [BENCHMARK_TICKER]

    LOG.info("Fetching ETF prices for %s", tickers)
    daily_prices = price_data.fetch_adjusted_close(tickers, start=args.start, end=args.end, cache_dir=cache_dir)
    weekly_returns = price_data.to_weekly_returns(daily_prices)

    returns_universe = weekly_returns[list(ETF_TICKERS)]

    def _build_weights(signal_long: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
        signal_long = portfolio.apply_publication_lag(signal_long, lag_days=args.lag_days)
        signal_wide = portfolio.pivot_signal(signal_long)
        aligned = portfolio.align_to_rebalance_dates(signal_wide, rebalance_dates)
        tradeable = returns_universe.loc[rebalance_dates].notna()
        return portfolio.signal_to_weights(aligned, tradeable_mask=tradeable)

    # New strategy: fade the *change* in commercial positioning, rebalanced
    # ~monthly to keep turnover down.
    LOG.info(
        "Computing commercial-shift signal (shift=%dw, smooth=%dw, invert=%s, rebalance=%dw)",
        args.shift_weeks, args.smooth_span, args.invert, args.rebalance_weeks,
    )
    shift_signal = signals.compute_commercial_shift_signal(
        cot_raw,
        shift_weeks=args.shift_weeks,
        lookback=args.lookback_weeks,
        min_periods=args.min_lookback_weeks,
        smooth_span=args.smooth_span,
        invert=args.invert,
    )
    rebalance_dates = portfolio.subsample_rebalance_dates(weekly_returns.index, args.rebalance_weeks)
    shift_weights = _build_weights(shift_signal, rebalance_dates)

    # Reference: the original level "follow the commercials" strategy, weekly.
    level_signal = signals.compute_commercial_zscore(
        cot_raw, lookback=args.lookback_weeks, min_periods=args.min_lookback_weeks
    )
    level_weights = _build_weights(level_signal, weekly_returns.index)

    LOG.info("Running backtests")
    shift_result = backtest.run_backtest(
        shift_weights, returns_universe, tx_cost_bps=args.tx_cost_bps, cash_return=args.cash_return
    )
    level_result = backtest.run_backtest(
        level_weights, returns_universe, tx_cost_bps=args.tx_cost_bps, cash_return=args.cash_return
    )
    benchmark_ew = backtest.equal_weight_benchmark(weekly_returns, list(ETF_TICKERS))
    benchmark_spy = weekly_returns[BENCHMARK_TICKER]

    summary = compare_strategies(
        {
            "fade_commercial_shift": shift_result.returns,
            "follow_commercial_level": level_result.returns,
            "equal_weight_universe": benchmark_ew,
            "spy_buy_and_hold": benchmark_spy,
        }
    )
    summary["ann_turnover"] = [
        _annualized_turnover(shift_result.turnover),
        _annualized_turnover(level_result.turnover),
        0.0,  # static buy-and-hold benchmarks
        0.0,
    ]
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    summary.to_csv(output_dir / "performance_summary.csv")
    shift_result.nav.to_csv(output_dir / "strategy_nav.csv", header=["nav"])
    shift_result.held_weights.to_csv(output_dir / "weights_history.csv")
    LOG.info("Wrote results to %s", output_dir)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        shift_result.nav.plot(ax=ax, label="Fade commercial shift (~monthly)")
        level_result.nav.plot(ax=ax, label="Follow commercial level (weekly)")
        (1 + benchmark_ew.fillna(0)).cumprod().plot(ax=ax, label="Equal-weight universe")
        (1 + benchmark_spy.fillna(0)).cumprod().plot(ax=ax, label="SPY buy & hold")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.set_title("COT sector strategies vs benchmarks")
        fig.tight_layout()
        fig.savefig(output_dir / "equity_curve.png", dpi=150)
        LOG.info("Saved equity_curve.png")
    except ImportError:
        LOG.warning("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
