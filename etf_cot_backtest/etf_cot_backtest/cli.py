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
    LOOKBACK_WEEKS,
    MIN_LOOKBACK_WEEKS,
    PUBLICATION_LAG_DAYS,
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
    p.add_argument("--lag-days", type=int, default=PUBLICATION_LAG_DAYS)
    p.add_argument("--tx-cost-bps", type=float, default=TRANSACTION_COST_BPS)
    p.add_argument("--cash-return", type=float, default=0.0, help="Per-period return assumed on uninvested cash")
    return p.parse_args(argv)


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

    LOG.info("Computing commercial positioning z-score signal")
    signal_long = signals.compute_commercial_zscore(
        cot_raw, lookback=args.lookback_weeks, min_periods=args.min_lookback_weeks
    )
    signal_long = portfolio.apply_publication_lag(signal_long, lag_days=args.lag_days)
    signal_wide = portfolio.pivot_signal(signal_long)

    rebalance_dates = weekly_returns.index
    aligned_signal = portfolio.align_to_rebalance_dates(signal_wide, rebalance_dates)
    tradeable_mask = weekly_returns[list(ETF_TICKERS)].notna()
    weights = portfolio.signal_to_weights(aligned_signal, tradeable_mask=tradeable_mask)

    LOG.info("Running backtest")
    result = backtest.run_backtest(
        weights, weekly_returns[list(ETF_TICKERS)], tx_cost_bps=args.tx_cost_bps, cash_return=args.cash_return
    )
    benchmark_ew = backtest.equal_weight_benchmark(weekly_returns, list(ETF_TICKERS))
    benchmark_spy = weekly_returns[BENCHMARK_TICKER]

    summary = compare_strategies(
        {
            "cot_follow_commercials": result.returns,
            "equal_weight_universe": benchmark_ew,
            "spy_buy_and_hold": benchmark_spy,
        }
    )
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    summary.to_csv(output_dir / "performance_summary.csv")
    result.nav.to_csv(output_dir / "strategy_nav.csv", header=["nav"])
    result.held_weights.to_csv(output_dir / "weights_history.csv")
    LOG.info("Wrote results to %s", output_dir)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        result.nav.plot(ax=ax, label="COT (follow commercials)")
        (1 + benchmark_ew.fillna(0)).cumprod().plot(ax=ax, label="Equal-weight universe")
        (1 + benchmark_spy.fillna(0)).cumprod().plot(ax=ax, label="SPY buy & hold")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.set_title("COT-driven ETF portfolio vs benchmarks")
        fig.tight_layout()
        fig.savefig(output_dir / "equity_curve.png", dpi=150)
        LOG.info("Saved equity_curve.png")
    except ImportError:
        LOG.warning("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
