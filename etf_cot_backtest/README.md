# ETF / COT backtest — "follow the commercials"

Backtests a sector-rotation strategy across the 11 Select Sector SPDR ETFs that
allocates based on the CFTC Commitments of Traders (COT) report: weight toward
sectors where **commercial hedgers** (producers, merchants, end-users — the
report's "smart money") are more net-long their sector's futures than their
own recent history, on the theory that they have better information about
their own market than speculators do.

## Strategy

1. **Signal.** For each market, `net_pct_oi = (commercial_long -
   commercial_short) / open_interest`, then a rolling z-score of that ratio
   against the market's own trailing 3-year history (`signals.py`). Scaling by
   open interest makes markets of very different sizes comparable; the
   z-score makes markets with different structural commercial biases (e.g.
   commercials are structurally net short gold, net long the dollar index)
   comparable to each other.
2. **Publication lag.** CFTC reports Tuesday's positioning but doesn't release
   it until the following Friday. The backtest shifts every signal's
   effective date forward by `--lag-days` (default 3) so it can never trade on
   information before it was public (`portfolio.apply_publication_lag`).
3. **Weights.** Long-only: weight is proportional to *positive* z-scores only;
   markets with a non-positive signal get zero weight, and if every market is
   non-positive the whole portfolio sits in cash for that period
   (`portfolio.signal_to_weights`).
4. **Backtest.** Weekly rebalance grid (matching COT's weekly cadence), with a
   configurable transaction-cost drag charged on turnover at the moment a
   position changes (`backtest.run_backtest`).
5. **Benchmarks.** Equal-weight buy-and-hold of the same sector universe, and
   SPY buy-and-hold (`backtest.equal_weight_benchmark`, `metrics.compare_strategies`).

### ETF universe (`config.UNIVERSE`)

| ETF  | Sector                  | CFTC Legacy report market                            |
|------|-------------------------|--------------------------------------------------------|
| XLF  | Financials              | E-MINI S&P FINANCIAL INDEX - CME                        |
| XLE  | Energy                  | E-MINI S&P ENERGY INDEX - CME                            |
| XLK  | Technology              | E-MINI S&P TECHNOLOGY INDEX - CME                        |
| XLV  | Health Care             | E-MINI S&P HEALTH CARE INDEX - CME                       |
| XLI  | Industrials             | E-MINI S&P INDUSTRIAL INDEX - CME                        |
| XLY  | Consumer Discretionary  | E-MINI S&P CONSUMER DISCRETIONARY INDEX - CME            |
| XLP  | Consumer Staples        | E-MINI S&P CONSUMER STAPLES INDEX - CME                  |
| XLU  | Utilities               | E-MINI S&P UTILITIES INDEX - CME                         |
| XLB  | Materials               | E-MINI S&P MATERIALS INDEX - CME                         |
| XLRE | Real Estate             | E-MINI S&P REAL ESTATE INDEX - CME (+ alt names)         |
| XLC  | Communication Services  | E-MINI S&P COMMUNICATION SERVICES INDEX - CME (+ alt names) |

CFTC's market-name text has changed wording several times across decades, so
each entry in `config.py` lists every known/likely alias. The original nine
sector futures launched together in 2011 with consistent naming; Real Estate
(2016) and Communication Services (2018) were added later and CME's own
marketing materials use slightly different naming for those two, so their
exact CFTC `market_name` text is a best guess pending verification against a
live download. If a future CFTC file uses wording not listed here,
`cftc_data.fetch_legacy_cot` logs a warning naming the unmatched ETF, along
with candidate `market_name` values found in the same download — check those
and add the alias.

SPY (`config.BENCHMARK_TICKER`) is fetched separately as an external
buy-and-hold benchmark; it is not part of the COT-driven sector universe.

## Known simplifications

- Weights are held constant between rebalances rather than drifting with
  price moves — equivalent to assuming continuous rebalancing back to target,
  not literal buy-and-hold drift. Standard simplification for a weekly
  systematic backtest, but worth knowing about.
- Cash is assumed to earn `--cash-return` (default 0%), not a real T-bill
  rate.
- Earliest ETF inception (XLC, 2018) limits how far back a fair sector-universe
  comparison can start even though the COT data itself goes back to 1986;
  the backtest dynamically excludes any ETF not yet listed at a given
  rebalance date (`portfolio.signal_to_weights(tradeable_mask=...)`) rather
  than hard-coding a start date.

## Running it for real

**This sandbox's network policy blocks cftc.gov and Yahoo Finance** (only
GitHub and PyPI are reachable), so the live fetch can't be exercised here.
Everything is built and tested against synthetic fixtures instead — run it
for real from an environment with normal outbound network access (your own
machine, a GitHub Action, etc.):

```bash
pip install -r requirements.txt
python -m etf_cot_backtest.cli --start 2011-01-01 --cache-dir ./data_cache --output-dir ./output
```

This downloads the CFTC Legacy futures-only COT report and ETF price history
(cached locally so re-runs are fast), builds the signal and weights, runs the
backtest, prints a comparison table, and writes `performance_summary.csv`,
`strategy_nav.csv`, `weights_history.csv`, and `equity_curve.png` to
`--output-dir`.

## Tests

```bash
cd etf_cot_backtest
pytest
```

All 24 tests run against synthetic data with no network access required:
unit tests for CFTC column/alias matching, the signal math, lag handling,
weight construction, backtest accounting (hand-verified against manual
calculations), and performance metrics, plus one integration test running
the full pipeline end-to-end on a synthetic multi-year, multi-sector dataset.
