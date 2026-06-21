# ETF / COT backtest — "follow the commercials"

Backtests a multi-asset ETF portfolio that allocates based on the CFTC
Commitments of Traders (COT) report: weight toward markets where **commercial
hedgers** (producers, merchants, end-users — the report's "smart money") are
more net-long than their own recent history, on the theory that they have
better information about their own market than speculators do.

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
5. **Benchmarks.** Equal-weight buy-and-hold of the same ETF universe, and SPY
   buy-and-hold (`backtest.equal_weight_benchmark`, `metrics.compare_strategies`).

### ETF universe (`config.UNIVERSE`)

| ETF  | Market               | CFTC Legacy report market                  |
|------|-----------------------|---------------------------------------------|
| GLD  | Gold                  | GOLD - COMMODITY EXCHANGE INC.               |
| SLV  | Silver                | SILVER - COMMODITY EXCHANGE INC.             |
| USO  | WTI Crude Oil         | WTI-PHYSICAL - NYMEX (+ historical aliases)  |
| UNG  | Natural Gas           | NATURAL GAS - NYMEX                          |
| CPER | Copper                | COPPER-GRADE #1 - COMEX                      |
| UUP  | US Dollar Index       | USD INDEX - ICE FUTURES U.S.                 |
| FXE  | Euro FX               | EURO FX - CME                                |
| FXY  | Japanese Yen          | JAPANESE YEN - CME                           |
| FXA  | Australian Dollar     | AUSTRALIAN DOLLAR - CME                      |
| TLT  | 20+ Year Treasuries   | U.S. TREASURY BONDS - CBOT                   |
| IEF  | 7-10 Year Treasuries  | UST 10Y NOTE - CBOT                          |
| SPY  | S&P 500               | E-MINI S&P 500 - CME                         |

CFTC's market-name text has changed wording several times across decades
(e.g. crude oil's contract was renamed more than once), so each entry in
`config.py` lists every known historical alias. If a future CFTC file uses
wording not listed here, `cftc_data.fetch_legacy_cot` logs a warning naming
the unmatched ETF — check the raw `market_name` values and add the alias.

## Known simplifications

- Weights are held constant between rebalances rather than drifting with
  price moves — equivalent to assuming continuous rebalancing back to target,
  not literal buy-and-hold drift. Standard simplification for a weekly
  systematic backtest, but worth knowing about.
- Cash is assumed to earn `--cash-return` (default 0%), not a real T-bill
  rate.
- Earliest ETF inception (CPER, 2011) limits how far back a fair multi-asset
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

All 21 tests run against synthetic data with no network access required:
unit tests for the signal math, lag handling, weight construction, backtest
accounting (hand-verified against manual calculations), and performance
metrics, plus one integration test running the full pipeline end-to-end on a
synthetic multi-year, multi-asset dataset.
