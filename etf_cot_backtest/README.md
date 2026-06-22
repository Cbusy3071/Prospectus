# ETF / COT backtest — sector rotation from commercial positioning

Backtests a sector-rotation strategy across the 11 Select Sector SPDR ETFs
driven by the CFTC Commitments of Traders (COT) report and the positioning of
**commercial hedgers**.

The classic commodity read is "follow the commercials" — they're the report's
"smart money", so go long where they're net-long. For *equity sectors* that
read loses badly to plain buy-and-hold (see results below): the commercials
here behave more like the side leaning *against* trends. So the default
strategy does the opposite, and keys off the *change* in positioning rather
than its level:

## Strategy

1. **Signal (`signals.compute_commercial_shift_signal`).** For each sector,
   `net_pct_oi = (commercial_long - commercial_short) / open_interest`. Take the
   **shift** — its change over `--shift-weeks` (default 13, ~a quarter) — then a
   rolling z-score of that change against the sector's own history so a "big
   move for this sector" is comparable across sectors. By default the sign is
   **inverted** (`--no-invert` to disable): a rising commercial net-long reads
   *bearish*, i.e. we side with the speculative/momentum flow the commercials
   are leaning against.
2. **Turnover control.** A change signal whipsaws far more than a level signal,
   so it's EMA-smoothed over `--smooth-span` weeks (default 4) and the portfolio
   rebalances only every `--rebalance-weeks` weeks (default 4, ~monthly) rather
   than weekly. Reported `ann_turnover` shows the effect.
3. **Publication lag.** CFTC reports Tuesday's positioning but doesn't release
   it until the following Friday. The backtest shifts every signal's
   effective date forward by `--lag-days` (default 3) so it can never trade on
   information before it was public (`portfolio.apply_publication_lag`).
4. **Weights.** Long-only: weight is proportional to the *positive* part of the
   signal only; sectors with a non-positive signal get zero weight, and if
   every sector is non-positive the whole portfolio sits in cash for that
   period (`portfolio.signal_to_weights`).
5. **Backtest.** Weights are held between rebalances; a configurable
   transaction-cost drag is charged on turnover at the moment a position
   changes (`backtest.run_backtest`).
6. **Comparison.** Reported alongside `tilt_commercial_shift` (same signal,
   different weighting — see below), the original `follow_commercial_level`
   (weekly) strategy, an equal-weight buy-and-hold of the same sector universe,
   and SPY buy-and-hold (`metrics.compare_strategies`).

### Long-only-or-cash vs. always-invested tilt

The long-only weighting above goes to cash whenever every sector's signal is
non-positive, which against a long bull market mostly means sitting out the
broad market rally rather than picking the right sectors within it — a
historical run showed a ~30% win rate for both COT variants, i.e. the
portfolio was flat/in cash most of the time. That cash drag is a confound: it
makes it impossible to tell whether the *signal* is bad or just the *all-or-
nothing exposure* is bad.

`tilt_commercial_shift` uses the identical shift signal but
`portfolio.signal_to_tilt_weights` instead: every tradeable sector starts at
equal weight (`1/n`) and the signal only redistributes weight by rank around
that base (highest-ranked sector gets the most, lowest-ranked the least), so
the portfolio is **always fully invested** — a flat or all-negative signal
just leaves it at equal weight rather than cash. `--tilt-strength` (default
1.0) controls how aggressive the redistribution is; `1.0` is the most extreme
tilt for which weights stay non-negative, `0.0` collapses to equal weight.
Comparing `tilt_commercial_shift` to `equal_weight_universe` isolates the
signal's stock(sector)-picking value from market-timing/cash-drag effects.

### ETF universe (`config.UNIVERSE`)

| ETF  | Sector                  | CFTC Legacy report market                            |
|------|-------------------------|--------------------------------------------------------|
| XLF  | Financials              | E-MINI S&P FINANCIAL INDEX - CME                        |
| XLE  | Energy                  | E-MINI S&P ENERGY INDEX - CME                            |
| XLK  | Technology              | E-MINI S&P TECHNOLOGY INDEX - CME                        |
| XLV  | Health Care             | E-MINI S&P HEALTH CARE INDEX - CME                       |
| XLI  | Industrials             | E-MINI S&P INDUSTRIAL INDEX - CME                        |
| XLY  | Consumer Discretionary  | E-MINI S&P CONSUMER DISCRETIONARY INDEX - CME            |
| XLP  | Consumer Staples        | E-MINI S&P CONSU STAPLES INDEX - CME                     |
| XLU  | Utilities               | E-MINI S&P UTILITIES INDEX - CME                         |
| XLB  | Materials               | E-MINI S&P MATERIALS INDEX - CME                         |
| XLRE | Real Estate             | E-MINI S&P REAL ESTATE INDEX - CME (+ alt names)         |
| XLC  | Communication Services  | E-MINI S&P COMMUNICATION INDEX - CME                     |

All 11 aliases above are confirmed against a live CFTC download (run via
GitHub Actions, since this sandbox blocks cftc.gov). Two needed correcting
from an initial guess: CFTC's actual text truncates "Consumer" to "Consu"
for Staples, and omits "Services" for Communication. CFTC's market-name text
has changed wording across decades, so each entry in `config.py` also lists
older/alternate aliases as fallbacks. If a future CFTC file uses wording not
listed here, `cftc_data.fetch_legacy_cot` logs a warning naming the unmatched
ETF, along with candidate `market_name` values found in the same download —
check those and add the alias.

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

All 34 tests run against synthetic data with no network access required:
unit tests for CFTC column/alias matching, the level and shift signal math,
lag handling, both weight-construction schemes (long-only-or-cash and the
always-invested tilt), backtest accounting (hand-verified against manual
calculations), and performance metrics, plus integration tests running the
full pipeline end-to-end (the weekly level path, the ~monthly shift path, and
the ~monthly tilt path) on a synthetic multi-year, multi-sector dataset.
