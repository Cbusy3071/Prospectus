"""ETF <-> CFTC futures market mapping and strategy defaults.

The CFTC "Legacy" Commitments of Traders report identifies markets by a free-text
``Market_and_Exchange_Names`` field rather than a stable numeric key in the bulk
text files, and that text has changed wording several times across decades.
Each entry below lists every known/likely alias so the fetcher can match a
market regardless of which era of the file it appears in.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketMapping:
    etf: str
    asset_class: str
    description: str
    cftc_market_aliases: tuple = field(default_factory=tuple)


# Equity sector universe: the 11 Select Sector SPDR ETFs, each paired with its
# CME E-mini S&P Select Sector futures contract -- the closest available COT
# proxy for sector-level commercial positioning. CME launched the original
# nine sectors in 2011; Real Estate (2016) and Communication Services (2018)
# were added later and CME's own marketing materials use slightly different
# naming for those two, so their aliases below include more guesses. If a
# guess is wrong, cftc_data._map_to_etf logs candidate market_name values
# pulled from the live data to fix it from a single run.
UNIVERSE = [
    MarketMapping(
        etf="XLF",
        asset_class="equity_sector",
        description="Financial Select Sector",
        cftc_market_aliases=("E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="XLE",
        asset_class="equity_sector",
        description="Energy Select Sector",
        cftc_market_aliases=("E-MINI S&P ENERGY INDEX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="XLK",
        asset_class="equity_sector",
        description="Technology Select Sector",
        cftc_market_aliases=("E-MINI S&P TECHNOLOGY INDEX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="XLV",
        asset_class="equity_sector",
        description="Health Care Select Sector",
        cftc_market_aliases=("E-MINI S&P HEALTH CARE INDEX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="XLI",
        asset_class="equity_sector",
        description="Industrial Select Sector",
        cftc_market_aliases=("E-MINI S&P INDUSTRIAL INDEX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="XLY",
        asset_class="equity_sector",
        description="Consumer Discretionary Select Sector",
        cftc_market_aliases=(
            "E-MINI S&P CONSUMER DISCRETIONARY INDEX - CHICAGO MERCANTILE EXCHANGE",
            "E-MINI S&P CONSUMER DISC INDEX - CHICAGO MERCANTILE EXCHANGE",
        ),
    ),
    MarketMapping(
        etf="XLP",
        asset_class="equity_sector",
        description="Consumer Staples Select Sector",
        cftc_market_aliases=(
            "E-MINI S&P CONSU STAPLES INDEX - CHICAGO MERCANTILE EXCHANGE",
            "E-MINI S&P CONSUMER STAPLES INDEX - CHICAGO MERCANTILE EXCHANGE",
        ),
    ),
    MarketMapping(
        etf="XLU",
        asset_class="equity_sector",
        description="Utilities Select Sector",
        cftc_market_aliases=("E-MINI S&P UTILITIES INDEX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="XLB",
        asset_class="equity_sector",
        description="Materials Select Sector",
        cftc_market_aliases=("E-MINI S&P MATERIALS INDEX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="XLRE",
        asset_class="equity_sector",
        description="Real Estate Select Sector",
        cftc_market_aliases=(
            "E-MINI S&P REAL ESTATE INDEX - CHICAGO MERCANTILE EXCHANGE",
            "E-MINI REAL ESTATE SELECT SECTOR INDEX - CHICAGO MERCANTILE EXCHANGE",
            "E-MINI S&P REAL ESTATE SELECT SECTOR INDEX - CHICAGO MERCANTILE EXCHANGE",
        ),
    ),
    MarketMapping(
        etf="XLC",
        asset_class="equity_sector",
        description="Communication Services Select Sector",
        cftc_market_aliases=(
            "E-MINI S&P COMMUNICATION INDEX - CHICAGO MERCANTILE EXCHANGE",
            "E-MINI S&P COMMUNICATION SERVICES INDEX - CHICAGO MERCANTILE EXCHANGE",
            "E-MINI S&P COMMUNICATION SERVICES SELECT SECTOR INDEX - CHICAGO MERCANTILE EXCHANGE",
        ),
    ),
]

ETF_TICKERS = tuple(m.etf for m in UNIVERSE)

# Strategy defaults.
LOOKBACK_WEEKS = 156           # ~3 years of weekly COT prints for the z-score window
MIN_LOOKBACK_WEEKS = 52        # require at least 1 year of history before trusting a z-score
SHIFT_WEEKS = 13               # measure the *change* in commercial positioning over ~a quarter
SIGNAL_SMOOTH_SPAN = 4         # EMA span (weeks) to damp whipsaw in the change signal
INVERT_SIGNAL = True           # fade the commercials: rising commercial net-long reads bearish
REBALANCE_WEEKS = 4            # rebalance every Nth weekly print (~monthly) to cut turnover
TILT_STRENGTH = 1.0            # 0 = equal-weight, 1 = max tilt that still keeps weights non-negative
MEANREV_THRESHOLD = 1.0        # z-score deadband: only trade once positioning is this extreme
PUBLICATION_LAG_DAYS = 3       # CFTC reports Tuesday positioning, releases it the following Friday
TRANSACTION_COST_BPS = 5.0     # round-trip cost assumption per unit of turnover
BENCHMARK_TICKER = "SPY"
