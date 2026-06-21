"""ETF <-> CFTC futures market mapping and strategy defaults.

The CFTC "Legacy" Commitments of Traders report identifies markets by a free-text
``Market_and_Exchange_Names`` field rather than a stable numeric key in the bulk
text files, and that text has changed wording several times across decades
(e.g. crude oil's contract name changed from "CRUDE OIL, LIGHT SWEET" to
"WTI-PHYSICAL"). Each entry below lists every known historical alias so the
fetcher can match a market regardless of which era of the file it appears in.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketMapping:
    etf: str
    asset_class: str
    description: str
    cftc_market_aliases: tuple = field(default_factory=tuple)


# Multi-asset universe: each ETF is paired with the CFTC futures market whose
# Commitments of Traders positioning is the closest available proxy for it.
UNIVERSE = [
    MarketMapping(
        etf="GLD",
        asset_class="commodity",
        description="Gold",
        cftc_market_aliases=("GOLD - COMMODITY EXCHANGE INC.",),
    ),
    MarketMapping(
        etf="SLV",
        asset_class="commodity",
        description="Silver",
        cftc_market_aliases=("SILVER - COMMODITY EXCHANGE INC.",),
    ),
    MarketMapping(
        etf="USO",
        asset_class="commodity",
        description="WTI Crude Oil",
        cftc_market_aliases=(
            "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE",
            "CRUDE OIL, LIGHT SWEET-WTI - NEW YORK MERCANTILE EXCHANGE",
            "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
        ),
    ),
    MarketMapping(
        etf="UNG",
        asset_class="commodity",
        description="Natural Gas",
        cftc_market_aliases=(
            "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
            "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE",
        ),
    ),
    MarketMapping(
        etf="CPER",
        asset_class="commodity",
        description="Copper",
        cftc_market_aliases=(
            "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
            "COPPER- #1 - COMMODITY EXCHANGE INC.",
        ),
    ),
    MarketMapping(
        etf="UUP",
        asset_class="currency",
        description="US Dollar Index",
        cftc_market_aliases=(
            "USD INDEX - ICE FUTURES U.S.",
            "U.S. DOLLAR INDEX - ICE FUTURES U.S.",
        ),
    ),
    MarketMapping(
        etf="FXE",
        asset_class="currency",
        description="Euro FX",
        cftc_market_aliases=("EURO FX - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="FXY",
        asset_class="currency",
        description="Japanese Yen",
        cftc_market_aliases=("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="FXA",
        asset_class="currency",
        description="Australian Dollar",
        cftc_market_aliases=("AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",),
    ),
    MarketMapping(
        etf="TLT",
        asset_class="rates",
        description="20+ Year Treasury Bonds",
        cftc_market_aliases=(
            "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
            "LONG-TERM U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
        ),
    ),
    MarketMapping(
        etf="IEF",
        asset_class="rates",
        description="7-10 Year Treasury Notes",
        cftc_market_aliases=(
            "UST 10Y NOTE - CHICAGO BOARD OF TRADE",
            "U.S. TREASURY NOTES, 10-YR - CHICAGO BOARD OF TRADE",
            "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
        ),
    ),
    MarketMapping(
        etf="SPY",
        asset_class="equity_index",
        description="S&P 500",
        cftc_market_aliases=(
            "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
            "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
        ),
    ),
]

ETF_TICKERS = tuple(m.etf for m in UNIVERSE)

# Strategy defaults.
LOOKBACK_WEEKS = 156           # ~3 years of weekly COT prints for the z-score window
MIN_LOOKBACK_WEEKS = 52        # require at least 1 year of history before trusting a z-score
PUBLICATION_LAG_DAYS = 3       # CFTC reports Tuesday positioning, releases it the following Friday
TRANSACTION_COST_BPS = 5.0     # round-trip cost assumption per unit of turnover
BENCHMARK_TICKER = "SPY"
