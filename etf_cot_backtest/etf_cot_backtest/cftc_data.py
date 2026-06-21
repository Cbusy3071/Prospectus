"""Fetch and tidy the CFTC Legacy Commitments of Traders (futures-only) report.

CFTC publishes this as a one-off historical bulk file covering 1986-2016 plus a
single zip per calendar year after that, at stable, long-standing URLs under
cftc.gov/files/dea/history/. This module downloads those zips, extracts the
fixed-width/CSV text file inside, and reduces each to the handful of columns
the strategy needs: open interest and commercial long/short positions per
market per report date.

Note: this sandbox's network policy blocks cftc.gov, so the fetch functions
here cannot be exercised in this environment. They are written against the
documented, stable CFTC bulk-file URL/column conventions and exercised in the
test suite via local fixtures instead -- run them for real from an
environment with normal outbound network access.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .config import UNIVERSE

LOG = logging.getLogger(__name__)

BASE_URL = "https://www.cftc.gov/files/dea/history"
BULK_HISTORY_URL = f"{BASE_URL}/deacot1986_2016.zip"
BULK_HISTORY_TXT = "FUT86_16.txt"
ANNUAL_URL_TEMPLATE = f"{BASE_URL}/deacot{{year}}.zip"
ANNUAL_TXT = "annual.txt"

# Columns we need, with fallbacks for naming variations across eras of the
# file (confirmed by inspecting the actual 1986-2016 bulk historical file,
# which uses a "Word Word-Sub (All)" convention rather than the underscored
# "Word_Word_All" convention used by CFTC's newer Socrata-style annual
# exports -- both are kept as candidates since the bulk and annual files are
# different vintages). The first matching candidate found in the loaded
# frame is used.
COLUMN_CANDIDATES = {
    "market": ["Market_and_Exchange_Names", "Market and Exchange Names"],
    "report_date": [
        "As of Date in Form YYYY-MM-DD",
        "Report_Date_as_YYYY-MM-DD",
        "Report_Date_as_MM_DD_YYYY",
    ],
    "report_date_yymmdd": ["As_of_Date_In_Form_YYMMDD", "As of Date in Form YYMMDD"],
    "open_interest": ["Open Interest (All)", "Open_Interest_All", "Open_Interest"],
    "commercial_long": [
        "Commercial Positions-Long (All)",
        "Comm_Positions_Long_All",
        "Commercial_Positions_Long_All",
        "Commercial_Positions_Long",
    ],
    "commercial_short": [
        "Commercial Positions-Short (All)",
        "Comm_Positions_Short_All",
        "Commercial_Positions_Short_All",
        "Commercial_Positions_Short",
    ],
    "noncommercial_long": [
        "Noncommercial Positions-Long (All)",
        "NonComm_Positions_Long_All",
        "Noncommercial_Positions_Long_All",
        "Noncommercial_Positions_Long",
    ],
    "noncommercial_short": [
        "Noncommercial Positions-Short (All)",
        "NonComm_Positions_Short_All",
        "Noncommercial_Positions_Short_All",
        "Noncommercial_Positions_Short",
    ],
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _parse_yymmdd(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.zfill(6)
    yy = s.str[0:2].astype(int)
    mm = s.str[2:4]
    dd = s.str[4:6]
    century = yy.apply(lambda y: 1900 + y if y >= 50 else 2000 + y)
    return pd.to_datetime(century.astype(str) + "-" + mm + "-" + dd, errors="coerce")


def _download_zip_text(url: str, inner_filename: str, timeout: int = 60) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Some annual zips vary the inner filename's case; match case-insensitively.
        names = {n.lower(): n for n in zf.namelist()}
        actual = names.get(inner_filename.lower())
        if actual is None and len(zf.namelist()) == 1:
            actual = zf.namelist()[0]
        if actual is None:
            raise FileNotFoundError(
                f"Expected {inner_filename!r} inside {url}, found {zf.namelist()}"
            )
        with zf.open(actual) as fh:
            return pd.read_csv(fh, low_memory=False)


def _tidy_raw(raw: pd.DataFrame) -> pd.DataFrame:
    col = {key: _find_column(raw, candidates) for key, candidates in COLUMN_CANDIDATES.items()}
    missing_required = [
        k for k in ("market", "open_interest", "commercial_long", "commercial_short") if col[k] is None
    ]
    if missing_required:
        raise ValueError(
            f"CFTC file missing expected columns for: {missing_required}. "
            f"Actual columns in file: {raw.columns.tolist()}"
        )

    if col["report_date"] is not None:
        date = pd.to_datetime(raw[col["report_date"]], errors="coerce")
    elif col["report_date_yymmdd"] is not None:
        date = _parse_yymmdd(raw[col["report_date_yymmdd"]])
    else:
        raise ValueError("CFTC file has no recognizable report-date column")

    tidy = pd.DataFrame(
        {
            "date": date,
            "market_name": raw[col["market"]].astype(str).str.strip(),
            "open_interest": pd.to_numeric(raw[col["open_interest"]], errors="coerce"),
            "commercial_long": pd.to_numeric(raw[col["commercial_long"]], errors="coerce"),
            "commercial_short": pd.to_numeric(raw[col["commercial_short"]], errors="coerce"),
        }
    )
    if col["noncommercial_long"] is not None:
        tidy["noncommercial_long"] = pd.to_numeric(raw[col["noncommercial_long"]], errors="coerce")
    if col["noncommercial_short"] is not None:
        tidy["noncommercial_short"] = pd.to_numeric(raw[col["noncommercial_short"]], errors="coerce")

    return tidy.dropna(subset=["date", "market_name"])


def _map_to_etf(tidy: pd.DataFrame) -> pd.DataFrame:
    """Collapse the free-text market_name column down to our ETF universe."""
    alias_to_etf = {}
    for mapping in UNIVERSE:
        for alias in mapping.cftc_market_aliases:
            alias_to_etf[alias.upper().strip()] = mapping.etf

    tidy = tidy.copy()
    tidy["etf"] = tidy["market_name"].str.upper().str.strip().map(alias_to_etf)
    matched = tidy.dropna(subset=["etf"])

    matched_etfs = set(matched["etf"].unique())
    configured_etfs = {m.etf for m in UNIVERSE}
    missing = configured_etfs - matched_etfs
    if missing:
        LOG.warning(
            "No CFTC rows matched the configured aliases for: %s. "
            "The CFTC market name text may have changed; check market_name values "
            "in the raw download and update config.UNIVERSE aliases.",
            sorted(missing),
        )
    return matched.drop(columns=["market_name"])


def fetch_legacy_cot(start_year: int = 1986, end_year: int | None = None, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Download and tidy the Legacy futures-only COT report for start_year..end_year.

    Returns a long-format DataFrame with one row per (date, etf) containing
    open_interest, commercial_long, commercial_short (and noncommercial_*
    where available). Requires outbound network access to cftc.gov.
    """
    import datetime as _dt

    if end_year is None:
        end_year = _dt.date.today().year

    cache_path = Path(cache_dir) / "legacy_cot_raw.parquet" if cache_dir else None
    if cache_path and cache_path.exists():
        LOG.info("Loading cached COT data from %s", cache_path)
        return pd.read_parquet(cache_path)

    frames = []
    if start_year <= 2016:
        LOG.info("Downloading bulk historical COT file: %s", BULK_HISTORY_URL)
        raw = _download_zip_text(BULK_HISTORY_URL, BULK_HISTORY_TXT)
        frames.append(_tidy_raw(raw))

    for year in range(max(start_year, 2017), end_year + 1):
        url = ANNUAL_URL_TEMPLATE.format(year=year)
        LOG.info("Downloading COT file for %s: %s", year, url)
        try:
            raw = _download_zip_text(url, ANNUAL_TXT)
        except requests.HTTPError as exc:
            LOG.warning("Could not download %s (%s); skipping", url, exc)
            continue
        frames.append(_tidy_raw(raw))

    combined = pd.concat(frames, ignore_index=True)
    tidy = _map_to_etf(combined)
    tidy = tidy.sort_values(["etf", "date"]).reset_index(drop=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_parquet(cache_path)

    return tidy
