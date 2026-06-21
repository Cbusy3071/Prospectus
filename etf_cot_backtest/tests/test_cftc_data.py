import pandas as pd

from etf_cot_backtest.cftc_data import _map_to_etf, _tidy_raw


def _raw_legacy_row(market_name, report_date, open_interest, comm_long, comm_short):
    return {
        "Market_and_Exchange_Names": market_name,
        "Report_Date_as_YYYY-MM-DD": report_date,
        "Open_Interest_All": open_interest,
        "Commercial_Positions_Long_All": comm_long,
        "Commercial_Positions_Short_All": comm_short,
        "Noncommercial_Positions_Long_All": comm_long // 2,
        "Noncommercial_Positions_Short_All": comm_short // 2,
    }


def test_tidy_raw_extracts_expected_columns():
    raw = pd.DataFrame(
        [
            _raw_legacy_row("GOLD - COMMODITY EXCHANGE INC.", "2020-01-07", 100, 60, 40),
            _raw_legacy_row("GOLD - COMMODITY EXCHANGE INC.", "2020-01-14", 110, 55, 45),
        ]
    )
    tidy = _tidy_raw(raw)
    assert list(tidy["market_name"]) == ["GOLD - COMMODITY EXCHANGE INC."] * 2
    assert tidy["date"].tolist() == [pd.Timestamp("2020-01-07"), pd.Timestamp("2020-01-14")]
    assert tidy["commercial_long"].tolist() == [60, 55]


def test_tidy_raw_falls_back_to_yymmdd_date_column():
    raw = pd.DataFrame(
        [
            {
                "Market_and_Exchange_Names": "SILVER - COMMODITY EXCHANGE INC.",
                "As_of_Date_In_Form_YYMMDD": "200107",
                "Open_Interest_All": 50,
                "Commercial_Positions_Long_All": 20,
                "Commercial_Positions_Short_All": 10,
            }
        ]
    )
    tidy = _tidy_raw(raw)
    assert tidy["date"].iloc[0] == pd.Timestamp("2020-01-07")


def test_tidy_raw_accepts_abbreviated_comm_columns():
    # Real CFTC Legacy files abbreviate "Commercial" to "Comm" / "Noncommercial"
    # to "NonComm" in their actual column headers.
    raw = pd.DataFrame(
        [
            {
                "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
                "Report_Date_as_YYYY-MM-DD": "2020-01-07",
                "Open_Interest_All": 100,
                "Comm_Positions_Long_All": 60,
                "Comm_Positions_Short_All": 40,
                "NonComm_Positions_Long_All": 30,
                "NonComm_Positions_Short_All": 20,
            }
        ]
    )
    tidy = _tidy_raw(raw)
    assert tidy["commercial_long"].tolist() == [60]
    assert tidy["commercial_short"].tolist() == [40]
    assert tidy["noncommercial_long"].tolist() == [30]
    assert tidy["noncommercial_short"].tolist() == [20]


def test_map_to_etf_matches_known_alias():
    tidy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-07", "2020-01-07"]),
            "market_name": ["GOLD - COMMODITY EXCHANGE INC.", "SOME UNMAPPED MARKET - CBOT"],
            "open_interest": [100, 200],
            "commercial_long": [60, 90],
            "commercial_short": [40, 80],
        }
    )
    mapped = _map_to_etf(tidy)
    assert list(mapped["etf"]) == ["GLD"]
    assert "market_name" not in mapped.columns
