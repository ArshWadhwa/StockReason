"""
Unit tests for StockReason v0.8 Historical Universe Reconstruction.
"""

import pytest
import pandas as pd
from datetime import datetime

from src.v08.universe import (
    get_universe,
    get_universe_details,
    get_eligible_universe,
    get_all_tickers,
    SURVIVORSHIP_SAFE,
    BENCHMARK_INDICES,
    MACRO_TICKERS,
    UNIVERSE_TIERS,
)
from src.v08.contracts import validate_universe_dataframe


def test_universe_sizes_and_tiers():
    u16 = get_universe("U16")
    u100 = get_universe("U100")
    u200 = get_universe("U200")
    u300 = get_universe("U300")
    u500 = get_universe("U500")

    assert len(u16) == 16, f"Expected 16 in U16, got {len(u16)}"
    assert len(u100) == 100, f"Expected 100 in U100, got {len(u100)}"
    assert len(u200) == 200, f"Expected 200 in U200, got {len(u200)}"
    assert len(u300) == 300, f"Expected 300 in U300, got {len(u300)}"
    assert len(u500) == 500, f"Expected 500 in U500, got {len(u500)}"

    # Check strict hierarchical containment
    assert set(u16).issubset(set(u100))
    assert set(u100).issubset(set(u200))
    assert set(u200).issubset(set(u300))
    assert set(u300).issubset(set(u500))


def test_unique_tickers():
    u500 = get_universe("U500")
    assert len(u500) == len(set(u500)), "Tickers in U500 must be unique"
    for t in u500:
        assert t.endswith(".NS"), f"Ticker {t} must end with .NS"


def test_survivorship_safe_and_delisted_cases():
    assert SURVIVORSHIP_SAFE is True
    u500 = get_universe("U500")

    # Verify historically traded delisted / merged names are present
    delisted_cases = ["RCOM.NS", "JETAIRWAYS.NS", "UNITECH.NS", "DHFL.NS", "HDFC.NS"]
    for sym in delisted_cases:
        assert sym in u500, f"Historical case {sym} must be in U500 to mitigate survivorship bias"


def test_point_in_time_eligibility():
    # 1. Jet Airways listed 2005, grounded/delisted 2019
    el_2015 = get_eligible_universe("2015-06-01", tier="U500")
    el_2024 = get_eligible_universe("2024-06-01", tier="U500")
    assert "JETAIRWAYS.NS" in el_2015
    assert "JETAIRWAYS.NS" not in el_2024

    # 2. Zomato listed 2021-07-23
    el_2018 = get_eligible_universe("2018-01-01", tier="U500")
    el_2023 = get_eligible_universe("2023-01-01", tier="U500")
    assert "ZOMATO.NS" not in el_2018
    assert "ZOMATO.NS" in el_2023

    # 3. Reliance listed 1995
    el_2012 = get_eligible_universe("2012-01-01", tier="U500")
    assert "RELIANCE.NS" in el_2012
    assert "RELIANCE.NS" in el_2024


def test_universe_dataframe_contract():
    df = get_universe_details("U500")
    assert len(df) == 500
    is_valid, errors = validate_universe_dataframe(df)
    assert is_valid, f"Universe dataframe failed validation: {errors}"

    # Verify sector diversification
    sectors = df["sector"].unique()
    assert len(sectors) >= 15, f"Expected at least 15 sectors, got {len(sectors)}"


def test_all_tickers_with_benchmarks_and_macros():
    all_t = get_all_tickers(include_benchmarks=True, include_macro=True)
    assert len(all_t) == 500 + len(BENCHMARK_INDICES) + len(MACRO_TICKERS)
    for b in BENCHMARK_INDICES:
        assert b in all_t
    for m in MACRO_TICKERS:
        assert m in all_t
