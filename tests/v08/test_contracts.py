"""
Tests for StockReason v0.8 Core Contracts and Schema Validators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.v08.contracts import (
    SecurityMetadata,
    validate_ohlcv_dataframe,
    validate_universe_dataframe,
    compute_purge_window,
    REQUIRED_OHLCV_COLUMNS,
    REQUIRED_UNIVERSE_COLUMNS,
    SUPPORTED_HORIZONS,
)


def create_sample_ohlcv(
    n_days: int = 10,
    ticker: str = "TCS.NS",
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Helper to generate synthetically valid OHLCV dataframe."""
    dates = pd.date_range(start=start_date, periods=n_days, freq="B")
    base_price = 3500.0
    records = []
    for i, dt in enumerate(dates):
        open_p = base_price + i * 2.0
        high_p = open_p + 15.0
        low_p = open_p - 10.0
        close_p = open_p + 5.0
        records.append({
            "ticker": ticker,
            "date": dt,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "adj_close": close_p * 0.98,
            "volume": 100000 + i * 500,
        })
    return pd.DataFrame(records)


def test_valid_ohlcv_contract():
    df = create_sample_ohlcv(10)
    is_valid, errors = validate_ohlcv_dataframe(df)
    assert is_valid, f"Validation failed unexpectedly with errors: {errors}"
    assert len(errors) == 0


def test_ohlcv_missing_columns():
    df = create_sample_ohlcv(5).drop(columns=["adj_close", "volume"])
    is_valid, errors = validate_ohlcv_dataframe(df)
    assert not is_valid
    assert any("Missing required columns" in err for err in errors)


def test_ohlcv_duplicate_keys():
    df = create_sample_ohlcv(5)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    is_valid, errors = validate_ohlcv_dataframe(df)
    assert not is_valid
    assert any("Duplicate" in err for err in errors)


def test_ohlcv_non_positive_prices():
    df = create_sample_ohlcv(5)
    df.loc[2, "close"] = -10.0
    is_valid, errors = validate_ohlcv_dataframe(df)
    assert not is_valid
    assert any("Non-positive price" in err for err in errors)


def test_ohlcv_boundary_violations():
    df = create_sample_ohlcv(5)
    # High lower than open
    df.loc[1, "high"] = df.loc[1, "open"] - 5.0
    # Low higher than close
    df.loc[3, "low"] = df.loc[3, "close"] + 5.0
    is_valid, errors = validate_ohlcv_dataframe(df)
    assert not is_valid
    assert any("High price lower" in err for err in errors)
    assert any("Low price higher" in err for err in errors)


def test_ohlcv_negative_volume():
    df = create_sample_ohlcv(5)
    df.loc[0, "volume"] = -500
    is_valid, errors = validate_ohlcv_dataframe(df)
    assert not is_valid
    assert any("Negative volume" in err for err in errors)


def test_ohlcv_future_dates():
    df = create_sample_ohlcv(5)
    df.loc[4, "date"] = pd.to_datetime("2035-01-01")
    is_valid, errors = validate_ohlcv_dataframe(df)
    assert not is_valid
    assert any("Future dates" in err for err in errors)


def test_security_metadata_eligibility():
    # Listed in 2015, delisted in 2022
    sec = SecurityMetadata(
        ticker="DELISTED.NS",
        symbol_nse="DELISTED",
        isin="INE000000001",
        company_name="Old Telecom Ltd",
        sector="Telecommunication",
        listing_date="2015-01-01",
        delisting_date="2022-06-30",
        is_active=False,
        survivorship_status="delisted",
    )
    # Before listing
    assert not sec.is_eligible("2012-01-01")
    # During listing
    assert sec.is_eligible("2018-05-15")
    # After delisting
    assert not sec.is_eligible("2024-01-01")

    # Active security
    active_sec = SecurityMetadata(
        ticker="RELIANCE.NS",
        symbol_nse="RELIANCE",
        isin="INE002A01018",
        company_name="Reliance Industries Ltd",
        sector="Energy",
        listing_date="1995-11-29",
        is_active=True,
    )
    assert active_sec.is_eligible("2012-01-01")
    assert active_sec.is_eligible("2026-11-01")


def test_validate_universe_dataframe():
    valid_data = pd.DataFrame([{
        "ticker": "RELIANCE.NS",
        "symbol_nse": "RELIANCE",
        "isin": "INE002A01018",
        "company_name": "Reliance Industries Ltd",
        "sector": "Energy",
        "listing_date": "1995-11-29",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "active",
    }])
    is_valid, errors = validate_universe_dataframe(valid_data)
    assert is_valid, f"Unexpected errors: {errors}"

    # Missing column
    invalid_data = valid_data.drop(columns=["survivorship_status"])
    is_valid, errors = validate_universe_dataframe(invalid_data)
    assert not is_valid


def test_compute_purge_window():
    assert compute_purge_window(1, 5) == 6
    assert compute_purge_window(5, 5) == 10
    assert compute_purge_window(21, 5) == 26
    assert compute_purge_window(63, 5) == 68
    assert compute_purge_window(126, 5) == 131

    with pytest.raises(ValueError):
        compute_purge_window(0)
