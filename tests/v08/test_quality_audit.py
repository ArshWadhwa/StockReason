"""
Unit tests for StockReason v0.8 Dataset Quality Gates and Audit Engine.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.v08.quality_audit import audit_dataset_quality
from src.v08.storage import write_parquet_atomic


@pytest.fixture
def temp_audit_env(tmp_path):
    """Sets up synthetic canonical prices and universe parquet files for testing quality audit."""
    canonical_dir = tmp_path / "canonical" / "prices"
    canonical_dir.mkdir(parents=True)
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True)
    report_path = meta_dir / "quality_report.json"

    # Create synthetic universe parquet
    u_data = [
        {"ticker": "STOCK_A.NS", "symbol_nse": "STOCK_A", "isin": "INE001A", "company_name": "Company A", "sector": "Information Technology", "listing_date": "2005-01-01", "delisting_date": None, "is_active": True, "survivorship_status": "active", "tier": "U100"},
        {"ticker": "STOCK_B.NS", "symbol_nse": "STOCK_B", "isin": "INE002B", "company_name": "Company B", "sector": "Financial Services", "listing_date": "2015-01-01", "delisting_date": None, "is_active": True, "survivorship_status": "active", "tier": "U100"},
    ]
    u_parquet = meta_dir / "universe.parquet"
    pd.DataFrame(u_data).to_parquet(str(u_parquet), index=False)

    # Stock A has 16 years (2008 to 2024) -> 15+ years category
    dates_a = pd.date_range("2008-01-01", "2024-01-01", periods=100)
    df_a = pd.DataFrame({
        "ticker": "STOCK_A.NS",
        "date": dates_a,
        "close": np.full(100, 100.0),
        "adj_close": np.full(100, 100.0),
        "split_bonus_anomaly": [False] * 100,
    })
    write_parquet_atomic(df_a, str(canonical_dir / "STOCK_A.NS.parquet"))

    # Stock B has 3 years (2021 to 2024) -> <5 years category
    dates_b = pd.date_range("2021-01-01", "2024-01-01", periods=50)
    df_b = pd.DataFrame({
        "ticker": "STOCK_B.NS",
        "date": dates_b,
        "close": np.full(50, 50.0),
        "adj_close": np.full(50, 50.0),
        "split_bonus_anomaly": [False] * 49 + [True],
    })
    write_parquet_atomic(df_b, str(canonical_dir / "STOCK_B.NS.parquet"))

    return {
        "canonical_dir": str(canonical_dir),
        "universe_parquet": str(u_parquet),
        "report_path": str(report_path),
    }


def test_quality_audit_metrics_presence(temp_audit_env):
    report = audit_dataset_quality(
        canonical_dir=temp_audit_env["canonical_dir"],
        universe_parquet=temp_audit_env["universe_parquet"],
        output_report_path=temp_audit_env["report_path"],
    )

    assert "metrics" in report
    metrics = report["metrics"]

    # Verify all 14 mandatory fields are present
    assert "1_total_verified_tickers" in metrics
    assert "2_total_historical_stock_day_rows" in metrics
    assert "3_earliest_available_record" in metrics
    assert "3_latest_available_record" in metrics
    assert "4_years_of_coverage_breakdown" in metrics
    assert "5_missing_trading_sessions_note" in metrics
    assert "6_duplicate_records_removed" in metrics
    assert "7_unresolved_abnormal_movements" in metrics
    assert "8_verified_corporate_action_coverage" in metrics
    assert "9_source_failures_count" in metrics
    assert "10_point_in_time_eligibility_coverage" in metrics
    assert "11_sector_distribution_tickers" in metrics
    assert "12_records_excluded_and_reasons" in metrics
    assert "13_most_recent_completed_session" in metrics
    assert "14_survivorship_and_provenance_limitations" in metrics

    # Verification of values
    assert metrics["1_total_verified_tickers"] == 2
    assert metrics["2_total_historical_stock_day_rows"] == 150
    assert metrics["4_years_of_coverage_breakdown"]["15_years_or_more"] == 1
    assert metrics["4_years_of_coverage_breakdown"]["less_than_5_years"] == 1
    assert metrics["7_unresolved_abnormal_movements"] == 1
    assert os.path.exists(temp_audit_env["report_path"])
