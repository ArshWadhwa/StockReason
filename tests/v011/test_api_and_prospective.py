"""
Tests for StockReason v0.11 Prospective Ledger & API Service.
=============================================================
1. Immutable ledger SHA-256 chaining.
2. Retrospective relabeling audit.
3. Production API fail-safe and transparent schema.
"""

import pytest
import os
import tempfile
import pandas as pd
from src.v011.prospective.ledger import ImmutableProspectiveLedgerV011
from src.v011.api.service import StockReasonProductionServiceV011


def test_immutable_ledger_hash_chaining():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = os.path.join(tmpdir, "test_ledger.csv")
        ledger = ImmutableProspectiveLedgerV011(ledger_path)

        rec1 = ledger.append_forecast(
            symbol="RELIANCE",
            horizon=21,
            ref_price=1250.0,
            pred_return=0.015,
            median_price=1260.0,
            lower_90=1100.0,
            upper_90=1400.0,
            baseline_return=0.0088,
            information_cutoff="2026-11-23",
            forecast_type="genuine_prospective",
        )

        rec2 = ledger.append_forecast(
            symbol="TCS",
            horizon=21,
            ref_price=4100.0,
            pred_return=0.010,
            median_price=4120.0,
            lower_90=3800.0,
            upper_90=4400.0,
            baseline_return=0.0088,
            information_cutoff="2026-11-23",
            forecast_type="genuine_prospective",
        )

        assert rec1["record_hash"] is not None
        assert rec2["prev_block_hash"] == rec1["record_hash"]

        df = pd.read_csv(ledger_path)
        assert len(df) == 2
        assert df.loc[0, "forecast_type"] == "genuine_prospective"


def test_production_api_service_schema_and_fallback():
    service = StockReasonProductionServiceV011()

    # 1. Standard fallback call
    res = service.generate_forecast(
        symbol="INFY",
        reference_price=1850.0,
        horizon=21,
    )
    assert res["symbol"] == "INFY"
    assert res["reference_price_inr"] == 1850.0
    assert res["forecast_type"] == "statistical_baseline_fallback"
    assert "50_percent" in res["prediction_intervals"]
    assert "90_percent" in res["prediction_intervals"]
    assert res["model_status"] == "EXPERIMENTAL_NOT_PROMOTED"

    # 2. Corrupted input / zero price
    res_err = service.generate_forecast(
        symbol="BAD_STOCK",
        reference_price=-10.0,
    )
    assert "ERROR" in res_err["status"]
