"""
Tests for StockReason v0.9.2 Dual Forecast Service & Parity Verification.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.v091.features.pipeline import FEATURE_COLS_21, StockReasonPipelineV091
from src.v092.api.dual_forecast_service import StockReasonDualForecastServiceV092
from src.v092.calibration.intervals import MultiHorizonConformalCalibrator


def test_dual_forecast_service_execution_and_parity():
    # Load panel to fit toy Ridge models for parity test
    panel_df = pd.read_parquet(
        "data/v0.9.1/features/panel_features.parquet",
        columns=["symbol", "date", "close", "target_return_21d", "target_demeaned_21d"] + FEATURE_COLS_21
    )
    panel_df["date_str"] = pd.to_datetime(panel_df["date"]).dt.strftime("%Y-%m-%d")
    clean = panel_df[panel_df["symbol"].isin(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"])].dropna(subset=FEATURE_COLS_21 + ["target_return_21d"])
    
    scaler_a = StandardScaler().fit(clean[FEATURE_COLS_21].values)
    model_a = Ridge(alpha=100.0, random_state=42).fit(scaler_a.transform(clean[FEATURE_COLS_21].values), clean["target_return_21d"].values)
    pipe_a = StockReasonPipelineV091(scaler=scaler_a, model=model_a, version="0.9.2")

    scaler_b = StandardScaler().fit(clean[FEATURE_COLS_21].values)
    model_b = Ridge(alpha=100.0, random_state=42).fit(scaler_b.transform(clean[FEATURE_COLS_21].values), clean["target_demeaned_21d"].values)
    pipe_b = StockReasonPipelineV091(scaler=scaler_b, model=model_b, version="0.9.2")

    calibrator = MultiHorizonConformalCalibrator(horizons=[21])
    calibrator.fit_horizon(21, clean["target_return_21d"].values, model_a.predict(scaler_a.transform(clean[FEATURE_COLS_21].values)), clean["vol_60d"].values)

    service = StockReasonDualForecastServiceV092(
        engine_a_pipeline=pipe_a,
        engine_b_pipeline=pipe_b,
        conformal_calibrator=calibrator,
    )

    # Test forecast for RELIANCE at latest_row date
    latest_row = clean[clean["symbol"] == "RELIANCE"].iloc[-1]
    res = service.forecast_security("RELIANCE", horizon_days=21, as_of_date=latest_row["date_str"])
    
    assert res["symbol"] == "RELIANCE"
    assert "engine_a_absolute_price" in res
    assert "engine_b_relative_ranking" in res
    assert res["probability_positive_return"] is None # Gated
    assert res["engine_a_absolute_price"]["reference_price_inr"] > 0
    assert res["engine_a_absolute_price"]["predicted_target_price_inr"] > 0
    assert len(res["engine_a_absolute_price"]["interval_95_pct_inr"]) == 2

    # Parity test: compare batch prediction vs service prediction
    latest_row = clean[clean["symbol"] == "RELIANCE"].iloc[-1]
    f_batch = latest_row[FEATURE_COLS_21].values.astype(float).reshape(1, -1)
    pred_batch_a = float(pipe_a.predict(f_batch)[0])
    pred_service_a = res["engine_a_absolute_price"]["predicted_absolute_return"]
    
    diff = abs(pred_batch_a - pred_service_a)
    assert diff < 1e-4 # Parity holds within numerical tolerance
