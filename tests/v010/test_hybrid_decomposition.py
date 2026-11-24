"""
Unit tests for StockReason v0.10 Hybrid Return Decomposition.
"""

import pytest
import pandas as pd
import numpy as np

from src.v010.decomposition.hybrid_engine import HybridReturnDecompositionEngine
from src.v091.features.pipeline import FEATURE_COLS_21


def test_hybrid_return_decomposition_fit_and_predict():
    # Load sample panel data
    panel_df = pd.read_parquet(
        "data/v0.9.1/features/panel_features.parquet",
        columns=["date", "symbol", "close", "target_return_21d"] + FEATURE_COLS_21
    )
    panel_df["date_str"] = pd.to_datetime(panel_df["date"]).dt.strftime("%Y-%m-%d")
    
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
    sub = panel_df[panel_df["symbol"].isin(symbols)].dropna(subset=FEATURE_COLS_21 + ["target_return_21d", "close"]).copy()
    
    train_df = sub[sub["date_str"] < "2023-01-01"]
    val_df = sub[sub["date_str"] >= "2023-01-01"]

    engine = HybridReturnDecompositionEngine(
        market_model_type="ridge",
        idiosyncratic_model_type="ridge",
        horizon=21,
    )
    engine.fit(train_df, features=FEATURE_COLS_21)
    
    preds = engine.predict(val_df, features=FEATURE_COLS_21)
    assert len(preds["pred_total_return"]) == len(val_df)
    assert len(preds["pred_inr_price"]) == len(val_df)
    assert all(preds["pred_inr_price"] > 0)
    assert "component_market" in preds
    assert "component_sector" in preds
    assert "component_idiosyncratic" in preds

    metrics = engine.evaluate(val_df, features=FEATURE_COLS_21)
    assert metrics["return_mae"] > 0
    assert metrics["inr_price_mae"] > 0
    assert "Hybrid Decomposition" in metrics["model_name"]
