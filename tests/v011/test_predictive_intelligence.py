"""
Tests for StockReason v0.11 Predictive Intelligence.
====================================================
1. Statistical baselines evaluation.
2. Baseline plus residual learning engine.
3. Target formulation price reconstruction.
4. Selective prediction coverage gating.
"""

import pytest
import numpy as np
import pandas as pd

from src.v011.evaluation.baselines import compute_price_metrics, evaluate_ten_strict_baselines_v011
from src.v011.models.residual_learning import BaselinePlusResidualLearningEngine
from src.v011.models.target_formulations import compare_target_formulations_v011
from src.v011.models.selective_prediction import evaluate_selective_prediction_v011
from src.v011.models.neural_diagnostics import CorrectedLightweightLinearNet
import torch


def test_baseline_metrics_computation():
    y_true = np.array([0.05, -0.02, 0.08, 0.00])
    y_pred = np.array([0.04, -0.01, 0.06, 0.01])
    close = np.array([100.0, 200.0, 50.0, 1000.0])

    m = compute_price_metrics(y_true, y_pred, close)
    assert m["return_mae"] > 0.0
    assert m["inr_price_mae"] > 0.0
    assert m["directional_accuracy_pct"] == 75.0 # 3 out of 4 agree in sign


def test_residual_learning_engine():
    np.random.seed(42)
    n = 200
    df_tr = pd.DataFrame({
        "close": np.random.uniform(100, 2000, n),
        "f1": np.random.normal(0, 1, n),
        "f2": np.random.normal(0, 1, n),
        "target_return_21d": np.random.normal(0.02, 0.05, n),
    })
    df_val = pd.DataFrame({
        "close": np.random.uniform(100, 2000, 50),
        "f1": np.random.normal(0, 1, 50),
        "f2": np.random.normal(0, 1, 50),
        "target_return_21d": np.random.normal(0.02, 0.05, 50),
    })

    engine = BaselinePlusResidualLearningEngine(residual_model_type="ridge", horizon=21)
    engine.fit(df_tr, features=["f1", "f2"], dev_df=df_val)

    preds = engine.predict(df_val, features=["f1", "f2"])
    assert len(preds) == len(df_val)
    assert 0.0 <= engine.alpha <= 1.0

    eval_res = engine.evaluate(df_val, features=["f1", "f2"])
    assert "return_mae" in eval_res
    assert eval_res["inr_price_mae"] > 0.0


def test_selective_prediction():
    y_true = np.array([0.10, -0.08, 0.01, -0.01, 0.05, -0.04, 0.00, 0.02])
    y_pred = np.array([0.12, -0.07, 0.005, -0.002, 0.04, -0.03, 0.001, 0.01])
    close = np.array([100.0] * 8)
    widths = np.array([10.0] * 8)
    base_pred = np.full(8, 0.01)

    df_sel = evaluate_selective_prediction_v011(
        y_true, y_pred, close, widths, base_pred, coverage_levels=[1.0, 0.50]
    )
    assert len(df_sel) == 2
    assert df_sel.loc[0, "target_coverage_pct"] == 100
    assert df_sel.loc[1, "target_coverage_pct"] == 50
    # Directional accuracy on 50% highest confidence should be high
    assert df_sel.loc[1, "directional_acc_pct"] >= df_sel.loc[0, "directional_acc_pct"]


def test_corrected_lightweight_linear_net():
    model = CorrectedLightweightLinearNet(input_dim=10)
    x = torch.randn(16, 10)
    out = model(x)
    assert out.shape == (16,)
    # Verify outputs are bounded within [-0.50, 1.00]
    assert (out >= -0.50).all()
    assert (out <= 1.00).all()
