"""
StockReason v0.10 — Multi-Horizon Independent vs Multi-Task Comparison.
========================================================================
Remediates Phase 4 (Section 9):
Compares:
1. Approach A: Independent models (separately trained per horizon).
2. Approach B: Multi-Task Neural Network (joint encoder with normalized loss).
3. Approach C: Hierarchical consistency verification (testing whether expected return
   scales monotonically with horizon sqrt(H) or violates basic bounds).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.v092.models.baselines import compute_absolute_price_metrics


HORIZONS = [1, 5, 21, 63, 126]


def compare_independent_vs_multitask(
    X_train: np.ndarray,
    y_train_by_horizon: Dict[int, np.ndarray],
    X_val: np.ndarray,
    y_val_by_horizon: Dict[int, np.ndarray],
    val_close: np.ndarray,
    multitask_results: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compares independent single-horizon Ridge models vs Multi-Task Neural Network
    across identical out-of-sample evaluation rows.
    """
    records = []

    # 1. Independent models (Ridge benchmark)
    for h in HORIZONS:
        y_tr_h = y_train_by_horizon[h]
        y_val_h = y_val_by_horizon[h]

        valid_tr = ~np.isnan(y_tr_h)
        valid_val = ~np.isnan(y_val_h)

        ridge = Ridge(alpha=100.0, random_state=42).fit(X_train[valid_tr], y_tr_h[valid_tr])
        p_ridge = ridge.predict(X_val)

        m_ridge = compute_absolute_price_metrics(y_val_h[valid_val], p_ridge[valid_val], val_close[valid_val])
        records.append({
            "approach": "Approach A: Independent Ridge",
            "horizon": f"{h}d",
            **m_ridge,
        })

    # 2. Multi-Task Neural Network results
    df_mt = multitask_results["metrics_table"].copy()
    for _, r in df_mt.iterrows():
        records.append({
            "approach": "Approach B: Multi-Task Neural Net",
            "horizon": r["horizon"],
            "return_mae": r["return_mae"],
            "return_rmse": r["return_rmse"],
            "inr_price_mae": r["inr_price_mae"],
            "normalized_price_error_pct": r["normalized_price_error_pct"],
            "directional_accuracy_pct": r["directional_accuracy_pct"],
        })

    return pd.DataFrame(records)


def verify_hierarchical_horizon_consistency(
    predictions_by_horizon: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    """
    Evaluates whether multi-horizon forecasts exhibit temporal consistency:
    e.g., standard deviation and range should expand with horizon H.
    """
    std_by_horizon = {}
    mae_implied_by_horizon = {}
    for h in HORIZONS:
        preds = predictions_by_horizon[h]
        clean_preds = preds[~np.isnan(preds)]
        std_by_horizon[f"{h}d"] = round(float(np.std(clean_preds)), 4)
        mae_implied_by_horizon[f"{h}d"] = round(float(np.mean(np.abs(clean_preds))), 4)

    is_monotonic_std = bool(all(
        std_by_horizon[f"{HORIZONS[i]}d"] <= std_by_horizon[f"{HORIZONS[i+1]}d"]
        for i in range(len(HORIZONS) - 1)
    ))

    return {
        "prediction_std_by_horizon": std_by_horizon,
        "mean_abs_prediction_by_horizon": mae_implied_by_horizon,
        "monotonic_dispersion_expansion": is_monotonic_std,
    }
