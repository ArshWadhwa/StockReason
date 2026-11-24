"""
StockReason v0.11 — Confidence-Based Selective Prediction Engine.
================================================================
Remediates Phase 4 (Section 27):
1. Principle:
   Financial markets exhibit low signal-to-noise ratios. The system should abstain
   from directional or trading claims when signals are weak, evaluating selective
   prediction at specified coverage levels: 100%, 75%, 50%, 30%, 20%, 10%.
2. Confidence Score Formulation:
   confidence_score = |pred_return| / (normalized_interval_width + 1e-4)
   where normalized_interval_width = (Q_90 - Q_10) / close.
3. Evaluates on each coverage subset:
   - Eligible forecast count
   - Directional Accuracy & Balanced Accuracy
   - Matthews Correlation Coefficient (MCC)
   - Return MAE & INR Price MAE & MAPE
   - Matched Unconditional Historical Mean Baseline
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.v011.evaluation.baselines import compute_price_metrics


def evaluate_selective_prediction_v011(
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
    close_prices: np.ndarray,
    interval_widths: np.ndarray,
    baseline_pred_ret: np.ndarray,
    coverage_levels: List[float] = [1.0, 0.75, 0.50, 0.30, 0.20, 0.10],
) -> pd.DataFrame:
    """
    Evaluates model performance as a function of confidence-based coverage threshold.
    """
    # Normalized interval width
    norm_width = np.maximum(interval_widths / np.maximum(close_prices, 1.0), 0.01)
    
    # Confidence score: higher signal magnitude relative to uncertainty width
    confidence_scores = np.abs(y_pred_ret) / norm_width

    results = []

    for cov in coverage_levels:
        if cov >= 0.999:
            selected_idx = np.ones(len(y_true_ret), dtype=bool)
        else:
            thresh = np.quantile(confidence_scores, 1.0 - cov)
            selected_idx = confidence_scores >= thresh

        n_selected = int(selected_idx.sum())
        if n_selected < min(4, len(y_true_ret)):
            continue

        y_true_sub = y_true_ret[selected_idx]
        y_pred_sub = y_pred_ret[selected_idx]
        close_sub = close_prices[selected_idx]
        base_sub = baseline_pred_ret[selected_idx]

        m_model = compute_price_metrics(y_true_sub, y_pred_sub, close_sub)
        m_base = compute_price_metrics(y_true_sub, base_sub, close_sub)

        # MCC
        y_true_bin = (y_true_sub > 0).astype(int)
        y_pred_bin = (y_pred_sub > 0).astype(int)
        mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin)) if len(np.unique(y_true_bin)) > 1 and len(np.unique(y_pred_bin)) > 1 else 0.0

        results.append({
            "target_coverage_pct": int(cov * 100),
            "actual_coverage_pct": round(n_selected / len(y_true_ret) * 100.0, 1),
            "eligible_forecasts": n_selected,
            "return_mae": m_model["return_mae"],
            "inr_price_mae": m_model["inr_price_mae"],
            "mape_pct": m_model["mape_pct"],
            "directional_acc_pct": m_model["directional_accuracy_pct"],
            "balanced_acc_pct": m_model["balanced_accuracy_pct"],
            "mcc": round(mcc, 4),
            "baseline_return_mae": m_base["return_mae"],
            "baseline_inr_mae": m_base["inr_price_mae"],
            "return_mae_reduction_pct": round((m_base["return_mae"] - m_model["return_mae"]) / m_base["return_mae"] * 100.0, 2),
        })

    return pd.DataFrame(results)
