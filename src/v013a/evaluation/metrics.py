"""StockReason v0.13A — Exact Forecasting Metrics & Mathematical Contracts.

Section D Requirements:
1. Exact Return MAE & Return RMSE.
2. Exact INR Price MAE & Price RMSE.
3. True Price MAPE (denominator is actual nominal future price, NOT today's reference price).
4. MedAPE (median percentage error).
5. Relative loss improvement: 100 * (Baseline - Candidate) / Baseline.
6. Directional classification & Abstention:
   UP   : return > 0
   DOWN : return < 0
   FLAT : return == 0
   Zero-return forecasts are FLAT abstentions; conditional vs unconditional hit rates.
7. Mathematical reference fixtures (Case A & Case B).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

logger = logging.getLogger("v013a_metrics")


def compute_return_mae(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    """Computes exact return MAE: mean(abs(R_actual - R_predicted))."""
    diff = np.abs(y_true_ret - y_pred_ret)
    return float(np.mean(diff))


def compute_return_rmse(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    """Computes exact return RMSE: sqrt(mean((R_actual - R_predicted)^2))."""
    diff_sq = (y_true_ret - y_pred_ret) ** 2
    return float(np.sqrt(np.mean(diff_sq)))


def compute_price_mae(p_true_future: np.ndarray, p_pred_future: np.ndarray) -> float:
    """Computes exact INR price MAE: mean(abs(P_actual_future - P_predicted_future))."""
    diff = np.abs(p_true_future - p_pred_future)
    return float(np.mean(diff))


def compute_price_rmse(p_true_future: np.ndarray, p_pred_future: np.ndarray) -> float:
    """Computes exact INR price RMSE: sqrt(mean((P_actual_future - P_predicted_future)^2))."""
    diff_sq = (p_true_future - p_pred_future) ** 2
    return float(np.sqrt(np.mean(diff_sq)))


def compute_true_price_mape(
    p_true_future: np.ndarray,
    p_pred_future: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Computes exact true price MAPE:
    MAPE = 100 * mean(abs(P_actual_future - P_predicted_future) / abs(P_actual_future)).
    
    The denominator MUST be the actual future nominal price.
    Guards against actual future price <= 0, NaN, or Inf.
    """
    valid = (p_true_future > 0) & np.isfinite(p_true_future) & np.isfinite(p_pred_future)
    if np.sum(valid) == 0:
        return float("nan")
    
    y_t = p_true_future[valid]
    y_p = p_pred_future[valid]
    ape = np.abs(y_t - y_p) / np.abs(y_t)
    return float(np.mean(ape) * 100.0)


def compute_true_price_medape(
    p_true_future: np.ndarray,
    p_pred_future: np.ndarray,
) -> float:
    """
    Computes exact true price MedAPE:
    MedAPE = 100 * median(abs(P_actual_future - P_predicted_future) / abs(P_actual_future)).
    """
    valid = (p_true_future > 0) & np.isfinite(p_true_future) & np.isfinite(p_pred_future)
    if np.sum(valid) == 0:
        return float("nan")
    
    y_t = p_true_future[valid]
    y_p = p_pred_future[valid]
    ape = np.abs(y_t - y_p) / np.abs(y_t)
    return float(np.median(ape) * 100.0)


def compute_relative_loss_improvement(baseline_loss: float, candidate_loss: float) -> float:
    """Computes relative percentage improvement: 100 * (Baseline - Candidate) / Baseline."""
    if baseline_loss <= 0:
        return 0.0
    return float(100.0 * (baseline_loss - candidate_loss) / baseline_loss)


def compute_directional_metrics_with_abstention(
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes rigorous directional classification metrics handling zero-return abstentions.

    Classes:
    UP   : return > tolerance
    DOWN : return < -tolerance
    FLAT : abs(return) <= tolerance

    A zero-return prediction (e.g. No-Change baseline) is FLAT.
    It receives NO credit for predicting UP or DOWN moves.
    """
    valid = np.isfinite(y_true_ret) & np.isfinite(y_pred_ret)
    y_t = y_true_ret[valid]
    y_p = y_pred_ret[valid]
    n_total = len(y_t)

    if n_total == 0:
        return {}

    # Discretize true
    true_up = y_t > tolerance
    true_down = y_t < -tolerance
    true_flat = np.abs(y_t) <= tolerance

    # Discretize pred
    pred_up = y_p > tolerance
    pred_down = y_p < -tolerance
    pred_flat = np.abs(y_p) <= tolerance

    n_pred_up = int(np.sum(pred_up))
    n_pred_down = int(np.sum(pred_down))
    n_pred_flat = int(np.sum(pred_flat))

    n_true_up = int(np.sum(true_up))
    n_true_down = int(np.sum(true_down))
    n_true_flat = int(np.sum(true_flat))

    coverage = float((n_pred_up + n_pred_down) / n_total)
    abstention_rate = float(n_pred_flat / n_total)

    # Unconditional hit rate:
    # Among all non-flat actual outcomes, abstentions count as misses.
    actual_directional_mask = ~true_flat
    n_actual_dir = int(np.sum(actual_directional_mask))
    if n_actual_dir > 0:
        unconditional_hits = int(np.sum((pred_up & true_up) | (pred_down & true_down)))
        unconditional_hit_rate = float(unconditional_hits / n_actual_dir)
    else:
        unconditional_hit_rate = 0.0

    # Conditional metrics: evaluate only where both true and pred are non-flat
    eval_mask = actual_directional_mask & (~pred_flat)
    n_conditional = int(np.sum(eval_mask))

    if n_conditional > 0:
        y_t_cond = (y_t[eval_mask] > tolerance).astype(int)  # 1 for UP, 0 for DOWN
        y_p_cond = (y_p[eval_mask] > tolerance).astype(int)

        cond_acc = float(np.mean(y_t_cond == y_p_cond))
        
        # Confusion matrix: [ [TN, FP], [FN, TP] ] where 1=UP, 0=DOWN
        cm = confusion_matrix(y_t_cond, y_p_cond, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        up_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        down_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        specificity = down_recall
        balanced_acc = float(0.5 * (up_recall + down_recall)) if np.isfinite(up_recall) and np.isfinite(down_recall) else float("nan")

        mcc = float(matthews_corrcoef(y_t_cond, y_p_cond)) if (len(np.unique(y_t_cond)) > 1 and len(np.unique(y_p_cond)) > 1) else 0.0
    else:
        cond_acc = None
        up_recall = None
        down_recall = None
        specificity = None
        balanced_acc = None
        mcc = None
        cm = np.zeros((2, 2), dtype=int)

    return {
        "total_observations": n_total,
        "n_pred_up": n_pred_up,
        "n_pred_down": n_pred_down,
        "n_pred_flat": n_pred_flat,
        "n_true_up": n_true_up,
        "n_true_down": n_true_down,
        "n_true_flat": n_true_flat,
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "conditional_observations": n_conditional,
        "conditional_directional_accuracy": round(cond_acc, 4) if cond_acc is not None else None,
        "unconditional_directional_hit_rate": round(unconditional_hit_rate, 4),
        "balanced_accuracy": round(balanced_acc, 4) if balanced_acc is not None else None,
        "mcc": round(mcc, 4) if mcc is not None else None,
        "up_recall": round(up_recall, 4) if up_recall is not None else None,
        "down_recall": round(down_recall, 4) if down_recall is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
    }


def verify_mathematical_reference_fixtures() -> Dict[str, bool]:
    """
    Verifies exact behavior against mathematical reference fixtures:
    Case A: Ref=100, Actual=120, Pred=110 -> Return Err=0.10, Price Err=10, MAPE=8.333333%
    Case B: Ref=100, Actual=80, Pred=90 -> Return Err=0.10, Price Err=10, MAPE=12.5%
    """
    # Case A
    p_ref_a = 100.0
    p_act_a = 120.0
    p_pred_a = 110.0
    r_act_a = (p_act_a / p_ref_a) - 1.0  # 0.20
    r_pred_a = (p_pred_a / p_ref_a) - 1.0  # 0.10

    ret_err_a = compute_return_mae(np.array([r_act_a]), np.array([r_pred_a]))
    price_err_a = compute_price_mae(np.array([p_act_a]), np.array([p_pred_a]))
    mape_a = compute_true_price_mape(np.array([p_act_a]), np.array([p_pred_a]))

    case_a_pass = (
        np.isclose(ret_err_a, 0.10)
        and np.isclose(price_err_a, 10.0)
        and np.isclose(mape_a, 8.333333333, atol=1e-5)
    )

    # Case B
    p_ref_b = 100.0
    p_act_b = 80.0
    p_pred_b = 90.0
    r_act_b = (p_act_b / p_ref_b) - 1.0  # -0.20
    r_pred_b = (p_pred_b / p_ref_b) - 1.0  # -0.10

    ret_err_b = compute_return_mae(np.array([r_act_b]), np.array([r_pred_b]))
    price_err_b = compute_price_mae(np.array([p_act_b]), np.array([p_pred_b]))
    mape_b = compute_true_price_mape(np.array([p_act_b]), np.array([p_pred_b]))

    case_b_pass = (
        np.isclose(ret_err_b, 0.10)
        and np.isclose(price_err_b, 10.0)
        and np.isclose(mape_b, 12.5, atol=1e-5)
    )

    return {
        "case_a_pass": bool(case_a_pass),
        "case_b_pass": bool(case_b_pass),
        "case_a_mape": mape_a,
        "case_b_mape": mape_b,
        "distinct_mapes_for_same_return_error": bool(not np.isclose(mape_a, mape_b)),
    }
