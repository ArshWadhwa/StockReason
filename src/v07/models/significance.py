"""
Statistical significance testing for model comparison in StockReason v0.7.

Implements:
- Diebold-Mariano test (Diebold & Mariano 1995; Harvey, Leybourne, & Newbold 1997)
  for comparing predictive accuracy of two forecast models under h-step autocorrelation.
- Convenience functions to compare models against Zero and Ridge baselines.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Union
import numpy as np
from scipy import stats


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    hln_correction: bool = True,
) -> Tuple[float, float]:
    """
    Computes the Diebold-Mariano test statistic and p-value for two competing forecasts.

    Null Hypothesis H0: E[L(e1) - L(e2)] = 0 (equal forecast accuracy).
    - If alternative == 'two-sided': H1: E[d] != 0
    - If alternative == 'less': H1: E[d] < 0 (model 1 has lower loss than model 2)
    - If alternative == 'greater': H1: E[d] > 0 (model 1 has higher loss than model 2)

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred1 : np.ndarray
        Predictions from Model 1 (candidate model).
    y_pred2 : np.ndarray
        Predictions from Model 2 (benchmark baseline model, e.g. Zero or Ridge).
    h : int
        Forecast horizon (used for Bartlett autocovariance kernel lag up to h - 1).
    loss : {'squared', 'absolute'}
        Loss function to evaluate. Default is 'squared'.
    alternative : {'two-sided', 'less', 'greater'}
        Alternative hypothesis. Default is 'two-sided'.
    hln_correction : bool
        Whether to apply Harvey-Leybourne-Newbold small sample correction.

    Returns
    -------
    Tuple[float, float]
        (dm_statistic, p_value)
    """
    y_t = np.asarray(y_true, dtype=float).flatten()
    p1 = np.asarray(y_pred1, dtype=float).flatten()
    p2 = np.asarray(y_pred2, dtype=float).flatten()

    # Filter out NaNs
    valid = ~(np.isnan(y_t) | np.isnan(p1) | np.isnan(p2))
    y_t, p1, p2 = y_t[valid], p1[valid], p2[valid]

    N = len(y_t)
    if N < 3:
        return 0.0, 1.0

    # 1. Compute loss differential d_t
    e1 = y_t - p1
    e2 = y_t - p2

    if loss == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    d_bar = float(np.mean(d))

    # If predictions are identical, statistic is 0 and p-value is 1.0
    if np.allclose(d, 0.0):
        return 0.0, 1.0

    # 2. Compute autocovariances gamma_k up to lag h - 1 with Bartlett spectral kernel
    # gamma_0 = sample variance
    d_centered = d - d_bar
    gamma_0 = float(np.mean(d_centered ** 2))

    max_lag = max(0, h - 1)
    var_d = gamma_0

    for k in range(1, max_lag + 1):
        # Sample autocovariance at lag k
        gamma_k = float(np.sum(d_centered[k:] * d_centered[:-k]) / N)
        weight = 1.0 - (k / h)
        var_d += 2.0 * weight * gamma_k

    if var_d <= 1e-12:
        return 0.0, 1.0

    # Standard DM statistic
    dm_stat = d_bar / np.sqrt(var_d / N)

    # 3. Small-sample correction (Harvey, Leybourne, Newbold 1997)
    if hln_correction and N > 1:
        corr_factor = np.sqrt((N + 1 - 2 * h + (h * (h - 1) / N)) / N)
        dm_stat = dm_stat * corr_factor
        df = max(1, N - 1)
        dist = stats.t(df=df)
    else:
        dist = stats.norm(0, 1)

    # 4. Compute p-value based on alternative
    if alternative == "two-sided":
        p_val = 2.0 * (1.0 - dist.cdf(abs(dm_stat)))
    elif alternative == "less":
        p_val = dist.cdf(dm_stat)
    elif alternative == "greater":
        p_val = 1.0 - dist.cdf(dm_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(dm_stat), float(np.clip(p_val, 0.0, 1.0))


def compare_against_baselines(
    y_true: np.ndarray,
    candidate_preds: Dict[str, np.ndarray],
    zero_preds: np.ndarray,
    ridge_preds: np.ndarray,
    h: int = 5,
    loss: Literal["squared", "absolute"] = "squared",
) -> Dict[str, Dict[str, float]]:
    """
    Runs Diebold-Mariano test for each candidate model against Zero and Ridge baselines.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping model_name -> {
            'dm_stat_vs_zero': float,
            'p_val_vs_zero': float,
            'dm_stat_vs_ridge': float,
            'p_val_vs_ridge': float,
        }
    """
    results: Dict[str, Dict[str, float]] = {}

    for name, pred in candidate_preds.items():
        stat_z, p_z = diebold_mariano_test(y_true, pred, zero_preds, h=h, loss=loss)
        stat_r, p_r = diebold_mariano_test(y_true, pred, ridge_preds, h=h, loss=loss)
        results[name] = {
            "dm_stat_vs_zero": stat_z,
            "p_val_vs_zero": p_z,
            "dm_stat_vs_ridge": stat_r,
            "p_val_vs_ridge": p_r,
        }

    return results
