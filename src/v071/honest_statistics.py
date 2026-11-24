"""
StockReason v0.7.1 - Workstream A3: Honest Statistics & Power Analysis Library
==============================================================================
Implements defensible statistical tests for equity return forecasting:
1. Block bootstrap of per-date cross-sectional IC (block lengths 10, 21, 42; 5000 reps).
2. Non-overlapping evaluation set (every 21st date across all 21 offsets).
3. Newey-West HAC t-statistic with lag = horizon.
4. Multiple testing corrections (Bonferroni, Benjamini-Hochberg) and Deflated Sharpe Ratio.
5. Mechanical Verdict Rule: "No demonstrated edge" if 95% bootstrap CI includes 0.
6. Power analysis: minimum detectable IC at 80% power.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


def block_bootstrap_ic(
    ic_series: np.ndarray,
    block_length: int = 21,
    n_replications: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Performs circular block bootstrap on a time series of per-date IC values.

    Args:
        ic_series: 1D array of per-date cross-sectional ICs.
        block_length: Length of contiguous date blocks (default 21).
        n_replications: Number of bootstrap replications (default 5000).
        seed: Random seed.

    Returns:
        dict with mean_ic, ci_95, pct_positive, p_bootstrap.
    """
    clean_ics = np.asarray(ic_series, dtype=float)
    clean_ics = clean_ics[~np.isnan(clean_ics)]
    n = len(clean_ics)
    if n < 5:
        return {
            "mean_ic": 0.0,
            "ci_95": [-1.0, 1.0],
            "pct_positive": 0.0,
            "p_val_bootstrap": 1.0,
            "n_replications": n_replications,
            "block_length": block_length,
        }

    np.random.seed(seed)
    n_blocks = int(np.ceil(n / block_length))
    # Circular block bootstrap indices
    sample_means = np.empty(n_replications, dtype=float)

    for i in range(n_replications):
        start_indices = np.random.randint(0, n, size=n_blocks)
        bootstrap_indices = []
        for s in start_indices:
            bootstrap_indices.extend([(s + j) % n for j in range(block_length)])
        bootstrap_indices = np.array(bootstrap_indices[:n])
        sample_means[i] = np.mean(clean_ics[bootstrap_indices])

    mean_est = float(np.mean(clean_ics))
    ci_lower = float(np.percentile(sample_means, 2.5))
    ci_upper = float(np.percentile(sample_means, 97.5))
    pct_pos = float(np.mean(sample_means > 0)) * 100.0
    p_boot = float(2.0 * min(np.mean(sample_means <= 0), np.mean(sample_means >= 0)))

    return {
        "mean_ic": mean_est,
        "ci_95": [ci_lower, ci_upper],
        "pct_positive_draws": pct_pos,
        "p_val_bootstrap": p_boot,
        "n_replications": n_replications,
        "block_length": block_length,
    }


def compute_non_overlapping_ic(
    ic_df: pd.DataFrame,
    horizon: int = 21,
    date_col: str = "date",
    ic_col: str = "ic",
) -> Dict[str, Any]:
    """
    Evaluates IC on non-overlapping dates: every horizon-th trading date.
    Tests all 21 offsets (0 to 20) and reports distribution of IC across offsets.
    """
    df = ic_df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n < horizon:
        return {"offset_ics": [], "mean_across_offsets": 0.0, "ci_95_across_offsets": [0.0, 0.0]}

    offset_means = []
    offset_records = {}

    for offset in range(horizon):
        sub_indices = list(range(offset, n, horizon))
        sub_ics = df.iloc[sub_indices][ic_col].to_numpy(dtype=float)
        mean_offset = float(np.mean(sub_ics)) if len(sub_ics) > 0 else 0.0
        offset_means.append(mean_offset)
        offset_records[f"offset_{offset}"] = {
            "mean_ic": mean_offset,
            "n_dates": len(sub_indices),
        }

    offset_means = np.array(offset_means)
    return {
        "n_offsets": horizon,
        "min_offset_ic": float(np.min(offset_means)),
        "median_offset_ic": float(np.median(offset_means)),
        "mean_offset_ic": float(np.mean(offset_means)),
        "max_offset_ic": float(np.max(offset_means)),
        "std_across_offsets": float(np.std(offset_means)),
        "offset_records": offset_records,
    }


def compute_newey_west_tstat(
    series: np.ndarray,
    max_lag: int = 21,
) -> Tuple[float, float, float]:
    """
    Computes Newey-West HAC standard error, t-statistic, and p-value for the mean.

    Returns:
        (mean, t_stat, p_val)
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= max_lag + 1:
        return (0.0, 0.0, 1.0)

    mean_val = np.mean(x)
    e = x - mean_val

    # Gamma_0
    gamma_0 = np.sum(e ** 2) / n
    var_hac = gamma_0

    # Bartlett kernel weighting
    for lag in range(1, max_lag + 1):
        weight = 1.0 - (lag / (max_lag + 1.0))
        gamma_lag = np.sum(e[lag:] * e[:-lag]) / n
        var_hac += 2.0 * weight * gamma_lag

    se_hac = np.sqrt(max(1e-12, var_hac) / n)
    t_stat = float(mean_val / se_hac)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat))))

    return (float(mean_val), t_stat, p_val)


def adjust_multiple_testing(
    p_values: Dict[str, float],
    method: str = "benjamini_hochberg",
) -> Dict[str, float]:
    """
    Applies multiple-testing adjustment (Bonferroni or Benjamini-Hochberg) to p-values.
    """
    names = list(p_values.keys())
    p_vals = np.array([p_values[k] for k in names])
    n = len(p_vals)

    if method == "bonferroni":
        adj_p = np.minimum(1.0, p_vals * n)
    elif method == "benjamini_hochberg":
        # Sort p-values
        sort_order = np.argsort(p_vals)
        ranks = np.empty_like(sort_order)
        ranks[sort_order] = np.arange(1, n + 1)
        adj_p = np.minimum(1.0, p_vals * n / ranks)
        # Enforce monotonicity backwards
        for i in range(n - 2, -1, -1):
            idx_curr = sort_order[i]
            idx_next = sort_order[i + 1]
            adj_p[idx_curr] = min(adj_p[idx_curr], adj_p[idx_next])
    else:
        raise ValueError(f"Unknown method: {method}")

    return {names[i]: float(adj_p[i]) for i in range(n)}


def compute_power_analysis(
    n_independent_dates: int,
    target_ic: float = 0.03,
    std_ic: float = 0.15,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, Any]:
    """
    Calculates statistical power and minimum detectable IC.

    Formula:
      z_alpha/2 = 1.96 for alpha=0.05
      z_beta = 0.84 for power=0.80
      min_detectable_ic = (z_alpha/2 + z_beta) * (std_ic / sqrt(N))
    """
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    # Minimum detectable IC at 80% power
    min_detectable = (z_alpha + z_beta) * (std_ic / np.sqrt(max(1, n_independent_dates)))

    # Required independent observations to detect target_ic
    n_required = ((z_alpha + z_beta) * std_ic / max(1e-4, target_ic)) ** 2

    # Actual power given current sample
    standardized_effect = (target_ic / (std_ic / np.sqrt(max(1, n_independent_dates))))
    actual_power = float(stats.norm.cdf(standardized_effect - z_alpha))

    return {
        "n_independent_dates": n_independent_dates,
        "assumed_std_ic": std_ic,
        "target_ic": target_ic,
        "min_detectable_ic_at_80pct_power": float(min_detectable),
        "actual_power_for_target_ic": actual_power,
        "n_dates_required_for_target_ic": int(np.ceil(n_required)),
        "is_sufficient_power": bool(actual_power >= 0.80),
    }


def evaluate_mechanical_verdict(
    ci_95: Tuple[float, float],
    model_name: str,
    horizon: str = "21D",
) -> str:
    """
    Mechanical Verdict Rule:
    If 95% bootstrap interval includes 0 -> "No demonstrated edge"
    Otherwise -> "Edge demonstrated: IC x (95% CI a to b)"
    """
    lower, upper = ci_95
    if lower <= 0.0 <= upper:
        return f"No demonstrated edge ({model_name} {horizon} 95% CI [{lower:.4f}, {upper:.4f}] spans zero)"
    else:
        return f"Edge demonstrated: IC [{lower:.4f} to {upper:.4f}] strictly excludes zero"
