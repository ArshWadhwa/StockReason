"""
StockReason v0.9.2 — Paired Time-Block Difference Confidence Intervals.
========================================================================
Remediates Item 10:
- Directly compares LightGBM vs Momentum using paired time-block bootstrap tests.
- Evaluates difference on identical dates:
  1. Engine A (Return MAE):
     D_t = MAE_t(LGBM) - MAE_t(Momentum)
     95% Block-Bootstrap CI on mean(D_t).
  2. Engine B (Rank IC):
     Delta_t = RankIC_t(LGBM) - RankIC_t(Momentum)
     95% Block-Bootstrap CI on mean(Delta_t).
- If 95% CI contains 0 => Concludes no statistically significant edge over momentum.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_paired_time_block_bootstrap_ci(
    diff_series: np.ndarray,
    block_size: int = 21,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, bool]:
    """
    Computes stationary block bootstrap 95% confidence interval on paired difference series.
    Returns: (mean_diff, ci_lower, ci_upper, is_statistically_significant).
    """
    clean_diff = diff_series[~np.isnan(diff_series)]
    n = len(clean_diff)
    if n < block_size * 2:
        m = float(np.mean(clean_diff))
        return m, m, m, False

    np.random.seed(seed)
    n_blocks = int(np.ceil(n / block_size))
    
    boot_means = []
    for _ in range(n_bootstrap):
        start_indices = np.random.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([clean_diff[idx : idx + block_size] for idx in start_indices])[:n]
        boot_means.append(np.mean(sample))

    mean_diff = float(np.mean(clean_diff))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    # Significant if 0 is outside the 95% interval
    is_significant = bool((ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0))
    return mean_diff, ci_lower, ci_upper, is_significant


def compare_lgbm_vs_momentum_paired(
    df_eval: pd.DataFrame,
    pred_lgb_col: str,
    pred_mom_col: str,
    target_ret_col: str,
    target_demeaned_col: str,
    date_col: str = "date_str",
    block_size: int = 21,
) -> Dict[str, Any]:
    """
    Performs paired time-block comparison between LightGBM and Momentum across both engines.
    """
    dates = []
    engine_a_diffs = []
    engine_b_diffs = []

    for d, group in df_eval.groupby(date_col):
        sub = group.dropna(subset=[pred_lgb_col, pred_mom_col, target_ret_col, target_demeaned_col])
        if len(sub) < 10:
            continue

        dates.append(d)
        
        # Engine A: Return MAE difference (MAE_lgb - MAE_mom)
        mae_lgb = np.mean(np.abs(sub[target_ret_col] - sub[pred_lgb_col]))
        mae_mom = np.mean(np.abs(sub[target_ret_col] - sub[pred_mom_col]))
        engine_a_diffs.append(mae_lgb - mae_mom)

        # Engine B: Rank IC difference (RankIC_lgb - RankIC_mom)
        r_lgb, _ = stats.spearmanr(sub[pred_lgb_col], sub[target_demeaned_col])
        r_mom, _ = stats.spearmanr(sub[pred_mom_col], sub[target_demeaned_col])
        if not np.isnan(r_lgb) and not np.isnan(r_mom):
            engine_b_diffs.append(r_lgb - r_mom)
        else:
            engine_b_diffs.append(0.0)

    # Compute bootstrap intervals
    m_a, low_a, high_a, sig_a = compute_paired_time_block_bootstrap_ci(np.array(engine_a_diffs), block_size=block_size)
    m_b, low_b, high_b, sig_b = compute_paired_time_block_bootstrap_ci(np.array(engine_b_diffs), block_size=block_size)

    return {
        "engine_a_return_mae_diff": {
            "mean_diff": round(m_a, 4),
            "ci_95": f"[{low_a:.4f}, {high_a:.4f}]",
            "is_statistically_significant": sig_a,
            "interpretation": (
                "LightGBM has significantly lower return MAE than momentum"
                if (sig_a and m_a < 0)
                else "No statistically significant difference in return MAE over momentum"
            ),
        },
        "engine_b_rank_ic_diff": {
            "mean_diff": round(m_b, 4),
            "ci_95": f"[{low_b:.4f}, {high_b:.4f}]",
            "is_statistically_significant": sig_b,
            "interpretation": (
                "LightGBM has significantly higher Rank IC than momentum"
                if (sig_b and m_b > 0)
                else "No statistically significant edge in Rank IC over momentum (CI includes 0)"
            ),
        },
        "n_evaluated_dates": len(dates),
    }
