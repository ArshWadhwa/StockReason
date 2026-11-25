"""StockReason v0.13A — Date-Block Paired Bootstrap & Statistical Significance Engine.

Section I Requirements:
1. Paired chronological date-block bootstrap:
   - Eliminates cross-sectional bias by blocking on calendar trading dates.
   - For matched observation (i, t): Loss_Diff(i, t) = Candidate_Loss(i, t) - Baseline_Loss(i, t)
   - Daily cross-sectional aggregation: Date_Loss_Diff(t) = mean_i(Loss_Diff(i, t))
2. Moving-block bootstrap across block lengths: 1, 5, 10, 21, 42, 63 trading sessions.
3. 2,000 bootstrap resamples for deterministic, reproducible 95% confidence intervals.
4. Formal statistical interpretation:
   - Entire CI < 0: Evidence favoring candidate
   - Entire CI > 0: Evidence favoring baseline
   - CI spans 0: Improvement or degradation not statistically established at 95% level.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("v013a_bootstrap")


def run_paired_date_block_bootstrap(
    daily_diff_series: pd.Series,
    block_length: int = 21,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a moving date-block bootstrap on daily loss differences.
    daily_diff_series: pd.Series indexed by chronological trading date containing Date_Loss_Diff(t).
    """
    diffs = daily_diff_series.dropna().values
    n_dates = len(diffs)

    if n_dates < block_length:
        return {
            "block_length": block_length,
            "n_dates": n_dates,
            "mean_loss_diff": float(np.mean(diffs)) if n_dates > 0 else 0.0,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "verdict": "INSUFFICIENT_DATES",
        }

    rng = np.random.RandomState(random_seed)
    n_blocks_needed = int(np.ceil(n_dates / block_length))
    max_start_idx = n_dates - block_length + 1

    # Pre-extract all candidate blocks of length block_length
    candidate_blocks = [diffs[i : i + block_length] for i in range(max_start_idx)]
    n_candidates = len(candidate_blocks)

    resampled_means = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sampled_block_indices = rng.randint(0, n_candidates, size=n_blocks_needed)
        resampled_series = np.concatenate([candidate_blocks[idx] for idx in sampled_block_indices])[:n_dates]
        resampled_means[b] = np.mean(resampled_series)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(resampled_means, (alpha / 2.0) * 100.0))
    ci_upper = float(np.percentile(resampled_means, (1.0 - alpha / 2.0) * 100.0))
    mean_diff = float(np.mean(diffs))

    if ci_upper < 0.0:
        verdict = "SIGNIFICANT_CANDIDATE_ADVANTAGE"
    elif ci_lower > 0.0:
        verdict = "SIGNIFICANT_BASELINE_ADVANTAGE"
    else:
        verdict = "NO_STATISTICAL_DIFFERENCE_ESTABLISHED"

    return {
        "block_length_days": block_length,
        "n_trading_dates": n_dates,
        "n_resamples": n_resamples,
        "mean_loss_diff": round(mean_diff, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "verdict": verdict,
    }


def compute_date_block_bootstrap_sensitivity(
    df: pd.DataFrame,
    candidate_loss_col: str,
    baseline_loss_col: str,
    date_col: str = "date",
    block_lengths: List[int] = [1, 5, 10, 21, 42, 63],
    n_resamples: int = 2000,
) -> Dict[str, Any]:
    """
    Computes date-block bootstrap sensitivity across multiple block lengths
    for paired candidate vs baseline loss comparison.
    """
    df = df[[date_col, candidate_loss_col, baseline_loss_col]].dropna().copy()
    df["loss_diff"] = df[candidate_loss_col] - df[baseline_loss_col]
    
    # Date-level cross-sectional mean
    daily_diff = df.groupby(date_col)["loss_diff"].mean().sort_index()

    sensitivity_results = {}
    for blk in block_lengths:
        res = run_paired_date_block_bootstrap(
            daily_diff,
            block_length=blk,
            n_resamples=n_resamples,
            confidence_level=0.95,
        )
        sensitivity_results[f"block_{blk}d"] = res

    return {
        "candidate_col": candidate_loss_col,
        "baseline_col": baseline_loss_col,
        "total_observations": len(df),
        "total_trading_dates": len(daily_diff),
        "blocks": sensitivity_results,
    }
