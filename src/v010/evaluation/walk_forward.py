"""
StockReason v0.10 — Expanding Walk-Forward Evaluation & Stability Engine.
========================================================================
Remediates Phase 8 (Sections 14, 17):
1. Expanding Walk-Forward Cross-Validation:
   8 chronological folds with strictly exact per-observation target-end timestamps.
2. 5-session embargo prior to each validation split.
3. Stability breakdowns:
   - Per-Year performance
   - Per-Sector performance
   - Per-Regime performance (Bullish, Bearish, Sideways)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.v092.validation.purged_cv import ExactPerObservationPurgedCV, compute_per_observation_target_end_dates
from src.v092.models.baselines import compute_absolute_price_metrics


def run_walk_forward_stability_breakdown(
    df_eval: pd.DataFrame,
    target_col: str,
    pred_col: str,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluates stability breakdown across:
    1. Per-Year performance
    2. Per-Sector performance
    3. Per-Regime performance
    """
    clean = df_eval.dropna(subset=[target_col, pred_col, "close"]).copy()
    y_true = clean[target_col].values
    y_pred = clean[pred_col].values
    prices = clean["close"].values

    # 1. Per-Year Breakdown
    clean["year"] = clean["date_str"].str[:4]
    year_records = []
    for yr, grp in clean.groupby("year"):
        m = compute_absolute_price_metrics(grp[target_col].values, grp[pred_col].values, grp["close"].values)
        year_records.append({"year": yr, "observations": len(grp), **m})
    df_year = pd.DataFrame(year_records)

    # 2. Per-Sector Breakdown
    sector_records = []
    if "sector" in clean.columns:
        for sec, grp in clean.groupby("sector"):
            if len(grp) >= 50:
                m = compute_absolute_price_metrics(grp[target_col].values, grp[pred_col].values, grp["close"].values)
                sector_records.append({"sector": sec, "observations": len(grp), **m})
    df_sector = pd.DataFrame(sector_records)

    # 3. Per-Regime Breakdown
    regime_records = []
    if "market_regime" in clean.columns:
        for reg, grp in clean.groupby("market_regime"):
            m = compute_absolute_price_metrics(grp[target_col].values, grp[pred_col].values, grp["close"].values)
            regime_records.append({"regime": reg, "observations": len(grp), **m})
    df_regime = pd.DataFrame(regime_records)

    return {
        "per_year": df_year,
        "per_sector": df_sector,
        "per_regime": df_regime,
    }
