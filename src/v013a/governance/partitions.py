"""StockReason v0.13A — Partition Governance & Horizon-Specific Label Purging.

Section H Requirements:
1. Strict label-end realization purging across all 5 horizons (1, 5, 21, 63, 126).
   Enforces: label_realization_date < next_partition_start_date.
2. Identifies and purges observations whose future labels leak across split boundaries.
3. Generates artifacts/v013a/partition_governance.json recording complete audit metrics.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("v013a_partitions")

HORIZONS = [1, 5, 21, 63, 126]


def compute_trading_calendar_sessions(
    df: pd.DataFrame,
    date_col: str = "date",
) -> List[pd.Timestamp]:
    """Extracts unique sorted exchange trading dates from the dataset."""
    dates = pd.to_datetime(df[date_col].unique())
    return sorted(dates)


def create_purged_horizon_partitions(
    df: pd.DataFrame,
    horizon: int = 21,
    train_end_date: str = "2022-12-31",
    calib_start_date: str = "2023-01-01",
    calib_end_date: str = "2023-12-31",
    eval_start_date: str = "2024-01-01",
    eval_end_date: str = "2026-11-23",
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> Dict[str, Any]:
    """
    Creates strictly purged walk-forward partitions guaranteeing zero label realization leakage.

    For horizon H, an observation on date t has its target realized on session t + H.
    If session(t + H) >= calib_start_date, it is strictly purged from training.
    If session(t + H) >= eval_start_date, it is strictly purged from calibration.
    """
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])

    trading_dates = compute_trading_calendar_sessions(df, date_col=date_col)
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    n_sessions = len(trading_dates)

    t_end = pd.to_datetime(train_end_date)
    c_start = pd.to_datetime(calib_start_date)
    c_end = pd.to_datetime(calib_end_date)
    e_start = pd.to_datetime(eval_start_date)
    e_end = pd.to_datetime(eval_end_date)

    # Vectorized map of target realization date
    future_map = {
        trading_dates[i]: trading_dates[i + horizon]
        for i in range(len(trading_dates) - horizon)
    }
    df["target_realization_date"] = df[date_col].map(future_map)

    # Raw partition masks
    raw_train_mask = df[date_col] <= t_end
    raw_calib_mask = (df[date_col] >= c_start) & (df[date_col] <= c_end)
    raw_eval_mask = (df[date_col] >= e_start) & (df[date_col] <= e_end)

    pre_purge_train_count = int(raw_train_mask.sum())
    pre_purge_calib_count = int(raw_calib_mask.sum())

    # Purge enforcement:
    # Training: target_realization_date must be strictly < c_start
    purged_train_mask = raw_train_mask & (df["target_realization_date"] < c_start)
    purged_train_count = pre_purge_train_count - int(purged_train_mask.sum())

    # Calibration: target_realization_date must be strictly < e_start
    purged_calib_mask = raw_calib_mask & (df["target_realization_date"] < e_start)
    purged_calib_count = pre_purge_calib_count - int(purged_calib_mask.sum())

    # Evaluation
    eval_mask = raw_eval_mask & df["target_realization_date"].notna()

    train_df = df[purged_train_mask].copy()
    calib_df = df[purged_calib_mask].copy()
    eval_df = df[eval_mask].copy()

    max_train_realization = train_df["target_realization_date"].max() if len(train_df) > 0 else pd.NaT
    max_calib_realization = calib_df["target_realization_date"].max() if len(calib_df) > 0 else pd.NaT

    return {
        "horizon": horizon,
        "train_df": train_df,
        "calib_df": calib_df,
        "eval_df": eval_df,
        "pre_purge_train_count": pre_purge_train_count,
        "purged_train_count": purged_train_count,
        "retained_train_count": len(train_df),
        "pre_purge_calib_count": pre_purge_calib_count,
        "purged_calib_count": purged_calib_count,
        "retained_calib_count": len(calib_df),
        "retained_eval_count": len(eval_df),
        "max_train_realization_date": str(max_train_realization.date()) if pd.notna(max_train_realization) else None,
        "calib_start_date": str(c_start.date()),
        "max_calib_realization_date": str(max_calib_realization.date()) if pd.notna(max_calib_realization) else None,
        "eval_start_date": str(e_start.date()),
        "train_leakage_prevented": bool(pd.isna(max_train_realization) or (max_train_realization < c_start)),
        "calib_leakage_prevented": bool(pd.isna(max_calib_realization) or (max_calib_realization < e_start)),
    }


def build_partition_governance_registry(
    df: pd.DataFrame,
    horizons: List[int] = HORIZONS,
    output_json: str = "artifacts/v013a/partition_governance.json",
) -> Dict[str, Any]:
    """Generates complete multi-horizon partition governance registry."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    registry = {
        "governance_version": "0.13.0",
        "description": "StockReason v0.13A Strict Chronological Partition Governance Registry",
        "partitions": {
            "training": {
                "intended_purpose": "Historical parameter fitting only",
                "cutoff_rule": "date <= 2022-12-31 AND target_realization < 2023-01-01",
                "classification": "DEVELOPMENT_TRAIN",
            },
            "calibration": {
                "intended_purpose": "Conformal score calibration & downside threshold validation",
                "cutoff_rule": "2023-01-01 <= date <= 2023-12-31 AND target_realization < 2024-01-01",
                "classification": "DEVELOPMENT_CALIBRATION",
            },
            "historical_diagnostic_evaluation": {
                "intended_purpose": "Controlled out-of-sample diagnostic evaluation (NOT pristine holdout)",
                "cutoff_rule": "2024-01-01 <= date <= 2026-11-23",
                "classification": "HISTORICAL_DIAGNOSTIC_EVALUATION",
            },
            "prospective": {
                "intended_purpose": "True forward forecasting created before realization",
                "cutoff_rule": "date > 2026-11-23",
                "classification": "GENUINE_PROSPECTIVE",
            },
        },
        "horizons_audit": {},
    }

    for h in horizons:
        h_res = create_purged_horizon_partitions(df, horizon=h)
        registry["horizons_audit"][f"{h}d"] = {
            "horizon_days": h,
            "pre_purge_train_count": h_res["pre_purge_train_count"],
            "purged_train_count": h_res["purged_train_count"],
            "retained_train_count": h_res["retained_train_count"],
            "pre_purge_calib_count": h_res["pre_purge_calib_count"],
            "purged_calib_count": h_res["purged_calib_count"],
            "retained_calib_count": h_res["retained_calib_count"],
            "retained_eval_count": h_res["retained_eval_count"],
            "max_train_realization_date": h_res["max_train_realization_date"],
            "calib_start_date": h_res["calib_start_date"],
            "max_calib_realization_date": h_res["max_calib_realization_date"],
            "eval_start_date": h_res["eval_start_date"],
            "train_leakage_prevented": h_res["train_leakage_prevented"],
            "calib_leakage_prevented": h_res["calib_leakage_prevented"],
            "audit_status": "PASS" if (h_res["train_leakage_prevented"] and h_res["calib_leakage_prevented"]) else "FAIL",
        }
        logger.info(
            "Horizon %3dd Partition Audit: Purged %d train rows, %d calib rows | Status: %s",
            h, h_res["purged_train_count"], h_res["purged_calib_count"],
            registry["horizons_audit"][f"{h}d"]["audit_status"],
        )

    with open(output_json, "w") as f:
        json.dump(registry, f, indent=2)

    return registry
