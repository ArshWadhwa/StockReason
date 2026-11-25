"""Chronological Partition Governance & Horizon-Specific Label Purging."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.utils.calendar import compute_trading_calendar_sessions
from src.utils.config import PRIMARY_HORIZON


def create_purged_horizon_partitions(
    df: pd.DataFrame,
    horizon: int = PRIMARY_HORIZON,
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

    t_end = pd.to_datetime(train_end_date)
    c_start = pd.to_datetime(calib_start_date)
    c_end = pd.to_datetime(calib_end_date)
    e_start = pd.to_datetime(eval_start_date)
    e_end = pd.to_datetime(eval_end_date)

    future_map = {
        trading_dates[i]: trading_dates[i + horizon]
        for i in range(len(trading_dates) - horizon)
    }
    df["target_realization_date"] = df[date_col].map(future_map)

    raw_train_mask = df[date_col] <= t_end
    raw_calib_mask = (df[date_col] >= c_start) & (df[date_col] <= c_end)
    raw_eval_mask = (df[date_col] >= e_start) & (df[date_col] <= e_end)

    pre_purge_train_count = int(raw_train_mask.sum())
    pre_purge_calib_count = int(raw_calib_mask.sum())

    # Purging enforcement
    purged_train_mask = raw_train_mask & (df["target_realization_date"] < c_start)
    purged_train_count = pre_purge_train_count - int(purged_train_mask.sum())

    purged_calib_mask = raw_calib_mask & (df["target_realization_date"] < e_start)
    purged_calib_count = pre_purge_calib_count - int(purged_calib_mask.sum())

    eval_mask = raw_eval_mask & df["target_realization_date"].notna()

    train_df = df[purged_train_mask].copy()
    calib_df = df[purged_calib_mask].copy()
    eval_df = df[eval_mask].copy()

    return {
        "horizon": horizon,
        "train_df": train_df,
        "calib_df": calib_df,
        "eval_df": eval_df,
        "metrics": {
            "pre_purge_train_rows": pre_purge_train_count,
            "purged_train_rows": purged_train_count,
            "final_train_rows": len(train_df),
            "pre_purge_calib_rows": pre_purge_calib_count,
            "purged_calib_rows": purged_calib_count,
            "final_calib_rows": len(calib_df),
            "eval_rows": len(eval_df),
        }
    }
