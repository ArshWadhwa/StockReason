"""
StockReason v0.9.2 — Exact Per-Observation Target-End Timestamp Purging & Walk-Forward CV.
=============================================================================================
Remediates Item 6:
- Replaces estimated calendar-offset or market-wide index purging with ACTUAL
  per-observation target-end timestamps:
  target_end_date(i, t) = calendar date of security i's (t + H)-th trading session.
- Handles missing trading sessions, halts, and suspensions:
  If security i experiences a 15-day trading halt, its H-day target realization
  extends 15 calendar days further into the future than a stock trading continuously.
- Purge condition:
  For every training observation (i, t), we require:
  target_end_date(i, t) < val_start_date - embargo (embargo = 5 trading sessions).
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def compute_per_observation_target_end_dates(
    df: pd.DataFrame,
    horizons: List[int] = [1, 5, 21, 63, 126],
    date_col: str = "date_str",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """
    Computes for every sample (i, t) and horizon H the exact calendar date
    of security i's (t + H)-th traded session.
    """
    df_out = df.copy()
    if date_col not in df_out.columns:
        df_out[date_col] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")

    df_out = df_out.sort_values([symbol_col, date_col])

    for h in horizons:
        col_name = f"target_end_date_{h}d"
        # Shift date backwards within symbol group to find (t + h)-th traded session
        df_out[col_name] = df_out.groupby(symbol_col)[date_col].shift(-h)

    return df_out


class ExactPerObservationPurgedCV:
    def __init__(
        self,
        n_folds: int = 8,
        primary_horizons: List[int] = [5, 21],
        exploratory_horizons: List[int] = [1, 63, 126],
        embargo_sessions: int = 5,
        min_train_sessions: int = 750,
    ):
        self.n_folds = n_folds
        self.primary_horizons = list(primary_horizons)
        self.exploratory_horizons = list(exploratory_horizons)
        self.all_horizons = sorted(list(set(self.primary_horizons + self.exploratory_horizons)))
        self.embargo_sessions = embargo_sessions
        self.min_train_sessions = min_train_sessions

    def generate_folds(self, trading_dates: List[str]) -> List[Dict[str, Any]]:
        """Generates n_folds walk-forward splits from sorted trading dates."""
        dates = sorted(list(set(trading_dates)))
        # Reserve last 126 sessions for forward evaluation
        reserve_len = 126
        dev_dates = dates[:-reserve_len] if len(dates) > reserve_len + self.min_train_sessions else dates
        n_dev = len(dev_dates)
        eval_window = (n_dev - self.min_train_sessions) // self.n_folds

        folds = []
        for k in range(self.n_folds):
            train_end_idx = self.min_train_sessions + k * eval_window
            val_start_idx = train_end_idx
            val_end_idx = min(train_end_idx + eval_window, n_dev)

            train_sessions = dev_dates[:train_end_idx]
            val_sessions = dev_dates[val_start_idx:val_end_idx]

            folds.append({
                "fold": k + 1,
                "train_start_date": train_sessions[0],
                "train_end_date": train_sessions[-1],
                "val_start_date": val_sessions[0],
                "val_end_date": val_sessions[-1],
                "n_train_sessions": len(train_sessions),
                "n_val_sessions": len(val_sessions),
                "train_sessions": train_sessions,
                "val_sessions": val_sessions,
            })
        return folds

    def purge_train_by_actual_target_end(
        self,
        df_with_target_ends: pd.DataFrame,
        train_sessions: List[str],
        val_start_date: str,
        horizon: int,
        all_market_dates: List[str],
        date_col: str = "date_str",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filters training observations using each observation's ACTUAL target_end_date.
        Requires: target_end_date(i, t) < val_start_date - embargo_sessions in market calendar.
        """
        target_end_col = f"target_end_date_{horizon}d"
        if target_end_col not in df_with_target_ends.columns:
            raise KeyError(f"Column {target_end_col} missing. Call compute_per_observation_target_end_dates first.")

        date_to_idx = {d: i for i, d in enumerate(all_market_dates)}
        val_start_idx = date_to_idx.get(val_start_date)
        if val_start_idx is None:
            val_start_idx = min([i for i, d in enumerate(all_market_dates) if d >= val_start_date], default=len(all_market_dates))

        # Embargo boundary date
        embargo_cutoff_idx = max(0, val_start_idx - self.embargo_sessions)
        embargo_cutoff_date = all_market_dates[embargo_cutoff_idx]

        # Condition 1: Sample date must be within train window
        train_start = train_sessions[0]
        train_end = train_sessions[-1]
        is_train_session = (df_with_target_ends[date_col] >= train_start) & (df_with_target_ends[date_col] <= train_end)
        
        # Condition 2: Actual per-observation target end date must be strictly before embargo cutoff
        # (If target_end_date is NaN because stock delisted, we check observation date itself)
        actual_end_dates = df_with_target_ends[target_end_col].fillna(df_with_target_ends[date_col])
        is_purged_ok = actual_end_dates < embargo_cutoff_date

        purged_mask = is_train_session & is_purged_ok
        purged_indices = df_with_target_ends[purged_mask].index.values

        if len(purged_indices) == 0:
            largest_target_end = "None"
            is_leaky = False
            margin_days = 0
        else:
            largest_target_end = actual_end_dates.loc[purged_indices].max()
            is_leaky = largest_target_end >= val_start_date
            margin_days = (pd.to_datetime(val_start_date) - pd.to_datetime(largest_target_end)).days

        audit = {
            "val_start_date": val_start_date,
            "embargo_cutoff_date": embargo_cutoff_date,
            "largest_actual_target_end": largest_target_end,
            "embargo_sessions": self.embargo_sessions,
            "margin_calendar_days": margin_days,
            "leakage_detected": is_leaky,
            "n_purged_samples": len(purged_indices),
        }
        return purged_indices, audit
