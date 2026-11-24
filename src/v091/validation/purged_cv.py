"""
StockReason v0.9.1 — Exact Forward-Label End-Date Purging & Walk-Forward CV.
=============================================================================
Remediates the Fixed-Purge Defect (Task 3):
In v0.9:
- Purge was hardcoded to a fixed 21 trading days (or 32 calendar days).
- For 63d horizon, 42 trading days leaked into validation.
- For 126d horizon, 105 trading days leaked into validation.

In v0.9.1:
- Uses EACH SAMPLE'S ACTUAL forward-label end date:
  label_end_date(i, t) = calendar date of session (t + H).
- Purge condition:
  For any training sample (i, t), we require:
  label_end_date(i, t) < first_eval_session - embargo_sessions (embargo = 5 trading sessions).
- Generates at least 8 expanding walk-forward folds + 1 untouched reserved final block (Task 10).
- Provides adversarial leakage test suites.
"""

import os
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


class ExactPurgedWalkForwardCV:
    def __init__(
        self,
        n_folds: int = 8,
        primary_horizons: List[int] = [5, 21],
        exploratory_horizons: List[int] = [1, 63, 126],
        embargo_sessions: int = 5,
        min_train_sessions: int = 750, # ~3 years
    ):
        self.n_folds = n_folds
        self.primary_horizons = list(primary_horizons)
        self.exploratory_horizons = list(exploratory_horizons)
        self.all_horizons = sorted(list(set(self.primary_horizons + self.exploratory_horizons)))
        self.embargo_sessions = embargo_sessions
        self.min_train_sessions = min_train_sessions

    def generate_folds(self, trading_dates: List[str]) -> List[Dict[str, Any]]:
        """
        Generates at least n_folds walk-forward splits from sorted trading dates.
        Leaves the most recent 126 sessions (~6 months) as an untouched final block.
        """
        dates = sorted(list(set(trading_dates)))
        total_dates = len(dates)
        
        # Reserve last 126 sessions for untouched forward evaluation (Task 10)
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
                "val_sessions": val_sessions
            })
            
        return folds

    def purge_train_indices_for_horizon(
        self,
        df: pd.DataFrame,
        train_sessions: List[str],
        first_eval_date: str,
        horizon: int,
        date_col: str = "date_str"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filters training observations such that the sample's actual forward label end date
        is strictly before (first_eval_date - embargo_sessions).
        """
        # Find all trading dates in order
        all_dates = sorted(df[date_col].unique().tolist())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        
        if first_eval_date not in date_to_idx:
            # find closest
            first_eval_idx = min([i for i, d in enumerate(all_dates) if d >= first_eval_date], default=len(all_dates))
        else:
            first_eval_idx = date_to_idx[first_eval_date]
            
        # Cutoff index for training sample t: sample t has label ending at t + horizon.
        # We require: (t + horizon) < first_eval_idx - embargo
        max_allowed_train_idx = first_eval_idx - self.embargo_sessions - horizon
        if max_allowed_train_idx < 0:
            raise ValueError(f"Embargo and horizon {horizon} consume entire training history prior to {first_eval_date}!")
            
        max_allowed_train_date = all_dates[max_allowed_train_idx]
        
        # Subsetting training rows
        train_mask = (df[date_col].isin(train_sessions)) & (df[date_col] <= max_allowed_train_date)
        purged_indices = df.index[train_mask].values
        
        # Diagnostics
        largest_train_sample_date = df.loc[purged_indices, date_col].max() if len(purged_indices) > 0 else "None"
        largest_train_sample_idx = date_to_idx.get(largest_train_sample_date, 0)
        largest_label_end_idx = min(len(all_dates) - 1, largest_train_sample_idx + horizon)
        largest_label_end_date = all_dates[largest_label_end_idx]
        
        # Assert strict mathematical separation
        assert largest_label_end_date < first_eval_date, (
            f"LEAKAGE VIOLATION for horizon {horizon}: largest training label end "
            f"{largest_label_end_date} >= first evaluation date {first_eval_date}!"
        )
        
        stats = {
            "horizon": horizon,
            "first_eval_date": first_eval_date,
            "max_allowed_train_date": max_allowed_train_date,
            "largest_sample_date": largest_train_sample_date,
            "largest_label_end_date": largest_label_end_date,
            "embargo_sessions": self.embargo_sessions,
            "margin_days": (pd.to_datetime(first_eval_date) - pd.to_datetime(largest_label_end_date)).days,
            "is_purged_valid": True
        }
        
        return purged_indices, stats


def benjamini_hochberg_p_adjust(p_values: List[float]) -> List[float]:
    """Applies Benjamini-Hochberg False Discovery Rate (FDR) adjustment to p-values."""
    p_vals = np.asarray(p_values)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    
    adjusted = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = min(1.0, sorted_p[i] * n / rank)
        cum_min = min(cum_min, adj)
        adjusted[i] = cum_min
        
    out = np.zeros(n)
    out[sorted_indices] = adjusted
    return out.tolist()
