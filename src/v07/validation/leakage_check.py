"""Leakage detection and assertion utilities for StockReason v0.7.

Guarantees zero information leakage between training and evaluation blocks
by rigorously checking:
1. Direct index intersection.
2. Direct observation date / cross-sectional overlap.
3. Label overlap (purging): training labels extending into evaluation periods.
4. Autocorrelation overlap (embargo): training observations occurring immediately
   after evaluation periods within the embargo window.
"""

from __future__ import annotations

import logging
from typing import Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LeakageError(ValueError):
    """Raised when information leakage is detected between train and evaluation splits."""
    pass


def _extract_subset(
    indices: Union[np.ndarray, pd.Index, Sequence[int], Sequence[str]],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Extract a DataFrame slice by integer position or index labels."""
    if len(indices) == 0:
        return panel.iloc[0:0].copy()

    # If boolean mask
    arr = np.asarray(indices)
    if arr.dtype == bool:
        if len(arr) != len(panel):
            raise ValueError(
                f"Boolean index length ({len(arr)}) does not match panel length ({len(panel)})."
            )
        return panel.iloc[arr].copy()

    # If integers matching positional range
    if np.issubdtype(arr.dtype, np.integer):
        # Check if indices are within [0, len(panel))
        if len(arr) > 0 and arr.min() >= 0 and arr.max() < len(panel):
            # Check whether panel.index is not an integer index or if iloc makes sense
            return panel.iloc[arr].copy()

    # Fallback to .loc for label indexing
    try:
        return panel.loc[indices].copy()
    except Exception:
        # Try iloc as last resort
        return panel.iloc[arr].copy()


def assert_no_leakage(
    train_idx: Union[np.ndarray, pd.Index, Sequence[int]],
    eval_idx: Union[np.ndarray, pd.Index, Sequence[int]],
    panel: pd.DataFrame,
    embargo_days: int = 0,
    max_horizon_days: int = 0,
    date_col: str = "date",
    label_end_col: str = "label_end",
    unique_dates: Sequence[pd.Timestamp] | None = None,
) -> None:
    """Assert that there is strictly zero leakage between train and evaluation sets.

    Parameters
    ----------
    train_idx : array-like
        Indices or positions of training samples in `panel`.
    eval_idx : array-like
        Indices or positions of evaluation (val or test) samples in `panel`.
    panel : pd.DataFrame
        Full panel dataset containing at least a date column (or DatetimeIndex).
    embargo_days : int, default 0
        Number of embargo trading/calendar days required after each evaluation block.
    max_horizon_days : int, default 0
        Maximum forward label horizon in days. Used if `label_end_col` is not present.
    date_col : str, default 'date'
        Name of the observation date column. If not present, index is checked.
    label_end_col : str, default 'label_end'
        Name of the column recording the end date of each observation's forward label.
    unique_dates : sequence of pd.Timestamp, optional
        Precomputed sorted unique calendar-ordered dates in the panel.

    Raises
    ------
    LeakageError
        If index overlap, date overlap, purging violation, or embargo violation is found.
    """
    if len(train_idx) == 0 or len(eval_idx) == 0:
        return

    # 1. Direct index intersection check
    train_set = set(np.asarray(train_idx))
    eval_set = set(np.asarray(eval_idx))
    idx_overlap = train_set.intersection(eval_set)
    if idx_overlap:
        raise LeakageError(
            f"Direct index overlap detected: {len(idx_overlap)} rows are present "
            f"in both train and eval splits (e.g., indices: {sorted(list(idx_overlap))[:5]})."
        )

    # Extract slices
    train_df = _extract_subset(train_idx, panel)
    eval_df = _extract_subset(eval_idx, panel)

    # Resolve dates
    if date_col in train_df.columns:
        train_dates = pd.to_datetime(train_df[date_col])
        eval_dates = pd.to_datetime(eval_df[date_col])
    elif isinstance(panel.index, pd.DatetimeIndex):
        train_dates = pd.to_datetime(train_df.index)
        eval_dates = pd.to_datetime(eval_df.index)
    else:
        raise ValueError(
            f"Cannot resolve dates: '{date_col}' column not found and index is not DatetimeIndex."
        )

    # 2. Direct date overlap check (Cross-sectional leakage prevention)
    train_date_set = set(train_dates)
    eval_date_set = set(eval_dates)
    common_dates = train_date_set.intersection(eval_date_set)
    if common_dates:
        sample_common = sorted(list(common_dates))[:3]
        raise LeakageError(
            f"Cross-sectional date overlap detected: {len(common_dates)} dates "
            f"appear in both training and evaluation sets (e.g. {sample_common}). "
            f"Calendar-ordered date boundaries must be disjoint."
        )

    # Resolve label_end dates for training rows
    if label_end_col in train_df.columns and not train_df[label_end_col].isna().all():
        train_label_ends = pd.to_datetime(train_df[label_end_col])
    else:
        # Fall back to date + max_horizon_days
        if max_horizon_days > 0:
            if unique_dates is not None:
                # Map observation date to trading date ahead
                u_dates = list(pd.to_datetime(unique_dates))
                date_to_pos = {d: i for i, d in enumerate(u_dates)}
                train_label_ends = train_dates.map(
                    lambda d: u_dates[min(date_to_pos[d] + max_horizon_days, len(u_dates) - 1)]
                    if d in date_to_pos else d + pd.Timedelta(days=max_horizon_days)
                )
            else:
                train_label_ends = train_dates + pd.to_timedelta(max_horizon_days, unit="D")
        else:
            train_label_ends = train_dates

    # 3. Purging and Embargo checks across all contiguous evaluation blocks
    # Evaluation dates can consist of 1 or more contiguous blocks (e.g. in CPCV)
    sorted_eval_dates = np.array(sorted(eval_date_set))
    if len(sorted_eval_dates) == 0:
        return

    # Find contiguous evaluation blocks
    if unique_dates is not None:
        all_u_dates = np.array(sorted(pd.to_datetime(unique_dates)))
    else:
        if date_col in panel.columns:
            all_u_dates = np.array(sorted(pd.to_datetime(panel[date_col].unique())))
        else:
            all_u_dates = np.array(sorted(pd.to_datetime(panel.index.unique())))

    # Determine blocks of eval dates in all_u_dates
    eval_pos = np.isin(all_u_dates, sorted_eval_dates)
    diffs = np.diff(np.concatenate(([0], eval_pos.view(np.int8), [0])))
    block_starts_idx = np.where(diffs == 1)[0]
    block_ends_idx = np.where(diffs == -1)[0] - 1

    for b_start_idx, b_end_idx in zip(block_starts_idx, block_ends_idx):
        eval_block_start = all_u_dates[b_start_idx]
        eval_block_end = all_u_dates[b_end_idx]

        # Check Purging: Training observations occurring strictly before eval_block_start
        # must have their forward-looking label end strictly before eval_block_start.
        pre_eval_mask = train_dates < eval_block_start
        if pre_eval_mask.any():
            violating_purge = pre_eval_mask & (train_label_ends >= eval_block_start)
            if violating_purge.any():
                violating_rows = train_df[violating_purge]
                max_le = train_label_ends[violating_purge].max()
                raise LeakageError(
                    f"Purging violation: {len(violating_rows)} training rows observed before "
                    f"eval block start ({eval_block_start.strftime('%Y-%m-%d')}) have forward label "
                    f"reaching into or beyond evaluation block (max label_end: "
                    f"{max_le.strftime('%Y-%m-%d')})."
                )

        # Check Embargo: Training observations occurring after eval_block_end
        # must not fall within the embargo window following eval_block_end.
        if embargo_days > 0:
            embargo_end_idx = min(b_end_idx + embargo_days, len(all_u_dates) - 1)
            embargo_end_date = all_u_dates[embargo_end_idx]

            post_eval_mask = train_dates > eval_block_end
            if post_eval_mask.any():
                violating_embargo = post_eval_mask & (train_dates <= embargo_end_date)
                if violating_embargo.any():
                    violating_rows = train_df[violating_embargo]
                    min_viol = train_dates[violating_embargo].min()
                    raise LeakageError(
                        f"Embargo violation: {len(violating_rows)} training rows fall within "
                        f"the {embargo_days}-day embargo window following eval block end "
                        f"({eval_block_end.strftime('%Y-%m-%d')} to "
                        f"{embargo_end_date.strftime('%Y-%m-%d')}). Earliest violating row: "
                        f"{min_viol.strftime('%Y-%m-%d')}."
                    )
