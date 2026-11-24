"""Purged and Embargoed Walk-Forward Cross-Validation.

Implements the standard validation engine for StockReason v0.7.
Ensures zero data leakage in financial panel data with overlapping label horizons
and cross-sectional dependencies across multiple assets.
"""

from __future__ import annotations

import logging
from typing import Generator, List, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.v07.validation.leakage_check import assert_no_leakage
except ImportError:
    from v07.validation.leakage_check import assert_no_leakage

logger = logging.getLogger(__name__)


class PurgedWalkForward:
    """Purged and Embargoed Walk-Forward Splitter for multi-asset panel time series.

    Structure:
    1. Expanding training window: anchored at dataset start, expanding with each fold.
    2. Purging: training rows whose forward-looking label overlaps with the validation
       or test period are dropped from the training set.
    3. Validation block: used for hyperparameter tuning / early stopping.
    4. Embargo: a buffer period following the validation block to prevent autocorrelation
       leakage into the test block.
    5. Test block: untouched out-of-sample evaluation period.
    6. Embargo: a buffer period following the test block.

    All boundaries are computed on calendar-ordered unique dates, ensuring that all
    tickers share identical date boundaries without cross-sectional misalignment.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    embargo_days : int, default 0
        Number of calendar-ordered unique dates for embargo following evaluation blocks.
    min_train_days : int, default 0
        Minimum number of unique dates required in the initial training block.
    val_days : int, default 0
        Number of unique dates allocated to the validation block.
    test_days : int, default 0
        Number of unique dates allocated to the test block.
    max_horizon_days : int, default 0
        Maximum forward label horizon in days. Used when `label_end` column is not
        explicitly present, or to enforce purging over the longest horizon.
    """

    def __init__(
        self,
        n_splits: int,
        embargo_days: int = 0,
        min_train_days: int = 0,
        val_days: int = 0,
        test_days: int = 0,
        max_horizon_days: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}.")
        if min_train_days < 0:
            raise ValueError(f"min_train_days must be >= 0, got {min_train_days}.")
        if val_days < 0:
            raise ValueError(f"val_days must be >= 0, got {val_days}.")
        if test_days < 0:
            raise ValueError(f"test_days must be >= 0, got {test_days}.")
        if max_horizon_days < 0:
            raise ValueError(f"max_horizon_days must be >= 0, got {max_horizon_days}.")

        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days
        self.val_days = val_days
        self.test_days = test_days
        self.max_horizon_days = max_horizon_days

    def _resolve_dates(
        self, panel: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Series, pd.Series]:
        """Extract sorted unique dates, per-row dates, and per-row label_ends."""
        if "date" in panel.columns:
            row_dates = pd.to_datetime(panel["date"]).reset_index(drop=True)
        elif isinstance(panel.index, pd.DatetimeIndex):
            row_dates = pd.Series(panel.index).reset_index(drop=True)
        else:
            raise ValueError(
                "Panel must have a 'date' column or a DatetimeIndex."
            )

        unique_dates = np.array(sorted(row_dates.unique()))
        n_unique = len(unique_dates)

        # Resolve row label_ends
        date_to_pos = {d: i for i, d in enumerate(unique_dates)}

        if "label_end" in panel.columns and not panel["label_end"].isna().all():
            row_label_ends = pd.to_datetime(panel["label_end"]).reset_index(drop=True)
            # If max_horizon_days is specified, ensure label_end extends at least max_horizon_days
            if self.max_horizon_days > 0:
                horizon_ends = row_dates.map(
                    lambda d: unique_dates[min(date_to_pos[d] + self.max_horizon_days, n_unique - 1)]
                )
                row_label_ends = np.maximum(row_label_ends, horizon_ends)
        else:
            if self.max_horizon_days > 0:
                row_label_ends = row_dates.map(
                    lambda d: unique_dates[min(date_to_pos[d] + self.max_horizon_days, n_unique - 1)]
                )
            else:
                row_label_ends = row_dates.copy()

        return unique_dates, row_dates, pd.Series(row_label_ends)

    def _compute_fold_boundaries(
        self, unique_dates: np.ndarray
    ) -> List[dict]:
        """Compute the unique date index ranges for each fold."""
        N = len(unique_dates)

        # Parameter inference if defaults / zeros were supplied
        test_days = self.test_days
        val_days = self.val_days
        min_train_days = self.min_train_days

        if test_days <= 0:
            # Allocate roughly 10% to test per split, minimum 5 days
            test_days = max(5, int(N / (self.n_splits + 2)))
        if val_days <= 0:
            val_days = test_days
        if min_train_days <= 0:
            min_train_days = max(10, N - (val_days + self.embargo_days + test_days) * self.n_splits)
            if min_train_days < 10:
                min_train_days = max(5, int(N * 0.2))

        # Check feasibility
        eval_span = val_days + self.embargo_days + test_days
        first_test_end = min_train_days + eval_span - 1
        last_test_end = N - 1

        if first_test_end > last_test_end:
            raise ValueError(
                f"Insufficient unique dates ({N}) for walk-forward validation: "
                f"min_train_days={min_train_days}, val_days={val_days}, "
                f"embargo_days={self.embargo_days}, test_days={test_days} "
                f"(requires at least {min_train_days + eval_span} dates)."
            )

        folds = []
        if self.n_splits == 1:
            val_start = last_test_end - eval_span + 1
            step = 0
        else:
            total_step_room = last_test_end - first_test_end
            step = total_step_room / (self.n_splits - 1)

        for k in range(self.n_splits):
            if self.n_splits == 1:
                curr_test_end = last_test_end
            else:
                curr_test_end = int(round(first_test_end + k * step))
                curr_test_end = min(curr_test_end, last_test_end)

            test_start = curr_test_end - test_days + 1
            val_end = test_start - self.embargo_days - 1
            val_start = val_end - val_days + 1
            train_end = val_start - 1

            if train_end < 0:
                raise ValueError(
                    f"Fold {k} generated negative train_end index ({train_end}). "
                    f"Please reduce val_days, test_days, or embargo_days."
                )

            fold_info = {
                "fold_id": k,
                "train_start_idx": 0,  # expanding window
                "train_end_idx": train_end,
                "val_start_idx": val_start,
                "val_end_idx": val_end,
                "test_start_idx": test_start,
                "test_end_idx": curr_test_end,
                "val_days": val_days,
                "test_days": test_days,
                "embargo_days": self.embargo_days,
            }
            folds.append(fold_info)

        return folds

    def split(
        self, panel: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, int], None, None]:
        """Generate indices for purged walk-forward train, val, and test splits.

        Parameters
        ----------
        panel : pd.DataFrame
            Panel data with 'date' column or DatetimeIndex, and optional 'label_end'.

        Yields
        ------
        train_idx : np.ndarray of int
            Integer positional indices for the purged training set.
        val_idx : np.ndarray of int
            Integer positional indices for the validation set.
        test_idx : np.ndarray of int
            Integer positional indices for the test set.
        fold_id : int
            Current fold identifier (0 to n_splits - 1).
        """
        unique_dates, row_dates, row_label_ends = self._resolve_dates(panel)
        fold_configs = self._compute_fold_boundaries(unique_dates)

        # Map dates to row indices
        date_series = pd.Series(row_dates.values, index=np.arange(len(panel)))
        label_end_series = pd.Series(row_label_ends.values, index=np.arange(len(panel)))

        for fold in fold_configs:
            k = fold["fold_id"]
            val_start_date = unique_dates[fold["val_start_idx"]]
            val_end_date = unique_dates[fold["val_end_idx"]]
            test_start_date = unique_dates[fold["test_start_idx"]]
            test_end_date = unique_dates[fold["test_end_idx"]]

            # Raw training mask (expanding window up to day before val_start)
            raw_train_mask = date_series < val_start_date

            # Purging: Drop training rows whose forward-looking label_end >= val_start_date
            purged_train_mask = raw_train_mask & (label_end_series < val_start_date)

            train_idx = np.where(purged_train_mask)[0]

            # Validation mask
            val_mask = (date_series >= val_start_date) & (date_series <= val_end_date)
            val_idx = np.where(val_mask)[0]

            # Test mask
            test_mask = (date_series >= test_start_date) & (date_series <= test_end_date)
            test_idx = np.where(test_mask)[0]

            # Strict assertion checks against any silent leakage
            assert_no_leakage(
                train_idx=train_idx,
                eval_idx=val_idx,
                panel=panel,
                embargo_days=self.embargo_days,
                max_horizon_days=self.max_horizon_days,
                unique_dates=unique_dates,
            )
            assert_no_leakage(
                train_idx=train_idx,
                eval_idx=test_idx,
                panel=panel,
                embargo_days=self.embargo_days,
                max_horizon_days=self.max_horizon_days,
                unique_dates=unique_dates,
            )

            # Check val vs test boundary
            if len(val_idx) > 0 and len(test_idx) > 0:
                val_set = set(val_idx)
                test_set = set(test_idx)
                if val_set.intersection(test_set):
                    raise ValueError(
                        f"Fold {k}: Validation and test indices overlap!"
                    )
                if val_end_date >= test_start_date:
                    raise ValueError(
                        f"Fold {k}: Validation end ({val_end_date}) >= test start ({test_start_date})."
                    )

            logger.info(
                f"[PurgedWalkForward] Fold {k}: Train rows={len(train_idx)} "
                f"(purged {np.sum(raw_train_mask) - len(train_idx)} rows), "
                f"Val rows={len(val_idx)} [{val_start_date.strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}], "
                f"Test rows={len(test_idx)} [{test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}]."
            )

            yield train_idx, val_idx, test_idx, k

    def get_fold_summary(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return a structured summary DataFrame of all fold date ranges and purge statistics."""
        unique_dates, row_dates, row_label_ends = self._resolve_dates(panel)
        fold_configs = self._compute_fold_boundaries(unique_dates)

        date_series = pd.Series(row_dates.values, index=np.arange(len(panel)))
        label_end_series = pd.Series(row_label_ends.values, index=np.arange(len(panel)))

        summary_rows = []
        for fold in fold_configs:
            k = fold["fold_id"]
            train_start_date = unique_dates[fold["train_start_idx"]]
            val_start_date = unique_dates[fold["val_start_idx"]]
            val_end_date = unique_dates[fold["val_end_idx"]]
            test_start_date = unique_dates[fold["test_start_idx"]]
            test_end_date = unique_dates[fold["test_end_idx"]]

            raw_train_mask = date_series < val_start_date
            purged_train_mask = raw_train_mask & (label_end_series < val_start_date)

            raw_train_count = int(np.sum(raw_train_mask))
            purged_train_count = int(np.sum(purged_train_mask))
            rows_purged = raw_train_count - purged_train_count

            val_mask = (date_series >= val_start_date) & (date_series <= val_end_date)
            test_mask = (date_series >= test_start_date) & (date_series <= test_end_date)

            val_count = int(np.sum(val_mask))
            test_count = int(np.sum(test_mask))

            summary_rows.append(
                {
                    "fold_id": k,
                    "train_start": train_start_date.strftime("%Y-%m-%d"),
                    "train_end": unique_dates[fold["train_end_idx"]].strftime("%Y-%m-%d"),
                    "raw_train_rows": raw_train_count,
                    "purged_rows": rows_purged,
                    "kept_train_rows": purged_train_count,
                    "val_start": val_start_date.strftime("%Y-%m-%d"),
                    "val_end": val_end_date.strftime("%Y-%m-%d"),
                    "val_rows": val_count,
                    "embargo_days": self.embargo_days,
                    "test_start": test_start_date.strftime("%Y-%m-%d"),
                    "test_end": test_end_date.strftime("%Y-%m-%d"),
                    "test_rows": test_count,
                }
            )

        return pd.DataFrame(summary_rows)
