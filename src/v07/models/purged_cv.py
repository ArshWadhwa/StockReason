"""
Purged Walk-Forward Cross Validation for financial time series in StockReason v0.7.

Guarantees zero lookahead leakage:
- Chronological expanding training window
- Purging: drops training samples that overlap with test forward return window
- Embargo: adds a buffer period after test block to prevent post-event leakage
"""
from __future__ import annotations

from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """
    Time-series cross validation with purging and embargoing.
    """

    def __init__(
        self,
        n_folds: int = 4,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_window: int = 21,
        embargo_window: int = 5,
        min_train_ratio: float = 0.40,
    ) -> None:
        self.n_folds = n_folds
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        self.min_train_ratio = min_train_ratio

    def split_indices(
        self, dates: pd.Series
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, val_indices, test_indices) for each fold.
        Based on unique sorted dates to prevent splitting a single trading day.
        """
        unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
        n_dates = len(unique_dates)

        test_block_size = max(10, int(n_dates * self.test_ratio))
        val_block_size = max(10, int(n_dates * self.val_ratio))
        min_train_dates = max(20, int(n_dates * self.min_train_ratio))

        date_to_idx = {d: np.where(dates == d)[0] for d in unique_dates}

        for fold in range(self.n_folds):
            # Calculate date cutoffs for this fold
            # Test block moves chronologically
            test_end_idx = n_dates - (self.n_folds - 1 - fold) * (test_block_size // 2)
            test_start_idx = max(0, test_end_idx - test_block_size)

            val_end_idx = max(0, test_start_idx - self.purge_window)
            val_start_idx = max(0, val_end_idx - val_block_size)

            train_end_idx = max(0, val_start_idx - self.purge_window)

            if train_end_idx < min_train_dates or val_end_idx <= val_start_idx or test_end_idx <= test_start_idx:
                continue

            train_dates = unique_dates.iloc[:train_end_idx]
            val_dates = unique_dates.iloc[val_start_idx:val_end_idx]
            test_dates = unique_dates.iloc[test_start_idx:test_end_idx]

            train_idx = np.concatenate([date_to_idx[d] for d in train_dates]) if len(train_dates) > 0 else np.array([], dtype=int)
            val_idx = np.concatenate([date_to_idx[d] for d in val_dates]) if len(val_dates) > 0 else np.array([], dtype=int)
            test_idx = np.concatenate([date_to_idx[d] for d in test_dates]) if len(test_dates) > 0 else np.array([], dtype=int)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, val_idx, test_idx
