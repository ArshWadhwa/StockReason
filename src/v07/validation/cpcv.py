"""Combinatorial Purged Cross-Validation (CPCV).

OPTIONAL / ADVANCED VALIDATION ENGINE.

Based on Marcos López de Prado, *Advances in Financial Machine Learning* (2018), Chapter 12.
Generates all C(N, k) train/test combinations across N chronological groups with purging
and embargo, enabling the construction of multiple independent historical backtest paths
without lookahead or leakage bias.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, Generator, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.v07.validation.leakage_check import assert_no_leakage
except ImportError:
    from v07.validation.leakage_check import assert_no_leakage

logger = logging.getLogger(__name__)


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV).

    Divides the timeline of calendar-ordered unique dates into N contiguous groups.
    For each combination of k test groups (C(N, k) combinations):
    - The test set is the union of the k test groups.
    - The training set contains the remaining N - k groups, purged of labels that overlap
      with any test group start, and embargoed after each test group end.

    Parameters
    ----------
    n_groups : int, default 6
        Total number of chronological groups (N).
    k_test_groups : int, default 2
        Number of test groups per combination (k).
    embargo_days : int, default 0
        Number of calendar-ordered unique dates for embargo following each test group.
    max_horizon_days : int, default 0
        Maximum forward label horizon in days for purging.
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test_groups: int = 2,
        embargo_days: int = 0,
        max_horizon_days: int = 0,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}.")
        if k_test_groups < 1 or k_test_groups >= n_groups:
            raise ValueError(
                f"k_test_groups must be in [1, n_groups - 1], got {k_test_groups} (n_groups={n_groups})."
            )
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}.")
        if max_horizon_days < 0:
            raise ValueError(f"max_horizon_days must be >= 0, got {max_horizon_days}.")

        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.embargo_days = embargo_days
        self.max_horizon_days = max_horizon_days

    @property
    def n_combinations(self) -> int:
        """Total number of CV combinations C(N, k)."""
        import math
        return math.comb(self.n_groups, self.k_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of distinct historical backtest paths C(N-1, k-1)."""
        import math
        return math.comb(self.n_groups - 1, self.k_test_groups - 1)

    def _resolve_dates(
        self, panel: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Series, pd.Series]:
        """Extract sorted unique dates, per-row dates, and per-row label_ends."""
        if "date" in panel.columns:
            row_dates = pd.to_datetime(panel["date"]).reset_index(drop=True)
        elif isinstance(panel.index, pd.DatetimeIndex):
            row_dates = pd.Series(panel.index).reset_index(drop=True)
        else:
            raise ValueError("Panel must have a 'date' column or a DatetimeIndex.")

        unique_dates = np.array(sorted(row_dates.unique()))
        n_unique = len(unique_dates)

        date_to_pos = {d: i for i, d in enumerate(unique_dates)}

        if "label_end" in panel.columns and not panel["label_end"].isna().all():
            row_label_ends = pd.to_datetime(panel["label_end"]).reset_index(drop=True)
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

    def _partition_groups(
        self, unique_dates: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Partition unique date indices into N contiguous groups [start_idx, end_idx]."""
        N = len(unique_dates)
        if N < self.n_groups:
            raise ValueError(
                f"Insufficient unique dates ({N}) to partition into {self.n_groups} groups."
            )

        group_bounds = []
        bounds = np.linspace(0, N, self.n_groups + 1, dtype=int)
        for i in range(self.n_groups):
            start = bounds[i]
            end = bounds[i + 1] - 1
            group_bounds.append((start, end))

        return group_bounds

    def split(
        self, panel: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray, int, Tuple[int, ...]], None, None]:
        """Generate train/test split indices for each CPCV combination.

        Parameters
        ----------
        panel : pd.DataFrame
            Panel data with 'date' column or DatetimeIndex, and optional 'label_end'.

        Yields
        ------
        train_idx : np.ndarray of int
            Integer positional indices for the purged and embargoed training set.
        test_idx : np.ndarray of int
            Integer positional indices for the test set across the k chosen groups.
        combo_id : int
            Sequential index of the combination (0 to C(N, k) - 1).
        test_groups : tuple of int
            Indices of the groups selected for testing in this combination.
        """
        unique_dates, row_dates, row_label_ends = self._resolve_dates(panel)
        group_bounds = self._partition_groups(unique_dates)

        date_series = pd.Series(row_dates.values, index=np.arange(len(panel)))
        label_end_series = pd.Series(row_label_ends.values, index=np.arange(len(panel)))

        # Assign each row to a group
        row_group = np.full(len(panel), -1, dtype=int)
        for g_id, (g_start, g_end) in enumerate(group_bounds):
            g_start_date = unique_dates[g_start]
            g_end_date = unique_dates[g_end]
            mask = (date_series >= g_start_date) & (date_series <= g_end_date)
            row_group[mask] = g_id

        # Generate all combinations of k test groups
        all_combinations = list(itertools.combinations(range(self.n_groups), self.k_test_groups))

        for combo_id, test_groups in enumerate(all_combinations):
            test_group_set = set(test_groups)
            train_group_set = set(range(self.n_groups)) - test_group_set

            # Test mask
            test_mask = np.isin(row_group, list(test_group_set))
            test_idx = np.where(test_mask)[0]

            # Initial train mask: all rows in training groups
            train_mask = np.isin(row_group, list(train_group_set))

            # Apply purging and embargo for each test group
            for tg in test_groups:
                g_start, g_end = group_bounds[tg]
                test_start_date = unique_dates[g_start]
                test_end_date = unique_dates[g_end]

                # Purge training samples occurring before test_start_date
                # whose label_end >= test_start_date
                pre_test_train = train_mask & (date_series < test_start_date)
                purge_mask = pre_test_train & (label_end_series >= test_start_date)
                train_mask[purge_mask] = False

                # Embargo training samples occurring after test_end_date
                if self.embargo_days > 0:
                    embargo_end_pos = min(g_end + self.embargo_days, len(unique_dates) - 1)
                    embargo_end_date = unique_dates[embargo_end_pos]
                    embargo_mask = (
                        train_mask
                        & (date_series > test_end_date)
                        & (date_series <= embargo_end_date)
                    )
                    train_mask[embargo_mask] = False

            train_idx = np.where(train_mask)[0]

            # Strict assertion verification
            assert_no_leakage(
                train_idx=train_idx,
                eval_idx=test_idx,
                panel=panel,
                embargo_days=self.embargo_days,
                max_horizon_days=self.max_horizon_days,
                unique_dates=unique_dates,
            )

            logger.info(
                f"[CPCV] Combo {combo_id}/{len(all_combinations)}: Test groups={test_groups}, "
                f"Train rows={len(train_idx)}, Test rows={len(test_idx)}."
            )

            yield train_idx, test_idx, combo_id, test_groups


def stitch_backtest_paths(
    cv_predictions: Union[Dict[int, pd.DataFrame], pd.DataFrame],
    n_groups: int = 6,
    k_test_groups: int = 2,
    group_col: str = "group_id",
    date_col: str = "date",
) -> Dict[int, pd.DataFrame]:
    """Stitch out-of-sample predictions from CPCV into multiple sequential backtest paths.

    In Combinatorial Purged Cross-Validation, each chronological group appears as a test group
    in C(N-1, k-1) combinations. These can be partitioned into C(N-1, k-1) full historical
    backtest paths where each path spans the entire timeline from group 0 to group N-1.

    Parameters
    ----------
    cv_predictions : dict of {combo_id: pd.DataFrame} or pd.DataFrame
        Predictions generated for each CPCV combination. Must contain rows for tested groups.
    n_groups : int, default 6
        Total number of chronological groups (N).
    k_test_groups : int, default 2
        Number of test groups per combination (k).
    group_col : str, default 'group_id'
        Column indicating the group ID of each row (if in DataFrame).
    date_col : str, default 'date'
        Date column for chronological sorting.

    Returns
    -------
    dict of {path_id: pd.DataFrame}
        A dictionary mapping path index (0 to C(N-1, k-1) - 1) to a full, chronological
        DataFrame of out-of-sample predictions spanning all N groups.
    """
    import math

    n_combos = math.comb(n_groups, k_test_groups)
    n_paths = math.comb(n_groups - 1, k_test_groups - 1)

    all_combos = list(itertools.combinations(range(n_groups), k_test_groups))

    # Map each group to the list of combination IDs where it served as a test group
    group_to_combos: Dict[int, List[int]] = {g: [] for g in range(n_groups)}
    for combo_id, test_groups in enumerate(all_combos):
        for tg in test_groups:
            group_to_combos[tg].append(combo_id)

    # Each group has exactly n_paths combinations: len(group_to_combos[g]) == n_paths
    # Build n_paths paths. In path p, group g uses combination group_to_combos[g][p].
    paths: Dict[int, List[pd.DataFrame]] = {p: [] for p in range(n_paths)}

    for p in range(n_paths):
        for g in range(n_groups):
            combo_id = group_to_combos[g][p]

            if isinstance(cv_predictions, dict):
                combo_df = cv_predictions.get(combo_id)
                if combo_df is not None:
                    # Filter for rows belonging to group g if group_col exists
                    if group_col in combo_df.columns:
                        g_slice = combo_df[combo_df[group_col] == g].copy()
                    else:
                        g_slice = combo_df.copy()
                    g_slice["path_id"] = p
                    g_slice["combo_id"] = combo_id
                    g_slice["group_id"] = g
                    paths[p].append(g_slice)
            elif isinstance(cv_predictions, pd.DataFrame):
                mask = (cv_predictions["combo_id"] == combo_id)
                if group_col in cv_predictions.columns:
                    mask = mask & (cv_predictions[group_col] == g)
                g_slice = cv_predictions[mask].copy()
                g_slice["path_id"] = p
                paths[p].append(g_slice)

    # Concatenate and sort each path chronologically
    stitched_paths: Dict[int, pd.DataFrame] = {}
    for p in range(n_paths):
        if len(paths[p]) > 0:
            path_df = pd.concat(paths[p], ignore_index=True)
            if date_col in path_df.columns:
                path_df = path_df.sort_values(by=date_col).reset_index(drop=True)
            stitched_paths[p] = path_df
        else:
            stitched_paths[p] = pd.DataFrame()

    return stitched_paths
