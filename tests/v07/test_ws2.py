"""Comprehensive unit and integration tests for StockReason v0.7 validation engine (WS2).

Tests:
1. Synthetic panel with known dates and known horizons (h=5, h=21).
2. Verification of exact purge row counts for h=5 and h=21 across multi-ticker panels.
3. Verification of embargo window separation between evaluation blocks.
4. Verification that no ticker's data appears across train, val, and test splits on the same date.
5. Verification that `assert_no_leakage` detects:
   - Direct index overlap
   - Direct date overlap
   - Purging lookahead label overlap
   - Embargo violation
6. Verification of expanding training window property.
7. Verification of Combinatorial Purged Cross-Validation (CPCV) combinations, leakage checks,
   and backtest path stitching.
8. Validation parameter constraints and error handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v07.validation.cpcv import CombinatorialPurgedCV, stitch_backtest_paths
from src.v07.validation.leakage_check import LeakageError, assert_no_leakage
from src.v07.validation.purged_walk_forward import PurgedWalkForward


@pytest.fixture
def synthetic_panel_h5_h21():
    """Create a deterministic synthetic panel with 100 business days and 3 tickers."""
    tickers = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
    n_days = 100
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)
    u_dates = list(dates)

    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        for i, d in enumerate(dates):
            # Known horizons: h=5 and h=21 trading days forward
            idx_5d = min(i + 5, n_days - 1)
            idx_21d = min(i + 21, n_days - 1)

            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "close": 1000.0 + (ticker_idx + 1) * 100.0 + i,
                    "volume": 10000 + i * 50,
                    "label_end_5d": u_dates[idx_5d],
                    "label_end_21d": u_dates[idx_21d],
                    "label_end": u_dates[idx_21d],  # Default to longest horizon
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    return df


def test_purged_walk_forward_basic_split(synthetic_panel_h5_h21):
    """Verify basic split functionality, fold counts, and non-empty outputs."""
    df = synthetic_panel_h5_h21
    n_splits = 3
    splitter = PurgedWalkForward(
        n_splits=n_splits,
        embargo_days=3,
        min_train_days=30,
        val_days=15,
        test_days=15,
        max_horizon_days=5,
    )

    splits = list(splitter.split(df))
    assert len(splits) == n_splits

    for train_idx, val_idx, test_idx, fold_id in splits:
        assert len(train_idx) > 0, f"Fold {fold_id} train_idx is empty"
        assert len(val_idx) > 0, f"Fold {fold_id} val_idx is empty"
        assert len(test_idx) > 0, f"Fold {fold_id} test_idx is empty"
        assert fold_id in [0, 1, 2]


def test_exact_purge_count_h5(synthetic_panel_h5_h21):
    """Verify that for horizon h=5, exactly 5 trading days * 3 tickers = 15 rows are purged."""
    df = synthetic_panel_h5_h21.copy()
    # Explicitly set label_end to the 5-day forward date
    df["label_end"] = df["label_end_5d"]

    n_tickers = len(df["ticker"].unique())
    assert n_tickers == 3

    h = 5
    splitter = PurgedWalkForward(
        n_splits=3,
        embargo_days=2,
        min_train_days=40,
        val_days=15,
        test_days=15,
        max_horizon_days=h,
    )

    summary = splitter.get_fold_summary(df)

    for _, fold_row in summary.iterrows():
        # Every fold must purge exactly h * n_tickers rows
        expected_purged = h * n_tickers  # 5 * 3 = 15
        actual_purged = fold_row["purged_rows"]
        assert actual_purged == expected_purged, (
            f"Fold {fold_row['fold_id']}: Expected exactly {expected_purged} purged rows "
            f"for h={h}, but got {actual_purged}."
        )

        # Kept train rows must equal raw train rows minus purged rows
        assert fold_row["kept_train_rows"] == fold_row["raw_train_rows"] - expected_purged


def test_exact_purge_count_h21(synthetic_panel_h5_h21):
    """Verify that for horizon h=21, exactly 21 trading days * 3 tickers = 63 rows are purged."""
    df = synthetic_panel_h5_h21.copy()
    # Explicitly set label_end to the 21-day forward date
    df["label_end"] = df["label_end_21d"]

    n_tickers = len(df["ticker"].unique())
    assert n_tickers == 3

    h = 21
    splitter = PurgedWalkForward(
        n_splits=2,
        embargo_days=3,
        min_train_days=45,
        val_days=15,
        test_days=15,
        max_horizon_days=h,
    )

    summary = splitter.get_fold_summary(df)

    for _, fold_row in summary.iterrows():
        expected_purged = h * n_tickers  # 21 * 3 = 63
        actual_purged = fold_row["purged_rows"]
        assert actual_purged == expected_purged, (
            f"Fold {fold_row['fold_id']}: Expected exactly {expected_purged} purged rows "
            f"for h={h}, but got {actual_purged}."
        )

        assert fold_row["kept_train_rows"] == fold_row["raw_train_rows"] - expected_purged


def test_multi_horizon_purges_longest(synthetic_panel_h5_h21):
    """Verify that when multi-horizon data is provided, purging uses the longest horizon."""
    df = synthetic_panel_h5_h21.copy()
    # Provide both h=5 and h=21, but max_horizon_days is set to 21
    splitter = PurgedWalkForward(
        n_splits=2,
        embargo_days=2,
        min_train_days=45,
        val_days=15,
        test_days=15,
        max_horizon_days=21,
    )

    summary = splitter.get_fold_summary(df)
    n_tickers = len(df["ticker"].unique())

    for _, fold_row in summary.iterrows():
        # Must purge 21 days * 3 tickers = 63 rows, not 5 days * 3 tickers = 15
        assert fold_row["purged_rows"] == 21 * n_tickers


def test_embargo_isolation(synthetic_panel_h5_h21):
    """Verify that the embargo window cleanly isolates validation from test blocks."""
    df = synthetic_panel_h5_h21
    embargo_days = 5
    val_days = 12
    test_days = 12

    splitter = PurgedWalkForward(
        n_splits=3,
        embargo_days=embargo_days,
        min_train_days=35,
        val_days=val_days,
        test_days=test_days,
        max_horizon_days=5,
    )

    u_dates = np.array(sorted(df["date"].unique()))
    date_to_pos = {d: i for i, d in enumerate(u_dates)}

    for train_idx, val_idx, test_idx, fold_id in splitter.split(df):
        val_dates = df.iloc[val_idx]["date"]
        test_dates = df.iloc[test_idx]["date"]

        val_end_pos = date_to_pos[val_dates.max()]
        test_start_pos = date_to_pos[test_dates.min()]

        # The distance between val_end and test_start must equal embargo_days + 1
        trading_days_gap = test_start_pos - val_end_pos - 1
        assert trading_days_gap == embargo_days, (
            f"Fold {fold_id}: Embargo gap was {trading_days_gap} trading days, "
            f"expected {embargo_days}."
        )


def test_no_cross_sectional_date_leakage(synthetic_panel_h5_h21):
    """Verify that no date appears in multiple sets, preventing any cross-ticker leakage."""
    df = synthetic_panel_h5_h21
    splitter = PurgedWalkForward(
        n_splits=4,
        embargo_days=4,
        min_train_days=30,
        val_days=12,
        test_days=12,
        max_horizon_days=5,
    )

    for train_idx, val_idx, test_idx, fold_id in splitter.split(df):
        train_dates = set(df.iloc[train_idx]["date"])
        val_dates = set(df.iloc[val_idx]["date"])
        test_dates = set(df.iloc[test_idx]["date"])

        # Pairwise disjoint date sets
        assert train_dates.isdisjoint(val_dates), f"Fold {fold_id}: train and val share dates!"
        assert train_dates.isdisjoint(test_dates), f"Fold {fold_id}: train and test share dates!"
        assert val_dates.isdisjoint(test_dates), f"Fold {fold_id}: val and test share dates!"

        # Ensure no ticker appears on the same date across sets
        train_pairs = set(zip(df.iloc[train_idx]["ticker"], df.iloc[train_idx]["date"]))
        val_pairs = set(zip(df.iloc[val_idx]["ticker"], df.iloc[val_idx]["date"]))
        test_pairs = set(zip(df.iloc[test_idx]["ticker"], df.iloc[test_idx]["date"]))

        assert train_pairs.isdisjoint(val_pairs)
        assert train_pairs.isdisjoint(test_pairs)
        assert val_pairs.isdisjoint(test_pairs)


def test_expanding_training_window(synthetic_panel_h5_h21):
    """Verify that the training window strictly expands across successive folds."""
    df = synthetic_panel_h5_h21
    splitter = PurgedWalkForward(
        n_splits=4,
        embargo_days=3,
        min_train_days=30,
        val_days=10,
        test_days=10,
        max_horizon_days=5,
    )

    train_sizes = []
    for train_idx, val_idx, test_idx, fold_id in splitter.split(df):
        train_sizes.append(len(train_idx))

    # In expanding window, each subsequent fold must have more training data
    for i in range(len(train_sizes) - 1):
        assert train_sizes[i + 1] > train_sizes[i], (
            f"Fold {i+1} train rows ({train_sizes[i+1]}) not strictly greater than "
            f"Fold {i} train rows ({train_sizes[i]})."
        )


def test_assert_no_leakage_catches_deliberate_leaks(synthetic_panel_h5_h21):
    """Verify that assert_no_leakage raises LeakageError on deliberate leaks."""
    df = synthetic_panel_h5_h21
    splitter = PurgedWalkForward(
        n_splits=2,
        embargo_days=3,
        min_train_days=40,
        val_days=15,
        test_days=15,
        max_horizon_days=5,
    )

    train_idx, val_idx, test_idx, fold_id = next(splitter.split(df))

    # Baseline: Clean split must pass with no error
    assert_no_leakage(train_idx, val_idx, df, embargo_days=3, max_horizon_days=5)
    assert_no_leakage(train_idx, test_idx, df, embargo_days=3, max_horizon_days=5)

    # 1. Deliberate direct index overlap
    leaky_train_idx = np.append(train_idx, val_idx[0])
    with pytest.raises(LeakageError, match="Direct index overlap detected"):
        assert_no_leakage(leaky_train_idx, val_idx, df)

    # 2. Deliberate date overlap (sharing a date)
    # Pick a row from val, copy it with a fake new index
    val_date = df.iloc[val_idx[0]]["date"]
    # Force a training row to have that same date
    fake_df = df.copy()
    fake_df.loc[train_idx[0], "date"] = val_date
    with pytest.raises(LeakageError, match="Cross-sectional date overlap detected"):
        assert_no_leakage(train_idx, val_idx, fake_df)

    # 3. Deliberate purging violation: training row with label_end >= val_start
    val_start = df.iloc[val_idx]["date"].min()
    unpurged_df = df.copy()
    # Find a training row observed before val_start, make its label_end reach into val
    train_pre_val = np.where(df["date"] < val_start)[0]
    bad_train_idx = train_pre_val[-1]  # row immediately before val_start
    unpurged_df.loc[bad_train_idx, "label_end"] = val_start + pd.Timedelta(days=5)

    with pytest.raises(LeakageError, match="Purging violation"):
        assert_no_leakage(
            np.append(train_idx, bad_train_idx),
            val_idx,
            unpurged_df,
            label_end_col="label_end",
        )

    # 4. Deliberate embargo violation: training row inside post-eval embargo window
    val_end = df.iloc[val_idx]["date"].max()
    u_dates = np.array(sorted(df["date"].unique()))
    val_end_pos = np.where(u_dates == val_end)[0][0]
    embargo_date = u_dates[val_end_pos + 1]  # date in the embargo window

    embargo_df = df.copy()
    # Find row on embargo_date
    embargo_rows = np.where(df["date"] == embargo_date)[0]
    if len(embargo_rows) > 0:
        bad_embargo_idx = embargo_rows[0]
        with pytest.raises(LeakageError, match="Embargo violation"):
            assert_no_leakage(
                np.append(train_idx, bad_embargo_idx),
                val_idx,
                embargo_df,
                embargo_days=3,
            )


def test_cpcv_combinations_and_paths(synthetic_panel_h5_h21):
    """Verify CPCV combinations count, leakage verification, and backtest path stitching."""
    df = synthetic_panel_h5_h21
    n_groups = 6
    k_test_groups = 2
    embargo_days = 2
    max_horizon_days = 5

    cpcv = CombinatorialPurgedCV(
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        embargo_days=embargo_days,
        max_horizon_days=max_horizon_days,
    )

    # Expected combinations C(6, 2) = 15
    assert cpcv.n_combinations == 15
    # Expected paths C(5, 1) = 5
    assert cpcv.n_paths == 5

    splits = list(cpcv.split(df))
    assert len(splits) == 15

    preds_by_combo = {}
    for train_idx, test_idx, combo_id, test_groups in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert len(test_groups) == k_test_groups

        test_slice = df.iloc[test_idx].copy()
        test_slice["pred"] = test_slice["close"] * 1.01
        preds_by_combo[combo_id] = test_slice

    # Stitch backtest paths
    paths = stitch_backtest_paths(preds_by_combo, n_groups=n_groups, k_test_groups=k_test_groups)
    assert len(paths) == cpcv.n_paths

    for path_id, path_df in paths.items():
        assert not path_df.empty
        # Verify chronological ordering
        assert path_df["date"].is_monotonic_increasing
        # Verify all groups are represented
        assert set(path_df["group_id"].unique()) == set(range(n_groups))


def test_parameter_validation():
    """Verify input argument bounds and error messages."""
    # Negative splits
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        PurgedWalkForward(n_splits=0)

    # Negative embargo
    with pytest.raises(ValueError, match="embargo_days must be >= 0"):
        PurgedWalkForward(n_splits=3, embargo_days=-1)

    # Insufficient dates
    dates = pd.date_range("2024-01-01", periods=10)
    small_df = pd.DataFrame({"date": dates, "ticker": "TCS.NS", "label_end": dates})
    splitter = PurgedWalkForward(
        n_splits=3, min_train_days=10, val_days=5, test_days=5, embargo_days=2
    )
    with pytest.raises(ValueError, match="Insufficient unique dates"):
        list(splitter.split(small_df))

    # CPCV invalid group parameters
    with pytest.raises(ValueError, match="n_groups must be >= 2"):
        CombinatorialPurgedCV(n_groups=1)
    with pytest.raises(ValueError, match="k_test_groups must be in"):
        CombinatorialPurgedCV(n_groups=4, k_test_groups=4)
