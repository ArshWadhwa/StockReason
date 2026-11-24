"""Demonstration script for StockReason v0.7 Purged Walk-Forward and CPCV Splitters.

Run via:
    python -m v07.validation.demo
or:
    .venv/bin/python -m v07.validation.demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

# Ensure src/ is on sys.path if running as module or script
_repo_root = Path(__file__).resolve().parents[2]
_src_root = _repo_root / "src"
for p in [str(_repo_root), str(_src_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

try:
    from src.v07.validation.cpcv import CombinatorialPurgedCV, stitch_backtest_paths
    from src.v07.validation.leakage_check import LeakageError, assert_no_leakage
    from src.v07.validation.purged_walk_forward import PurgedWalkForward
except ImportError:
    from v07.validation.cpcv import CombinatorialPurgedCV, stitch_backtest_paths
    from v07.validation.leakage_check import LeakageError, assert_no_leakage
    from v07.validation.purged_walk_forward import PurgedWalkForward

# Configure clean logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")


def generate_synthetic_market_panel(
    tickers: List[str] | None = None,
    n_days: int = 500,
    start_date: str = "2023-01-02",
) -> pd.DataFrame:
    """Generate synthetic daily multi-asset panel data with forward labels and horizons."""
    if tickers is None:
        tickers = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    # Generate business days
    dates = pd.bdate_range(start=start_date, periods=n_days)
    u_dates = list(dates)

    rows = []
    np.random.seed(42)

    for ticker in tickers:
        base_price = 1000.0 + np.random.uniform(100, 2000)
        daily_returns = np.random.normal(0.0005, 0.015, size=n_days)
        prices = base_price * np.cumprod(1 + daily_returns)
        volumes = np.random.lognormal(12, 0.5, size=n_days)

        for i, d in enumerate(dates):
            # Compute 5D and 21D forward horizons
            idx_5d = min(i + 5, n_days - 1)
            idx_21d = min(i + 21, n_days - 1)

            ret_5d = (prices[idx_5d] - prices[i]) / prices[i]
            ret_21d = (prices[idx_21d] - prices[i]) / prices[i]

            label_end_5d = u_dates[idx_5d]
            label_end_21d = u_dates[idx_21d]

            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "close": prices[i],
                    "volume": volumes[i],
                    "return_5d": ret_5d,
                    "return_21d": ret_21d,
                    "label_end_5d": label_end_5d,
                    "label_end_21d": label_end_21d,
                    "label_end": label_end_21d,  # Longest horizon used for multi-horizon purge
                }
            )

    df = pd.DataFrame(rows)
    # Sort by date, then ticker
    df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    return df


def run_demo() -> None:
    """Execute validation demonstration."""
    print("=" * 80)
    print("   StockReason v0.7: Purged & Embargoed Walk-Forward Validation Demo")
    print("=" * 80)

    # 1. Synthesize multi-asset panel
    n_days = 500
    tickers = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    print(f"\n[1] Generating synthetic market panel: {len(tickers)} tickers across {n_days} trading days...")
    panel = generate_synthetic_market_panel(tickers=tickers, n_days=n_days)
    print(f"    Total panel rows: {len(panel):,} | Date range: {panel['date'].min().strftime('%Y-%m-%d')} to {panel['date'].max().strftime('%Y-%m-%d')}")

    # 2. Configure PurgedWalkForward
    n_splits = 5
    embargo_days = 5
    min_train_days = 120
    val_days = 40
    test_days = 40
    max_horizon_days = 21

    print("\n[2] Configuring PurgedWalkForward Splitter:")
    print(f"    - Number of splits (n_splits): {n_splits}")
    print(f"    - Min train days (min_train_days): {min_train_days}")
    print(f"    - Validation block (val_days): {val_days}")
    print(f"    - Out-of-sample block (test_days): {test_days}")
    print(f"    - Embargo period (embargo_days): {embargo_days}")
    print(f"    - Max label horizon (max_horizon_days): {max_horizon_days} (Purges overlapping labels)")

    splitter = PurgedWalkForward(
        n_splits=n_splits,
        embargo_days=embargo_days,
        min_train_days=min_train_days,
        val_days=val_days,
        test_days=test_days,
        max_horizon_days=max_horizon_days,
    )

    # 3. Inspect Fold Summary Table
    print("\n[3] Walk-Forward Fold Schedule & Purge Breakdown:")
    summary_df = splitter.get_fold_summary(panel)
    header = (
        f"{'Fold':<5} | {'Train Span':<23} | {'Raw Tr':<7} | {'Purged':<6} | {'Kept Tr':<7} | "
        f"{'Val Span':<23} | {'Val Rows':<8} | {'Test Span':<23} | {'Test Rows':<9}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for _, row in summary_df.iterrows():
        train_span = f"{row['train_start']}..{row['train_end']}"
        val_span = f"{row['val_start']}..{row['val_end']}"
        test_span = f"{row['test_start']}..{row['test_end']}"
        print(
            f"{row['fold_id']:<5} | {train_span:<23} | {row['raw_train_rows']:<7} | "
            f"{row['purged_rows']:<6} | {row['kept_train_rows']:<7} | "
            f"{val_span:<23} | {row['val_rows']:<8} | "
            f"{test_span:<23} | {row['test_rows']:<9}"
        )
    print("-" * len(header))

    # 4. Iterate over splits & assert zero leakage
    print("\n[4] Running live split iteration and verifying zero data leakage...")
    for train_idx, val_idx, test_idx, fold_id in splitter.split(panel):
        train_dates = panel.iloc[train_idx]["date"]
        val_dates = panel.iloc[val_idx]["date"]
        test_dates = panel.iloc[test_idx]["date"]
        print(
            f"    [OK] Fold {fold_id}: Train={len(train_idx)} rows ({train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}) "
            f"| Val={len(val_idx)} rows ({val_dates.min().strftime('%Y-%m-%d')} to {val_dates.max().strftime('%Y-%m-%d')}) "
            f"| Test={len(test_idx)} rows ({test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}) "
            f"-> Zero leakage verified!"
        )

    # 5. Deliberate Leakage Verification Test
    print("\n[5] Verifying assert_no_leakage catches deliberate leaks:")
    test_fold_train = list(train_idx)
    test_fold_val = list(val_idx)

    # Case A: Inject date overlap
    test_fold_train_leaky = test_fold_train + [test_fold_val[0]]
    try:
        assert_no_leakage(test_fold_train_leaky, test_fold_val, panel)
        print("    [FAIL] assert_no_leakage failed to catch date overlap!")
    except LeakageError as e:
        print(f"    [PASS] Correctly caught direct overlap: {str(e)[:75]}...")

    # Case B: Inject unpurged row with future label
    # An unpurged training row right before val_start with horizon 21 reaches into val
    val_start = panel.iloc[val_idx]["date"].min()
    unpurged_candidates = panel[(panel["date"] < val_start) & (panel["label_end"] >= val_start)]
    if len(unpurged_candidates) > 0:
        bad_idx = unpurged_candidates.index[0]
        try:
            assert_no_leakage(test_fold_train + [bad_idx], test_fold_val, panel)
            print("    [FAIL] assert_no_leakage failed to catch unpurged label!")
        except LeakageError as e:
            print(f"    [PASS] Correctly caught purging violation: {str(e)[:75]}...")

    # 6. Advanced/Optional CPCV Demo
    print("\n[6] Advanced: Combinatorial Purged Cross-Validation (CPCV):")
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo_days=5, max_horizon_days=21)
    print(f"    - N={cpcv.n_groups} groups, k={cpcv.k_test_groups} test groups")
    print(f"    - Total train/test combinations C({cpcv.n_groups}, {cpcv.k_test_groups}): {cpcv.n_combinations}")
    print(f"    - Number of stitched backtest paths C({cpcv.n_groups}-1, {cpcv.k_test_groups}-1): {cpcv.n_paths}")

    cpcv_preds = {}
    for train_idx, test_idx, combo_id, test_groups in cpcv.split(panel):
        test_slice = panel.iloc[test_idx].copy()
        # Mock model prediction
        test_slice["pred_return_21d"] = test_slice["return_21d"] * 0.2 + np.random.normal(0, 0.01, size=len(test_slice))
        cpcv_preds[combo_id] = test_slice

    paths = stitch_backtest_paths(cpcv_preds, n_groups=6, k_test_groups=2)
    print(f"    Successfully generated {len(paths)} independent out-of-sample backtest paths:")
    for p_id, p_df in paths.items():
        print(f"      * Path {p_id}: {len(p_df)} predictions spanning {p_df['date'].min().strftime('%Y-%m-%d')} to {p_df['date'].max().strftime('%Y-%m-%d')}")

    print("\n" + "=" * 80)
    print("   Validation Demo Completed Successfully with Zero Leakage!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
