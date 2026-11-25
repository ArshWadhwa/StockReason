"""Canonical Dataset & Research Integrity Verification Tests."""

import json
import os
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import pyarrow.parquet as pq
from src.utils.config import CANONICAL_DATASET_PATH, CANONICAL_SHA256, RESULTS_METRICS_DIR
from src.utils.metrics import compute_file_sha256


def test_canonical_dataset_hash():
    """Verify Canonical dataset SHA-256 hash."""
    assert CANONICAL_DATASET_PATH.exists(), f"Canonical dataset missing at {CANONICAL_DATASET_PATH}"
    actual_hash = compute_file_sha256(CANONICAL_DATASET_PATH)
    assert actual_hash == CANONICAL_SHA256, f"Hash mismatch: {actual_hash} != {CANONICAL_SHA256}"


def test_feature_family_integration_audit():
    """Verify feature family populated vs unpopulated counts."""
    audit_path = RESULTS_METRICS_DIR / "feature_family_audit.csv"
    if not audit_path.exists():
        audit_path = Path("reports/v013c_feature_family_audit.csv")

    assert os.path.exists(audit_path), f"Feature family audit missing at {audit_path}"
    df = pd.read_csv(audit_path)

    assert len(df) == 80, f"Expected 80 registered features, got {len(df)}"
    pop_counts = df.groupby(["family", "populated"]).size().unstack(fill_value=0)

    assert pop_counts.loc["F0", True] == 17
    assert pop_counts.loc["F1", True] == 7
    assert pop_counts.loc["F2", True] == 5
    assert pop_counts.loc["F3", True] == 20
    assert pop_counts.loc["F4", True] == 0, "F4 should have 0 populated features"
    assert pop_counts.loc["F5", True] == 0, "F5 should have 0 populated features"
    assert pop_counts.loc["F6", True] == 19
    assert pop_counts[True].sum() == 68, "Expected 68 total populated features"


def test_baseline_reproduction():
    """Verify exact matched no-change baseline reproduction on 2024-2026 cohort."""
    summary_path = RESULTS_METRICS_DIR / "model_comparison.json"
    if not summary_path.exists():
        summary_path = Path("artifacts/v013c/v013c_predictive_experiment_summary.json")

    assert os.path.exists(summary_path), f"Experiment summary missing at {summary_path}"
    with open(summary_path) as f:
        data = json.load(f)

    nc = data["no_change_reference"]
    assert nc["eval_observations"] == 836297
    assert abs(nc["true_price_mape"] - 8.7648) < 0.001
    assert abs(nc["inr_price_mae"] - 103.404) < 0.01


def test_experiment_summary_artifacts():
    """Verify experiment summary CSV and parquet predictions exist and are valid."""
    csv_path = RESULTS_METRICS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        csv_path = Path("artifacts/v013c/v013c_predictive_experiment_summary.csv")
    assert os.path.exists(csv_path)
    df_sum = pd.read_csv(csv_path)
    assert len(df_sum) == 15, f"Expected 15 models/baselines, got {len(df_sum)}"

    parquet_path = Path("results/predictions/predictions.parquet")
    if not parquet_path.exists():
        parquet_path = Path("artifacts/v013c/v013c_evaluation_predictions.parquet")
    assert os.path.exists(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    assert pf.metadata.num_rows == 836297
