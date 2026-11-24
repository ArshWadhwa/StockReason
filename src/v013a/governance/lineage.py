"""StockReason v0.13A — Model Artifact Lineage & Contamination Isolation.

Section J Requirements:
1. Inspects all existing trained model artifacts across project versions.
2. Computes SHA-256 cryptographic hashes.
3. Classifies each artifact into strict governance statuses:
   - VERIFIED_FOR_DIAGNOSTIC_USE
   - RETRAIN_REQUIRED
   - INVALIDATED
   - UNVERIFIED_LINEAGE
   - BASELINE_ONLY
4. Documents contamination causes (unseparated nominal/adjusted targets, unpurged split boundaries, arbitrary conformal clamps).
5. Emits artifacts/v013a/model_artifact_lineage.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger("v013a_lineage")


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


MODEL_ARTIFACTS_MANIFEST = [
    {
        "model_id": "no_change_random_walk",
        "model_type": "Statistical Baseline (Zero-Return No-Change)",
        "artifact_path": "src/v013a/models/baselines.py",
        "training_dataset": "None (Non-parametric constant)",
        "training_cutoff": "N/A",
        "target_definition": "R_nominal = 0.0",
        "status": "BASELINE_ONLY",
        "known_defects": "Previously awarded inflated directional accuracy by misclassifying flat forecasts as down predictions.",
        "remediation_status": "CORRECTED_IN_V013A (Abstention model for binary directional calls)",
    },
    {
        "model_id": "historical_median_drift",
        "model_type": "Statistical Baseline (Rolling/Historical Median Return)",
        "artifact_path": "src/v013a/models/baselines.py",
        "training_dataset": "Training partition (< 2023-01-01 strictly purged)",
        "training_cutoff": "2022-12-31",
        "target_definition": "Historical median return strictly preceding origin",
        "status": "BASELINE_ONLY",
        "known_defects": "None when fitted strictly on training data.",
        "remediation_status": "VERIFIED_FOR_DIAGNOSTIC_USE",
    },
    {
        "model_id": "catboost_residual_v011",
        "model_type": "Gradient Boosted Decision Tree (CatBoost Residual)",
        "artifact_path": "artifacts/v0.11/best_residual_engine_v011.joblib",
        "training_dataset": "data/v0.9.1/features/panel_features.parquet",
        "training_cutoff": "2022-12-31 (Unpurged horizon overlap)",
        "target_definition": "target_return_21d (CA-adjusted total return conflated with nominal price return)",
        "status": "RETRAIN_REQUIRED",
        "known_defects": "Trained on targets with unseparated corporate actions and unpurged training boundary overlap.",
        "remediation_status": "REQUIRES_RETRAINING_STAGE_C",
    },
    {
        "model_id": "calibrated_quantile_forecaster_v011",
        "model_type": "Conformal Quantile Multiplier Registry",
        "artifact_path": "artifacts/v0.11/calibrated_quantile_forecaster_v011.joblib",
        "training_dataset": "Calibration partition 2023",
        "training_cutoff": "2023-12-31",
        "target_definition": "target_return_21d",
        "status": "RETRAIN_REQUIRED",
        "known_defects": "Hardcoded clamp max(0.5, ...) forced interval widths to be exactly half raw width.",
        "remediation_status": "REPLACED_BY_V013A_GENUINE_CONFORMAL_ENGINE",
    },
    {
        "model_id": "hybrid_pipeline_v010",
        "model_type": "Hybrid Multi-Stage Pipeline",
        "artifact_path": "artifacts/v0.10/engine_a_hybrid_pipeline.joblib",
        "training_dataset": "data/v0.10/train.parquet",
        "training_cutoff": "2022-12-31",
        "target_definition": "target_return_21d",
        "status": "INVALIDATED",
        "known_defects": "Trained on legacy unquarantined data with known corporate action discrepancy contamination.",
        "remediation_status": "DEPRECATED_HISTORICAL_ONLY",
    },
    {
        "model_id": "engine_b_v092",
        "model_type": "Absolute Price LightGBM Regressor",
        "artifact_path": "artifacts/v0.9.2/engine_b_pipeline.joblib",
        "training_dataset": "data/v0.9.1/features/panel_features.parquet",
        "training_cutoff": "2022-12-31",
        "target_definition": "target_price_21d",
        "status": "INVALIDATED",
        "known_defects": "Trained without horizon-specific label realization purging.",
        "remediation_status": "DEPRECATED_HISTORICAL_ONLY",
    },
    {
        "model_id": "lstm_primary_legacy",
        "model_type": "Deep Learning LSTM Regressor",
        "artifact_path": "artifacts/models/best_lstm.pth",
        "training_dataset": "Legacy data/processed",
        "training_cutoff": "Unrecorded",
        "target_definition": "Unreconciled price return",
        "status": "INVALIDATED",
        "known_defects": "Unverified lineage, no label purging, superseded by modern tabular baselines.",
        "remediation_status": "DEPRECATED_HISTORICAL_ONLY",
    },
    {
        "model_id": "gru_21d_v09",
        "model_type": "Deep Learning GRU",
        "artifact_path": "artifacts/v0.9/models/gru_21d.pth",
        "training_dataset": "data/v0.9/features",
        "training_cutoff": "2022-12-31",
        "target_definition": "target_return_21d",
        "status": "INVALIDATED",
        "known_defects": "Unpurged training boundary, target discrepancy contamination.",
        "remediation_status": "DEPRECATED_HISTORICAL_ONLY",
    },
    {
        "model_id": "econometric_ewma_v013a",
        "model_type": "Recursive Econometric Volatility Estimator (lambda=0.94)",
        "artifact_path": "src/v013a/volatility/engine.py",
        "training_dataset": "Strictly preceding decimal return observations",
        "training_cutoff": "Walk-forward recursive origin",
        "target_definition": "Annualized Realized Variance / Volatility",
        "status": "VERIFIED_FOR_DIAGNOSTIC_USE",
        "known_defects": "Previous v0.12 implementation mistakenly grouped raw closing prices in rupees instead of daily returns.",
        "remediation_status": "CORRECTED_AND_VERIFIED_IN_V013A",
    },
    {
        "model_id": "econometric_har_rv_v013a",
        "model_type": "Heterogeneous Autoregressive Realized Volatility Regressor",
        "artifact_path": "src/v013a/volatility/engine.py",
        "training_dataset": "Training partition (< 2023-01-01 strictly purged)",
        "training_cutoff": "2022-12-31",
        "target_definition": "Forward annualized realized variance V_forward(t, H)",
        "status": "VERIFIED_FOR_DIAGNOSTIC_USE",
        "known_defects": "Previous v0.12 implementation had intercept 276.01 and was clipped to 1.50.",
        "remediation_status": "CORRECTED_AND_VERIFIED_IN_V013A",
    },
    {
        "model_id": "split_conformal_calibrator_v013a",
        "model_type": "Nonparametric Split-Conformal Calibrator",
        "artifact_path": "src/v013a/calibration/conformal.py",
        "training_dataset": "Calibration partition 2023 (strictly purged)",
        "training_cutoff": "2023-12-31",
        "target_definition": "Residual nonconformity scores s_i = |y_i - f(x_i)|",
        "status": "VERIFIED_FOR_DIAGNOSTIC_USE",
        "known_defects": "Previous implementation clamped multipliers to 0.5.",
        "remediation_status": "CORRECTED_AND_VERIFIED_IN_V013A",
    },
]


def build_model_artifact_lineage_registry(
    output_json: str = "artifacts/v013a/model_artifact_lineage.json",
) -> Dict[str, Any]:
    """Generates complete cryptographic lineage registry for all model artifacts."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    registry = {
        "registry_version": "0.13.0",
        "description": "StockReason v0.13A Model Artifact Lineage & Contamination Isolation Registry",
        "total_artifacts_audited": len(MODEL_ARTIFACTS_MANIFEST),
        "status_summary": {},
        "artifacts": [],
    }

    status_counts: Dict[str, int] = {}

    for item in MODEL_ARTIFACTS_MANIFEST:
        path = item["artifact_path"]
        sha = compute_file_sha256(path) if os.path.exists(path) else "SOURCE_FILE_IN_REPO" if path.endswith(".py") else "MISSING"
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        record = dict(item)
        record["sha256"] = sha
        record["file_exists"] = bool(os.path.exists(path))
        registry["artifacts"].append(record)

    registry["status_summary"] = status_counts

    with open(output_json, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Model lineage registry built: %d artifacts audited, summary: %s",
                len(MODEL_ARTIFACTS_MANIFEST), status_counts)
    return registry
