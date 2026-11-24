"""
Unit tests for Workstream A8: Promotion gate and Lockbox protocol security.
"""

import os
import pytest
from src.v071.honest_statistics import evaluate_mechanical_verdict
from src.v071.lockbox_evaluate import run_lockbox_evaluation, verify_lockbox_purity
from src.v07.common.lockbox import UNLOCK_ENV_VAR, UNLOCK_MAGIC_VALUE, get_config_hash


def test_gate_rejects_model_with_ci_including_zero():
    # CatBoost CI spans zero
    cb_ci = (-0.0056, 0.0126)
    verdict = evaluate_mechanical_verdict(cb_ci, "catboost", "21D")
    assert "No demonstrated edge" in verdict


def test_lockbox_refuses_when_hash_mismatches():
    with pytest.raises(ValueError, match="Config hash mismatch"):
        run_lockbox_evaluation(
            frozen_config_path="config/v071.yaml",
            expected_config_hash="DEADBEEF12345678",
        )


def test_lockbox_refuses_without_unlock_env_var(monkeypatch):
    monkeypatch.delenv(UNLOCK_ENV_VAR, raising=False)
    with pytest.raises(PermissionError):
        run_lockbox_evaluation(
            frozen_config_path="config/v071.yaml",
            expected_config_hash=None,
        )
