"""Model Portability & Parity Verification Test."""

import os
import json
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
from src.inference.model_loader import ModelLoader
from src.inference.predictor import StockReasonPredictor
from scripts.download_models import download_models, sha256_file, EXPECTED_CHECKSUMS


def test_model_portability_and_parity():
    """
    Verifies that model artifacts can be staged, downloaded into a clean temporary directory,
    re-loaded, and generate numerically identical predictions to pre-migration baselines.
    """
    # 1. Load pre-migration reference fixture
    fixture_path = Path("tests/fixtures/pre_migration_reference.json")
    assert fixture_path.exists(), "Reference fixture missing"
    with open(fixture_path, "r") as f:
        fixture = json.load(f)

    # 2. Check local models
    local_loader = ModelLoader("models")
    lgb_local = local_loader.load_lightgbm()
    assert lgb_local is not None, "Local LightGBM could not be loaded"
    assert len(lgb_local.feature_names) == 21

    cb_local = local_loader.load_catboost()
    assert cb_local is not None, "Local CatBoost could not be loaded"

    # 3. Create clean temporary directory to simulate clean teammate download
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 4. Download into temporary directory
        success = download_models(tmp_path)
        assert success, "Download into clean environment failed"

        # 5. Verify checksums in temporary directory
        for fname, exp_hash in EXPECTED_CHECKSUMS.items():
            assert (tmp_path / fname).exists()
            assert sha256_file(tmp_path / fname) == exp_hash

        # 6. Load downloaded models from clean temp dir
        clean_loader = ModelLoader(tmp_path)
        lgb_clean = clean_loader.load_lightgbm()
        cb_clean = clean_loader.load_catboost()
        assert lgb_clean is not None
        assert cb_clean is not None

        # 7. Test inference parity on sample input
        X_test = np.random.RandomState(42).randn(10, 21).astype(np.float32)

        pred_lgb_local = lgb_local.predict(X_test)
        pred_lgb_clean = lgb_clean.predict(X_test)
        np.testing.assert_allclose(pred_lgb_local, pred_lgb_clean, rtol=1e-7, atol=1e-7)

        pred_cb_local = cb_local.predict(X_test)
        pred_cb_clean = cb_clean.predict(X_test)
        np.testing.assert_allclose(pred_cb_local, pred_cb_clean, rtol=1e-7, atol=1e-7)

    # 8. Verify No-Change baseline numerical reproduction
    nc = fixture.get("no_change_reference", {})
    assert abs(nc.get("true_price_mape", 0.0) - 8.7648) < 0.001
    assert abs(nc.get("inr_price_mae", 0.0) - 103.404) < 0.01
