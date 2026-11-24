"""
StockReason v0.7 - Workstream 10 Test Suite
============================================
Comprehensive test suite covering:
  1. Metrics library (per-date cross-sectional IC, Rank IC, ICIR, MAE, RMSE, Brier, breakdowns)
  2. Report generator (Markdown & JSON output, bootstrap CI, edge verdict)
  3. Model promotion gate (rejection of inferior models, acceptance of qualifying models, registry JSON)
  4. Drift monitor v2 (rolling Rank IC, conformal coverage collapse, feature PSI drift)
  5. Pipeline orchestration & freshness enforcement (fail loudly on stale data, archive creation)
  6. Backend API integration (new v0.7 fields, /api/model/report, updated /api/system/status)
  7. Frontend build verification test (tsc -b && vite build)
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, date, timedelta
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.v07.reporting.metrics import (
    compute_metrics,
    compute_per_date_ic,
    compute_bootstrap_ci,
    compute_calibration_metrics,
)
from src.v07.reporting.make_report import (
    make_report,
    evaluate_model_performance,
    compute_net_backtest_summary,
)
from src.v07.reporting.promote_model import (
    evaluate_promotion_gate,
    promote_model,
    load_model_registry,
)
from src.v07.reporting.drift import (
    calculate_psi,
    compute_rolling_rank_ic,
    compute_rolling_coverage,
    check_drift,
)
from src.v07.reporting.pipeline import (
    check_data_freshness,
    run_v07_pipeline,
)
from src.m4_dashboard.backend.main import app

ROOT_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic_pred_df():
    """Generates synthetic multi-day cross-sectional prediction fixture."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    records = []
    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        for s in stocks:
            true_ret = float(rng.normal(0.001, 0.02))
            pred_ret = float(0.3 * true_ret + rng.normal(0.0, 0.015))
            prob_up = float(1.0 / (1.0 + np.exp(-pred_ret * 40.0)))
            records.append({
                "date": d_str,
                "ticker": s,
                "y_pred": pred_ret,
                "y_true": true_ret,
                "y_prob": prob_up,
                "regime": "TRENDING" if d.day % 2 == 0 else "VOLATILE",
                "sector": "IT" if s in ["TCS.NS", "INFY.NS"] else "Banking",
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 1. Metrics Library Tests
# ─────────────────────────────────────────────────────────────
class TestMetricsLibrary:
    def test_compute_metrics_keys_and_values(self, synthetic_pred_df):
        metrics = compute_metrics(synthetic_pred_df)
        required_keys = {
            "ic", "rank_ic", "icir", "rank_icir", "mae", "rmse",
            "dir_acc", "dir_acc_vs_base", "brier", "n"
        }
        assert required_keys.issubset(metrics.keys()), f"Missing keys: {required_keys - set(metrics.keys())}"

        assert metrics["n"] == len(synthetic_pred_df)
        assert metrics["n_dates"] == 25
        assert isinstance(metrics["rank_ic"], float)
        assert isinstance(metrics["rank_ic_ci_95"], (tuple, list))
        assert len(metrics["rank_ic_ci_95"]) == 2
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert 0.0 <= metrics["dir_acc"] <= 1.0
        assert metrics["brier"] is not None

    def test_per_date_cross_sectional_ic(self, synthetic_pred_df):
        per_date = compute_per_date_ic(synthetic_pred_df, pred_col="y_pred", target_col="y_true", date_col="date")
        assert not per_date.empty
        assert len(per_date) == 25
        assert "rank_ic" in per_date.columns
        assert "ic" in per_date.columns
        # Verify cross-sectional averaging matches
        mean_rank_ic = per_date["rank_ic"].mean()
        metrics = compute_metrics(synthetic_pred_df)
        assert pytest.approx(metrics["rank_ic"], abs=1e-4) == mean_rank_ic

    def test_calibration_metrics(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.85, 0.15, 0.65, 0.25])
        cal = compute_calibration_metrics(y_true, y_prob, n_bins=5)
        assert "ece" in cal
        assert "brier" in cal
        assert cal["ece"] < 0.25
        assert cal["brier"] < 0.15

    def test_bootstrap_ci(self):
        vals = np.random.normal(0.05, 0.02, 100)
        low, high = compute_bootstrap_ci(vals, n_bootstraps=500, ci=0.95, seed=42)
        assert low < high
        assert low > 0.0  # Positive distribution


# ─────────────────────────────────────────────────────────────
# 2. Evaluation Report Generator Tests
# ─────────────────────────────────────────────────────────────
class TestReportGenerator:
    def test_make_report_synthetic(self, synthetic_pred_df, tmp_path):
        out_dir = tmp_path / "reports_test"
        md_path, json_path, data = make_report(
            model_name="test_candidate",
            pred_df=synthetic_pred_df,
            output_dir=out_dir,
            as_of_date="2026-11-24",
            seed=42,
        )

        assert md_path.exists()
        assert json_path.exists()

        # Check JSON report
        with open(json_path, "r", encoding="utf-8") as f:
            report_json = json.load(f)

        assert report_json["model_name"] == "test_candidate"
        assert "verdict" in report_json
        assert "Edge demonstrated" in report_json["verdict"] or "No demonstrated edge" in report_json["verdict"]
        assert "config_hash" in report_json
        assert "git_commit" in report_json
        assert "data_window" in report_json
        assert "metrics" in report_json
        assert "backtest" in report_json
        assert "calibration" in report_json

        # Check Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        assert "# StockReason v0.7 Model Evaluation Report: `test_candidate`" in md_text
        assert "Rank IC (Spearman)" in md_text
        assert "Net Annualized Sharpe Ratio" in md_text
        assert "Expected Calibration Error" in md_text


# ─────────────────────────────────────────────────────────────
# 3. Model Promotion Gate Tests
# ─────────────────────────────────────────────────────────────
class TestPromotionGate:
    def test_promotion_gate_qualifying_model(self, tmp_path):
        candidate_report = {
            "model_name": "candidate_alpha",
            "git_commit": "abc1234",
            "config_hash": "cfg_alpha",
            "metrics": {
                "rank_ic": 0.09,
                "rank_icir": 2.1,
                "rank_ic_ci_95": (0.03, 0.15),
                "rank_ic_pval": 0.005,
            },
            "backtest": {"sharpe": 1.6, "max_drawdown": 0.08},
            "calibration": {"ece": 0.04, "brier": 0.21},
            "baseline_metrics": {"rank_ic": 0.02},
            "verdict": "Edge demonstrated: Rank IC 0.0900 (CI 0.0300 to 0.1500)",
        }
        incumbent_report = {
            "model_name": "production_incumbent",
            "metrics": {"rank_ic": 0.05, "rank_ic_ci_95": (0.01, 0.09)},
            "backtest": {"sharpe": 1.2},
            "calibration": {"ece": 0.05},
        }

        gate_res = evaluate_promotion_gate(candidate_report, incumbent_report=incumbent_report)
        assert gate_res["promoted"] is True
        assert gate_res["checks"]["beats_incumbent_rank_ic"] is True
        assert gate_res["checks"]["beats_baseline_rank_ic"] is True
        assert gate_res["checks"]["no_sharpe_degradation"] is True
        assert gate_res["checks"]["calibration_within_tolerance"] is True

        # Test persistence in registry
        reg_path = tmp_path / "model_registry.json"
        promoted, _ = promote_model(candidate_report, registry_path=reg_path)
        assert promoted is True

        reg = load_model_registry(reg_path)
        assert reg["active_production"]["model_name"] == "candidate_alpha"

    def test_promotion_gate_rejects_inferior_model(self, tmp_path):
        # Worse model with negative Rank IC and degraded Sharpe
        worse_report = {
            "model_name": "candidate_worse",
            "metrics": {
                "rank_ic": -0.01,
                "rank_icir": -0.2,
                "rank_ic_ci_95": (-0.08, 0.06),
                "rank_ic_pval": 0.45,
            },
            "backtest": {"sharpe": -0.4, "max_drawdown": 0.25},
            "calibration": {"ece": 0.18, "brier": 0.35},
            "baseline_metrics": {"rank_ic": 0.03},
        }
        incumbent_report = {
            "model_name": "production_champion",
            "metrics": {"rank_ic": 0.06, "rank_ic_ci_95": (0.02, 0.10)},
            "backtest": {"sharpe": 1.3},
            "calibration": {"ece": 0.05},
        }

        gate_res = evaluate_promotion_gate(worse_report, incumbent_report=incumbent_report)
        assert gate_res["promoted"] is False
        assert gate_res["checks"]["beats_incumbent_rank_ic"] is False
        assert gate_res["checks"]["beats_baseline_rank_ic"] is False
        assert gate_res["checks"]["no_sharpe_degradation"] is False
        assert gate_res["checks"]["calibration_within_tolerance"] is False
        assert len(gate_res["reasons"]) > 0

        # Verify rejection in registry does NOT replace active production
        reg_path = tmp_path / "model_registry.json"
        # Seed incumbent first
        promote_model({
            "model_name": "incumbent",
            "metrics": {"rank_ic": 0.06, "rank_ic_ci_95": (0.02, 0.10)},
            "backtest": {"sharpe": 1.3},
            "calibration": {"ece": 0.05},
            "baseline_metrics": {"rank_ic": 0.02},
        }, registry_path=reg_path)

        promoted, _ = promote_model(worse_report, registry_path=reg_path)
        assert promoted is False
        reg = load_model_registry(reg_path)
        assert reg["active_production"]["model_name"] == "incumbent"


# ─────────────────────────────────────────────────────────────
# 4. Drift Monitor Tests
# ─────────────────────────────────────────────────────────────
class TestDriftMonitor:
    def test_psi_calculation(self):
        ref = np.random.normal(10, 2, 500)
        same = np.random.normal(10, 2, 500)
        shifted = np.random.normal(16, 2, 500)

        psi_same = calculate_psi(ref, same)
        psi_shifted = calculate_psi(ref, shifted)

        assert psi_same < 0.10  # Stable
        assert psi_shifted > 0.25  # Significant drift

    def test_conformal_coverage_collapse_alert(self):
        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        records = []
        for d in dates:
            for s in ["S1", "S2", "S3"]:
                records.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "ticker": s,
                    "y_pred": 0.01,
                    "y_true": 0.05,
                    "interval_lower_80": 0.00,
                    "interval_upper_80": 0.02,  # y_true falls outside interval (0% coverage)
                })
        df = pd.DataFrame(records)
        res = check_drift(df, window=10)
        assert res["status"] == "ALERT"
        assert any("Coverage Collapse" in a for a in res["alerts"])


# ─────────────────────────────────────────────────────────────
# 5. Pipeline Orchestration & Freshness Tests
# ─────────────────────────────────────────────────────────────
class TestPipelineOrchestration:
    def test_freshness_fails_loudly_on_stale_data(self, tmp_path):
        stale_csv = tmp_path / "stale_price.csv"
        # Old data from 2020
        df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "ticker": ["RELIANCE.NS", "RELIANCE.NS"]})
        df.to_csv(stale_csv, index=False)

        is_fresh, msg, _ = check_data_freshness(stale_csv, max_stale_days=5, reference_date=date(2026, 11, 24))
        assert is_fresh is False
        assert "CRITICAL FRESHNESS FAILURE" in msg

    def test_run_v07_pipeline_creates_archive(self, tmp_path):
        summary = run_v07_pipeline(
            max_stale_days=10,
            as_of_date="2026-11-23",
            output_dir=tmp_path / "v07_out",
        )
        assert summary["freshness_status"] == "PASSED"
        assert summary["stocks_evaluated"] > 0
        archive_path = Path(summary["archive_dir"])
        assert archive_path.exists()
        assert (archive_path / "predictions.json").exists()
        assert (archive_path / "signals.json").exists()


# ─────────────────────────────────────────────────────────────
# 6. Backend API Integration Tests
# ─────────────────────────────────────────────────────────────
class TestBackendAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_system_status_endpoint(self, client):
        res = client.get("/api/system/status")
        assert res.status_code == 200
        data = res.json()
        assert "data_as_of" in data
        assert "model_version" in data
        assert "rolling_oos_rank_ic" in data
        assert "edge_verdict" in data
        assert "is_stale" in data

    def test_model_report_endpoint(self, client):
        res = client.get("/api/model/report")
        assert res.status_code == 200
        data = res.json()
        assert "verdict" in data
        assert "metrics" in data
        assert "rank_ic" in data["metrics"]
        assert "calibration" in data

    def test_stock_signal_new_v07_fields(self, client):
        res = client.get("/api/stocks/RELIANCE.NS/signal")
        assert res.status_code == 200
        data = res.json()
        assert "prob_up_calibrated" in data
        assert "interval_80" in data
        assert "invalidation_conditions" in data
        assert isinstance(data["invalidation_conditions"], list)

    def test_stock_predictions_new_v07_fields(self, client):
        res = client.get("/api/stocks/RELIANCE.NS/predictions")
        assert res.status_code == 200
        data = res.json()
        assert "prob_up_calibrated" in data
        assert "interval_80" in data


# ─────────────────────────────────────────────────────────────
# 7. Frontend Build Verification Test
# ─────────────────────────────────────────────────────────────
class TestFrontendBuild:
    def test_frontend_build_succeeds(self):
        frontend_dir = ROOT_DIR / "frontend"
        if not (frontend_dir / "node_modules").exists():
            pytest.skip("frontend/node_modules not found; skipping build test")

        res = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, f"Frontend build failed:\n{res.stderr}\n{res.stdout}"
