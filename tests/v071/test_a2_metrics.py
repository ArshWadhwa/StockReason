"""
Unit tests for Workstream A2: Lockbox enforcement & IC metric correctness.
"""

import os
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.v07.common.lockbox import (
    load_panel,
    UNLOCK_ENV_VAR,
    UNLOCK_MAGIC_VALUE,
    LOCKBOX_START_DATE,
    EMBARGO_START_DATE,
)
from src.v071.metrics import (
    compute_ic,
    summarize_ic,
    compute_directional_accuracy,
)


class TestLockboxContract:
    def test_load_panel_excludes_lockbox_by_default(self):
        panel = load_panel(exclude_lockbox=True, exclude_embargo=True)
        assert not panel.empty
        max_date = panel["date"].max()
        # Trainable cutoff should be prior to embargo start
        assert max_date < pd.to_datetime(EMBARGO_START_DATE)

    def test_unlocked_access_without_env_var_raises(self, monkeypatch):
        monkeypatch.delenv(UNLOCK_ENV_VAR, raising=False)
        with pytest.raises(PermissionError, match="LOCKBOX ACCESS DENIED"):
            load_panel(exclude_lockbox=False)

    def test_unlocked_access_with_env_var_succeeds_and_logs(self, monkeypatch, tmp_path):
        monkeypatch.setenv(UNLOCK_ENV_VAR, UNLOCK_MAGIC_VALUE)
        panel = load_panel(exclude_lockbox=False)
        assert not panel.empty
        # Verify log file was written
        log_path = "reports/v0.7.1/lockbox_access.log"
        assert os.path.exists(log_path)
        with open(log_path, "r") as f:
            lines = f.readlines()
        assert any(UNLOCK_MAGIC_VALUE[:10] in line or "caller=" in line for line in lines)


class TestMetricContract:
    @pytest.fixture
    def random_panel(self):
        np.random.seed(42)
        n_dates = 10
        n_stocks = 25
        dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
        records = []
        for d in dates:
            p = np.random.randn(n_stocks)
            t = np.random.randn(n_stocks)
            for i in range(n_stocks):
                records.append({
                    "date": d,
                    "ticker": f"STOCK_{i}",
                    "pred": p[i],
                    "target": t[i],
                })
        return pd.DataFrame(records)

    def test_constant_prediction_yields_exact_zero_ic(self, random_panel):
        df = random_panel.copy()
        df["pred"] = 0.05  # Constant across all stocks and dates
        ic_df = compute_ic(df, pred_col="pred", target_col="target", min_names_per_date=20)
        assert len(ic_df) > 0
        assert np.all(ic_df["ic"] == 0.0)

    def test_perfect_prediction_yields_plus_one(self, random_panel):
        df = random_panel.copy()
        df["pred"] = df["target"]
        ic_df = compute_ic(df, pred_col="pred", target_col="target", method="spearman")
        assert len(ic_df) > 0
        assert np.allclose(ic_df["ic"], 1.0, atol=1e-7)

    def test_reversed_prediction_yields_minus_one(self, random_panel):
        df = random_panel.copy()
        df["pred"] = -df["target"]
        ic_df = compute_ic(df, pred_col="pred", target_col="target", method="spearman")
        assert len(ic_df) > 0
        assert np.allclose(ic_df["ic"], -1.0, atol=1e-7)

    def test_matches_scipy_spearmanr_date_by_date(self, random_panel):
        df = random_panel.copy()
        ic_df = compute_ic(df, pred_col="pred", target_col="target", method="spearman", min_names_per_date=5)
        for _, row in ic_df.iterrows():
            sub = df[df["date"] == row["date"]]
            expected_res = stats.spearmanr(sub["pred"], sub["target"])
            expected = expected_res.statistic if hasattr(expected_res, "statistic") else expected_res[0]
            assert np.isclose(row["ic"], expected, atol=1e-7)

    def test_handles_ties_nans_and_small_universes(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        records = []
        # Date 0: 30 stocks with ties & NaNs
        for i in range(30):
            records.append({
                "date": dates[0],
                "pred": 1.0 if i < 15 else 2.0,
                "target": np.nan if i == 0 else (1.0 if i > 15 else 0.5),
            })
        # Date 1: only 5 stocks (below min_names_per_date=20)
        for i in range(5):
            records.append({"date": dates[1], "pred": float(i), "target": float(i)})
        # Date 2: all NaNs
        for i in range(25):
            records.append({"date": dates[2], "pred": np.nan, "target": 1.0})

        df = pd.DataFrame(records)
        ic_df = compute_ic(df, min_names_per_date=20)
        # Only date 0 should remain (date 1 dropped for n<20, date 2 dropped for NaNs)
        assert len(ic_df) == 1
        assert ic_df.iloc[0]["date"] == dates[0]
        assert not np.isnan(ic_df.iloc[0]["ic"])

    def test_directional_accuracy_base_rate(self):
        # Panel where target is positive 70% of the time
        np.random.seed(42)
        n = 1000
        targets = np.where(np.random.rand(n) < 0.70, 0.05, -0.05)
        # Constant zero prediction
        df_zero = pd.DataFrame({"pred": np.zeros(n), "target": targets})
        res_zero = compute_directional_accuracy(df_zero)
        # Zero prediction (pred > 0 is all False) matches exactly the down moves (1 - pos_base_rate)
        assert np.isclose(res_zero["dir_acc"], 100.0 - res_zero["pos_base_rate"], atol=1e-5)
        assert np.isclose(res_zero["base_rate"], res_zero["pos_base_rate"], atol=1e-5)

        # Constant up prediction matches exactly the positive moves
        df_up = pd.DataFrame({"pred": np.ones(n), "target": targets})
        res_up = compute_directional_accuracy(df_up)
        assert np.isclose(res_up["dir_acc"], res_up["pos_base_rate"], atol=1e-5)
        assert np.isclose(res_up["base_rate"], res_up["pos_base_rate"], atol=1e-5)
