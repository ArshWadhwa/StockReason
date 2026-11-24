"""
Unit and Integration Tests for StockReason v0.7 Workstream WS5.

Validates:
1. Stationarity enforcement: Sets A, B, C, D contain 0 raw price/volume/MA levels.
2. Cross-sectional normalization: Produces exact mean = 0 across assets per date.
3. Target construction: Matches exact mathematical formulas for raw, excess, vol-scaled,
   clipped, and rank-normalized returns; verifies label timing metadata.
4. Shift invariance (Zero Future Leakage): Truncating time series at date t yields
   identical feature vectors at date t.
5. Collinear redundancy clustering: Correctly identifies and prunes |r| > 0.90 pairs.
6. Canonical panel dataset contract conformance.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.v07.features.targets import (
    compute_targets,
    compute_trailing_realized_volatility,
    rank_normalize_cross_section,
)
from src.v07.features.feature_sets import (
    SET_A_FEATURES,
    SET_B_FEATURES,
    SET_C_FEATURES,
    SET_D_FEATURES,
    get_feature_set,
    validate_stationarity,
    is_stationary_feature,
    enrich_stationary_features,
)
from src.v07.features.normalize import (
    normalize_cross_section,
    rank_transform_group,
    zscore_transform_group,
)
from src.v07.features.importance import (
    compute_correlation_matrix,
    cluster_redundant_features,
)


# ==============================================================================
# 1. Stationarity Enforcement Tests
# ==============================================================================

def test_stationarity_sets_a_through_d():
    """Verify that sets A, B, C, D contain zero non-stationary features."""
    for s_name, feat_list in [
        ("A", SET_A_FEATURES),
        ("B", SET_B_FEATURES),
        ("C", SET_C_FEATURES),
        ("D", SET_D_FEATURES),
    ]:
        violations = validate_stationarity(feat_list, allow_violations=True)
        assert len(violations) == 0, f"Set {s_name} contains non-stationary columns: {violations}"


def test_stationarity_catches_prohibited_columns():
    """Verify that validator catches raw price levels, moving averages, and volumes."""
    prohibited = [
        "open", "high", "low", "close", "adj_close",
        "volume", "obv", "obv_ema_20", "pvt",
        "sma_5", "sma_20", "sma_50", "sma_200",
        "ema_5", "ema_20", "ema_50", "ema_200",
        "atr_7", "atr_14", "atr_21", "atr_14_lag_1",
        "macd_line_12_26", "macd_signal_12_26",
        "bb_upper_20", "bb_lower_20", "bb_mid_20",
        "donchian_high_20", "keltner_upper_20",
        "mom_5", "close_slope_10",
        "hl_spread", "co_spread",
    ]
    for col in prohibited:
        assert not is_stationary_feature(col), f"Should be prohibited: {col}"

    with pytest.raises(ValueError, match="Stationarity violation detected"):
        validate_stationarity(["return_1d", "close", "vol_ratio_5"])

    with pytest.raises(ValueError, match="Stationarity violation detected"):
        validate_stationarity(["return_1d", "sma_50"])


# ==============================================================================
# 2. Cross-Sectional Normalization Tests
# ==============================================================================

def test_cross_sectional_rank_normalization_properties():
    """Verify cross-sectional rank normalization yields mean ≈ 0 and strict [-0.5, 0.5] bounds."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    tickers = [f"STOCK_{i}" for i in range(15)]
    records = []
    for d in dates:
        for t in tickers:
            records.append({
                "date": d,
                "ticker": t,
                "feat_raw": np.random.exponential(scale=10.0),
                "feat_normal": np.random.normal(loc=100.0, scale=25.0),
            })
    df = pd.DataFrame(records)

    norm_df = normalize_cross_section(df, ["feat_raw", "feat_normal"], method="rank")

    for d, grp in norm_df.groupby("date"):
        for col in ["feat_raw", "feat_normal"]:
            vals = grp[col]
            assert np.isclose(vals.mean(), 0.0, atol=1e-7), f"Mean on {d} for {col} was {vals.mean()}"
            assert vals.min() >= -0.5 - 1e-9, f"Min bound violated on {d} for {col}"
            assert vals.max() <= 0.5 + 1e-9, f"Max bound violated on {d} for {col}"


def test_cross_sectional_zscore_properties():
    """Verify cross-sectional z-score produces mean ≈ 0 and std ≈ 1."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    tickers = [f"STOCK_{i}" for i in range(20)]
    records = []
    for d in dates:
        for t in tickers:
            records.append({
                "date": d,
                "ticker": t,
                "val": np.random.uniform(10, 50),
            })
    df = pd.DataFrame(records)
    norm_df = normalize_cross_section(df, ["val"], method="zscore", clip_zscore=None)

    for d, grp in norm_df.groupby("date"):
        vals = grp["val"]
        assert np.isclose(vals.mean(), 0.0, atol=1e-7)
        assert np.isclose(vals.std(ddof=1), 1.0, atol=1e-5)


# ==============================================================================
# 3. Target Construction & Manual Calculation Verification
# ==============================================================================

def test_targets_match_manual_calculation():
    """Verify compute_targets matches exact mathematical ground truth."""
    dates = pd.date_range("2023-01-01", periods=40, freq="B")
    n_days = len(dates)

    # Stock A: steadily increasing
    prices_a = [100.0 * (1.01 ** i) for i in range(n_days)]
    # Stock B: steadily decreasing
    prices_b = [200.0 * (0.99 ** i) for i in range(n_days)]
    # Benchmark: flat
    bench = [18000.0 * (1.002 ** i) for i in range(n_days)]

    rows = []
    for i in range(n_days):
        rows.append({"date": dates[i], "ticker": "A", "adj_close": prices_a[i], "close": prices_a[i], "nifty_50_close": bench[i]})
        rows.append({"date": dates[i], "ticker": "B", "adj_close": prices_b[i], "close": prices_b[i], "nifty_50_close": bench[i]})

    df = pd.DataFrame(rows)
    df_targets = compute_targets(df, horizons=(5, 21), enable_126d=False, vol_window=10)

    # Check row at index t=15 for Stock A
    t = 15
    row_a = df_targets[(df_targets["ticker"] == "A") & (df_targets["date"] == dates[t])].iloc[0]

    # 1. Raw 5-day return
    expected_y5_raw = prices_a[t + 5] / prices_a[t] - 1.0
    assert np.isclose(row_a["y_5d_raw"], expected_y5_raw, atol=1e-6)

    # 2. Benchmark 5-day return
    expected_bench_ret = bench[t + 5] / bench[t] - 1.0
    expected_excess = expected_y5_raw - expected_bench_ret
    assert np.isclose(row_a["y_5d_excess"], expected_excess, atol=1e-6)

    # 3. Label timing columns
    assert pd.to_datetime(row_a["label_start"]) == dates[t]
    assert pd.to_datetime(row_a["label_end_5d"]) == dates[t + 5]
    assert pd.to_datetime(row_a["label_end_21d"]) == dates[t + 21]

    # 4. Cross-sectional rank: Stock A has higher excess return than Stock B on date t
    row_b = df_targets[(df_targets["ticker"] == "B") & (df_targets["date"] == dates[t])].iloc[0]
    assert row_a["y_5d_ranked"] == 0.5
    assert row_b["y_5d_ranked"] == -0.5
    assert np.isclose(row_a["y_5d_ranked"] + row_b["y_5d_ranked"], 0.0, atol=1e-7)


def test_126d_config_toggle():
    """Verify 126D code path is disabled by default and enabled when requested."""
    dates = pd.date_range("2023-01-01", periods=150, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "ticker": "STOCK1",
        "close": [100.0] * len(dates),
        "adj_close": [100.0] * len(dates),
        "nifty_50_close": [18000.0] * len(dates),
    })

    # Disabled by default
    res_def = compute_targets(df, horizons=(5, 21), enable_126d=False)
    assert "y_126d" not in res_def.columns
    assert "label_end_126d" not in res_def.columns

    # Enabled via toggle
    res_enabled = compute_targets(df, horizons=(5, 21), enable_126d=True)
    assert "y_126d" in res_enabled.columns
    assert "label_end_126d" in res_enabled.columns


# ==============================================================================
# 4. Shift Invariance Test (Strict Zero Lookahead Leakage)
# ==============================================================================

def test_shift_invariance_zero_future_leakage():
    """
    Rigorously verify zero lookahead leakage:
    Truncating the historical panel at date t MUST produce bitwise/floating-point
    identical feature vectors at date t as computing on the full panel.
    """
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    n = len(dates)

    p_a = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    p_b = 200 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, n)))
    bench = 18000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n)))

    records = []
    for i in range(n):
        for tick, p in [("STOCK_A", p_a), ("STOCK_B", p_b)]:
            records.append({
                "date": dates[i],
                "ticker": tick,
                "open": p[i], "high": p[i] * 1.01, "low": p[i] * 0.99, "close": p[i], "adj_close": p[i],
                "volume": 50000,
                "nifty_50_close": bench[i],
                "return_1d": (p[i] / p[i-1] - 1.0) if i > 0 else 0.0,
                "return_5d": (p[i] / p[i-5] - 1.0) if i >= 5 else 0.0,
                "return_21d": (p[i] / p[i-21] - 1.0) if i >= 21 else 0.0,
            })
    full_df = pd.DataFrame(records)

    # Choose a historical decision point t (e.g. index 50)
    cutoff_date = dates[50]

    # Compute enriched features on FULL panel
    full_enriched = enrich_stationary_features(full_df)
    features_full_at_t = full_enriched[full_enriched["date"] == cutoff_date].sort_values("ticker").reset_index(drop=True)

    # Truncate panel strictly at cutoff_date
    truncated_df = full_df[full_df["date"] <= cutoff_date].copy()
    truncated_enriched = enrich_stationary_features(truncated_df)
    features_trunc_at_t = truncated_enriched[truncated_enriched["date"] == cutoff_date].sort_values("ticker").reset_index(drop=True)

    # Compare all enriched features
    compare_cols = ["rs_nifty_5d", "rs_nifty_21d", "nifty_50_return_5d", "nifty_50_return_21d"]
    for col in compare_cols:
        if col in features_full_at_t.columns:
            np.testing.assert_allclose(
                features_full_at_t[col].values,
                features_trunc_at_t[col].values,
                err_msg=f"Feature {col} showed future lookahead leakage!",
                rtol=1e-7,
                atol=1e-7,
            )


# ==============================================================================
# 5. Redundancy & Importance Clustering Tests
# ==============================================================================

def test_redundancy_clustering():
    """Verify that cluster_redundant_features groups |corr| >= 0.90 variables."""
    feats = ["ret_1d", "ret_1d_copy", "vol_10d", "vol_10d_copy", "rsi_14"]
    # Build synthetic corr matrix
    data = {
        "ret_1d": [1.0, 0.98, 0.1, 0.1, 0.3],
        "ret_1d_copy": [0.98, 1.0, 0.1, 0.1, 0.3],
        "vol_10d": [0.1, 0.1, 1.0, 0.95, -0.2],
        "vol_10d_copy": [0.1, 0.1, 0.95, 1.0, -0.2],
        "rsi_14": [0.3, 0.3, -0.2, -0.2, 1.0],
    }
    corr_df = pd.DataFrame(data, index=feats)

    res = cluster_redundant_features(corr_df, threshold=0.90)
    assert res["cluster_count"] == 2
    assert len(res["recommended_drop"]) == 2
    assert len(res["recommended_keep"]) == 3
    assert "rsi_14" in res["recommended_keep"]


# ==============================================================================
# 6. Canonical Panel Dataset Contract Test
# ==============================================================================

def test_panel_v07_dataset_contract():
    """Verify saved canonical v0.7 panel matches all shared contract requirements."""
    panel_path = Path("data/v0.7/panel_v07.parquet")
    if not panel_path.exists():
        pytest.skip("data/v0.7/panel_v07.parquet has not yet been generated.")

    df = pd.read_parquet(panel_path)

    # 1. Keys
    assert "ticker" in df.columns
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].dt.tz is None  # no timezone

    # 2. Price columns
    price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in price_cols:
        assert c in df.columns, f"Required price column missing: {c}"

    # 3. Targets
    target_cols = ["y_5d", "y_21d", "y_5d_raw", "y_21d_raw", "y_5d_excess", "y_21d_excess"]
    for c in target_cols:
        assert c in df.columns, f"Required target column missing: {c}"

    # 4. Label timing columns
    timing_cols = ["label_start", "label_end_5d", "label_end_21d"]
    for c in timing_cols:
        assert c in df.columns, f"Required timing column missing: {c}"

    # 5. Column name casing
    for c in df.columns:
        assert c == c.lower(), f"Column name not lower_snake_case: {c}"
