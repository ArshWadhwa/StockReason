"""
Unit tests for StockReason v0.10 Advanced Feature Engineering & Ablation.
"""

import pytest
import numpy as np
import pandas as pd

from src.v010.features.cross_sectional import compute_daily_cross_sectional_features
from src.v010.features.announcements import CorporateAnnouncementsFeatureExtractor
from src.v010.features.ablation import run_v010_feature_ablation
from src.v091.features.pipeline import FEATURE_COLS_21


def test_cross_sectional_features_computation():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=5, freq="B").strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        for i in range(20):
            rows.append({
                "date_str": d,
                "symbol": f"STOCK_{i}",
                "close": 100.0 + np.random.randn() * 10.0,
                "volume": 1000.0 + np.random.rand() * 5000.0,
                "turnover": 500000.0,
                "ret_21d": 0.02 + np.random.randn() * 0.05,
                "vol_60d": 0.20 + np.random.rand() * 0.10,
                "close_sma_50": 98.0,
            })
    df_sample = pd.DataFrame(rows)

    df_feats = compute_daily_cross_sectional_features(df_sample)
    assert "mom_pctile_21d" in df_feats.columns
    assert "vol_pctile_60d" in df_feats.columns
    assert "turnover_pctile" in df_feats.columns
    assert "market_breadth_sma50" in df_feats.columns
    assert "cs_dispersion_21d" in df_feats.columns
    assert df_feats["mom_pctile_21d"].between(0.0, 1.0).all()
    assert df_feats["market_breadth_sma50"].between(0.0, 1.0).all()


def test_corporate_announcements_extraction():
    extractor = CorporateAnnouncementsFeatureExtractor()
    df_ann = extractor.load_announcements()
    assert isinstance(df_ann, pd.DataFrame)

    # Test with dummy panel
    df_panel = pd.DataFrame({
        "symbol": ["RELIANCE", "TCS"],
        "date_str": ["2024-05-01", "2024-05-01"],
    })
    merged = extractor.compute_symbol_announcement_features(df_panel)
    assert "ann_count_30d" in merged.columns
    assert "ann_has_recent_results" in merged.columns
