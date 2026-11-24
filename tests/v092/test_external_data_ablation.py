"""
Tests for StockReason v0.9.2 External Data Ingestion & Ablation.
"""

import pytest
import pandas as pd
import numpy as np

from src.v092.features.external_data import (
    load_annual_financial_ratios,
    load_sector_indices_features,
)
from src.v092.features.ablation import run_absolute_price_matched_ablation


def test_annual_financials_publication_lag():
    fin_df = load_annual_financial_ratios()
    if not fin_df.empty:
        # Check that publication availability date is lagged to Oct 1 of the year
        sample = fin_df.iloc[0]
        assert str(sample["fy_end_year"]) in sample["available_from_date"]
        assert sample["available_from_date"].endswith("-10-01")
        assert "roe" in fin_df.columns
        assert "profit_margin" in fin_df.columns


def test_sector_indices_loading():
    sec_dfs = load_sector_indices_features()
    assert len(sec_dfs) > 0
    # At least NIFTY_500 or NIFTY_BANK or NIFTY_IT should exist
    assert any(k in sec_dfs for k in ["NIFTY_500", "NIFTY_BANK", "NIFTY_IT"])


def test_matched_ablation_execution():
    panel_df = pd.read_parquet(
        "data/v0.9.1/features/panel_features.parquet",
        columns=["symbol", "date", "close", "target_return_21d"] + [
            "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d",
            "vol_20d", "vol_60d", "dist_ma20", "dist_ma50", "dist_ma200",
            "rsi_14", "macd_diff", "hl_spread", "vol_ratio_20d", "log_turnover",
            "nifty_ret_1d", "nifty_ret_21d", "vix_level", "vix_change_5d", "rel_strength_21d"
        ]
    )
    # Run ablation on subset for speed in test
    sub_sample = panel_df[panel_df["symbol"].isin(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"])].copy()
    ablation_df = run_absolute_price_matched_ablation(sub_sample, horizon=21)
    assert len(ablation_df) >= 2
    assert "inr_price_mae" in ablation_df.columns
    assert "return_mae" in ablation_df.columns
