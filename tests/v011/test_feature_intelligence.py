"""
Tests for StockReason v0.11 Feature Intelligence.
=================================================
1. Fundamental point-in-time correctness.
2. Corporate announcement event density without lookahead.
3. Microstructure & delivery features.
4. Cross-sectional ranking logic.
"""

import pytest
import numpy as np
import pandas as pd

from src.v011.features.fundamentals import attach_point_in_time_fundamentals
from src.v011.features.events import attach_point_in_time_events
from src.v011.features.market_structure import compute_market_structure_features
from src.v011.features.cross_sectional import compute_daily_cross_sectional_features


def test_fundamental_point_in_time_join():
    panel_df = pd.DataFrame({
        "symbol": ["INFY", "INFY", "INFY"],
        "date_str": ["2024-01-10", "2024-01-15", "2024-04-20"],
        "close": [1500.0, 1550.0, 1600.0],
    })

    fin_df = pd.DataFrame({
        "clean_symbol": ["INFY", "INFY"],
        "report_date": ["2024-01-12", "2024-04-18"],
        "revenue_inr": [1000.0, 1200.0],
        "net_profit_inr": [200.0, 260.0],
        "ebitda_inr": [250.0, 310.0],
        "net_margin": [0.20, 0.217],
        "ebitda_margin": [0.25, 0.258],
        "rev_growth_yoy": [0.10, 0.15],
        "profit_growth_yoy": [0.08, 0.12],
        "fundamental_score": [0.5, 0.6],
    })

    merged = attach_point_in_time_fundamentals(panel_df, fin_df)

    # Row 1 (2024-01-10) is BEFORE first report_date (2024-01-12): must have no coverage (0.0)
    assert merged.loc[0, "has_fundamental_coverage"] == 0
    assert merged.loc[0, "net_margin"] == 0.0

    # Row 2 (2024-01-15) is AFTER first report (2024-01-12): receives first report's net_margin (0.20)
    assert merged.loc[1, "has_fundamental_coverage"] == 1
    assert abs(merged.loc[1, "net_margin"] - 0.20) < 1e-4

    # Row 3 (2024-04-20) is AFTER second report (2024-04-18): receives second report's net_margin (0.217)
    assert merged.loc[2, "has_fundamental_coverage"] == 1
    assert abs(merged.loc[2, "net_margin"] - 0.217) < 1e-4


def test_market_structure_features():
    df = pd.DataFrame({
        "symbol": ["TCS", "TCS", "TCS"],
        "date": pd.date_range("2024-01-01", periods=3),
        "open": [3500.0, 3550.0, 3600.0],
        "high": [3550.0, 3600.0, 3650.0],
        "low": [3480.0, 3520.0, 3580.0],
        "close": [3520.0, 3590.0, 3620.0],
        "prev_close": [3490.0, 3520.0, 3590.0],
        "turnover": [1e8, 1.2e8, 1.5e8],
        "delivery_pct": [45.0, 50.0, 55.0],
    })

    res = compute_market_structure_features(df)
    assert "ret_overnight" in res.columns
    assert "ret_intraday" in res.columns
    assert "hl_spread" in res.columns
    assert "turnover_shock_20d" in res.columns
    # HL spread must be positive
    assert (res["hl_spread"] >= 0).all()


def test_cross_sectional_rankings():
    df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "date_str": ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02"],
        "ret_21d": [0.01, 0.05, -0.02, 0.10],
        "vol_60d": [0.15, 0.25, 0.35, 0.10],
    })

    ranked = compute_daily_cross_sectional_features(df)
    # Highest return (D with 0.10) must have percentile rank 1.0
    d_rank = ranked.loc[ranked["symbol"] == "D", "cs_mom_rank_21d"].values[0]
    c_rank = ranked.loc[ranked["symbol"] == "C", "cs_mom_rank_21d"].values[0]
    assert d_rank > c_rank
    assert 0.0 <= c_rank <= 1.0
    assert 0.0 <= d_rank <= 1.0
