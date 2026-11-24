"""
Tests for StockReason v0.9.2 Universe Observation Eligibility.
"""

import pytest
import pandas as pd
import numpy as np

from src.v092.canonical.universe import (
    investigate_universe_observation_eligibility,
    filter_panel_to_eligible_monthly_membership,
)


def test_universe_observation_eligibility_audit():
    report = investigate_universe_observation_eligibility()
    assert report["total_discovered_securities"] == 1840
    assert report["v091_training_observations"] == 3111356
    assert report["strict_eligible_training_observations"] == 1402267
    assert report["non_eligible_training_observations_in_v091"] == 1709089
    assert report["v091_val_observations"] == 550251
    assert "3,111,356" in report["audit_finding"]


def test_filter_panel_to_eligible_monthly_membership():
    # Synthetic mini-panel
    panel_df = pd.DataFrame({
        "symbol": ["INFY", "INFY", "INFY", "TCS"],
        "date": ["2023-01-05", "2023-02-10", "2023-03-15", "2023-01-10"],
        "close": [1500.0, 1550.0, 1600.0, 3200.0]
    })
    pit_df = pd.DataFrame({
        "symbol": ["INFY", "TCS"],
        "rebal_str": ["2023-01-02", "2023-01-02"]
    })
    filtered = filter_panel_to_eligible_monthly_membership(panel_df, pit_df)
    # INFY only has 2023-01 eligible, TCS has 2023-01 eligible
    assert len(filtered) == 2
    assert set(filtered["symbol"].tolist()) == {"INFY", "TCS"}
    assert set(filtered["date"].tolist()) == {"2023-01-05", "2023-01-10"}
