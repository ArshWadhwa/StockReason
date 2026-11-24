"""
StockReason v0.11 — Historical Stress Testing & Epoch Analysis.
================================================================
Remediates Phase 6 (Section 39):
Evaluates model stability and baseline-relative edge across 7 distinct historical market epochs:
1. 2011–2012: Eurozone Debt Crisis & Global Consolidation (Bearish/Sideways)
2. 2014–2015: Indian General Election Bull Surge (Bullish Expansion)
3. 2018–2019: NBFC Liquidity Crisis & Small-Cap Crash (Credit Stress)
4. 2020 Q1–Q2: COVID-19 Pandemic Volatility Expansion (Extreme Shock)
5. 2020 Q3–2021: Post-Pandemic Liquidity Rally (Strong Trend)
6. 2022: Global Central Bank Rate-Hike Cycle & Inflation (Sideways/Volatile)
7. 2023–2026: Broad Domestic Market Expansion & Modern Regime (Evaluation Period)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from src.v011.evaluation.baselines import compute_price_metrics


EPOCHS = {
    "2011_2012_eurozone_crisis": ("2011-01-01", "2013-01-01"),
    "2014_2015_election_bull": ("2014-01-01", "2016-01-01"),
    "2018_2019_nbfc_liquidity_crisis": ("2018-01-01", "2020-01-01"),
    "2020_covid_shock_and_recovery": ("2020-01-01", "2021-01-01"),
    "2021_post_covid_rally": ("2021-01-01", "2022-01-01"),
    "2022_rate_hike_cycle": ("2022-01-01", "2023-01-01"),
    "2023_2026_modern_expansion": ("2023-01-01", "2026-11-25"),
}


def evaluate_historical_stress_testing_v011(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "target_return_21d",
    date_col: str = "date_str",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Evaluates forecasting stability and baseline-relative performance across all 7 historical epochs.
    """
    results = []

    for epoch_name, (start_d, end_d) in EPOCHS.items():
        sub = df[(df[date_col] >= start_d) & (df[date_col] < end_d)].dropna(subset=[pred_col, target_col, price_col])
        if len(sub) < 50:
            continue

        y_true = sub[target_col].values
        y_pred = sub[pred_col].values
        close = sub[price_col].values

        m_model = compute_price_metrics(y_true, y_pred, close)

        # Matched historical mean baseline for that epoch
        mean_ret = float(np.mean(y_true))
        m_base = compute_price_metrics(y_true, np.full(len(y_true), mean_ret), close)

        results.append({
            "market_epoch": epoch_name,
            "date_range": f"{start_d} to {end_d}",
            "observation_count": len(sub),
            "model_return_mae": m_model["return_mae"],
            "baseline_return_mae": m_base["return_mae"],
            "model_inr_price_mae": m_model["inr_price_mae"],
            "baseline_inr_price_mae": m_base["inr_price_mae"],
            "model_mape_pct": m_model["mape_pct"],
            "balanced_accuracy_pct": m_model["balanced_accuracy_pct"],
            "outperformed_baseline": m_model["return_mae"] < m_base["return_mae"],
            "relative_edge_pct": round((m_base["return_mae"] - m_model["return_mae"]) / m_base["return_mae"] * 100.0, 2),
        })

    return pd.DataFrame(results)
