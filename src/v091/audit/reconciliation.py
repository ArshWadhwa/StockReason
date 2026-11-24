"""
StockReason v0.9.1 — Earlier Reports vs System Snapshot Reconciliation.
========================================================================
Forensically documents and explains each discrepancy between earlier reports
(v0.7 / v0.8) and the independently verified system snapshot (v0.9/v0.9.1):

1. Feature Counts (21 vs 42/55/113):
   - Earlier claims of 42, 55, and 113 features were carried over verbatim from
     legacy v0.7 documents (where news sentiment and extensive technical indicators
     were included).
   - Production v0.9 pipeline actually uses exactly 21 features (Feature Set B).
   - Correct figure: 21 features in production.

2. Validation RMSE (0.118 vs 0.31):
   - v0.7 reported RMSE ~ 0.31 on raw, unscaled prices or poorly converged neural models.
   - Production v0.9 RMSE is 0.118 (0.118 return units ~ 11.8% return error over 21 days).
   - Correct figure: 0.118 for 21-day return forecasting.

3. Prediction Interval Width (sigma 0.1169 vs 0.076, +-22.9% vs +-15%):
   - Realized empirical residual standard deviation on validation is sigma = 0.1169 (+-22.9% for 95%).
   - Production API in `forecast_service.py` artificially hardcoded calibrated_sigma = 0.076
     (1.96 * 0.076 = 14.896% ~ +-15%), providing narrow, under-covered intervals for all stocks.
   - Correct figure: sigma = 0.1169 (true empirical uncertainty).

4. Price MAE (INR 162.93 vs INR 228.22):
   - INR 162.93 was reported on a normalized or lower-priced subset.
   - When evaluated across the complete out-of-sample test cohort including high-priced stocks,
     the true price MAE is INR 228.22.
   - Correct figure: INR 228.22.

5. Directional Accuracy (62.47% vs 53.76%):
   - 62.47% was reported in v0.7 under an in-sample or close-to-close lookahead setting.
   - True out-of-sample directional accuracy of the collapsed LightGBM model is 53.76%,
     which is mathematically identical to the Always-Up naive baseline (53.76%).
   - Earlier claim of 62.47% is formally WITHDRAWN.

6. Security Counts (84 vs 88 vs 99):
   - 84 distinct equities had full unbroken training history from 2011 to 2022.
   - 88 distinct equities were present in the validation partition (2023-2024).
   - 99 distinct equities traded during the 2025-2026 test partition.
   - In v0.9.1, the point-in-time universe expands to 500 equities per month (1,870 unique symbols).
"""

from typing import Dict, List, Any
import pandas as pd


RECONCILIATION_TABLE = [
    {
        "metric_or_claim": "Feature Count",
        "earlier_report_claim": "42 / 55 / 113 features",
        "snapshot_audited_figure": "21 features (Set B)",
        "cause_of_discrepancy": "Legacy tables from v0.7 text were carried forward without updating; production train.py and builder.py only ever generated 21 features.",
        "correct_figure": "21 features",
        "status": "Corrected; legacy claims withdrawn."
    },
    {
        "metric_or_claim": "Validation RMSE",
        "earlier_report_claim": "0.3100",
        "snapshot_audited_figure": "0.1180",
        "cause_of_discrepancy": "0.31 reflected unconverged v0.7 deep learning loss or nominal scale; 0.118 is true 21-day return RMSE on standardized features.",
        "correct_figure": "0.1180",
        "status": "Verified on code."
    },
    {
        "metric_or_claim": "Interval Width (Sigma)",
        "earlier_report_claim": "Sigma = 0.076 (+-15.0% interval)",
        "snapshot_audited_figure": "Sigma = 0.1169 (+-22.9% interval)",
        "cause_of_discrepancy": "API hardcoded sigma=0.076 to make intervals look artificially tight, achieving only 76% empirical coverage instead of 95%.",
        "correct_figure": "Sigma = 0.1169 (empirical residual)",
        "status": "Remediated in v0.9.1 via conformal prediction."
    },
    {
        "metric_or_claim": "Price MAE (INR)",
        "earlier_report_claim": "INR 162.93",
        "snapshot_audited_figure": "INR 228.22",
        "cause_of_discrepancy": "Earlier evaluation omitted high-priced test equities (MRF, Page Ind) or evaluated on an earlier sample.",
        "correct_figure": "INR 228.22",
        "status": "Verified across full cohort."
    },
    {
        "metric_or_claim": "Directional Accuracy",
        "earlier_report_claim": "62.47%",
        "snapshot_audited_figure": "53.76% (Identical to Always-Up)",
        "cause_of_discrepancy": "Earlier figure was in-sample or leaked; true out-of-sample collapsed model predicts UP 100% of the time, matching market base rate (53.76%).",
        "correct_figure": "53.76%",
        "status": "62.47% claim formally WITHDRAWN."
    },
    {
        "metric_or_claim": "Security Counts",
        "earlier_report_claim": "Claims of 100 or 500 stocks",
        "snapshot_audited_figure": "84 (Train), 88 (Val), 99 (Test)",
        "cause_of_discrepancy": "Ingestion took a single October 2026 snapshot of 100 stocks; only 84 had 2011 history; 16 dropped out.",
        "correct_figure": "84/88/99 in v0.9; 500 per month (1,870 total) in v0.9.1",
        "status": "Remediated to true point-in-time universe."
    }
]


def get_reconciliation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(RECONCILIATION_TABLE)
