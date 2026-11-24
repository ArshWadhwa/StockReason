"""
StockReason v0.9.2 — Master Execution Script for Absolute Price Forecasting Pipeline.
======================================================================================
Executes end-to-end evaluation of:
1. Universe observation eligibility audit (Item 5).
2. Generalized corporate actions & segment gap masking (Item 7).
3. Exact per-observation target-end timestamp purging (Item 6).
4. Matched-cohort ablation with annual financials and sector indices (Item 8).
5. Engine A: Multi-horizon absolute price forecasting vs strict baselines (Items 2, 3, 4).
6. Engine B: Cross-sectional relative stock ranking (Item 1).
7. Paired time-block difference bootstrap test: LightGBM vs Momentum (Item 10).
8. Multi-horizon conformal interval calibration and coverage evaluation (Item 9).
9. Inference parity verification and master reporting (Items 11, 12).
"""

import os
import sys
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

from src.v091.features.pipeline import FEATURE_COLS_21, StockReasonPipelineV091
from src.v092.canonical.universe import investigate_universe_observation_eligibility
from src.v092.canonical.integrity import (
    detect_unadjusted_corporate_actions_and_gaps,
    mask_generalized_unadjusted_target_windows,
)
from src.v092.validation.purged_cv import (
    compute_per_observation_target_end_dates,
    ExactPerObservationPurgedCV,
)
from src.v092.features.ablation import run_absolute_price_matched_ablation
from src.v092.models.baselines import evaluate_engine_a_baselines, compute_absolute_price_metrics
from src.v092.models.engine_a import train_engine_a_models
from src.v092.models.engine_b import evaluate_engine_b_ranking
from src.v092.models.comparison import compare_lgbm_vs_momentum_paired
from src.v092.calibration.intervals import MultiHorizonConformalCalibrator
from src.v092.api.dual_forecast_service import StockReasonDualForecastServiceV092

OUTPUT_DIR = "reports/v0.9.2"
ARTIFACTS_DIR = "artifacts/v0.9.2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    start_time = time.time()
    print("================================================================================")
    print("StockReason v0.9.2: Running Complete Absolute Price Forecasting Pipeline")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # Step 1: Universe Observation Eligibility Audit (Item 5)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Auditing Point-in-Time Universe Observation Eligibility...")
    eligibility_report = investigate_universe_observation_eligibility()
    df_eligibility = pd.DataFrame([{
        "Metric": k, "Value": str(v)
    } for k, v in eligibility_report.items() if k != "audit_finding"])
    df_eligibility.to_csv(os.path.join(OUTPUT_DIR, "universe_observation_eligibility.csv"), index=False)
    print(f"  -> Discovered Securities: {eligibility_report['total_discovered_securities']}")
    print(f"  -> v0.9.1 Training Rows: {eligibility_report['v091_training_observations']:,}")
    print(f"  -> Strict Active Constituent Rows: {eligibility_report['strict_eligible_training_observations']:,} ({eligibility_report['training_eligibility_pct']}%)")
    print(f"  -> Non-Eligible Rows: {eligibility_report['non_eligible_training_observations_in_v091']:,}")

    # -------------------------------------------------------------------------
    # Step 2: Generalized Corporate Actions & Segment Gap Audit (Item 7)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Executing Generalized Corporate Action & Segment Gap Audit...")
    panel_path = "data/v0.9.1/features/panel_features.parquet"
    panel_df = pd.read_parquet(panel_path)
    if "date_str" not in panel_df.columns:
        panel_df["date_str"] = pd.to_datetime(panel_df["date"]).dt.strftime("%Y-%m-%d")

    events_df = detect_unadjusted_corporate_actions_and_gaps(panel_df)
    events_df.to_csv(os.path.join(OUTPUT_DIR, "generalized_corporate_action_audit.csv"), index=False)
    print(f"  -> Total Detected Anomaly / Corporate Action Events: {len(events_df)}")

    # Mask target windows
    horizons = [1, 5, 21, 63, 126]
    panel_masked, _, masked_counts = mask_generalized_unadjusted_target_windows(
        panel_df, events_df=events_df, horizons=horizons
    )
    df_masked = pd.DataFrame([
        {"horizon": f"{h}d", "masked_target_windows": count} for h, count in masked_counts.items()
    ])
    df_masked.to_csv(os.path.join(OUTPUT_DIR, "masked_windows_summary.csv"), index=False)
    print(f"  -> Masked Windows per Horizon: {masked_counts}")

    # -------------------------------------------------------------------------
    # Step 3: Exact Per-Observation Forward Purging Audit (Item 6)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Exact Per-Observation Target-End Forward Purge Audit...")
    panel_with_ends = compute_per_observation_target_end_dates(panel_masked, horizons=horizons)
    all_market_dates = sorted(panel_with_ends["date_str"].unique().tolist())

    cv = ExactPerObservationPurgedCV(embargo_sessions=5)
    folds = cv.generate_folds(all_market_dates)

    purge_records = []
    for f in folds:
        for h in horizons:
            _, audit = cv.purge_train_by_actual_target_end(
                panel_with_ends,
                train_sessions=f["train_sessions"],
                val_start_date=f["val_start_date"],
                horizon=h,
                all_market_dates=all_market_dates,
            )
            purge_records.append({
                "fold": f["fold"],
                "horizon": h,
                **audit
            })
    df_purge = pd.DataFrame(purge_records)
    df_purge.to_csv(os.path.join(OUTPUT_DIR, "exact_per_observation_purge_audit.csv"), index=False)
    has_leakage = bool(df_purge["leakage_detected"].any())
    print(f"  -> Evaluated {len(df_purge)} Fold-Horizon combinations. Any leakage detected: {has_leakage}")

    # -------------------------------------------------------------------------
    # Step 4: Matched-Cohort Ablation with Annual Financials & Sectors (Item 8)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Matched-Cohort Ablation (Technicals vs Market vs Sector vs Financials)...")
    ablation_df = run_absolute_price_matched_ablation(panel_masked, horizon=21)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "matched_cohort_ablation_absolute.csv"), index=False)
    print("  -> Matched Cohort Ablation Results:")
    print(ablation_df[["feature_group", "feature_count", "return_mae", "inr_price_mae", "normalized_price_error_pct", "rank_ic"]])

    # -------------------------------------------------------------------------
    # Step 5: Multi-Horizon Engine A: Absolute Price Forecasting vs Strict Baselines
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training & Evaluating Engine A Absolute Price Forecasting across Horizons...")
    f_cols = FEATURE_COLS_21
    clean_panel = panel_masked.dropna(subset=f_cols + ["target_return_21d", "close"]).sort_values("date_str")

    train_data = clean_panel[clean_panel["date_str"] < "2023-01-01"]
    val_data = clean_panel[(clean_panel["date_str"] >= "2023-01-01") & (clean_panel["date_str"] < "2025-01-01")]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_data[f_cols].values)
    X_val = scaler.transform(val_data[f_cols].values)
    val_close = val_data["close"].values

    engine_a_all_records = []
    conformal_calibrator = MultiHorizonConformalCalibrator(horizons=horizons)
    engine_a_predictions = {}

    for h in horizons:
        target_ret_col = f"target_return_{h}d"
        if target_ret_col not in train_data.columns:
            continue

        y_tr_h = train_data[target_ret_col].values
        y_val_h = val_data[target_ret_col].values

        # 1. Baselines for horizon h
        df_base_h = evaluate_engine_a_baselines(train_data, val_data, horizon=h)
        for _, r in df_base_h.iterrows():
            engine_a_all_records.append({"horizon": f"{h}d", **r.to_dict()})

        # 2. Machine Learning Challengers (Ridge, LightGBM, CatBoost, GRU)
        train_res = train_engine_a_models(
            X_tr, y_tr_h, X_val, y_val_h, val_close, horizon=h, include_nn=(h == 21)
        )
        for _, r in train_res["metrics_table"].iterrows():
            engine_a_all_records.append({"horizon": f"{h}d", **r.to_dict()})

        # Store predictions for paired comparison & conformal intervals
        engine_a_predictions[h] = train_res["predictions"]["LightGBM"]
        val_vols = val_data["vol_60d"].values if "vol_60d" in val_data.columns else np.full(len(val_data), 0.25)
        conformal_calibrator.fit_horizon(h, y_val_h, train_res["predictions"]["LightGBM"], val_vols)

    df_engine_a = pd.DataFrame(engine_a_all_records)
    df_engine_a.to_csv(os.path.join(OUTPUT_DIR, "engine_a_absolute_price_evaluations.csv"), index=False)
    print("  -> Top Engine A 21-Day Price Forecasting Metrics:")
    print(df_engine_a[df_engine_a["horizon"] == "21d"][["model_or_baseline", "return_mae", "inr_price_mae", "normalized_price_error_pct", "directional_accuracy_pct"]])

    # -------------------------------------------------------------------------
    # Step 6: Multi-Horizon Conformal Calibration Coverage Evaluation (Item 9)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Evaluating Conformal Coverage across all Horizons...")
    coverage_df = conformal_calibrator.evaluate_multi_horizon_coverage(val_data, engine_a_predictions)
    coverage_df.to_csv(os.path.join(OUTPUT_DIR, "multi_horizon_conformal_coverage.csv"), index=False)
    print("  -> Conformal Empirical Coverage Table:")
    print(coverage_df)

    # -------------------------------------------------------------------------
    # Step 7: Engine B: Cross-Sectional Stock Ranking Evaluation (Item 1)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Evaluating Engine B Cross-Sectional Stock Ranking...")
    y_tr_b = train_data["target_demeaned_21d"].values
    y_val_b = val_data["target_demeaned_21d"].values

    engine_b_res = evaluate_engine_b_ranking(X_tr, y_tr_b, X_val, y_val_b, val_data, horizon=21)
    df_engine_b = engine_b_res["metrics_table"]
    df_engine_b.to_csv(os.path.join(OUTPUT_DIR, "engine_b_ranking_evaluations.csv"), index=False)
    print("  -> Engine B Ranking Evaluations:")
    print(df_engine_b)

    # -------------------------------------------------------------------------
    # Step 8: Paired Time-Block Difference Confidence Intervals (Item 10)
    # -------------------------------------------------------------------------
    print("\n[Step 8] Computing Paired Time-Block Difference Confidence Intervals (LGBM vs Momentum)...")
    val_comp = val_data[["date_str"]].copy()
    val_comp["pred_lgb"] = engine_a_predictions[21]
    val_comp["pred_mom"] = engine_b_res["predictions"]["Momentum_12_1"]
    val_comp["target_ret"] = val_data["target_return_21d"].values
    val_comp["target_demeaned"] = val_data["target_demeaned_21d"].values

    paired_res = compare_lgbm_vs_momentum_paired(
        val_comp,
        pred_lgb_col="pred_lgb",
        pred_mom_col="pred_mom",
        target_ret_col="target_ret",
        target_demeaned_col="target_demeaned",
    )
    with open(os.path.join(OUTPUT_DIR, "paired_lgbm_vs_momentum_bootstrap.json"), "w") as f:
        json.dump(paired_res, f, indent=2)
    print(f"  -> Engine A Return MAE Diff: {paired_res['engine_a_return_mae_diff']['mean_diff']} (CI: {paired_res['engine_a_return_mae_diff']['ci_95']})")
    print(f"  -> Engine B Rank IC Diff: {paired_res['engine_b_rank_ic_diff']['mean_diff']} (CI: {paired_res['engine_b_rank_ic_diff']['ci_95']})")

    # -------------------------------------------------------------------------
    # Step 9: Save Serialized Production Pipelines & Verify Parity (Items 11, 12)
    # -------------------------------------------------------------------------
    print("\n[Step 9] Serializing Models & Verifying Parity...")
    mask_a = ~np.isnan(train_data["target_return_21d"].values)
    ridge_a = Ridge(alpha=100.0, random_state=42).fit(X_tr[mask_a], train_data["target_return_21d"].values[mask_a])
    pipe_a = StockReasonPipelineV091(scaler=scaler, model=ridge_a, version="0.9.2")
    joblib.dump(pipe_a, os.path.join(ARTIFACTS_DIR, "engine_a_pipeline.joblib"))

    mask_b = ~np.isnan(train_data["target_demeaned_21d"].values)
    ridge_b = Ridge(alpha=100.0, random_state=42).fit(X_tr[mask_b], train_data["target_demeaned_21d"].values[mask_b])
    pipe_b = StockReasonPipelineV091(scaler=scaler, model=ridge_b, version="0.9.2")
    joblib.dump(pipe_b, os.path.join(ARTIFACTS_DIR, "engine_b_pipeline.joblib"))
    joblib.dump(conformal_calibrator, os.path.join(ARTIFACTS_DIR, "multi_horizon_conformal_calibrator.joblib"))

    service = StockReasonDualForecastServiceV092(
        engine_a_pipeline=pipe_a,
        engine_b_pipeline=pipe_b,
        conformal_calibrator=conformal_calibrator,
    )

    # 50 random samples parity test
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_data), size=50, replace=False)
    parity_diffs = []
    for idx in sample_indices:
        row = val_data.iloc[idx]
        f_vec = row[f_cols].values.astype(float).reshape(1, -1)
        pred_batch_a = float(pipe_a.predict(f_vec)[0])
        res_live = service.forecast_security(row["symbol"], horizon_days=21, as_of_date=row["date_str"])
        pred_service_a = res_live["engine_a_absolute_price"]["predicted_absolute_return"]
        diff = abs(pred_batch_a - pred_service_a)
        parity_diffs.append({"symbol": row["symbol"], "date": row["date_str"], "diff": diff})

    df_parity = pd.DataFrame(parity_diffs)
    df_parity.to_csv(os.path.join(OUTPUT_DIR, "inference_parity_test_results.csv"), index=False)
    max_parity_diff = float(df_parity["diff"].max())
    print(f"  -> Maximum Absolute Parity Difference: {max_parity_diff:.2e} (Passed: {max_parity_diff < 1e-4})")

    # Master audit results JSON
    master_results = {
        "pipeline_version": "0.9.2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": round(time.time() - start_time, 2),
        "eligibility_audit": eligibility_report,
        "masked_windows_summary": masked_counts,
        "purge_audit_has_leakage": bool(has_leakage),
        "paired_comparison": paired_res,
        "max_parity_diff": float(max_parity_diff),
        "conformal_coverage": coverage_df.to_dict(orient="records"),
    }
    with open(os.path.join(OUTPUT_DIR, "master_audit_results.json"), "w") as f:
        json.dump(master_results, f, indent=2)

    print("\n================================================================================")
    print(f"Pipeline Finished Successfully in {round(time.time() - start_time, 2)} seconds!")
    print("================================================================================")


if __name__ == "__main__":
    main()
