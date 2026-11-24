"""Empirical uncertainty evaluation pipeline for StockReason v0.7.

Runs full evaluation across all four horizons (1D, 1W, 1M, 6M):
1. Loads historical price features and prepares strictly causal forward targets.
2. Runs LSTM multi-horizon model with 50 MC-dropout passes.
3. Evaluates directional calibration (Isotonic/Platt) on holdout OOF predictions:
   - Brier score, log loss, ECE before and after calibration.
   - Saves reliability diagrams to artifacts/v0.7/plots/calibration_<horizon>.png.
4. Evaluates Conformal Prediction intervals at nominal 80% coverage:
   - Split-conformal vs Conformalized Quantile Regression (CQR) vs Adaptive Conformal (ACI).
   - Winkler score, empirical coverage, mean/median width.
   - Coverage breakdown by market regime (TRENDING, CHOPPY, VOLATILE) and over time.
5. Benchmark comparison with legacy MC-dropout interval (mean ± 1.96·std).
6. Writes reproducibility config to artifacts/v0.7/uncertainty_config.json
   and complete evaluation metrics to artifacts/v0.7/uncertainty_eval_results.json.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from src.m3_modeling.lstm_model import MultiHorizonLSTM, mc_dropout_inference
from src.v07.uncertainty.conformal import (
    AdaptiveConformalInference,
    ConformalizedQuantileRegression,
    SplitConformalRegressor,
    compare_intervals_with_mc_dropout,
    compute_coverage,
    compute_interval_metrics,
    evaluate_coverage_by_regime,
    evaluate_coverage_over_time,
    run_rolling_conformal_simulation,
)
from src.v07.uncertainty.direction import (
    DirectionalCalibrator,
    compute_brier_score,
    compute_ece,
    compute_log_loss,
    derive_directional_probabilities,
    generate_reliability_diagram,
)
from src.v07.uncertainty.schema import (
    StockUncertaintyForecast,
    UncertaintyHorizonOutput,
    build_uncertainty_horizon,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uncertainty_eval")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

HORIZON_MAP = {
    "1D": {"shift": 1, "target_col": "Target_1D", "label": "1-Day"},
    "1W": {"shift": 5, "target_col": "Target_5D", "label": "1-Week (5-Day)"},
    "1M": {"shift": 21, "target_col": "Target_21D", "label": "1-Month (21-Day)"},
    "6M": {"shift": 126, "target_col": "Target_126D", "label": "6-Month (126-Day)"},
}

SEQUENCE_LENGTH = 60
MC_PASSES = 50
NOMINAL_CONFORMAL_COVERAGE = 0.80  # alpha = 0.20
NOMINAL_MC_COVERAGE = 0.95        # alpha = 0.05
COST_BPS = 15.0                   # 15 basis points = 0.0015


def load_data_and_models(
    price_csv: Path,
    model_path: Path,
    feature_scaler_path: Path,
    target_scaler_path: Path,
) -> Tuple[pd.DataFrame, MultiHorizonLSTM, Any, Any]:
    """Load price data, trained LSTM model, and scalers."""
    logger.info(f"Loading price features from {price_csv}")
    df = pd.read_csv(price_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute forward targets per ticker if not already present
    for h_code, h_info in HORIZON_MAP.items():
        t_col = h_info["target_col"]
        if t_col not in df.columns:
            shift_days = h_info["shift"]
            df[t_col] = df.groupby("ticker")["close"].shift(-shift_days) / df["close"] - 1.0

    # Benchmark forward return (NIFTY 50 return)
    if "nifty_50_close" in df.columns:
        df["nifty_forward_1d"] = (
            df.groupby("ticker")["nifty_50_close"].shift(-1) / df["nifty_50_close"] - 1.0
        )
    else:
        df["nifty_forward_1d"] = 0.0

    f_scaler = joblib.load(feature_scaler_path)
    t_scaler = joblib.load(target_scaler_path)
    num_features = f_scaler.n_features_in_

    model = MultiHorizonLSTM(
        input_dim=num_features,
        hidden_dim=64,
        num_layers=2,
        output_dim=4,
        dropout_rate=0.3,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    logger.info("Loaded MultiHorizonLSTM and scalers successfully.")

    return df, model, f_scaler, t_scaler


def extract_features_and_targets(
    df: pd.DataFrame,
    f_scaler: Any,
    t_scaler: Any,
) -> pd.DataFrame:
    """Build sequence windows, exclude lookahead, align targets and metadata."""
    exclude_cols = {
        "Target_1D", "Target_5D", "Target_21D", "Target_126D",
        "ticker", "date", "market_regime", "nifty_forward_1d",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    num_model_features = f_scaler.n_features_in_

    records = []
    tickers = sorted(df["ticker"].unique())

    for ticker in tickers:
        df_t = df[df["ticker"] == ticker].sort_values("date").copy()
        if len(df_t) < SEQUENCE_LENGTH + 126:
            continue

        raw_feats = df_t[feature_cols].values
        # Pad or trim to fit scaler
        if raw_feats.shape[1] < num_model_features:
            pad = np.zeros((raw_feats.shape[0], num_model_features - raw_feats.shape[1]))
            raw_feats = np.hstack([raw_feats, pad])
        else:
            raw_feats = raw_feats[:, :num_model_features]

        scaled_feats = f_scaler.transform(raw_feats)

        dates = df_t["date"].values
        closes = df_t["close"].values
        regimes = df_t["market_regime"].values if "market_regime" in df_t.columns else np.array(["UNKNOWN"] * len(df_t))
        t1d = df_t["Target_1D"].values
        t5d = df_t["Target_5D"].values
        t21d = df_t["Target_21D"].values
        t126d = df_t["Target_126D"].values
        nifty_bench = df_t["nifty_forward_1d"].values

        for i in range(SEQUENCE_LENGTH, len(df_t) - 126):
            # Window strictly from [i - SEQUENCE_LENGTH : i]
            # Prediction origin is date at i - 1
            rec = {
                "ticker": ticker,
                "date": pd.to_datetime(dates[i - 1]),
                "current_price": float(closes[i - 1]),
                "regime": str(regimes[i - 1]),
                "Target_1D": float(t1d[i - 1]),
                "Target_5D": float(t5d[i - 1]),
                "Target_21D": float(t21d[i - 1]),
                "Target_126D": float(t126d[i - 1]),
                "benchmark_1d": float(nifty_bench[i - 1]) if not np.isnan(nifty_bench[i - 1]) else 0.0,
                "window": scaled_feats[i - SEQUENCE_LENGTH : i],
            }
            records.append(rec)

    logger.info(f"Extracted {len(records)} sequence records across {len(tickers)} tickers.")
    return records


def run_mc_dropout_batch(
    model: MultiHorizonLSTM,
    windows: np.ndarray,
    t_scaler: Any,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run MC Dropout across windows in batches, returns (mean_pred, std_pred) in return scale."""
    model.train()  # Activate dropout
    n_samples = len(windows)
    mean_preds = []
    std_preds = []

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = torch.tensor(windows[start_idx:end_idx], dtype=torch.float32)

            passes = []
            for _ in range(MC_PASSES):
                passes.append(model(batch_x).numpy())
            passes = np.array(passes)  # [MC_PASSES, batch, 4]

            b_mean = np.mean(passes, axis=0)
            b_std = np.std(passes, axis=0)

            # Inverse transform from scaled targets
            # t_scaler: scaled = (y - min) / (max - min) * 2 - 1 -> inverse_transform
            b_mean_ret = t_scaler.inverse_transform(b_mean)
            b_std_ret = np.abs(
                t_scaler.inverse_transform(b_mean + b_std) - b_mean_ret
            )

            mean_preds.append(b_mean_ret)
            std_preds.append(b_std_ret)

    all_mean = np.vstack(mean_preds)
    all_std = np.vstack(std_preds)
    return all_mean, all_std


def run_uncertainty_evaluation() -> Dict[str, Any]:
    """Main evaluation execution function."""
    base_dir = Path(__file__).resolve().parents[3]
    price_csv = base_dir / "data" / "processed" / "price_features.csv"
    model_path = base_dir / "artifacts" / "models" / "best_lstm.pth"
    f_scaler_path = base_dir / "artifacts" / "models" / "feature_scaler.pkl"
    t_scaler_path = base_dir / "artifacts" / "models" / "target_scaler.pkl"

    out_plot_dir = base_dir / "artifacts" / "v0.7" / "plots"
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    out_artifacts_dir = base_dir / "artifacts" / "v0.7"
    out_artifacts_dir.mkdir(parents=True, exist_ok=True)

    df, model, f_scaler, t_scaler = load_data_and_models(
        price_csv, model_path, f_scaler_path, t_scaler_path
    )
    records = extract_features_and_targets(df, f_scaler, t_scaler)

    # Sort strictly chronologically by date
    records.sort(key=lambda r: r["date"])
    dates = np.array([r["date"] for r in records])
    tickers = np.array([r["ticker"] for r in records])
    prices = np.array([r["current_price"] for r in records])
    regimes = np.array([r["regime"] for r in records])
    windows = np.array([r["window"] for r in records])

    targets = {
        "1D": np.array([r["Target_1D"] for r in records]),
        "1W": np.array([r["Target_5D"] for r in records]),
        "1M": np.array([r["Target_21D"] for r in records]),
        "6M": np.array([r["Target_126D"] for r in records]),
    }
    benchmarks_1d = np.array([r["benchmark_1d"] for r in records])

    # Temporal split: Train/Pre-cal (60%), Calibration OOF (20%), Out-of-sample Test (20%)
    n_total = len(records)
    n_train = int(n_total * 0.60)
    n_cal = int(n_total * 0.20)
    cal_slice = slice(n_train, n_train + n_cal)
    test_slice = slice(n_train + n_cal, n_total)

    logger.info(
        f"Temporal splits: Total={n_total}, Train/Warmup={n_train} (to {dates[n_train-1].strftime('%Y-%m-%d')}), "
        f"Calibration={n_cal} ({dates[cal_slice.start].strftime('%Y-%m-%d')} to {dates[cal_slice.stop-1].strftime('%Y-%m-%d')}), "
        f"Test={n_total - n_train - n_cal} ({dates[test_slice.start].strftime('%Y-%m-%d')} to {dates[test_slice.stop-1].strftime('%Y-%m-%d')})"
    )

    # Run inference on all windows (cached for speed)
    cache_path = out_artifacts_dir / "cache_mc_preds.npz"
    if cache_path.exists():
        logger.info(f"Loading cached MC-dropout predictions from {cache_path}")
        cache_data = np.load(cache_path)
        mean_preds = cache_data["mean_preds"]
        std_preds = cache_data["std_preds"]
    else:
        logger.info(f"Running {MC_PASSES} MC-dropout passes on all {n_total} windows...")
        mean_preds, std_preds = run_mc_dropout_batch(model, windows, t_scaler)
        np.savez_compressed(cache_path, mean_preds=mean_preds, std_preds=std_preds)
        logger.info(f"Saved MC-dropout cache to {cache_path}")

    results: Dict[str, Any] = {
        "meta": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_version": "0.7.0",
            "model_architecture": "MultiHorizonLSTM (2-layer, hidden=64, MC-Dropout=0.3)",
            "random_seed": RANDOM_SEED,
            "mc_passes": MC_PASSES,
            "cost_bps": COST_BPS,
            "sample_counts": {
                "total": n_total,
                "calibration_oof": n_cal,
                "test": n_total - n_train - n_cal,
            },
            "date_ranges": {
                "calibration": f"{dates[cal_slice.start].strftime('%Y-%m-%d')} to {dates[cal_slice.stop-1].strftime('%Y-%m-%d')}",
                "test": f"{dates[test_slice.start].strftime('%Y-%m-%d')} to {dates[test_slice.stop-1].strftime('%Y-%m-%d')}",
            },
            "environment": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
            },
        },
        "horizons": {},
        "regime_breakdown": {},
        "comparison_table": [],
    }

    horizon_keys = ["1D", "1W", "1M", "6M"]

    for h_idx, h_code in enumerate(horizon_keys):
        logger.info(f"Evaluating Horizon: {h_code} ({HORIZON_MAP[h_code]['label']})...")

        y_true_all = targets[h_code]
        y_pred_all = mean_preds[:, h_idx]
        std_all = std_preds[:, h_idx]

        # ── 1. Directional Calibration ────────────────────────────────
        # Uncalibrated directional probability via normal CDF
        prob_dir_all = derive_directional_probabilities(
            predicted_return=y_pred_all,
            uncertainty_std=std_all,
            benchmark_return=benchmarks_1d if h_code == "1D" else 0.0,
            cost_bps=COST_BPS,
        )
        p_up_uncal_all = prob_dir_all["prob_up"]
        y_binary_all = (y_true_all > 0.0).astype(float)

        # Calibrate on OOF calibration slice only
        cal_p_uncal = p_up_uncal_all[cal_slice]
        cal_y_bin = y_binary_all[cal_slice]

        calibrator = DirectionalCalibrator(method="auto")
        calibrator.fit(cal_p_uncal, cal_y_bin)

        # Evaluate on out-of-sample Test slice
        test_p_uncal = p_up_uncal_all[test_slice]
        test_y_bin = y_binary_all[test_slice]
        test_p_cal = calibrator.predict_proba(test_p_uncal)

        # Metrics before and after
        cal_eval = calibrator.evaluate(
            y_true=test_y_bin,
            y_prob_uncal=test_p_uncal,
            y_prob_cal=test_p_cal,
        )

        # Generate and save reliability plot
        plot_path = out_plot_dir / f"calibration_{h_code}.png"
        generate_reliability_diagram(
            y_true=test_y_bin,
            y_prob=test_p_cal,
            n_bins=10,
            title=f"StockReason v0.7: Directional Reliability ({h_code} Horizon)",
            save_path=plot_path,
            y_prob_uncalibrated=test_p_uncal,
            label=f"Calibrated ({calibrator.method_used_})",
            label_uncalibrated="Uncalibrated (Heuristic / Normal CDF)",
        )

        # ── 2. Conformal Prediction Intervals (80% Nominal) ────────────
        # Split-Conformal
        sc = SplitConformalRegressor(alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE)
        sc.fit(y_true_cal=y_true_all[cal_slice], y_pred_cal=y_pred_all[cal_slice])
        sc_lower_test, sc_upper_test = sc.predict_interval(y_pred_all[test_slice])
        sc_metrics = compute_interval_metrics(
            y_true_all[test_slice], sc_lower_test, sc_upper_test, alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE
        )

        # Conformalized Quantile Regression (CQR)
        # Using base quantile regression on scaled windows or rolling residuals
        cqr = ConformalizedQuantileRegression(
            alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE, n_estimators=40, max_depth=3
        )
        # Fit base quantile models on train slice (taking 2D representation: last step features)
        X_train_cqr = windows[slice(0, n_train), -1, :20]
        y_train_cqr = y_true_all[slice(0, n_train)]
        X_cal_cqr = windows[cal_slice, -1, :20]
        y_cal_cqr = y_true_all[cal_slice]
        X_test_cqr = windows[test_slice, -1, :20]

        cqr.fit_base_models(X_train_cqr, y_train_cqr)
        cqr.calibrate(y_cal_cqr, X_cal=X_cal_cqr)
        cqr_lower_test, cqr_upper_test = cqr.predict_interval(X_test=X_test_cqr)
        cqr_metrics = compute_interval_metrics(
            y_true_all[test_slice], cqr_lower_test, cqr_upper_test, alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE
        )

        # Adaptive Conformal Inference (ACI) on Test
        # Initialize ACI with calibration residuals
        cal_res = np.abs(y_true_all[cal_slice] - y_pred_all[cal_slice])
        aci = AdaptiveConformalInference(
            target_alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE,
            gamma=0.03,
            window_size=126,
        )
        aci.reset(initial_residuals=cal_res)
        aci_lowers, aci_uppers = [], []
        test_y = y_true_all[test_slice]
        test_pred = y_pred_all[test_slice]

        for t_i in range(len(test_y)):
            l_t, u_t = aci.predict_interval(test_pred[t_i])
            aci.step(y_true=test_y[t_i], lower=l_t, upper=u_t, y_pred=test_pred[t_i])
            aci_lowers.append(l_t)
            aci_uppers.append(u_t)

        aci_metrics = compute_interval_metrics(
            test_y, np.array(aci_lowers), np.array(aci_uppers), alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE
        )

        # ── 3. MC-Dropout Comparison (Legacy Heuristic Interval) ────────
        # Legacy interval: mean ± 1.96·std (claimed nominal 95%)
        mc_comp = compare_intervals_with_mc_dropout(
            y_true=test_y,
            point_pred=test_pred,
            mc_std=std_all[test_slice],
            lower_conformal=sc_lower_test,
            upper_conformal=sc_upper_test,
            nominal_conformal=NOMINAL_CONFORMAL_COVERAGE,
            nominal_mc=NOMINAL_MC_COVERAGE,
        )

        # Regime breakdown on Test slice
        test_regimes = regimes[test_slice]
        regime_eval = evaluate_coverage_by_regime(
            y_true=test_y,
            lower=sc_lower_test,
            upper=sc_upper_test,
            regimes=test_regimes,
            alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE,
        )

        # Coverage over time (60-day rolling)
        test_dates = dates[test_slice]
        over_time_eval = evaluate_coverage_over_time(
            dates=test_dates,
            y_true=test_y,
            lower=sc_lower_test,
            upper=sc_upper_test,
            rolling_window=60,
            alpha=1.0 - NOMINAL_CONFORMAL_COVERAGE,
        )

        horizon_summary = {
            "horizon": h_code,
            "label": HORIZON_MAP[h_code]["label"],
            "directional_calibration": {
                "calibrator_method": calibrator.method_used_,
                "n_calibration_samples": calibrator.n_samples_,
                "test_evaluation": cal_eval,
            },
            "conformal_split": sc_metrics,
            "conformal_cqr": cqr_metrics,
            "conformal_aci": aci_metrics,
            "mc_dropout_comparison": mc_comp,
            "regime_coverage": regime_eval,
            "rolling_coverage_summary": {
                "mean_rolling_cov": float(np.mean(over_time_eval["rolling_coverage"])),
                "min_rolling_cov": float(np.min(over_time_eval["rolling_coverage"])),
                "max_rolling_cov": float(np.max(over_time_eval["rolling_coverage"])),
            },
            "plot_saved": str(plot_path),
        }
        results["horizons"][h_code] = horizon_summary

        # Add to comparison table
        results["comparison_table"].append({
            "horizon": h_code,
            "method": "Legacy MC-Dropout",
            "nominal_coverage": "95.0%",
            "empirical_coverage": f"{mc_comp['mc_dropout']['empirical_coverage']*100:.1f}%",
            "coverage_gap": f"{mc_comp['mc_dropout']['coverage_gap']*100:+.1f}%",
            "mean_interval_width": f"{mc_comp['mc_dropout']['mean_width']*100:.2f}%",
            "winkler_score": f"{mc_comp['mc_dropout']['winkler_score']:.4f}",
            "calibrated_status": "Uncalibrated (Heuristic)",
        })
        results["comparison_table"].append({
            "horizon": h_code,
            "method": "Split Conformal",
            "nominal_coverage": "80.0%",
            "empirical_coverage": f"{sc_metrics['empirical_coverage']*100:.1f}%",
            "coverage_gap": f"{sc_metrics['coverage_gap']*100:+.1f}%",
            "mean_interval_width": f"{sc_metrics['mean_width']*100:.2f}%",
            "winkler_score": f"{sc_metrics['winkler_score']:.4f}",
            "calibrated_status": "Exact Conformal (Marginal)",
        })
        results["comparison_table"].append({
            "horizon": h_code,
            "method": "CQR (Quantile Regression)",
            "nominal_coverage": "80.0%",
            "empirical_coverage": f"{cqr_metrics['empirical_coverage']*100:.1f}%",
            "coverage_gap": f"{cqr_metrics['coverage_gap']*100:+.1f}%",
            "mean_interval_width": f"{cqr_metrics['mean_width']*100:.2f}%",
            "winkler_score": f"{cqr_metrics['winkler_score']:.4f}",
            "calibrated_status": "Adaptive Heteroscedastic",
        })
        results["comparison_table"].append({
            "horizon": h_code,
            "method": "Adaptive ACI",
            "nominal_coverage": "80.0%",
            "empirical_coverage": f"{aci_metrics['empirical_coverage']*100:.1f}%",
            "coverage_gap": f"{aci_metrics['coverage_gap']*100:+.1f}%",
            "mean_interval_width": f"{aci_metrics['mean_width']*100:.2f}%",
            "winkler_score": f"{aci_metrics['winkler_score']:.4f}",
            "calibrated_status": "Regime Adaptive (Online)",
        })

    # Save results and configs
    config_file = out_artifacts_dir / "uncertainty_config.json"
    results_file = out_artifacts_dir / "uncertainty_eval_results.json"

    with open(config_file, "w") as f:
        json.dump(results["meta"], f, indent=2)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation config to {config_file}")
    logger.info(f"Saved complete evaluation results to {results_file}")

    return results


if __name__ == "__main__":
    run_uncertainty_evaluation()
