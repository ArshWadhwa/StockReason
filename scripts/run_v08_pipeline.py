"""
StockReason v0.8 End-to-End Pipeline Execution Script.
======================================================
Orchestrates:
1. Storage hierarchy initialization under data/v0.8/.
2. Historical data acquisition (2011-11-01 to latest session) with resume and caching.
3. Canonical data cleaning, split anomaly detection, and turnover-invariant adjustment.
4. Feature engineering and multi-horizon target calculation (1, 5, 21, 63, 126d).
5. 14-point dataset quality audit and gate validation.
6. Scaling experiments evaluation across historical depth (5y, 10y, 15y) and breadth (96, 100, 300, 500 stocks).
"""

import os
import sys
sys.path.insert(0, ".")

# CRITICAL ON MACOS: Set OpenMP flag and import torch BEFORE lightgbm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.v08.universe import get_universe, get_universe_details, BENCHMARK_INDICES, MACRO_TICKERS
from src.v08.storage import init_storage_directories, write_parquet_atomic, compute_directory_manifest
from src.v08.download import fetch_ticker_ohlcv, sanitize_ticker
from src.v08.cleaner import clean_stock_dataframe, clean_universe_panel
from src.v08.features import compute_technical_features, compute_multi_horizon_targets
from src.v08.quality_audit import audit_dataset_quality
from src.v08.scaling_experiments import run_scaling_benchmark_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v08.pipeline_runner")


def run_v08_pipeline(
    target_tier: str = "U100",
    start_date: str = "2011-11-01",
    end_date: Optional[str] = None,
    output_dir: str = "data/v0.8",
    rate_limit_delay: float = 0.05,
) -> Dict[str, Any]:
    """
    Executes full v0.8 data expansion and validation pipeline.
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info(f"STARTING STOCKREASON v0.8 HISTORICAL DATASET EXPANSION PIPELINE")
    logger.info(f"Target Start Date: {start_date} | Target Universe: {target_tier}")
    logger.info("=" * 70)

    # 1. Initialize storage layout
    init_storage_directories(output_dir)
    raw_dir = os.path.join(output_dir, "raw", "prices")
    canonical_dir = os.path.join(output_dir, "canonical", "prices")
    meta_dir = os.path.join(output_dir, "metadata")
    reports_dir = "reports/v0.8"
    artifacts_dir = "artifacts/v0.8"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 2. Retrieve universe
    universe_df = get_universe_details("U500")
    target_tickers = get_universe(target_tier)
    # Include benchmarks and macros
    all_query_tickers = target_tickers + [b for b in BENCHMARK_INDICES if b not in target_tickers] + [m for m in MACRO_TICKERS if m not in target_tickers]

    # Include key historical delisted/distressed names
    delisted_subset = ["RCOM.NS", "JETAIRWAYS.NS", "UNITECH.NS", "DHFL.NS", "HDFC.NS", "CADILAHC.NS", "LTI.NS", "SUZLON.NS", "YESBANK.NS"]
    for sym in delisted_subset:
        if sym not in all_query_tickers:
            all_query_tickers.append(sym)

    logger.info(f"Acquiring 15-year history for {len(all_query_tickers)} securities...")

    download_results = {}
    cleaned_frames = []
    total_raw_rows = 0

    for idx, sym in enumerate(all_query_tickers, start=1):
        # A. Resilient fetch with caching & resume
        dl_res = fetch_ticker_ohlcv(
            ticker=sym,
            start_date=start_date,
            end_date=end_date,
            output_dir=raw_dir,
            checkpoint_path=os.path.join(meta_dir, "download_checkpoint.json"),
            provenance_path=os.path.join(meta_dir, "provenance.json"),
            rate_limit_delay=rate_limit_delay,
            incremental=True,
            v07_fallback_dir="data/v0.7/raw",
        )
        download_results[sym] = dl_res

        # B. Clean and canonicalize if successful
        if dl_res["status"] in ["success", "cached"] and dl_res["filepath"]:
            raw_stock_df = pd.read_parquet(dl_res["filepath"])
            total_raw_rows += len(raw_stock_df)

            cleaned_df, clean_stats = clean_stock_dataframe(raw_stock_df, sym)
            cleaned_out_path = os.path.join(canonical_dir, f"{sanitize_ticker(sym)}.parquet")
            write_parquet_atomic(cleaned_df, cleaned_out_path, optimize=True)

            if not sym.startswith("^") and not sym.endswith("=X") and not sym == "CL=F":
                cleaned_frames.append(cleaned_df)

        if idx % 20 == 0 or idx == len(all_query_tickers):
            logger.info(f"Acquisition & Cleaning Progress: [{idx}/{len(all_query_tickers)}]...")

    logger.info(f"Data Acquisition complete: {len(cleaned_frames)} clean equity series prepared ({total_raw_rows} raw observations).")

    # 3. Assemble canonical equity panel
    full_canonical_panel = pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else pd.DataFrame()
    canonical_panel_path = os.path.join(output_dir, "canonical", "canonical_stock_panel.parquet")
    write_parquet_atomic(full_canonical_panel, canonical_panel_path, optimize=True)
    logger.info(f"Saved canonical stock panel to {canonical_panel_path} ({len(full_canonical_panel)} rows).")

    # 4. Feature engineering and multi-horizon target calculation
    logger.info("Computing stationary technical features and multi-horizon prediction targets (1, 5, 21, 63, 126d)...")
    feature_panel = compute_technical_features(full_canonical_panel)
    panel_with_targets = compute_multi_horizon_targets(feature_panel, horizons=[1, 5, 21, 63, 126])

    processed_features_path = os.path.join(output_dir, "processed", "panel_features_targets.parquet")
    write_parquet_atomic(panel_with_targets, processed_features_path, optimize=True)
    logger.info(f"Saved processed feature tensor to {processed_features_path} ({len(panel_with_targets)} rows x {panel_with_targets.shape[1]} columns).")

    # 5. Dataset Quality Gate Audit
    quality_report = audit_dataset_quality(
        canonical_dir=canonical_dir,
        universe_parquet=os.path.join(output_dir, "canonical", "universe_membership", "universe_500.parquet"),
        output_report_path=os.path.join(meta_dir, "quality_report.json"),
    )
    logger.info(f"Dataset Quality Audit Status: {quality_report.get('overall_status', 'UNKNOWN')}")

    # 6. Compute Directory Manifests
    manifest_raw = compute_directory_manifest(raw_dir, os.path.join(meta_dir, "manifest_raw.json"))
    manifest_canonical = compute_directory_manifest(canonical_dir, os.path.join(meta_dir, "dataset_manifest.json"))

    # 7. Execute Scaling Experiments (Configs A through F)
    logger.info("Executing scaling experiments across historical depth and universe breadth...")
    common_test_names = target_tickers[:96]
    scaling_results = run_scaling_benchmark_experiment(
        panel_df=panel_with_targets,
        common_test_tickers=common_test_names,
        test_start_date="2025-05-01",
        horizon=21,
    )

    scaling_results_path = os.path.join(artifacts_dir, "scaling_experiment_results.json")
    with open(scaling_results_path, "w") as f:
        json.dump(scaling_results, f, indent=2)
    logger.info(f"Saved scaling experiment results to {scaling_results_path}")

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"PIPELINE RUN COMPLETE in {elapsed}s.")

    return {
        "status": "COMPLETED",
        "elapsed_seconds": elapsed,
        "raw_rows": total_raw_rows,
        "canonical_rows": len(full_canonical_panel),
        "processed_features_rows": len(panel_with_targets),
        "feature_count": panel_with_targets.shape[1],
        "quality_report": quality_report,
        "scaling_results": scaling_results,
    }


if __name__ == "__main__":
    run_v08_pipeline()
