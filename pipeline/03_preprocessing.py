#!/usr/bin/env python3
"""Pipeline Step 03: Preprocessing & Feature Quality Validation."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.ingestion import load_dataset
from src.preprocessing.validation import audit_feature_availability, check_constant_features, check_feature_collinearity
from src.features.pipeline import POPULATED_FEATURE_NAMES
from src.utils.config import CANONICAL_DATASET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("03_preprocessing")


def main():
    logger.info("Executing Pipeline Step 03: Feature Quality & Preprocessing Audit...")
    sample_cols = ["date", "symbol", "close", "ret_1d", "vol_20d", "nifty50_ret_1d", "india_vix_level"]
    df = load_dataset(CANONICAL_DATASET_PATH, columns=sample_cols)

    # Audit availability
    avail_df = audit_feature_availability(df, sample_cols)
    logger.info("Audit summary:\n%s", avail_df[["feature", "present", "coverage_pct", "unique_values"]])

    # Check constants and collinearity
    constants = check_constant_features(df, sample_cols)
    logger.info("Constant features detected: %s", constants)

    collinear = check_feature_collinearity(df, sample_cols, threshold=0.95)
    logger.info("Collinear pairs (>0.95): %s", collinear)

    print("STEP_03_PREPROCESSING_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
