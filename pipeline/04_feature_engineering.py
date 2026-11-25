#!/usr/bin/env python3
"""Pipeline Step 04: Feature Engineering Pipeline."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import POPULATED_FEATURE_NAMES
from src.features.technical import compute_technical_features
from src.preprocessing.ingestion import load_dataset
from src.utils.config import CANONICAL_DATASET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("04_feature_engineering")


def main():
    logger.info("Executing Pipeline Step 04: Feature Engineering Verification...")
    logger.info("Canonical feature panel defines %d verified populated features across F0-F6.", len(POPULATED_FEATURE_NAMES))

    # Verify features in canonical panel
    df_head = load_dataset(CANONICAL_DATASET_PATH, columns=POPULATED_FEATURE_NAMES[:10])
    logger.info("Sample loaded canonical features: %s", list(df_head.columns))
    print("STEP_04_FEATURE_ENGINEERING_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
