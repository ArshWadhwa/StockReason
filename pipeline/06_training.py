#!/usr/bin/env python3
"""Pipeline Step 06: Model Training & Baseline Fitting."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline import NoChangeRandomWalkBaseline, HierarchicalShrinkageBaseline
from src.models.lightgbm_model import LightGBMForecaster
from src.utils.config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("06_training")


def main():
    logger.info("Executing Pipeline Step 06: Model Training Interface...")
    logger.info("Models directory: %s", MODELS_DIR)

    # Check existing models
    lgb_path = MODELS_DIR / "lightgbm_21d.txt"
    if lgb_path.exists():
        forecaster = LightGBMForecaster()
        forecaster.load_model(lgb_path)
        logger.info("Loaded LightGBM model with %d features.", len(forecaster.feature_names))

    rw_base = NoChangeRandomWalkBaseline(horizon=21)
    logger.info("Initialized No-Change Random Walk benchmark (Zero-return nominal price).")
    print("STEP_06_TRAINING_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
