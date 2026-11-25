#!/usr/bin/env python3
"""Pipeline Step 05: Chronological Data Splitting & Horizon Purging."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.ingestion import load_dataset
from src.preprocessing.splitting import create_purged_horizon_partitions
from src.utils.config import CANONICAL_DATASET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("05_data_splitting")


def main():
    logger.info("Executing Pipeline Step 05: Chronological Partition Splitting (21D Purged)...")
    cols = ["date", "symbol", "close", "target_price_21d", "target_return_nominal_21d"]
    # Load sample rows to demonstrate partition logic
    df = load_dataset(CANONICAL_DATASET_PATH, columns=cols)

    partitions = create_purged_horizon_partitions(
        df,
        horizon=21,
        train_end_date="2022-12-31",
        calib_start_date="2023-01-01",
        calib_end_date="2023-12-31",
        eval_start_date="2024-01-01",
    )

    m = partitions["metrics"]
    logger.info("Partition metrics: %s", m)
    print("STEP_05_DATA_SPLITTING_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
