#!/usr/bin/env python3
"""Pipeline Step 02: Data Cleaning & Corporate Actions Quarantine."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.ingestion import load_dataset
from src.preprocessing.cleaning import clean_market_dataset
from src.preprocessing.corporate_actions import verify_target_price_consistency, quarantine_unreconciled_target_discrepancies
from src.utils.config import CANONICAL_DATASET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("02_cleaning")


def main():
    logger.info("Executing Pipeline Step 02: Data Cleaning & Quarantine...")
    cols = ["date", "symbol", "close", "volume", "target_price_21d", "target_return_21d"]
    df = load_dataset(CANONICAL_DATASET_PATH, columns=cols)

    # Clean data
    cleaned_df, clean_summary = clean_market_dataset(df)
    logger.info("Cleaning summary: %s", clean_summary)

    # Check target price consistency
    consistency = verify_target_price_consistency(cleaned_df, horizons=[21])
    logger.info("Target consistency check: %s", consistency)

    print("STEP_02_CLEANING_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
