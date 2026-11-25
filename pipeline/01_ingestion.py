#!/usr/bin/env python3
"""Pipeline Step 01: Data Ingestion."""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.ingestion import load_dataset
from src.utils.config import CANONICAL_DATASET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("01_ingestion")


def main():
    logger.info("Executing Pipeline Step 01: Data Ingestion...")
    logger.info("Target dataset path: %s", CANONICAL_DATASET_PATH)

    if not CANONICAL_DATASET_PATH.exists():
        logger.error("Dataset not found at %s", CANONICAL_DATASET_PATH)
        sys.exit(1)

    df = load_dataset(CANONICAL_DATASET_PATH, columns=["date", "symbol", "close", "volume"])
    logger.info("Successfully ingested dataset. Rows: %d, Symbols: %d", len(df), df["symbol"].nunique())
    print("STEP_01_INGESTION_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
