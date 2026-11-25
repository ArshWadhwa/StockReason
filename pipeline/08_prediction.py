#!/usr/bin/env python3
"""Pipeline Step 08: Forecast Generation & Inference."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import StockReasonPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("08_prediction")


def main():
    logger.info("Executing Pipeline Step 08: Inference & Forecast Generation...")
    predictor = StockReasonPredictor(model_type="baseline")

    # Sample single prediction
    resp = predictor.predict_single(
        symbol="RELIANCE",
        close=2500.0,
        reference_date="2026-11-25",
        horizon=21,
    )
    logger.info("Sample forecast generated:\n%s", resp.model_dump_json(indent=2))
    print("STEP_08_PREDICTION_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
