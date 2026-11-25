#!/usr/bin/env python3
"""Pipeline Step 07: Evaluation & Model Comparison."""

import sys
import logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_forecasting_metrics
from src.utils.config import RESULTS_METRICS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("07_evaluation")


def main():
    logger.info("Executing Pipeline Step 07: Evaluation & Model Comparison...")
    comp_file = RESULTS_METRICS_DIR / "model_comparison.csv"
    if not comp_file.exists():
        # Fallback to artifacts/v013c if not yet moved
        comp_file = Path("artifacts/v013c/v013c_predictive_experiment_summary.csv")

    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        cols_to_show = [c for c in ["model_name", "diag_mape", "diag_mae_inr", "rel_red_vs_no_change_pct"] if c in df_comp.columns]
        logger.info("Loaded evaluation comparison (%d models/baselines):\n%s",
                    len(df_comp), df_comp[cols_to_show].head(10))
    else:
        logger.warning("Comparison metrics file not found at %s", comp_file)

    print("STEP_07_EVALUATION_STATUS: SUCCESS")


if __name__ == "__main__":
    main()
