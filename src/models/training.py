"""Model Training Orchestration Module."""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.models.baseline import HierarchicalShrinkageBaseline, NoChangeRandomWalkBaseline
from src.models.lightgbm_model import LightGBMForecaster
from src.models.catboost_model import CatBoostForecaster
from src.evaluation.metrics import compute_forecasting_metrics

logger = logging.getLogger("model_training")


def train_production_models(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int = 21,
) -> Dict[str, Any]:
    """
    Trains production LightGBM and CatBoost models strictly on training partition,
    using calibration partition for validation and early stopping.
    """
    logger.info("Training production baseline, LightGBM, and CatBoost models...")
    t0 = time.time()

    # 1. Fit Hierarchical Shrinkage Baseline
    base_model = HierarchicalShrinkageBaseline(horizon=horizon)
    base_model.fit(train_df, ret_col=f"target_return_nominal_{horizon}d")
    base_model.tune_alpha(cal_df, ret_col=f"target_return_nominal_{horizon}d", target_price_col=f"target_price_{horizon}d")

    # 2. Extract feature arrays
    X_train = train_df[feature_cols].fillna(0.0).astype(np.float32).values
    X_cal = cal_df[feature_cols].fillna(0.0).astype(np.float32).values

    # Train target: log return or direct return
    y_train = train_df[f"target_return_nominal_{horizon}d"].astype(np.float32).values
    y_cal = cal_df[f"target_return_nominal_{horizon}d"].astype(np.float32).values

    # 3. Train LightGBM
    lgb_model = LightGBMForecaster(feature_names=feature_cols)
    lgb_model.fit(X_train, y_train, X_val=X_cal, y_val=y_cal)

    # 4. Train CatBoost
    cb_model = CatBoostForecaster(feature_names=feature_cols)
    cb_model.fit(X_train, y_train, X_val=X_cal, y_val=y_cal)

    elapsed = round(time.time() - t0, 2)
    logger.info("Trained production models in %.2f seconds.", elapsed)

    return {
        "baseline": base_model,
        "lightgbm": lgb_model,
        "catboost": cb_model,
        "training_duration": elapsed,
        "feature_count": len(feature_cols),
    }
