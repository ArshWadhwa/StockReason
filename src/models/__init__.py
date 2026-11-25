"""StockReason Models Package."""

from src.models.baseline import (
    NoChangeRandomWalkBaseline,
    HistoricalMeanBaseline,
    HistoricalMedianBaseline,
    HierarchicalShrinkageBaseline,
)
from src.models.lightgbm_model import LightGBMForecaster
from src.models.catboost_model import CatBoostForecaster
from src.models.residual import BaselinePlusResidualLearningEngine
from src.models.adaptive import compute_recency_weights
from src.models.training import train_production_models

__all__ = [
    "NoChangeRandomWalkBaseline",
    "HistoricalMeanBaseline",
    "HistoricalMedianBaseline",
    "HierarchicalShrinkageBaseline",
    "LightGBMForecaster",
    "CatBoostForecaster",
    "BaselinePlusResidualLearningEngine",
    "compute_recency_weights",
    "train_production_models",
]
