"""
StockReason v0.7 Models and Baselines.
"""
try:
    from src.v07.models.base import BaseForecaster, extract_tabular_features
    from src.v07.models.baselines import ZeroForecaster, MarketForecaster
    from src.v07.models.linear import RidgeForecaster, ElasticNetForecaster
    from src.v07.models.tree import LightGBMForecaster, CatBoostForecaster
    from src.v07.models.deep import LSTMForecaster, GRUForecaster, DLinearForecaster
except ImportError:
    from v07.models.base import BaseForecaster, extract_tabular_features
    from v07.models.baselines import ZeroForecaster, MarketForecaster
    from v07.models.linear import RidgeForecaster, ElasticNetForecaster
    from v07.models.tree import LightGBMForecaster, CatBoostForecaster
    from v07.models.deep import LSTMForecaster, GRUForecaster, DLinearForecaster

__all__ = [
    "BaseForecaster",
    "extract_tabular_features",
    "ZeroForecaster",
    "MarketForecaster",
    "RidgeForecaster",
    "ElasticNetForecaster",
    "LightGBMForecaster",
    "CatBoostForecaster",
    "LSTMForecaster",
    "GRUForecaster",
    "DLinearForecaster",
]
