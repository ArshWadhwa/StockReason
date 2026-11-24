"""
StockReason v0.7 Features & Targets Package.
"""

from .targets import (
    compute_targets,
    compute_trailing_realized_volatility,
    rank_normalize_cross_section,
)
from .feature_sets import (
    SET_A_FEATURES,
    SET_B_FEATURES,
    SET_C_FEATURES,
    SET_D_FEATURES,
    get_feature_set,
    validate_stationarity,
    is_stationary_feature,
    enrich_stationary_features,
)
from .normalize import (
    normalize_cross_section,
    rank_transform_group,
    zscore_transform_group,
)
from .importance import (
    compute_correlation_matrix,
    cluster_redundant_features,
    compute_lgbm_and_permutation_importance,
)

__all__ = [
    "compute_targets",
    "compute_trailing_realized_volatility",
    "rank_normalize_cross_section",
    "SET_A_FEATURES",
    "SET_B_FEATURES",
    "SET_C_FEATURES",
    "SET_D_FEATURES",
    "get_feature_set",
    "validate_stationarity",
    "is_stationary_feature",
    "enrich_stationary_features",
    "normalize_cross_section",
    "rank_transform_group",
    "zscore_transform_group",
    "compute_correlation_matrix",
    "cluster_redundant_features",
    "compute_lgbm_and_permutation_importance",
]
