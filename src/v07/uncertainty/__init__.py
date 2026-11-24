"""StockReason v0.7 Uncertainty Package.

Provides calibrated directional probabilities and distribution-free conformal prediction intervals.
"""

from src.v07.uncertainty.conformal import (
    AdaptiveConformalInference,
    ConformalizedQuantileRegression,
    SplitConformalRegressor,
    compare_intervals_with_mc_dropout,
    compute_conformal_quantile,
    compute_coverage,
    compute_interval_metrics,
    compute_winkler_score,
    evaluate_coverage_by_regime,
    evaluate_coverage_over_time,
    run_rolling_conformal_simulation,
)
from src.v07.uncertainty.direction import (
    DirectionalCalibrator,
    compute_brier_score,
    compute_ece,
    compute_log_loss,
    compute_reliability_curve,
    derive_directional_probabilities,
    generate_reliability_diagram,
)
from src.v07.uncertainty.schema import (
    DeprecatedConfidenceWarning,
    StockUncertaintyForecast,
    UncertaintyHorizonOutput,
    build_uncertainty_horizon,
    upgrade_legacy_horizon_output,
)

__all__ = [
    # Direction
    "derive_directional_probabilities",
    "compute_brier_score",
    "compute_log_loss",
    "compute_ece",
    "compute_reliability_curve",
    "generate_reliability_diagram",
    "DirectionalCalibrator",
    # Conformal
    "compute_conformal_quantile",
    "compute_coverage",
    "compute_winkler_score",
    "compute_interval_metrics",
    "evaluate_coverage_by_regime",
    "evaluate_coverage_over_time",
    "SplitConformalRegressor",
    "ConformalizedQuantileRegression",
    "AdaptiveConformalInference",
    "run_rolling_conformal_simulation",
    "compare_intervals_with_mc_dropout",
    # Schema
    "DeprecatedConfidenceWarning",
    "UncertaintyHorizonOutput",
    "StockUncertaintyForecast",
    "build_uncertainty_horizon",
    "upgrade_legacy_horizon_output",
]
