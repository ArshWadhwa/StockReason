"""Pydantic schemas and API serialization helpers for StockReason v0.7 uncertainty.

Defines the trustworthy forecasting output schema:
- prob_up: Calibrated directional probability P(return > 0)
- prob_up_calibrated: bool flag confirming post-hoc calibration was applied
- prob_excess_up: P(return > benchmark)
- prob_net_up: P(return > transaction costs)
- interval_80: [lower, upper] conformal prediction interval at nominal 80% coverage
- interval_coverage_observed: Historical empirical coverage on out-of-fold validation data
- n_calibration_samples: Size of holdout/OOF calibration set
- model_version: Model version string (e.g. '0.7.0')
- forecast_origin_date: Date on which forecast was generated (ISO format)
- confidence: DEPRECATED heuristic confidence score (maintained for backward compatibility)
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DeprecatedConfidenceWarning(UserWarning):
    """Warning emitted when deprecated heuristic confidence is accessed or supplied."""
    pass


class UncertaintyHorizonOutput(BaseModel):
    """
    Schema for horizon forecast with calibrated directional probabilities
    and conformal prediction intervals.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    expected_return: float = Field(
        ...,
        description="Point estimate of fractional expected return over horizon (e.g. 0.012 for +1.2%).",
    )
    predicted_price: float = Field(
        ...,
        description="Implied absolute price at target horizon: current_price * (1 + expected_return).",
    )
    prob_up: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Directional probability P(return > 0).",
    )
    prob_up_calibrated: bool = Field(
        default=True,
        description="True if prob_up was calibrated via Isotonic Regression or Platt scaling on OOF data.",
    )
    prob_excess_up: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Directional probability P(return > benchmark_return).",
    )
    prob_net_up: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Directional probability P(return > transaction_costs).",
    )
    interval_80: Tuple[float, float] = Field(
        ...,
        description="Nominal 80% conformal prediction interval [lower_price, upper_price].",
    )
    interval_coverage_observed: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Empirical coverage rate observed on validation / calibration set.",
    )
    n_calibration_samples: int = Field(
        default=0,
        ge=0,
        description="Number of calibration samples used to construct the conformal interval.",
    )
    model_version: str = Field(
        default="0.7.0",
        description="Model and calibration engine version string.",
    )
    forecast_origin_date: str = Field(
        ...,
        description="Forecast origin date in ISO format YYYY-MM-DD.",
    )

    # Backward compatibility fields (Deprecated in v0.7)
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="[DEPRECATED in v0.7] Legacy heuristic confidence score. Replaced by prob_up.",
    )
    uncertainty_std: Optional[float] = Field(
        default=None,
        description="Epistemic standard deviation from MC-Dropout passes (legacy).",
    )
    lower_bound: Optional[float] = Field(
        default=None,
        description="Legacy lower bound (synced with interval_80[0]).",
    )
    upper_bound: Optional[float] = Field(
        default=None,
        description="Legacy upper bound (synced with interval_80[1]).",
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Informational note regarding uncertainty or sample size.",
    )

    @field_validator("forecast_origin_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            # Validate ISO date format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
            datetime.fromisoformat(v.split(" ")[0])
            return v
        except Exception as e:
            raise ValueError(f"Invalid forecast_origin_date format '{v}': {e}")

    @model_validator(mode="after")
    def sync_legacy_and_deprecated_fields(self) -> "UncertaintyHorizonOutput":
        # Ensure interval_80 has lower <= upper
        l, u = self.interval_80
        if l > u:
            self.interval_80 = (u, l)

        # Sync legacy lower_bound & upper_bound if not provided
        if self.lower_bound is None:
            self.lower_bound = self.interval_80[0]
        if self.upper_bound is None:
            self.upper_bound = self.interval_80[1]

        # Handle deprecated confidence
        if self.confidence is None:
            # Map legacy confidence to calibrated directional probability
            self.confidence = self.prob_up
        elif "confidence" in self.model_fields_set and self.confidence != self.prob_up:
            # Notify of deprecation if confidence was explicitly passed with differing value
            warnings.warn(
                "The 'confidence' field is deprecated in StockReason v0.7 and will be removed in a future release. "
                "Use 'prob_up' and 'interval_80' for rigorous calibrated uncertainty.",
                DeprecatedConfidenceWarning,
                stacklevel=2,
            )

        return self


class StockUncertaintyForecast(BaseModel):
    """
    Complete forecast schema for a single ticker across all horizons.
    """
    model_config = ConfigDict(extra="allow")

    ticker: str
    prediction_date: str
    current_price: float
    horizons: Dict[str, UncertaintyHorizonOutput]
    ensemble: Optional[Dict[str, Any]] = None
    shap_explainability: Optional[List[Dict[str, Any]]] = None


def build_uncertainty_horizon(
    expected_return: float,
    current_price: float,
    prob_up: float,
    interval_80: Tuple[float, float],
    forecast_origin_date: str,
    prob_up_calibrated: bool = True,
    prob_excess_up: Optional[float] = None,
    prob_net_up: Optional[float] = None,
    interval_coverage_observed: Optional[float] = None,
    n_calibration_samples: int = 0,
    model_version: str = "0.7.0",
    uncertainty_std: Optional[float] = None,
    confidence_note: Optional[str] = None,
) -> UncertaintyHorizonOutput:
    """
    Factory helper to build a validated UncertaintyHorizonOutput.
    """
    pred_price = round(current_price * (1.0 + expected_return), 2)
    return UncertaintyHorizonOutput(
        expected_return=round(float(expected_return), 6),
        predicted_price=pred_price,
        prob_up=round(float(prob_up), 4),
        prob_up_calibrated=prob_up_calibrated,
        prob_excess_up=round(float(prob_excess_up), 4) if prob_excess_up is not None else None,
        prob_net_up=round(float(prob_net_up), 4) if prob_net_up is not None else None,
        interval_80=(round(float(interval_80[0]), 2), round(float(interval_80[1]), 2)),
        interval_coverage_observed=(
            round(float(interval_coverage_observed), 4)
            if interval_coverage_observed is not None
            else None
        ),
        n_calibration_samples=n_calibration_samples,
        model_version=model_version,
        forecast_origin_date=forecast_origin_date,
        uncertainty_std=round(float(uncertainty_std), 6) if uncertainty_std is not None else None,
        confidence_note=confidence_note,
    )


def upgrade_legacy_horizon_output(
    legacy_data: Dict[str, Any],
    current_price: float,
    forecast_origin_date: str,
    calibration_q: Optional[float] = None,
    prob_up_calibrated: bool = False,
    n_calibration_samples: int = 0,
    interval_coverage_observed: Optional[float] = None,
) -> UncertaintyHorizonOutput:
    """
    Upgrade a legacy v0.6 horizon prediction dict into a v0.7 UncertaintyHorizonOutput.
    """
    ret = float(legacy_data.get("expected_return", 0.0))
    std = float(legacy_data.get("uncertainty_std", 0.02))
    pred_price = float(legacy_data.get("predicted_price", current_price * (1 + ret)))

    # If conformal margin Q is supplied, compute conformal price interval;
    # else fall back to legacy bounds with warning.
    if calibration_q is not None and calibration_q > 0:
        l_price = current_price * (1.0 + ret - calibration_q)
        u_price = current_price * (1.0 + ret + calibration_q)
    else:
        l_price = float(legacy_data.get("lower_bound", current_price * (1.0 + ret - 1.96 * std)))
        u_price = float(legacy_data.get("upper_bound", current_price * (1.0 + ret + 1.96 * std)))

    # Use legacy confidence or derive Gaussian prob_up if confidence was heuristic
    legacy_conf = legacy_data.get("confidence")
    if legacy_conf is not None and not prob_up_calibrated:
        prob_up = float(legacy_conf)
    else:
        # Approximate Gaussian P(R > 0)
        from scipy.stats import norm
        prob_up = float(norm.cdf(ret / max(std, 1e-4)))

    return UncertaintyHorizonOutput(
        expected_return=ret,
        predicted_price=pred_price,
        prob_up=min(1.0, max(0.0, prob_up)),
        prob_up_calibrated=prob_up_calibrated,
        interval_80=(l_price, u_price),
        interval_coverage_observed=interval_coverage_observed,
        n_calibration_samples=n_calibration_samples,
        model_version="0.7.0",
        forecast_origin_date=forecast_origin_date,
        uncertainty_std=std,
        confidence_note=legacy_data.get("confidence_note"),
    )
