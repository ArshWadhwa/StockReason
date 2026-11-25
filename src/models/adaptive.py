"""Bounded Adaptive & Exponential Recency Weighting Module."""

from typing import Optional
import numpy as np
import pandas as pd


def compute_recency_weights(
    dates: pd.Series,
    halflife_years: float = 3.0,
    min_weight: float = 0.20,
    max_weight: float = 1.00,
) -> np.ndarray:
    """
    Computes bounded exponential recency weights:
    decay = exp(-ln(2) * delta_days / (halflife_years * 365.25))
    clipped to [min_weight, max_weight].
    """
    dt = pd.to_datetime(dates)
    max_dt = dt.max()
    delta_days = (max_dt - dt).dt.total_seconds() / (24 * 3600.0)

    halflife_days = halflife_years * 365.25
    decay_lambda = np.log(2.0) / halflife_days
    raw_weights = np.exp(-decay_lambda * delta_days.values)

    weights = np.clip(raw_weights, min_weight, max_weight).astype(np.float32)
    # Normalize mean weight to 1.0
    return weights / np.mean(weights)
