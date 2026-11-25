"""StockReason — Production API Service & Transparent Intelligence Engine."""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, date
import numpy as np
import pandas as pd
from src.utils.calendar import get_target_trading_date
from src.inference.predictor import StockReasonPredictor
from src.inference.forecast import ForecastResponse


class StockReasonAPIService:
    """
    Transparent API Service enforcing fail-safe fallback,
    nominal-price reconstruction, and exchange calendar compliance.
    """
    def __init__(
        self,
        baseline_price_mae: float = 103.40,
        model_version: str = "0.14.0",
        is_probability_calibrated: bool = False,
    ):
        self.baseline_price_mae = baseline_price_mae
        self.model_version = model_version
        self.is_probability_calibrated = is_probability_calibrated
        self.predictor = StockReasonPredictor(
            model_type="baseline",
            is_probability_calibrated=is_probability_calibrated,
        )

    def generate_forecast(
        self,
        symbol: str,
        current_price: float,
        features: Optional[Dict[str, float]] = None,
        reference_date: Optional[str] = None,
        horizon_days: int = 21,
    ) -> Dict[str, Any]:
        """
        Generates production forecast following transparency schema.
        """
        resp = self.predictor.predict_single(
            symbol=symbol,
            close=current_price,
            features=features,
            reference_date=reference_date,
            horizon=horizon_days,
        )
        data = resp.model_dump()
        data["model_version"] = self.model_version
        data["baseline_reference_mae_inr"] = self.baseline_price_mae
        return data
