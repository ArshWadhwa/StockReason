"""Production Forecast Data Structure & Response Schema."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ForecastResponse(BaseModel):
    symbol: str
    target_horizon_days: int = 21
    target_trading_date: str
    reference_close_price: float
    predicted_price: float
    predicted_return: float
    model_name: str
    model_status: str
    price_prediction_interval_50: Optional[List[float]] = None
    price_prediction_interval_80: Optional[List[float]] = None
    price_prediction_interval_90: Optional[List[float]] = None
    probability_positive_return: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
