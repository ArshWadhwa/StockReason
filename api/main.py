"""StockReason — FastAPI Production Intelligence Backend."""

import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.service import StockReasonAPIService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stockreason_api")

app = FastAPI(
    title="StockReason Intelligence API",
    description="Multi-Horizon Quantitative Forecasting & Decision Support API for Indian Equity Markets",
    version="0.14.0",
)

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = StockReasonAPIService()


class ForecastRequest(BaseModel):
    symbol: str
    current_price: float = Field(..., gt=0)
    reference_date: Optional[str] = None
    horizon_days: int = Field(default=21, ge=1, le=126)
    features: Optional[Dict[str, float]] = None


@app.get("/api/health")
def health():
    return {
        "status": "HEALTHY",
        "version": "0.14.0",
        "environment": "production",
    }


@app.get("/api/system/status")
def system_status():
    return {
        "status": "ACTIVE",
        "version": "0.14.0",
        "primary_horizon_days": 21,
        "default_baseline": "No-Change Random Walk",
        "baseline_mae_inr": 103.40,
        "baseline_mape_pct": 8.7648,
        "ml_promotion_status": "EXPERIMENTAL_NOT_PROMOTED",
    }


@app.post("/api/forecast")
def create_forecast(req: ForecastRequest):
    try:
        return service.generate_forecast(
            symbol=req.symbol,
            current_price=req.current_price,
            features=req.features,
            reference_date=req.reference_date,
            horizon_days=req.horizon_days,
        )
    except Exception as e:
        logger.exception("Forecast generation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stocks/{symbol}/forecast")
def get_stock_forecast(
    symbol: str,
    price: float = Query(..., gt=0),
    horizon: int = Query(default=21, ge=1, le=126),
):
    try:
        return service.generate_forecast(
            symbol=symbol,
            current_price=price,
            horizon_days=horizon,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
