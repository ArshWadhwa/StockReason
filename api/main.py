"""StockReason — FastAPI Production Intelligence Backend."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.data_hub import StockReasonDataHub
from api.service import StockReasonAPIService
from src.utils.config import RESULTS_FIGURES_DIR

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

data_hub = StockReasonDataHub()
service = StockReasonAPIService(data_hub=data_hub)


class ForecastRequest(BaseModel):
    symbol: str
    current_price: float = Field(..., gt=0)
    reference_date: Optional[str] = None
    horizon_days: int = Field(default=21, ge=1, le=126)
    features: Optional[Dict[str, float]] = None
    model_type: str = Field(default="baseline")


@app.get("/api/health")
def health():
    return {
        "status": "HEALTHY",
        "version": "0.14.0",
        "environment": "production",
        "ready_for_week4_swap": True,
        "tickers_available": len(data_hub.get_available_tickers()),
    }


@app.get("/api/system/status")
def system_status():
    return data_hub.get_system_status()


@app.post("/api/forecast")
def create_forecast(req: ForecastRequest):
    try:
        return service.generate_forecast(
            symbol=req.symbol,
            current_price=req.current_price,
            features=req.features,
            reference_date=req.reference_date,
            horizon_days=req.horizon_days,
            model_type=req.model_type,
        )
    except Exception as e:
        logger.exception("Forecast generation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stocks/{symbol}/forecast")
def get_stock_forecast(
    symbol: str,
    price: float = Query(..., gt=0),
    horizon: int = Query(default=21, ge=1, le=126),
    model_type: str = Query(default="baseline"),
):
    try:
        return service.generate_forecast(
            symbol=symbol,
            current_price=price,
            horizon_days=horizon,
            model_type=model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Market, Stock & Decision Support Endpoints ──────────────────────────────


@app.get("/api/market/universes")
def get_market_universes():
    return data_hub.get_universes()


@app.get("/api/market/overview")
def get_market_overview():
    return data_hub.get_market_overview()


@app.get("/api/stocks/{ticker}/prices")
def get_stock_prices(ticker: str, days: int = Query(default=180, ge=5, le=1500)):
    try:
        return data_hub.get_stock_prices(ticker, days=days)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/stocks/{ticker}/sentiment")
def get_stock_sentiment(ticker: str):
    return data_hub.get_stock_sentiment(ticker)


@app.get("/api/stocks/{ticker}/predictions")
def get_stock_predictions(ticker: str):
    try:
        return data_hub.get_stock_predictions(ticker)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/stocks/{ticker}/signal")
def get_stock_signal(ticker: str):
    try:
        return data_hub.get_stock_signal(ticker)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/stocks/{ticker}/risk")
def get_stock_risk(ticker: str):
    try:
        return data_hub.get_stock_risk_profile(ticker)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/signals/ranked")
def get_ranked_signals(include_all: bool = Query(default=False)):
    return data_hub.get_ranked_signals(include_all=include_all)


@app.get("/api/backtest")
def get_backtest():
    return data_hub.get_backtest()


@app.get("/api/digest/daily")
def get_daily_digest():
    return data_hub.get_daily_digest()


@app.get("/api/model/report")
def get_model_report():
    return data_hub.get_model_report()


# ── v0.14.0 Research, Benchmark & Governance Endpoints ──────────────────────


@app.get("/api/research/benchmarks")
def get_research_benchmarks():
    return data_hub.get_research_benchmarks()


@app.get("/api/research/features")
def get_research_features():
    return data_hub.get_research_features()


@app.get("/api/research/manifest")
def get_research_manifest():
    return data_hub.get_research_manifest()


@app.get("/api/research/figures/{filename}")
def get_research_figure(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid figure filename")
    fig_path = (RESULTS_FIGURES_DIR / safe_name).resolve()
    if not fig_path.exists() or fig_path.parent != RESULTS_FIGURES_DIR.resolve():
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(fig_path, media_type="image/png")
