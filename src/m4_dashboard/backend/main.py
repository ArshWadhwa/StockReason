import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    HealthResponse, UniverseResponse, PriceHistoryResponse,
    SentimentResponse, PredictionResponse, SignalResponse,
    BacktestResponse, DailyDigest
)
from .data_service import DataService
from .signal_engine import RuleBasedSignalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m4_backend")

app = FastAPI(
    title="StockReason Intelligence API",
    description="Confidence-Aware Decision Support API for NIFTY 50 and Sectoral Indices (M4 Track)",
    version="1.0.0"
)

# Enable CORS for React Frontend (standard Vite dev ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_service = DataService()
signal_engine = RuleBasedSignalEngine()

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Week 1 Core Milestone: Health check endpoint confirming API status & contract compatibility."""
    return HealthResponse()

@app.get("/api/market/universes", tags=["Market Data"])
def get_universes():
    """Returns covered indices and stocks across NIFTY 50, Nifty Bank, Nifty IT, Nifty Next 50."""
    return data_service.get_universes()

@app.get("/api/market/overview", tags=["Market Data"])
def get_market_overview():
    """Returns macro context, index benchmarks, India VIX level, and current market regime."""
    return data_service.get_market_overview()

@app.get("/api/stocks/{ticker}/prices", response_model=PriceHistoryResponse, tags=["Price & Indicators"])
def get_stock_prices(ticker: str, days: int = Query(180, ge=10, le=365)):
    """Returns OHLCV price series + technical indicator columns (SMA, EMA, RSI, MACD, Bollinger Bands)."""
    data = data_service.get_stock_prices(ticker, days=days)
    if not data:
        raise HTTPException(status_code=404, detail=f"Stock ticker '{ticker}' not found in active universe")
    return data

@app.get("/api/stocks/{ticker}/sentiment", response_model=SentimentResponse, tags=["News & Sentiment"])
def get_stock_sentiment(ticker: str):
    """Returns FinBERT headline scores, daily aggregate sentiment, and sentiment-price divergence alerts."""
    data = data_service.get_stock_sentiment(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Sentiment records for ticker '{ticker}' not found")
    return data

@app.get("/api/stocks/{ticker}/predictions", response_model=PredictionResponse, tags=["Model Predictions"])
def get_stock_predictions(ticker: str):
    """Returns multi-horizon return forecasts (1D, 1W, 1M, 6M) with MC Dropout confidence, ensemble check, and SHAP."""
    data = data_service.get_stock_prediction(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Predictions for ticker '{ticker}' not found")
    return data

@app.get("/api/stocks/{ticker}/signal", response_model=SignalResponse, tags=["Signal Engine"])
def get_stock_signal(ticker: str):
    """Returns rule-based Buy / Hold / Avoid decision with full transparent reasoning trail."""
    data = data_service.get_stock_signal(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Signal for ticker '{ticker}' not found")
    return data

@app.get("/api/signals/ranked", tags=["Signal Engine"])
def get_ranked_signals():
    """Returns all universe tickers ranked by signal conviction score (Watchlist/Portfolio View)."""
    return data_service.get_ranked_signals()

@app.get("/api/backtest", response_model=BacktestResponse, tags=["Backtesting & Validation"])
def get_backtest_results():
    """Returns backtest equity curve, metrics (Sharpe, max drawdown, win rate), and random-strategy baseline comparison."""
    return data_service.get_backtest()

@app.get("/api/digest/daily", response_model=DailyDigest, tags=["Daily Digest"])
def get_daily_digest():
    """Returns auto-generated plain-English summary of signal changes and market commentary."""
    return data_service.get_daily_digest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
