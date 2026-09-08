import sys
import os
# Add project root to python path to allow running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import logging
import asyncio
import subprocess
from datetime import datetime, time as dtime
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.m4_dashboard.backend.models import (
    HealthResponse, UniverseResponse, PriceHistoryResponse,
    SentimentResponse, PredictionResponse, SignalResponse,
    BacktestResponse, DailyDigest
)
from src.m4_dashboard.backend.data_service import DataService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m4_backend")

# ── Generate predictions before starting the API ──────────────────────
logger.info("Running M3 prediction pipeline…")
try:
    from src.m3_modeling.predict import generate_predictions
    generate_predictions()
    logger.info("M3 predictions generated successfully")
except Exception as e:
    logger.warning(f"M3 prediction generation failed: {e}. Will use existing predictions.json if available.")

app = FastAPI(
    title="StockReason Intelligence API",
    description="Confidence-Aware Decision Support API for NIFTY 50 — Real Data Mode",
    version="3.0.0"
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


# ══════════════════════════════════════════════════════════════════════
# Smart Background Refresh Architecture (Issue #4)
# ══════════════════════════════════════════════════════════════════════
# Instead of re-running the entire M1+M2+M3 pipeline every 60 seconds,
# we split into three tiers:
#
# 1. LIVE PRICE POLLER: Every 60s during market hours — lightweight
#    yfinance call to update current prices in memory.
#
# 2. DAILY PIPELINE: Once per day after market close (4:00 PM IST) —
#    full M1+M2+M3 refresh from refresh_data.py.
#
# 3. WEEKLY RETRAIN CHECK: Sunday midnight — checks drift and triggers
#    retraining if accuracy has degraded.
# ══════════════════════════════════════════════════════════════════════

def _is_market_hours() -> bool:
    """Check if current time is within IST market hours (9:15 AM - 3:30 PM)."""
    now = datetime.now()
    market_open = dtime(9, 15)
    market_close = dtime(15, 30)
    # Monday=0, Sunday=6
    return now.weekday() < 5 and market_open <= now.time() <= market_close


async def live_price_poller():
    """
    Tier 1: Lightweight live price polling every 60 seconds.
    Only fetches current prices from Yahoo Finance — no disk I/O, no ML inference.
    Only runs during market hours to avoid wasting API calls.
    """
    while True:
        try:
            if _is_market_hours():
                logger.info("[Poller] Fetching live prices (market hours)...")
                tickers = list(data_service.prices.keys())
                if tickers:
                    live_prices = data_service._fetch_live_prices(tickers)
                    # Update in-memory prices only
                    for ticker, live_data in live_prices.items():
                        if ticker in data_service.prices:
                            prev_close = data_service.prices[ticker].get("previous_close")
                            new_price = live_data["price"]
                            data_service.prices[ticker]["current_price"] = new_price
                            data_service.prices[ticker]["live_price_source"] = "yahoo_finance_live"
                            if prev_close and prev_close > 0:
                                data_service.prices[ticker]["day_change_pct"] = round(
                                    ((new_price - prev_close) / prev_close) * 100, 2
                                )
                    logger.info(f"[Poller] Updated {len(live_prices)} live prices")
            else:
                logger.debug("[Poller] Outside market hours — skipping")
        except Exception as e:
            logger.error(f"[Poller] Error: {e}")

        await asyncio.sleep(60)


async def daily_pipeline_runner():
    """
    Tier 2: Full M1+M2+M3 pipeline refresh once per day after market close.
    Runs at ~4:00 PM IST (16:00). Checks every 5 minutes if it's time.
    """
    last_run_date = None

    while True:
        now = datetime.now()
        today = now.date()
        current_time = now.time()

        # Run once per day, after 4:00 PM IST, on weekdays only
        should_run = (
            now.weekday() < 5 and           # Weekday
            current_time >= dtime(16, 0) and  # After 4 PM
            last_run_date != today            # Haven't run today
        )

        if should_run:
            logger.info("[Daily] Starting full M1+M2+M3 pipeline refresh...")
            try:
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "refresh_data.py",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=root_dir
                )
                stdout, stderr = await process.communicate()
                if process.returncode == 0:
                    logger.info("[Daily] Pipeline refresh complete. Reloading DataService.")
                    data_service._load_all()
                    last_run_date = today
                else:
                    logger.error(f"[Daily] Pipeline failed (code {process.returncode}):\n{stderr.decode()}")
            except Exception as e:
                logger.error(f"[Daily] Error: {e}")

            # Also run prediction tracker for drift monitoring
            try:
                from src.m3_modeling.prediction_tracker import compute_accuracy_report
                report = compute_accuracy_report()
                if report.get('drift_detected'):
                    logger.warning("[Daily] ⚠️  Model drift detected! Consider retraining.")
            except Exception as e:
                logger.warning(f"[Daily] Prediction tracker error: {e}")

        # Check every 5 minutes
        await asyncio.sleep(300)


async def weekly_retrain_check():
    """
    Tier 3: Weekly check (Sunday midnight) to see if model needs retraining.
    Only triggers retraining if drift was detected during the week.
    """
    last_check_week = None

    while True:
        now = datetime.now()
        current_week = now.isocalendar()[1]

        # Run on Sunday (weekday=6) after midnight, once per week
        should_check = (
            now.weekday() == 6 and
            last_check_week != current_week
        )

        if should_check:
            logger.info("[Weekly] Running weekly retrain check...")
            try:
                from src.m3_modeling.prediction_tracker import check_drift
                drift = check_drift()

                if drift:
                    logger.warning("[Weekly] Drift detected — triggering model retraining!")
                    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "src.m3_modeling.train_pipeline",
                        "--data_dir", "data/processed",
                        "--epochs", "50",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=root_dir
                    )
                    stdout, stderr = await process.communicate()
                    if process.returncode == 0:
                        logger.info("[Weekly] Retraining complete! Regenerating predictions...")
                        # Re-generate predictions with new model
                        pred_process = await asyncio.create_subprocess_exec(
                            sys.executable, "-m", "src.m3_modeling.predict",
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=root_dir
                        )
                        await pred_process.communicate()
                        data_service._load_all()
                    else:
                        logger.error(f"[Weekly] Retraining failed:\n{stderr.decode()}")
                else:
                    logger.info("[Weekly] No drift detected — model is healthy, skipping retrain.")

                last_check_week = current_week
            except Exception as e:
                logger.error(f"[Weekly] Error: {e}")

        # Check every hour
        await asyncio.sleep(3600)


@app.on_event("startup")
async def startup_event():
    """Start the three-tier background refresh system."""
    asyncio.create_task(live_price_poller())
    asyncio.create_task(daily_pipeline_runner())
    asyncio.create_task(weekly_retrain_check())
    logger.info("Background tasks started: Live Poller (60s) | Daily Pipeline (4PM) | Weekly Retrain (Sunday)")


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint confirming API status."""
    return HealthResponse()

@app.get("/api/system/status", tags=["System"])
def get_system_status():
    """Returns system status including last data refresh timestamp."""
    return data_service.get_system_status()

@app.get("/api/system/accuracy", tags=["System"])
def get_accuracy_report():
    """Returns model accuracy and drift monitoring report."""
    return data_service.get_accuracy_report()

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

@app.get("/api/stocks/{ticker}/predictions", tags=["Model Predictions"])
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
def get_ranked_signals(include_all: bool = Query(False)):
    """Returns all universe tickers ranked by signal conviction score (Watchlist/Portfolio View)."""
    return data_service.get_ranked_signals(include_all=include_all)

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
