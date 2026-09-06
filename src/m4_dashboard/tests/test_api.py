import pytest
from fastapi.testclient import TestClient
from src.m4_dashboard.backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["track"] == "M4 - Product & Dashboard"
    assert data["ready_for_week4_swap"] is True

def test_market_universes(client):
    res = client.get("/api/market/universes")
    assert res.status_code == 200
    data = res.json()
    assert "indices" in data
    assert "stocks" in data
    assert len(data["stocks"]) > 0

def test_stock_prices(client):
    res = client.get("/api/stocks/RELIANCE.NS/prices?days=60")
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "RELIANCE.NS"
    assert len(data["history"]) <= 60
    assert "sma_20" in data["history"][-1]
    assert "rsi_14" in data["history"][-1]

def test_stock_sentiment(client):
    res = client.get("/api/stocks/INFY.NS/sentiment")
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "INFY.NS"
    assert "avg_sentiment_score" in data
    assert "headlines" in data
    assert len(data["headlines"]) > 0

def test_stock_prediction(client):
    res = client.get("/api/stocks/TCS.NS/predictions")
    assert res.status_code == 200
    data = res.json()
    assert "horizons" in data
    assert "1D" in data["horizons"]
    assert "1M" in data["horizons"]
    assert "ensemble" in data
    assert "shap_explainability" in data

def test_stock_signal(client):
    res = client.get("/api/stocks/RELIANCE.NS/signal")
    assert res.status_code == 200
    data = res.json()
    assert data["signal"] in ["BUY", "HOLD", "AVOID"]
    assert "reasoning" in data
    assert len(data["reasoning"]) > 0

def test_ranked_signals(client):
    res = client.get("/api/signals/ranked")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_backtest_endpoint(client):
    res = client.get("/api/backtest")
    assert res.status_code == 200
    data = res.json()
    assert "strategy_metrics" in data
    assert "random_baseline_metrics" in data
    assert "equity_curve" in data

def test_daily_digest(client):
    res = client.get("/api/digest/daily")
    assert res.status_code == 200
    data = res.json()
    assert "market_summary" in data
    assert "signal_upgrades" in data
