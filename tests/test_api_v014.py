"""Unit Tests for Consolidated StockReason API."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["version"] == "0.14.0"


def test_api_system_status(client):
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ACTIVE"
    assert data["baseline_mae_inr"] == 103.40
    assert data["baseline_mape_pct"] == 8.7648


def test_api_post_forecast(client):
    payload = {
        "symbol": "TCS",
        "current_price": 3800.0,
        "horizon_days": 21,
    }
    res = client.post("/api/forecast", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "TCS"
    assert data["reference_close_price"] == 3800.0
    assert data["predicted_price"] == 3800.0
    assert data["predicted_return"] == 0.0
    assert data["model_name"] == "NoChangeRandomWalk"
    assert "STATISTICAL_BASELINE_FALLBACK" in data["model_status"]
    assert "target_trading_date" in data


def test_api_get_forecast(client):
    res = client.get("/api/stocks/INFY/forecast?price=1600.0&horizon=21")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "INFY"
    assert data["predicted_price"] == 1600.0
