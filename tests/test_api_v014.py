"""Unit & Integration Tests for Consolidated StockReason API."""

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
    assert data["tickers_loaded"] >= 16


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
    assert data["price_prediction_interval_80"] is not None


def test_api_get_forecast(client):
    res = client.get("/api/stocks/INFY/forecast?price=1600.0&horizon=21")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "INFY"
    assert data["predicted_price"] == 1600.0


def test_api_market_and_stock_endpoints(client):
    res_u = client.get("/api/market/universes")
    assert res_u.status_code == 200
    assert len(res_u.json()["stocks"]) >= 16

    res_o = client.get("/api/market/overview")
    assert res_o.status_code == 200
    assert len(res_o.json()["indices"]) >= 3

    res_p = client.get("/api/stocks/RELIANCE.NS/prices?days=60")
    assert res_p.status_code == 200
    assert len(res_p.json()["history"]) == 60

    res_s = client.get("/api/stocks/RELIANCE.NS/sentiment")
    assert res_s.status_code == 200
    assert "headlines" in res_s.json()

    res_pr = client.get("/api/stocks/RELIANCE.NS/predictions")
    assert res_pr.status_code == 200
    assert "1M" in res_pr.json()["horizons"]

    res_sig = client.get("/api/stocks/RELIANCE.NS/signal")
    assert res_sig.status_code == 200
    assert res_sig.json()["signal"] in ("BUY", "HOLD", "AVOID")

    res_risk = client.get("/api/stocks/RELIANCE.NS/risk")
    assert res_risk.status_code == 200
    risk = res_risk.json()
    assert len(risk["conformal_intervals"]) == 4
    assert "ewma_94_pct" in risk["volatility_latest"]
    assert "garch_params" in risk
    assert "probability_calibration" in risk


def test_api_portfolio_and_research_endpoints(client):
    res_ranked = client.get("/api/signals/ranked?include_all=true")
    assert res_ranked.status_code == 200
    assert len(res_ranked.json()) >= 16

    res_bt = client.get("/api/backtest")
    assert res_bt.status_code == 200
    assert len(res_bt.json()["equity_curve"]) > 10

    res_dig = client.get("/api/digest/daily")
    assert res_dig.status_code == 200
    assert "market_summary" in res_dig.json()

    res_rep = client.get("/api/model/report")
    assert res_rep.status_code == 200
    assert res_rep.json()["has_edge"] is False

    res_bench = client.get("/api/research/benchmarks")
    assert res_bench.status_code == 200
    assert len(res_bench.json()["model_results"]) == 15

    res_feat = client.get("/api/research/features")
    assert res_feat.status_code == 200
    assert res_feat.json()["populated_features_count"] == 68

    res_man = client.get("/api/research/manifest")
    assert res_man.status_code == 200
    figs = res_man.json()["figures"]
    assert len(figs) >= 5

    res_fig = client.get(figs[0]["url"])
    assert res_fig.status_code == 200
    assert res_fig.headers["content-type"] == "image/png"
