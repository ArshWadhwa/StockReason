import pytest
from src.m4_dashboard.backend.backtester import BacktestEngine

def test_backtester_metrics():
    # Synthetic positive returns
    returns = [0.01, 0.015, -0.005, 0.02, -0.01, 0.012]
    metrics = BacktestEngine.calculate_metrics(returns)
    assert metrics["cumulative_return"] > 0
    assert metrics["total_trades"] == 6
    assert metrics["win_rate"] == round((4 / 6) * 100, 2)
    assert metrics["sharpe_ratio"] > 0

def test_backtester_simulation_vs_random():
    price_series = [
        {"close": 100.0},
        {"close": 102.0},
        {"close": 101.0},
        {"close": 104.0},
        {"close": 106.0}
    ]
    signals = ["BUY", "HOLD", "BUY", "AVOID"]
    res = BacktestEngine.simulate_strategy_vs_random(price_series, signals)
    assert "strategy" in res
    assert "benchmark" in res
    assert "random_baseline" in res
    assert res["strategy"]["total_trades"] == 4
