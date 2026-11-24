"""
Unit tests for Workstreams A5 & A6: Strategy engine, hysteresis, and attribution regression.
"""

import pytest
import numpy as np
import pandas as pd

from src.v071.strategy_and_benchmarks import (
    run_hysteresis_backtest,
    run_attribution_regression,
)


@pytest.fixture
def synthetic_market():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    tickers = [f"STOCK_{i}" for i in range(30)]

    pred_records = []
    ret_records = []

    for d in dates:
        # Stationary random predictions and returns
        p = np.random.randn(len(tickers))
        r = np.random.normal(0.0005, 0.015, len(tickers))
        for i, tck in enumerate(tickers):
            pred_records.append({"date": d, "ticker": tck, "pred": p[i]})
            ret_records.append({"date": d, "ticker": tck, "return_1d": r[i]})

    preds_df = pd.DataFrame(pred_records)
    returns_df = pd.DataFrame(ret_records)
    return preds_df, returns_df


def test_hysteresis_turnover_monotonically_decreases(synthetic_market):
    preds_df, returns_df = synthetic_market

    # Band = K (plain top-K, no hysteresis buffer)
    res_band_k = run_hysteresis_backtest(preds_df, returns_df, k=10, band=10, rebalance_freq=10)
    # Band = 1.5K
    res_band_15k = run_hysteresis_backtest(preds_df, returns_df, k=10, band=15, rebalance_freq=10)
    # Band = 2K
    res_band_20k = run_hysteresis_backtest(preds_df, returns_df, k=10, band=20, rebalance_freq=10)

    # As hysteresis band widens, turnover must decrease monotonically
    assert res_band_k["annualized_turnover"] >= res_band_15k["annualized_turnover"]
    assert res_band_15k["annualized_turnover"] >= res_band_20k["annualized_turnover"]


def test_min_trade_size_reduces_trades(synthetic_market):
    preds_df, returns_df = synthetic_market

    res_no_min = run_hysteresis_backtest(preds_df, returns_df, k=10, band=15, min_trade_size=0.000)
    res_with_min = run_hysteresis_backtest(preds_df, returns_df, k=10, band=15, min_trade_size=0.03)

    assert res_with_min["total_trades"] <= res_no_min["total_trades"]


def test_attribution_regression_recovers_planted_alpha_and_beta():
    np.random.seed(42)
    n = 2000
    mkt = np.random.normal(0.0004, 0.01, n)
    rf = 0.065
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0

    planted_beta = 1.25
    planted_alpha_ann = 0.05  # 5% annual alpha
    planted_alpha_daily = planted_alpha_ann / 252.0

    noise = np.random.normal(0.0, 0.003, n)
    noise = noise - np.mean(noise)  # zero mean
    # Strategy excess = alpha + beta * mkt_excess + noise
    strat = daily_rf + planted_alpha_daily + planted_beta * (mkt - daily_rf) + noise

    res = run_attribution_regression(strat, mkt, risk_free_rate=rf)

    assert abs(res["beta"] - planted_beta) < 0.02
    assert abs(res["alpha_ann"] - planted_alpha_ann) < 0.01
    assert res["r2"] > 0.80
