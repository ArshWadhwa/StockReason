"""
Unit tests for Workstream A7: Cost assumptions verification and cost model.
"""

import pytest
from src.v071.cost_model import (
    calculate_trade_cost,
    verify_config_freshness,
    OFFICIAL_VERIFICATION_RECORDS,
)


def test_cost_parameters_verification_table():
    assert len(OFFICIAL_VERIFICATION_RECORDS) >= 8
    params = [r["parameter"] for r in OFFICIAL_VERIFICATION_RECORDS]
    assert "brokerage_delivery" in params
    assert "stt_equity_delivery" in params
    assert "nse_txn_charge_cash" in params
    assert "sebi_turnover_fee" in params
    assert "stamp_duty_delivery" in params
    assert "gst_rate" in params
    assert "dp_charges_per_sell" in params
    assert "slippage_baseline_bps" in params


def test_config_freshness_verification():
    passed, msg = verify_config_freshness("config/v071.yaml", max_months=12)
    assert passed is True, msg


def test_buy_vs_sell_charges():
    val = 100000.0  # ₹1 lakh trade
    buy_res = calculate_trade_cost(val, is_buy=True)
    sell_res = calculate_trade_cost(val, is_buy=False)

    # Stamp duty should only apply to Buy
    assert buy_res["stamp_duty"] > 0
    assert sell_res["stamp_duty"] == 0

    # DP charge should only apply to Sell
    assert sell_res["dp_fee"] > 0
    assert buy_res["dp_fee"] == 0

    # STT applies to both Buy and Sell
    assert buy_res["stt"] > 0
    assert sell_res["stt"] > 0
    assert buy_res["stt"] == sell_res["stt"]


def test_cost_multipliers():
    val = 100000.0
    c_base = calculate_trade_cost(val, is_buy=True, cost_multiplier=1.0)
    c_half = calculate_trade_cost(val, is_buy=True, cost_multiplier=0.5)
    c_double = calculate_trade_cost(val, is_buy=True, cost_multiplier=2.0)

    assert abs(c_half["total_cost"] - 0.5 * c_base["total_cost"]) < 1e-5
    assert abs(c_double["total_cost"] - 2.0 * c_base["total_cost"]) < 1e-5
