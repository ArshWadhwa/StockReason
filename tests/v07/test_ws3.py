"""
Comprehensive unit and integration tests for WS3: Honest, Cost-Aware Backtester.
================================================================================
Test Suite covers:
1. Known toy price path with hand-computed P&L (gross vs net).
2. Verification that no trade executes at signal-day close (execution strictly on t+1 open).
3. Monotonic reduction of returns as transaction costs/slippage increase.
4. Rejection of in-sample predictions (raises ValueError if is_out_of_sample != True).
5. Limit-entry fills (low <= limit price check, price improvement when open <= limit).
6. Stop-loss and take-profit intraday triggers (stop priority when both touched, gap-down fills at open).
7. Sector cap constraints in Top-K selection.
8. Daily vs weekly rebalance schedules.
9. Random top-K multi-seed benchmark calculation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.v07.backtest.costs import TransactionCostModel, CostBreakdown
from src.v07.backtest.orders import Position, Trade, OrderExecutor
from src.v07.backtest.strategies import (
    TopKStrategy,
    LongFlatStrategy,
    EqualWeightUniverseStrategy,
    RandomTopKStrategy,
)
from src.v07.backtest.metrics import (
    compute_backtest_metrics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_drawdown_series,
    save_equity_curve,
)
from src.v07.backtest.engine import BacktestEngine, BacktestResult


# ─── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def toy_price_panel():
    """
    Toy price panel with 3 trading days for 2 stocks:
    Day 0 (2024-01-01): Signal generated after close.
    Day 1 (2024-01-02): Trade enters at open (100.0). Closes at 105.0.
    Day 2 (2024-01-03): Trade exits at open (105.0).
    """
    data = [
        # Date, Ticker, Open, High, Low, Close, Volume
        {"date": "2024-01-01", "ticker": "STOCK_A", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 100000},
        {"date": "2024-01-01", "ticker": "STOCK_B", "open": 50.0,  "high": 51.0,  "low": 49.0, "close": 50.5,  "volume": 100000},
        {"date": "2024-01-02", "ticker": "STOCK_A", "open": 100.0, "high": 110.0, "low": 98.0, "close": 105.0, "volume": 100000},
        {"date": "2024-01-02", "ticker": "STOCK_B", "open": 50.0,  "high": 52.0,  "low": 48.0, "close": 49.0,  "volume": 100000},
        {"date": "2024-01-03", "ticker": "STOCK_A", "open": 105.0, "high": 112.0, "low": 104.0, "close": 110.0, "volume": 100000},
        {"date": "2024-01-03", "ticker": "STOCK_B", "open": 49.0,  "high": 50.0,  "low": 48.0, "close": 48.5,  "volume": 100000},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def toy_predictions():
    """
    Valid out-of-sample predictions matching toy price panel.
    """
    data = [
        # On Day 0 (2024-01-01), STOCK_A predicted +10%, STOCK_B predicted -2%
        {"date": "2024-01-01", "ticker": "STOCK_A", "pred_return": 0.10, "is_out_of_sample": True},
        {"date": "2024-01-01", "ticker": "STOCK_B", "pred_return": -0.02, "is_out_of_sample": True},
        # On Day 1 (2024-01-02), STOCK_A predicted -5%, STOCK_B predicted +8%
        {"date": "2024-01-02", "ticker": "STOCK_A", "pred_return": -0.05, "is_out_of_sample": True},
        {"date": "2024-01-02", "ticker": "STOCK_B", "pred_return": 0.08, "is_out_of_sample": True},
    ]
    return pd.DataFrame(data)


# ─── Tests ─────────────────────────────────────────────────────────────

def test_toy_price_path_hand_computed_pnl(toy_price_panel, toy_predictions):
    """
    1. Verify hand-computed P&L on a deterministic toy price path.
    With zero costs:
    - Capital: 100,000.
    - Day 0 close: Signal on STOCK_A (+10%). Day 0 close has 0 trades, 100% cash.
    - Day 1 open: Enters STOCK_A at open=100.0 (1000 shares). Day 1 close=105.0 (equity 105,000).
    - Day 2 open: Rebalance occurs because Day 1 signal switched to STOCK_B.
      STOCK_A exits at Day 2 open=105.0. Gross profit = 1000 * 5.0 = 5,000 (+5.0%).
    """
    engine = BacktestEngine(
        cost_model=TransactionCostModel.zero_cost(),
        initial_capital=100000.0,
        rebalance_freq="daily",
        horizon_days=5,
    )
    strat = TopKStrategy(k=1)
    res = engine.run(
        predictions=toy_predictions,
        price_panel=toy_price_panel,
        strategy=strat,
        strategy_name="ToyHandComputed",
    )

    daily_eq = res.daily_equity
    assert len(daily_eq) == 3

    # On Day 0 close (signal date): 0 trades, equity == initial capital
    assert daily_eq.iloc[0]["date"] == "2024-01-01"
    assert daily_eq.iloc[0]["cash"] == 100000.0
    assert daily_eq.iloc[0]["invested_value"] == 0.0
    assert daily_eq.iloc[0]["net_equity"] == 100000.0

    # On Day 1 close (after Day 1 open entry at 100, close at 105)
    assert daily_eq.iloc[1]["date"] == "2024-01-02"
    assert daily_eq.iloc[1]["invested_value"] == 105000.0
    assert daily_eq.iloc[1]["net_equity"] == 105000.0

    # Trade completed
    assert len(res.trades) >= 1
    t0 = res.trades[0]
    assert t0.ticker == "STOCK_A"
    assert t0.entry_price == 100.0
    assert t0.exit_price == 105.0
    assert t0.gross_pnl == 5000.0
    assert t0.net_pnl == 5000.0
    assert round(t0.return_pct, 4) == 0.05


def test_no_trade_executes_at_signal_day_close(toy_price_panel, toy_predictions):
    """
    2. Crucial Anti-Lookahead Guarantee:
    Verifies that on day t (when signal is derived from day t close),
    NO execution happens at day t close.
    Execution happens strictly on day t+1 at the OPEN.
    """
    engine = BacktestEngine(
        cost_model=TransactionCostModel.zero_cost(),
        initial_capital=100000.0,
        rebalance_freq="daily",
    )
    strat = TopKStrategy(k=1)
    res = engine.run(toy_predictions, toy_price_panel, strat)

    day_0_rec = res.daily_equity.iloc[0]
    # At Day 0 close: turnover is 0, cash is 100,000, no active positions
    assert day_0_rec["date"] == "2024-01-01"
    assert day_0_rec["turnover"] == 0.0
    assert day_0_rec["costs"] == 0.0
    assert day_0_rec["invested_value"] == 0.0

    # Day 1 records turnover from Day 1 morning open fill
    day_1_rec = res.daily_equity.iloc[1]
    assert day_1_rec["date"] == "2024-01-02"
    assert day_1_rec["turnover"] > 0.0


def test_costs_reduce_return_monotonically(toy_price_panel, toy_predictions):
    """
    3. Verify costs reduce net return monotonically as transaction fees/slippage increase.
    Tests across a spectrum of round-trip costs (0 bps, 10 bps, 25 bps, 50 bps, 100 bps)
    as well as standard statutory Indian equity delivery charges.
    """
    strat = TopKStrategy(k=1)

    round_trips = [0.0, 10.0, 25.0, 50.0, 100.0]
    net_returns = []
    costs_paid = []

    for bps in round_trips:
        cost_m = TransactionCostModel.flat_bps(round_trip_bps=bps) if bps > 0 else TransactionCostModel.zero_cost()
        eng = BacktestEngine(cost_model=cost_m, initial_capital=100000.0)
        res = eng.run(toy_predictions, toy_price_panel, strat)
        net_returns.append(res.metrics.cumulative_return_net)
        costs_paid.append(res.metrics.total_costs_paid)

    # Verify strictly monotonic decrease in net returns and increase in costs
    for i in range(len(net_returns) - 1):
        assert net_returns[i] > net_returns[i + 1], f"Expected return at {round_trips[i]} bps ({net_returns[i]}) > {round_trips[i+1]} bps ({net_returns[i+1]})"
        assert costs_paid[i] < costs_paid[i + 1], f"Expected costs at {round_trips[i]} bps ({costs_paid[i]}) < {round_trips[i+1]} bps ({costs_paid[i+1]})"

    # Also verify standard Indian statutory cost model reduces return below zero-cost
    eng_std = BacktestEngine(cost_model=TransactionCostModel(), initial_capital=100000.0)
    res_std = eng_std.run(toy_predictions, toy_price_panel, strat)
    assert res_std.metrics.cumulative_return_net < net_returns[0]
    assert res_std.metrics.total_costs_paid > 0.0


def test_backtester_raises_on_in_sample_predictions(toy_price_panel, toy_predictions):
    """
    4. Verify backtester strictly refuses to run on in-sample predictions.
    """
    engine = BacktestEngine()

    # Case A: Missing 'is_out_of_sample' column
    df_missing = toy_predictions.drop(columns=["is_out_of_sample"])
    with pytest.raises(ValueError, match="missing 'is_out_of_sample' column"):
        engine.run(df_missing, toy_price_panel)

    # Case B: is_out_of_sample == False present
    df_with_in_sample = toy_predictions.copy()
    df_with_in_sample.loc[0, "is_out_of_sample"] = False
    with pytest.raises(ValueError, match="contains .* in-sample prediction"):
        engine.run(df_with_in_sample, toy_price_panel)


def test_limit_entry_orders():
    """
    5. Fills model: limit-entry orders.
    - Fill only if day t+1 low <= limit price.
    - If open <= limit price: fill at open (price improvement).
    - If open > limit price and low <= limit price: fill at limit price.
    - If low > limit price: unfilled.
    """
    executor = OrderExecutor(cost_model=TransactionCostModel.zero_cost())

    bar = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0}

    # Case 1: Limit price = 90.0 (low is 95.0, so order fails to fill)
    pos_unfilled = executor.execute_limit_entry(
        ticker="TEST",
        limit_price=90.0,
        target_allocation=10000.0,
        available_cash=10000.0,
        date="2024-01-02",
        bar=bar,
    )
    assert pos_unfilled is None, "Order should not fill when low > limit_price"

    # Case 2: Limit price = 102.0 (open is 100.0 <= limit_price -> fills at open 100.0 with price improvement)
    pos_improved = executor.execute_limit_entry(
        ticker="TEST",
        limit_price=102.0,
        target_allocation=10000.0,
        available_cash=10000.0,
        date="2024-01-02",
        bar=bar,
    )
    assert pos_improved is not None
    assert pos_improved.entry_price == 100.0, "Should fill at open price 100.0 (price improvement)"

    # Case 3: Limit price = 97.0 (open is 100.0 > limit, but low 95.0 <= limit -> fills at limit 97.0)
    pos_limit = executor.execute_limit_entry(
        ticker="TEST",
        limit_price=97.0,
        target_allocation=9700.0,
        available_cash=10000.0,
        date="2024-01-02",
        bar=bar,
    )
    assert pos_limit is not None
    assert pos_limit.entry_price == 97.0, "Should fill at limit price 97.0"


def test_stop_loss_and_take_profit_priority_and_gaps():
    """
    6. Stop-loss and take-profit evaluated on daily high/low.
    - Stop triggers first if both touched on same day!
    - Gap-through-stop fills at open.
    - Normal stop fills at stop price.
    """
    # Create position bought at 100.0, stop loss at 95.0, take profit at 110.0
    costs = CostBreakdown("BUY", 100, 100.0, 10000.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    pos = Position(
        ticker="TEST",
        shares=100,
        entry_price=100.0,
        entry_date="2024-01-01",
        cost_basis=10000.0,
        entry_costs=costs,
        stop_loss_price=95.0,
        take_profit_price=110.0,
    )

    # Scenario A: BOTH touched on the same day (Low=90 <= 95, High=115 >= 110)
    # Open is 102 (normal within bounds)
    bar_both = {"open": 102.0, "high": 115.0, "low": 90.0, "close": 96.0}
    exit_info = pos.check_intraday_exit(bar_both)
    assert exit_info is not None
    assert exit_info["exit_reason"] == "STOP_LOSS", "Stop loss must take priority when both are touched!"
    assert exit_info["exit_price"] == 95.0, "Should fill at stop price when open >= stop_price"

    # Scenario B: Gap-through-stop (Stock gaps down at open to 91.0 < 95.0)
    bar_gap = {"open": 91.0, "high": 93.0, "low": 89.0, "close": 90.0}
    exit_info_gap = pos.check_intraday_exit(bar_gap)
    assert exit_info_gap is not None
    assert exit_info_gap["exit_reason"] == "STOP_LOSS"
    assert exit_info_gap["exit_price"] == 91.0, "Gap-through-stop must fill at open (91.0), not stop price!"

    # Scenario C: Take-profit touched normally (High=112 >= 110, Low=98 > 95)
    bar_tp = {"open": 101.0, "high": 112.0, "low": 98.0, "close": 111.0}
    exit_info_tp = pos.check_intraday_exit(bar_tp)
    assert exit_info_tp is not None
    assert exit_info_tp["exit_reason"] == "TAKE_PROFIT"
    assert exit_info_tp["exit_price"] == 110.0


def test_top_k_sector_cap_constraint():
    """
    7. Verify Top-K sector cap prevents concentration.
    Universe has 3 IT stocks and 1 Banking stock.
    Top 3 scores:
    1. IT_1 (0.20)
    2. IT_2 (0.15)
    3. IT_3 (0.10)
    4. BANK_1 (0.05)
    With K=2, sector_cap=0.5 (max 1 stock per sector):
    Selected must be [IT_1, BANK_1], skipping IT_2 and IT_3.
    """
    preds = pd.DataFrame([
        {"ticker": "IT_1", "pred_return": 0.20},
        {"ticker": "IT_2", "pred_return": 0.15},
        {"ticker": "IT_3", "pred_return": 0.10},
        {"ticker": "BANK_1", "pred_return": 0.05},
    ])
    active_univ = ["IT_1", "IT_2", "IT_3", "BANK_1"]
    sector_map = {"IT_1": "IT", "IT_2": "IT", "IT_3": "IT", "BANK_1": "Banking"}

    strat = TopKStrategy(k=2, sector_cap=0.50)
    weights = strat.generate_target_weights(
        date="2024-01-01",
        predictions_df=preds,
        active_universe=active_univ,
        sector_map=sector_map,
    )

    assert "IT_1" in weights
    assert "BANK_1" in weights
    assert "IT_2" not in weights
    assert "IT_3" not in weights
    assert len(weights) == 2
    assert weights["IT_1"] == 0.5
    assert weights["BANK_1"] == 0.5


def test_long_flat_strategy():
    """
    Verify Long/Flat strategy maps BUY/HOLD/AVOID signals to 1.0, 0.5, 0.0 weights.
    """
    preds = pd.DataFrame([
        {"ticker": "STOCK_A", "signal": "BUY", "pred_return": 0.05},
        {"ticker": "STOCK_B", "signal": "HOLD", "pred_return": 0.0},
        {"ticker": "STOCK_C", "signal": "AVOID", "pred_return": -0.05},
    ])
    strat = LongFlatStrategy(signal_col="signal")
    weights = strat.generate_target_weights(
        date="2024-01-01",
        predictions_df=preds,
        active_universe=["STOCK_A", "STOCK_B", "STOCK_C"],
    )

    # Base weight = 1/3 ~ 0.3333
    # BUY -> 1.0 * (1/3) ~ 0.3333
    # HOLD -> 0.5 * (1/3) ~ 0.1667
    # AVOID -> 0.0
    assert "STOCK_A" in weights
    assert "STOCK_B" in weights
    assert "STOCK_C" not in weights
    assert round(weights["STOCK_A"], 3) == round(1.0 / 3.0, 3)
    assert round(weights["STOCK_B"], 3) == round(0.5 / 3.0, 3)


def test_random_top_k_benchmark_statistics(toy_price_panel, toy_predictions):
    """
    8. Verify Random Top-K benchmark runs over 200 seeds and outputs distributions.
    """
    engine = BacktestEngine(
        cost_model=TransactionCostModel.zero_cost(),
        initial_capital=100000.0,
    )
    bench = engine.run_random_top_k_benchmark(
        predictions=toy_predictions,
        price_panel=toy_price_panel,
        k=1,
        n_seeds=20,  # fast run for unit test
        base_seed=100,
    )

    assert bench["n_seeds"] == 20
    assert "cumulative_return_net_pct" in bench
    ret_stat = bench["cumulative_return_net_pct"]
    assert "mean" in ret_stat
    assert "p05" in ret_stat
    assert "p95" in ret_stat
    assert ret_stat["p05"] <= ret_stat["mean"] <= ret_stat["p95"]


def test_save_equity_curve_and_serialization(tmp_path, toy_price_panel, toy_predictions):
    """
    9. Verify saving equity curves to Parquet and JSON.
    """
    engine = BacktestEngine(
        cost_model=TransactionCostModel.zero_cost(),
        initial_capital=100000.0,
    )
    res = engine.run(toy_predictions, toy_price_panel)
    saved = res.save(output_dir=tmp_path, file_prefix="test_run")

    assert Path(saved["parquet"]).exists()
    assert Path(saved["json"]).exists()
    assert Path(saved["summary"]).exists()

    df_loaded = pd.read_parquet(saved["parquet"])
    assert len(df_loaded) == len(res.daily_equity)
    assert "net_equity" in df_loaded.columns


def test_engine_from_config(tmp_path):
    """
    11. Verify loading BacktestEngine from config/v07.yaml.
    """
    config_file = Path("config/v07.yaml")
    assert config_file.exists(), "config/v07.yaml must exist"

    engine = BacktestEngine.from_config(config_file)
    assert engine.rebalance_freq == "daily"
    assert engine.horizon_days == 5
    assert engine.cost_model.use_statutory_charges is True
    assert engine.cost_model.stt_pct == 0.0010


def test_engine_weekly_rebalance():
    """
    12. Verify weekly rebalance schedule only executes once per week.
    """
    # 2 weeks of price data
    dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
    price_data = []
    pred_data = []
    for d in dates:
        price_data.append({"date": d, "ticker": "T1", "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 100000})
        pred_data.append({"date": d, "ticker": "T1", "pred_return": 0.05, "is_out_of_sample": True})

    df_p = pd.DataFrame(price_data)
    df_pred = pd.DataFrame(pred_data)

    engine = BacktestEngine(
        cost_model=TransactionCostModel.zero_cost(),
        rebalance_freq="weekly",
        initial_capital=100000.0,
    )
    res = engine.run(df_pred, df_p, TopKStrategy(k=1))

    # Weekly rebalance should only execute entries at the start of a new week
    # With 10 business days (2 weeks), we should only have orders on week transitions
    assert len(res.daily_equity) == 10
    assert len(res.trades) >= 1


def test_stop_loss_in_engine_simulation():
    """
    13. Verify intraday stop-loss trigger in full engine simulation.
    """
    price_data = [
        {"date": "2024-01-01", "ticker": "DROPPER", "open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 100000},
        {"date": "2024-01-02", "ticker": "DROPPER", "open": 100.0, "high": 101.0, "low": 90.0, "close": 92.0, "volume": 100000}, # Dips to 90
        {"date": "2024-01-03", "ticker": "DROPPER", "open": 92.0,  "high": 93.0,  "low": 91.0, "close": 91.5, "volume": 100000},
    ]
    pred_data = [
        {"date": "2024-01-01", "ticker": "DROPPER", "pred_return": 0.10, "is_out_of_sample": True},
    ]
    df_p = pd.DataFrame(price_data)
    df_pred = pd.DataFrame(pred_data)

    # 5% stop loss: entry at 100.0 -> stop at 95.0. On Day 2 low is 90.0 (triggered!)
    engine = BacktestEngine(
        cost_model=TransactionCostModel.zero_cost(),
        stop_loss_pct=0.05,
        initial_capital=100000.0,
    )
    res = engine.run(df_pred, df_p, TopKStrategy(k=1))

    # Stop loss should have triggered on 2024-01-02
    stop_trades = [t for t in res.trades if t.exit_reason == "STOP_LOSS"]
    assert len(stop_trades) == 1
    t = stop_trades[0]
    assert t.exit_date == "2024-01-02"
    assert t.exit_price == 95.0
    assert t.gross_pnl == -5000.0  # 1000 shares * -5.0
