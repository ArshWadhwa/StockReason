"""
Empirical Comparison: Old-Style Frictionless Backtest vs v0.7 Honest Cost-Aware Backtest.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.m4_dashboard.backend.backtester import BacktestEngine as OldBacktestEngine
from src.v07.backtest.engine import BacktestEngine as NewBacktestEngine
from src.v07.backtest.strategies import TopKStrategy, LongFlatStrategy, EqualWeightUniverseStrategy
from src.v07.backtest.costs import TransactionCostModel


def build_sample_predictions(df_prices: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates out-of-sample predictions across tickers over the evaluation window
    using an exponentially weighted momentum signal computed strictly at close(t).
    Horizon h = 5 days. Target y = (close[t+5] - close[t]) / close[t].
    """
    df = df_prices.sort_values(by=["ticker", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    records = []
    tickers = sorted(df["ticker"].unique())

    # Sector mapping
    sector_map = {
        "RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT",
        "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "BHARTIARTL.NS": "Telecom",
        "ITC.NS": "FMCG", "LT.NS": "Infrastructure", "SBIN.NS": "Banking",
        "AXISBANK.NS": "Banking", "KOTAKBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
        "BANKBARODA.NS": "Banking", "PNB.NS": "Banking", "AUBANK.NS": "Banking",
        "IDFCFIRSTB.NS": "Banking",
    }

    for tkr in tickers:
        df_tkr = df[df["ticker"] == tkr].copy().reset_index(drop=True)
        # 20-day momentum signal at close(t)
        df_tkr["signal_raw"] = df_tkr["close"].pct_change(20).ewm(span=5).mean()
        # 5-day forward return (target)
        df_tkr["y_true"] = df_tkr["close"].shift(-5) / df_tkr["close"] - 1.0

        df_eval = df_tkr[(df_tkr["date"] >= start_date) & (df_tkr["date"] <= end_date)]

        for _, row in df_eval.iterrows():
            pred_ret = float(row["signal_raw"]) if pd.notna(row["signal_raw"]) else 0.0
            prob_up = 1.0 / (1.0 + np.exp(-15.0 * pred_ret))  # logistic map
            y_val = float(row["y_true"]) if pd.notna(row["y_true"]) else 0.0

            records.append({
                "ticker": tkr,
                "date": row["date"],
                "horizon": "5d",
                "fold_id": 1,
                "model_name": "sample_ewma_momentum",
                "seed": 42,
                "pred_return": pred_ret,
                "pred_prob_up": prob_up,
                "q10": pred_ret - 0.02,
                "q50": pred_ret,
                "q90": pred_ret + 0.02,
                "y_true": y_val,
                "is_out_of_sample": True,
            })

    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("RUNNING WS3 BACKTEST COMPARISON: OLD VS NEW ENGINE")
    print("=" * 70)

    # 1. Load price panel
    df_raw = pd.read_parquet("data/processed/price_features.parquet")
    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")

    eval_start = "2024-01-01"
    eval_end = "2024-12-31"

    # 2. Generate Prediction Archive
    print(f"Generating out-of-sample predictions ({eval_start} to {eval_end})...")
    df_preds = build_sample_predictions(df_raw, eval_start, eval_end)
    out_dir = Path("artifacts/v0.7/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "sample_baseline_5d.parquet"
    df_preds.to_parquet(pred_path, index=False)
    print(f"Saved {len(df_preds)} predictions to {pred_path}")

    # Sector mapping for 16 stocks
    sector_map = {
        "RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT",
        "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "BHARTIARTL.NS": "Telecom",
        "ITC.NS": "FMCG", "LT.NS": "Infrastructure", "SBIN.NS": "Banking",
        "AXISBANK.NS": "Banking", "KOTAKBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
        "BANKBARODA.NS": "Banking", "PNB.NS": "Banking", "AUBANK.NS": "Banking",
        "IDFCFIRSTB.NS": "Banking",
    }

    # Filter price panel to evaluation period + buffer
    df_prices = df_raw[(df_raw["date"] >= "2023-12-15") & (df_raw["date"] <= "2025-01-15")].copy()

    # -------------------------------------------------------------
    # RUN 1: Old-Style Frictionless Backtest (src/m4_dashboard)
    # -------------------------------------------------------------
    print("\n--- Running Old-Style Backtester (Frictionless, Same-Close) ---")
    # Old engine ran per ticker: simulate_strategy_vs_random(price_series, signals)
    # Let's run it across the stocks and aggregate portfolio returns
    old_results = {}
    for tkr in df_preds["ticker"].unique():
        tkr_prices = df_prices[df_prices["ticker"] == tkr].sort_values("date")
        tkr_preds = df_preds[df_preds["ticker"] == tkr].sort_values("date")
        
        # Merge on date
        merged = pd.merge(tkr_prices[["date", "close"]], tkr_preds[["date", "pred_return"]], on="date")
        if len(merged) < 2:
            continue
        
        price_series = [{"close": float(c)} for c in merged["close"]]
        signals = ["BUY" if r > 0.005 else ("AVOID" if r < -0.005 else "HOLD") for r in merged["pred_return"]]
        
        res_tkr = OldBacktestEngine.simulate_strategy_vs_random(price_series, signals)
        old_results[tkr] = res_tkr

    # Average metrics across stocks for old engine
    old_strat_cums = [r["strategy"]["cumulative_return"] for r in old_results.values() if "strategy" in r]
    old_strat_sharpes = [r["strategy"]["sharpe_ratio"] for r in old_results.values() if "strategy" in r]
    old_strat_dds = [r["strategy"]["max_drawdown"] for r in old_results.values() if "strategy" in r]
    old_strat_winrates = [r["strategy"]["win_rate"] for r in old_results.values() if "strategy" in r]

    avg_old_cum = float(np.mean(old_strat_cums))
    avg_old_sharpe = float(np.mean(old_strat_sharpes))
    avg_old_dd = float(np.mean(old_strat_dds))
    avg_old_winrate = float(np.mean(old_strat_winrates))

    print(f"Old Backtester (Avg across universe):")
    print(f"  Cumulative Return: {avg_old_cum:.2f}%")
    print(f"  Sharpe Ratio:      {avg_old_sharpe:.2f}")
    print(f"  Max Drawdown:      {avg_old_dd:.2f}%")
    print(f"  Win Rate:          {avg_old_winrate:.2f}%")

    # -------------------------------------------------------------
    # RUN 2: New Honest Backtester - Zero Cost (t+1 Open Execution)
    # -------------------------------------------------------------
    print("\n--- Running New Backtester: Zero Cost (Isolating t+1 Open Lag) ---")
    eng_zero = NewBacktestEngine(
        cost_model=TransactionCostModel.zero_cost(),
        initial_capital=1_000_000.0,
        rebalance_freq="daily",
        horizon_days=5,
        sector_map=sector_map,
    )
    strat_topk = TopKStrategy(k=5, sector_cap=0.30)
    res_zero = eng_zero.run(
        predictions=df_preds,
        price_panel=df_prices,
        strategy=strat_topk,
        strategy_name="TopK_ZeroCost_tPlus1Open",
    )
    m_zero = res_zero.metrics.to_dict()
    print(f"New Backtester (Zero Cost, t+1 Open):")
    print(f"  Cumulative Return: {m_zero['cumulative_return_net_pct']}%")
    print(f"  CAGR:              {m_zero['cagr_net_pct']}%")
    print(f"  Sharpe Ratio:      {m_zero['annualized_sharpe_net']}")
    print(f"  Max Drawdown:      {m_zero['max_drawdown_net_pct']}%")
    print(f"  Total Trades:      {m_zero['total_trades']}")

    # -------------------------------------------------------------
    # RUN 3: New Honest Backtester - Full Cost (Indian Delivery + Slippage)
    # -------------------------------------------------------------
    print("\n--- Running New Backtester: Full Indian Delivery Costs ---")
    cost_model = TransactionCostModel(
        brokerage_pct=0.0003,
        brokerage_flat_cap=20.0,
        stt_pct=0.0010,
        exchange_txn_pct=0.0000345,
        sebi_turnover_pct=0.000001,
        stamp_duty_pct=0.00015,
        gst_pct=0.18,
        base_slippage_bps=5.0,
        impact_factor=0.05,
    )
    eng_cost = NewBacktestEngine(
        cost_model=cost_model,
        initial_capital=1_000_000.0,
        rebalance_freq="daily",
        horizon_days=5,
        sector_map=sector_map,
    )
    res_cost = eng_cost.run(
        predictions=df_preds,
        price_panel=df_prices,
        strategy=strat_topk,
        strategy_name="TopK_CostAware_IndianEquity",
    )
    m_cost = res_cost.metrics.to_dict()
    print(f"New Backtester (Full Cost, t+1 Open):")
    print(f"  Gross Return:      {m_cost['cumulative_return_gross_pct']}%")
    print(f"  Net Return:        {m_cost['cumulative_return_net_pct']}%")
    print(f"  Cost Drag:         {m_cost['cost_drag_bps']} bps")
    print(f"  Total Costs Paid:  ₹{m_cost['total_costs_paid']:,.2f}")
    print(f"  Net Sharpe:        {m_cost['annualized_sharpe_net']}")
    print(f"  Net Max Drawdown:  {m_cost['max_drawdown_net_pct']}%")
    print(f"  Turnover (ann.):   {m_cost['annualized_turnover']}x")
    print(f"  Hit Rate:          {m_cost['hit_rate_pct']}%")

    # -------------------------------------------------------------
    # RUN 4: Weekly Rebalance (Cost reduction)
    # -------------------------------------------------------------
    print("\n--- Running New Backtester: Weekly Rebalancing ---")
    eng_weekly = NewBacktestEngine(
        cost_model=cost_model,
        initial_capital=1_000_000.0,
        rebalance_freq="weekly",
        horizon_days=5,
        sector_map=sector_map,
    )
    res_weekly = eng_weekly.run(
        predictions=df_preds,
        price_panel=df_prices,
        strategy=strat_topk,
        strategy_name="TopK_Weekly_IndianEquity",
    )
    m_weekly = res_weekly.metrics.to_dict()
    print(f"Weekly Rebalance:")
    print(f"  Net Return:        {m_weekly['cumulative_return_net_pct']}%")
    print(f"  Cost Drag:         {m_weekly['cost_drag_bps']} bps")
    print(f"  Total Costs Paid:  ₹{m_weekly['total_costs_paid']:,.2f}")
    print(f"  Turnover (ann.):   {m_weekly['annualized_turnover']}x")

    # -------------------------------------------------------------
    # RUN 5: Benchmarks: Equal-Weight Universe & Random Top-K (200 seeds)
    # -------------------------------------------------------------
    print("\n--- Running Equal-Weight Universe Benchmark ---")
    eng_ew = NewBacktestEngine(
        cost_model=cost_model,
        initial_capital=1_000_000.0,
        rebalance_freq="weekly",
        sector_map=sector_map,
    )
    res_ew = eng_ew.run(
        predictions=df_preds,
        price_panel=df_prices,
        strategy=EqualWeightUniverseStrategy(),
        strategy_name="EqualWeight_Universe",
    )
    m_ew = res_ew.metrics.to_dict()
    print(f"Equal-Weight Universe:")
    print(f"  Net Return:        {m_ew['cumulative_return_net_pct']}%")
    print(f"  Net Sharpe:        {m_ew['annualized_sharpe_net']}")
    print(f"  Net Max Drawdown:  {m_ew['max_drawdown_net_pct']}%")

    print("\n--- Running Random Top-K Benchmark (200 seeds) ---")
    bench_random = eng_cost.run_random_top_k_benchmark(
        predictions=df_preds,
        price_panel=df_prices,
        k=5,
        n_seeds=200,
        base_seed=42,
    )
    print(f"Random Top-K (200 seeds):")
    print(f"  Mean Net Return:   {bench_random['cumulative_return_net_pct']['mean']}% (std: {bench_random['cumulative_return_net_pct']['std']}%)")
    print(f"  5th-95th Pctile:   [{bench_random['cumulative_return_net_pct']['p05']}%, {bench_random['cumulative_return_net_pct']['p95']}%]")
    print(f"  Mean Sharpe:       {bench_random['annualized_sharpe_net']['mean']}")
    print(f"  Mean Max Drawdown: {bench_random['max_drawdown_net_pct']['mean']}%")

    # Save artifacts
    art_dir = Path("artifacts/v0.7/backtest_comparison")
    art_dir.mkdir(parents=True, exist_ok=True)
    res_cost.save(art_dir, "honest_topk_daily")
    res_weekly.save(art_dir, "honest_topk_weekly")
    res_ew.save(art_dir, "equal_weight_benchmark")

    comparison_summary = {
        "old_frictionless": {
            "cumulative_return_pct": avg_old_cum,
            "sharpe_ratio": avg_old_sharpe,
            "max_drawdown_pct": avg_old_dd,
            "win_rate_pct": avg_old_winrate,
        },
        "new_zero_cost_tplus1_open": m_zero,
        "new_honest_daily_cost": m_cost,
        "new_honest_weekly_cost": m_weekly,
        "equal_weight_benchmark": m_ew,
        "random_topk_200_seeds": bench_random,
    }

    with open(art_dir / "comparison_metrics.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"\nSaved comparison metrics to {art_dir / 'comparison_metrics.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
