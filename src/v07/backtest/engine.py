"""
Honest, Cost-Aware Backtest Engine for StockReason v0.7.
======================================================
Key Architecture Principles:
1. Strict Out-of-Sample Enforcement: Refuses to run if any prediction is not marked is_out_of_sample == True.
2. Next-Day Open Execution: Signals computed at close of day t execute strictly at the open of day t+1.
3. Realistic Indian Equity Costs: Models brokerage, STT, exchange txn charges, SEBI fee, stamp duty, GST, and liquidity slippage.
4. Intraday Limit & Stop Execution: Limit orders execute only if low <= limit price; stop-loss takes priority if both stop and take-profit are touched; gap-through-stop executes at open.
5. Benchmark Harness: Compares strategy against Buy-and-Hold (Nifty 50), Equal-Weight Universe, and Random Top-K (averaged over 200 seeds with 5th and 95th percentiles).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from pathlib import Path
import math
import yaml
import numpy as np
import pandas as pd

from src.v07.backtest.costs import TransactionCostModel, CostBreakdown
from src.v07.backtest.orders import Position, Trade, OrderExecutor
from src.v07.backtest.strategies import (
    BaseStrategy, TopKStrategy, LongFlatStrategy, EqualWeightUniverseStrategy, RandomTopKStrategy
)
from src.v07.backtest.metrics import (
    BacktestMetrics, compute_backtest_metrics, save_equity_curve
)


@dataclass
class BacktestResult:
    strategy_name: str
    metrics: BacktestMetrics
    daily_equity: pd.DataFrame
    trades: List[Trade]
    parameters: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "metrics": self.metrics.to_dict(),
            "total_trades": len(self.trades),
        }

    def save(
        self,
        output_dir: Union[str, Path],
        file_prefix: str = "backtest"
    ) -> Dict[str, Path]:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        parquet_path = out_p / f"{file_prefix}_equity.parquet"
        json_path = out_p / f"{file_prefix}_equity.json"
        summary_path = out_p / f"{file_prefix}_summary.json"

        save_equity_curve(self.daily_equity, parquet_path=parquet_path, json_path=json_path)

        with open(summary_path, "w") as f:
            yaml.dump(self.summary(), f, default_flow_style=False)

        return {
            "parquet": parquet_path,
            "json": json_path,
            "summary": summary_path,
        }


class BacktestEngine:
    """
    Core Backtest Engine enforcing forecast-origin alignment, t+1 open execution,
    realistic cost drag, limit fills, and stop-loss logic.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        initial_capital: float = 1_000_000.0,
        risk_free_rate: float = 0.0,
        rebalance_freq: str = "daily",  # "daily" | "weekly" | "horizon"
        horizon_days: int = 5,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        limit_entry_pct: Optional[float] = None,  # e.g. 0.002 = buy at max 0.2% below close(t)
        sector_map: Optional[Dict[str, str]] = None,
    ):
        self.cost_model = cost_model or TransactionCostModel()
        self.initial_capital = float(initial_capital)
        self.risk_free_rate = float(risk_free_rate)
        self.rebalance_freq = rebalance_freq.lower()
        self.horizon_days = max(1, int(horizon_days))
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.limit_entry_pct = limit_entry_pct
        self.sector_map = sector_map or {}
        self.order_executor = OrderExecutor(self.cost_model)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "BacktestEngine":
        p = Path(config_path)
        if not p.exists():
            return cls()
        with open(p, "r") as f:
            cfg = yaml.safe_load(f) or {}

        bt_cfg = cfg.get("backtest", {})
        cost_cfg = bt_cfg.get("costs", {})
        exec_cfg = bt_cfg.get("execution", {})

        cost_model = TransactionCostModel.from_dict(cost_cfg)
        return cls(
            cost_model=cost_model,
            initial_capital=float(bt_cfg.get("initial_capital", 1_000_000.0)),
            rebalance_freq=str(bt_cfg.get("rebalance_freq", "daily")),
            horizon_days=int(bt_cfg.get("horizon_days", 5)),
            stop_loss_pct=exec_cfg.get("stop_loss_pct"),
            take_profit_pct=exec_cfg.get("take_profit_pct"),
            limit_entry_pct=exec_cfg.get("limit_entry_pct"),
        )

    def validate_predictions(self, predictions_df: pd.DataFrame) -> None:
        """
        Validates prediction archive schema and strictly enforces out-of-sample flag.
        Refuses to run if any prediction is not marked is_out_of_sample == True.
        """
        required_cols = {"ticker", "date", "pred_return"}
        missing = required_cols - set(predictions_df.columns)
        if missing:
            raise ValueError(f"Prediction archive missing required columns: {missing}")

        if "is_out_of_sample" not in predictions_df.columns:
            raise ValueError(
                "Prediction archive missing 'is_out_of_sample' column. "
                "The engine refuses to run on unverified predictions."
            )

        # Check if all rows are out-of-sample
        is_oos = predictions_df["is_out_of_sample"]
        # Allow boolean True, 1, or string 'True'/'true'
        valid_oos = is_oos.apply(lambda x: x is True or x == 1 or str(x).lower() == "true")
        if not valid_oos.all():
            invalid_count = int((~valid_oos).sum())
            raise ValueError(
                f"Prediction archive contains {invalid_count} in-sample prediction(s). "
                "Only out-of-sample predictions (is_out_of_sample == True) are permitted."
            )

    def prepare_price_panel(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes price panel columns and date format.
        Expected columns: date, ticker, open, high, low, close, volume.
        """
        df = price_panel.copy()
        df.columns = [c.lower() for c in df.columns]

        req = {"date", "ticker", "open", "high", "low", "close"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Price panel missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if "volume" not in df.columns:
            df["volume"] = 1_000_000.0

        return df

    def run(
        self,
        predictions: Union[pd.DataFrame, str, Path],
        price_panel: pd.DataFrame,
        strategy: Optional[BaseStrategy] = None,
        strategy_name: str = "TopK_Strategy",
    ) -> BacktestResult:
        """
        Runs full backtest simulation.
        Signals at date t close -> Execution at date t+1 open.
        """
        if isinstance(predictions, (str, Path)):
            p = Path(predictions)
            if p.suffix == ".parquet":
                df_preds = pd.read_parquet(p)
            elif p.suffix == ".csv":
                df_preds = pd.read_csv(p)
            elif p.suffix == ".json":
                df_preds = pd.read_json(p)
            else:
                raise ValueError(f"Unsupported prediction file format: {p}")
        else:
            df_preds = predictions.copy()

        # Validate out-of-sample
        self.validate_predictions(df_preds)
        df_preds["date"] = pd.to_datetime(df_preds["date"]).dt.strftime("%Y-%m-%d")

        # Prepare price data
        df_prices = self.prepare_price_panel(price_panel)
        trading_dates = sorted(df_prices["date"].unique())
        if len(trading_dates) < 2:
            raise ValueError("Price panel must contain at least 2 distinct trading dates.")

        date_to_idx = {d: i for i, d in enumerate(trading_dates)}

        # Group prices by date and ticker for O(1) lookups
        price_lookup: Dict[str, Dict[str, Dict[str, float]]] = {}
        median_volumes: Dict[str, float] = {}
        
        # Precompute median volumes per ticker
        vol_groups = df_prices.groupby("ticker")["volume"].median().to_dict()
        median_volumes = vol_groups

        for d, group in df_prices.groupby("date"):
            price_lookup[d] = {}
            for _, row in group.iterrows():
                price_lookup[d][row["ticker"]] = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                }

        # Predictions grouped by date (date t)
        pred_lookup: Dict[str, pd.DataFrame] = {
            d: grp for d, grp in df_preds.groupby("date")
        }

        # Strategy default
        if strategy is None:
            strategy = TopKStrategy(k=10, sector_cap=0.30)

        # Simulation state
        cash = self.initial_capital
        gross_pnl_accum = 0.0
        active_positions: Dict[str, Position] = {}
        closed_trades: List[Trade] = []
        daily_records: List[Dict[str, Any]] = []

        # Identify rebalance dates (in terms of execution date t+1)
        pending_rebalance_orders: Optional[Dict[str, float]] = None

        prev_week = None

        for idx, date_t in enumerate(trading_dates):
            current_bar_date = date_t
            curr_prices = price_lookup.get(current_bar_date, {})

            # 1. MORNING (OPEN of day t):
            # Execute any pending orders generated from day t-1 signals
            day_turnover = 0.0
            day_costs = 0.0

            if pending_rebalance_orders is not None:
                target_weights = pending_rebalance_orders
                pending_rebalance_orders = None

                # Current portfolio equity estimate at open
                curr_open_equity = cash
                for tkr, pos in active_positions.items():
                    open_p = curr_prices.get(tkr, {}).get("open", pos.entry_price)
                    curr_open_equity += pos.shares * open_p

                # Determine positions to close
                tickers_to_close = [
                    tkr for tkr, pos in active_positions.items()
                    if tkr not in target_weights or pos.days_held >= self.horizon_days
                ]

                for tkr in tickers_to_close:
                    pos = active_positions.pop(tkr)
                    exit_bar = curr_prices.get(tkr)
                    exit_price = exit_bar["open"] if exit_bar else pos.entry_price
                    med_vol = median_volumes.get(tkr)
                    trade = self.order_executor.execute_exit(
                        pos, exit_price, current_bar_date, "REBALANCE", med_vol
                    )
                    closed_trades.append(trade)
                    gross_proceeds = trade.shares * exit_price
                    exit_cost = trade.exit_costs.total_cost if trade.exit_costs else 0.0
                    cash += (gross_proceeds - exit_cost)
                    gross_pnl_accum += trade.gross_pnl
                    day_turnover += gross_proceeds
                    day_costs += exit_cost

                # Determine positions to enter
                total_avail_cash = cash
                for tkr, weight in target_weights.items():
                    if tkr in active_positions:
                        continue  # Already holding

                    bar = curr_prices.get(tkr)
                    if not bar or bar["open"] <= 0:
                        continue

                    target_dollar = weight * curr_open_equity
                    med_vol = median_volumes.get(tkr)

                    pos = None
                    if self.limit_entry_pct is not None and self.limit_entry_pct > 0:
                        # Limit order at (1 - limit_entry_pct) of prior close or open
                        limit_p = bar["open"] * (1.0 - self.limit_entry_pct)
                        pos = self.order_executor.execute_limit_entry(
                            ticker=tkr,
                            limit_price=limit_p,
                            target_allocation=target_dollar,
                            available_cash=cash,
                            date=current_bar_date,
                            bar=bar,
                            stop_loss_pct=self.stop_loss_pct,
                            take_profit_pct=self.take_profit_pct,
                            target_horizon_days=self.horizon_days,
                            median_volume=med_vol,
                        )
                    else:
                        pos = self.order_executor.execute_market_entry(
                            ticker=tkr,
                            target_allocation=target_dollar,
                            available_cash=cash,
                            date=current_bar_date,
                            bar=bar,
                            stop_loss_pct=self.stop_loss_pct,
                            take_profit_pct=self.take_profit_pct,
                            target_horizon_days=self.horizon_days,
                            median_volume=med_vol,
                        )

                    if pos:
                        active_positions[tkr] = pos
                        cash -= pos.cost_basis
                        day_turnover += pos.shares * pos.entry_price
                        day_costs += pos.entry_costs.total_cost

            # 2. INTRADAY of day t:
            # Check Stop-Loss and Take-Profit for active positions
            intraday_exits = []
            for tkr, pos in list(active_positions.items()):
                bar = curr_prices.get(tkr)
                if not bar:
                    continue

                exit_info = pos.check_intraday_exit(bar)
                if exit_info:
                    active_positions.pop(tkr)
                    med_vol = median_volumes.get(tkr)
                    trade = self.order_executor.execute_exit(
                        pos,
                        exit_price=exit_info["exit_price"],
                        exit_date=current_bar_date,
                        exit_reason=exit_info["exit_reason"],
                        median_volume=med_vol,
                    )
                    closed_trades.append(trade)
                    gross_proceeds = trade.shares * trade.exit_price
                    exit_cost = trade.exit_costs.total_cost if trade.exit_costs else 0.0
                    cash += (gross_proceeds - exit_cost)
                    gross_pnl_accum += trade.gross_pnl
                    day_turnover += gross_proceeds
                    day_costs += exit_cost

            # 3. END OF DAY t (CLOSE):
            # Compute portfolio valuation
            invested_value = 0.0
            unrealized_gross_pnl = 0.0

            for tkr, pos in active_positions.items():
                close_p = curr_prices.get(tkr, {}).get("close", pos.entry_price)
                val = pos.shares * close_p
                invested_value += val
                unrealized_gross_pnl += pos.shares * (close_p - pos.entry_price)
                pos.update_day()

            net_equity = cash + invested_value
            gross_equity = self.initial_capital + gross_pnl_accum + unrealized_gross_pnl

            daily_records.append({
                "date": current_bar_date,
                "gross_equity": gross_equity,
                "net_equity": net_equity,
                "cash": cash,
                "invested_value": invested_value,
                "turnover": day_turnover,
                "costs": day_costs,
            })

            # 4. POST-CLOSE OF DAY t:
            # Generate signals from day t predictions to execute on day t+1 OPEN
            if current_bar_date in pred_lookup and (idx + 1) < len(trading_dates):
                # Check rebalance schedule
                next_date = trading_dates[idx + 1]
                should_rebalance = False

                if self.rebalance_freq == "daily":
                    should_rebalance = True
                elif self.rebalance_freq == "weekly":
                    dt_obj = pd.to_datetime(next_date)
                    next_week = dt_obj.isocalendar()[1]
                    if prev_week is None or next_week != prev_week:
                        should_rebalance = True
                        prev_week = next_week
                elif self.rebalance_freq == "horizon":
                    # Check if any position reached horizon or if portfolio has cash
                    has_expiring = any(p.days_held >= self.horizon_days for p in active_positions.values())
                    if has_expiring or not active_positions:
                        should_rebalance = True

                if should_rebalance:
                    preds_t = pred_lookup[current_bar_date]
                    active_universe = list(curr_prices.keys())
                    target_weights = strategy.generate_target_weights(
                        date=current_bar_date,
                        predictions_df=preds_t,
                        active_universe=active_universe,
                        sector_map=self.sector_map,
                    )
                    pending_rebalance_orders = target_weights

        # Close any remaining open positions at the final close date
        final_date = trading_dates[-1]
        final_prices = price_lookup.get(final_date, {})
        for tkr, pos in list(active_positions.items()):
            exit_p = final_prices.get(tkr, {}).get("close", pos.entry_price)
            med_vol = median_volumes.get(tkr)
            trade = self.order_executor.execute_exit(
                pos, exit_p, final_date, "FINAL_CLOSE", med_vol
            )
            closed_trades.append(trade)

        daily_df = pd.DataFrame(daily_records)
        metrics = compute_backtest_metrics(
            daily_df=daily_df,
            trades=closed_trades,
            risk_free_rate=self.risk_free_rate,
        )

        params = {
            "strategy_name": strategy_name,
            "initial_capital": self.initial_capital,
            "rebalance_freq": self.rebalance_freq,
            "horizon_days": self.horizon_days,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "limit_entry_pct": self.limit_entry_pct,
            "cost_model": self.cost_model.__dict__,
        }

        return BacktestResult(
            strategy_name=strategy_name,
            metrics=metrics,
            daily_equity=daily_df,
            trades=closed_trades,
            parameters=params,
        )

    def run_random_top_k_benchmark(
        self,
        predictions: Union[pd.DataFrame, str, Path],
        price_panel: pd.DataFrame,
        k: int = 10,
        n_seeds: int = 200,
        base_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs Random Top-K selection across N seeds (default 200) and computes
        mean, std, 5th percentile, and 95th percentile metrics.
        """
        metrics_list = []
        for i in range(n_seeds):
            seed = base_seed + i
            strat = RandomTopKStrategy(k=k, seed=seed)
            res = self.run(
                predictions=predictions,
                price_panel=price_panel,
                strategy=strat,
                strategy_name=f"RandomTopK_seed_{seed}",
            )
            metrics_list.append(res.metrics.to_dict())

        df_bench = pd.DataFrame(metrics_list)
        summary = {
            "n_seeds": n_seeds,
            "cumulative_return_net_pct": {
                "mean": round(float(df_bench["cumulative_return_net_pct"].mean()), 2),
                "std": round(float(df_bench["cumulative_return_net_pct"].std()), 2),
                "p05": round(float(df_bench["cumulative_return_net_pct"].quantile(0.05)), 2),
                "p95": round(float(df_bench["cumulative_return_net_pct"].quantile(0.95)), 2),
            },
            "cagr_net_pct": {
                "mean": round(float(df_bench["cagr_net_pct"].mean()), 2),
                "p05": round(float(df_bench["cagr_net_pct"].quantile(0.05)), 2),
                "p95": round(float(df_bench["cagr_net_pct"].quantile(0.95)), 2),
            },
            "annualized_sharpe_net": {
                "mean": round(float(df_bench["annualized_sharpe_net"].mean()), 2),
                "p05": round(float(df_bench["annualized_sharpe_net"].quantile(0.05)), 2),
                "p95": round(float(df_bench["annualized_sharpe_net"].quantile(0.95)), 2),
            },
            "max_drawdown_net_pct": {
                "mean": round(float(df_bench["max_drawdown_net_pct"].mean()), 2),
                "p05": round(float(df_bench["max_drawdown_net_pct"].quantile(0.05)), 2),
                "p95": round(float(df_bench["max_drawdown_net_pct"].quantile(0.95)), 2),
            },
        }
        return summary
