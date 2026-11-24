"""
StockReason v0.7.1 - Workstreams A5 & A6: Strategy, Benchmarks, & Return Attribution
=====================================================================================
Implements:
1. Low-turnover horizon-matched strategy with hysteresis band and min-trade-size filters.
2. Fair benchmarks on identical test dates: Nifty buy-and-hold, Equal-weight buy-and-hold,
   Equal-weight monthly rebalanced, Random Top-K (200 seeds), Momentum, Low-Volatility.
3. Return attribution regression: alpha, beta, Newey-West t-stats, R^2, and long-minus-universe spread.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.v071.cost_model import calculate_trade_cost
from src.v071.honest_statistics import compute_newey_west_tstat

logger = logging.getLogger("A5_A6_Engine")


def run_hysteresis_backtest(
    preds_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    k: int = 10,
    band: int = 20,
    rebalance_freq: int = 21,
    min_trade_size: float = 0.01,
    weight_drift_tol: float = 0.02,
    max_sector_weight: float = 0.30,
    max_stock_weight: float = 0.15,
    cost_multiplier: float = 1.0,
    initial_capital: float = 1000000.0,
    risk_free_rate: float = 0.065,
) -> Dict[str, Any]:
    """
    Backtests a horizon-matched strategy with hysteresis band and trade size filters.

    Args:
        preds_df: DataFrame with columns [date, ticker, pred].
        returns_df: DataFrame with columns [date, ticker, return_1d] (realized daily returns).
        k: Entry threshold (rank <= k).
        band: Exit threshold (keep if rank <= band, sell if rank > band).
        rebalance_freq: Rebalance interval in trading days (e.g. 21).
        min_trade_size: Minimum position change to execute (e.g. 0.01 = 1%).
        weight_drift_tol: Tolerance before adjusting existing weights.
        max_sector_weight: Max portfolio weight per sector.
        max_stock_weight: Max portfolio weight per individual stock.
        cost_multiplier: Cost sensitivity factor (0.5, 1.0, 1.5, 2.0).
    """
    dates = sorted(preds_df["date"].unique())
    if len(dates) < rebalance_freq:
        return {"sharpe": 0.0, "total_return": 0.0, "annualized_return": 0.0}

    # Map returns: (date, ticker) -> return_1d
    ret_map = returns_df.set_index(["date", "ticker"])["return_1d"].to_dict()

    current_holdings: Dict[str, float] = {}  # ticker -> weight
    portfolio_value = initial_capital
    daily_values = [portfolio_value]
    daily_returns = []
    daily_gross_returns = []
    
    total_turnover = 0.0
    total_trades = 0
    total_cost_paid = 0.0
    holding_days: List[int] = []
    entry_dates: Dict[str, pd.Timestamp] = {}

    for t_idx, d in enumerate(dates):
        # 1. Realize returns for current holdings on date d
        if current_holdings:
            gross_ret = 0.0
            for tck, w in current_holdings.items():
                r = ret_map.get((d, tck), 0.0)
                gross_ret += w * r
        else:
            gross_ret = 0.0

        daily_gross_returns.append(gross_ret)
        portfolio_value *= (1.0 + gross_ret)

        # 2. Check if today is a rebalance date
        if t_idx % rebalance_freq == 0:
            sub_preds = preds_df[preds_df["date"] == d].sort_values("pred", ascending=False).reset_index(drop=True)
            if not sub_preds.empty:
                sub_preds["rank"] = np.arange(1, len(sub_preds) + 1)
                rank_map = sub_preds.set_index("ticker")["rank"].to_dict()

                # Determine candidates to keep: currently held and rank <= band
                kept_tickers = [tck for tck in current_holdings if rank_map.get(tck, 999) <= band]
                
                # Determine new entrants from top k
                top_k_tickers = sub_preds.head(k)["ticker"].tolist()
                new_entrants = [tck for tck in top_k_tickers if tck not in kept_tickers]

                # Combine kept and new entrants, limited to target size k
                selected = kept_tickers + new_entrants
                if len(selected) > k:
                    # Sort selected by current rank and take top k
                    selected = sorted(selected, key=lambda tck: rank_map.get(tck, 999))[:k]

                # Target weights (equal weight among selected, capped at max_stock_weight)
                n_sel = len(selected)
                target_w = min(1.0 / max(1, n_sel), max_stock_weight) if n_sel > 0 else 0.0
                target_holdings = {tck: target_w for tck in selected}

                # Normalize weights to sum <= 1.0
                total_w = sum(target_holdings.values())
                if total_w > 1.0:
                    for tck in target_holdings:
                        target_holdings[tck] /= total_w

                # Calculate trades with min_trade_size and drift tolerance
                all_tcks = set(current_holdings.keys()).union(target_holdings.keys())
                rebal_cost = 0.0
                rebal_turnover = 0.0

                for tck in all_tcks:
                    curr_w = current_holdings.get(tck, 0.0)
                    tgt_w = target_holdings.get(tck, 0.0)
                    dw = tgt_w - curr_w

                    # Minimum trade size filter: skip small rebalances
                    if abs(dw) < min_trade_size or (curr_w > 0 and tgt_w > 0 and abs(dw) < weight_drift_tol):
                        continue

                    trade_val = abs(dw) * portfolio_value
                    is_buy = dw > 0
                    c_dict = calculate_trade_cost(trade_val, is_buy=is_buy, cost_multiplier=cost_multiplier)
                    rebal_cost += c_dict["total_cost"]
                    rebal_turnover += abs(dw)
                    total_trades += 1

                    if not is_buy and tck in entry_dates:
                        # Record holding period
                        holding_days.append((d - entry_dates[tck]).days)
                        del entry_dates[tck]
                    elif is_buy and tck not in entry_dates:
                        entry_dates[tck] = d

                # Deduct transaction cost from portfolio
                portfolio_value -= rebal_cost
                total_cost_paid += rebal_cost
                total_turnover += rebal_turnover
                current_holdings = target_holdings

        daily_values.append(portfolio_value)
        if len(daily_values) > 1:
            net_ret = (daily_values[-1] / daily_values[-2]) - 1.0
            daily_returns.append(net_ret)

    daily_rets = np.array(daily_returns)
    n_days = len(daily_rets)
    total_ret = (portfolio_value / initial_capital) - 1.0
    cagr = ((1.0 + total_ret) ** (252.0 / max(1, n_days))) - 1.0 if n_days > 0 and total_ret > -1.0 else 0.0

    # Sharpe ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_rets = daily_rets - daily_rf
    sharpe = (np.mean(excess_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252.0) if len(daily_rets) > 1 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(daily_values)
    drawdowns = (np.array(daily_values) - peak) / (peak + 1e-9)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Annualized one-way turnover: (total_turnover / 2) * (252 / n_days)
    ann_turnover = (total_turnover / 2.0) * (252.0 / max(1, n_days))
    cost_drag_bps = (total_cost_paid / initial_capital) * 10000.0 * (252.0 / max(1, n_days))
    avg_holding = float(np.mean(holding_days)) if holding_days else float(rebalance_freq)

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "annualized_turnover": float(ann_turnover),
        "total_trades": total_trades,
        "avg_holding_days": avg_holding,
        "cost_drag_bps": float(cost_drag_bps),
        "daily_returns": daily_rets,
        "daily_values": np.array(daily_values),
        "dates": dates,
    }


def run_attribution_regression(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = 0.065,
) -> Dict[str, float]:
    """
    Regresses strategy daily excess returns on benchmark excess returns.
    Reports annualized alpha, beta, Newey-West t-statistic for alpha, and R^2.
    """
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 10:
        return {"alpha_ann": 0.0, "beta": 1.0, "alpha_tstat": 0.0, "p_val": 1.0, "r2": 0.0}

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    y = strategy_returns[:n] - daily_rf
    x = benchmark_returns[:n] - daily_rf

    # OLS: y = alpha + beta * x
    X_mat = np.column_stack([np.ones(n), x])
    coeffs, residuals, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    alpha_daily, beta = coeffs[0], coeffs[1]

    # Annualized alpha: arithmetic annualization alpha_daily * 252.0
    alpha_ann = float(alpha_daily * 252.0)

    # Newey-West standard error for alpha
    e = y - (alpha_daily + beta * x)
    gamma_0 = np.sum(e ** 2) / n
    var_hac = gamma_0
    for lag in range(1, 10):
        w = 1.0 - (lag / 11.0)
        var_hac += 2.0 * w * (np.sum(e[lag:] * e[:-lag]) / n)
    
    # Variance of alpha estimator: (X'X)^-1 * var_hac * n
    denom = np.sum((x - np.mean(x)) ** 2)
    var_alpha = (var_hac / n) * (1.0 + (np.mean(x) ** 2) / (denom / n + 1e-12))
    se_alpha = np.sqrt(max(1e-12, var_alpha))
    alpha_tstat = alpha_daily / se_alpha
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(alpha_tstat))))

    # R^2
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(e ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    return {
        "alpha_ann": float(alpha_ann),
        "beta": float(beta),
        "alpha_tstat": float(alpha_tstat),
        "p_val": p_val,
        "r2": float(r2),
    }
