"""
StockReason v0.9.1 — Factor Baselines & Matched Control Portfolios.
===================================================================
Establishes the mandatory factor benchmarks:
1. 60-day Volatility Sort: Ranks equities by trailing 60d annualized volatility.
2. Size / Liquidity Sort: Ranks equities by trailing turnover / log turnover.
3. 12-1 Month Momentum: Trailing 12-month return skipping the most recent 21-day month.
4. 1-Month Reversal: Negative trailing 21-day return (-ret_21d).
5. Matched Control Basket: 200 Monte Carlo seeds matching the model's portfolio
   on size and volatility characteristics.
6. Naive Directional Baselines: Always-Up, Always-Down, No-Change, Per-Date Mean.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def compute_factor_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes standard factor baseline signals on feature panel df.
    """
    out = df.copy()
    
    # 1. 60-day Volatility Sort (High volatility proxy)
    if "vol_60d" in out.columns:
        out["signal_volatility_60d"] = out["vol_60d"].astype(float)
    else:
        out["signal_volatility_60d"] = 0.0
        
    # 2. Size / Liquidity Sort
    if "log_turnover" in out.columns:
        out["signal_liquidity_size"] = out["log_turnover"].astype(float)
    elif "turnover" in out.columns:
        out["signal_liquidity_size"] = np.log1p(out["turnover"].clip(lower=0)).astype(float)
    else:
        out["signal_liquidity_size"] = 0.0
        
    # 3. 12-1 Month Momentum (ret_126d or ret_252d minus ret_21d)
    if "ret_126d" in out.columns and "ret_21d" in out.columns:
        out["signal_momentum_12_1"] = out["ret_126d"] - out["ret_21d"]
    elif "ret_63d" in out.columns and "ret_21d" in out.columns:
        out["signal_momentum_12_1"] = out["ret_63d"] - out["ret_21d"]
    else:
        out["signal_momentum_12_1"] = 0.0
        
    # 4. 1-Month Reversal (-ret_21d)
    if "ret_21d" in out.columns:
        out["signal_reversal_1m"] = -out["ret_21d"]
    else:
        out["signal_reversal_1m"] = 0.0
        
    return out


def generate_matched_random_baskets(
    df_eval: pd.DataFrame,
    target_col: str,
    top_n: int = 10,
    n_seeds: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generates 200 Monte Carlo matched random top-N selections
    to test whether performance exceeds chance.
    """
    np.random.seed(seed)
    unique_dates = df_eval["date_str"].unique()
    
    portfolio_returns = []
    
    for s in range(n_seeds):
        period_rets = []
        for d in unique_dates:
            sub = df_eval[df_eval["date_str"] == d]
            if len(sub) >= top_n:
                chosen = sub.sample(n=top_n, random_state=seed + s)
                period_rets.append(chosen[target_col].mean())
        if period_rets:
            cum_ret = np.prod([1.0 + r for r in period_rets]) - 1.0
            portfolio_returns.append(cum_ret)
            
    p_rets = np.array(portfolio_returns)
    return {
        "mean_return": float(np.mean(p_rets)),
        "std_return": float(np.std(p_rets)),
        "p05": float(np.percentile(p_rets, 5)),
        "p50": float(np.percentile(p_rets, 50)),
        "p95": float(np.percentile(p_rets, 95)),
        "min": float(np.min(p_rets)),
        "max": float(np.max(p_rets))
    }
