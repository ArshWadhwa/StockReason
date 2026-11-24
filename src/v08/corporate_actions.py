"""
StockReason v0.8 Corporate Actions and Price Adjustment Auditor.
==============================================================
Validates splits, bonus issues, dividends, and corporate reorganizations.
Preserves both actual tradable historical prices and adjusted price series with
mathematically consistent adjustment factors across all OHLCV fields.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("v08.corporate_actions")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class CorporateActionEvent:
    ticker: str
    ex_date: str
    action_type: str  # 'SPLIT', 'BONUS', 'SPECIAL_DIVIDEND', 'RIGHTS'
    split_ratio: float  # e.g., 2.0 for 2-for-1 split (post/pre), 0.5 for reverse split
    dividend_amount: float = 0.0
    notes: str = ""


def detect_corporate_actions_from_series(
    df: pd.DataFrame,
    divergence_threshold: float = 0.20,
) -> List[CorporateActionEvent]:
    """
    Detects corporate actions (splits, bonuses) by identifying discrepancies
    between unadjusted price steps and adjusted price steps:
    ratio_step = (close[t] / close[t-1]) / (adj_close[t] / adj_close[t-1])
    """
    if df.empty or len(df) < 2:
        return []

    g = df.sort_values("date").reset_index(drop=True)
    close = g["close"].values
    adj_close = g["adj_close"].values
    dates = g["date"].dt.strftime("%Y-%m-%d").values
    ticker = g["ticker"].iloc[0] if "ticker" in g.columns else "UNKNOWN"

    events = []
    for t in range(1, len(g)):
        c_prev, c_curr = close[t - 1], close[t]
        a_prev, a_curr = adj_close[t - 1], adj_close[t]

        if c_prev <= 0 or a_prev <= 0 or c_curr <= 0 or a_curr <= 0:
            continue

        raw_step = c_curr / c_prev
        adj_step = a_curr / a_prev
        divergence = abs(raw_step / adj_step - 1.0)

        if divergence >= divergence_threshold:
            # Implied split ratio: how many new shares per old share
            implied_ratio = round(c_prev / (c_curr * (a_prev / a_curr)), 2)
            if implied_ratio >= 1.5 or implied_ratio <= 0.7:
                action_type = "SPLIT" if implied_ratio >= 1.0 else "REVERSE_SPLIT"
                events.append(CorporateActionEvent(
                    ticker=ticker,
                    ex_date=dates[t],
                    action_type=action_type,
                    split_ratio=implied_ratio,
                    notes=f"Detected price discontinuity: raw step {raw_step:.3f} vs adj step {adj_step:.3f}"
                ))

    return events


def compute_consistent_adjusted_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constructs a fully consistent adjusted OHLCV DataFrame.
    Adjustment factor: adj_factor = adj_close / close.
    Applies adj_factor consistently across Open, High, Low, and Close,
    and reciprocal 1/adj_factor to Volume to preserve traded rupee turnover.
    """
    out = df.copy()
    close = out["close"].replace(0, np.nan)
    adj_close = out["adj_close"]

    adj_factor = (adj_close / close).fillna(1.0).astype(np.float32)
    out["adj_factor"] = adj_factor

    # Adjusted OHLC
    out["adj_open"] = (out["open"] * adj_factor).astype(np.float32)
    out["adj_high"] = (out["high"] * adj_factor).astype(np.float32)
    out["adj_low"] = (out["low"] * adj_factor).astype(np.float32)
    # Volume adjusted such that adj_volume * adj_close == volume * close (turnover invariance)
    out["adj_volume"] = (out["volume"] / (adj_factor + 1e-9)).astype(np.int64)

    return out
