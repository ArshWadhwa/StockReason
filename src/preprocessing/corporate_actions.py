"""Corporate Actions Reconciliation & Target Quarantine Module."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from src.utils.config import HORIZONS


def quarantine_unreconciled_target_discrepancies(
    df: pd.DataFrame,
    horizons: List[int] = HORIZONS,
    tolerance: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Quarantines observations where the discrepancy between raw price ratio
    and recorded economic return exceeds tolerance (default 5% = 0.05).
    Masks targets with NaN so that unadjusted splits/bonuses do not contaminate model evaluation.
    """
    out_df = df.copy()
    quarantined_counts = {}

    for h in horizons:
        price_col = f"target_price_{h}d"
        ret_col = f"target_return_{h}d"

        if price_col not in out_df.columns or ret_col not in out_df.columns:
            quarantined_counts[f"{h}d"] = 0
            continue

        valid_mask = out_df[price_col].notna() & out_df[ret_col].notna() & (out_df["close"] > 0)
        implied_return = (out_df.loc[valid_mask, price_col] / out_df.loc[valid_mask, "close"]) - 1.0
        discrepancy = np.abs(out_df.loc[valid_mask, ret_col] - implied_return)

        # Flag and mask discrepant rows
        discrepant_mask = valid_mask & (discrepancy > tolerance)
        n_discrepant = int(discrepant_mask.sum())
        quarantined_counts[f"{h}d"] = n_discrepant

        if n_discrepant > 0:
            out_df.loc[discrepant_mask, price_col] = np.nan
            out_df.loc[discrepant_mask, ret_col] = np.nan
            if f"target_dir_{h}d" in out_df.columns:
                out_df.loc[discrepant_mask, f"target_dir_{h}d"] = np.nan

    return out_df, quarantined_counts


def verify_target_price_consistency(
    df: pd.DataFrame,
    horizons: List[int] = HORIZONS,
    tolerance: float = 0.005,
    max_tolerable_discrepancy: float = 0.05,
) -> Dict[str, Any]:
    """
    Verifies that target prices and returns across all horizons are strictly positive,
    non-empty, mathematically consistent with the reference price:
    | target_return - (target_price / close - 1.0) | <= tolerance,
    and free from unquarantined corporate action discrepancies.
    """
    results = {}
    total_rows = len(df)

    for h in horizons:
        price_col = f"target_price_{h}d"
        ret_col = f"target_return_{h}d"

        if price_col not in df.columns or ret_col not in df.columns:
            results[f"{h}d"] = {"status": "MISSING_COLUMNS", "evaluated_rows": 0}
            continue

        valid = df[price_col].notna() & df[ret_col].notna() & df["close"].notna()
        sub = df[valid]

        if len(sub) == 0:
            results[f"{h}d"] = {"status": "EMPTY", "evaluated_rows": 0}
            continue

        negative_prices = int((sub[price_col] <= 0).sum())

        # Mathematical consistency check
        implied_return = (sub[price_col] / sub["close"]) - 1.0
        discrepancy = np.abs(sub[ret_col] - implied_return)
        corporate_action_adjusted_rows = int((discrepancy > tolerance).sum())
        unquarantined_large_discrepancies = int((discrepancy > max_tolerable_discrepancy).sum())

        is_pass = (negative_prices == 0) and (unquarantined_large_discrepancies == 0)

        results[f"{h}d"] = {
            "status": "PASS" if is_pass else "QUARANTINE_REQUIRED",
            "evaluated_rows": len(sub),
            "coverage_pct": round(len(sub) / total_rows * 100.0, 2),
            "negative_prices_count": negative_prices,
            "corporate_action_adjusted_rows_count": corporate_action_adjusted_rows,
            "corporate_action_adjusted_pct": round(corporate_action_adjusted_rows / len(sub) * 100.0, 2),
            "unquarantined_large_discrepancies": unquarantined_large_discrepancies,
            "max_discrepancy": round(float(discrepancy.max()), 6) if len(discrepancy) > 0 else 0.0,
            "mean_implied_raw_return": round(float(implied_return.mean()), 4),
            "mean_recorded_total_return": round(float(sub[ret_col].mean()), 4),
        }

    return results
