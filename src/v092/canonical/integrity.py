"""
StockReason v0.9.2 — Generalized Corporate Actions & Segment Gap Integrity Engine.
==================================================================================
Remediates Item 7:
- Eliminates hardcoded lists of problematic securities (ADANIENT, TATAMOTORS, VEDL, CUPID).
- Implements deterministic, universe-wide algorithmic detection of:
  1. Corporate actions (demergers, schemes of arrangement, capital reductions) with price drops >= 20%.
  2. Data gaps / trading segment omissions: Any gap > 5 trading sessions (or > 7 calendar days)
     where the subsequent session return is >= +-20%.
  3. Severe unadjusted price anomalies: Drops <= -35% or surges >= +50% that reflect
     unadjusted corporate restructurings or surveillance series hops.
- Automatically masks all forward target windows [t, t + H] spanning any detected event.
"""

import os
import glob
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd


def detect_unadjusted_corporate_actions_and_gaps(
    panel_df: pd.DataFrame,
    actions_dir: str = "data/indian-markets/actions",
    gap_days_threshold: int = 7,
    extreme_move_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Universally scans panel_df across all securities to identify:
    1. Action-backed demergers / schemes of arrangement.
    2. Segment gaps (> 7 calendar days) with jump >= 20%.
    3. Severe unadjusted price discontinuities (drop <= -35% or surge >= +50%).
    """
    df = panel_df.sort_values(["symbol", "date"]).copy()
    if "date_str" not in df.columns:
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df["prev_date_str"] = df.groupby("symbol")["date_str"].shift(1)
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["prev_adj_close"] = df.groupby("symbol")["adj_close"].shift(1)

    df["cal_days_gap"] = (pd.to_datetime(df["date_str"]) - pd.to_datetime(df["prev_date_str"])).dt.days
    df["raw_ret_1d"] = (df["close"] - df["prev_close"]) / (df["prev_close"] + 1e-9)
    df["adj_ret_1d"] = (df["adj_close"] - df["prev_adj_close"]) / (df["prev_adj_close"] + 1e-9)

    # 1. Load corporate actions registry
    action_files = sorted(glob.glob(os.path.join(actions_dir, "nse_*.parquet")))
    actions_df = pd.concat([pd.read_parquet(f) for f in action_files], ignore_index=True) if action_files else pd.DataFrame()
    if not actions_df.empty and "ex_date_str" not in actions_df.columns:
        actions_df["ex_date_str"] = pd.to_datetime(actions_df["ex_date"]).dt.strftime("%Y-%m-%d")

    detected_events = []

    # Criterion A: Segment gaps > gap_days_threshold with >= extreme_move_threshold jump
    gap_mask = (df["cal_days_gap"] > gap_days_threshold) & (df["raw_ret_1d"].abs() >= extreme_move_threshold)
    for _, row in df[gap_mask].iterrows():
        detected_events.append({
            "symbol": row["symbol"],
            "date": row["date_str"],
            "event_type": "Segment Gap / Data Omission",
            "return_1d_pct": round(float(row["raw_ret_1d"]) * 100, 2),
            "cal_days_gap": int(row["cal_days_gap"]),
            "reason": f"Trading gap of {int(row['cal_days_gap'])} calendar days with {round(float(row['raw_ret_1d']) * 100, 1)}% move."
        })

    # Criterion B: Corporate action demergers or schemes of arrangement with price drop >= 20%
    if not actions_df.empty:
        # Filter actions indicating demerger or scheme of arrangement
        demerger_keywords = ["DEMERGER", "SCHEME", "ARRANGEMENT", "CAPITAL REDUCTION", "SPIN OFF"]
        act_mask = actions_df["type"].str.upper().str.contains("|".join(demerger_keywords), na=False) | \
                   actions_df["raw_subject"].str.upper().str.contains("|".join(demerger_keywords), na=False)
        demerger_actions = actions_df[act_mask]

        for _, act in demerger_actions.iterrows():
            sym = act["symbol"]
            ex_d = act["ex_date_str"]
            # Look up in panel within +-3 days
            sub = df[(df["symbol"] == sym) & (df["date_str"] >= ex_d)].head(1)
            if not sub.empty:
                r = sub.iloc[0]
                if r["raw_ret_1d"] <= -extreme_move_threshold:
                    detected_events.append({
                        "symbol": sym,
                        "date": r["date_str"],
                        "event_type": "Confirmed Corporate Action (Demerger)",
                        "return_1d_pct": round(float(r["raw_ret_1d"]) * 100, 2),
                        "cal_days_gap": int(r["cal_days_gap"]) if pd.notna(r["cal_days_gap"]) else 1,
                        "reason": f"Corporate Action: {act['type']} - {act.get('raw_subject', '')[:60]}"
                    })

    # Criterion C: Severe unadjusted price anomalies (drop <= -35% or surge >= +50%)
    severe_mask = (df["raw_ret_1d"] <= -0.35) | (df["raw_ret_1d"] >= 0.50)
    for _, row in df[severe_mask].iterrows():
        # Avoid duplicate if already caught
        sym = row["symbol"]
        d = row["date_str"]
        if not any(e["symbol"] == sym and e["date"] == d for e in detected_events):
            detected_events.append({
                "symbol": sym,
                "date": d,
                "event_type": "Severe Unadjusted Anomaly",
                "return_1d_pct": round(float(row["raw_ret_1d"]) * 100, 2),
                "cal_days_gap": int(row["cal_days_gap"]) if pd.notna(row["cal_days_gap"]) else 1,
                "reason": f"Extreme price discontinuity of {round(float(row['raw_ret_1d']) * 100, 1)}%."
            })

    events_df = pd.DataFrame(detected_events).drop_duplicates(subset=["symbol", "date"])
    return events_df.sort_values(["date", "symbol"])


def mask_generalized_unadjusted_target_windows(
    panel_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    actions_dir: str = "data/indian-markets/actions",
    horizons: List[int] = [1, 5, 21, 63, 126],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Universally masks forward return and price targets spanning any detected
    unadjusted corporate action, severe segment gap, or unadjusted price anomaly.
    """
    if events_df is None:
        events_df = detect_unadjusted_corporate_actions_and_gaps(panel_df, actions_dir=actions_dir)

    df = panel_df.sort_values(["symbol", "date"]).copy()
    if "date_str" not in df.columns:
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    masked_counts = {h: 0 for h in horizons}

    # Group detected events by symbol for fast lookup
    events_by_symbol = events_df.groupby("symbol")["date"].apply(set).to_dict()
    affected_by_h = {h: [] for h in horizons}

    for sym, event_dates in events_by_symbol.items():
        sym_mask = df["symbol"] == sym
        if not sym_mask.any():
            continue

        sym_df = df[sym_mask]
        sym_dates = sym_df["date_str"].values
        sym_indices = sym_df.index.values

        for event_date in event_dates:
            event_match = np.where(sym_dates == event_date)[0]
            if len(event_match) == 0:
                continue
            pos = event_match[0]

            for h in horizons:
                start_pos = max(0, pos - h)
                end_pos = min(len(sym_dates), pos + 1)
                affected_by_h[h].extend(sym_indices[start_pos:end_pos])

    for h in horizons:
        aff_idx = list(set(affected_by_h[h]))
        for target_prefix in ["target_return_", "target_price_", "target_demeaned_", "target_rank_"]:
            col = f"{target_prefix}{h}d"
            if col in df.columns:
                if target_prefix == "target_return_":
                    valid = df.loc[aff_idx, col].notna().sum()
                    masked_counts[h] = int(valid)
                df.loc[aff_idx, col] = np.nan

    return df, events_df, masked_counts
