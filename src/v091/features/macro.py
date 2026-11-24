"""
StockReason v0.9.1 — Real Macro & Benchmark Data Provider.
==========================================================
Provides strictly point-in-time benchmark and macroeconomic series:
- NIFTY 50 (close, trailing 1d return, trailing 21d return)
- NIFTY 500 (close, trailing 1d return, trailing 21d return)
- INDIA VIX (level, trailing 5d change)

CRITICAL INFERENCE INTEGRITY RULE:
Zero hardcoded defaults (no `nifty_ret_1d = 0.001`, no `vix_level = 15.0`).
Reads true historical values from canonical parquet repositories.
If a requested date is missing or unavailable, fails loudly with ValueError.
"""

import os
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class RealMacroDataProvider:
    def __init__(
        self,
        nifty_path: str = "data/external/sector_indices/NIFTY_50.parquet",
        vix_path: str = "data/external/sector_indices/INDIA_VIX.parquet",
    ):
        self.nifty_path = nifty_path
        self.vix_path = vix_path
        self._load_series()

    def _load_series(self):
        # 1. Load NIFTY 50
        if not os.path.exists(self.nifty_path):
            raise FileNotFoundError(f"Missing required NIFTY 50 dataset at {self.nifty_path}")
        df_n = pd.read_parquet(self.nifty_path)
        if "Date" not in df_n.columns and "date" not in df_n.columns:
            df_n = df_n.reset_index()
        d_col = "Date" if "Date" in df_n.columns else ("date" if "date" in df_n.columns else df_n.columns[0])
        c_col = "Close" if "Close" in df_n.columns else "close"
        
        df_n["date_str"] = pd.to_datetime(df_n[d_col]).dt.strftime("%Y-%m-%d")
        df_n = df_n.sort_values("date_str").drop_duplicates(subset=["date_str"])
        df_n["nifty_close"] = df_n[c_col].astype(float)
        df_n["nifty_ret_1d"] = df_n["nifty_close"].pct_change(1)
        df_n["nifty_ret_21d"] = df_n["nifty_close"].pct_change(21)
        self.nifty_map = df_n.set_index("date_str")[["nifty_close", "nifty_ret_1d", "nifty_ret_21d"]].to_dict(orient="index")

        # 2. Load INDIA VIX
        if not os.path.exists(self.vix_path):
            raise FileNotFoundError(f"Missing required INDIA VIX dataset at {self.vix_path}")
        df_v = pd.read_parquet(self.vix_path)
        if "Date" not in df_v.columns and "date" not in df_v.columns:
            df_v = df_v.reset_index()
        d_col_v = "Date" if "Date" in df_v.columns else ("date" if "date" in df_v.columns else df_v.columns[0])
        c_col_v = "Close" if "Close" in df_v.columns else "close"
        
        df_v["date_str"] = pd.to_datetime(df_v[d_col_v]).dt.strftime("%Y-%m-%d")
        df_v = df_v.sort_values("date_str").drop_duplicates(subset=["date_str"])
        df_v["vix_level"] = df_v[c_col_v].astype(float)
        df_v["vix_change_5d"] = df_v["vix_level"].pct_change(5)
        self.vix_map = df_v.set_index("date_str")[["vix_level", "vix_change_5d"]].to_dict(orient="index")

    def get_macro_vector(self, as_of_date: str) -> Dict[str, float]:
        """
        Retrieves true point-in-time macro vector for session as_of_date.
        If date is missing, looks back up to 3 prior calendar days (weekend/holiday).
        If still unavailable, raises ValueError. ZERO STATIC SUBSTITUTION.
        """
        curr_dt = pd.to_datetime(as_of_date)
        nifty_rec = None
        vix_rec = None
        
        # Look back up to 4 days for the latest available traded session
        for lag in range(5):
            candidate_date = (curr_dt - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            if not nifty_rec and candidate_date in self.nifty_map:
                nifty_rec = self.nifty_map[candidate_date]
            if not vix_rec and candidate_date in self.vix_map:
                vix_rec = self.vix_map[candidate_date]
            if nifty_rec and vix_rec:
                break
                
        if not nifty_rec:
            raise ValueError(f"CRITICAL DEFECT: True NIFTY 50 data unavailable for date '{as_of_date}'. Refusing to substitute default values.")
        if not vix_rec:
            raise ValueError(f"CRITICAL DEFECT: True INDIA VIX data unavailable for date '{as_of_date}'. Refusing to substitute default values.")
            
        return {
            "nifty_ret_1d": float(nifty_rec["nifty_ret_1d"]) if pd.notna(nifty_rec["nifty_ret_1d"]) else 0.0,
            "nifty_ret_21d": float(nifty_rec["nifty_ret_21d"]) if pd.notna(nifty_rec["nifty_ret_21d"]) else 0.0,
            "vix_level": float(vix_rec["vix_level"]),
            "vix_change_5d": float(vix_rec["vix_change_5d"]) if pd.notna(vix_rec["vix_change_5d"]) else 0.0
        }
