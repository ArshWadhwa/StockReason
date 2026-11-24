"""
sequences.py
------------
Implementation of the forecast-origin sequence builder and lazy PyTorch Dataset
for StockReason v0.7.

Forecast-Origin Convention:
  A forecast made at date t uses ONLY information available at the close of day t.
  - Input window: rows [t - seq_len + 1 ... t] (inclusive of day t, length = seq_len).
  - Target for horizon h: (adj_close[t + h] / adj_close[t]) - 1.0.
  - label_start: date of day t.
  - label_end_h: date of day t + h.
  - Windows NEVER cross ticker boundaries.
  - Windows are built lazily through PyTorch SequenceDataset to prevent RAM blowup.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Lazy PyTorch Dataset for multi-horizon sequence forecasting.
    
    Instead of pre-allocating an (N, seq_len, num_features) 3D array in memory,
    this dataset stores 2D feature matrices per ticker and slices them on-the-fly
    in __getitem__.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int = 60,
        horizons: Optional[List[int]] = None,
        price_col: str = "adj_close",
        return_meta: bool = True,
    ):
        """
        Args:
            panel: DataFrame with [ticker, date, price_col, *feature_cols].
            feature_cols: List of column names used as model input features.
            seq_len: Sequence length L (input window size). Default: 60.
            horizons: List of forward horizons in trading days, e.g. [5, 21].
            price_col: Column name for price returns (default: 'adj_close').
            return_meta: Whether __getitem__ returns (x, y, meta). If False, returns (x, y).
        """
        super().__init__()
        if horizons is None:
            horizons = [5, 21]
        self.feature_cols = list(feature_cols)
        self.seq_len = int(seq_len)
        self.horizons = [int(h) for h in horizons]
        self.max_horizon = max(self.horizons) if self.horizons else 0
        self.price_col = price_col
        self.return_meta = return_meta

        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if not self.horizons:
            raise ValueError("horizons list cannot be empty")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"All horizons must be positive integers: {self.horizons}")

        # Ensure required columns exist
        missing_features = [col for col in self.feature_cols if col not in panel.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns in panel: {missing_features}")

        if self.price_col not in panel.columns:
            if "close" in panel.columns:
                import warnings
                warnings.warn(
                    f"'{self.price_col}' not found in panel; falling back to 'close'. "
                    f"Note: corporate actions rule mandates 'adj_close' for returns.",
                    UserWarning,
                )
                self.price_col = "close"
            else:
                raise ValueError(f"Price column '{self.price_col}' not found in panel.")

        # Prepare panel data
        df = panel.copy()
        if "date" not in df.columns:
            raise ValueError("Panel must contain a 'date' column.")
        df["date"] = pd.to_datetime(df["date"])

        if "ticker" not in df.columns:
            df["ticker"] = "DEFAULT"

        # Organize 2D arrays per ticker
        self.ticker_features: Dict[str, np.ndarray] = {}
        self.ticker_prices: Dict[str, np.ndarray] = {}
        self.ticker_dates: Dict[str, np.ndarray] = {}
        
        metadata_rows: List[Dict[str, Any]] = []
        sample_index_entries: List[Tuple[str, int, int, np.ndarray, pd.Timestamp, Dict[int, pd.Timestamp]]] = []

        sample_counter = 0

        # Group by ticker
        grouped = df.groupby("ticker", sort=True)
        for ticker, grp in grouped:
            grp_sorted = grp.sort_values("date").reset_index(drop=True)
            n_rows = len(grp_sorted)

            feat_arr = grp_sorted[self.feature_cols].to_numpy(dtype=np.float32)
            price_arr = grp_sorted[self.price_col].to_numpy(dtype=np.float64)
            date_arr = grp_sorted["date"].to_numpy()

            self.ticker_features[str(ticker)] = feat_arr
            self.ticker_prices[str(ticker)] = price_arr
            self.ticker_dates[str(ticker)] = date_arr

            # Check if sufficient rows for sequence + max_horizon
            min_required = self.seq_len + self.max_horizon
            if n_rows < min_required:
                continue

            # Forecast-origin convention:
            # t runs from seq_len - 1 to n_rows - 1 - max_horizon
            # Input window is rows [t - seq_len + 1 ... t] (inclusive of t, length = seq_len)
            # Target for horizon h is adj_close[t + h] / adj_close[t] - 1
            for t in range(self.seq_len - 1, n_rows - self.max_horizon):
                start_idx = t - self.seq_len + 1
                end_idx = t  # inclusive
                
                price_t = price_arr[t]
                if price_t <= 0 or np.isnan(price_t):
                    continue

                label_start = pd.Timestamp(date_arr[t])
                
                targets_for_sample = []
                label_ends: Dict[int, pd.Timestamp] = {}
                row_meta: Dict[str, Any] = {
                    "sample_idx": sample_counter,
                    "ticker": str(ticker),
                    "t_idx": t,
                    "label_start": label_start,
                }

                skip_sample = False
                for h in self.horizons:
                    price_th = price_arr[t + h]
                    if price_th <= 0 or np.isnan(price_th):
                        skip_sample = True
                        break
                    ret_h = float(price_th / price_t - 1.0)
                    end_date_h = pd.Timestamp(date_arr[t + h])
                    targets_for_sample.append(ret_h)
                    label_ends[h] = end_date_h
                    row_meta[f"label_end_{h}"] = end_date_h
                    row_meta[f"target_{h}"] = ret_h

                if skip_sample:
                    continue

                targets_arr = np.array(targets_for_sample, dtype=np.float32)
                sample_index_entries.append((str(ticker), start_idx, end_idx, targets_arr, label_start, label_ends))
                metadata_rows.append(row_meta)
                sample_counter += 1

        self.index_entries = sample_index_entries
        self.metadata = pd.DataFrame(metadata_rows) if metadata_rows else pd.DataFrame(
            columns=["sample_idx", "ticker", "t_idx", "label_start"]
            + [f"label_end_{h}" for h in self.horizons]
            + [f"target_{h}" for h in self.horizons]
        )

    def __len__(self) -> int:
        return len(self.index_entries)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], Tuple[torch.Tensor, torch.Tensor]]:
        ticker, start_idx, end_idx, targets_arr, label_start, label_ends = self.index_entries[idx]
        
        # Lazy slice: [start_idx : end_idx + 1] has exact length seq_len
        # end_idx is inclusive, so python slice ends at end_idx + 1
        window_feat = self.ticker_features[ticker][start_idx : end_idx + 1]
        
        x = torch.from_numpy(window_feat.copy())
        y = torch.from_numpy(targets_arr.copy())

        if not self.return_meta:
            return x, y

        meta = {
            "sample_idx": idx,
            "ticker": ticker,
            "t_idx": end_idx,
            "label_start": label_start,
            "label_ends": label_ends,
        }
        for h, end_dt in label_ends.items():
            meta[f"label_end_{h}"] = end_dt

        return x, y, meta

    def get_metadata(self) -> pd.DataFrame:
        """Returns the metadata DataFrame for all samples."""
        return self.metadata

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Eagerly materializes all sequences into RAM as (X, Y, metadata).
        Use with caution on very large datasets.
        """
        n = len(self)
        if n == 0:
            x_empty = torch.empty((0, self.seq_len, len(self.feature_cols)), dtype=torch.float32)
            y_empty = torch.empty((0, len(self.horizons)), dtype=torch.float32)
            return x_empty, y_empty, self.metadata.copy()

        all_x = np.empty((n, self.seq_len, len(self.feature_cols)), dtype=np.float32)
        all_y = np.empty((n, len(self.horizons)), dtype=np.float32)

        for i, (ticker, start_idx, end_idx, targets_arr, _, _) in enumerate(self.index_entries):
            all_x[i] = self.ticker_features[ticker][start_idx : end_idx + 1]
            all_y[i] = targets_arr

        return torch.from_numpy(all_x), torch.from_numpy(all_y), self.metadata.copy()


def build_sequences(
    panel: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 60,
    horizons: Optional[List[int]] = None,
    price_col: str = "adj_close",
    return_meta: bool = True,
) -> SequenceDataset:
    """
    Constructs a SequenceDataset following the forecast-origin convention.
    
    Args:
        panel: DataFrame containing historical price and feature data.
        feature_cols: List of feature column names.
        seq_len: Number of historical days in input sequence (L).
        horizons: List of forward target horizons in trading days (e.g. [5, 21]).
        price_col: Name of adjusted price column (default: 'adj_close').
        return_meta: If True, __getitem__ returns (x, y, meta).
        
    Returns:
        SequenceDataset: Lazy PyTorch Dataset.
    """
    return SequenceDataset(
        panel=panel,
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizons=horizons,
        price_col=price_col,
        return_meta=return_meta,
    )


def get_live_inference_window(
    panel: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 60,
    ticker: Optional[str] = None,
    price_col: str = "adj_close",
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Extracts the live-inference input window (the most recent seq_len rows)
    for each ticker, guaranteeing exact format match with training windows.
    
    Args:
        panel: DataFrame containing features up to the latest available date.
        feature_cols: List of feature column names.
        seq_len: Sequence length (must match model sequence length).
        ticker: Optional single ticker to extract. If None, extracts for all tickers in panel.
        price_col: Name of adjusted price column.
        
    Returns:
        Tuple[torch.Tensor, pd.DataFrame]:
            - Tensor of shape (num_tickers, seq_len, num_features).
            - Metadata DataFrame with [ticker, as_of_date, current_adj_close].
    """
    df = panel.copy()
    if "date" not in df.columns:
        raise ValueError("Panel must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])

    if "ticker" not in df.columns:
        df["ticker"] = "DEFAULT"

    if ticker is not None:
        df = df[df["ticker"] == ticker]
        if df.empty:
            raise ValueError(f"Ticker '{ticker}' not found in panel.")

    meta_rows = []
    tensors_list = []

    for t_name, grp in df.groupby("ticker", sort=True):
        grp_sorted = grp.sort_values("date").reset_index(drop=True)
        if len(grp_sorted) < seq_len:
            raise ValueError(
                f"Ticker '{t_name}' has only {len(grp_sorted)} rows, but seq_len={seq_len} is required."
            )

        # The live window is exactly the last seq_len rows [len-seq_len : len]
        window_df = grp_sorted.iloc[-seq_len:]
        latest_row = grp_sorted.iloc[-1]

        feat_arr = window_df[feature_cols].to_numpy(dtype=np.float32)
        tensors_list.append(feat_arr)

        meta_rows.append({
            "ticker": str(t_name),
            "as_of_date": pd.Timestamp(latest_row["date"]),
            "current_adj_close": float(latest_row[price_col]) if price_col in latest_row else float("nan"),
            "seq_len": seq_len,
        })

    if not tensors_list:
        raise ValueError("No tickers had sufficient data for live inference window.")

    batch_x = torch.from_numpy(np.stack(tensors_list, axis=0))
    meta_df = pd.DataFrame(meta_rows)
    return batch_x, meta_df
