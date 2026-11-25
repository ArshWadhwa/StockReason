"""Data Ingestion & Dataset Loading Module."""

from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import pyarrow.parquet as pq
from src.utils.config import DEFAULT_CANONICAL_DATASET, LEGACY_CANONICAL_DATASET


def load_dataset(
    path: Optional[Union[str, Path]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Loads dataset from Parquet or CSV.
    Defaults to canonical dataset if path is not specified.
    """
    if path is None:
        if DEFAULT_CANONICAL_DATASET.exists():
            path = DEFAULT_CANONICAL_DATASET
        elif LEGACY_CANONICAL_DATASET.exists():
            path = LEGACY_CANONICAL_DATASET
        else:
            raise FileNotFoundError("Canonical dataset parquet not found at standard paths.")

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    elif path.suffix in [".csv", ".txt"]:
        return pd.read_csv(path, usecols=columns)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
