"""
StockReason v0.8 Partitioned Parquet Storage Engine.
===================================================
Provides:
- Memory-efficient partitioned parquet storage under data/v0.8/.
- Atomic writes preventing partially corrupted datasets.
- Optimized dtypes (float32, category) reducing RAM consumption by ~50%.
- Dataset versioning, SHA-256 checksums, and manifest creation.
"""

import os
import sys
import time
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Generator
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("v08.storage")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

STORAGE_LAYOUT = {
    "raw_prices": "data/v0.8/raw/prices",
    "raw_corp_actions": "data/v0.8/raw/corporate_actions",
    "raw_universe": "data/v0.8/raw/universe",
    "canonical_prices": "data/v0.8/canonical/prices",
    "canonical_universe": "data/v0.8/canonical/universe_membership",
    "processed_features": "data/v0.8/processed/features",
    "processed_targets": "data/v0.8/processed/targets",
    "metadata": "data/v0.8/metadata",
}


def init_storage_directories(base_dir: str = "data/v0.8") -> None:
    """Creates the full v0.8 storage directory hierarchy."""
    for key, path in STORAGE_LAYOUT.items():
        os.makedirs(path, exist_ok=True)
    logger.info(f"Initialized v0.8 storage directory hierarchy under {base_dir}")


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes numeric and string dtypes to minimize memory footprint:
    - float64 -> float32
    - object strings -> category (for tickers and sectors)
    - int64 -> int64 or int32 for volumes / counts
    - dates -> datetime64[ns]
    """
    if df.empty:
        return df

    out = df.copy()
    for col in out.columns:
        dt = out[col].dtype
        if col in ["ticker", "sector", "survivorship_status"]:
            out[col] = out[col].astype("category")
        elif col in ["date"]:
            out[col] = pd.to_datetime(out[col]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
        elif dt == "float64":
            out[col] = out[col].astype(np.float32)
        elif dt == "int64" and col not in ["volume", "adj_volume"]:
            if out[col].max() < 2147483647 and out[col].min() > -2147483648:
                out[col] = out[col].astype(np.int32)
    return out


def write_parquet_atomic(
    df: pd.DataFrame,
    filepath: str,
    compression: str = "snappy",
    optimize: bool = True,
) -> str:
    """
    Atomically writes a DataFrame to a Parquet file using a temporary file.
    Returns SHA-256 hash of the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if optimize:
        df_to_write = optimize_dtypes(df)
    else:
        df_to_write = df

    temp_path = f"{filepath}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
    df_to_write.to_parquet(temp_path, index=False, compression=compression)
    os.replace(temp_path, filepath)

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_directory_manifest(
    dir_path: str,
    output_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a dataset manifest with file sizes, row counts, and SHA-256 checksums.
    """
    if not os.path.exists(dir_path):
        return {"error": f"Directory not found: {dir_path}", "files": []}

    files_info = []
    total_bytes = 0
    total_rows = 0

    for root, _, filenames in os.walk(dir_path):
        for fname in sorted(filenames):
            if fname.endswith(".parquet"):
                fpath = os.path.join(root, fname)
                size_bytes = os.path.getsize(fpath)
                total_bytes += size_bytes

                # Read parquet metadata without loading full table
                try:
                    pq_file = pq.ParquetFile(fpath)
                    rows = pq_file.metadata.num_rows
                    total_rows += rows
                except Exception:
                    rows = 0

                sha256 = hashlib.sha256()
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)

                rel_path = os.path.relpath(fpath, dir_path)
                files_info.append({
                    "file": rel_path,
                    "size_bytes": size_bytes,
                    "num_rows": rows,
                    "sha256": sha256.hexdigest(),
                })

    manifest = {
        "directory": dir_path,
        "total_files": len(files_info),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "total_rows": total_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files_info,
    }

    if output_manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_manifest_path)), exist_ok=True)
        with open(output_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest
