"""
StockReason v0.7.1 - Lockbox Data Access Module
===============================================
Enforces strict segregation between the exploratory/training panel and the final
lockbox test partition.

Specification:
- Lockbox is the last 6 months of available trading dates in the panel.
- Embargo is the max_horizon + embargo_window (21 + 5 = 26 trading days) immediately
  prior to the lockbox start date.
- Default calls (exclude_lockbox=True) strip the lockbox and embargo dates.
- Accessing lockbox data (exclude_lockbox=False) requires the environment variable:
    STOCKREASON_LOCKBOX_UNLOCK="I_AM_THE_LEAD_AGENT_AND_CONFIG_IS_FROZEN"
- Every unlocked access appends an entry to reports/v0.7.1/lockbox_access.log.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

UNLOCK_ENV_VAR = "STOCKREASON_LOCKBOX_UNLOCK"
UNLOCK_MAGIC_VALUE = "I_AM_THE_LEAD_AGENT_AND_CONFIG_IS_FROZEN"
DEFAULT_PANEL_PATH = "data/v0.7/panel_v07.parquet"
LOCKBOX_LOG_PATH = "reports/v0.7.1/lockbox_access.log"

# Default frozen dates for 2021-11-18 -> 2026-09-07 panel
LOCKBOX_START_DATE = "2026-03-09"
LOCKBOX_END_DATE = "2026-09-07"
EMBARGO_START_DATE = "2026-01-29"
EMBARGO_END_DATE = "2026-03-06"
TRAINABLE_END_DATE = "2026-01-28"


def get_git_commit() -> str:
    """Returns the current git commit hash, or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_config_hash(config_path: str = "config/v071.yaml") -> str:
    """Computes SHA256 of the frozen configuration file."""
    p = Path(config_path)
    if not p.exists():
        p = Path("config/v07.yaml")
    if p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return "no_config"


def log_lockbox_access(caller: str, config_hash: str) -> None:
    """Logs an unlocked lockbox access event."""
    log_file = Path(LOCKBOX_LOG_PATH)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    commit = get_git_commit()
    line = f"{timestamp} | caller={caller} | commit={commit} | config_hash={config_hash}\n"
    with open(log_file, "a") as f:
        f.write(line)
    logger.warning("LOCKBOX ACCESSED: %s", line.strip())


def get_lockbox_date_bounds(df: pd.DataFrame, date_col: str = "date") -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Computes exact lockbox start, embargo start, and max date from data.
    Lockbox: last 6 calendar months of trading dates.
    Embargo: 26 trading days preceding lockbox start.
    """
    dates = pd.to_datetime(pd.Series(df[date_col].unique())).sort_values().reset_index(drop=True)
    if len(dates) == 0:
        raise ValueError("DataFrame contains no dates.")
    max_date = dates.iloc[-1]
    approx_lockbox_start = max_date - pd.DateOffset(months=6)
    lockbox_dates = dates[dates >= approx_lockbox_start]
    lockbox_start = lockbox_dates.iloc[0]
    
    pre_lockbox_dates = dates[dates < lockbox_start]
    embargo_days = 26
    if len(pre_lockbox_dates) > embargo_days:
        embargo_start = pre_lockbox_dates.iloc[-embargo_days]
    else:
        embargo_start = pre_lockbox_dates.iloc[0]
        
    return lockbox_start, embargo_start, max_date


def load_panel(
    path: str = DEFAULT_PANEL_PATH,
    exclude_lockbox: bool = True,
    exclude_embargo: bool = True,
    date_col: str = "date",
    config_path: str = "config/v071.yaml",
) -> pd.DataFrame:
    """
    Loads equity panel data with strict lockbox enforcement.

    Args:
        path: Path to parquet panel dataset.
        exclude_lockbox: If True (default), removes lockbox dates.
                         If False, requires environment variable unlock.
        exclude_embargo: If True, also removes the embargo window preceding lockbox.
        date_col: Name of date column.
        config_path: Path to config for hashing.

    Returns:
        Filtered or full panel DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Panel file not found at {path}")

    df = pd.read_parquet(path)
    df[date_col] = pd.to_datetime(df[date_col])

    if exclude_lockbox:
        # Determine cutoff dates
        lockbox_start, embargo_start, _ = get_lockbox_date_bounds(df, date_col=date_col)
        cutoff = embargo_start if exclude_embargo else lockbox_start
        filtered_df = df[df[date_col] < cutoff].copy()
        logger.info(
            "Loaded panel with exclude_lockbox=True. Cutoff date: %s. Rows: %d -> %d",
            cutoff.strftime("%Y-%m-%d"),
            len(df),
            len(filtered_df),
        )
        return filtered_df

    # Unlocked access requested
    env_val = os.environ.get(UNLOCK_ENV_VAR, "")
    if env_val != UNLOCK_MAGIC_VALUE:
        raise PermissionError(
            f"LOCKBOX ACCESS DENIED! exclude_lockbox=False requires environment variable "
            f"{UNLOCK_ENV_VAR}='{UNLOCK_MAGIC_VALUE}'. Got: '{env_val}'"
        )

    # Authorized access: log it
    caller_frame = inspect.stack()[1]
    caller = f"{caller_frame.filename}:{caller_frame.lineno} in {caller_frame.function}"
    cfg_hash = get_config_hash(config_path)
    log_lockbox_access(caller, cfg_hash)

    return df
