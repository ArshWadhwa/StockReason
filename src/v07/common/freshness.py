"""
freshness.py
------------
Data freshness monitor and gatekeeper for StockReason v0.7.

Reports staleness per source:
  1. Price parquet/csv
  2. Sentiment parquet/csv
  3. Predictions JSON

Calculates elapsed NSE trading days between each source's latest timestamp
and today's date using the NSE trading holiday calendar.
Refuses to publish forecasts older than a configurable limit (default: 3 trading days).
Exposes the data-as-of date and freshness status for API payloads.

CLI usage:
  python -m v07.common.freshness [--config config/v07.yaml] [--max-stale 3] [--as-of YYYY-MM-DD]
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.m1_price_data.nse_holidays import is_trading_day
from src.v07.common.config import get_project_root, load_v07_config
from src.v07.common.validation import count_trading_days_elapsed


class StaleDataError(RuntimeError):
    """Raised when data staleness exceeds the allowable threshold for publishing forecasts."""
    pass


@dataclass
class SourceFreshness:
    source_name: str
    resolved_path: str
    exists: bool
    latest_date: Optional[str]
    stale_trading_days: Optional[int]
    is_fresh: bool
    error: Optional[str] = None


@dataclass
class FreshnessReport:
    as_of_date: str
    max_stale_trading_days: int
    sources: Dict[str, SourceFreshness] = field(default_factory=dict)
    overall_latest_price_date: Optional[str] = None
    max_stale_days_observed: int = 0
    can_publish_forecast: bool = True
    summary: str = ""

    def to_api_payload(self) -> Dict[str, Any]:
        """Returns standard payload for API responses and downstream payloads."""
        return {
            "data_as_of": self.overall_latest_price_date,
            "checked_at": self.as_of_date,
            "trading_days_stale": self.max_stale_days_observed,
            "max_allowed_stale_days": self.max_stale_trading_days,
            "can_publish_forecast": self.can_publish_forecast,
            "status": "FRESH" if self.can_publish_forecast else "STALE",
            "sources": {
                name: {
                    "path": s.resolved_path,
                    "exists": s.exists,
                    "latest_date": s.latest_date,
                    "stale_trading_days": s.stale_trading_days,
                    "is_fresh": s.is_fresh,
                }
                for name, s in self.sources.items()
            },
        }


def extract_latest_price_date(path: Path) -> Optional[date]:
    """Reads the maximum date from price parquet or csv."""
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=["date"])
    else:
        df = pd.read_csv(path, usecols=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date()


def extract_latest_sentiment_date(path: Path) -> Optional[date]:
    """Reads the maximum date from sentiment parquet or csv."""
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=["date"])
    else:
        df = pd.read_csv(path, usecols=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date()


def extract_latest_predictions_date(path: Path) -> Optional[date]:
    """Reads the latest prediction_date from predictions.json."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        return None

    # Check top-level metadata first
    if "prediction_date" in data:
        return pd.to_datetime(data["prediction_date"]).date()
    if "data_as_of" in data:
        return pd.to_datetime(data["data_as_of"]).date()

    # Search per-ticker entries
    dates = []
    for k, v in data.items():
        if isinstance(v, dict):
            if "prediction_date" in v:
                dates.append(pd.to_datetime(v["prediction_date"]).date())
            elif "date" in v:
                dates.append(pd.to_datetime(v["date"]).date())

    return max(dates) if dates else None


def check_source_freshness(
    source_name: str,
    path: Path,
    extractor: Any,
    as_of: date,
    max_stale_days: int,
) -> SourceFreshness:
    """Checks freshness for a single file source."""
    if not path.exists():
        return SourceFreshness(
            source_name=source_name,
            resolved_path=str(path),
            exists=False,
            latest_date=None,
            stale_trading_days=None,
            is_fresh=False,
            error=f"File not found: {path}",
        )

    try:
        latest = extractor(path)
        if latest is None:
            return SourceFreshness(
                source_name=source_name,
                resolved_path=str(path),
                exists=True,
                latest_date=None,
                stale_trading_days=None,
                is_fresh=False,
                error="File is empty or contains no valid dates.",
            )

        stale_days = count_trading_days_elapsed(latest, as_of)
        is_fresh = stale_days <= max_stale_days

        return SourceFreshness(
            source_name=source_name,
            resolved_path=str(path),
            exists=True,
            latest_date=str(latest),
            stale_trading_days=stale_days,
            is_fresh=is_fresh,
        )
    except Exception as e:
        return SourceFreshness(
            source_name=source_name,
            resolved_path=str(path),
            exists=True,
            latest_date=None,
            stale_trading_days=None,
            is_fresh=False,
            error=str(e),
        )


def check_freshness(
    config_path: Optional[str] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
    max_stale_trading_days: Optional[int] = None,
    price_path: Optional[Union[str, Path]] = None,
    sentiment_path: Optional[Union[str, Path]] = None,
    predictions_path: Optional[Union[str, Path]] = None,
) -> FreshnessReport:
    """
    Evaluates freshness across all data sources.
    
    Args:
        config_path: Path to v07.yaml.
        as_of_date: Reference date (defaults to today).
        max_stale_trading_days: Maximum allowable stale trading days (default from config or 3).
        price_path: Custom override path for price dataset.
        sentiment_path: Custom override path for sentiment dataset.
        predictions_path: Custom override path for predictions JSON.
    """
    root = get_project_root()
    cfg = load_v07_config(config_path)

    if max_stale_trading_days is None:
        max_stale_trading_days = cfg.get("freshness", {}).get("max_stale_trading_days", 3)

    ref_date = pd.to_datetime(as_of_date).date() if as_of_date else datetime.now().date()

    paths_cfg = cfg.get("paths", {})

    # Resolve price path (check parquet, fallback to csv if parquet missing)
    if price_path:
        p_path = Path(price_path)
    else:
        p_path = root / paths_cfg.get("price_features_parquet", "data/processed/price_features.parquet")
        if not p_path.exists():
            csv_candidate = root / paths_cfg.get("price_features_csv", "data/processed/price_features.csv")
            if csv_candidate.exists():
                p_path = csv_candidate

    # Resolve sentiment path
    if sentiment_path:
        s_path = Path(sentiment_path)
    else:
        s_path = root / paths_cfg.get("sentiment_features_parquet", "data/processed/sentiment_features.parquet")
        if not s_path.exists():
            csv_candidate = root / paths_cfg.get("sentiment_features_csv", "data/processed/sentiment_features.csv")
            if csv_candidate.exists():
                s_path = csv_candidate

    # Resolve predictions path
    if predictions_path:
        pred_path = Path(predictions_path)
    else:
        pred_path = root / paths_cfg.get("predictions_json", "data/processed/predictions.json")

    sources: Dict[str, SourceFreshness] = {}

    sources["price"] = check_source_freshness("price", p_path, extract_latest_price_date, ref_date, max_stale_trading_days)
    sources["sentiment"] = check_source_freshness("sentiment", s_path, extract_latest_sentiment_date, ref_date, max_stale_trading_days)
    sources["predictions"] = check_source_freshness("predictions", pred_path, extract_latest_predictions_date, ref_date, max_stale_trading_days)

    price_date = sources["price"].latest_date
    stale_days_list = [s.stale_trading_days for s in sources.values() if s.stale_trading_days is not None]
    max_stale_observed = max(stale_days_list) if stale_days_list else 999

    # For publishing forecasts, price data must exist and be <= max_stale_trading_days
    price_fresh = sources["price"].is_fresh and sources["price"].exists
    can_publish = price_fresh

    summary_items = []
    for name, s in sources.items():
        if s.exists:
            summary_items.append(f"{name}: {s.latest_date} ({s.stale_trading_days}d stale)")
        else:
            summary_items.append(f"{name}: MISSING")

    summary = f"As of {ref_date} | Can publish: {can_publish} | " + " | ".join(summary_items)

    return FreshnessReport(
        as_of_date=str(ref_date),
        max_stale_trading_days=max_stale_trading_days,
        sources=sources,
        overall_latest_price_date=price_date,
        max_stale_days_observed=max_stale_observed,
        can_publish_forecast=can_publish,
        summary=summary,
    )


def enforce_freshness_for_forecast(
    config_path: Optional[str] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
    max_stale_trading_days: Optional[int] = None,
    price_path: Optional[Union[str, Path]] = None,
) -> FreshnessReport:
    """
    Validates data freshness and RAISES StaleDataError if price data is stale.
    Use in prediction or forecasting pipelines before executing models.
    """
    report = check_freshness(
        config_path=config_path,
        as_of_date=as_of_date,
        max_stale_trading_days=max_stale_trading_days,
        price_path=price_path,
    )

    if not report.can_publish_forecast:
        price_src = report.sources.get("price")
        price_date = price_src.latest_date if price_src else "UNKNOWN"
        stale_days = price_src.stale_trading_days if price_src else "UNKNOWN"
        raise StaleDataError(
            f"Forecast publication rejected due to stale market data. "
            f"Data as of {price_date} is {stale_days} trading days stale. "
            f"(Configured threshold: {report.max_stale_trading_days} trading days)."
        )

    return report


def get_api_freshness_payload(
    config_path: Optional[str] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> Dict[str, Any]:
    """Convenience helper to expose data-as-of status in API payload responses."""
    report = check_freshness(config_path=config_path, as_of_date=as_of_date)
    return report.to_api_payload()


def main():
    parser = argparse.ArgumentParser(description="StockReason v0.7 Data Freshness Monitor")
    parser.add_argument("--config", default="config/v07.yaml", help="Path to config file")
    parser.add_argument("--max-stale", type=int, default=None, help="Maximum allowed stale trading days")
    parser.add_argument("--as-of", default=None, help="Evaluation date (YYYY-MM-DD), default today")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if stale")
    args = parser.parse_args()

    report = check_freshness(
        config_path=args.config,
        as_of_date=args.as_of,
        max_stale_trading_days=args.max_stale,
    )

    print("================================================================================")
    print("                    STOCKREASON v0.7 DATA FRESHNESS REPORT                      ")
    print("================================================================================")
    print(f"Evaluation Date (Reference) : {report.as_of_date}")
    print(f"Configured Max Stale Days   : {report.max_stale_trading_days} trading days")
    print(f"Forecast Publishing Allowed : {'YES [PASS]' if report.can_publish_forecast else 'NO [REJECT]'}")
    print("--------------------------------------------------------------------------------")
    print(f"{'Source':<15} {'Latest Date':<14} {'Stale (Days)':<14} {'Status':<10} {'Path'}")
    print("--------------------------------------------------------------------------------")

    for name, s in report.sources.items():
        dt_str = s.latest_date if s.latest_date else "N/A"
        stale_str = f"{s.stale_trading_days} days" if s.stale_trading_days is not None else "N/A"
        status_str = "FRESH" if s.is_fresh else ("MISSING" if not s.exists else "STALE")
        rel_path = s.resolved_path
        print(f"{name:<15} {dt_str:<14} {stale_str:<14} {status_str:<10} {rel_path}")

    print("================================================================================")
    print(report.summary)
    print("================================================================================")

    if args.strict and not report.can_publish_forecast:
        sys.exit(1)


if __name__ == "__main__":
    main()
