#!/usr/bin/env python3
"""
StockReason — Daily Data & Feature Refresh Script
Usage:
    python refresh_data.py

Runs the complete M1 data pipeline:
Fetch Raw Data -> Clean & Align -> Merge Market Context -> 100+ Features -> Regime Detection -> Export price_features.parquet
"""
import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.m1_price_data.build_features import build_pipeline

if __name__ == "__main__":
    print("Launching StockReason M1 Data & Feature Pipeline...")
    build_pipeline()
