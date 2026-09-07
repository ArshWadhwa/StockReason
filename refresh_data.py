#!/usr/bin/env python3
"""
StockReason — Unified Data & Feature Refresh Script
Usage:
    python refresh_data.py           # Runs M1 (Price) + M2 (News & Sentiment) + M3 (Predictions)
    python refresh_data.py --m1      # Runs M1 Price & Market Data pipeline only
    python refresh_data.py --m2      # Runs M2 News & Sentiment pipeline only
    python refresh_data.py --m3      # Runs M3 Prediction Generation only
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.m1_price_data.build_features import build_pipeline as build_m1_pipeline
from src.m2_sentiment.pipeline import run_sentiment_pipeline as build_m2_pipeline


def main():
    parser = argparse.ArgumentParser(description="StockReason Data & Sentiment Pipeline Refresh")
    parser.add_argument("--m1", action="store_true", help="Run M1 Price Data pipeline only")
    parser.add_argument("--m2", action="store_true", help="Run M2 News & Sentiment pipeline only")
    parser.add_argument("--m3", action="store_true", help="Run M3 Prediction Generation only")
    parser.add_argument("--fast", action="store_true", help="Use fast baseline scoring for M2")
    args = parser.parse_args()

    run_all = not (args.m1 or args.m2 or args.m3)

    if args.m1 or run_all:
        print("\n=======================================================")
        print(">>> Launching StockReason M1: Price & Technical Pipeline")
        print("=======================================================")
        try:
            build_m1_pipeline()
        except Exception as e:
            print(f"[M1 Error] {e}")

    if args.m2 or run_all:
        print("\n=======================================================")
        print(">>> Launching StockReason M2: News & Sentiment Pipeline")
        print("=======================================================")
        try:
            build_m2_pipeline(use_finbert=not args.fast)
        except Exception as e:
            print(f"[M2 Error] {e}")

    if args.m3 or run_all:
        print("\n=======================================================")
        print(">>> Launching StockReason M3: Prediction Generation")
        print("=======================================================")
        try:
            from src.m3_modeling.predict import generate_predictions
            generate_predictions()
        except Exception as e:
            print(f"[M3 Error] {e}")

    print("\n[Refresh Complete] Datasets updated in data/processed/")


if __name__ == "__main__":
    main()
