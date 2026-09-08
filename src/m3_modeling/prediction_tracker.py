"""
Prediction Tracker — Drift Monitoring & Rolling Accuracy
=========================================================
Reads predictions_log.jsonl and price_features.parquet to compute
rolling directional accuracy for each ticker and horizon.

Outputs accuracy_report.json to data/processed/ for the dashboard.

Usage:
    python -m src.m3_modeling.prediction_tracker
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_PATH = PROCESSED_DIR / "predictions_log.jsonl"
PRICE_PARQUET = PROCESSED_DIR / "price_features.parquet"
REPORT_PATH = PROCESSED_DIR / "accuracy_report.json"

# If rolling accuracy drops below this, recommend retraining
DRIFT_THRESHOLD = 0.52  # 52% — near random for directional prediction

HORIZON_FORWARD_DAYS = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "6M": 126,
}


def load_prediction_history():
    """Load all prediction entries from the JSONL log."""
    if not LOG_PATH.exists():
        return []

    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_price_data():
    """Load price features for realized return computation."""
    if not PRICE_PARQUET.exists():
        return None
    df = pd.read_parquet(PRICE_PARQUET)
    if 'ticker' not in df.columns and 'ticker' in df.index.names:
        df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_realized_returns(df_prices, ticker, prediction_date, horizons_to_check):
    """
    Compute actual realized returns for a given prediction date and ticker.
    Returns dict of {horizon_label: actual_return} or None if data not available.
    """
    df_t = df_prices[df_prices['ticker'] == ticker].sort_values('date')
    pred_date = pd.to_datetime(prediction_date)

    # Find the row closest to prediction date
    idx = df_t[df_t['date'] <= pred_date].index
    if len(idx) == 0:
        return {}

    base_idx = idx[-1]
    base_price = float(df_t.loc[base_idx, 'close'])

    realized = {}
    for h_label, forward_days in horizons_to_check.items():
        # Find the price `forward_days` trading days after the prediction
        future_rows = df_t[df_t['date'] > pred_date].head(forward_days + 5)
        if len(future_rows) >= forward_days:
            future_price = float(future_rows.iloc[forward_days - 1]['close'])
            realized[h_label] = (future_price - base_price) / base_price
        # else: not enough future data yet — skip this horizon

    return realized


def compute_accuracy_report():
    """
    Main function: computes rolling directional accuracy per ticker per horizon.
    Returns the report dict and writes it to disk.
    """
    entries = load_prediction_history()
    if not entries:
        print("[Tracker] No prediction log entries found. Run predict.py first.")
        return {"status": "no_data", "drift_detected": False}

    df_prices = load_price_data()
    if df_prices is None:
        print("[Tracker] No price data found.")
        return {"status": "no_price_data", "drift_detected": False}

    print(f"[Tracker] Processing {len(entries)} prediction log entries...")

    # Flatten predictions: list of (timestamp, ticker, horizon, predicted_return, predicted_direction)
    records = []
    for entry in entries:
        ts = entry.get('timestamp', '')
        preds = entry.get('predictions', {})
        for ticker, pred_data in preds.items():
            horizons = pred_data.get('horizons', {})
            current_price = pred_data.get('current_price', 0)

            # Determine the prediction date (use first date from price data near this timestamp)
            pred_date = ts[:10] if ts else None
            if not pred_date:
                continue

            for h_label, h_data in horizons.items():
                exp_ret = h_data.get('expected_return', 0)
                records.append({
                    'timestamp': ts,
                    'prediction_date': pred_date,
                    'ticker': ticker,
                    'horizon': h_label,
                    'predicted_return': exp_ret,
                    'predicted_direction': 1 if exp_ret > 0 else (-1 if exp_ret < 0 else 0),
                    'confidence': h_data.get('confidence', 0.5),
                })

    if not records:
        return {"status": "no_records", "drift_detected": False}

    df_preds = pd.DataFrame(records)

    # Compute realized returns for each prediction
    realized_data = []
    for _, row in df_preds.iterrows():
        h_label = row['horizon']
        if h_label not in HORIZON_FORWARD_DAYS:
            continue

        realized = compute_realized_returns(
            df_prices, row['ticker'], row['prediction_date'],
            {h_label: HORIZON_FORWARD_DAYS[h_label]}
        )

        if h_label in realized:
            actual_ret = realized[h_label]
            actual_dir = 1 if actual_ret > 0 else (-1 if actual_ret < 0 else 0)
            correct = 1 if row['predicted_direction'] == actual_dir else 0
            realized_data.append({
                'ticker': row['ticker'],
                'horizon': h_label,
                'prediction_date': row['prediction_date'],
                'predicted_return': row['predicted_return'],
                'actual_return': actual_ret,
                'correct_direction': correct,
                'confidence': row['confidence'],
            })

    if not realized_data:
        print("[Tracker] No realized returns available yet (predictions too recent).")
        return {
            "status": "awaiting_outcomes",
            "drift_detected": False,
            "predictions_tracked": len(records),
            "outcomes_available": 0,
        }

    df_results = pd.DataFrame(realized_data)
    print(f"[Tracker] {len(df_results)} predictions with realized outcomes")

    # ── Per-ticker, per-horizon rolling accuracy ──────────────────────
    ticker_reports = {}
    drift_detected = False

    for (ticker, horizon), grp in df_results.groupby(['ticker', 'horizon']):
        grp = grp.sort_values('prediction_date')
        n = len(grp)

        # Rolling window: last 30 predictions or all available
        window = min(30, n)
        recent = grp.tail(window)

        dir_accuracy = recent['correct_direction'].mean()
        mae = float(np.mean(np.abs(recent['predicted_return'] - recent['actual_return'])))
        avg_confidence = float(recent['confidence'].mean())

        if dir_accuracy < DRIFT_THRESHOLD and n >= 10:
            drift_detected = True
            status = "DRIFT_DETECTED"
        elif dir_accuracy < 0.55 and n >= 10:
            status = "MARGINAL"
        else:
            status = "HEALTHY"

        if ticker not in ticker_reports:
            ticker_reports[ticker] = {}

        ticker_reports[ticker][horizon] = {
            "directional_accuracy": round(float(dir_accuracy), 4),
            "mae": round(mae, 6),
            "avg_confidence": round(avg_confidence, 4),
            "sample_count": n,
            "window_size": window,
            "status": status,
        }

    # ── Aggregate summary ─────────────────────────────────────────────
    all_correct = df_results['correct_direction'].values
    overall_accuracy = float(np.mean(all_correct)) if len(all_correct) > 0 else 0.0

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "drift_detected" if drift_detected else "healthy",
        "drift_detected": drift_detected,
        "overall_directional_accuracy": round(overall_accuracy, 4),
        "total_predictions_evaluated": len(df_results),
        "total_predictions_logged": len(records),
        "drift_threshold": DRIFT_THRESHOLD,
        "per_ticker": ticker_reports,
    }

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Tracker] Accuracy report → {REPORT_PATH}")
    print(f"[Tracker] Overall directional accuracy: {overall_accuracy:.2%}")
    if drift_detected:
        print("[Tracker] ⚠️  DRIFT DETECTED — consider retraining the model")

    return report


def check_drift():
    """Quick check: returns True if model drift is detected."""
    report = compute_accuracy_report()
    return report.get('drift_detected', False)


if __name__ == "__main__":
    compute_accuracy_report()
