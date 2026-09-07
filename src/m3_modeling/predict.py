"""
M3 Prediction Generator
========================
Loads trained LSTM model, runs inference on latest data from
data/processed/, and writes predictions.json consumed by the M4 dashboard.

Usage:
    python -m src.m3_modeling.predict            # from project root
    python src/m3_modeling/predict.py             # direct
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.m3_modeling.lstm_model import MultiHorizonLSTM, mc_dropout_inference


# ── Paths ──────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_PATH = PROCESSED_DIR / "predictions.json"

PRICE_PARQUET = PROCESSED_DIR / "price_features.parquet"
SENTIMENT_PARQUET = PROCESSED_DIR / "sentiment_features.parquet"

LSTM_PATH = MODEL_DIR / "best_lstm.pth"

FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.pkl"

SEQUENCE_LENGTH = 60
MC_PASSES = 50
HORIZONS = ["1D", "5D", "21D", "126D"]
HORIZON_LABELS = {"1D": "1D", "5D": "1W", "21D": "1M", "126D": "6M"}

# ── Human-readable feature name map ───────────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "rsi_14": "RSI (14-day)",
    "rsi_7": "RSI (7-day)",
    "rsi_21": "RSI (21-day)",
    "rsi_28": "RSI (28-day)",
    "macd_line_12_26": "MACD Line (12,26)",
    "macd_hist_12_26": "MACD Histogram",
    "macd_signal_12_26": "MACD Signal",
    "dist_sma_50": "Distance from 50-day SMA",
    "dist_sma_20": "Distance from 20-day SMA",
    "dist_sma_200": "Distance from 200-day SMA",
    "dist_ema_50": "Distance from 50-day EMA",
    "dist_ema_20": "Distance from 20-day EMA",
    "sma_50": "50-day Simple Moving Average",
    "sma_20": "20-day Simple Moving Average",
    "sma_200": "200-day Simple Moving Average",
    "ema_50": "50-day Exponential Moving Average",
    "bb_width_20": "Bollinger Band Width (20)",
    "bb_pct_b_20": "Bollinger %B (20)",
    "bb_upper_20": "Bollinger Upper Band (20)",
    "bb_lower_20": "Bollinger Lower Band (20)",
    "atr_14": "Average True Range (14)",
    "natr_14": "Normalized ATR (14)",
    "volatility_20d": "20-day Volatility",
    "volatility_60d": "60-day Volatility",
    "volatility_5d": "5-day Volatility",
    "volume": "Trading Volume",
    "vol_ratio_20": "Volume Ratio (20-day)",
    "vol_ratio_5": "Volume Ratio (5-day)",
    "obv": "On-Balance Volume",
    "cmf_20": "Chaikin Money Flow (20)",
    "pvt": "Price Volume Trend",
    "adx_14": "ADX Trend Strength (14)",
    "plus_di_14": "Positive Direction Index",
    "minus_di_14": "Negative Direction Index",
    "stoch_k_14": "Stochastic %K (14)",
    "stoch_d_14": "Stochastic %D (14)",
    "williams_r_14": "Williams %R (14)",
    "cci_14": "Commodity Channel Index (14)",
    "cci_20": "Commodity Channel Index (20)",
    "return_1d": "1-day Return",
    "return_5d": "5-day Return",
    "return_21d": "21-day Return",
    "return_63d": "63-day Return",
    "close": "Closing Price",
    "open": "Opening Price",
    "high": "Session High",
    "low": "Session Low",
    "hl_spread_pct": "High-Low Spread %",
    "co_spread_pct": "Close-Open Spread %",
    "close_slope_20": "20-day Price Slope",
    "close_slope_10": "10-day Price Slope",
    "close_slope_5": "5-day Price Slope",
    "vix_close": "India VIX Level",
    "vix_close_return": "India VIX Daily Change",
    "nifty_50_close": "Nifty 50 Level",
    "nifty_50_close_return": "Nifty 50 Daily Return",
    "nifty_bank_close": "Nifty Bank Level",
    "nifty_bank_close_return": "Nifty Bank Daily Return",
    "sma_50_200_ratio": "SMA 50/200 Golden Cross Ratio",
    "sma_20_50_ratio": "SMA 20/50 Ratio",
    "donchian_pos_20": "Donchian Channel Position (20)",
    "keltner_pos_20": "Keltner Channel Position (20)",
    "sentiment_mean": "News Sentiment Score",
    "sentiment_pos_ratio": "Positive News Ratio",
    "sentiment_neg_ratio": "Negative News Ratio",
    "sentiment_momentum_3d": "3-day Sentiment Momentum",
    "sentiment_momentum_7d": "7-day Sentiment Momentum",
}

FEATURE_DESCRIPTIONS = {
    "rsi_14": "Measures if the stock is overbought (>70) or oversold (<30) over 14 days",
    "macd_line_12_26": "Shows momentum direction — positive means bullish momentum building",
    "macd_hist_12_26": "Histogram gap between MACD and signal — widening means stronger trend",
    "dist_sma_50": "How far price is from its 50-day average — positive means trading above trend",
    "dist_sma_20": "How far price is from its 20-day average — measures short-term deviation",
    "dist_sma_200": "How far price is from its 200-day average — key long-term trend indicator",
    "atr_14": "Average daily price range over 14 days — higher means more volatile",
    "volatility_20d": "Annualized price volatility over 20 trading days",
    "bb_width_20": "Width of Bollinger Bands — narrow means potential breakout ahead",
    "bb_pct_b_20": "Position within Bollinger Bands — above 1.0 means overbought",
    "adx_14": "Trend strength — above 25 means strong trend, below 20 means ranging",
    "volume": "Number of shares traded — high volume confirms price moves",
    "vol_ratio_20": "Today's volume vs 20-day average — >1 means above-average activity",
    "obv": "Cumulative volume direction — rising OBV confirms uptrend",
    "cmf_20": "Money flow in/out — positive means institutional buying pressure",
    "stoch_k_14": "Momentum oscillator — above 80 is overbought, below 20 is oversold",
    "williams_r_14": "Overbought/oversold oscillator — near 0 is overbought, near -100 oversold",
    "return_1d": "Yesterday's price change percentage",
    "return_5d": "Price change over the past week",
    "return_21d": "Price change over the past month",
    "close": "Most recent closing price",
    "close_slope_20": "Linear trend direction over 20 days — positive means upward slope",
    "vix_close": "India VIX — market fear gauge. High VIX means expected turbulence",
    "nifty_50_close_return": "Broader market return — shows if market tailwind or headwind",
    "sma_50_200_ratio": "Golden cross ratio — above 1.0 means 50-day SMA above 200-day (bullish)",
    "sentiment_mean": "Average news sentiment from FinBERT NLP — positive means bullish headlines",
    "sentiment_pos_ratio": "Fraction of recent news articles with positive tone",
    "sentiment_momentum_3d": "How fast sentiment is changing over 3 days",
}


def _get_feature_cols(df):
    """Select numeric feature columns, excluding targets and identifiers."""
    exclude = {'ticker', 'date', 'market_regime',
               'Target_1D', 'Target_5D', 'Target_21D', 'Target_126D'}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def generate_predictions() -> dict:
    """
    Main entry point. Returns the predictions dict AND writes it to disk.
    """
    print("[M3] Loading data …")
    df_price = pd.read_parquet(PRICE_PARQUET)
    if 'ticker' not in df_price.columns and 'ticker' in df_price.index.names:
        df_price = df_price.reset_index()
    elif 'ticker' not in df_price.columns:
        print("[M3 Error] price_features.parquet is missing 'ticker'. Please run refresh_data.py")
        return {}

    df_price['date'] = pd.to_datetime(df_price['date'])

    # Merge sentiment
    if SENTIMENT_PARQUET.exists():
        df_sent = pd.read_parquet(SENTIMENT_PARQUET)
        if 'ticker' not in df_sent.columns and 'ticker' in df_sent.index.names:
            df_sent = df_sent.reset_index()
        df_sent['date'] = pd.to_datetime(df_sent['date'])
        df_price = pd.merge(df_price, df_sent, on=['date', 'ticker'], how='left')
        df_price.fillna(0, inplace=True)

    feature_cols = _get_feature_cols(df_price)
    tickers = sorted(df_price['ticker'].unique())
    print(f"[M3] {len(tickers)} tickers, {len(feature_cols)} features")

    # ── Load models & scalers ──────────────────────────────────────────
    f_scaler = joblib.load(FEATURE_SCALER_PATH)
    t_scaler = joblib.load(TARGET_SCALER_PATH)

    # Determine number of features the LSTM was trained on
    num_model_features = f_scaler.n_features_in_


    # Load LSTM
    lstm = MultiHorizonLSTM(
        input_dim=num_model_features,
        hidden_dim=64,
        num_layers=2,
        output_dim=4,
        dropout_rate=0.3,
    )
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location="cpu", weights_only=True))

    # ── Per-ticker inference ───────────────────────────────────────────
    predictions = {}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for ticker in tickers:
        df_t = df_price[df_price['ticker'] == ticker].sort_values('date').copy()
        if len(df_t) < SEQUENCE_LENGTH + 1:
            print(f"  [SKIP] {ticker}: only {len(df_t)} rows")
            continue

        latest_row = df_t.iloc[-1]
        current_price = float(latest_row['close'])
        latest_date = str(latest_row['date'])[:10]

        # Build feature matrix — use the same cols the scaler saw
        # The scaler was fit on the training data's numeric cols.
        # We need to match the same number of columns.
        available_feature_cols = [c for c in feature_cols if c in df_t.columns]

        # Pad or trim to match scaler expectation
        feat_matrix = df_t[available_feature_cols].values
        if feat_matrix.shape[1] < num_model_features:
            pad = np.zeros((feat_matrix.shape[0], num_model_features - feat_matrix.shape[1]))
            feat_matrix = np.hstack([feat_matrix, pad])
        elif feat_matrix.shape[1] > num_model_features:
            feat_matrix = feat_matrix[:, :num_model_features]

        # Scale
        try:
            scaled = f_scaler.transform(feat_matrix)
        except Exception as e:
            print(f"  [SKIP] {ticker}: scaler error — {e}")
            continue

        # Last SEQUENCE_LENGTH window
        window = scaled[-SEQUENCE_LENGTH:]
        x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        # ── LSTM MC Dropout ────────────────────────────────────────────
        mean_pred, std_pred = mc_dropout_inference(lstm, x_tensor, num_passes=MC_PASSES)
        # Inverse-transform predictions back to return space
        mean_ret = t_scaler.inverse_transform(mean_pred)[0]   # shape (4,)
        std_ret = t_scaler.inverse_transform(
            mean_pred + std_pred
        )[0] - t_scaler.inverse_transform(mean_pred)[0]       # approx std in return space
        std_ret = np.abs(std_ret)

        # ── Build horizons dict ────────────────────────────────────────
        horizons = {}
        for i, h_raw in enumerate(HORIZONS):
            h_label = HORIZON_LABELS[h_raw]
            ret = float(mean_ret[i])
            std = float(std_ret[i])
            pred_price = current_price * (1 + ret)
            conf = float(max(0.0, min(1.0, 1.0 / (1.0 + std * 10))))
            horizons[h_label] = {
                "expected_return": round(ret, 6),
                "predicted_price": round(pred_price, 2),
                "confidence": round(conf, 4),
                "uncertainty_std": round(std, 6),
                "lower_bound": round(current_price * (1 + ret - 1.96 * std), 2),
                "upper_bound": round(current_price * (1 + ret + 1.96 * std), 2),
            }

        # ── Ensemble comparison (legacy compatibility) ───────────────────
        lstm_1m = float(mean_ret[2])
        ensemble = {
            "lstm_return_1M": round(lstm_1m, 6),
            "disagreement_detected": False,
            "disagreement_delta": 0.0,
        }

        # ── Feature importance (LSTM gradient attribution for 21D horizon) ──
        shap_list = []
        lstm.eval()
        x_attr = x_tensor.clone().detach().requires_grad_(True)
        pred_attr = lstm(x_attr)
        # 21D horizon is index 2
        pred_attr[0, 2].backward()
        
        if x_attr.grad is not None:
            # Saliency attribution: sum absolute gradient across time sequence
            grad_abs = x_attr.grad.abs().sum(dim=1).squeeze(0).numpy()
            raw_grad = x_attr.grad.sum(dim=1).squeeze(0).numpy()
            
            imp_per_feat = grad_abs[:num_model_features]
            raw_grad_feat = raw_grad[:num_model_features]
            
            total = imp_per_feat.sum()
            if total > 0:
                imp_per_feat /= total
                
            feat_names = available_feature_cols[:num_model_features]
            top_indices = np.argsort(imp_per_feat)[::-1][:8]
            
            for idx in top_indices:
                if imp_per_feat[idx] < 0.005:
                    continue
                feat_name = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
                # Direction matches the sign of the raw gradient
                direction = "positive" if raw_grad_feat[idx] >= 0 else "negative"
                
                shap_list.append({
                    "feature": feat_name,
                    "feature_display": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name.replace('_', ' ').title()),
                    "feature_description": FEATURE_DESCRIPTIONS.get(feat_name, ""),
                    "impact": round(float(imp_per_feat[idx]) * (1 if direction == "positive" else -1), 6),
                    "importance_score": round(float(imp_per_feat[idx]), 4),
                    "direction": direction,
                    "current_value": round(float(latest_row.get(feat_name, 0)), 4) if feat_name in latest_row.index else None,
                })

        predictions[ticker] = {
            "ticker": ticker,
            "prediction_date": latest_date,
            "current_price": round(current_price, 2),
            "horizons": horizons,
            "ensemble": ensemble,
            "shap_explainability": shap_list,
            "generated_at": now_str,
        }
        print(f"  ✓ {ticker}: 1M return={lstm_1m*100:+.2f}%, conf={horizons['1M']['confidence']:.0%}")

    # ── Write to disk ──────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\n[M3] Wrote predictions for {len(predictions)} tickers → {OUTPUT_PATH}")

    return predictions


if __name__ == "__main__":
    generate_predictions()
