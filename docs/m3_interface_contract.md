# M3 Interface Contract

This document defines the expected inputs from M1 (Data) and M2 (News) and the expected outputs from M3 (Modeling) for M4 (Dashboard) to consume.

## 1. Model Input Schema
M3 expects a unified, cleaned tabular dataset for each ticker, indexed by Date.

**Format**: Parquet or CSV.
**Key Columns**:
*   `Date` (YYYY-MM-DD)
*   `Ticker` (String)
*   `Open`, `High`, `Low`, `Close`, `Volume` (Numeric)
*   **M1 Indicators**: e.g., `SMA_20`, `RSI_14`, `MACD`, `BB_lower`, `BB_upper`, `Volatility_30d` (Numeric)
*   **M2 Sentiment**: `FinBERT_Sentiment_Score` (-1.0 to 1.0)
*   **M1 Context**: `India_VIX`, `Sector_Index_Close` (Numeric)

## 2. Model Output Schema (For M4)
M3 will provide predictions via JSON (or a database table) that M4's FastAPI backend will serve.

**Structure per Ticker per Date**:
```json
{
  "date": "2024-01-01",
  "ticker": "RELIANCE.NS",
  "predictions": {
    "1D": {
      "expected_return_pct": 0.5,
      "confidence_score": 0.85,
      "signal": "Buy"
    },
    "1W": {
      "expected_return_pct": 1.2,
      "confidence_score": 0.60,
      "signal": "Hold"
    },
    "1M": {
      "expected_return_pct": 3.5,
      "confidence_score": 0.90,
      "signal": "Buy"
    },
    "6M": {
      "expected_return_pct": 8.0,
      "confidence_score": 0.75,
      "signal": "Buy"
    }
  },
  "shap_summary": {
    "top_positive_features": ["RSI_14", "FinBERT_Sentiment_Score"],
    "top_negative_features": ["Volatility_30d"]
  }
}
```

### Definitions:
*   `expected_return_pct`: Predicted percentage return for the horizon.
*   `confidence_score`: Derived from Monte Carlo Dropout variance (0.0 to 1.0).
*   `signal`: Rule-based label (Buy/Hold/Avoid) based on return and confidence.
