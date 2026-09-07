# Model Output Schema Specification (M3 Contract)

This document establishes the official interface contract between Module 3 (`m3_modeling`) and Module 4 (`m4_dashboard`).

---

## 1. Prediction Payload Format

Module 3 generates multi-horizon predictions with Monte Carlo Dropout uncertainty estimates and gradient-based attribution vectors.

### Horizon Forecast Structure
Each ticker forecast contains predictions for four distinct time horizons:
- `1D` (Next Day)
- `1W` (Next 5 Trading Days)
- `1M` (Next 21 Trading Days)
- `6M` (Next 126 Trading Days)

For each horizon:
- `expected_return`: Projected fractional return (e.g., `0.035` = +3.5%)
- `predicted_price`: Target price in INR based on latest close
- `confidence`: Confidence score in `[0.0, 1.0]` calibrated from MC Dropout variance
- `uncertainty_std`: Standard deviation of returns across 50 MC Dropout stochastic passes
- `lower_bound`: 95% confidence interval floor
- `upper_bound`: 95% confidence interval ceiling

### Ensemble Structure
- `lstm_return_1M`: LSTM model predicted return
- `disagreement_detected`: Legacy boolean (currently false)
- `disagreement_delta`: Legacy delta (currently 0.0)

### Feature Explainability Field
List of top feature attributions explaining the direction and magnitude of the primary 1M prediction:
- `feature`: Feature name (e.g. `rsi_14`, `dist_sma_50`, `daily_sentiment_score`, `india_vix_return_5d`)
- `impact`: Raw SHAP value
- `importance_score`: Normalized importance `[0.0, 1.0]`
- `direction`: `"positive"` (drives return higher) or `"negative"` (drives return lower)
