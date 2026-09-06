# M4 REST API Contract Specification

This document details the FastAPI REST endpoints served by Module 4 (`src/m4_dashboard/backend/main.py`), consumed by the React front-end application.

---

## Base URL
Default local development URL: `http://localhost:8000`

---

## Endpoints

### 1. Health & System Status
- **Method**: `GET`
- **Route**: `/api/health`

### 2. Supported Market Universes & Tickers
- **Method**: `GET`
- **Route**: `/api/market/universes`

### 3. Market Overview & Key Macro Indicators
- **Method**: `GET`
- **Route**: `/api/market/overview`

### 4. Historical Prices & Technical Indicators
- **Method**: `GET`
- **Route**: `/api/stocks/{ticker}/prices`

### 5. News Headlines & Sentiment Scores
- **Method**: `GET`
- **Route**: `/api/stocks/{ticker}/sentiment`

### 6. Model Multi-Horizon Predictions & Uncertainty
- **Method**: `GET`
- **Route**: `/api/stocks/{ticker}/predictions`

### 7. Rule-Based Signal Engine Calls
- **Method**: `GET`
- **Route**: `/api/stocks/{ticker}/signal`

### 8. All Stocks Ranked Signals (Watchlist/Portfolio View)
- **Method**: `GET`
- **Route**: `/api/signals/ranked`

### 9. Historical Backtest & Baseline Comparison
- **Method**: `GET`
- **Route**: `/api/backtest`

### 10. Daily 'What Changed' Digest
- **Method**: `GET`
- **Route**: `/api/digest/daily`
