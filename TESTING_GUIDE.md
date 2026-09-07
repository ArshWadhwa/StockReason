# StockReason — Quick Verification Guide

A concise checklist to verify that both backend and frontend are operating correctly.

---

## 1. Quick Automated Test (30s)

Verify core data models, feature generation, and sentiment failover:

```bash
python -m pytest tests/
```
> **Expected**: `11 passed`

---

## 2. Start Backend & Verify

In Terminal 1:
```bash
python main.py
```
> Server runs on `http://localhost:8000`

### What to check:
1. **Health Check**: Open `http://localhost:8000/api/health` in your browser.  
   *Expected JSON*: `{"status": "ok", "ready_for_week4_swap": true}`
2. **Interactive Docs**: Open `http://localhost:8000/docs`.  
   *Expected*: Swagger UI listing all endpoints (`/api/market/...`, `/api/stocks/...`, `/api/signals/...`).
3. **News & Sentiment Check**: Open `http://localhost:8000/api/stocks/RELIANCE.NS/sentiment`.  
   *Expected*: Returns sentiment scores and news articles (Finnhub + Google News fallback).

---

## 3. Start Frontend & Verify

In Terminal 2:
```bash
cd frontend
npm run dev
```
> App runs on `http://localhost:5173`

### What to look for in the UI:
1. **Live Header Tape**: Top ticker tape displays indices (`NIFTY 50`, `BANK NIFTY`, `INDIA VIX`) with price changes.
2. **Stock Selector**: Switch between stocks (e.g., `RELIANCE.NS`, `INFY.NS`, `TCS.NS`). Price chart and indicators render cleanly.
3. **News & Sentiment Panel**: Shows headline cards with colored polarity badges (`Positive`, `Negative`, `Neutral`) and confidence scores.
4. **Decision Engine & Signal Card**: Displays model prediction, confidence percentage, and actionable signal (`BUY`, `HOLD`, or `AVOID`).
5. **Backtest / Digest Views**: Switch tabs to verify equity curve simulation and daily market summary cards.

---

## 4. Optional: Refresh Live Data Pipelines

To run full real-time data fetching (price + news + model scoring):
```bash
# Fast mode (uses lightweight textblob baseline for quick turnaround)
python refresh_data.py --fast

# Full mode (uses transformer FinBERT model)
python refresh_data.py
```
