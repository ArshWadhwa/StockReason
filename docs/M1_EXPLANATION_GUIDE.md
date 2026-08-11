# M1 — Presentation & Explanation Guide: Stock & Market Data Pipeline

This document is your **presentation cheat sheet and technical guide** for explaining your work on **Module 1 (`m1_price_data`)** to your professor, teacher, project guide, and teammates.

---

## 💡 Executive Summary (The 30-Second Pitch)

> *"In the **StockReason** project, my responsibility was **Module 1: Price & Market Data Engineering**. I built an automated, production-grade Python data pipeline that fetches 5+ years of historical market data for Indian stock universes (NIFTY 50, Nifty Bank, Nifty IT, Nifty Next 50) along with market benchmarks (NIFTY 50, Nifty Bank, Nifty IT), market volatility (India VIX), and macroeconomic drivers (USD/INR exchange rate, Crude Oil).*
> 
> *The pipeline automatically cleans the raw time series, merges market context, engineers **over 160 technical indicators and historical lag features with zero data leakage**, classifies market volatility regimes (`TRENDING`, `CHOPPY`, `VOLATILE`), and exports a single, high-performance dataset (`price_features.parquet`) that feeds directly into the downstream Sentiment (M2), Machine Learning Modeling (M3), and Dashboard (M4) tracks."*

---

## 🏗️ What Does Your Code Do Step-by-Step?

Your pipeline works like a factory line in 5 automated stages:

```
    [1. Fetch Raw Data]
(Stocks, Indices, VIX, USD/INR, Crude Oil)
           │
           ▼
   [2. Clean & Align]
(Deduplicate, Impute NaNs, Align Dates)
           │
           ▼
[3. Feature Engineering]
(160+ Technical Indicators: RSI, MACD, BB, ATR, Vol)
           │
           ▼
 [4. Anti-Leakage Lags & Regime]
(Shifted Historical Lags + TRENDING/CHOPPY/VOLATILE)
           │
           ▼
[5. Export & Handover]
(price_features.parquet + Quality Reports)
```

---

### Step 1: Data Acquisition (`src/m1_price_data/fetcher.py`)
- **Stock Universe**: Fetches daily OHLCV (`Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`) for pilot & full universes across NIFTY 50, Nifty Bank, Nifty IT, and Nifty Next 50.
- **Market Context & Benchmarks**:
  - `NIFTY 50` (`^NSEI`): Broad market benchmark.
  - `NIFTY Bank` (`^NSEBANK`): Sector benchmark for financial stocks.
  - `NIFTY IT` (`^CNXIT`): Sector benchmark for IT stocks.
  - `India VIX` (`^INDIAVIX`): Market volatility / fear gauge.
- **Macro Drivers**:
  - `USD/INR` (`INR=X`): Currency exchange rate impacting IT exports and foreign institutional investment.
  - `Crude Oil WTI` (`CL=F`): Commodity price driving inflation and import costs for India.

---

### Step 2: Data Cleaning & Alignment (`src/m1_price_data/cleaner.py`)
- **Deduplication**: Removes duplicate date-ticker entries.
- **Missing Value Handling**: Replaces bad/negative price anomalies and performs forward-fill (`ffill`) followed by backward-fill (`bfill`).
- **Calendar Alignment**: Align stocks with Indian trading holidays and forward-fills global commodity/forex market gaps so every stock row has exact matching market context.
- **Robust Ticker Filtering**: Excludes delisted or insufficient-data tickers gracefully without crashing the pipeline and logs them to `reports/excluded_tickers.csv`.

---

### Step 3: Feature Engineering (`src/m1_price_data/technical_features.py`)
Generates **over 160 technical indicators** grouped into key categories:
1. **Trend & Moving Averages**: SMA/EMA (5 to 200 days), distance metrics, golden/death cross ratios ($SMA_{50}/SMA_{200}$).
2. **Momentum Oscillators**: RSI (7, 14, 21, 28 days), MACD lines, signal lines & histograms, Stochastic %K/%D, Williams %R, Commodity Channel Index (CCI).
3. **Volatility & Bands**: Bollinger Bands (%B position, Bandwidth), Keltner Channels, Donchian Channels, ATR, Normalized ATR (NATR), Parkinson & Garman-Klass Volatility estimators.
4. **Volume & Money Flow**: On-Balance Volume (OBV), Chaikin Money Flow (CMF 20), Price-Volume Trend (PVT), Volume surge ratios.
5. **Directional Trend Strength**: ADX 14, +DI, -DI.

---

### Step 4: Zero Data Leakage Lags & Regime Detection (`lag_features.py` & `regime_detection.py`)
- **Anti-Leakage Shift Operations**: Computes historical return and volume lags ($t-1, t-2, t-5, t-10, t-21$) using explicit `.shift(1+)`.
  - *Why this is critical*: Guarantees that feature values at time $t$ **only** use information known at or before $t-1$, preventing any future data leakage into downstream machine learning models (LSTM / XGBoost).
- **Market Regime Classification**: Categorizes market dynamics into 3 distinct regimes:
  - `VOLATILE`: High India VIX (>18) or elevated ATR ratio.
  - `TRENDING`: High directional ADX strength (>25).
  - `CHOPPY`: Sideways, range-bound market with low trend strength.

---

### Step 5: Export & Automated Refresh (`build_features.py` & `refresh_data.py`)
- Exports the final processed dataset to `data/processed/price_features.parquet` (**44,029 rows**, **216 columns**).
- Produces automated quality metadata: `reports/data_quality_report.csv` and `reports/excluded_tickers.csv`.
- Allows single-command daily updates:
  ```bash
  python refresh_data.py
  ```

---

## 🎯 How to Present This to Your Teacher / Professor

### Key Highlights to Emphasize:
1. **End-to-End Automation**: *"It is not a manual collection of Jupyter notebooks. Running one command (`python refresh_data.py`) executes the entire pipeline end-to-end."*
2. **Production-Ready & Robust**: *"The pipeline handles missing dates, delisted stocks, and exchange holiday differences gracefully without failing."*
3. **Prevented Data Leakage**: *"We strictly enforced time-shift operations on historical lag features to prevent lookahead bias in model training."*
4. **Decoupled Architecture**: *"I documented the dataset schema and feature dictionary so my teammates (M2 Sentiment, M3 Modeling, M4 Dashboard) can load the data effortlessly in one line (`pd.read_parquet`)."*

---

## ❓ Anticipated Questions from Teachers & How to Answer

### Q1: *"Why did you use Parquet format instead of CSV?"*
> **Answer**: *"Parquet is a binary columnar storage format that compresses data heavily (saving up to 80% disk space), preserves native pandas data types (like `datetime64` and `float64`), and loads significantly faster during model training compared to CSV files."*

### Q2: *"How did you prevent data leakage in your lag features?"*
> **Answer**: *"Every lag feature (e.g. `return_1d_lag_1`) is constructed using an explicit `.shift(1)` operation per ticker. This ensures that feature values at index $t$ contain strictly historical data from timestamp $t-1$ or earlier, so the LSTM model never sees future price information."*

### Q3: *"What is the purpose of India VIX and USD/INR in stock price modeling?"*
> **Answer**: *"Individual stock prices are heavily influenced by broad market sentiment and macro factors. India VIX measures market fear and volatility, while USD/INR impacts export-heavy sectors like IT and broad capital flows. Including these gives our downstream models critical market-context features."*

### Q4: *"How do your teammates use your output?"*
> **Answer**: *"M2 (Sentiment) uses the closing price and trend features to calculate sentiment-price divergence. M3 (Modeling) loads `price_features.parquet` directly as input features for LSTM and XGBoost models. M4 (Dashboard) displays the price series and market regimes."*
