# Raw Data Schema Specification

This document publishes the official schema specifications for raw historical price and index dataset files output by the Module 1 (`m1_price_data`) fetcher.

---

## 1. Stock OHLCV Dataset Schema

- **Storage Location**: `data/raw/ohlcv/*.parquet` and `data/raw/ohlcv/pilot_stocks.parquet`
- **Format**: Parquet (Snappy compressed)
- **Timeframe**: Daily (`1d`)

| Column Name | Data Type | Nullable | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `datetime64[ns]` | No | Trading date (YYYY-MM-DD) | `2024-01-15` |
| `ticker` | `string` | No | Yahoo Finance ticker symbol | `RELIANCE.NS` |
| `open` | `float64` | No | Opening price in INR | `2740.50` |
| `high` | `float64` | No | Day high price in INR | `2765.00` |
| `low` | `float64` | No | Day low price in INR | `2731.10` |
| `close` | `float64` | No | Closing price in INR | `2755.80` |
| `adj_close` | `float64` | No | Adjusted closing price (dividends/splits) | `2755.80` |
| `volume` | `int64` | No | Trading volume (number of shares) | `4523100` |

---

## 2. Index & Volatility Levels Schema

- **Storage Location**:
  - `data/raw/indices/nifty_50.parquet` (Index ticker: `^NSEI`)
  - `data/raw/indices/nifty_bank.parquet` (Index ticker: `^NSEBANK`)
  - `data/raw/indices/india_vix.parquet` (Index ticker: `^INDIAVIX`)
- **Format**: Parquet (Snappy compressed)
- **Timeframe**: Daily (`1d`)

| Column Name | Data Type | Nullable | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `datetime64[ns]` | No | Trading date (YYYY-MM-DD) | `2024-01-15` |
| `symbol` | `string` | No | Yahoo Finance index symbol | `^NSEI` |
| `open` | `float64` | No | Opening index level / VIX level | `22000.10` |
| `high` | `float64` | No | Day high level | `22150.40` |
| `low` | `float64` | No | Day low level | `21950.00` |
| `close` | `float64` | No | Closing index level / VIX level | `22097.45` |
| `adj_close` | `float64` | No | Adjusted closing level | `22097.45` |
| `volume` | `int64` | Yes | Trading volume (0 for VIX/indices if unrecorded) | `0` |
