# Sentiment Data Schema Specification

This document publishes the official schema specifications for news headlines and aggregated sentiment dataset files output by the Module 2 (`m2_sentiment`) pipeline.

---

## 1. Raw & Scored News Headlines Schema

- **Storage Location**: `data/raw/news/*.parquet` or `data/raw/news/headlines.parquet`
- **Format**: Parquet (Snappy compressed) / SQLite
- **Granularity**: Article / Headline level with publication timestamps

| Column Name | Data Type | Nullable | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns, UTC]` | No | UTC timestamp when the article was published | `2024-01-15 09:30:00+00:00` |
| `date` | `date` / `string` | No | Market trading date alignment (YYYY-MM-DD) | `2024-01-15` |
| `ticker` | `string` | No | Associated stock/index ticker symbol | `RELIANCE.NS` |
| `title` | `string` | No | Cleaned news headline text | `Reliance Q3 profit rises 11% on retail boost` |
| `source` | `string` | Yes | News publisher / feed source | `Economic Times` |
| `url` | `string` | Yes | Source URL (optional) | `https://economictimes.indiatimes.com/...` |
| `score_finbert` | `float64` | No | Compound FinBERT score: $P(\text{pos}) - P(\text{neg}) \in [-1.0, 1.0]$ | `0.8421` |
| `score_textblob`| `float64` | No | Baseline TextBlob polarity score $\in [-1.0, 1.0]$ | `0.4500` |
| `sentiment_label`| `string` | No | Categorical label: `positive`, `neutral`, `negative` | `positive` |
| `confidence` | `float64` | No | Softmax confidence / probability of the chosen label | `0.9214` |

---

## 2. Daily Aggregated Sentiment Features Schema (`sentiment_features.parquet`)

- **Storage Location**: `data/processed/sentiment_features.parquet`
- **Format**: Parquet (Snappy compressed)
- **Granularity**: Daily (`1d`) per stock ticker

This table is consumed directly by **M3 (Modeling)** for LSTM feature merging and **M4 (Dashboard/API)** for sentiment cards and divergence badges.

| Column Name | Data Type | Nullable | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `datetime64[ns]` | No | Trading date (YYYY-MM-DD) | `2024-01-15` |
| `ticker` | `string` | No | Stock or index ticker symbol | `RELIANCE.NS` |
| `news_count` | `int64` | No | Total number of relevant headlines on this date | `5` |
| `sentiment_mean` | `float64` | No | Average daily FinBERT sentiment score $\in [-1.0, 1.0]$ | `0.6540` |
| `sentiment_pos_ratio` | `float64` | No | Fraction of positive headlines $[0.0, 1.0]$ | `0.8000` |
| `sentiment_neg_ratio` | `float64` | No | Fraction of negative headlines $[0.0, 1.0]$ | `0.0000` |
| `sentiment_momentum_3d` | `float64` | No | 3-day change in sentiment score ($S_t - S_{t-3}$) | `+0.2150` |
| `sentiment_momentum_7d` | `float64` | No | 7-day change in sentiment score ($S_t - S_{t-7}$) | `+0.1100` |
| `sentiment_divergence_flag` | `int64` | No | Divergence alert: `1` (Bullish Div), `-1` (Bearish Div), `0` (None) | `1` |
| `divergence_desc` | `string` | Yes | Human-readable explanation of price-sentiment divergence | `Bullish Divergence: Price -3.2% vs Sentiment +0.65` |
