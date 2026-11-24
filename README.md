# StockReason `v0.7.0-dev`
> **Trustworthy Forecasting and Evaluation Engine for NSE Equities**

StockReason is a production-grade machine learning and forecasting framework for National Stock Exchange (NSE) Indian equities. It provides leak-free time-series modeling, purged walk-forward cross-validation, honest post-market cost-aware backtesting, conformal prediction intervals, and transparent feature attribution.

---

## What's New in v0.7: The Trustworthy Engine

StockReason v0.7 eliminates five foundational architectural vulnerabilities present in conventional financial ML pipelines:
1. **Forecast-Origin Alignment**: Ensures training sequences $[t-L+1 \dots t]$ use information available strictly at market close of day $t$ to forecast $\text{Close}_{t+h}/\text{Close}_t - 1$, fixing legacy 1-day timing misalignment.
2. **Purged Walk-Forward CV**: Uses calendar date alignment, label-end forward purging, and post-evaluation embargoes to guarantee zero lookahead leakage across multi-horizon labels.
3. **Honest Post-Market Backtesting**: Simulates execution strictly at the **open of day $t+1$** and incorporates a full statutory Indian equity delivery cost model (STT 0.1%, stamp duty 0.015%, NSE turnover fee, SEBI fee, 18% GST, liquidity-scaled slippage).
4. **Calibrated Uncertainty & Conformal Intervals**: Replaces misleading MC-dropout intervals with Conformal Prediction Intervals (CQR/ACI) achieving ~80.2% empirical coverage on nominal 80%, alongside calibrated directional probabilities.
5. **Stationary Feature Sets**: Replaces non-stationary price/volume levels with volatility-scaled excess returns and stationary normalized technical indicators (improving Information Ratio from 0.68 to 0.99).

---

## Installation & Environment Setup

The repository uses Python 3.12 and [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency resolution:

```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt lightgbm catboost scipy statsmodels
```

---

## Running Tests

All 135 unit and integration tests run deterministically:

```bash
# Set OMP_NUM_THREADS=1 on macOS arm64 to prevent multi-threaded OpenMP contention
OMP_NUM_THREADS=1 pytest -q
```

---

## v0.7 CLI & Execution Commands

### 1. Unified Trustworthy Refresh Pipeline
Runs the complete 7-stage pipeline: Freshness check $\to$ Data update $\to$ Feature build $\to$ Model inference $\to$ Calibration $\to$ Signals $\to$ Archive:
```bash
python refresh_data.py --v07
```

### 2. Market Data Freshness Gate
Checks price and sentiment data freshness against the official NSE holiday calendar:
```bash
python -m v07.common.freshness --strict
```

### 3. Purged Walk-Forward Cross-Validation Demo
Demonstrates fold date boundaries, forward-label purge counts, and automated zero-leakage assertions:
```bash
python -m v07.validation.demo
```

### 4. 9-Model Comparison Harness
Trains and evaluates Zero, Market, Ridge, ElasticNet, LightGBM, CatBoost, LSTM, GRU, and DLinear models with Diebold-Mariano significance testing:
```bash
python -m v07.models.run_comparison --config config/v07.yaml
```

### 5. Feature Ablation Benchmark
Trains gradient boosted trees across Feature Sets A through E (Stationary technicals vs Market context vs NLP vs Legacy control):
```bash
python -m v07.features.run_ablation
```

### 6. Universe Scaling Experiment
Evaluates performance across tiered universes (U16 to U300) and history lengths (5 years vs 10 years):
```bash
python -m v07.data_expansion.run_scaling_experiment
```

### 7. Model Evaluation Report Generator
Produces a reproducible evaluation report with bootstrap confidence intervals and edge verdict:
```bash
python -m v07.reporting.make_report --model catboost
```

### 8. Statistical Model Promotion Gate
Verifies whether a candidate model qualifies to replace the incumbent based on out-of-sample Rank IC, backtest Sharpe, and calibration error:
```bash
python -m v07.reporting.promote_model --candidate catboost
```

---

## Launching the Application

### Backend API (FastAPI)
```bash
cd src/m4_dashboard/backend
uvicorn main:app --reload --port 8000
```
- Interactive API documentation available at `http://localhost:8000/docs`
- Quantitative model report endpoint: `GET /api/model/report`
- System status and freshness endpoint: `GET /api/system/status`

### Frontend Dashboard (React + TypeScript + Vite)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` to explore the interactive dashboard featuring the new **Model Reliability Panel**, **Calibrated 80% Intervals**, and **Falsifiable Invalidation Conditions**.

---

## Repository Architecture

```
StockReason/
├── config/
│   ├── v07.yaml                # Master v0.7 parameters & transaction cost schedule
│   ├── v07_features.yaml       # Stationary feature set definitions
│   └── settings.yaml           # Global system paths
├── src/
│   └── v07/
│       ├── common/             # Freshness gate, sequence builder, validation
│       ├── validation/         # Purged walk-forward CV, embargo, CPCV, leakage checks
│       ├── backtest/           # Cost-aware backtester, Indian statutory fees, fill engine
│       ├── models/             # BaseForecaster, tree/linear/deep models, DM tests
│       ├── features/           # Volatility-scaled excess targets, feature sets A-E
│       ├── data_expansion/     # U16-U300 universes, yfinance caching downloader, cleaner
│       ├── news/               # Point-in-time event store, 15:30 IST rollover, deduplication
│       ├── uncertainty/        # Isotonic/Platt calibration, conformal intervals (CQR/ACI)
│       ├── signals/            # Signal Engine v2, portfolio rules, brackets, explanations
│       └── reporting/          # Metrics, make_report, promote_model, drift monitor
├── tests/
│   └── v07/                    # Comprehensive v0.7 pytest test suite (124 tests)
├── reports/
│   └── v0.7/                   # Local-only research and evaluation reports
└── frontend/                   # React dashboard with Model Reliability Panel
```

---

## License
MIT License
