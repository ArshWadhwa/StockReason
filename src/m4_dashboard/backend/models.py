from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0"
    track: str = "M4 - Product & Dashboard"
    data_mode: str = "real"
    ready_for_week4_swap: bool = True

class TickerItem(BaseModel):
    ticker: str
    name: str
    sector: str
    universe: str
    current_price: Optional[float] = None
    day_change_pct: Optional[float] = None

class IndexItem(BaseModel):
    symbol: str
    name: str
    category: str
    base_level: Optional[float] = None

class UniverseResponse(BaseModel):
    indices: List[Dict[str, Any]]
    stocks: List[Dict[str, Any]]

class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_line_12_26: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    regime: Optional[str] = None

class PriceHistoryResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    universe: str
    current_price: float
    day_change_pct: float
    quote_status: Optional[str] = None
    quote_updated_at: Optional[str] = None
    history: List[PricePoint]

class HeadlineItem(BaseModel):
    id: str
    headline: str
    source: str
    published_at: str
    sentiment_score: float
    sentiment_label: str
    confidence: float

class DivergenceAlert(BaseModel):
    has_divergence: bool
    message: str

class SentimentResponse(BaseModel):
    ticker: str
    avg_sentiment_score: float
    sentiment_label: str
    article_count: int
    bullish_ratio: float
    bearish_ratio: float
    divergence_alert: DivergenceAlert
    headlines: List[HeadlineItem]

class HorizonForecast(BaseModel):
    expected_return: float
    predicted_price: float
    confidence: float
    uncertainty_std: float
    lower_bound: float
    upper_bound: float

class EnsembleComparison(BaseModel):
    lstm_return_1M: float
    secondary_return_1M: Optional[float] = None
    disagreement_detected: bool = False
    disagreement_delta: float = 0.0

class ShapFeature(BaseModel):
    feature: str
    feature_display: Optional[str] = None
    feature_description: Optional[str] = None
    impact: float
    importance_score: float
    direction: str
    current_value: Optional[float] = None

class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    current_price: float
    horizons: Dict[str, HorizonForecast]
    ensemble: EnsembleComparison
    shap_explainability: List[ShapFeature]
    # v0.7 trustworthy engine extensions (WS8/WS9/WS10)
    prob_up: Optional[float] = None
    prob_up_calibrated: Optional[float] = None
    interval_80: Optional[List[float]] = None
    interval_coverage_observed: Optional[float] = None
    feature_attribution: Optional[Dict[str, float]] = None
    forecast_origin_date: Optional[str] = None
    invalidation_conditions: Optional[List[str]] = None

class EntryExitBracket(BaseModel):
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    atr_14: float
    bracket_type: str

class SignalResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    universe: str
    current_price: float
    signal: str
    signal_score: float
    confidence: float
    expected_return_1M: float
    sentiment_score: float
    regime: str
    ensemble_agreement: bool
    reasoning: List[str]
    entry_exit: Optional[EntryExitBracket] = None
    # v0.7 trustworthy engine extensions (WS8/WS9/WS10)
    prob_up: Optional[float] = None
    prob_up_calibrated: Optional[float] = None
    interval_80: Optional[List[float]] = None
    interval_coverage_observed: Optional[float] = None
    feature_attribution: Optional[Dict[str, float]] = None
    forecast_origin_date: Optional[str] = None
    invalidation_conditions: Optional[List[str]] = None

class BacktestMetrics(BaseModel):
    cumulative_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

class BacktestResponse(BaseModel):
    strategy_metrics: BacktestMetrics
    benchmark_metrics: Dict[str, Any]
    random_baseline_metrics: Dict[str, Any]
    confidence_calibration: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]

class DailyDigest(BaseModel):
    digest_date: str
    market_summary: str
    signal_upgrades: List[Dict[str, Any]]
    signal_downgrades: List[Dict[str, Any]]
    top_opportunities: List[str]
    high_risk_alerts: List[str]

class SystemStatusResponse(BaseModel):
    status: str = "operational"
    data_as_of: str
    model_version: str = "0.7.0"
    rolling_oos_rank_ic: Optional[float] = None
    edge_verdict: str = "No demonstrated edge"
    is_stale: bool = False
    freshness_message: str = "Data is up to date"
    last_refreshed: str
    tickers_loaded: int = 0
    predictions_loaded: int = 0
    sentiment_loaded: int = 0
    signals_generated: int = 0
    data_source: str = "real"

class ModelReportResponse(BaseModel):
    model_name: str
    timestamp: str
    as_of_date: str
    verdict: str
    has_edge: bool
    metrics: Dict[str, Any]
    calibration: Dict[str, Any]
    backtest: Dict[str, Any]
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    delta_rank_ic: Optional[float] = None
    baseline_metrics: Optional[Dict[str, Any]] = None
    universe: Optional[str] = None
    feature_count: Optional[int] = None
