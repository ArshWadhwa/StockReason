export interface UniverseStock {
  ticker: string;
  name: string;
  sector: string;
  universe: string;
  current_price: number;
  day_change_pct: number;
}

export interface UniverseIndex {
  symbol: string;
  name: string;
  category: string;
  base_level?: number;
}

export interface MarketOverviewData {
  indices: Array<{
    symbol: string;
    name: string;
    level: number;
    change_pct: number;
    status: string;
  }>;
  macro: Array<{
    name: string;
    value: number;
    change_pct: number;
  }>;
  market_regime: string;
  active_coverage_count: number;
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma_20: number;
  sma_50: number;
  rsi_14: number;
  macd_line_12_26: number;
  macd_signal: number;
  bb_upper: number;
  bb_lower: number;
  regime: string;
}

export interface StockPriceData {
  ticker: string;
  name: string;
  sector: string;
  universe: string;
  current_price: number;
  day_change_pct: number;
  quote_status?: 'LIVE' | 'LAST_CLOSE' | 'UNAVAILABLE' | string;
  quote_updated_at?: string;
  history: PricePoint[];
}

export interface HeadlineItem {
  id: string;
  headline: string;
  source: string;
  published_at: string;
  sentiment_score: number;
  sentiment_label: 'positive' | 'neutral' | 'negative';
  confidence: number;
}

export interface SentimentData {
  ticker: string;
  avg_sentiment_score: number;
  sentiment_label: 'bullish' | 'neutral' | 'bearish';
  article_count: number;
  bullish_ratio: number;
  bearish_ratio: number;
  divergence_alert: {
    has_divergence: boolean;
    message: string;
  };
  headlines: HeadlineItem[];
}

export interface HorizonForecast {
  expected_return: number;
  predicted_price: number;
  confidence: number;
  uncertainty_std: number;
  lower_bound: number;
  upper_bound: number;
}

export interface PredictionData {
  ticker: string;
  prediction_date: string;
  current_price: number;
  horizons: {
    '1D': HorizonForecast;
    '1W': HorizonForecast;
    '1M': HorizonForecast;
    '6M': HorizonForecast;
  };
  ensemble?: {
    lstm_return_1M: number;
    disagreement_detected: boolean;
    disagreement_delta: number;
  };
  shap_explainability: Array<{
    feature: string;
    feature_display?: string;
    feature_description?: string;
    impact: number;
    importance_score: number;
    direction: 'positive' | 'negative';
    current_value?: number;
  }>;
  // v0.7 trustworthy engine fields
  prob_up?: number;
  prob_up_calibrated?: number;
  interval_80?: [number, number];
  interval_coverage_observed?: number;
  feature_attribution?: Record<string, number>;
  forecast_origin_date?: string;
  invalidation_conditions?: string[];
}

export interface SignalData {
  ticker: string;
  name: string;
  sector: string;
  universe: string;
  current_price: number;
  signal: 'BUY' | 'HOLD' | 'AVOID';
  signal_score: number;
  confidence: number;
  expected_return_1M: number;
  sentiment_score: number;
  regime: string;
  ensemble_agreement: boolean;
  reasoning: string[];
  // v0.7 trustworthy engine fields
  prob_up?: number;
  prob_up_calibrated?: number;
  interval_80?: [number, number];
  interval_coverage_observed?: number;
  feature_attribution?: Record<string, number>;
  forecast_origin_date?: string;
  invalidation_conditions?: string[];
}

export interface BacktestData {
  strategy_metrics: {
    cumulative_return: number;
    annualized_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
  };
  benchmark_metrics: {
    name: string;
    cumulative_return: number;
    annualized_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
  };
  random_baseline_metrics: {
    name: string;
    cumulative_return: number;
    annualized_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
  };
  confidence_calibration: Array<{
    confidence_bin: string;
    predicted_prob: number;
    actual_accuracy: number;
    sample_count: number;
  }>;
  equity_curve: Array<{
    date: string;
    strategy: number;
    benchmark: number;
    random_baseline: number;
  }>;
}

export interface DailyDigestData {
  digest_date: string;
  market_summary: string;
  signal_upgrades: Array<{
    ticker: string;
    previous_signal: string;
    new_signal: string;
    primary_catalyst: string;
  }>;
  signal_downgrades: Array<{
    ticker: string;
    previous_signal: string;
    new_signal: string;
    primary_catalyst: string;
  }>;
  top_opportunities: string[];
  high_risk_alerts: string[];
}

export interface SystemStatus {
  status?: string;
  data_as_of?: string;
  model_version?: string;
  rolling_oos_rank_ic?: number | null;
  edge_verdict?: string;
  is_stale?: boolean;
  freshness_message?: string;
  last_refreshed: string;
  tickers_loaded: number;
  predictions_loaded: number;
  sentiment_loaded: number;
  signals_generated: number;
  data_source: string;
}

export interface ModelReport {
  model_name: string;
  timestamp: string;
  as_of_date: string;
  verdict: string;
  has_edge: boolean;
  metrics: {
    ic: number;
    rank_ic: number;
    icir: number;
    rank_icir: number;
    mae: number;
    rmse: number;
    dir_acc: number;
    dir_acc_vs_base: number;
    brier?: number;
    n: number;
    n_dates: number;
    rank_ic_ci_95?: [number, number];
    by_regime?: Record<string, any>;
    by_sector?: Record<string, any>;
  };
  calibration: {
    ece: number;
    brier: number;
    bins: any[];
  };
  backtest: {
    sharpe: number;
    max_drawdown: number;
    annualized_return: number;
    annualized_turnover: number;
    transaction_cost_bps?: number;
  };
  git_commit?: string;
  config_hash?: string;
  delta_rank_ic?: number;
  baseline_name?: string;
  universe?: string;
  feature_count?: number;
}
