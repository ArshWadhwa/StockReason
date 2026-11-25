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
  base_level?: number | null;
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
  url?: string | null;
}

export interface SentimentData {
  ticker: string;
  avg_sentiment_score: number;
  sentiment_label: 'bullish' | 'neutral' | 'bearish';
  article_count: number;
  bullish_ratio: number;
  bearish_ratio: number;
  sentiment_momentum_3d?: number;
  sentiment_momentum_7d?: number;
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
  prob_up?: number | null;
  prob_up_calibrated?: number | null;
  interval_80?: [number, number] | null;
  interval_coverage_observed?: number | null;
  feature_attribution?: Record<string, number>;
  forecast_origin_date?: string;
  target_trading_date?: string;
  calibration_action?: string;
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
  prob_up?: number | null;
  prob_up_calibrated?: number | null;
  interval_80?: [number, number] | null;
  interval_coverage_observed?: number | null;
  feature_attribution?: Record<string, number>;
  forecast_origin_date?: string;
  invalidation_conditions?: string[];
}

export interface ForecastRequest {
  symbol: string;
  current_price: number;
  reference_date?: string | null;
  horizon_days: number;
  model_type: 'baseline' | 'lightgbm' | 'catboost';
  features?: Record<string, number>;
}

export interface ForecastResult {
  symbol: string;
  target_horizon_days: number;
  target_trading_date: string;
  reference_close_price: number;
  predicted_price: number;
  predicted_return: number;
  model_name: string;
  model_status: string;
  model_version?: string;
  baseline_reference_mae_inr?: number;
  price_prediction_interval_50?: [number, number] | null;
  price_prediction_interval_80?: [number, number] | null;
  price_prediction_interval_90?: [number, number] | null;
  probability_positive_return?: number | null;
  warnings: string[];
}

export interface ConformalIntervalItem {
  coverage_label: string;
  nominal_coverage: number;
  empirical_coverage: number;
  coverage_error: number;
  return_lower: number;
  return_upper: number;
  price_lower: number;
  price_upper: number;
  winkler_score: number;
  miss_lower_tail: number;
  miss_upper_tail: number;
  calibrated_multiplier: number;
}

export interface StockRiskProfile {
  ticker: string;
  as_of_date: string;
  current_price: number;
  horizon_days: number;
  target_trading_date: string;
  volatility_latest: {
    trailing_60d_pct: number;
    ewma_94_pct: number;
    garch_11_pct: number;
    har_rv_pct: number;
  };
  qlike_loss: {
    trailing_60d: number | null;
    ewma_94: number | null;
    garch_11: number | null;
    har_rv: number | null;
  };
  garch_params: {
    omega: number;
    alpha: number;
    beta: number;
    persistence: number;
    converged: boolean;
  };
  har_rv_coefs: {
    daily: number;
    weekly: number;
    monthly: number;
  };
  volatility_series: Array<{
    date: string;
    ewma_vol_pct: number;
    trailing_60d_vol_pct: number;
    garch_vol_pct: number;
    har_rv_vol_pct: number;
  }>;
  conformal_intervals: ConformalIntervalItem[];
  probability_calibration: {
    brier_score_model: number | null;
    brier_score_reference: number | null;
    brier_skill_score: number;
    base_rate: number | null;
    is_calibrated: boolean;
    action: string;
  };
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
  version?: string;
  primary_horizon_days?: number;
  default_baseline?: string;
  baseline_mae_inr?: number;
  baseline_mape_pct?: number;
  ml_promotion_status?: string;
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

export interface BenchmarkModelEntry {
  model_name: string;
  category: string;
  algorithm: string;
  target_type: string;
  num_features: number;
  train_duration_sec: number;
  rel_error_red_vs_no_change_pct: number;
  rel_error_red_vs_strongest_base_pct: number;
  hurdle_assessment: string;
  cal_metrics: Record<string, number>;
  diag_metrics: Record<string, number>;
}

export interface ResearchBenchmarksData {
  release: string;
  horizon: string;
  evaluation_cohort_obs: number;
  no_change_reference: {
    true_price_mape: number;
    inr_price_mae: number;
    eval_observations: number;
  };
  strongest_statistical_baseline: {
    id: string;
    name: string;
    cal_mape: number;
    diag_mape: number;
    diag_mae: number;
  };
  f8_development_selection: {
    selected_name: string;
    selected_features_count: number;
    selected_features: string[];
    cal_mape: number;
    diag_mape: number;
  };
  best_development_model: string;
  model_results: BenchmarkModelEntry[];
  regime_results: Record<string, {
    n_obs: number;
    pct_of_cohort: number;
    no_change_mape: number;
    best_model_mape: number;
    relative_error_reduction_pct: number;
    best_model_rank_ic: number;
    best_model_dir_acc: number;
  }>;
}

export interface FeatureRegistryData {
  registry_version: string;
  release: string;
  total_unique_features: number;
  populated_features_count: number;
  quarantined_features_count: number;
  families: Array<{
    family: string;
    description: string;
    status: string;
    total_count: number;
    populated_count: number;
    features: string[];
  }>;
}

export interface ModelManifestData {
  manifest: {
    manifest_version: string;
    repository: string;
    evaluation_horizon: string;
    training_cutoff: string;
    calibration_period: string;
    diagnostic_period: string;
    training_dataset_sha256: string;
    models: Array<{
      model_id: string;
      algorithm: string;
      horizon: string;
      target: string;
      artifact_file: string | null;
      sha256?: string;
      feature_count?: number;
      status: string;
      mape: number;
      mae_inr: number;
      directional_accuracy: number;
      rank_ic: number;
      limitations: string;
    }>;
  };
  config: Record<string, any>;
  figures: Array<{
    filename: string;
    title: string;
    url: string;
  }>;
}
