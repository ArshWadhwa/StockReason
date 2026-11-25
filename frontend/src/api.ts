import type {
  UniverseIndex,
  UniverseStock,
  MarketOverviewData,
  StockPriceData,
  SentimentData,
  PredictionData,
  SignalData,
  BacktestData,
  DailyDigestData,
  SystemStatus,
  ModelReport,
  ForecastRequest,
  ForecastResult,
  StockRiskProfile,
  ResearchBenchmarksData,
  FeatureRegistryData,
  ModelManifestData,
} from './types';

export const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch from ${url} (HTTP ${res.status})`);
  }
  return res.json();
}

async function postJson<T, B>(url: string, body: B): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export const api = {
  getHealth: () =>
    fetchJson<{ status: string; version?: string; ready_for_week4_swap?: boolean }>(`${API_BASE}/health`),

  getSystemStatus: () => fetchJson<SystemStatus>(`${API_BASE}/system/status`),

  getUniverses: () => fetchJson<{ indices: UniverseIndex[]; stocks: UniverseStock[] }>(`${API_BASE}/market/universes`),

  getMarketOverview: () => fetchJson<MarketOverviewData>(`${API_BASE}/market/overview`),

  getStockPrices: (ticker: string, days = 180) =>
    fetchJson<StockPriceData>(`${API_BASE}/stocks/${ticker}/prices?days=${days}`),

  getStockSentiment: (ticker: string) =>
    fetchJson<SentimentData>(`${API_BASE}/stocks/${ticker}/sentiment`),

  getStockPredictions: (ticker: string) =>
    fetchJson<PredictionData>(`${API_BASE}/stocks/${ticker}/predictions`),

  getStockSignal: (ticker: string) =>
    fetchJson<SignalData>(`${API_BASE}/stocks/${ticker}/signal`),

  getStockRisk: (ticker: string) =>
    fetchJson<StockRiskProfile>(`${API_BASE}/stocks/${ticker}/risk`),

  runForecast: (payload: ForecastRequest) =>
    postJson<ForecastResult, ForecastRequest>(`${API_BASE}/forecast`, payload),

  getStockForecast: (symbol: string, price: number, horizon = 21, modelType = 'baseline') =>
    fetchJson<ForecastResult>(
      `${API_BASE}/stocks/${symbol}/forecast?price=${price}&horizon=${horizon}&model_type=${modelType}`
    ),

  getRankedSignals: (includeAll = false) =>
    fetchJson<SignalData[]>(`${API_BASE}/signals/ranked${includeAll ? '?include_all=true' : ''}`),

  getBacktest: () => fetchJson<BacktestData>(`${API_BASE}/backtest`),

  getDailyDigest: () => fetchJson<DailyDigestData>(`${API_BASE}/digest/daily`),

  getModelReport: () => fetchJson<ModelReport>(`${API_BASE}/model/report`),

  getResearchBenchmarks: () => fetchJson<ResearchBenchmarksData>(`${API_BASE}/research/benchmarks`),

  getResearchFeatures: () => fetchJson<FeatureRegistryData>(`${API_BASE}/research/features`),

  getResearchManifest: () => fetchJson<ModelManifestData>(`${API_BASE}/research/manifest`),
};
