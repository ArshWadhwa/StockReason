import type {
  UniverseIndex, UniverseStock, MarketOverviewData,
  StockPriceData, SentimentData, PredictionData,
  SignalData, BacktestData, DailyDigestData, SystemStatus
} from './types';

const API_BASE = 'http://localhost:8000/api';

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch from ${url} (HTTP ${res.status})`);
  }
  return res.json();
}

export const api = {
  getHealth: () => fetchJson<{ status: string; ready_for_week4_swap: boolean }>(`${API_BASE}/health`),
  
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
  
  getRankedSignals: (includeAll = false) =>
    fetchJson<SignalData[]>(`${API_BASE}/signals/ranked${includeAll ? '?include_all=true' : ''}`),
  
  getBacktest: () =>
    fetchJson<BacktestData>(`${API_BASE}/backtest`),
  
  getDailyDigest: () =>
    fetchJson<DailyDigestData>(`${API_BASE}/digest/daily`),
};
