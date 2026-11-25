import { useState, useEffect, useMemo } from 'react';
import { api } from './api';
import type {
  UniverseStock,
  MarketOverviewData,
  StockPriceData,
  SentimentData,
  PredictionData,
  SignalData,
  StockRiskProfile,
  BacktestData,
  DailyDigestData,
  SystemStatus,
} from './types';

import { LandingHome } from './components/LandingHome';
import { PriceChartView } from './components/PriceChartView';
import { SignalsView } from './components/SignalsView';
import { RiskVolatilityView } from './components/RiskVolatilityView';
import { SentimentView } from './components/SentimentView';
import { BacktestView } from './components/BacktestView';
import { ModelBenchmarksView } from './components/ModelBenchmarksView';
import { DigestView } from './components/DigestView';
import { BrandMark } from './components/BrandMark';

import {
  LineChart,
  Zap,
  ShieldCheck,
  Newspaper,
  BarChart3,
  Award,
  FileText,
  ChevronRight,
  Search,
} from 'lucide-react';

import Lenis from 'lenis';
import 'lenis/dist/lenis.css';

type PageMode = 'home' | 'dashboard';
type DashboardTab = 'chart' | 'signals' | 'risk' | 'sentiment' | 'backtest' | 'benchmarks' | 'digest';

const VALID_TABS: DashboardTab[] = ['chart', 'signals', 'risk', 'sentiment', 'backtest', 'benchmarks', 'digest'];

function getInitialRoute(): PageMode {
  if (typeof window !== 'undefined' && window.location.pathname.startsWith('/dashboard')) {
    return 'dashboard';
  }
  return 'home';
}

function getInitialTicker(): string {
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search);
    const ticker = params.get('ticker');
    if (ticker) return ticker;
  }
  return 'RELIANCE.NS';
}

function getInitialTab(): DashboardTab {
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search);
    const tab = params.get('tab') as DashboardTab | null;
    if (tab && VALID_TABS.includes(tab)) return tab;
  }
  return 'chart';
}

export function App() {
  const [pageMode, setPageMode] = useState<PageMode>(getInitialRoute);
  const [activeTab, setActiveTab] = useState<DashboardTab>(getInitialTab);
  const [selectedTicker, setSelectedTicker] = useState<string>(getInitialTicker);
  const [tickerFilter, setTickerFilter] = useState<string>('');

  // App States
  const [loading, setLoading] = useState<boolean>(true);
  const [stockLoading, setStockLoading] = useState<boolean>(false);
  const [switchingTicker, setSwitchingTicker] = useState<string | null>(null); // tracks in-flight ticker switch
  const [error, setError] = useState<string | null>(null);

  // Data Stores
  const [stocks, setStocks] = useState<UniverseStock[]>([]);
  const [overview, setOverview] = useState<MarketOverviewData | null>(null);
  const [rankedSignals, setRankedSignals] = useState<SignalData[]>([]);
  const [priceData, setPriceData] = useState<StockPriceData | null>(null);
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [signalData, setSignalData] = useState<SignalData | null>(null);
  const [riskData, setRiskData] = useState<StockRiskProfile | null>(null);
  const [backtestData, setBacktestData] = useState<BacktestData | null>(null);
  const [digestData, setDigestData] = useState<DailyDigestData | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    const handlePopState = () => {
      const isDashboard = window.location.pathname.startsWith('/dashboard');
      setPageMode(isDashboard ? 'dashboard' : 'home');

      const params = new URLSearchParams(window.location.search);
      const tickerParam = params.get('ticker');
      if (tickerParam) setSelectedTicker(tickerParam);

      const tabParam = params.get('tab') as DashboardTab | null;
      if (tabParam && VALID_TABS.includes(tabParam)) setActiveTab(tabParam);
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Initial Data Load
  useEffect(() => {
    async function loadInitialData() {
      try {
        setLoading(true);
        setError(null);

        const [univRes, overRes, sigsRes, btRes, digRes, statusRes] = await Promise.all([
          api.getUniverses(),
          api.getMarketOverview(),
          api.getRankedSignals(true),
          api.getBacktest(),
          api.getDailyDigest(),
          api.getSystemStatus(),
        ]);

        setStocks(univRes.stocks);
        setOverview(overRes);
        setRankedSignals(sigsRes);
        setBacktestData(btRes);
        setDigestData(digRes);
        setSystemStatus(statusRes);

        if (univRes.stocks.length > 0) {
          setSelectedTicker((prev) => {
            const params = new URLSearchParams(window.location.search);
            const urlTicker = params.get('ticker');
            if (urlTicker && univRes.stocks.some((s) => s.ticker === urlTicker)) {
              return urlTicker;
            }
            if (prev && univRes.stocks.some((s) => s.ticker === prev)) {
              return prev;
            }
            return univRes.stocks[0].ticker;
          });
        }
      } catch (err: any) {
        console.error('Failed to load initial market data:', err);
        setError('Could not connect to FastAPI local backend at http://localhost:8000.');
      } finally {
        setLoading(false);
      }
    }
    loadInitialData();
  }, []);

  // Fetch Stock-Specific Details — seamless: keep old data visible during switch
  useEffect(() => {
    if (!selectedTicker) return;

    let cancelled = false;

    async function loadStockDetails() {
      // For first load: show skeleton. For ticker switches: keep old data visible.
      const isFirstLoad = !priceData;
      if (isFirstLoad) setStockLoading(true);
      else setSwitchingTicker(selectedTicker);

      setError(null);

      try {
        const [pData, sData, prData, sigData, rkData] = await Promise.all([
          api.getStockPrices(selectedTicker),
          api.getStockSentiment(selectedTicker),
          api.getStockPredictions(selectedTicker),
          api.getStockSignal(selectedTicker),
          api.getStockRisk(selectedTicker),
        ]);
        if (cancelled) return;
        // Atomic swap — no intermediate empty state
        setPriceData(pData);
        setSentimentData(sData);
        setPredictionData(prData);
        setSignalData(sigData);
        setRiskData(rkData);
      } catch (err: any) {
        if (cancelled) return;
        console.error(`Failed to fetch details for ${selectedTicker}:`, err);
        setError(`Failed to load data for ${selectedTicker}`);
      } finally {
        if (!cancelled) {
          setStockLoading(false);
          setSwitchingTicker(null);
        }
      }
    }

    loadStockDetails();
    const refreshTimer = window.setInterval(() => { loadStockDetails(); }, 60_000);
    return () => {
      cancelled = true;
      window.clearInterval(refreshTimer);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTicker]);

  // Lenis Smooth Scroll (Landing Mode)
  useEffect(() => {
    if (pageMode !== 'home') return;

    const lenis = new Lenis({
      duration: 1.25,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: 'vertical',
      gestureOrientation: 'vertical',
      smoothWheel: true,
      wheelMultiplier: 0.9,
      touchMultiplier: 1.5,
    });

    let rafId: number;
    function raf(time: number) {
      lenis.raf(time);
      rafId = requestAnimationFrame(raf);
    }
    rafId = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(rafId);
      lenis.destroy();
    };
  }, [pageMode]);

  const filteredStocks = useMemo(() => {
    const q = tickerFilter.trim().toLowerCase();
    if (!q) return stocks;
    return stocks.filter(
      (s) =>
        s.ticker.toLowerCase().includes(q) ||
        s.name.toLowerCase().includes(q) ||
        s.sector.toLowerCase().includes(q)
    );
  }, [stocks, tickerFilter]);

  const handleOpenDashboard = (ticker?: string, tab?: DashboardTab) => {
    const targetTicker = ticker || selectedTicker;
    const targetTab = tab || 'chart';

    if (ticker) setSelectedTicker(ticker);
    setActiveTab(targetTab);
    setPageMode('dashboard');

    const params = new URLSearchParams();
    if (targetTicker) params.set('ticker', targetTicker);
    if (targetTab && targetTab !== 'chart') params.set('tab', targetTab);
    const queryString = params.toString() ? `?${params.toString()}` : '';
    const targetUrl = `/dashboard${queryString}`;

    if (window.location.pathname + window.location.search !== targetUrl) {
      window.history.pushState(null, '', targetUrl);
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleGoHome = () => {
    setPageMode('home');
    if (window.location.pathname !== '/') {
      window.history.pushState(null, '', '/');
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleTickerChange = (ticker: string) => {
    setSelectedTicker(ticker);
    if (pageMode === 'dashboard') {
      const params = new URLSearchParams(window.location.search);
      params.set('ticker', ticker);
      const queryString = params.toString() ? `?${params.toString()}` : '';
      window.history.replaceState(null, '', `/dashboard${queryString}`);
    }
  };

  const handleTabChange = (tab: DashboardTab) => {
    setActiveTab(tab);
    if (pageMode === 'dashboard') {
      const params = new URLSearchParams(window.location.search);
      if (tab === 'chart') {
        params.delete('tab');
      } else {
        params.set('tab', tab);
      }
      const queryString = params.toString() ? `?${params.toString()}` : '';
      window.history.replaceState(null, '', `/dashboard${queryString}`);
    }
  };

  /* ═══════════════════ LANDING HOME MODE ═══════════════════ */
  if (pageMode === 'home') {
    return (
      <div style={{ background: 'var(--bg)', minHeight: '100vh' }}>
        <header className="shell">
          <nav className="landing-nav">
            <a
              href="/"
              className="landing-nav-brand"
              onClick={(e) => {
                e.preventDefault();
                handleGoHome();
              }}
              style={{ textDecoration: 'none', color: 'inherit', cursor: 'pointer' }}
            >
              <div className="brand-icon">
                <BrandMark />
              </div>
              StockReason
            </a>

            <a
              href="/dashboard"
              className="landing-nav-login"
              onClick={(e) => {
                e.preventDefault();
                handleOpenDashboard();
              }}
              style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center' }}
            >
              Open Terminal →
            </a>
          </nav>
        </header>

        <LandingHome
          overview={overview}
          signals={rankedSignals}
          onOpenDashboard={handleOpenDashboard}
        />
      </div>
    );
  }

  /* ═══════════════════ DASHBOARD MODE ═══════════════════ */
  const selectedStock = stocks.find((s) => s.ticker === selectedTicker);

  const activeSignal =
    signalData && signalData.ticker === selectedTicker
      ? signalData.signal
      : rankedSignals.find((s) => s.ticker === selectedTicker)?.signal;

  const getSuggestion = (sig?: string) => {
    if (sig === 'BUY') {
      return { label: 'BUY', color: '#16a34a', bg: '#e8f5e9', border: '#c8e6c9', icon: '▲' };
    }
    if (sig === 'HOLD') {
      return { label: 'HOLD', color: '#b45309', bg: '#fefce8', border: '#fde68a', icon: '◆' };
    }
    if (sig === 'AVOID' || sig === 'SELL') {
      return { label: 'AVOID', color: '#dc2626', bg: '#fef2f2', border: '#fecaca', icon: '▼' };
    }
    return { label: 'HOLD', color: '#b45309', bg: '#fefce8', border: '#fde68a', icon: '◆' };
  };

  const suggestion = getSuggestion(activeSignal);

  return (
    <div className="dashboard-mode">
      <div className="landing-bg-decorations" aria-hidden="true">
        <svg className="bg-svg-pattern" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="stock-grid-dash" width="80" height="80" patternUnits="userSpaceOnUse">
              <path d="M 80 0 L 0 0 0 80" fill="none" stroke="rgba(28, 35, 49, 0.03)" strokeWidth="1" />
              <circle cx="80" cy="80" r="1.5" fill="rgba(28, 35, 49, 0.05)" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#stock-grid-dash)" />
        </svg>
      </div>

      {/* Navbar */}
      <header className="shell" style={{ position: 'relative', zIndex: 10 }}>
        <nav className="landing-nav">
          <a
            href="/"
            className="landing-nav-brand"
            onClick={(e) => {
              e.preventDefault();
              handleGoHome();
            }}
            style={{ textDecoration: 'none', color: 'inherit', cursor: 'pointer' }}
          >
            <div className="brand-icon">
              <BrandMark />
            </div>
            StockReason
          </a>

          {selectedStock && (
            <div className="dash-nav-stock-badge">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="dash-nav-stock-name" title={selectedStock.name}>
                  {selectedStock.name}
                </span>
                <span className="dash-nav-stock-ticker">
                  {selectedTicker.replace('.NS', '')}
                </span>
              </div>

              <div style={{ width: 1, height: 16, background: '#e2e8f0' }} />

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '11px', color: '#5A6478', fontFamily: 'var(--sans)', fontWeight: '500' }}>
                  Signal
                </span>
                <span
                  className="dash-nav-suggestion"
                  style={{
                    color: suggestion.color,
                    background: suggestion.bg,
                    border: `1px solid ${suggestion.border}`,
                  }}
                >
                  <span style={{ fontSize: '9px' }}>{suggestion.icon}</span>
                  {suggestion.label}
                </span>
              </div>
            </div>
          )}
        </nav>
      </header>

      <div className="shell" style={{ position: 'relative', zIndex: 10, paddingBottom: 60 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '270px 1fr', gap: '24px', alignItems: 'start' }}>
          {/* Sidebar Controls */}
          <aside className="dash-sidebar">
            <div>
              <div style={{ fontFamily: 'var(--sans)', fontSize: '11px', fontWeight: '600', color: 'var(--dash-muted)', marginBottom: '8px' }}>
                Stock Selector ({stocks.length} NSE Pilot)
              </div>

              <div style={{ position: 'relative', marginBottom: '8px' }}>
                <Search size={13} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                <input
                  type="text"
                  value={tickerFilter}
                  onChange={(e) => setTickerFilter(e.target.value)}
                  placeholder="Filter symbol or sector..."
                  style={{
                    width: '100%',
                    padding: '7px 10px 7px 28px',
                    borderRadius: '8px',
                    border: '1px solid var(--rule)',
                    fontSize: '12px',
                    fontFamily: 'var(--sans)',
                    color: 'var(--ink)',
                    background: 'var(--paper)',
                  }}
                />
              </div>

              <select
                className="dash-select"
                value={selectedTicker}
                onChange={(e) => handleTickerChange(e.target.value)}
              >
                {(filteredStocks.length > 0 ? filteredStocks : stocks).map((s) => (
                  <option key={s.ticker} value={s.ticker}>
                    {s.name} ({s.ticker.replace('.NS', '')})
                  </option>
                ))}
              </select>

              {selectedStock && (
                <div style={{
                  marginTop: '10px',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  background: 'var(--paper)',
                  border: '1px solid var(--rule)',
                  fontSize: '11px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <span style={{ fontFamily: 'var(--mono)', fontWeight: '700', fontSize: '14px', color: 'var(--ink)' }}>
                      ₹{selectedStock.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </span>
                    <span style={{
                      fontFamily: 'var(--mono)',
                      fontWeight: '700',
                      color: selectedStock.day_change_pct >= 0 ? 'var(--gain)' : 'var(--loss)',
                    }}>
                      {selectedStock.day_change_pct >= 0 ? '+' : ''}{selectedStock.day_change_pct.toFixed(2)}%
                    </span>
                  </div>
                  <div style={{ color: 'var(--dash-muted)', marginTop: '4px' }}>
                    {selectedStock.sector} · {selectedStock.universe}
                  </div>
                </div>
              )}
            </div>

            <div style={{ borderTop: '1px solid #edf2f7', paddingTop: '16px' }}>
              <div style={{ fontFamily: 'var(--sans)', fontSize: '11px', fontWeight: '600', color: 'var(--dash-muted)', marginBottom: '10px' }}>
                Analysis views
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {[
                  { id: 'chart', label: 'Price Action & Chart', icon: LineChart },
                  { id: 'signals', label: 'Signals & Forecast Lab', icon: Zap },
                  { id: 'risk', label: 'Risk & Volatility', icon: ShieldCheck },
                  { id: 'sentiment', label: 'News & FinBERT', icon: Newspaper },
                  { id: 'backtest', label: 'Backtest & Calibration', icon: BarChart3 },
                  { id: 'benchmarks', label: 'Model Benchmarks & Audit', icon: Award },
                  { id: 'digest', label: 'Daily Digest', icon: FileText },
                ].map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      className={`dash-tab-btn ${isActive ? 'active' : ''}`}
                      onClick={() => handleTabChange(tab.id as DashboardTab)}
                    >
                      <Icon size={16} strokeWidth={isActive ? 2.4 : 1.8} style={{ color: isActive ? 'var(--accent)' : '#64748b' }} />
                      <span style={{ flex: 1 }}>{tab.label}</span>
                      {isActive && <ChevronRight size={14} style={{ color: 'var(--accent)' }} />}
                    </button>
                  );
                })}
              </div>
            </div>
          </aside>

          {/* Dashboard Main Workspace */}
          <main style={{ minWidth: 0 }}>
            {/* v0.14.0 Engine & Governance Banner */}
            {systemStatus && (
              <div style={{
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '10px',
                padding: '10px 16px',
                marginBottom: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '10px',
                fontSize: '12px',
                fontFamily: 'var(--sans)',
                boxShadow: '0 1px 3px rgba(0,0,0,0.03)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ color: 'var(--dash-muted)', fontWeight: '500' }}>Data as-of:</span>
                    <strong style={{ fontFamily: 'var(--mono)', color: 'var(--ink)' }}>{systemStatus.data_as_of}</strong>
                  </div>
                  <div style={{ width: 1, height: 14, background: '#cbd5e1' }} />
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ color: 'var(--dash-muted)', fontWeight: '500' }}>Engine:</span>
                    <strong style={{ fontFamily: 'var(--mono)', color: 'var(--accent)' }}>v{systemStatus.model_version || systemStatus.version || '0.14.0'}</strong>
                  </div>
                  <div style={{ width: 1, height: 14, background: '#cbd5e1' }} />
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ color: 'var(--dash-muted)', fontWeight: '500' }}>Baseline MAE:</span>
                    <strong style={{ fontFamily: 'var(--mono)', color: 'var(--ink)' }}>₹{(systemStatus.baseline_mae_inr ?? 103.4).toFixed(2)}</strong>
                  </div>
                  {systemStatus.rolling_oos_rank_ic !== undefined && systemStatus.rolling_oos_rank_ic !== null && (
                    <>
                      <div style={{ width: 1, height: 14, background: '#cbd5e1' }} />
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ color: 'var(--dash-muted)', fontWeight: '500' }}>OOS Rank IC:</span>
                        <strong style={{ fontFamily: 'var(--mono)', color: '#1A7F4B' }}>
                          +{systemStatus.rolling_oos_rank_ic.toFixed(4)}
                        </strong>
                      </div>
                    </>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  {systemStatus.ml_promotion_status && (
                    <span style={{
                      background: '#fffbeb',
                      color: '#b45309',
                      border: '1px solid #fde68a',
                      padding: '3px 8px',
                      borderRadius: '6px',
                      fontWeight: '700',
                      fontSize: '10px',
                      fontFamily: 'var(--mono)',
                    }}>
                      ML GATE: {systemStatus.ml_promotion_status}
                    </span>
                  )}
                </div>
              </div>
            )}

            {error && (
              <div style={{
                background: 'var(--danger-bg)',
                border: '1px solid var(--danger-border)',
                color: 'var(--danger)',
                padding: '14px',
                borderRadius: '10px',
                marginBottom: '16px',
                fontFamily: 'var(--mono)',
                fontSize: '12px',
              }}>
                {error}
              </div>
            )}

            {activeTab === 'chart' && (
              <PriceChartView
                priceData={priceData}
                loading={loading || stockLoading}
                switchingTicker={switchingTicker}
                topStocks={rankedSignals}
                universeStocks={stocks}
                selectedTicker={selectedTicker}
                onSelectTicker={(t) => handleTickerChange(t)}
              />
            )}

            {activeTab === 'signals' && (
              <SignalsView
                signal={signalData}
                prediction={predictionData}
                loading={loading || stockLoading}
                onOpenBenchmarks={() => handleTabChange('benchmarks')}
              />
            )}

            {activeTab === 'risk' && (
              <RiskVolatilityView
                risk={riskData}
                loading={loading || stockLoading}
              />
            )}

            {activeTab === 'sentiment' && (
              <SentimentView
                sentiment={sentimentData}
                loading={loading || stockLoading}
              />
            )}

            {activeTab === 'backtest' && (
              <BacktestView
                backtest={backtestData}
                loading={loading}
              />
            )}

            {activeTab === 'benchmarks' && <ModelBenchmarksView />}

            {activeTab === 'digest' && (
              <DigestView
                digest={digestData}
                onSelectStock={(t) => handleTickerChange(t)}
                loading={loading}
              />
            )}

            <footer style={{
              marginTop: '40px',
              padding: '20px 0',
              borderTop: '1px solid var(--rule)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '12px',
              fontFamily: 'var(--sans)',
              fontSize: '12px',
              color: 'var(--dash-muted)',
            }}>
              <div>StockReason v0.14.0 · NSE Quantitative Intelligence Terminal</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: '11px' }}>
                Active: <strong style={{ color: 'var(--ink)' }}>{selectedTicker.replace('.NS', '')}</strong>
                {systemStatus ? `  ·  Data: ${systemStatus.data_as_of}  ·  ${systemStatus.tickers_loaded} equities` : ''}
              </div>
            </footer>
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
