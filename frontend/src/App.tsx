import { useState, useEffect } from 'react';
import { api } from './api';
import type {
  UniverseStock, MarketOverviewData,
  StockPriceData, SentimentData, PredictionData,
  SignalData, BacktestData, DailyDigestData, SystemStatus
} from './types';

import { LandingHome } from './components/LandingHome';
import { PriceChartView } from './components/PriceChartView';
import { SignalsView } from './components/SignalsView';
import { SentimentView } from './components/SentimentView';
import { BacktestView } from './components/BacktestView';
import { DigestView } from './components/DigestView';

import {
  LineChart,
  Zap,
  Newspaper,
  BarChart3,
  FileText,
  ChevronRight
} from 'lucide-react';

import Lenis from 'lenis';
import 'lenis/dist/lenis.css';

type PageMode = 'home' | 'dashboard';
type DashboardTab = 'chart' | 'signals' | 'sentiment' | 'backtest' | 'digest';

const VALID_TABS: DashboardTab[] = ['chart', 'signals', 'sentiment', 'backtest', 'digest'];

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

  // App States
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Data Stores
  const [stocks, setStocks] = useState<UniverseStock[]>([]);
  const [overview, setOverview] = useState<MarketOverviewData | null>(null);
  const [rankedSignals, setRankedSignals] = useState<SignalData[]>([]);
  const [priceData, setPriceData] = useState<StockPriceData | null>(null);
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [signalData, setSignalData] = useState<SignalData | null>(null);
  const [backtestData, setBacktestData] = useState<BacktestData | null>(null);
  const [digestData, setDigestData] = useState<DailyDigestData | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  // Listen to browser Back / Forward (popstate)
  useEffect(() => {
    const handlePopState = () => {
      const isDashboard = window.location.pathname.startsWith('/dashboard');
      setPageMode(isDashboard ? 'dashboard' : 'home');

      const params = new URLSearchParams(window.location.search);
      const tickerParam = params.get('ticker');
      if (tickerParam) {
        setSelectedTicker(tickerParam);
      }
      const tabParam = params.get('tab') as DashboardTab | null;
      if (tabParam && VALID_TABS.includes(tabParam)) {
        setActiveTab(tabParam);
      }
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
          api.getRankedSignals(true), // true = show all ticker strip names on landing home preview
          api.getBacktest(),
          api.getDailyDigest(),
          api.getSystemStatus()
        ]);

        setStocks(univRes.stocks);
        setOverview(overRes);
        setRankedSignals(sigsRes);
        setBacktestData(btRes);
        setDigestData(digRes);
        setSystemStatus(statusRes);

        if (univRes.stocks.length > 0) {
          setSelectedTicker(prev => {
            const params = new URLSearchParams(window.location.search);
            const urlTicker = params.get('ticker');
            if (urlTicker && univRes.stocks.some(s => s.ticker === urlTicker)) {
              return urlTicker;
            }
            if (prev && univRes.stocks.some(s => s.ticker === prev)) {
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

  // Fetch Stock-Specific Details
  useEffect(() => {
    if (!selectedTicker) return;

    async function loadStockDetails() {
      try {
        setError(null);
        const [pData, sData, prData, sigData] = await Promise.all([
          api.getStockPrices(selectedTicker),
          api.getStockSentiment(selectedTicker),
          api.getStockPredictions(selectedTicker),
          api.getStockSignal(selectedTicker)
        ]);
        setPriceData(pData);
        setSentimentData(sData);
        setPredictionData(prData);
        setSignalData(sigData);
      } catch (err: any) {
        console.error(`Failed to fetch details for ${selectedTicker}:`, err);
        setError(`Failed to load data for ${selectedTicker}`);
      }
    }
    loadStockDetails();
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
        {/* Light Navbar */}
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
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
                  <polyline points="16 7 22 7 22 13" />
                </svg>
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
              Open Dashboard →
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
  const selectedStock = stocks.find(s => s.ticker === selectedTicker);

  const activeSignal = (signalData && signalData.ticker === selectedTicker)
    ? signalData.signal
    : rankedSignals.find(s => s.ticker === selectedTicker)?.signal;

  const getSuggestion = (sig?: string) => {
    if (sig === 'BUY') {
      return { label: 'BUY', color: '#16a34a', bg: '#e8f5e9', border: '#c8e6c9', icon: '▲' };
    }
    if (sig === 'HOLD') {
      return { label: 'HOLD', color: '#b45309', bg: '#fefce8', border: '#fde68a', icon: '◆' };
    }
    if (sig === 'AVOID' || sig === 'SELL') {
      return { label: 'SELL', color: '#dc2626', bg: '#fef2f2', border: '#fecaca', icon: '▼' };
    }
    return { label: 'HOLD', color: '#b45309', bg: '#fefce8', border: '#fde68a', icon: '◆' };
  };

  const suggestion = getSuggestion(activeSignal);

  return (
    <div className="dashboard-mode">
      {/* ═══════════ AMBIENT BACKGROUND DECORATIONS ═══════════ */}
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

      {/* Light Navbar matching Homepage */}
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
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
                <polyline points="16 7 22 7 22 13" />
              </svg>
            </div>
            StockReason
          </a>

          {/* Right side: Current opened stock name & suggestion */}
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
                Stock
              </div>
              <select
                className="dash-select"
                value={selectedTicker}
                onChange={(e) => handleTickerChange(e.target.value)}
              >
                {stocks.map(s => (
                  <option key={s.ticker} value={s.ticker}>
                    {s.name} ({s.ticker.replace('.NS', '')})
                  </option>
                ))}
              </select>
            </div>

            <div style={{ borderTop: '1px solid #edf2f7', paddingTop: '16px' }}>
              <div style={{ fontFamily: 'var(--sans)', fontSize: '11px', fontWeight: '600', color: 'var(--dash-muted)', marginBottom: '10px' }}>
                Analysis views
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {[
                  { id: 'chart', label: 'Price Action & Chart', icon: LineChart },
                  { id: 'signals', label: 'AI Signals & SHAP', icon: Zap },
                  { id: 'sentiment', label: 'News & FinBERT', icon: Newspaper },
                  { id: 'backtest', label: 'Backtest vs Baseline', icon: BarChart3 },
                  { id: 'digest', label: 'Daily Digest', icon: FileText },
                ].map(tab => {
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
            {error && (
              <div style={{
                background: 'var(--danger-bg)',
                border: '1px solid var(--danger-border)',
                color: 'var(--danger)',
                padding: '14px',
                borderRadius: '10px',
                marginBottom: '16px',
                fontFamily: 'var(--mono)',
                fontSize: '12px'
              }}>
                {error}
              </div>
            )}

            {activeTab === 'chart' && (
              <PriceChartView
                priceData={priceData}
                loading={loading}
              />
            )}

            {activeTab === 'signals' && (
              <SignalsView
                signal={signalData}
                prediction={predictionData}
                loading={loading}
              />
            )}

            {activeTab === 'sentiment' && (
              <SentimentView
                sentiment={sentimentData}
                loading={loading}
              />
            )}

            {activeTab === 'backtest' && (
              <BacktestView
                backtest={backtestData}
                loading={loading}
              />
            )}

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
              color: 'var(--dash-muted)'
            }}>
              <div>StockReason · NIFTY 50 analysis terminal</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: '11px' }}>Active: <strong style={{ color: 'var(--ink)' }}>{selectedTicker.replace('.NS', '')}</strong>{systemStatus ? `  ·  Refreshed: ${systemStatus.last_refreshed} IST` : ''}</div>
            </footer>
          </main>

        </div>
      </div>
    </div>
  );
}

export default App;
