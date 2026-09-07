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

type PageMode = 'home' | 'dashboard';
type DashboardTab = 'chart' | 'signals' | 'sentiment' | 'backtest' | 'digest';

export function App() {
  const [pageMode, setPageMode] = useState<PageMode>('home');
  const [activeTab, setActiveTab] = useState<DashboardTab>('chart');
  const [selectedTicker, setSelectedTicker] = useState<string>('RELIANCE.NS');

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
          setSelectedTicker(univRes.stocks[0].ticker);
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

  const handleOpenDashboard = (ticker?: string) => {
    if (ticker) setSelectedTicker(ticker);
    setPageMode('dashboard');
    setActiveTab('chart');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div>
      {/* Top Header */}
      <div className="shell" style={{ paddingBottom: 0 }}>
        <header className="topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div
              className="brand"
              onClick={() => setPageMode('home')}
              style={{ cursor: 'pointer' }}
            >
              StockReason
            </div>
            <div className="navnote">
              {pageMode === 'home' ? 'NIFTY 50 / signal-led market reading / preview' : 'NIFTY 50 · AI Analysis Workspace'}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {pageMode === 'home' ? (
              <button
                className="btn primary"
                onClick={() => handleOpenDashboard()}
                style={{ fontSize: '12px', padding: '8px 16px' }}
              >
                Open Nifty 50 UI →
              </button>
            ) : (
              <button
                className="btn secondary"
                onClick={() => setPageMode('home')}
                style={{ fontSize: '12px', padding: '8px 16px' }}
              >
                ← Back to Home
              </button>
            )}
          </div>
        </header>
      </div>

      {/* Scrolling Ticker Tape (Home Page only) */}
      {pageMode === 'home' && (
        <div className="ticker-wrap" style={{ marginTop: '20px' }}>
          <div className="ticker-track">
            <div className="ticker-line">
              {(rankedSignals.length ? rankedSignals.concat(rankedSignals) : []).map((stock, i) => (
                <div
                  key={i}
                  className="ticker-item"
                  onClick={() => handleOpenDashboard(stock.ticker)}
                >
                  <span className="ticker-name">{stock.ticker.replace('.NS', '')}</span>
                  <span className="ticker-price">₹{stock.current_price.toFixed(1)}</span>
                  <span className={stock.signal === 'BUY' ? 'ticker-up' : (stock.signal === 'AVOID' ? 'ticker-down' : 'ticker-price')}>
                    {stock.signal === 'BUY' ? '▲' : (stock.signal === 'AVOID' ? '▼' : '●')} {stock.signal}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* PAGE 1: PURE LANDING HOME */}
      {pageMode === 'home' && (
        <main className="shell" style={{ paddingTop: '10px' }}>
          <LandingHome
            overview={overview}
            signals={rankedSignals}
            onOpenDashboard={handleOpenDashboard}
          />
          <footer className="footer-line">
            <div>Built for faster market reading.</div>
            <div>StockReason / NIFTY 50 / live tape concept</div>
          </footer>
        </main>
      )}

      {/* PAGE 2: SEPARATE NIFTY 50 DASHBOARD WORKSPACE */}
      {pageMode === 'dashboard' && (
        <div className="shell" style={{ paddingTop: '20px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '24px', alignItems: 'start' }}>
            
            {/* Sidebar Controls */}
            <aside style={{
              background: '#111111',
              border: '1px solid #1e1e1e',
              borderRadius: '8px',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '18px'
            }}>
              <div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '13px', fontWeight: '700', color: '#f2f2f2' }}>
                  StockReason
                </div>
                <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '2px' }}>
                  NIFTY 50 · AI Analysis
                </div>
              </div>

              <div style={{ borderTop: '1px solid #1c1c1c', paddingTop: '14px' }}>
                <label style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--muted)', textTransform: 'uppercase' }}>
                  Company
                </label>
                <select
                  value={selectedTicker}
                  onChange={(e) => setSelectedTicker(e.target.value)}
                  style={{
                    width: '100%',
                    marginTop: '6px',
                    background: '#1a1a1a',
                    color: '#f0f0f0',
                    border: '1px solid #2a2a2a',
                    borderRadius: '6px',
                    padding: '8px 10px',
                    fontFamily: 'var(--mono)',
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  {stocks.map(s => (
                    <option key={s.ticker} value={s.ticker}>
                      {s.name} ({s.ticker.replace('.NS', '')})
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ borderTop: '1px solid #1c1c1c', paddingTop: '14px' }}>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
                  Analysis Views
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {[
                    { id: 'chart', label: 'Price Action & Chart' },
                    { id: 'signals', label: 'AI Signals & SHAP' },
                    { id: 'sentiment', label: 'News & FinBERT' },
                    { id: 'backtest', label: 'Backtest vs Baseline' },
                    { id: 'digest', label: 'Daily Digest' },
                  ].map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as DashboardTab)}
                      style={{
                        padding: '8px 12px',
                        borderRadius: '4px',
                        border: '1px solid',
                        borderColor: activeTab === tab.id ? '#333' : 'transparent',
                        background: activeTab === tab.id ? '#1c1c1c' : 'transparent',
                        color: activeTab === tab.id ? '#fff' : 'var(--muted)',
                        fontFamily: 'var(--mono)',
                        fontSize: '11px',
                        textAlign: 'left',
                        cursor: 'pointer'
                      }}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ borderTop: '1px solid #1c1c1c', paddingTop: '14px', fontSize: '11px', color: '#666', fontFamily: 'var(--mono)' }}>
                <div>Data sources:</div>
                <div style={{ marginTop: '4px' }}>• Yahoo Finance — OHLCV</div>
                <div>• News / FinBERT — NLP</div>
                <div>• PyTorch LSTM + MC Dropout</div>
                {systemStatus && (
                  <div style={{ marginTop: '10px', padding: '8px', background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: '4px' }}>
                    <div style={{ color: '#4ade80', fontWeight: '600', fontSize: '10px', marginBottom: '4px' }}>● LIVE DATA</div>
                    <div style={{ fontSize: '10px', color: '#888' }}>Last refreshed:</div>
                    <div style={{ fontSize: '11px', color: '#ccc', marginTop: '2px' }}>{systemStatus.last_refreshed}</div>
                    <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>
                      {systemStatus.tickers_loaded} stocks · {systemStatus.predictions_loaded} predictions
                    </div>
                  </div>
                )}
              </div>
            </aside>

            {/* Dashboard Main Workspace */}
            <main style={{ minWidth: 0 }}>
              {error && (
                <div style={{
                  background: 'var(--danger-bg)',
                  border: '1px solid var(--danger-border)',
                  color: 'var(--danger)',
                  padding: '12px',
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
                  onSelectStock={(t) => setSelectedTicker(t)}
                  loading={loading}
                />
              )}

              <footer className="footer-line" style={{ marginTop: '30px' }}>
                <div>StockReason / NIFTY 50 Analysis Terminal — {systemStatus?.data_source === 'real' ? '100% Real Data' : 'Loading…'}</div>
                <div>Active Ticker: {selectedTicker}{systemStatus ? ` · Refreshed: ${systemStatus.last_refreshed}` : ''}</div>
              </footer>
            </main>

          </div>
        </div>
      )}
    </div>
  );
}

export default App;
