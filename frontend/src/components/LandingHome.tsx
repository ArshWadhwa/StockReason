import React, { useEffect, useRef, useCallback } from 'react';
import type { MarketOverviewData, SignalData } from '../types';
import {
  TrendingUp,
  TrendingDown,
  ArrowUpRight,
  BarChart3,
  Brain,
  Compass,
  Cpu,
  CheckCircle2,
  Zap,
  LineChart,
} from 'lucide-react';

interface LandingHomeProps {
  overview: MarketOverviewData | null;
  signals: SignalData[];
  onOpenDashboard: (ticker?: string) => void;
}

/* ─── helper: deterministic colour from string ─── */
const tickerColors = [
  '#1da1f2', '#4285f4', '#34a853', '#ea4335',
  '#ff6f00', '#7b1fa2', '#00897b', '#c62828',
  '#0d47a1', '#2e7d32', '#ef6c00', '#ad1457',
];
function colorFor(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return tickerColors[Math.abs(h) % tickerColors.length];
}

/* ─── candlestick data for decorative chart ─── */
const CANDLES = [
  { o: 240, c: 248, h: 252, l: 236, g: true, vol: 65 },
  { o: 248, c: 242, h: 254, l: 238, g: false, vol: 45 },
  { o: 242, c: 250, h: 256, l: 240, g: true, vol: 80 },
  { o: 250, c: 246, h: 258, l: 244, g: false, vol: 40 },
  { o: 246, c: 255, h: 260, l: 244, g: true, vol: 90 },
  { o: 255, c: 252, h: 262, l: 248, g: false, vol: 50 },
  { o: 252, c: 258, h: 264, l: 250, g: true, vol: 70 },
  { o: 258, c: 254, h: 266, l: 250, g: false, vol: 35 },
  { o: 254, c: 262, h: 268, l: 252, g: true, vol: 85 },
  { o: 262, c: 256, h: 270, l: 254, g: false, vol: 55 },
  { o: 256, c: 264, h: 272, l: 254, g: true, vol: 95 },
  { o: 264, c: 260, h: 274, l: 256, g: false, vol: 45 },
  { o: 260, c: 268, h: 276, l: 258, g: true, vol: 75 },
  { o: 268, c: 264, h: 278, l: 260, g: false, vol: 40 },
  { o: 264, c: 272, h: 280, l: 262, g: true, vol: 88 },
  { o: 272, c: 268, h: 282, l: 264, g: false, vol: 50 },
  { o: 268, c: 276, h: 284, l: 266, g: true, vol: 92 },
  { o: 276, c: 272, h: 286, l: 268, g: false, vol: 48 },
  { o: 272, c: 280, h: 288, l: 270, g: true, vol: 100 },
  { o: 280, c: 276, h: 290, l: 272, g: false, vol: 52 },
  { o: 276, c: 284, h: 292, l: 274, g: true, vol: 84 },
  { o: 284, c: 278, h: 294, l: 276, g: false, vol: 60 },
  { o: 278, c: 286, h: 296, l: 276, g: true, vol: 94 },
  { o: 286, c: 282, h: 298, l: 278, g: false, vol: 58 },
];

/* ─── mini sparkline SVG generator ─── */
function renderSparkline(isUp: boolean) {
  const points = isUp
    ? "0,16 12,14 24,18 36,11 48,13 60,7 72,9 84,3"
    : "0,4 12,7 24,5 36,12 48,10 60,16 72,13 84,18";
  const strokeColor = isUp ? "#2d6a2e" : "#e53935";
  return (
    <svg width="84" height="22" viewBox="0 0 84 22" fill="none" className="stock-sparkline">
      <polyline
        points={points}
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/* ─── scroll-reveal hook ─── */
function useScrollReveal() {
  const ref = useRef<HTMLDivElement>(null);

  const setup = useCallback(() => {
    if (!ref.current) return;
    const els = ref.current.querySelectorAll('.reveal');
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -30px 0px' }
    );
    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const cleanup = setup();
    return cleanup;
  }, [setup]);

  return ref;
}

export const LandingHome: React.FC<LandingHomeProps> = ({
  overview,
  signals,
  onOpenDashboard,
}) => {
  const containerRef = useScrollReveal();
  const topBuy = signals.find(s => s.signal === 'BUY') || signals[0];
  const topStocks = signals.slice(0, 4);

  // Take a curated slice of signals for ticker ribbon
  interface RibbonStockItem {
    ticker: string;
    name: string;
    current_price: number;
    expected_return_1M: number;
    signal: string;
  }

  const ribbonStocks: RibbonStockItem[] = signals.length > 0
    ? signals.slice(0, 10).map(s => ({
        ticker: s.ticker,
        name: s.name,
        current_price: s.current_price,
        expected_return_1M: s.expected_return_1M,
        signal: s.signal,
      }))
    : [
        { ticker: 'RELIANCE.NS', name: 'Reliance Industries', current_price: 1298.5, expected_return_1M: 0.0128, signal: 'BUY' },
        { ticker: 'TCS.NS', name: 'Tata Consultancy', current_price: 2261.0, expected_return_1M: 0.0075, signal: 'BUY' },
        { ticker: 'INFY.NS', name: 'Infosys', current_price: 1086.5, expected_return_1M: 0.0052, signal: 'BUY' },
        { ticker: 'HDFCBANK.NS', name: 'HDFC Bank', current_price: 705.2, expected_return_1M: -0.0031, signal: 'HOLD' },
        { ticker: 'ICICIBANK.NS', name: 'ICICI Bank', current_price: 1412.0, expected_return_1M: 0.0142, signal: 'BUY' },
        { ticker: 'LT.NS', name: 'Larsen & Toubro', current_price: 3964.0, expected_return_1M: 0.0089, signal: 'BUY' },
        { ticker: 'ITC.NS', name: 'ITC Limited', current_price: 264.0, expected_return_1M: 0.0041, signal: 'HOLD' },
        { ticker: 'BHARTIARTL.NS', name: 'Bharti Airtel', current_price: 1840.0, expected_return_1M: 0.0165, signal: 'BUY' },
      ];

  return (
    <div ref={containerRef} className="landing-wrapper">
      {/* ═══════════ BACKGROUND DECORATIONS (SVGS & GLOWS) ═══════════ */}
      <div className="landing-bg-decorations" aria-hidden="true">
        {/* Ambient Glowing Orbs */}
        <div className="bg-glow-orb orb-1" />
        <div className="bg-glow-orb orb-2" />
        <div className="bg-glow-orb orb-3" />

        {/* Decorative SVG Stock Grid & Candlesticks pattern */}
        <svg className="bg-svg-pattern" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="stock-grid-pattern" width="80" height="80" patternUnits="userSpaceOnUse">
              <path d="M 80 0 L 0 0 0 80" fill="none" stroke="rgba(45, 106, 46, 0.035)" strokeWidth="1" />
              <circle cx="80" cy="80" r="1.5" fill="rgba(45, 106, 46, 0.07)" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#stock-grid-pattern)" />
        </svg>

        {/* Background Floating Stock SVG Motifs */}
        <div className="bg-stock-motif motif-hero-right">
          <svg width="320" height="180" viewBox="0 0 320 180" fill="none">
            <path
              d="M 10 150 Q 80 120, 140 130 T 260 50 T 310 20"
              stroke="rgba(45, 106, 46, 0.09)"
              strokeWidth="2.5"
              strokeDasharray="6 6"
              fill="none"
            />
            <path
              d="M 10 150 Q 80 120, 140 130 T 260 50 T 310 20 L 310 180 L 10 180 Z"
              fill="url(#grad-bg-motif)"
              opacity="0.5"
            />
            <defs>
              <linearGradient id="grad-bg-motif" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(76, 175, 80, 0.08)" />
                <stop offset="100%" stopColor="rgba(76, 175, 80, 0)" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      </div>

      {/* ═══════════ HERO ═══════════ */}
      <section className="shell">
        <div className="landing-hero">
          <div className="landing-hero-text reveal reveal-left">
            <h1>Make Better Investment Decisions With Alternative Data</h1>
            <p>
              Get the inside scoop on NIFTY 50 companies like never before.
              StockReason combines AI signals, FinBERT sentiment, and LSTM predictions to power smarter decisions.
            </p>
          </div>

          <div className="reveal reveal-right hero-chart-wrapper">
            {/* Floating Top Depth Badge */}
            <div className="floating-stat-badge badge-top-right float-anim">
              <div className="badge-icon-box green">
                <TrendingUp size={15} />
              </div>
              <div className="badge-text-box">
                <span className="badge-micro">ENSEMBLE CONVICTION</span>
                <span className="badge-title">4 / 4 Models Agree</span>
              </div>
            </div>

            {/* Floating Bottom Depth Badge */}
            <div className="floating-stat-badge badge-bottom-left float-anim-alt">
              <div className="badge-icon-box blue">
                <Brain size={15} />
              </div>
              <div className="badge-text-box">
                <span className="badge-micro">SENTIMENT VELOCITY</span>
                <span className="badge-title">{topBuy ? `Score: ${topBuy.sentiment_score.toFixed(2)}` : 'Score: +0.72'}</span>
              </div>
            </div>

            {/* Dimensional Hero Chart Card */}
            <div className="hero-chart-card depth-card hover-lift">
              <div className="hero-chart-header">
                <div className="hero-chart-ticker">
                  <div className="ticker-icon pulse-glow">
                    <LineChart size={18} color="#ffffff" strokeWidth={2.4} />
                  </div>
                  <div className="ticker-info">
                    <div className="ticker-title-row">
                      <h3>{topBuy ? topBuy.ticker.replace('.NS', '') : 'RELIANCE'}</h3>
                      <span className="nifty-tag">NIFTY 50</span>
                    </div>
                    <span className="company-fullname">
                      {topBuy?.name || 'Reliance Industries Ltd.'}
                    </span>
                  </div>
                </div>

                <div className="hero-chart-badge-group">
                  <span className="hero-chart-badge">
                    <TrendingUp size={13} style={{ marginRight: 4 }} />
                    {topBuy ? `${topBuy.expected_return_1M >= 0 ? '+' : ''}${(topBuy.expected_return_1M * 100).toFixed(2)}%` : '+1.28%'}
                  </span>
                </div>
              </div>

              <div className="hero-chart-tabs">
                <button>1D</button>
                <button>1W</button>
                <button className="active">1M</button>
                <button>3M</button>
                <button>1Y</button>
              </div>

              {/* Candlestick SVG chart with Volume & Trend Depth */}
              <div className="hero-chart-area">
                <svg width="100%" height="100%" viewBox="0 0 480 180" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="chartAreaGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#2d6a2e" stopOpacity="0.16" />
                      <stop offset="100%" stopColor="#2d6a2e" stopOpacity="0.0" />
                    </linearGradient>
                    <linearGradient id="emaGlow" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#2d6a2e" />
                      <stop offset="100%" stopColor="#4caf50" />
                    </linearGradient>
                  </defs>

                  {/* Horizontal Grid lines */}
                  {[0.2, 0.4, 0.6, 0.8].map((r, i) => (
                    <line
                      key={i}
                      x1="0"
                      y1={r * 180}
                      x2="480"
                      y2={r * 180}
                      stroke="rgba(0,0,0,0.06)"
                      strokeDasharray="3 3"
                      strokeWidth="1"
                    />
                  ))}

                  {/* Volume histogram bars at bottom */}
                  {CANDLES.map((c, i) => {
                    const x = 16 + i * 19;
                    const vHeight = (c.vol / 100) * 32;
                    return (
                      <rect
                        key={`vol-${i}`}
                        x={x - 4}
                        y={178 - vHeight}
                        width="8"
                        height={vHeight}
                        fill={c.g ? 'rgba(45, 106, 46, 0.22)' : 'rgba(229, 57, 53, 0.22)'}
                        rx="1.5"
                      />
                    );
                  })}

                  {/* Price labels */}
                  {['₹3,000', '₹2,800', '₹2,600', '₹2,400'].map((label, i) => (
                    <text key={i} x="424" y={32 + i * 38} fill="#94a3b8" fontSize="9" fontFamily="var(--mono)" fontWeight="500">
                      {label}
                    </text>
                  ))}

                  {/* Subtle Area fill under trend line */}
                  <path
                    d="M 16 142 L 35 138 L 54 140 L 73 134 L 92 128 L 111 126 L 130 120 L 149 122 L 168 114 L 187 116 L 206 108 L 225 110 L 244 102 L 263 104 L 282 96 L 301 98 L 320 90 L 339 92 L 358 84 L 377 86 L 396 78 L 415 82 L 434 74 L 453 76 L 453 180 L 16 180 Z"
                    fill="url(#chartAreaGradient)"
                  />

                  {/* Candlesticks */}
                  {CANDLES.map((c, i) => {
                    const maxP = 300, minP = 220;
                    const scale = (v: number) => 170 - ((v - minP) / (maxP - minP)) * 145;
                    const x = 16 + i * 19;
                    const oY = scale(c.o);
                    const cY = scale(c.c);
                    const hY = scale(c.h);
                    const lY = scale(c.l);
                    const color = c.g ? '#2d6a2e' : '#e53935';
                    return (
                      <g key={i}>
                        <line x1={x} y1={hY} x2={x} y2={lY} stroke={color} strokeWidth="1.2" strokeLinecap="round" />
                        <rect
                          x={x - 3.5}
                          y={Math.min(oY, cY)}
                          width="7"
                          height={Math.max(Math.abs(oY - cY), 3)}
                          fill={color}
                          rx="1.5"
                        />
                      </g>
                    );
                  })}

                  {/* Trendline overlay */}
                  <polyline
                    points="16,142 54,138 92,128 130,120 168,114 206,108 244,102 282,96 320,90 358,84 396,78 434,74 453,76"
                    fill="none"
                    stroke="url(#emaGlow)"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />

                  {/* Current price marker tag */}
                  <g transform="translate(372, 42)">
                    <rect x="0" y="0" width="86" height="24" rx="6" fill="#1e4d1f" />
                    <text x="43" y="16" fill="#ffffff" fontSize="10" fontFamily="var(--mono)" textAnchor="middle" fontWeight="700">
                      ₹{topBuy ? topBuy.current_price.toFixed(1) : '1,298.5'}
                    </text>
                  </g>
                </svg>
              </div>

              <div className="hero-chart-footer">
                <div className="hero-chart-footer-left">
                  <div className="signal-pill-wrap">
                    <span className="signal-pill-label">Signal</span>
                    <span className="signal-pill-val">
                      <Zap size={12} className="signal-pill-icon" />
                      {topBuy?.signal || 'BUY'}
                    </span>
                  </div>
                </div>
                <div className="hero-chart-footer-right">
                  <div className="price-big">₹{topBuy ? topBuy.current_price.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '1,298'}</div>
                  <div className="price-change-row">
                    {topBuy && topBuy.expected_return_1M >= 0 ? (
                      <span className="price-change up">
                        <TrendingUp size={12} />
                        +{Math.abs(topBuy.expected_return_1M * 100).toFixed(2)}% (1M)
                      </span>
                    ) : (
                      <span className="price-change down">
                        <TrendingDown size={12} />
                        -{topBuy ? Math.abs(topBuy.expected_return_1M * 100).toFixed(2) : '1.28'}% (1M)
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ STOCK TICKER RIBBON ═══════════ */}
      <div className="stock-ticker-ribbon">
        <div className="stock-ticker-track">
          {ribbonStocks.concat(ribbonStocks).map((stock, idx) => {
            const sym = stock.ticker.replace('.NS', '');
            const isPos = stock.expected_return_1M >= 0;
            return (
              <div
                className="ticker-strip-item"
                key={idx}
                onClick={() => onOpenDashboard(stock.ticker)}
              >
                <span className="ticker-strip-sym">{sym}</span>
                <span className="ticker-strip-price">₹{stock.current_price.toFixed(1)}</span>
                <span className={`ticker-strip-change ${isPos ? 'up' : 'down'}`}>
                  {isPos ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                  {isPos ? '+' : ''}{(stock.expected_return_1M * 100).toFixed(2)}%
                </span>
                {renderSparkline(isPos)}
              </div>
            );
          })}
        </div>
      </div>

      {/* ═══════════ METRICS DOCK (DEPTH CARDS) ═══════════ */}
      <div className="metrics-strip">
        <div className="metrics-strip-inner">
          {[
            {
              cat: 'AI Signals',
              name: `${signals.filter(s => s.signal === 'BUY').length} BUY signals`,
              val: `${signals.length}`,
              change: 'NIFTY 50 Active',
              up: true,
              icon: <BarChart3 size={18} />,
            },
            {
              cat: 'FinBERT Sentiment',
              name: topBuy?.name || 'Top Pick',
              val: topBuy ? topBuy.sentiment_score.toFixed(2) : '0.72',
              change: topBuy && topBuy.sentiment_score > 0.5 ? 'Bullish Sentiment' : 'Bearish Sentiment',
              up: topBuy ? topBuy.sentiment_score > 0.5 : true,
              icon: <Brain size={18} />,
            },
            {
              cat: 'Market Regime',
              name: overview?.market_regime?.replace(/_/g, ' ') || 'Low Volatility',
              val: `${overview?.active_coverage_count || 50}`,
              change: 'Constructive Market',
              up: true,
              icon: <Compass size={18} />,
            },
            {
              cat: 'LSTM Model Confidence',
              name: topBuy?.name || 'Top Pick',
              val: topBuy ? `${(topBuy.confidence * 100).toFixed(0)}%` : '85%',
              change: topBuy?.ensemble_agreement ? 'Ensemble Agrees' : 'Mixed Signals',
              up: topBuy?.ensemble_agreement ?? true,
              icon: <Cpu size={18} />,
            },
          ].map((m, i) => (
            <div className="metric-glass-card depth-card hover-lift" key={i}>
              <div className="metric-header-row">
                <span className="metric-category">{m.cat}</span>
                <div className="metric-icon-circle">{m.icon}</div>
              </div>

              <div className="metric-main-row">
                <span className="metric-value">{m.val}</span>
                <div className={`metric-change-badge ${m.up ? 'up' : 'down'}`}>
                  {m.up ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
                  <span>{m.change}</span>
                </div>
              </div>

              <div className="metric-footer-note">
                <span className="metric-stock-name">{m.name}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ═══════════ SECTION 1: TOP STOCKS (card LEFT, text RIGHT) ═══════════ */}
      <section className="shell">
        <div className="landing-section">
          <div className="landing-section-split">
            <div className="top-stocks-card depth-card reveal reveal-left hover-lift">
              <div className="top-stocks-header">
                <div>
                  <h3>Top Ranked NIFTY 50 Stocks</h3>
                  <p className="top-stocks-subtitle">Ranked by LSTM Expected Return & FinBERT Momentum</p>
                </div>
                <button className="view-all-btn" onClick={() => onOpenDashboard()}>
                  <span>View All</span>
                  <ArrowUpRight size={14} />
                </button>
              </div>

              <div className="stock-list-container">
                {topStocks.length > 0 ? topStocks.map((stock, i) => {
                  const ticker = stock.ticker.replace('.NS', '');
                  const changeVal = (stock.expected_return_1M * 100);
                  const isUp = changeVal >= 0;
                  return (
                    <div
                      className="stock-list-item depth-item hover-lift"
                      key={i}
                      onClick={() => onOpenDashboard(stock.ticker)}
                    >
                      <div className="stock-list-left">
                        <span className="stock-rank-badge">#{i + 1}</span>
                        <div className="stock-logo" style={{ background: colorFor(ticker) }}>
                          {ticker.slice(0, 2)}
                        </div>
                        <div className="stock-list-info">
                          <div className="stock-symbol-row">
                            <span className="stock-symbol">{ticker}</span>
                            <span className="signal-mini-badge">{stock.signal}</span>
                          </div>
                          <span className="stock-name">{stock.name}</span>
                        </div>
                      </div>

                      <div className="stock-list-center">
                        {renderSparkline(isUp)}
                      </div>

                      <div className="stock-list-right">
                        <span className="stock-price">
                          ₹{stock.current_price.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
                        </span>
                        <span className={`stock-change-pill ${isUp ? 'up' : 'down'}`}>
                          {isUp ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
                          {isUp ? '+' : ''}{changeVal.toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  );
                }) : (
                  [
                    { sym: 'RELIANCE', name: 'Reliance Industries', price: '₹2,948.5', change: '+1.28%', up: true },
                    { sym: 'TCS', name: 'Tata Consultancy Services', price: '₹3,842.1', change: '+0.75%', up: true },
                    { sym: 'INFY', name: 'Infosys Ltd.', price: '₹1,456.3', change: '+0.52%', up: true },
                    { sym: 'HDFCBANK', name: 'HDFC Bank', price: '₹1,612.8', change: '-0.31%', up: false },
                  ].map((s, i) => (
                    <div className="stock-list-item depth-item hover-lift" key={i} onClick={() => onOpenDashboard()}>
                      <div className="stock-list-left">
                        <span className="stock-rank-badge">#{i + 1}</span>
                        <div className="stock-logo" style={{ background: colorFor(s.sym) }}>{s.sym.slice(0, 2)}</div>
                        <div className="stock-list-info">
                          <span className="stock-symbol">{s.sym}</span>
                          <span className="stock-name">{s.name}</span>
                        </div>
                      </div>
                      <div className="stock-list-center">
                        {renderSparkline(s.up)}
                      </div>
                      <div className="stock-list-right">
                        <span className="stock-price">{s.price}</span>
                        <span className={`stock-change-pill ${s.up ? 'up' : 'down'}`}>{s.change}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="section-text-block reveal reveal-right">
              <h2>What Stocks are Trending?</h2>
              <p>
                Discover NIFTY 50 opportunities using AI-powered
                signals that combine LSTM price predictions, FinBERT
                sentiment analysis, technical indicators, and macro
                context to surface the highest-conviction picks.
              </p>

              <div className="section-bullet-group">
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Real-time NSE market depth and OHLCV ingestion</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Cross-validated return forecasts for 1-month horizons</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Ensemble agreement scoring to eliminate false signals</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ SECTION 2: SHAP EXPLAINABILITY (text LEFT, image RIGHT) ═══════════ */}
      <section className="shell">
        <div className="landing-section">
          <div className="landing-section-split">
            <div className="section-text-block reveal reveal-left">
              <h2>AI Signals Powered by SHAP Explainability</h2>
              <p>
                Every BUY / HOLD / AVOID signal is backed by SHAP
                feature attribution, showing exactly which technical
                and sentiment factors drove the recommendation.
                No black boxes — full transparency.
              </p>

              <div className="section-bullet-group">
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Quantifies the exact contribution of RSI, MACD, and Bollinger bands</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Separates news headline sentiment impact from technical momentum</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Visual waterfalls explain why a stock was upgraded or downgraded</span>
                </div>
              </div>
            </div>

            <div className="section-illustration-wrapper reveal reveal-right">
              <div className="illustration-depth-card hover-float">
                <img src="/images/stock-picks.jpg" alt="Stock picks illustration" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ SECTION 3: REAL-TIME SENTIMENT (image LEFT, text RIGHT) ═══════════ */}
      <section className="shell">
        <div className="landing-section">
          <div className="landing-section-split">
            <div className="section-illustration-wrapper reveal reveal-left">
              <div className="illustration-depth-card hover-float">
                <img src="/images/stock-alerts.jpg" alt="Stock alerts app" style={{ maxWidth: 340 }} />
              </div>
            </div>

            <div className="section-text-block reveal reveal-right">
              <h2>Real-Time Sentiment from FinBERT NLP</h2>
              <p>
                Track headline bias, sentiment shifts, and divergence
                alerts across your NIFTY 50 watchlist. FinBERT scores
                every headline so you catch sentiment changes before
                the market prices them in.
              </p>

              <div className="section-bullet-group">
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Fine-tuned specifically on financial disclosures and news</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Detects subtle tone changes in earnings transcripts and releases</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Signals sentiment divergences before they show up in technical charts</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ SECTION 4: BACKTEST (text LEFT, image RIGHT) ═══════════ */}
      <section className="shell">
        <div className="landing-section">
          <div className="landing-section-split">
            <div className="section-text-block reveal reveal-left">
              <h2>Backtest Against the Market</h2>
              <p>
                Compare StockReason's signal-based strategy against
                NIFTY 50 buy-and-hold and random baselines. View equity
                curves, Sharpe ratios, win rates, and drawdown analysis
                to validate the edge.
              </p>

              <div className="section-bullet-group">
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Historical simulations with realistic slippage and transaction costs</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Walk-forward validation prevents look-ahead and overfitting bias</span>
                </div>
                <div className="bullet-item">
                  <CheckCircle2 size={16} className="bullet-icon" />
                  <span>Metrics on maximum drawdown, Sharpe ratio, and annualized alpha</span>
                </div>
              </div>
            </div>

            <div className="section-illustration-wrapper reveal reveal-right">
              <div className="illustration-depth-card hover-float">
                <img src="/images/portfolio.jpg" alt="Portfolio dashboard" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ TESTIMONIALS / TERMINAL LAUNCH CTA ═══════════ */}
      <section className="shell">
        <div className="cta-depth-card reveal reveal-up hover-lift">
          <div className="cta-content">
            <h2>Built for Faster, Smarter<br />Market Reading</h2>
            <p>
              LSTM predictions · FinBERT sentiment · SHAP explainability · Backtest validation
            </p>

            <div className="cta-buttons-row">
              <button className="cta-primary-btn" onClick={() => onOpenDashboard()}>
                <span>Launch AI Terminal</span>
                <ArrowUpRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ FOOTER ═══════════ */}
      <div className="landing-footer">
        <div className="landing-footer-inner shell">
          <div className="footer-left">
            <div className="footer-brand">
              <div className="brand-icon-mini">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
                  <polyline points="16 7 22 7 22 13" />
                </svg>
              </div>
              <span>StockReason</span>
            </div>
            <p className="footer-desc">AI-Driven Quantitative Intelligence for Indian Equities.</p>
          </div>
          <div className="footer-right">
            <p>Built for faster market reading. StockReason © 2024</p>
          </div>
        </div>
      </div>
    </div>
  );
};
