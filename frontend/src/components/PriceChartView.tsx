import React, { useState } from 'react';
import type { StockPriceData } from '../types';

interface PriceChartViewProps {
  priceData: StockPriceData | null;
  loading: boolean;
}

export const PriceChartView: React.FC<PriceChartViewProps> = ({ priceData, loading }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (loading || !priceData) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel" style={{ height: '140px' }} />
        <div className="ledger-panel" style={{ height: '360px' }} />
      </div>
    );
  }

  const history = priceData.history;
  const closes = history.map(h => h.close);
  const minPrice = Math.min(...closes) * 0.98;
  const maxPrice = Math.max(...closes) * 1.02;
  const priceRange = maxPrice - minPrice || 1;

  const width = 880;
  const height = 300;

  const toSvgPoints = (arr: (number | undefined)[]) =>
    arr.map((val, idx) => {
      if (val === undefined) return null;
      const x = (idx / (arr.length - 1)) * width;
      const y = height - ((val - minPrice) / priceRange) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).filter(Boolean).join(' ');

  const pricePoints = toSvgPoints(closes);
  const areaPolygon = `0,${height} ${pricePoints} ${width},${height}`;

  const latest = history[history.length - 1];
  const prev = history[history.length - 2] || latest;
  const changePct = ((latest.close - prev.close) / prev.close) * 100;
  const return1w = ((latest.close - history[Math.max(0, history.length - 6)].close) / history[Math.max(0, history.length - 6)].close) * 100;
  const return1m = ((latest.close - history[Math.max(0, history.length - 22)].close) / history[Math.max(0, history.length - 22)].close) * 100;

  const activePoint = hoverIndex !== null && hoverIndex >= 0 && hoverIndex < history.length
    ? history[hoverIndex]
    : latest;
  const activeX = hoverIndex !== null ? (hoverIndex / (history.length - 1)) * width : width;
  const activeY = height - ((activePoint.close - minPrice) / priceRange) * height;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Stock Overview Header */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
              <h2 style={{ fontSize: '26px', fontWeight: '700', color: '#f5f5f5', letterSpacing: '-0.02em' }}>
                {priceData.name}
              </h2>
              <span style={{ fontFamily: 'var(--mono)', fontSize: '12px', color: '#888' }}>
                {priceData.ticker}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '16px', marginTop: '6px' }}>
              <span style={{ fontSize: '32px', fontWeight: '700', fontFamily: 'var(--mono)', color: '#ffffff' }}>
                ₹{activePoint.close.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span style={{
                fontSize: '14px',
                fontFamily: 'var(--mono)',
                fontWeight: '600',
                color: changePct >= 0 ? 'var(--success)' : 'var(--danger)'
              }}>
                {changePct >= 0 ? `+${changePct.toFixed(2)}%` : `${changePct.toFixed(2)}%`}
              </span>
              {hoverIndex !== null && (
                <span style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#888' }}>
                  {activePoint.date}
                </span>
              )}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span className="ledger-tag tag-buy">BUY SIGNAL</span>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#777', marginTop: '6px' }}>
              REGIME: {latest.regime.replace(/_/g, ' ').toUpperCase()}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', marginTop: '14px', borderTop: '1px solid #1e1e1e', paddingTop: '10px', fontSize: '11px', fontFamily: 'var(--mono)', color: '#888' }}>
          <span>SECTOR: {priceData.sector}</span>
          <span style={{ color: '#333' }}>|</span>
          <span>UNIVERSE: {priceData.universe}</span>
          <span style={{ color: '#333' }}>|</span>
          <span style={{ color: 'var(--success)' }}>MC CONFIDENCE: 88%</span>
        </div>
      </div>

      {/* Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
        <div className="ledger-cell">
          <div className="ledger-label">1D Return</div>
          <div className={`ledger-metric ${changePct >= 0 ? 'ledger-metric-brass' : 'ledger-metric-rust'}`}>
            {changePct >= 0 ? `+${changePct.toFixed(2)}%` : `${changePct.toFixed(2)}%`}
          </div>
          <div className="ledger-note">vs prior close</div>
        </div>

        <div className="ledger-cell">
          <div className="ledger-label">1W Return</div>
          <div className={`ledger-metric ${return1w >= 0 ? 'ledger-metric-brass' : 'ledger-metric-rust'}`}>
            {return1w >= 0 ? `+${return1w.toFixed(2)}%` : `${return1w.toFixed(2)}%`}
          </div>
          <div className="ledger-note">5-day rolling return</div>
        </div>

        <div className="ledger-cell">
          <div className="ledger-label">1M Return</div>
          <div className={`ledger-metric ${return1m >= 0 ? 'ledger-metric-brass' : 'ledger-metric-rust'}`}>
            {return1m >= 0 ? `+${return1m.toFixed(2)}%` : `${return1m.toFixed(2)}%`}
          </div>
          <div className="ledger-note">21 trading sessions</div>
        </div>
      </div>

      {/* Chart Canvas Panel */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div>
            <div style={{ fontSize: '15px', fontWeight: '600', color: '#f0f0f0' }}>
              Historical Price Action
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#777', marginTop: '2px' }}>
              180 daily trading sessions · Hover across curve to inspect any date
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--success)' }}>
            ── Close Price (₹)
          </div>
        </div>

        {/* Clean Visible High-Contrast Price Curve */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: `${height}px`,
            background: '#090a0d',
            border: '1px solid #1c222b',
            borderRadius: '4px',
            overflow: 'hidden'
          }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = (e.clientX - rect.left) / rect.width;
            const idx = Math.min(history.length - 1, Math.max(0, Math.round(relX * (history.length - 1))));
            setHoverIndex(idx);
          }}
          onMouseLeave={() => setHoverIndex(null)}
        >
          <svg
            viewBox={`0 0 ${width} ${height}`}
            style={{ width: '100%', height: '100%', display: 'block' }}
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4ade80" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#4ade80" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Horizontal Grid lines */}
            {[0.2, 0.4, 0.6, 0.8].map((ratio, i) => {
              const y = height * ratio;
              const priceVal = maxPrice - (ratio * priceRange);
              return (
                <g key={i}>
                  <line
                    x1="0"
                    y1={y}
                    x2={width}
                    y2={y}
                    stroke="#1a202c"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                  />
                  <text
                    x="10"
                    y={y - 4}
                    fill="#555"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    ₹{priceVal.toFixed(0)}
                  </text>
                </g>
              );
            })}

            {/* Price Area Fill */}
            <polygon fill="url(#chartGradient)" points={areaPolygon} />

            {/* Main Price Line - Bold and high contrast */}
            <polyline
              fill="none"
              stroke="#4ade80"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={pricePoints}
            />

            {/* Hover Crosshair & Marker */}
            <line
              x1={activeX}
              y1={0}
              x2={activeX}
              y2={height}
              stroke="#ffffff"
              strokeWidth="1"
              strokeDasharray="2 2"
              opacity="0.4"
            />
            <circle
              cx={activeX}
              cy={activeY}
              r="4.5"
              fill="#ffffff"
              stroke="#4ade80"
              strokeWidth="2.5"
            />
          </svg>
        </div>

        {/* Dynamic Coordinate Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', fontFamily: 'var(--mono)', fontSize: '11px', color: '#777' }}>
          <div>
            Low: <span style={{ color: '#aaa' }}>₹{minPrice.toFixed(1)}</span> · High: <span style={{ color: '#aaa' }}>₹{maxPrice.toFixed(1)}</span>
          </div>
          <div>
            Session: <span style={{ color: '#ccc' }}>{activePoint.date}</span> · Close: <span style={{ color: '#fff', fontWeight: '600' }}>₹{activePoint.close.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
