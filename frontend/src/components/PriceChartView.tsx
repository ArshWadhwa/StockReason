import React, { useState, useMemo } from 'react';
import type { StockPriceData, PricePoint } from '../types';

interface PriceChartViewProps {
  priceData: StockPriceData | null;
  loading: boolean;
}

type TimeRange = '1M' | '3M' | '6M' | 'ALL';
type ChartStyle = 'candles' | 'line';

export const PriceChartView: React.FC<PriceChartViewProps> = ({ priceData, loading }) => {
  const [chartStyle, setChartStyle] = useState<ChartStyle>('candles');
  const [timeRange, setTimeRange] = useState<TimeRange>('3M');
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const allHistory = priceData?.history || [];

  // Filter history based on selected time range
  const displayedHistory = useMemo(() => {
    if (!allHistory || allHistory.length === 0) return [];
    switch (timeRange) {
      case '1M':
        return allHistory.slice(-22);
      case '3M':
        return allHistory.slice(-66);
      case '6M':
        return allHistory.slice(-130);
      case 'ALL':
      default:
        return allHistory;
    }
  }, [allHistory, timeRange]);

  if (loading || !priceData || allHistory.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel" style={{ height: '140px' }} />
        <div className="ledger-panel" style={{ height: '420px' }} />
      </div>
    );
  }

  const latest = allHistory[allHistory.length - 1];
  const prev = allHistory[allHistory.length - 2] || latest;
  const dayChangePct = ((latest.close - prev.close) / prev.close) * 100;
  const dayChangeVal = latest.close - prev.close;

  const return1w = ((latest.close - allHistory[Math.max(0, allHistory.length - 6)].close) / allHistory[Math.max(0, allHistory.length - 6)].close) * 100;
  const return1m = ((latest.close - allHistory[Math.max(0, allHistory.length - 22)].close) / allHistory[Math.max(0, allHistory.length - 22)].close) * 100;

  // 180D / all-time range calculations
  const allLows = allHistory.map(h => h.low || h.close);
  const allHighs = allHistory.map(h => h.high || h.close);
  const periodLow = Math.min(...allLows);
  const periodHigh = Math.max(...allHighs);
  const rangeSpan = periodHigh - periodLow || 1;
  const currentPosPct = Math.min(100, Math.max(0, ((latest.close - periodLow) / rangeSpan) * 100));

  // Chart dimensions & scaling for displayed slice
  const width = 920;
  const totalHeight = 380;
  const priceAreaHeight = 290;
  const volumeAreaHeight = 70;
  const paddingLeft = 10;
  const paddingRight = 65;
  const usableWidth = width - paddingLeft - paddingRight;

  const dispLows = displayedHistory.map(h => h.low || h.close);
  const dispHighs = displayedHistory.map(h => h.high || h.close);
  const minLow = Math.min(...dispLows);
  const maxHigh = Math.max(...dispHighs);
  const pricePad = (maxHigh - minLow) * 0.08 || 1;
  const minPrice = minLow - pricePad;
  const maxPrice = maxHigh + pricePad;
  const priceRange = maxPrice - minPrice || 1;

  const maxVol = Math.max(...displayedHistory.map(h => h.volume || 0)) || 1;

  const numCandles = displayedHistory.length;
  const candleSlotWidth = usableWidth / Math.max(1, numCandles);
  const candleBodyWidth = Math.max(2.5, Math.min(16, candleSlotWidth * 0.7));

  // Active point for HUD / Crosshair
  const activePoint: PricePoint = hoverIndex !== null && hoverIndex >= 0 && hoverIndex < displayedHistory.length
    ? displayedHistory[hoverIndex]
    : latest;

  const activeIndex = hoverIndex !== null ? hoverIndex : displayedHistory.length - 1;
  const activeX = paddingLeft + (activeIndex + 0.5) * candleSlotWidth;
  const activeY = priceAreaHeight - ((activePoint.close - minPrice) / priceRange) * priceAreaHeight;
  const activeCandleChange = ((activePoint.close - activePoint.open) / (activePoint.open || 1)) * 100;

  // SVG points for fallback line mode
  const linePoints = displayedHistory.map((h, i) => {
    const x = paddingLeft + (i + 0.5) * candleSlotWidth;
    const y = priceAreaHeight - ((h.close - minPrice) / priceRange) * priceAreaHeight;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const areaPolygon = `${paddingLeft},${priceAreaHeight} ${linePoints} ${paddingLeft + (displayedHistory.length - 0.5) * candleSlotWidth},${priceAreaHeight}`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Stock Overview Header */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h2 style={{ fontFamily: 'var(--display)', fontSize: '26px', fontWeight: '700', color: 'var(--text-primary)', letterSpacing: '-0.02em', margin: 0 }}>
                {priceData.name}
              </h2>
              <span style={{
                fontFamily: 'var(--mono)',
                fontSize: '11px',
                color: '#475569',
                background: '#f1f5f9',
                padding: '3px 8px',
                borderRadius: '6px',
                border: '1px solid #e2e8f0',
                fontWeight: '600'
              }}>
                {priceData.ticker}
              </span>
              <span style={{
                fontFamily: 'var(--mono)',
                fontSize: '11px',
                color: '#16a34a',
                background: '#ecfdf5',
                padding: '3px 8px',
                borderRadius: '6px',
                border: '1px solid #a7f3d0',
                fontWeight: '600'
              }}>
                NSE
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '14px', marginTop: '12px' }}>
              <span style={{ fontSize: '36px', fontWeight: '700', fontFamily: 'var(--mono)', color: 'var(--text-primary)', letterSpacing: '-0.03em' }}>
                ₹{latest.close.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span style={{
                fontSize: '13px',
                fontFamily: 'var(--mono)',
                fontWeight: '700',
                padding: '4px 10px',
                borderRadius: '8px',
                color: dayChangePct >= 0 ? '#15803d' : '#b91c1c',
                background: dayChangePct >= 0 ? '#ecfdf5' : '#fef2f2',
                border: `1px solid ${dayChangePct >= 0 ? '#a7f3d0' : '#fecaca'}`,
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                <span>{dayChangePct >= 0 ? '▲' : '▼'}</span>
                <span>{dayChangeVal >= 0 ? `+₹${dayChangeVal.toFixed(2)}` : `-₹${Math.abs(dayChangeVal).toFixed(2)}`}</span>
                <span>({dayChangePct >= 0 ? `+${dayChangePct.toFixed(2)}%` : `${dayChangePct.toFixed(2)}%`})</span>
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px' }}>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: '11px',
              padding: '4px 12px',
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '20px',
              color: '#334155',
              fontWeight: '600',
              letterSpacing: '0.04em'
            }}>
              MARKET REGIME: <strong style={{ color: '#0f172a' }}>{(latest.regime || 'trending').replace(/_/g, ' ').toUpperCase()}</strong>
            </div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: '#64748b' }}>
              Sector: <strong style={{ color: '#0f172a' }}>{priceData.sector}</strong> · Universe: <strong style={{ color: '#0f172a' }}>{priceData.universe}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Non-repetitive Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
        {/* Card 1: 180-Day Range with Gauge (Replaces repetitive 1D return) */}
        <div className="ledger-cell" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="ledger-label">180-Day Range & Position</div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: '6px' }}>
              <span style={{ fontSize: '12px', fontFamily: 'var(--mono)', color: '#64748b' }}>
                L: <strong style={{ color: '#0f172a' }}>₹{periodLow.toFixed(1)}</strong>
              </span>
              <span style={{ fontSize: '12px', fontFamily: 'var(--mono)', color: '#64748b' }}>
                H: <strong style={{ color: '#0f172a' }}>₹{periodHigh.toFixed(1)}</strong>
              </span>
            </div>
          </div>
          <div style={{ marginTop: '10px' }}>
            <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px', position: 'relative', overflow: 'hidden' }}>
              <div style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${currentPosPct}%`,
                background: 'linear-gradient(90deg, #16a34a, #22c55e)',
                borderRadius: '3px'
              }} />
            </div>
            <div className="ledger-note" style={{ marginTop: '6px', textAlign: 'right' }}>
              At {currentPosPct.toFixed(0)}% of 180D band
            </div>
          </div>
        </div>

        {/* Card 2: 1-Week Momentum */}
        <div className="ledger-cell">
          <div className="ledger-label">1-Week Momentum</div>
          <div className={`ledger-metric ${return1w >= 0 ? 'ledger-metric-brass' : 'ledger-metric-rust'}`} style={{ marginTop: '4px' }}>
            {return1w >= 0 ? `+${return1w.toFixed(2)}%` : `${return1w.toFixed(2)}%`}
          </div>
          <div className="ledger-note" style={{ marginTop: '4px' }}>5-day rolling return</div>
        </div>

        {/* Card 3: 1-Month Trend */}
        <div className="ledger-cell">
          <div className="ledger-label">1-Month Trend</div>
          <div className={`ledger-metric ${return1m >= 0 ? 'ledger-metric-brass' : 'ledger-metric-rust'}`} style={{ marginTop: '4px' }}>
            {return1m >= 0 ? `+${return1m.toFixed(2)}%` : `${return1m.toFixed(2)}%`}
          </div>
          <div className="ledger-note" style={{ marginTop: '4px' }}>21 trading sessions return</div>
        </div>
      </div>

      {/* Chart Canvas Panel */}
      <div className="ledger-panel" style={{ padding: '20px' }}>
        {/* Chart Controls Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ fontFamily: 'var(--display)', fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              Historical Price & Volume Action
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
              Candlestick price discovery · Volume distribution below
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Chart Type Toggle: Candles vs Line */}
            <div style={{
              display: 'inline-flex',
              background: '#f1f5f9',
              padding: '3px',
              borderRadius: '8px',
              border: '1px solid #e2e8f0'
            }}>
              <button
                type="button"
                onClick={() => setChartStyle('candles')}
                style={{
                  border: 'none',
                  background: chartStyle === 'candles' ? '#ffffff' : 'transparent',
                  color: chartStyle === 'candles' ? '#0f172a' : '#64748b',
                  fontSize: '11px',
                  fontFamily: 'var(--mono)',
                  fontWeight: '700',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  boxShadow: chartStyle === 'candles' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                  transition: 'all 0.15s ease'
                }}
              >
                🕯️ Candles
              </button>
              <button
                type="button"
                onClick={() => setChartStyle('line')}
                style={{
                  border: 'none',
                  background: chartStyle === 'line' ? '#ffffff' : 'transparent',
                  color: chartStyle === 'line' ? '#0f172a' : '#64748b',
                  fontSize: '11px',
                  fontFamily: 'var(--mono)',
                  fontWeight: '700',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  boxShadow: chartStyle === 'line' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                  transition: 'all 0.15s ease'
                }}
              >
                📈 Line
              </button>
            </div>

            {/* Timeframe Range Selector */}
            <div style={{
              display: 'inline-flex',
              background: '#f1f5f9',
              padding: '3px',
              borderRadius: '8px',
              border: '1px solid #e2e8f0'
            }}>
              {(['1M', '3M', '6M', 'ALL'] as const).map(range => (
                <button
                  key={range}
                  type="button"
                  onClick={() => setTimeRange(range)}
                  style={{
                    border: 'none',
                    background: timeRange === range ? '#2d6a2e' : 'transparent',
                    color: timeRange === range ? '#ffffff' : '#64748b',
                    fontSize: '11px',
                    fontFamily: 'var(--mono)',
                    fontWeight: '700',
                    padding: '4px 9px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    boxShadow: timeRange === range ? '0 2px 6px rgba(45, 106, 46, 0.25)' : 'none',
                    transition: 'all 0.15s ease'
                  }}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Floating HUD Bar on Hover / Active Candle */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          padding: '8px 14px',
          background: '#f8fafc',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          marginBottom: '12px',
          fontFamily: 'var(--mono)',
          fontSize: '11px',
          color: '#475569',
          flexWrap: 'wrap'
        }}>
          <div>
            Date: <strong style={{ color: '#0f172a' }}>{activePoint.date}</strong>
          </div>
          <div>
            O: <strong style={{ color: '#0f172a' }}>₹{activePoint.open?.toFixed(2)}</strong>
          </div>
          <div>
            H: <strong style={{ color: '#16a34a' }}>₹{activePoint.high?.toFixed(2)}</strong>
          </div>
          <div>
            L: <strong style={{ color: '#dc2626' }}>₹{activePoint.low?.toFixed(2)}</strong>
          </div>
          <div>
            C: <strong style={{ color: '#0f172a' }}>₹{activePoint.close?.toFixed(2)}</strong>
          </div>
          <div>
            Chg: <strong style={{ color: activeCandleChange >= 0 ? '#16a34a' : '#dc2626' }}>
              {activeCandleChange >= 0 ? `+${activeCandleChange.toFixed(2)}%` : `${activeCandleChange.toFixed(2)}%`}
            </strong>
          </div>
          {activePoint.volume > 0 && (
            <div>
              Vol: <strong style={{ color: '#475569' }}>{(activePoint.volume / 1000000).toFixed(2)}M</strong>
            </div>
          )}
          {hoverIndex !== null && (
            <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#16a34a', fontWeight: '600' }}>
              ● INSPECTING CANDLE
            </span>
          )}
        </div>

        {/* SVG Canvas with Candlesticks & Volume */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: `${totalHeight}px`,
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            overflow: 'hidden',
            boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.02)'
          }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = (e.clientX - rect.left - paddingLeft) / usableWidth;
            const idx = Math.min(displayedHistory.length - 1, Math.max(0, Math.floor(relX * displayedHistory.length)));
            setHoverIndex(idx);
          }}
          onMouseLeave={() => setHoverIndex(null)}
        >
          <svg
            viewBox={`0 0 ${width} ${totalHeight}`}
            style={{ width: '100%', height: '100%', display: 'block' }}
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2d6a2e" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#2d6a2e" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Horizontal Gridlines & Price Labels */}
            {[0.15, 0.35, 0.55, 0.75].map((ratio, i) => {
              const y = priceAreaHeight * ratio;
              const priceVal = maxPrice - (ratio * priceRange);
              return (
                <g key={i}>
                  <line
                    x1={paddingLeft}
                    y1={y}
                    x2={width - paddingRight}
                    y2={y}
                    stroke="#f1f5f9"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                  />
                  <text
                    x={width - paddingRight + 8}
                    y={y + 3}
                    fill="#94a3b8"
                    fontSize="10"
                    fontFamily="var(--mono)"
                    fontWeight="500"
                  >
                    ₹{priceVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Separator line between Price and Volume areas */}
            <line
              x1={paddingLeft}
              y1={priceAreaHeight}
              x2={width - paddingRight}
              y2={priceAreaHeight}
              stroke="#e2e8f0"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
            <text
              x={width - paddingRight + 8}
              y={priceAreaHeight + 14}
              fill="#cbd5e1"
              fontSize="9"
              fontFamily="var(--mono)"
            >
              VOL
            </text>

            {/* ───── Candlestick Mode ───── */}
            {chartStyle === 'candles' && (
              <g id="candlestick-group">
                {displayedHistory.map((point, i) => {
                  const x = paddingLeft + (i + 0.5) * candleSlotWidth;
                  const isBullish = (point.close >= point.open);
                  const strokeColor = isBullish ? '#16a34a' : '#dc2626';
                  const fillColor = isBullish ? '#22c55e' : '#ef4444';

                  const yHigh = priceAreaHeight - ((point.high - minPrice) / priceRange) * priceAreaHeight;
                  const yLow = priceAreaHeight - ((point.low - minPrice) / priceRange) * priceAreaHeight;
                  const yOpen = priceAreaHeight - ((point.open - minPrice) / priceRange) * priceAreaHeight;
                  const yClose = priceAreaHeight - ((point.close - minPrice) / priceRange) * priceAreaHeight;

                  const bodyY = Math.min(yOpen, yClose);
                  const bodyHeight = Math.max(2, Math.abs(yClose - yOpen));

                  // Volume bar
                  const volHeight = (point.volume / maxVol) * volumeAreaHeight;
                  const volY = totalHeight - volHeight - 4;

                  const isHovered = hoverIndex === i;

                  return (
                    <g key={i} opacity={hoverIndex !== null && !isHovered ? 0.75 : 1}>
                      {/* Volume Bar */}
                      <rect
                        x={x - candleBodyWidth / 2}
                        y={volY}
                        width={candleBodyWidth}
                        height={volHeight}
                        fill={isBullish ? 'rgba(34, 197, 94, 0.28)' : 'rgba(239, 68, 68, 0.28)'}
                        stroke={isBullish ? 'rgba(22, 163, 74, 0.6)' : 'rgba(220, 38, 38, 0.6)'}
                        strokeWidth="0.8"
                        rx="1"
                      />

                      {/* Upper & Lower Wick */}
                      <line
                        x1={x}
                        y1={yHigh}
                        x2={x}
                        y2={yLow}
                        stroke={strokeColor}
                        strokeWidth={isHovered ? 2 : 1.2}
                        shapeRendering="crispEdges"
                      />

                      {/* Candle Body */}
                      <rect
                        x={x - candleBodyWidth / 2}
                        y={bodyY}
                        width={candleBodyWidth}
                        height={bodyHeight}
                        fill={fillColor}
                        stroke={strokeColor}
                        strokeWidth={isHovered ? 2 : 1}
                        rx="1.5"
                      />
                    </g>
                  );
                })}
              </g>
            )}

            {/* ───── Line Mode ───── */}
            {chartStyle === 'line' && (
              <g id="line-group">
                {/* Area Gradient Fill */}
                <polygon fill="url(#lineGradient)" points={areaPolygon} />

                {/* Polyline */}
                <polyline
                  fill="none"
                  stroke="#2d6a2e"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={linePoints}
                />

                {/* Volume Bars */}
                {displayedHistory.map((point, i) => {
                  const x = paddingLeft + (i + 0.5) * candleSlotWidth;
                  const volHeight = (point.volume / maxVol) * volumeAreaHeight;
                  const volY = totalHeight - volHeight - 4;
                  return (
                    <rect
                      key={i}
                      x={x - candleBodyWidth / 2}
                      y={volY}
                      width={candleBodyWidth}
                      height={volHeight}
                      fill="rgba(45, 106, 46, 0.2)"
                      rx="1"
                    />
                  );
                })}
              </g>
            )}

            {/* Crosshairs & Target Marker */}
            {hoverIndex !== null && (
              <g id="crosshairs">
                {/* Vertical Crosshair */}
                <line
                  x1={activeX}
                  y1={0}
                  x2={activeX}
                  y2={totalHeight}
                  stroke="#2d6a2e"
                  strokeWidth="1.2"
                  strokeDasharray="3 3"
                  opacity="0.6"
                />

                {/* Horizontal Crosshair at Close Level */}
                <line
                  x1={paddingLeft}
                  y1={activeY}
                  x2={width - paddingRight}
                  y2={activeY}
                  stroke="#2d6a2e"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                  opacity="0.4"
                />

                {/* Price Label on Right Axis */}
                <rect
                  x={width - paddingRight + 4}
                  y={activeY - 9}
                  width="55"
                  height="18"
                  fill="#2d6a2e"
                  rx="4"
                />
                <text
                  x={width - paddingRight + 8}
                  y={activeY + 3}
                  fill="#ffffff"
                  fontSize="9"
                  fontFamily="var(--mono)"
                  fontWeight="700"
                >
                  ₹{activePoint.close.toFixed(1)}
                </text>

                {/* Pin Circle Marker on Line Mode */}
                {chartStyle === 'line' && (
                  <>
                    <circle
                      cx={activeX}
                      cy={activeY}
                      r="8"
                      fill="rgba(45, 106, 46, 0.2)"
                    />
                    <circle
                      cx={activeX}
                      cy={activeY}
                      r="4"
                      fill="#2d6a2e"
                      stroke="#ffffff"
                      strokeWidth="2"
                    />
                  </>
                )}
              </g>
            )}
          </svg>
        </div>

        {/* Footer Coordinate Summary */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '14px', fontFamily: 'var(--mono)', fontSize: '11px', color: '#64748b' }}>
          <div>
            Showing <strong style={{ color: '#0f172a' }}>{displayedHistory.length}</strong> daily candles ({displayedHistory[0]?.date} → {latest.date})
          </div>
          <div>
            Range Low: <strong style={{ color: '#0f172a' }}>₹{minLow.toFixed(1)}</strong> · Range High: <strong style={{ color: '#0f172a' }}>₹{maxHigh.toFixed(1)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
};
