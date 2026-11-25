import React, { useState, useMemo } from 'react';
import type { StockPriceData, PricePoint, SignalData, UniverseStock } from '../types';

interface PriceChartViewProps {
  priceData: StockPriceData | null;
  loading: boolean;
  switchingTicker?: string | null;
  topStocks?: SignalData[];
  universeStocks?: UniverseStock[];
  selectedTicker?: string;
  onSelectTicker?: (ticker: string) => void;
}

type TimeRange = '1M' | '3M' | '6M' | 'ALL';
type ChartStyle = 'candles' | 'line';

export const PriceChartView: React.FC<PriceChartViewProps> = ({
  priceData,
  loading,
  switchingTicker,
  topStocks = [],
  universeStocks = [],
  selectedTicker,
  onSelectTicker,
}) => {
  const [chartStyle, setChartStyle] = useState<ChartStyle>('candles');
  const [timeRange, setTimeRange] = useState<TimeRange>('3M');
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [showSma20, setShowSma20] = useState<boolean>(true);
  const [showSma50, setShowSma50] = useState<boolean>(true);
  const [showBollinger, setShowBollinger] = useState<boolean>(false);

  const allHistory = useMemo(() => priceData?.history || [], [priceData?.history]);

  // ── Sparkline helper: deterministic 9-pt decorative mini-chart ──────────
  function buildSparkline(dayChangePct: number, signal: string, ret1m: number | null): string {
    const trend = signal === 'BUY' ? 1 : signal === 'AVOID' ? -1 : 0;
    const seed = Math.abs(dayChangePct * 1000 + (ret1m ?? 0) * 100) + 1;
    const pts: number[] = [];
    let val = 50;
    for (let i = 0; i < 9; i++) {
      const noise = ((seed * (i + 7) * 2654435761) % 97) / 97 - 0.5;
      val = Math.min(90, Math.max(10, val + trend * 5 + noise * 14 + dayChangePct * 0.3));
      pts.push(val);
    }
    return pts.map((v, i) => `${(i * 32).toFixed(0)},${(100 - v).toFixed(0)}`).join(' ');
  }

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

  // Show skeleton only on initial load (priceData is null), not during ticker switches
  if (loading && !priceData) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel skeleton" style={{ height: '140px' }} />
        <div className="ledger-panel skeleton" style={{ height: '420px' }} />
      </div>
    );
  }

  if (!priceData || allHistory.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel skeleton" style={{ height: '140px' }} />
        <div className="ledger-panel skeleton" style={{ height: '420px' }} />
      </div>
    );
  }

  const latest = allHistory[allHistory.length - 1];
  const prev = allHistory[allHistory.length - 2] || latest;

  const currentPrice = priceData.current_price ?? latest.close;
  const dayChangePct = priceData.day_change_pct !== undefined && priceData.day_change_pct !== null
    ? priceData.day_change_pct
    : ((latest.close - prev.close) / (prev.close || 1)) * 100;
  const dayChangeVal = prev && prev.close && priceData.current_price
    ? priceData.current_price - prev.close
    : (latest.close - prev.close);

  const return1w = ((currentPrice - allHistory[Math.max(0, allHistory.length - 6)].close) / allHistory[Math.max(0, allHistory.length - 6)].close) * 100;
  const return1m = ((currentPrice - allHistory[Math.max(0, allHistory.length - 22)].close) / allHistory[Math.max(0, allHistory.length - 22)].close) * 100;

  const allLows = allHistory.map(h => h.low || h.close);
  const allHighs = allHistory.map(h => h.high || h.close);
  const periodLow = Math.min(...allLows);
  const periodHigh = Math.max(...allHighs);
  const rangeSpan = periodHigh - periodLow || 1;
  const currentPosPct = Math.min(100, Math.max(0, ((currentPrice - periodLow) / rangeSpan) * 100));

  const width = 920;
  const totalHeight = 380;
  const priceAreaHeight = 290;
  const volumeAreaHeight = 70;
  const paddingLeft = 10;
  const paddingRight = 65;
  const usableWidth = width - paddingLeft - paddingRight;

  const dispLows = displayedHistory.map(h => showBollinger ? Math.min(h.low || h.close, h.bb_lower || h.close) : (h.low || h.close));
  const dispHighs = displayedHistory.map(h => showBollinger ? Math.max(h.high || h.close, h.bb_upper || h.close) : (h.high || h.close));
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

  const activePoint: PricePoint = hoverIndex !== null && hoverIndex >= 0 && hoverIndex < displayedHistory.length
    ? displayedHistory[hoverIndex]
    : latest;

  const activeIndex = hoverIndex !== null ? hoverIndex : displayedHistory.length - 1;
  const activeX = paddingLeft + (activeIndex + 0.5) * candleSlotWidth;
  const activeY = priceAreaHeight - ((activePoint.close - minPrice) / priceRange) * priceAreaHeight;
  const activeCandleChange = ((activePoint.close - activePoint.open) / (activePoint.open || 1)) * 100;

  const toPricePoints = (getter: (p: PricePoint) => number) =>
    displayedHistory.map((h, i) => {
      const x = paddingLeft + (i + 0.5) * candleSlotWidth;
      const y = priceAreaHeight - ((getter(h) - minPrice) / priceRange) * priceAreaHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

  const linePoints = toPricePoints((h) => h.close);
  const sma20Points = toPricePoints((h) => h.sma_20 || h.close);
  const sma50Points = toPricePoints((h) => h.sma_50 || h.close);
  const bbUpperPoints = toPricePoints((h) => h.bb_upper || h.close);
  const bbLowerPoints = toPricePoints((h) => h.bb_lower || h.close);
  const areaPolygon = `${paddingLeft},${priceAreaHeight} ${linePoints} ${paddingLeft + (displayedHistory.length - 0.5) * candleSlotWidth},${priceAreaHeight}`;

  const rsiColor = latest.rsi_14 >= 70 ? '#dc2626' : (latest.rsi_14 <= 30 ? '#16a34a' : '#0f172a');
  const macdBullish = latest.macd_line_12_26 >= latest.macd_signal;

  const dayChangeLookup = new Map<string, number>();
  universeStocks.forEach((s) => dayChangeLookup.set(s.ticker, s.day_change_pct));

  const topTenRows = topStocks.length > 0
    ? topStocks.slice(0, 10).map((s, idx) => ({
      rank: idx + 1,
      ticker: s.ticker,
      name: s.name,
      price: s.current_price,
      dayChangePct: dayChangeLookup.get(s.ticker) ?? 0,
      signal: s.signal,
      score: s.signal_score,
      ret1m: s.expected_return_1M,
    }))
    : universeStocks.slice(0, 10).map((s, idx) => ({
      rank: idx + 1,
      ticker: s.ticker,
      name: s.name,
      price: s.current_price,
      dayChangePct: s.day_change_pct,
      signal: 'HOLD' as const,
      score: null as number | null,
      ret1m: null as number | null,
    }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* ── Top 10 Ranked Stocks — Interactive Horizontal Strip ── */}
      {topTenRows.length > 0 && (
        <div className="ledger-panel" style={{ padding: '14px 16px', position: 'relative', overflow: 'hidden' }}>
          {/* Switching overlay — subtle shimmer across the whole strip */}
          {switchingTicker && (
            <div style={{
              position: 'absolute', inset: 0, zIndex: 10,
              background: 'rgba(245,244,240,0.55)',
              backdropFilter: 'blur(1px)',
              pointerEvents: 'none',
              borderRadius: 'inherit',
            }} />
          )}

          {/* Header row */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <div>
              <span style={{ fontFamily: 'var(--sans)', fontSize: '14px', fontWeight: '700', color: 'var(--ink)' }}>
                Top 10 Ranked Equities
              </span>
            </div>
            <span style={{
              fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--accent)',
              background: 'var(--accent-subtle)', border: '1px solid var(--accent-border)',
              padding: '2px 8px', borderRadius: '20px', fontWeight: '700',
            }}>
              SIGNAL RANKED
            </span>
          </div>

          {/* Horizontal scrollable strip */}
          <div className="top10-strip">
            {topTenRows.map((row, animIdx) => {
              const isSelected = selectedTicker === row.ticker;
              const isSwitching = switchingTicker === row.ticker;
              const isDayUp = row.dayChangePct >= 0;

              const pastelThemes = [
                { bg: 'linear-gradient(145deg, #FFF5F5 0%, #FFE8E8 100%)', border: '#F8CACA', orb: 'rgba(239, 68, 68, 0.10)', rankBg: '#9B2C2C' },
                { bg: 'linear-gradient(145deg, #FFF8EB 0%, #FFEDD5 100%)', border: '#F6CFA0', orb: 'rgba(245, 158, 11, 0.11)', rankBg: '#9C4221' },
                { bg: 'linear-gradient(145deg, #FEFCE8 0%, #FEF3C7 100%)', border: '#EED98B', orb: 'rgba(234, 179, 8, 0.11)', rankBg: '#854D0E' },
                { bg: 'linear-gradient(145deg, #F0FDF4 0%, #DCFCE7 100%)', border: '#A7E3BC', orb: 'rgba(34, 197, 94, 0.10)', rankBg: '#166534' },
                { bg: 'linear-gradient(145deg, #F0FDFA 0%, #CCFBF1 100%)', border: '#99E2D4', orb: 'rgba(20, 184, 166, 0.10)', rankBg: '#115E59' },
                { bg: 'linear-gradient(145deg, #EFF6FF 0%, #DBEAFE 100%)', border: '#AFCBF7', orb: 'rgba(59, 130, 246, 0.10)', rankBg: '#1E40AF' },
                { bg: 'linear-gradient(145deg, #EEF2FF 0%, #E0E7FF 100%)', border: '#B8C4F7', orb: 'rgba(99, 102, 241, 0.10)', rankBg: '#3730A3' },
                { bg: 'linear-gradient(145deg, #F5F3FF 0%, #EDE9FE 100%)', border: '#C9BDF8', orb: 'rgba(139, 92, 246, 0.10)', rankBg: '#5B21B6' },
                { bg: 'linear-gradient(145deg, #FDF2F8 0%, #FCE7F3 100%)', border: '#F2B8D8', orb: 'rgba(236, 72, 153, 0.10)', rankBg: '#9D174D' },
                { bg: 'linear-gradient(145deg, #F8FAFC 0%, #E2E8F0 100%)', border: '#CBD5E1', orb: 'rgba(100, 116, 139, 0.10)', rankBg: '#334155' },
              ];
              const theme = pastelThemes[animIdx % pastelThemes.length];

              const sigStyle =
                row.signal === 'BUY'
                  ? { bg: 'var(--gain-bg)', color: 'var(--gain)', border: 'var(--gain-border)', glow: 'rgba(26,127,75,0.2)' }
                  : row.signal === 'AVOID'
                    ? { bg: 'var(--loss-bg)', color: 'var(--loss)', border: 'var(--loss-border)', glow: 'rgba(192,57,43,0.2)' }
                    : { bg: 'var(--hold-bg)', color: 'var(--hold)', border: 'var(--hold-border)', glow: 'rgba(148,108,0,0.2)' };

              const sparkPts = buildSparkline(row.dayChangePct, row.signal, row.ret1m);
              const sparkColor = isDayUp ? 'var(--gain)' : 'var(--loss)';

              return (
                <button
                  key={row.ticker}
                  type="button"
                  className={`top10-card${isSelected ? ' active' : ''}${isSwitching ? ' switching' : ''}`}
                  style={{
                    animationDelay: `${animIdx * 40}ms`,
                    background: theme.bg,
                    borderColor: isSelected ? 'var(--accent)' : theme.border,
                  }}
                  onClick={() => onSelectTicker?.(row.ticker)}
                  aria-pressed={isSelected}
                  aria-label={`Select ${row.ticker.replace('.NS', '')} — ${row.name}`}
                >
                  {/* Decorative pastel corner orb */}
                  <span
                    aria-hidden="true"
                    style={{
                      position: 'absolute',
                      top: '-18px',
                      right: '-18px',
                      width: '64px',
                      height: '64px',
                      borderRadius: '50%',
                      background: theme.orb,
                      pointerEvents: 'none',
                    }}
                  />

                  {/* Rank badge */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', position: 'relative' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                      <span className="top10-rank" style={{ background: isSelected ? 'var(--accent)' : theme.rankBg }}>
                        #{row.rank}
                      </span>
                      {isSelected && (
                        <span className="top10-active-dot" title="Currently selected" />
                      )}
                    </div>
                    <span style={{
                      fontFamily: 'var(--mono)', fontSize: '9px', fontWeight: '800',
                      padding: '2px 6px', borderRadius: '4px',
                      background: sigStyle.bg, color: sigStyle.color,
                      border: `1px solid ${sigStyle.border}`,
                      letterSpacing: '0.04em',
                    }}>
                      {row.signal}
                    </span>
                  </div>

                  {/* Ticker & name */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: '14px', fontWeight: '800', color: 'var(--ink)', letterSpacing: '-0.01em' }}>
                      {row.ticker.replace('.NS', '')}
                    </div>
                    <div style={{
                      fontFamily: 'var(--sans)', fontSize: '10px', color: 'var(--dash-muted)',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      maxWidth: '140px',
                    }}>
                      {row.name}
                    </div>
                  </div>

                  {/* Sparkline SVG */}
                  <div className="top10-sparkline">
                    <svg viewBox="0 0 256 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block' }}>
                      <defs>
                        <linearGradient id={`spk-grad-${row.rank}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={isDayUp ? '#1A7F4B' : '#C0392B'} stopOpacity="0.18" />
                          <stop offset="100%" stopColor={isDayUp ? '#1A7F4B' : '#C0392B'} stopOpacity="0.0" />
                        </linearGradient>
                      </defs>
                      <polyline
                        fill="none"
                        stroke={sparkColor}
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        points={sparkPts}
                        opacity="0.9"
                      />
                    </svg>
                  </div>

                  {/* Price row */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: '4px' }}>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: '12px', fontWeight: '700', color: 'var(--ink)' }}>
                      ₹{row.price.toLocaleString('en-IN', { maximumFractionDigits: 1 })}
                    </span>
                    <span style={{
                      fontFamily: 'var(--mono)', fontSize: '11px', fontWeight: '700',
                      color: isDayUp ? 'var(--gain)' : 'var(--loss)',
                    }}>
                      {isDayUp ? '▲' : '▼'}{Math.abs(row.dayChangePct).toFixed(2)}%
                    </span>
                  </div>

                  {/* 1M expected return chip */}
                  {row.ret1m != null && (
                    <div style={{ marginTop: '5px' }}>
                      <span style={{
                        fontFamily: 'var(--mono)', fontSize: '9px', fontWeight: '700',
                        padding: '1px 5px', borderRadius: '4px',
                        background: row.ret1m >= 0 ? '#EBF5EE' : '#FDECEA',
                        color: row.ret1m >= 0 ? 'var(--gain)' : 'var(--loss)',
                        border: `1px solid ${row.ret1m >= 0 ? 'var(--gain-border)' : 'var(--loss-border)'}`,
                      }}>
                        {row.ret1m >= 0 ? '+' : ''}{(row.ret1m * 100).toFixed(1)}% 1M
                      </span>
                    </div>
                  )}

                  {/* Loading spinner overlay on the active card during switch */}
                  {isSwitching && (
                    <div className="top10-card-spinner" aria-hidden="true">
                      <div className="top10-spinner-dot" />
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Stock Overview Header */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
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
              <span style={{
                fontFamily: 'var(--mono)',
                fontSize: '10px',
                color: priceData.quote_status === 'LIVE' ? '#15803d' : '#64748b',
                background: priceData.quote_status === 'LIVE' ? '#f0fdf4' : '#f8fafc',
                padding: '3px 8px',
                borderRadius: '6px',
                border: `1px solid ${priceData.quote_status === 'LIVE' ? '#bbf7d0' : '#e2e8f0'}`,
                fontWeight: '700',
              }}>
                {priceData.quote_status === 'LIVE' ? 'LIVE QUOTE' : `LAST CLOSE · ${priceData.quote_updated_at || latest.date}`}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '14px', marginTop: '12px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '36px', fontWeight: '700', fontFamily: 'var(--mono)', color: 'var(--text-primary)', letterSpacing: '-0.03em' }}>
                ₹{currentPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
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

      {/* Technical & Range Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '14px' }}>
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
              At {currentPosPct.toFixed(0)}% of {allHistory.length}D band
            </div>
          </div>
        </div>

        <div className="ledger-cell">
          <div className="ledger-label">1-Week Momentum</div>
          <div className={`ledger-metric ${return1w >= 0 ? 'ledger-metric-brass' : 'ledger-metric-rust'}`} style={{ marginTop: '4px' }}>
            {return1w >= 0 ? `+${return1w.toFixed(2)}%` : `${return1w.toFixed(2)}%`}
          </div>
          <div className="ledger-note" style={{ marginTop: '4px' }}>5-session rolling return</div>
        </div>

        <div className="ledger-cell">
          <div className="ledger-label">1-Month Trend</div>
          <div className={`ledger-metric ${return1m >= 0 ? 'ledger-metric-brass' : 'ledger-metric-rust'}`} style={{ marginTop: '4px' }}>
            {return1m >= 0 ? `+${return1m.toFixed(2)}%` : `${return1m.toFixed(2)}%`}
          </div>
          <div className="ledger-note" style={{ marginTop: '4px' }}>21-session return</div>
        </div>

        <div className="ledger-cell">
          <div className="ledger-label">RSI (14) & MACD</div>
          <div style={{ fontFamily: 'var(--serif)', fontSize: '26px', color: rsiColor, marginTop: '4px', lineHeight: '1.1' }}>
            {latest.rsi_14.toFixed(1)}
          </div>
          <div className="ledger-note" style={{ marginTop: '4px', fontFamily: 'var(--mono)' }}>
            MACD: <strong style={{ color: macdBullish ? 'var(--gain)' : 'var(--loss)' }}>
              {latest.macd_line_12_26.toFixed(2)}
            </strong> (sig {latest.macd_signal.toFixed(2)})
          </div>
        </div>
      </div>

      {/* Chart Canvas Panel */}
      <div className="ledger-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ fontFamily: 'var(--display)', fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              Historical Price, Technical Overlays & Volume
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
              Candlestick price discovery · Toggleable SMA-20, SMA-50, and Bollinger Bands
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            {/* Overlay Toggles */}
            <button
              type="button"
              onClick={() => setShowSma20(v => !v)}
              style={{
                border: `1px solid ${showSma20 ? '#3B6FD4' : '#e2e8f0'}`,
                background: showSma20 ? '#EBF0FB' : '#ffffff',
                color: showSma20 ? '#3B6FD4' : '#64748b',
                fontSize: '11px',
                fontFamily: 'var(--mono)',
                fontWeight: '700',
                padding: '4px 8px',
                borderRadius: '6px',
                cursor: 'pointer',
              }}
            >
              SMA 20
            </button>
            <button
              type="button"
              onClick={() => setShowSma50(v => !v)}
              style={{
                border: `1px solid ${showSma50 ? '#D97706' : '#e2e8f0'}`,
                background: showSma50 ? '#FEF3C7' : '#ffffff',
                color: showSma50 ? '#B45309' : '#64748b',
                fontSize: '11px',
                fontFamily: 'var(--mono)',
                fontWeight: '700',
                padding: '4px 8px',
                borderRadius: '6px',
                cursor: 'pointer',
              }}
            >
              SMA 50
            </button>
            <button
              type="button"
              onClick={() => setShowBollinger(v => !v)}
              style={{
                border: `1px solid ${showBollinger ? '#7C3AED' : '#e2e8f0'}`,
                background: showBollinger ? '#F5F3FF' : '#ffffff',
                color: showBollinger ? '#7C3AED' : '#64748b',
                fontSize: '11px',
                fontFamily: 'var(--mono)',
                fontWeight: '700',
                padding: '4px 8px',
                borderRadius: '6px',
                cursor: 'pointer',
              }}
            >
              BB (20)
            </button>

            <div style={{ display: 'inline-flex', background: '#f1f5f9', padding: '3px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
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
                }}
              >
                Candles
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
                }}
              >
                Line
              </button>
            </div>

            <div style={{ display: 'inline-flex', background: '#f1f5f9', padding: '3px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
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
                  }}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Floating HUD Bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
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
          <div>Date: <strong style={{ color: '#0f172a' }}>{activePoint.date}</strong></div>
          <div>O: <strong style={{ color: '#0f172a' }}>₹{activePoint.open?.toFixed(2)}</strong></div>
          <div>H: <strong style={{ color: '#16a34a' }}>₹{activePoint.high?.toFixed(2)}</strong></div>
          <div>L: <strong style={{ color: '#dc2626' }}>₹{activePoint.low?.toFixed(2)}</strong></div>
          <div>C: <strong style={{ color: '#0f172a' }}>₹{activePoint.close?.toFixed(2)}</strong></div>
          <div>
            Chg:{' '}
            <strong style={{ color: activeCandleChange >= 0 ? '#16a34a' : '#dc2626' }}>
              {activeCandleChange >= 0 ? `+${activeCandleChange.toFixed(2)}%` : `${activeCandleChange.toFixed(2)}%`}
            </strong>
          </div>
          {showSma20 && <div>SMA20: <strong style={{ color: '#3B6FD4' }}>₹{activePoint.sma_20?.toFixed(1)}</strong></div>}
          {showSma50 && <div>SMA50: <strong style={{ color: '#B45309' }}>₹{activePoint.sma_50?.toFixed(1)}</strong></div>}
          {activePoint.volume > 0 && <div>Vol: <strong style={{ color: '#475569' }}>{(activePoint.volume / 1000000).toFixed(2)}M</strong></div>}
        </div>

        {/* SVG Canvas */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: `${totalHeight}px`,
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            overflow: 'hidden',
          }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = (e.clientX - rect.left - paddingLeft) / usableWidth;
            const idx = Math.min(displayedHistory.length - 1, Math.max(0, Math.floor(relX * displayedHistory.length)));
            setHoverIndex(idx);
          }}
          onMouseLeave={() => setHoverIndex(null)}
        >
          <svg viewBox={`0 0 ${width} ${totalHeight}`} style={{ width: '100%', height: '100%', display: 'block' }} preserveAspectRatio="none">
            <defs>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2d6a2e" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#2d6a2e" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {[0.15, 0.35, 0.55, 0.75].map((ratio, i) => {
              const y = priceAreaHeight * ratio;
              const priceVal = maxPrice - (ratio * priceRange);
              return (
                <g key={i}>
                  <line x1={paddingLeft} y1={y} x2={width - paddingRight} y2={y} stroke="#f1f5f9" strokeWidth="1" strokeDasharray="4 4" />
                  <text x={width - paddingRight + 8} y={y + 3} fill="#94a3b8" fontSize="10" fontFamily="var(--mono)" fontWeight="500">
                    ₹{priceVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            <line x1={paddingLeft} y1={priceAreaHeight} x2={width - paddingRight} y2={priceAreaHeight} stroke="#e2e8f0" strokeWidth="1" strokeDasharray="2 2" />
            <text x={width - paddingRight + 8} y={priceAreaHeight + 14} fill="#cbd5e1" fontSize="9" fontFamily="var(--mono)">VOL</text>

            {chartStyle === 'candles' && (
              <g>
                {displayedHistory.map((point, i) => {
                  const x = paddingLeft + (i + 0.5) * candleSlotWidth;
                  const isBullish = point.close >= point.open;
                  const strokeColor = isBullish ? '#16a34a' : '#dc2626';
                  const fillColor = isBullish ? '#22c55e' : '#ef4444';
                  const yHigh = priceAreaHeight - ((point.high - minPrice) / priceRange) * priceAreaHeight;
                  const yLow = priceAreaHeight - ((point.low - minPrice) / priceRange) * priceAreaHeight;
                  const yOpen = priceAreaHeight - ((point.open - minPrice) / priceRange) * priceAreaHeight;
                  const yClose = priceAreaHeight - ((point.close - minPrice) / priceRange) * priceAreaHeight;
                  const bodyY = Math.min(yOpen, yClose);
                  const bodyHeight = Math.max(2, Math.abs(yClose - yOpen));
                  const volHeight = (point.volume / maxVol) * volumeAreaHeight;
                  const volY = totalHeight - volHeight - 4;
                  const isHovered = hoverIndex === i;

                  return (
                    <g key={i} opacity={hoverIndex !== null && !isHovered ? 0.75 : 1}>
                      <rect
                        x={x - candleBodyWidth / 2}
                        y={volY}
                        width={candleBodyWidth}
                        height={volHeight}
                        fill={isBullish ? 'rgba(34, 197, 94, 0.28)' : 'rgba(239, 68, 68, 0.28)'}
                        rx="1"
                      />
                      <line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={strokeColor} strokeWidth={isHovered ? 2 : 1.2} />
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

            {chartStyle === 'line' && (
              <g>
                <polygon fill="url(#lineGradient)" points={areaPolygon} />
                <polyline fill="none" stroke="#2d6a2e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" points={linePoints} />
                {displayedHistory.map((point, i) => {
                  const x = paddingLeft + (i + 0.5) * candleSlotWidth;
                  const volHeight = (point.volume / maxVol) * volumeAreaHeight;
                  const volY = totalHeight - volHeight - 4;
                  return <rect key={i} x={x - candleBodyWidth / 2} y={volY} width={candleBodyWidth} height={volHeight} fill="rgba(45, 106, 46, 0.2)" rx="1" />;
                })}
              </g>
            )}

            {/* Technical Indicator Overlays */}
            {showBollinger && (
              <>
                <polyline fill="none" stroke="#7C3AED" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.65" points={bbUpperPoints} />
                <polyline fill="none" stroke="#7C3AED" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.65" points={bbLowerPoints} />
              </>
            )}
            {showSma20 && <polyline fill="none" stroke="#3B6FD4" strokeWidth="1.6" points={sma20Points} />}
            {showSma50 && <polyline fill="none" stroke="#D97706" strokeWidth="1.6" points={sma50Points} />}

            {hoverIndex !== null && (
              <g>
                <line x1={activeX} y1={0} x2={activeX} y2={totalHeight} stroke="#2d6a2e" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.6" />
                <line x1={paddingLeft} y1={activeY} x2={width - paddingRight} y2={activeY} stroke="#2d6a2e" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />
                <rect x={width - paddingRight + 4} y={activeY - 9} width="55" height="18" fill="#2d6a2e" rx="4" />
                <text x={width - paddingRight + 8} y={activeY + 3} fill="#ffffff" fontSize="9" fontFamily="var(--mono)" fontWeight="700">
                  ₹{activePoint.close.toFixed(1)}
                </text>
              </g>
            )}
          </svg>
        </div>

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
