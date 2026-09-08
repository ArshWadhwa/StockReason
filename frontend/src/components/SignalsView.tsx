import React, { useState } from 'react';
import type { SignalData, PredictionData } from '../types';

interface SignalsViewProps {
  signal: SignalData | null;
  prediction: PredictionData | null;
  loading: boolean;
}

// Category classification for visual grouping
const FEATURE_CATEGORIES: Record<string, { label: string; color: string }> = {
  // Momentum
  rsi_14:           { label: 'Momentum', color: '#3B6FD4' },
  rsi_7:            { label: 'Momentum', color: '#3B6FD4' },
  rsi_21:           { label: 'Momentum', color: '#3B6FD4' },
  rsi_28:           { label: 'Momentum', color: '#3B6FD4' },
  macd_line_12_26:  { label: 'Momentum', color: '#3B6FD4' },
  macd_hist_12_26:  { label: 'Momentum', color: '#3B6FD4' },
  macd_signal_12_26:{ label: 'Momentum', color: '#3B6FD4' },
  stoch_k_14:       { label: 'Momentum', color: '#3B6FD4' },
  stoch_d_14:       { label: 'Momentum', color: '#3B6FD4' },
  williams_r_14:    { label: 'Momentum', color: '#3B6FD4' },
  cci_14:           { label: 'Momentum', color: '#3B6FD4' },
  cci_20:           { label: 'Momentum', color: '#3B6FD4' },

  // Trend
  dist_sma_50:      { label: 'Trend', color: '#1A7F4B' },
  dist_sma_20:      { label: 'Trend', color: '#1A7F4B' },
  dist_sma_200:     { label: 'Trend', color: '#1A7F4B' },
  dist_ema_50:      { label: 'Trend', color: '#1A7F4B' },
  dist_ema_20:      { label: 'Trend', color: '#1A7F4B' },
  sma_50_200_ratio: { label: 'Trend', color: '#1A7F4B' },
  sma_20_50_ratio:  { label: 'Trend', color: '#1A7F4B' },
  adx_14:           { label: 'Trend', color: '#1A7F4B' },
  close_slope_20:   { label: 'Trend', color: '#1A7F4B' },
  close_slope_10:   { label: 'Trend', color: '#1A7F4B' },
  close_slope_5:    { label: 'Trend', color: '#1A7F4B' },

  // Volatility
  atr_14:           { label: 'Volatility', color: '#946C00' },
  natr_14:          { label: 'Volatility', color: '#946C00' },
  volatility_20d:   { label: 'Volatility', color: '#946C00' },
  volatility_5d:    { label: 'Volatility', color: '#946C00' },
  volatility_60d:   { label: 'Volatility', color: '#946C00' },
  bb_width_20:      { label: 'Volatility', color: '#946C00' },
  bb_pct_b_20:      { label: 'Volatility', color: '#946C00' },

  // Volume
  volume:           { label: 'Volume', color: '#6B52C8' },
  vol_ratio_20:     { label: 'Volume', color: '#6B52C8' },
  vol_ratio_5:      { label: 'Volume', color: '#6B52C8' },
  obv:              { label: 'Volume', color: '#6B52C8' },
  cmf_20:           { label: 'Volume', color: '#6B52C8' },
  pvt:              { label: 'Volume', color: '#6B52C8' },

  // Sentiment
  sentiment_mean:         { label: 'Sentiment', color: '#B0457E' },
  sentiment_pos_ratio:    { label: 'Sentiment', color: '#B0457E' },
  sentiment_neg_ratio:    { label: 'Sentiment', color: '#B0457E' },
  sentiment_momentum_3d:  { label: 'Sentiment', color: '#B0457E' },
  sentiment_momentum_7d:  { label: 'Sentiment', color: '#B0457E' },

  // Macro
  vix_close:              { label: 'Macro', color: '#C0392B' },
  vix_close_return:       { label: 'Macro', color: '#C0392B' },
  nifty_50_close:         { label: 'Macro', color: '#C0392B' },
  nifty_50_close_return:  { label: 'Macro', color: '#C0392B' },
  nifty_bank_close_return:{ label: 'Macro', color: '#C0392B' },

  // Returns
  return_1d:    { label: 'Returns', color: '#2E7D8C' },
  return_5d:    { label: 'Returns', color: '#2E7D8C' },
  return_21d:   { label: 'Returns', color: '#2E7D8C' },
  return_63d:   { label: 'Returns', color: '#2E7D8C' },
  close:        { label: 'Price', color: '#2E7D8C' },
  hl_spread_pct:{ label: 'Price', color: '#2E7D8C' },
};

// Returns null for uncategorized — chip is suppressed, not shown as "Other"
const getCategory = (feature: string) =>
  FEATURE_CATEGORIES[feature] ?? null;

const DEFAULT_VISIBLE = 3;

export const SignalsView: React.FC<SignalsViewProps> = ({ signal, prediction, loading }) => {
  const [expandedFeature, setExpandedFeature] = useState<number | null>(null);
  const [showAllFeatures, setShowAllFeatures] = useState(false);

  if (loading || !signal || !prediction) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel skeleton" style={{ height: '180px' }} />
        <div className="ledger-panel skeleton" style={{ height: '300px' }} />
      </div>
    );
  }

  const horizons = prediction.horizons;
  const allFeatures = prediction.shap_explainability;

  // Relative bar scaling — bars sized relative to the top feature, not absolute 0–100%
  const maxScore = Math.max(...allFeatures.map(f => f.importance_score), 0.001);
  const visibleFeatures = showAllFeatures ? allFeatures : allFeatures.slice(0, DEFAULT_VISIBLE);

  const signalColor = signal.signal === 'BUY' ? '#1A7F4B' : (signal.signal === 'HOLD' ? '#946C00' : '#C0392B');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

      {/* ── Primary Signal Verdict — PRIMARY card treatment ── */}
      <div className="ledger-panel" style={{
        borderLeft: `4px solid ${signalColor}`,
        borderRadius: '12px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            {/* Signal badge + conviction inline */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
              <span className={`ledger-tag ${signal.signal === 'BUY' ? 'tag-buy' : (signal.signal === 'HOLD' ? 'tag-hold' : 'tag-avoid')}`}
                style={{ fontSize: '13px', padding: '5px 13px' }}>
                {signal.signal}
              </span>
              <span style={{ fontSize: '13px', color: 'var(--dash-muted)', fontFamily: 'var(--sans)' }}>
                Conviction: <strong style={{ color: 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}>{signal.signal_score} / 100</strong>
              </span>
            </div>

            {/* Company name — display font, not repeated ticker */}
            <div style={{
              fontFamily: 'var(--serif)',
              fontSize: '26px',
              fontWeight: '400',
              color: 'var(--ink)',
              marginTop: '10px',
              lineHeight: '1.15',
            }}>
              {signal.name}
            </div>
          </div>

          {/* Confidence score — primary metric, serif */}
          <div style={{ textAlign: 'right' }}>
            <div className="ledger-label">MC Dropout confidence</div>
            <div style={{
              fontFamily: 'var(--serif)',
              fontSize: '38px',
              fontWeight: '400',
              color: signalColor,
              fontVariantNumeric: 'tabular-nums',
              lineHeight: '1',
              marginTop: '4px',
            }}>
              {Math.round(signal.confidence * 100)}%
            </div>
          </div>
        </div>

        {/* Decision Reasoning Trail */}
        <div style={{ marginTop: '22px' }}>
          <div className="ledger-label" style={{ marginBottom: '10px' }}>Decision reasoning</div>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '6px', listStyleType: 'none' }}>
            {signal.reasoning.map((r, idx) => (
              <li key={idx} className="ledger-row" style={{
                fontSize: '13px',
                color: '#334155',
                display: 'flex',
                alignItems: 'baseline',
                gap: '10px',
                lineHeight: '1.5',
              }}>
                <span style={{ color: 'var(--gain)', fontWeight: '700', fontSize: '11px', flexShrink: 0 }}>✓</span>
                {r}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ── Multi-Horizon Cards — SECONDARY treatment ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
        {(['1D', '1W', '1M', '6M'] as const).map((hKey) => {
          const h = horizons[hKey];
          const isPos = h.expected_return >= 0;
          const labelMap = { '1D': 'Next day', '1W': '1 week', '1M': '1 month', '6M': '6 months' };
          return (
            <div key={hKey} className="ledger-cell">
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '8px',
              }}>
                <span style={{ fontSize: '12px', color: 'var(--dash-muted)', fontFamily: 'var(--sans)', fontWeight: '500' }}>
                  {labelMap[hKey]}
                </span>
                <span style={{
                  fontFamily: 'var(--mono)',
                  fontSize: '10px',
                  fontWeight: '700',
                  color: 'var(--ink)',
                  background: 'var(--paper)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                }}>
                  {hKey}
                </span>
              </div>
              <div style={{
                fontFamily: 'var(--serif)',
                fontSize: '26px',
                fontWeight: '400',
                color: isPos ? 'var(--gain)' : 'var(--loss)',
                fontVariantNumeric: 'tabular-nums',
                lineHeight: '1.1',
                marginBottom: '6px',
              }}>
                {isPos ? `+${(h.expected_return * 100).toFixed(2)}%` : `${(h.expected_return * 100).toFixed(2)}%`}
              </div>
              <div style={{
                fontSize: '12px',
                color: 'var(--dash-muted)',
                fontFamily: 'var(--mono)',
                fontVariantNumeric: 'tabular-nums',
              }}>
                ₹{h.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 1 })}
              </div>
              <div style={{
                fontSize: '11px',
                color: '#94a3b8',
                fontFamily: 'var(--mono)',
                marginTop: '3px',
                fontVariantNumeric: 'tabular-nums',
              }}>
                {Math.round(h.confidence * 100)}% conf · ±₹{Math.round(h.upper_bound - h.lower_bound)}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── SHAP Feature Attribution — progressive disclosure ── */}
      <div className="ledger-panel">
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '20px',
          flexWrap: 'wrap',
          gap: '12px',
        }}>
          <div>
            <div style={{
              fontFamily: 'var(--sans)',
              fontSize: '17px',
              fontWeight: '700',
              color: 'var(--ink)',
            }}>
              What's driving this prediction
            </div>
            <div style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '3px' }}>
              LSTM feature attribution · click any factor for detail
            </div>
          </div>
          <div style={{
            fontSize: '11px',
            fontFamily: 'var(--sans)',
            color: 'var(--accent)',
            background: 'var(--accent-subtle)',
            border: '1px solid var(--accent-border)',
            padding: '4px 10px',
            borderRadius: '6px',
            fontWeight: '600',
          }}>
            Multi-horizon LSTM + MC Dropout
          </div>
        </div>

        {allFeatures.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--dash-muted)', fontSize: '13px' }}>
            No feature attribution data available for this ticker.
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {visibleFeatures.map((f, i) => {
                const cat = getCategory(f.feature);
                const displayName = f.feature_display || f.feature
                  .replace(/_/g, ' ')
                  .replace(/\b\w/g, c => c.toUpperCase());
                const description = f.feature_description || '';
                const isExpanded = expandedFeature === i;

                // Relative bar: scaled to the top-ranked feature
                const barWidth = Math.round((f.importance_score / maxScore) * 100);

                return (
                  <div
                    key={i}
                    onClick={() => setExpandedFeature(isExpanded ? null : i)}
                    style={{
                      padding: '12px 14px',
                      background: isExpanded ? 'var(--accent-subtle)' : '#ffffff',
                      border: `1px solid ${isExpanded ? 'var(--accent-border)' : 'var(--rule)'}`,
                      borderRadius: '10px',
                      cursor: 'pointer',
                      transition: 'all 0.18s ease',
                    }}
                  >
                    {/* Main row */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>

                      {/* Rank badge — positional context */}
                      <span style={{
                        fontFamily: 'var(--mono)',
                        fontSize: '10px',
                        fontWeight: '700',
                        color: 'var(--dash-muted)',
                        width: '18px',
                        textAlign: 'right',
                        flexShrink: 0,
                      }}>
                        {i + 1}
                      </span>

                      {/* Feature name + category chip */}
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                          <span style={{
                            fontSize: '13px',
                            fontWeight: '600',
                            color: 'var(--ink)',
                            fontFamily: 'var(--sans)',
                          }}>
                            {displayName}
                          </span>
                          {/* Category chip — only shown when category is known */}
                          {cat && (
                            <span style={{
                              fontSize: '9px',
                              padding: '1px 6px',
                              borderRadius: '4px',
                              background: cat.color + '14',
                              color: cat.color,
                              fontFamily: 'var(--sans)',
                              fontWeight: '600',
                              border: `1px solid ${cat.color}28`,
                              letterSpacing: '0.2px',
                            }}>
                              {cat.label}
                            </span>
                          )}
                        </div>

                        {/* Relative importance bar */}
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                          marginTop: '7px',
                        }}>
                          <div style={{
                            flex: 1,
                            height: '5px',
                            background: 'var(--paper)',
                            borderRadius: '3px',
                            overflow: 'hidden',
                            position: 'relative',
                          }}>
                            <div style={{
                              position: 'absolute',
                              left: f.direction === 'negative' ? `${100 - barWidth}%` : '0',
                              width: `${barWidth}%`,
                              height: '100%',
                              background: f.direction === 'positive'
                                ? 'var(--gain)'
                                : 'var(--loss)',
                              borderRadius: '3px',
                              transition: 'width 0.35s ease',
                            }} />
                          </div>
                          <span style={{
                            fontSize: '12px',
                            fontWeight: '600',
                            fontFamily: 'var(--mono)',
                            fontVariantNumeric: 'tabular-nums',
                            color: f.direction === 'positive' ? 'var(--gain)' : 'var(--loss)',
                            minWidth: '52px',
                            textAlign: 'right',
                          }}>
                            {f.direction === 'positive' ? '▲' : '▼'} {(f.importance_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {/* Current value — mono, tabular */}
                      {f.current_value !== undefined && f.current_value !== null && (
                        <div style={{ textAlign: 'right', minWidth: '72px' }}>
                          <div style={{
                            fontSize: '10px',
                            color: 'var(--dash-muted)',
                            fontFamily: 'var(--sans)',
                            marginBottom: '2px',
                          }}>
                            Current
                          </div>
                          <div style={{
                            fontSize: '13px',
                            fontWeight: '700',
                            color: 'var(--ink)',
                            fontFamily: 'var(--mono)',
                            fontVariantNumeric: 'tabular-nums',
                          }}>
                            {typeof f.current_value === 'number'
                              ? (Math.abs(f.current_value) > 1000
                                ? f.current_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })
                                : f.current_value.toFixed(2))
                              : f.current_value}
                          </div>
                        </div>
                      )}

                      <span style={{
                        color: 'var(--dash-muted)',
                        fontSize: '11px',
                        transform: isExpanded ? 'rotate(90deg)' : 'none',
                        transition: 'transform 0.15s',
                        flexShrink: 0,
                      }}>
                        ▸
                      </span>
                    </div>

                    {/* Expanded description */}
                    {isExpanded && description && (
                      <div style={{
                        marginTop: '12px',
                        padding: '10px 12px 10px 30px',
                        fontSize: '12px',
                        lineHeight: '1.65',
                        color: '#475569',
                        borderTop: '1px solid var(--accent-border)',
                      }}>
                        <span style={{ color: cat?.color ?? 'var(--accent)', fontWeight: '600' }}>
                          What this means:{' '}
                        </span>
                        {description}
                        {f.direction === 'positive'
                          ? ' This factor is currently pushing the forecast higher.'
                          : ' This factor is currently pulling the forecast lower.'}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Progressive disclosure control */}
            {allFeatures.length > DEFAULT_VISIBLE && (
              <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'center' }}>
                <button
                  className="ledger-btn-accent"
                  onClick={() => {
                    setShowAllFeatures(v => !v);
                    if (showAllFeatures) setExpandedFeature(null);
                  }}
                >
                  {showAllFeatures
                    ? 'Show fewer factors'
                    : `Show all ${allFeatures.length} factors`}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
