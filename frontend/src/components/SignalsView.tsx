import React, { useState } from 'react';
import type { SignalData, PredictionData } from '../types';

interface SignalsViewProps {
  signal: SignalData | null;
  prediction: PredictionData | null;
  loading: boolean;
}

// Category classification for visual grouping
const FEATURE_CATEGORIES: Record<string, { label: string; color: string; icon: string }> = {
  // Technical / Momentum
  rsi_14: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  rsi_7: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  rsi_21: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  rsi_28: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  macd_line_12_26: { label: 'Momentum', color: '#60a5fa', icon: '📈' },
  macd_hist_12_26: { label: 'Momentum', color: '#60a5fa', icon: '📈' },
  macd_signal_12_26: { label: 'Momentum', color: '#60a5fa', icon: '📈' },
  stoch_k_14: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  stoch_d_14: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  williams_r_14: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  cci_14: { label: 'Momentum', color: '#60a5fa', icon: '📊' },
  cci_20: { label: 'Momentum', color: '#60a5fa', icon: '📊' },

  // Trend
  dist_sma_50: { label: 'Trend', color: '#4ade80', icon: '📐' },
  dist_sma_20: { label: 'Trend', color: '#4ade80', icon: '📐' },
  dist_sma_200: { label: 'Trend', color: '#4ade80', icon: '📐' },
  dist_ema_50: { label: 'Trend', color: '#4ade80', icon: '📐' },
  dist_ema_20: { label: 'Trend', color: '#4ade80', icon: '📐' },
  sma_50_200_ratio: { label: 'Trend', color: '#4ade80', icon: '📐' },
  sma_20_50_ratio: { label: 'Trend', color: '#4ade80', icon: '📐' },
  adx_14: { label: 'Trend', color: '#4ade80', icon: '📐' },
  close_slope_20: { label: 'Trend', color: '#4ade80', icon: '📐' },
  close_slope_10: { label: 'Trend', color: '#4ade80', icon: '📐' },
  close_slope_5: { label: 'Trend', color: '#4ade80', icon: '📐' },

  // Volatility
  atr_14: { label: 'Volatility', color: '#f59e0b', icon: '⚡' },
  natr_14: { label: 'Volatility', color: '#f59e0b', icon: '⚡' },
  volatility_20d: { label: 'Volatility', color: '#f59e0b', icon: '⚡' },
  volatility_5d: { label: 'Volatility', color: '#f59e0b', icon: '⚡' },
  volatility_60d: { label: 'Volatility', color: '#f59e0b', icon: '⚡' },
  bb_width_20: { label: 'Volatility', color: '#f59e0b', icon: '⚡' },
  bb_pct_b_20: { label: 'Volatility', color: '#f59e0b', icon: '⚡' },

  // Volume / Flow
  volume: { label: 'Volume', color: '#a78bfa', icon: '📦' },
  vol_ratio_20: { label: 'Volume', color: '#a78bfa', icon: '📦' },
  vol_ratio_5: { label: 'Volume', color: '#a78bfa', icon: '📦' },
  obv: { label: 'Volume', color: '#a78bfa', icon: '📦' },
  cmf_20: { label: 'Volume', color: '#a78bfa', icon: '💰' },
  pvt: { label: 'Volume', color: '#a78bfa', icon: '📦' },

  // Sentiment
  sentiment_mean: { label: 'Sentiment', color: '#f472b6', icon: '🧠' },
  sentiment_pos_ratio: { label: 'Sentiment', color: '#f472b6', icon: '🧠' },
  sentiment_neg_ratio: { label: 'Sentiment', color: '#f472b6', icon: '🧠' },
  sentiment_momentum_3d: { label: 'Sentiment', color: '#f472b6', icon: '🧠' },
  sentiment_momentum_7d: { label: 'Sentiment', color: '#f472b6', icon: '🧠' },

  // Macro
  vix_close: { label: 'Macro', color: '#fb923c', icon: '🌍' },
  vix_close_return: { label: 'Macro', color: '#fb923c', icon: '🌍' },
  nifty_50_close: { label: 'Macro', color: '#fb923c', icon: '🌍' },
  nifty_50_close_return: { label: 'Macro', color: '#fb923c', icon: '🌍' },
  nifty_bank_close_return: { label: 'Macro', color: '#fb923c', icon: '🌍' },

  // Price / Returns
  return_1d: { label: 'Returns', color: '#22d3ee', icon: '💹' },
  return_5d: { label: 'Returns', color: '#22d3ee', icon: '💹' },
  return_21d: { label: 'Returns', color: '#22d3ee', icon: '💹' },
  return_63d: { label: 'Returns', color: '#22d3ee', icon: '💹' },
  close: { label: 'Price', color: '#22d3ee', icon: '💹' },
  hl_spread_pct: { label: 'Price', color: '#22d3ee', icon: '💹' },
};

const getCategory = (feature: string) =>
  FEATURE_CATEGORIES[feature] || { label: 'Other', color: '#888', icon: '📋' };

export const SignalsView: React.FC<SignalsViewProps> = ({ signal, prediction, loading }) => {
  const [expandedFeature, setExpandedFeature] = useState<number | null>(null);

  if (loading || !signal || !prediction) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel" style={{ height: '180px' }} />
        <div className="ledger-panel" style={{ height: '300px' }} />
      </div>
    );
  }

  const horizons = prediction.horizons;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Primary Signal Verdict Card */}
      <div className="ledger-panel" style={{
        borderLeft: `5px solid ${
          signal.signal === 'BUY' ? '#16a34a' :
          (signal.signal === 'HOLD' ? '#b45309' : '#dc2626')
        }`
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span className={`ledger-tag ${signal.signal === 'BUY' ? 'tag-buy' : (signal.signal === 'HOLD' ? 'tag-hold' : 'tag-avoid')}`}>
                {signal.signal} SIGNAL
              </span>
              <span style={{ fontSize: '12px', color: '#64748b', fontFamily: 'var(--mono)' }}>
                Conviction: <strong style={{ color: '#0f172a' }}>{signal.signal_score} / 100</strong>
              </span>
            </div>
            <div style={{ fontFamily: 'var(--display)', fontSize: '26px', fontWeight: '700', color: 'var(--text-primary)', marginTop: '10px' }}>
              {signal.name} <span style={{ fontFamily: 'var(--mono)', fontSize: '18px', color: '#64748b', fontWeight: '500' }}>({signal.ticker})</span>
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div className="ledger-label">MC Dropout Confidence</div>
            <div style={{ fontSize: '32px', fontWeight: '700', color: '#1e4d1f', fontFamily: 'var(--mono)' }}>
              {Math.round(signal.confidence * 100)}%
            </div>
          </div>
        </div>

        {/* Decision Reasoning Trail */}
        <div style={{ marginTop: '22px' }}>
          <div className="ledger-label">Decision Reasoning Trail</div>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '8px', listStyleType: 'none', marginTop: '10px' }}>
            {signal.reasoning.map((r, idx) => (
              <li key={idx} style={{
                fontSize: '13px',
                color: '#334155',
                display: 'flex',
                alignItems: 'baseline',
                gap: '10px',
                background: '#f8fafc',
                padding: '10px 14px',
                borderRadius: '8px',
                border: '1px solid #e2e8f0',
                lineHeight: '1.5'
              }}>
                <span style={{ color: '#16a34a', fontWeight: '700', fontSize: '12px' }}>✓</span>
                {r}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Multi-Horizon Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
        {(['1D', '1W', '1M', '6M'] as const).map((hKey) => {
          const h = horizons[hKey];
          const isPos = h.expected_return >= 0;
          return (
            <div key={hKey} className="ledger-cell">
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#64748b', fontSize: '11px', fontFamily: 'var(--mono)' }}>
                <span>{hKey === '1D' ? 'NEXT DAY' : (hKey === '1W' ? '1 WEEK' : (hKey === '1M' ? '1 MONTH' : '6 MONTHS'))}</span>
                <span style={{ fontWeight: '700', color: '#0f172a' }}>{hKey}</span>
              </div>
              <div style={{
                fontSize: '24px',
                fontWeight: '700',
                margin: '8px 0',
                fontFamily: 'var(--mono)',
                color: isPos ? '#16a34a' : '#dc2626'
              }}>
                {isPos ? `+${(h.expected_return * 100).toFixed(2)}%` : `${(h.expected_return * 100).toFixed(2)}%`}
              </div>
              <div style={{ fontSize: '12px', color: '#475569', fontFamily: 'var(--mono)', fontWeight: '500' }}>
                Target: ₹{h.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 1 })}
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8', fontFamily: 'var(--mono)', marginTop: '4px' }}>
                Conf: {Math.round(h.confidence * 100)}% · ±₹{Math.round(h.upper_bound - h.lower_bound)}
              </div>
            </div>
          );
        })}
      </div>

      {/* SHAP Feature Attribution — Human Readable */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div style={{ fontFamily: 'var(--display)', fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              What's Driving This Prediction
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>
              LSTM feature attribution — click any feature for details
            </div>
          </div>
          <div style={{
            fontSize: '11px',
            fontFamily: 'var(--mono)',
            color: '#1e4d1f',
            background: '#e8f5e9',
            border: '1px solid #c8e6c9',
            padding: '4px 10px',
            borderRadius: '6px',
            fontWeight: '600'
          }}>
            Multi-Horizon LSTM + MC Dropout
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {prediction.shap_explainability.map((f, i) => {
            const cat = getCategory(f.feature);
            const displayName = f.feature_display || f.feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            const description = f.feature_description || '';
            const isExpanded = expandedFeature === i;
            const barWidth = Math.min(100, Math.round(f.importance_score * 100 * 3));

            return (
              <div
                key={i}
                onClick={() => setExpandedFeature(isExpanded ? null : i)}
                style={{
                  padding: '14px 16px',
                  background: isExpanded ? '#f0fdf4' : '#ffffff',
                  border: `1px solid ${isExpanded ? '#86efac' : '#e2e8f0'}`,
                  borderRadius: '10px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
                }}
              >
                {/* Main Row */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  {/* Category Icon */}
                  <span style={{ fontSize: '16px', width: '24px', textAlign: 'center' }}>{cat.icon}</span>

                  {/* Feature Name + Category Badge */}
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '13px', fontWeight: '600', color: '#0f172a', fontFamily: 'var(--mono)' }}>
                        {displayName}
                      </span>
                      <span style={{
                        fontSize: '9px',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        background: cat.color + '15',
                        color: cat.color,
                        fontFamily: 'var(--mono)',
                        fontWeight: '700',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                        border: `1px solid ${cat.color}30`
                      }}>
                        {cat.label}
                      </span>
                    </div>

                    {/* Importance Bar */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '6px' }}>
                      <div style={{
                        flex: 1,
                        height: '6px',
                        background: '#f1f5f9',
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
                            ? 'linear-gradient(90deg, #4ade80, #16a34a)'
                            : 'linear-gradient(270deg, #f87171, #dc2626)',
                          borderRadius: '3px',
                          transition: 'width 0.3s ease',
                        }} />
                      </div>
                      <span style={{
                        fontSize: '12px',
                        fontWeight: '700',
                        fontFamily: 'var(--mono)',
                        color: f.direction === 'positive' ? '#16a34a' : '#dc2626',
                        minWidth: '50px',
                        textAlign: 'right',
                      }}>
                        {f.direction === 'positive' ? '▲' : '▼'} {(f.importance_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Current Value */}
                  {f.current_value !== undefined && f.current_value !== null && (
                    <div style={{ textAlign: 'right', minWidth: '80px' }}>
                      <div style={{ fontSize: '10px', color: '#94a3b8', fontFamily: 'var(--mono)' }}>CURRENT</div>
                      <div style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a', fontFamily: 'var(--mono)' }}>
                        {typeof f.current_value === 'number'
                          ? (Math.abs(f.current_value) > 1000
                            ? f.current_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })
                            : f.current_value.toFixed(2))
                          : f.current_value}
                      </div>
                    </div>
                  )}

                  {/* Expand Arrow */}
                  <span style={{ color: '#94a3b8', fontSize: '12px', transform: isExpanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
                    ▸
                  </span>
                </div>

                {/* Expanded Description */}
                {isExpanded && description && (
                  <div style={{
                    marginTop: '12px',
                    padding: '12px 14px 10px 36px',
                    fontSize: '12px',
                    lineHeight: '1.6',
                    color: '#475569',
                    borderTop: '1px solid #e2e8f0',
                  }}>
                    <span style={{ color: cat.color, fontWeight: '700' }}>What this means: </span>
                    {description}
                    {f.direction === 'positive'
                      ? ' This feature is currently pushing the prediction higher.'
                      : ' This feature is currently pulling the prediction lower.'}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {prediction.shap_explainability.length === 0 && (
          <div style={{ padding: '24px', textAlign: 'center', color: '#94a3b8', fontSize: '13px', fontFamily: 'var(--mono)' }}>
            No feature attribution data available for this ticker.
          </div>
        )}
      </div>
    </div>
  );
};
