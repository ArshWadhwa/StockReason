import React from 'react';
import type { SignalData, PredictionData } from '../types';

interface SignalsViewProps {
  signal: SignalData | null;
  prediction: PredictionData | null;
  loading: boolean;
}

export const SignalsView: React.FC<SignalsViewProps> = ({ signal, prediction, loading }) => {
  if (loading || !signal || !prediction) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel" style={{ height: '180px' }} />
        <div className="ledger-panel" style={{ height: '300px' }} />
      </div>
    );
  }

  const horizons = prediction.horizons;
  const ensemble = prediction.ensemble;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Primary Signal Verdict Card */}
      <div className="ledger-panel" style={{
        borderLeft: `4px solid ${
          signal.signal === 'BUY' ? 'var(--brass-amber)' :
          (signal.signal === 'HOLD' ? 'var(--hold-gold)' : 'var(--rust-avoid)')
        }`
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span className={`ledger-tag ${signal.signal === 'BUY' ? 'tag-buy' : (signal.signal === 'HOLD' ? 'tag-hold' : 'tag-avoid')}`}>
                {signal.signal} SIGNAL
              </span>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-data)' }}>
                Conviction: <strong style={{ color: 'var(--text-parchment)' }}>{signal.signal_score} / 100</strong>
              </span>
            </div>
            <div style={{ fontFamily: 'var(--font-serif)', fontSize: '26px', fontWeight: '700', color: 'var(--text-parchment)', marginTop: '12px' }}>
              {signal.name} ({signal.ticker})
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div className="ledger-label">MC Dropout Confidence</div>
            <div style={{ fontSize: '30px', fontWeight: '600', color: 'var(--brass-amber)', fontFamily: 'var(--font-data)' }}>
              {Math.round(signal.confidence * 100)}%
            </div>
          </div>
        </div>

        {/* Ensemble Disagreement Banner */}
        {ensemble.disagreement_detected && (
          <div style={{
            marginTop: '18px',
            padding: '14px 18px',
            background: 'var(--rust-bg)',
            border: '1px solid var(--rust-border)',
            borderRadius: '2px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            <div>
              <div style={{ fontWeight: '700', color: 'var(--rust-avoid)', fontSize: '12px', fontFamily: 'var(--font-data)' }}>
                ▲ ENSEMBLE DISAGREEMENT ALERT (LSTM VS XGBOOST)
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                LSTM projects {ensemble.lstm_return_1M >= 0 ? `+${(ensemble.lstm_return_1M*100).toFixed(1)}%` : `${(ensemble.lstm_return_1M*100).toFixed(1)}%`} while XGBoost projects {ensemble.xgboost_return_1M >= 0 ? `+${(ensemble.xgboost_return_1M*100).toFixed(1)}%` : `${(ensemble.xgboost_return_1M*100).toFixed(1)}%`} (Spread: {(ensemble.disagreement_delta*100).toFixed(1)}%).
              </div>
            </div>
          </div>
        )}

        {/* Decision Reasoning Trail */}
        <div style={{ marginTop: '22px' }}>
          <div className="ledger-label">Decision Reasoning Trail</div>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '8px', listStyleType: 'none', marginTop: '10px' }}>
            {signal.reasoning.map((r, idx) => (
              <li key={idx} style={{
                fontSize: '13px',
                color: 'var(--text-parchment)',
                display: 'flex',
                alignItems: 'baseline',
                gap: '10px',
                background: '#0B0D0F',
                padding: '10px 14px',
                border: '1px solid var(--line-grid)'
              }}>
                <span style={{ color: 'var(--brass-amber)', fontFamily: 'var(--font-data)', fontSize: '11px' }}>▸</span>
                {r}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Multi-Horizon Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
        {(['1D', '1W', '1M', '6M'] as const).map((hKey) => {
          const h = horizons[hKey];
          return (
            <div key={hKey} className="ledger-cell">
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)', fontSize: '10px', fontFamily: 'var(--font-data)' }}>
                <span>{hKey === '1D' ? 'NEXT DAY' : (hKey === '1W' ? '1 WEEK' : (hKey === '1M' ? '1 MONTH' : '6 MONTHS'))}</span>
                <span style={{ fontWeight: '600' }}>{hKey}</span>
              </div>
              <div style={{
                fontSize: '22px',
                fontWeight: '600',
                margin: '8px 0',
                fontFamily: 'var(--font-data)',
                color: h.expected_return >= 0 ? 'var(--brass-amber)' : 'var(--rust-avoid)'
              }}>
                {h.expected_return >= 0 ? `+${(h.expected_return * 100).toFixed(2)}%` : `${(h.expected_return * 100).toFixed(2)}%`}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-data)' }}>
                Target: ₹{h.predicted_price.toFixed(1)}
              </div>
            </div>
          );
        })}
      </div>

      {/* SHAP Feature Attribution Card */}
      <div className="ledger-panel">
        <div style={{ fontFamily: 'var(--font-serif)', fontSize: '18px', fontWeight: '600', color: 'var(--text-parchment)', marginBottom: '16px' }}>
          SHAP Feature Attribution Matrix
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {prediction.shap_explainability.map((f, i) => (
            <div key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontFamily: 'var(--font-data)', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-parchment)' }}>{f.feature}</span>
                <span style={{ color: f.direction === 'positive' ? 'var(--brass-amber)' : 'var(--rust-avoid)' }}>
                  {f.direction === 'positive' ? `+${f.impact}` : `${f.impact}`}
                </span>
              </div>
              <div style={{ height: '4px', background: '#0B0D0F', border: '1px solid var(--line-grid)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${f.importance_score * 100}%`,
                  background: f.direction === 'positive' ? 'var(--brass-amber)' : 'var(--rust-avoid)'
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
