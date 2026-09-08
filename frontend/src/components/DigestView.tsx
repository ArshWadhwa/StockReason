import React from 'react';
import type { DailyDigestData } from '../types';

interface DigestViewProps {
  digest: DailyDigestData | null;
  onSelectStock: (ticker: string) => void;
  loading: boolean;
}

export const DigestView: React.FC<DigestViewProps> = ({
  digest,
  onSelectStock,
  loading
}) => {
  if (loading || !digest) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel" style={{ height: '160px' }} />
        <div className="ledger-panel" style={{ height: '300px' }} />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Executive Market Brief */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <h2 style={{ fontFamily: 'var(--display)', fontSize: '24px', fontWeight: '700', color: 'var(--text-primary)' }}>
            Daily "What Changed" Digest
          </h2>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: '11px',
            color: '#1e4d1f',
            background: '#e8f5e9',
            border: '1px solid #c8e6c9',
            padding: '4px 12px',
            borderRadius: '20px',
            fontWeight: '600'
          }}>
            EDITION: {digest.digest_date}
          </span>
        </div>
        <p style={{
          fontSize: '14px',
          lineHeight: '1.7',
          color: '#1e3a24',
          background: 'linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)',
          padding: '18px 20px',
          border: '1px solid #bbf7d0',
          borderRadius: '12px',
          boxShadow: '0 1px 2px rgba(22, 163, 74, 0.04)'
        }}>
          {digest.market_summary}
        </p>
      </div>

      {/* Upgrades & Downgrades Columns */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
        {/* Signal Upgrades */}
        <div className="ledger-panel">
          <div className="ledger-label" style={{ color: '#16a34a', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>▲</span> Signal Upgrades Today
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {digest.signal_upgrades.map((item) => (
              <div
                key={item.ticker}
                onClick={() => onSelectStock(item.ticker)}
                style={{
                  padding: '14px',
                  background: '#ffffff',
                  border: '1px solid #bbf7d0',
                  borderRadius: '10px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.02)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#f0fdf4';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                  e.currentTarget.style.borderColor = '#86efac';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = '#ffffff';
                  e.currentTarget.style.transform = 'none';
                  e.currentTarget.style.borderColor = '#bbf7d0';
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <strong style={{ color: '#0f172a', fontFamily: 'var(--mono)', fontSize: '13px' }}>{item.ticker}</strong>
                  <span style={{
                    fontSize: '11px',
                    fontFamily: 'var(--mono)',
                    color: '#1e4d1f',
                    background: '#e8f5e9',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontWeight: '700',
                    border: '1px solid #c8e6c9'
                  }}>
                    {item.previous_signal} → {item.new_signal}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: '#475569', lineHeight: '1.4' }}>
                  {item.primary_catalyst}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Signal Downgrades */}
        <div className="ledger-panel">
          <div className="ledger-label" style={{ color: '#dc2626', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>▼</span> Signal Downgrades / Flags
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {digest.signal_downgrades.map((item) => (
              <div
                key={item.ticker}
                onClick={() => onSelectStock(item.ticker)}
                style={{
                  padding: '14px',
                  background: '#ffffff',
                  border: '1px solid #fecaca',
                  borderRadius: '10px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.02)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#fef2f2';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                  e.currentTarget.style.borderColor = '#fca5a5';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = '#ffffff';
                  e.currentTarget.style.transform = 'none';
                  e.currentTarget.style.borderColor = '#fecaca';
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <strong style={{ color: '#0f172a', fontFamily: 'var(--mono)', fontSize: '13px' }}>{item.ticker}</strong>
                  <span style={{
                    fontSize: '11px',
                    fontFamily: 'var(--mono)',
                    color: '#b91c1c',
                    background: '#fef2f2',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontWeight: '700',
                    border: '1px solid #fecaca'
                  }}>
                    {item.previous_signal} → {item.new_signal}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: '#475569', lineHeight: '1.4' }}>
                  {item.primary_catalyst}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Conviction Universe Focus */}
      <div className="ledger-panel">
        <div className="ledger-panel-title" style={{ marginBottom: '16px' }}>Actionable Universe Focus</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>
          <div style={{ padding: '16px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '12px' }}>
            <div style={{ color: '#1e4d1f', fontSize: '11px', fontFamily: 'var(--mono)', fontWeight: '700', marginBottom: '10px' }}>
              HIGH AI CONVICTION LIST
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {digest.top_opportunities.map(t => (
                <button
                  key={t}
                  onClick={() => onSelectStock(t)}
                  style={{
                    padding: '6px 12px',
                    background: '#ffffff',
                    border: '1px solid #86efac',
                    borderRadius: '6px',
                    color: '#1e4d1f',
                    fontFamily: 'var(--mono)',
                    fontSize: '11px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = '#1e4d1f';
                    e.currentTarget.style.color = '#ffffff';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = '#ffffff';
                    e.currentTarget.style.color = '#1e4d1f';
                    e.currentTarget.style.transform = 'none';
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          <div style={{ padding: '16px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '12px' }}>
            <div style={{ color: '#b91c1c', fontSize: '11px', fontFamily: 'var(--mono)', fontWeight: '700', marginBottom: '10px' }}>
              ELEVATED UNCERTAINTY / AVOID LIST
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {digest.high_risk_alerts.map(t => (
                <button
                  key={t}
                  onClick={() => onSelectStock(t)}
                  style={{
                    padding: '6px 12px',
                    background: '#ffffff',
                    border: '1px solid #fca5a5',
                    borderRadius: '6px',
                    color: '#b91c1c',
                    fontFamily: 'var(--mono)',
                    fontSize: '11px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = '#b91c1c';
                    e.currentTarget.style.color = '#ffffff';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = '#ffffff';
                    e.currentTarget.style.color = '#b91c1c';
                    e.currentTarget.style.transform = 'none';
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
