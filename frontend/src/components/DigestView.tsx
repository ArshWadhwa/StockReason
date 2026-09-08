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
          <h2 style={{ fontFamily: 'var(--sans)', fontSize: '20px', fontWeight: '700', color: 'var(--ink)' }}>
            Daily digest
          </h2>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: '11px',
            color: 'var(--dash-muted)',
            background: 'var(--paper)',
            border: '1px solid var(--rule)',
            padding: '4px 12px',
            borderRadius: '20px',
            fontWeight: '600',
            fontVariantNumeric: 'tabular-nums',
          }}>
            {digest.digest_date}
          </span>
        </div>
        <p style={{
          fontSize: '14px',
          lineHeight: '1.7',
          color: 'var(--ink)',
          background: 'var(--paper)',
          padding: '16px 20px',
          border: '1px solid var(--rule)',
          borderRadius: '10px',
        }}>
          {digest.market_summary}
        </p>
      </div>

      {/* Upgrades & Downgrades Columns */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
        {/* Signal Upgrades */}
        <div className="ledger-panel">
        <div className="ledger-label" style={{ color: 'var(--gain)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>▲</span> Signal upgrades
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {digest.signal_upgrades.length === 0 ? (
              <p style={{ fontSize: '13px', color: 'var(--dash-muted)', padding: '12px 0' }}>No signal upgrades today.</p>
            ) : digest.signal_upgrades.map((item) => (
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
          <div className="ledger-label" style={{ color: 'var(--loss)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>▼</span> Signal downgrades
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {digest.signal_downgrades.length === 0 ? (
              <p style={{ fontSize: '13px', color: 'var(--dash-muted)', padding: '12px 0' }}>No signal downgrades today.</p>
            ) : digest.signal_downgrades.map((item) => (
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
        <div className="ledger-panel-title" style={{ marginBottom: '16px' }}>Actionable universe</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>
          <div style={{ padding: '16px', background: 'var(--gain-bg)', border: '1px solid var(--gain-border)', borderRadius: '10px' }}>
            <div style={{ color: 'var(--gain)', fontSize: '12px', fontFamily: 'var(--sans)', fontWeight: '600', marginBottom: '12px' }}>
              High-conviction opportunities
            </div>
            {digest.top_opportunities.length === 0 ? (
              <p style={{ fontSize: '13px', color: 'var(--dash-muted)' }}>No stocks currently meet high-conviction thresholds.</p>
            ) : (
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {digest.top_opportunities.map(t => (
                  <button
                    key={t}
                    onClick={() => onSelectStock(t)}
                    style={{
                      padding: '5px 11px',
                      background: '#ffffff',
                      border: '1px solid var(--gain-border)',
                      borderRadius: '6px',
                      color: 'var(--gain)',
                      fontFamily: 'var(--mono)',
                      fontSize: '11px',
                      fontWeight: '700',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'var(--gain)';
                      e.currentTarget.style.color = '#ffffff';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = '#ffffff';
                      e.currentTarget.style.color = 'var(--gain)';
                    }}
                  >
                    {t.replace('.NS', '')}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={{ padding: '16px', background: 'var(--loss-bg)', border: '1px solid var(--loss-border)', borderRadius: '10px' }}>
            <div style={{ color: 'var(--loss)', fontSize: '12px', fontFamily: 'var(--sans)', fontWeight: '600', marginBottom: '12px' }}>
              Elevated uncertainty / avoid
            </div>
            {digest.high_risk_alerts.length === 0 ? (
              <p style={{ fontSize: '13px', color: 'var(--dash-muted)' }}>No stocks flagged for elevated uncertainty.</p>
            ) : (
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {digest.high_risk_alerts.map(t => (
                  <button
                    key={t}
                    onClick={() => onSelectStock(t)}
                    style={{
                      padding: '5px 11px',
                      background: '#ffffff',
                      border: '1px solid var(--loss-border)',
                      borderRadius: '6px',
                      color: 'var(--loss)',
                      fontFamily: 'var(--mono)',
                      fontSize: '11px',
                      fontWeight: '700',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'var(--loss)';
                      e.currentTarget.style.color = '#ffffff';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = '#ffffff';
                      e.currentTarget.style.color = 'var(--loss)';
                    }}
                  >
                    {t.replace('.NS', '')}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
