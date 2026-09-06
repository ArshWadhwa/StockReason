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
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '12px' }}>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '24px', fontWeight: '700', color: 'var(--text-parchment)' }}>
            Daily "What Changed" Digest
          </h2>
          <span style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--slate-blue)' }}>
            EDITION: {digest.digest_date}
          </span>
        </div>
        <p style={{ fontSize: '15px', lineHeight: '1.7', color: 'var(--text-parchment)', background: '#07090C', padding: '16px', border: '1px solid var(--line-grid)' }}>
          {digest.market_summary}
        </p>
      </div>

      {/* Upgrades & Downgrades Columns */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
        {/* Signal Upgrades */}
        <div className="ledger-panel">
          <div className="ledger-label" style={{ color: 'var(--brass-amber)', marginBottom: '14px' }}>
            ▲ Signal Upgrades Today
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {digest.signal_upgrades.map((item) => (
              <div
                key={item.ticker}
                onClick={() => onSelectStock(item.ticker)}
                style={{ padding: '12px', background: '#0B0D0F', border: '1px solid var(--line-grid)', cursor: 'pointer' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <strong style={{ color: 'var(--text-parchment)', fontFamily: 'var(--font-data)' }}>{item.ticker}</strong>
                  <span style={{ fontSize: '11px', fontFamily: 'var(--font-data)', color: 'var(--brass-amber)' }}>
                    {item.previous_signal} → {item.new_signal}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  {item.primary_catalyst}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Signal Downgrades */}
        <div className="ledger-panel">
          <div className="ledger-label" style={{ color: 'var(--rust-avoid)', marginBottom: '14px' }}>
            ▼ Signal Downgrades / Flags
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {digest.signal_downgrades.map((item) => (
              <div
                key={item.ticker}
                onClick={() => onSelectStock(item.ticker)}
                style={{ padding: '12px', background: '#0B0D0F', border: '1px solid var(--line-grid)', cursor: 'pointer' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <strong style={{ color: 'var(--text-parchment)', fontFamily: 'var(--font-data)' }}>{item.ticker}</strong>
                  <span style={{ fontSize: '11px', fontFamily: 'var(--font-data)', color: 'var(--rust-avoid)' }}>
                    {item.previous_signal} → {item.new_signal}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  {item.primary_catalyst}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Conviction Universe Focus */}
      <div className="ledger-panel">
        <div className="ledger-panel-title" style={{ marginBottom: '14px' }}>Actionable Universe Focus</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
          <div style={{ padding: '14px', background: '#0B0D0F', border: '1px solid var(--line-grid)' }}>
            <div style={{ color: 'var(--brass-amber)', fontSize: '11px', fontFamily: 'var(--font-data)', fontWeight: '600', marginBottom: '8px' }}>
              HIGH AI CONVICTION LIST
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {digest.top_opportunities.map(t => (
                <button
                  key={t}
                  onClick={() => onSelectStock(t)}
                  style={{
                    padding: '4px 10px',
                    background: 'var(--brass-glow)',
                    border: '1px solid var(--brass-border)',
                    color: 'var(--brass-amber)',
                    fontFamily: 'var(--font-data)',
                    fontSize: '11px',
                    cursor: 'pointer'
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          <div style={{ padding: '14px', background: '#0B0D0F', border: '1px solid var(--line-grid)' }}>
            <div style={{ color: 'var(--rust-avoid)', fontSize: '11px', fontFamily: 'var(--font-data)', fontWeight: '600', marginBottom: '8px' }}>
              ELEVATED UNCERTAINTY / AVOID LIST
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {digest.high_risk_alerts.map(t => (
                <button
                  key={t}
                  onClick={() => onSelectStock(t)}
                  style={{
                    padding: '4px 10px',
                    background: 'var(--rust-bg)',
                    border: '1px solid var(--rust-border)',
                    color: 'var(--rust-avoid)',
                    fontFamily: 'var(--font-data)',
                    fontSize: '11px',
                    cursor: 'pointer'
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
