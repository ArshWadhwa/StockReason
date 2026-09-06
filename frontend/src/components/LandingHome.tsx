import React from 'react';
import type { MarketOverviewData, SignalData } from '../types';

interface LandingHomeProps {
  overview: MarketOverviewData | null;
  signals: SignalData[];
  onOpenDashboard: (ticker?: string) => void;
}

export const LandingHome: React.FC<LandingHomeProps> = ({
  overview,
  signals,
  onOpenDashboard,
}) => {
  const topStock = signals.find(s => s.signal === 'BUY') || signals[0];

  return (
    <div>
      {/* Editorial Hero */}
      <section className="hero">
        <div>
          <div className="eyebrow">Dark terminal meets editorial</div>
          <h1>Read the market before it moves.</h1>
          <p>
            StockReason compresses NIFTY 50 price action, sentiment, and macro context into a single disciplined view.
            It feels like a live terminal, but reads like a premium editorial analysis.
          </p>
          <div style={{ display: 'flex', gap: '12px', marginTop: '26px', flexWrap: 'wrap' }}>
            <button
              className="btn primary"
              onClick={() => onOpenDashboard(topStock?.ticker)}
            >
              Open Nifty 50 UI →
            </button>
            <a className="btn secondary" href="#features">
              View Architecture
            </a>
          </div>
        </div>

        <div className="hero-panel">
          <div className="panel-label">Session summary</div>
          <div className="panel-grid">
            <div className="stat-card">
              <div className="stat-title">Momentum</div>
              <div className="stat-value" style={{ color: 'var(--success)' }}>
                {topStock ? `+${(topStock.expected_return_1M * 100).toFixed(2)}%` : '+1.28%'}
              </div>
              <div className="stat-desc">
                {topStock ? `${topStock.ticker.replace('.NS','')} leads session with clean follow-through.` : 'RELIANCE closes higher with clean follow-through.'}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-title">Bias</div>
              <div className="stat-value" style={{ color: 'var(--success)' }}>BUY</div>
              <div className="stat-desc">Market structure remains constructive across leading names.</div>
            </div>
            <div className="stat-card">
              <div className="stat-title">Sentiment</div>
              <div className="stat-value">0.72</div>
              <div className="stat-desc">Headline tone is leaning positive with manageable event noise.</div>
            </div>
            <div className="stat-card">
              <div className="stat-title">Risk Regime</div>
              <div className="stat-value" style={{ color: 'var(--success)' }}>
                {overview?.market_regime.replace(/_/g, ' ').toUpperCase() || 'LOW VOL'}
              </div>
              <div className="stat-desc">Trend remains intact without excessive volatility expansion.</div>
            </div>
          </div>
        </div>
      </section>

      {/* What the platform carries grid */}
      <div style={{ marginTop: '40px' }} id="features">
        <div className="section-kicker">What the platform carries</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px', marginBottom: '32px' }}>
          <div className="feature-card">
            <div className="feature-meta">01</div>
            <h3 style={{ margin: '10px 0', fontSize: '18px', color: '#f2f2f2', letterSpacing: '-0.02em' }}>Market snapshot</h3>
            <p style={{ margin: 0, color: '#9f9f9f', fontSize: '14px', lineHeight: '1.6' }}>
              Terminal-grade price reading for NIFTY 50 names, layered with market structure and a compact execution view.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-meta">02</div>
            <h3 style={{ margin: '10px 0', fontSize: '18px', color: '#f2f2f2', letterSpacing: '-0.02em' }}>News & FinBERT</h3>
            <p style={{ margin: 0, color: '#9f9f9f', fontSize: '14px', lineHeight: '1.6' }}>
              Track the bias in headlines, sentiment shifts, and divergence alerts without leaving the main workflow.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-meta">03</div>
            <h3 style={{ margin: '10px 0', fontSize: '18px', color: '#f2f2f2', letterSpacing: '-0.02em' }}>Technical signals</h3>
            <p style={{ margin: 0, color: '#9f9f9f', fontSize: '14px', lineHeight: '1.6' }}>
              Trend, momentum, and confirmation signals arranged like a clean desk note rather than an overcrowded dashboard.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-meta">04</div>
            <h3 style={{ margin: '10px 0', fontSize: '18px', color: '#f2f2f2', letterSpacing: '-0.02em' }}>Macro context</h3>
            <p style={{ margin: 0, color: '#9f9f9f', fontSize: '14px', lineHeight: '1.6' }}>
              Keep India VIX, crude oil, and broader sectoral indices visible when the tape gets noisy.
            </p>
          </div>
        </div>
      </div>

      {/* Evidence layout */}
      <div className="section-kicker">Evidence layout</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px', marginBottom: '40px' }}>
        <div className="chart-shell">
          <div className="panel-label">Preview frame</div>
          <div style={{ border: '1px solid #1d1d1d', background: '#0d0d0d', padding: '22px', minHeight: '260px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div style={{ fontFamily: 'var(--mono)', color: '#8a8a8a', fontSize: '12px', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
              Price action scaffold
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', margin: '18px 0' }}>
              <div style={{ border: '1px solid #202020', padding: '12px', background: '#101010' }}>
                <div style={{ fontFamily: 'var(--mono)', color: '#8c8c8c', fontSize: '11px', marginBottom: '6px' }}>Open</div>
                <div style={{ fontFamily: 'var(--mono)', color: '#f0f0f0', fontSize: '17px' }}>2,948</div>
              </div>
              <div style={{ border: '1px solid #202020', padding: '12px', background: '#101010' }}>
                <div style={{ fontFamily: 'var(--mono)', color: '#8c8c8c', fontSize: '11px', marginBottom: '6px' }}>High</div>
                <div style={{ fontFamily: 'var(--mono)', color: '#f0f0f0', fontSize: '17px' }}>2,991</div>
              </div>
              <div style={{ border: '1px solid #202020', padding: '12px', background: '#101010' }}>
                <div style={{ fontFamily: 'var(--mono)', color: '#8c8c8c', fontSize: '11px', marginBottom: '6px' }}>Low</div>
                <div style={{ fontFamily: 'var(--mono)', color: '#f0f0f0', fontSize: '17px' }}>2,932</div>
              </div>
              <div style={{ border: '1px solid #202020', padding: '12px', background: '#101010' }}>
                <div style={{ fontFamily: 'var(--mono)', color: '#8c8c8c', fontSize: '11px', marginBottom: '6px' }}>Close</div>
                <div style={{ fontFamily: 'var(--mono)', color: '#f0f0f0', fontSize: '17px' }}>2,984</div>
              </div>
            </div>
            <div style={{ color: '#9a9a9a', fontSize: '13px', lineHeight: '1.6' }}>
              The interactive terminal dashboard lives behind the "Open Nifty 50 UI" door. Here, the landing page keeps visual language strict and structure quiet.
            </div>
          </div>
        </div>

        <div className="notes-shell">
          <div className="panel-label">Analyst notes</div>
          <div className="list-item">
            <div className="list-title">Signal hierarchy</div>
            <div className="list-body">Reads as a disciplined workspace: one lead signal, three supporting layers, zero unnecessary visual clutter.</div>
          </div>
          <div className="list-item">
            <div className="list-title">Typography rule</div>
            <div className="list-body">Use monospace for values, percentages, and market codes. Keep headlines in clean sans to preserve editorial clarity.</div>
          </div>
          <div className="list-item">
            <div className="list-title">Motion rule</div>
            <div className="list-body">Only the ticker tape moves. Everything else remains stable, measured, and data-first.</div>
          </div>
        </div>
      </div>
    </div>
  );
};
