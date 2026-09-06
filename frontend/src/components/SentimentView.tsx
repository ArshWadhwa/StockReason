import React from 'react';
import type { SentimentData } from '../types';

interface SentimentViewProps {
  sentiment: SentimentData | null;
  loading: boolean;
}

export const SentimentView: React.FC<SentimentViewProps> = ({ sentiment, loading }) => {
  if (loading || !sentiment) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel" style={{ height: '140px' }} />
        <div className="ledger-panel" style={{ height: '300px' }} />
      </div>
    );
  }

  const alert = sentiment.divergence_alert;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Sentiment Overview Panel */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '24px', fontWeight: '700', color: 'var(--text-parchment)' }}>
              FinBERT Sentiment Trajectory
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Financial-domain NLP scoring for {sentiment.ticker}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '14px' }}>
            <div className="ledger-cell" style={{ minWidth: '130px', padding: '10px 14px' }}>
              <div className="ledger-label">Daily Score</div>
              <div style={{
                fontSize: '22px',
                fontWeight: '600',
                fontFamily: 'var(--font-data)',
                color: sentiment.avg_sentiment_score >= 0 ? 'var(--brass-amber)' : 'var(--rust-avoid)'
              }}>
                {sentiment.avg_sentiment_score >= 0 ? `+${sentiment.avg_sentiment_score.toFixed(2)}` : sentiment.avg_sentiment_score.toFixed(2)}
              </div>
            </div>

            <div className="ledger-cell" style={{ minWidth: '130px', padding: '10px 14px' }}>
              <div className="ledger-label">Tone</div>
              <div style={{
                fontSize: '12px',
                fontWeight: '600',
                fontFamily: 'var(--font-data)',
                marginTop: '6px',
                color: sentiment.sentiment_label === 'bullish' ? 'var(--brass-amber)' : (sentiment.sentiment_label === 'bearish' ? 'var(--rust-avoid)' : 'var(--hold-gold)')
              }}>
                {sentiment.sentiment_label.toUpperCase()}
              </div>
            </div>
          </div>
        </div>

        {/* Divergence Alert */}
        <div style={{
          marginTop: '18px',
          padding: '12px 16px',
          background: alert.has_divergence ? 'var(--rust-bg)' : 'var(--slate-subtle)',
          border: `1px solid ${alert.has_divergence ? 'var(--rust-border)' : 'var(--line-strong)'}`
        }}>
          <div style={{ fontWeight: '700', color: alert.has_divergence ? 'var(--rust-avoid)' : 'var(--slate-blue)', fontSize: '11px', fontFamily: 'var(--font-data)' }}>
            {alert.has_divergence ? '▲ SENTIMENT-PRICE DIVERGENCE FLAG' : '✓ SENTIMENT & TECHNICAL TREND CONVERGENCE'}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {alert.message}
          </div>
        </div>
      </div>

      {/* Headlines Feed */}
      <div className="ledger-panel">
        <div className="ledger-panel-title" style={{ marginBottom: '14px' }}>
          Recent Financial News Headlines & FinBERT Scores
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {sentiment.headlines.map((item) => (
            <div
              key={item.id}
              style={{
                padding: '14px',
                border: '1px solid var(--line-grid)',
                background: 'var(--bg-ink)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '14px',
                flexWrap: 'wrap'
              }}
            >
              <div style={{ flex: '1 1 350px' }}>
                <div style={{ fontSize: '14px', color: 'var(--text-parchment)', marginBottom: '4px' }}>
                  {item.headline}
                </div>
                <div style={{ display: 'flex', gap: '10px', fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--font-data)' }}>
                  <span>{item.source}</span>
                  <span>•</span>
                  <span>{new Date(item.published_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <span className={`ledger-tag ${item.sentiment_label === 'positive' ? 'tag-buy' : (item.sentiment_label === 'negative' ? 'tag-avoid' : 'tag-hold')}`}>
                  {item.sentiment_label.toUpperCase()}
                </span>
                <div style={{
                  fontFamily: 'var(--font-data)',
                  fontWeight: '600',
                  fontSize: '13px',
                  color: item.sentiment_score >= 0 ? 'var(--brass-amber)' : 'var(--rust-avoid)'
                }}>
                  {item.sentiment_score >= 0 ? `+${item.sentiment_score.toFixed(2)}` : item.sentiment_score.toFixed(2)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
