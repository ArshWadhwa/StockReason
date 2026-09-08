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
        <div className="ledger-panel" style={{ height: '340px' }} />
      </div>
    );
  }

  const alert = sentiment.divergence_alert;
  const isToneBullish = sentiment.sentiment_label === 'bullish';
  const isToneBearish = sentiment.sentiment_label === 'bearish';

  const formatPublishDate = (dateStr: string) => {
    if (!dateStr) return 'Recent';
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) {
      return dateStr.slice(0, 16);
    }
    return d.toLocaleDateString('en-IN', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const cleanHeadline = (text: string) => {
    if (!text) return '';
    return text
      .replace(/[\s\u2014\u2013-]+score:\s*[+-]?\d+(\.\d+)?/gi, '')
      .trim();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Sentiment Overview Panel */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 style={{ fontFamily: 'var(--display)', fontSize: '24px', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>
              FinBERT Sentiment Trajectory
            </h2>
            <p style={{ fontSize: '13px', color: '#64748b', marginTop: '6px', margin: 0 }}>
              Domain-tuned NLP scoring and real-time financial news sentiment for <strong style={{ color: '#0f172a' }}>{sentiment.ticker}</strong>
            </p>
          </div>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {/* Daily NLP Score */}
            <div className="ledger-cell" style={{ minWidth: '130px', padding: '12px 18px' }}>
              <div className="ledger-label">FinBERT NLP Score</div>
              <div style={{
                fontSize: '26px',
                fontWeight: '700',
                fontFamily: 'var(--mono)',
                color: sentiment.avg_sentiment_score >= 0 ? '#16a34a' : '#dc2626',
                marginTop: '4px'
              }}>
                {sentiment.avg_sentiment_score >= 0 ? `+${sentiment.avg_sentiment_score.toFixed(2)}` : sentiment.avg_sentiment_score.toFixed(2)}
              </div>
            </div>

            {/* Overall Tone Classification */}
            <div className="ledger-cell" style={{ minWidth: '130px', padding: '12px 18px' }}>
              <div className="ledger-label">Sentiment Tone</div>
              <div style={{
                fontSize: '12px',
                fontWeight: '700',
                fontFamily: 'var(--mono)',
                marginTop: '8px',
                color: isToneBullish ? '#16a34a' : (isToneBearish ? '#dc2626' : '#b45309'),
                background: isToneBullish ? '#ecfdf5' : (isToneBearish ? '#fef2f2' : '#fefce8'),
                border: `1px solid ${isToneBullish ? '#a7f3d0' : (isToneBearish ? '#fecaca' : '#fde68a')}`,
                padding: '4px 10px',
                borderRadius: '20px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px'
              }}>
                <span>{isToneBullish ? '▲' : (isToneBearish ? '▼' : '◆')}</span>
                <span>{sentiment.sentiment_label.toUpperCase()}</span>
              </div>
            </div>

            {/* Bullish Ratio */}
            <div className="ledger-cell" style={{ minWidth: '130px', padding: '12px 18px' }}>
              <div className="ledger-label">Bullish Ratio</div>
              <div style={{
                fontSize: '26px',
                fontWeight: '700',
                fontFamily: 'var(--mono)',
                color: '#0f172a',
                marginTop: '4px'
              }}>
                {Math.round((sentiment.bullish_ratio || 0) * 100)}%
              </div>
            </div>
          </div>
        </div>

        {/* Divergence Alert */}
        <div style={{
          marginTop: '20px',
          padding: '14px 18px',
          borderRadius: '10px',
          background: alert.has_divergence ? '#fef2f2' : '#f0fdf4',
          border: `1px solid ${alert.has_divergence ? '#fecaca' : '#bbf7d0'}`
        }}>
          <div style={{
            fontWeight: '700',
            color: alert.has_divergence ? '#b91c1c' : '#1e4d1f',
            fontSize: '11px',
            fontFamily: 'var(--mono)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            letterSpacing: '0.04em'
          }}>
            <span>{alert.has_divergence ? '⚠️' : '✓'}</span>
            {alert.has_divergence ? 'SENTIMENT-PRICE DIVERGENCE FLAG' : 'SENTIMENT & TECHNICAL TREND CONVERGENCE'}
          </div>
          <div style={{ fontSize: '13px', color: '#475569', marginTop: '6px', lineHeight: '1.5' }}>
            {alert.message}
          </div>
        </div>
      </div>

      {/* Headlines Feed */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <div style={{ fontFamily: 'var(--display)', fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              Recent Financial News & Sentiment Assessment
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
              Financial RSS headlines parsed and evaluated with FinBERT NLP
            </div>
          </div>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: '11px',
            color: '#475569',
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            padding: '4px 10px',
            borderRadius: '6px',
            fontWeight: '600'
          }}>
            {sentiment.headlines.length} Articles Analyzed
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {sentiment.headlines.map((item) => {
            const isPos = item.sentiment_label === 'positive';
            const isNeg = item.sentiment_label === 'negative';
            const badgeColor = isPos ? '#15803d' : (isNeg ? '#b91c1c' : '#b45309');
            const badgeBg = isPos ? '#ecfdf5' : (isNeg ? '#fef2f2' : '#fefce8');
            const badgeBorder = isPos ? '#a7f3d0' : (isNeg ? '#fecaca' : '#fde68a');

            return (
              <div
                key={item.id}
                style={{
                  padding: '16px 18px',
                  border: '1px solid #e2e8f0',
                  background: '#ffffff',
                  borderRadius: '12px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '16px',
                  flexWrap: 'wrap',
                  boxShadow: '0 1px 3px rgba(18, 50, 24, 0.02)',
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ flex: '1 1 400px' }}>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '6px', lineHeight: '1.45' }}>
                    {cleanHeadline(item.headline)}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: '#64748b', fontFamily: 'var(--mono)' }}>
                    <span style={{ fontWeight: '600', color: '#334155' }}>{item.source}</span>
                    <span style={{ opacity: 0.4 }}>•</span>
                    <span>{formatPublishDate(item.published_at)}</span>
                  </div>
                </div>

                {/* Single Non-Redundant Sentiment Pill */}
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '6px 14px',
                  borderRadius: '20px',
                  fontFamily: 'var(--mono)',
                  fontSize: '12px',
                  fontWeight: '700',
                  color: badgeColor,
                  background: badgeBg,
                  border: `1px solid ${badgeBorder}`,
                  whiteSpace: 'nowrap'
                }}>
                  <span style={{ fontSize: '10px' }}>{isPos ? '▲' : (isNeg ? '▼' : '◆')}</span>
                  <span>{item.sentiment_label.toUpperCase()}</span>
                  <span style={{ opacity: 0.4 }}>·</span>
                  <span>{item.sentiment_score >= 0 ? `+${item.sentiment_score.toFixed(2)}` : item.sentiment_score.toFixed(2)}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
