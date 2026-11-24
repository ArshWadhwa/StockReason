import React, { useEffect, useState } from 'react';
import type { ModelReport } from '../types';
import { api } from '../api';
import { ShieldCheck, AlertTriangle, CheckCircle2 } from 'lucide-react';

export const ModelReliabilityPanel: React.FC = () => {
  const [report, setReport] = useState<ModelReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadReport() {
      try {
        setLoading(true);
        const data = await api.getModelReport();
        setReport(data);
      } catch (err: any) {
        console.error('Failed to load model reliability report:', err);
        setError('Could not fetch model reliability report');
      } finally {
        setLoading(false);
      }
    }
    loadReport();
  }, []);

  if (loading) {
    return <div className="ledger-panel skeleton" style={{ height: '240px' }} />;
  }

  if (error || !report) {
    return null;
  }

  const m = report.metrics;
  const ci = m.rank_ic_ci_95 || [0, 0];
  const hasEdge = report.has_edge;

  return (
    <div className="ledger-panel" style={{
      borderLeft: `4px solid ${hasEdge ? '#1A7F4B' : '#EAB308'}`,
      borderRadius: '12px',
      marginTop: '16px',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {hasEdge ? (
            <ShieldCheck size={22} style={{ color: '#1A7F4B' }} />
          ) : (
            <AlertTriangle size={22} style={{ color: '#EAB308' }} />
          )}
          <div>
            <div style={{ fontFamily: 'var(--sans)', fontSize: '16px', fontWeight: '700', color: 'var(--ink)' }}>
              Model Reliability & Quantitative Verification
            </div>
            <div style={{ fontSize: '11px', color: 'var(--dash-muted)', marginTop: '2px' }}>
              Engine version: <strong>{report.model_name}</strong> · As of: <strong>{report.as_of_date}</strong>
            </div>
          </div>
        </div>

        {/* Verdict Badge */}
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 12px',
          borderRadius: '20px',
          fontSize: '12px',
          fontWeight: '600',
          fontFamily: 'var(--sans)',
          background: hasEdge ? '#ecfdf5' : '#fefce8',
          color: hasEdge ? '#065f46' : '#854d0e',
          border: `1px solid ${hasEdge ? '#a7f3d0' : '#fef08a'}`,
        }}>
          {hasEdge ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
          <span>{report.verdict}</span>
        </div>
      </div>

      {/* Grid of Key Reliability Metrics */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '12px',
        marginBottom: '16px',
      }}>
        {/* Metric 1: Rank IC */}
        <div className="ledger-cell">
          <div className="ledger-label">Rolling Out-of-Sample Rank IC</div>
          <div style={{
            fontSize: '22px',
            fontFamily: 'var(--serif)',
            color: m.rank_ic > 0 ? '#1A7F4B' : '#DC2626',
            marginTop: '4px',
          }}>
            {m.rank_ic.toFixed(4)}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--dash-muted)', marginTop: '2px', fontFamily: 'var(--mono)' }}>
            95% CI: [{ci[0].toFixed(3)}, {ci[1].toFixed(3)}]
          </div>
        </div>

        {/* Metric 2: Rank ICIR */}
        <div className="ledger-cell">
          <div className="ledger-label">Annualized Rank ICIR</div>
          <div style={{
            fontSize: '22px',
            fontFamily: 'var(--serif)',
            color: 'var(--ink)',
            marginTop: '4px',
          }}>
            {m.rank_icir.toFixed(2)}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--dash-muted)', marginTop: '2px' }}>
            Information Ratio (ann.)
          </div>
        </div>

        {/* Metric 3: Directional Accuracy vs Base */}
        <div className="ledger-cell">
          <div className="ledger-label">Dir Acc vs Base Rate</div>
          <div style={{
            fontSize: '22px',
            fontFamily: 'var(--serif)',
            color: m.dir_acc_vs_base >= 0 ? '#1A7F4B' : '#DC2626',
            marginTop: '4px',
          }}>
            {m.dir_acc_vs_base >= 0 ? '+' : ''}{(m.dir_acc_vs_base * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--dash-muted)', marginTop: '2px' }}>
            Raw: {(m.dir_acc * 100).toFixed(1)}%
          </div>
        </div>

        {/* Metric 4: Calibration ECE */}
        <div className="ledger-cell">
          <div className="ledger-label">Expected Calibration Error</div>
          <div style={{
            fontSize: '22px',
            fontFamily: 'var(--serif)',
            color: report.calibration.ece <= 0.08 ? '#1A7F4B' : '#B45309',
            marginTop: '4px',
          }}>
            {report.calibration.ece.toFixed(4)}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--dash-muted)', marginTop: '2px' }}>
            Brier Score: {m.brier !== undefined && m.brier !== null ? m.brier.toFixed(4) : report.calibration.brier.toFixed(4)}
          </div>
        </div>
      </div>

      {/* Regime and Sector breakdown accordion / summary */}
      {m.by_regime && Object.keys(m.by_regime).length > 0 && (
        <div style={{
          padding: '10px 14px',
          background: 'var(--paper)',
          borderRadius: '8px',
          fontSize: '12px',
          color: '#475569',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
          alignItems: 'center',
        }}>
          <div style={{ fontWeight: '600', color: 'var(--ink)' }}>Regime Breakdowns:</div>
          {Object.entries(m.by_regime).map(([reg, data]: [string, any]) => (
            <div key={reg} style={{ display: 'flex', gap: '6px' }}>
              <span style={{ fontWeight: '500' }}>{reg}:</span>
              <span style={{ fontFamily: 'var(--mono)', color: data.rank_ic > 0 ? '#1A7F4B' : '#DC2626' }}>
                Rank IC {data.rank_ic.toFixed(3)}
              </span>
              <span style={{ color: 'var(--dash-muted)' }}>({data.n} rows)</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
