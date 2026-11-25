import React, { useState } from 'react';
import type { StockRiskProfile } from '../types';
import { ShieldAlert, ShieldCheck, Activity } from 'lucide-react';

interface RiskVolatilityViewProps {
  risk: StockRiskProfile | null;
  loading: boolean;
}

export const RiskVolatilityView: React.FC<RiskVolatilityViewProps> = ({ risk, loading }) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [showTrailing, setShowTrailing] = useState(true);
  const [showGarch, setShowGarch] = useState(true);
  const [showHar, setShowHar] = useState(true);

  if (loading || !risk) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel skeleton" style={{ height: '140px' }} />
        <div className="ledger-panel skeleton" style={{ height: '280px' }} />
        <div className="ledger-panel skeleton" style={{ height: '220px' }} />
      </div>
    );
  }

  const series = risk.volatility_series || [];
  const width = 880;
  const height = 240;
  const padL = 10;
  const padR = 54;
  const usableW = width - padL - padR;

  const allVals = series.flatMap((p) => {
    const arr = [p.ewma_vol_pct];
    if (showTrailing) arr.push(p.trailing_60d_vol_pct);
    if (showGarch) arr.push(p.garch_vol_pct);
    if (showHar) arr.push(p.har_rv_vol_pct);
    return arr;
  });
  const minV = Math.max(0, Math.min(...(allVals.length ? allVals : [10])) * 0.85);
  const maxV = Math.max(...(allVals.length ? allVals : [30])) * 1.08;
  const rangeV = maxV - minV || 1;

  const toPts = (getter: (p: typeof series[number]) => number) =>
    series
      .map((p, idx) => {
        const x = padL + (idx / Math.max(1, series.length - 1)) * usableW;
        const y = height - ((getter(p) - minV) / rangeV) * height;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  const activeIdx = hoverIdx !== null && hoverIdx < series.length ? hoverIdx : Math.max(0, series.length - 1);
  const activePoint = series[activeIdx];
  const activeX = padL + (activeIdx / Math.max(1, series.length - 1)) * usableW;

  const qlikes = risk.qlike_loss;
  const validQlikes = Object.entries(qlikes).filter(([, v]) => v !== null && v !== undefined) as Array<[string, number]>;
  const bestQlikeKey = validQlikes.length > 0
    ? validQlikes.reduce((best, curr) => (curr[1] < best[1] ? curr : best))[0]
    : null;

  const prob = risk.probability_calibration;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Header & Volatility Model Suite */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '18px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} style={{ color: 'var(--accent)' }} />
              <h2 style={{ fontFamily: 'var(--sans)', fontSize: '20px', fontWeight: '700', color: 'var(--ink)', margin: 0 }}>
                Econometric Volatility & Conformal Risk
              </h2>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '4px' }}>
              Annualized decimal-return volatility models & finite-sample split-conformal intervals for{' '}
              <strong style={{ color: 'var(--ink)' }}>{risk.ticker.replace('.NS', '')}</strong> · Horizon {risk.horizon_days}D (Target {risk.target_trading_date})
            </p>
          </div>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: '11px',
            background: 'var(--paper)',
            border: '1px solid var(--rule)',
            padding: '4px 10px',
            borderRadius: '6px',
            color: 'var(--dash-muted)',
            fontWeight: '600',
          }}>
            As-of: {risk.as_of_date} · Close ₹{risk.current_price.toLocaleString('en-IN')}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '12px' }}>
          {[
            { key: 'ewma_94', label: 'Recursive EWMA (λ=0.94)', val: risk.volatility_latest.ewma_94_pct, q: qlikes.ewma_94, color: 'var(--accent)' },
            { key: 'garch_11', label: 'GARCH(1,1) Constrained MLE', val: risk.volatility_latest.garch_11_pct, q: qlikes.garch_11, color: '#1A7F4B' },
            { key: 'har_rv', label: 'Fitted HAR-RV', val: risk.volatility_latest.har_rv_pct, q: qlikes.har_rv, color: '#7C3AED' },
            { key: 'trailing_60d', label: 'Trailing 60D Realized', val: risk.volatility_latest.trailing_60d_pct, q: qlikes.trailing_60d, color: '#64748b' },
          ].map((item) => {
            const isBest = bestQlikeKey === item.key;
            return (
              <div key={item.key} className="ledger-cell" style={{ borderTop: `3px solid ${item.color}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
                  <div className="ledger-label">{item.label}</div>
                  {isBest && (
                    <span style={{
                      fontSize: '9px',
                      fontFamily: 'var(--mono)',
                      fontWeight: '700',
                      background: 'var(--gain-bg)',
                      color: 'var(--gain)',
                      border: '1px solid var(--gain-border)',
                      padding: '1px 6px',
                      borderRadius: '4px',
                    }}>
                      BEST QLIKE
                    </span>
                  )}
                </div>
                <div style={{ fontFamily: 'var(--serif)', fontSize: '28px', color: 'var(--ink)', marginTop: '6px', lineHeight: '1.1' }}>
                  {item.val.toFixed(2)}%
                </div>
                <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)', marginTop: '6px' }}>
                  OOS QLIKE: <strong style={{ color: 'var(--ink)' }}>{item.q !== null && item.q !== undefined ? item.q.toFixed(4) : 'N/A'}</strong>
                </div>
              </div>
            );
          })}
        </div>

        {/* GARCH & HAR-RV Parameter Summary */}
        <div style={{
          marginTop: '14px',
          padding: '10px 14px',
          background: 'var(--paper)',
          borderRadius: '8px',
          border: '1px solid var(--rule)',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '18px',
          fontSize: '11px',
          fontFamily: 'var(--mono)',
          color: 'var(--dash-muted)',
        }}>
          <div>
            GARCH(1,1): <strong style={{ color: 'var(--ink)' }}>α={risk.garch_params.alpha.toFixed(4)}</strong>,{' '}
            <strong style={{ color: 'var(--ink)' }}>β={risk.garch_params.beta.toFixed(4)}</strong>,{' '}
            persistence <strong style={{ color: risk.garch_params.persistence < 0.99 ? 'var(--gain)' : 'var(--hold)' }}>
              {risk.garch_params.persistence.toFixed(4)}
            </strong>{' '}
            ({risk.garch_params.converged ? 'MLE converged' : 'fallback'})
          </div>
          <div>
            HAR-RV weights: <strong style={{ color: 'var(--ink)' }}>D={risk.har_rv_coefs.daily.toFixed(3)}</strong> ·{' '}
            <strong style={{ color: 'var(--ink)' }}>W={risk.har_rv_coefs.weekly.toFixed(3)}</strong> ·{' '}
            <strong style={{ color: 'var(--ink)' }}>M={risk.har_rv_coefs.monthly.toFixed(3)}</strong>
          </div>
        </div>
      </div>

      {/* Volatility Trajectory Chart */}
      {series.length > 0 && (
        <div className="ledger-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '14px' }}>
            <div>
              <div className="ledger-panel-title">Annualized Volatility Term-Structure (120 Sessions)</div>
              <div style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '2px' }}>
                Comparing EWMA (λ=0.94) against GARCH(1,1), HAR-RV, and 60-day realized volatility
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', fontSize: '11px', fontFamily: 'var(--mono)' }}>
              <span style={{ padding: '4px 9px', borderRadius: '6px', background: 'var(--accent-subtle)', color: 'var(--accent)', fontWeight: '700' }}>
                ── EWMA
              </span>
              <button
                type="button"
                onClick={() => setShowGarch(v => !v)}
                style={{
                  padding: '4px 9px',
                  borderRadius: '6px',
                  border: '1px solid var(--rule)',
                  background: showGarch ? '#ecfdf5' : '#ffffff',
                  color: showGarch ? '#1A7F4B' : 'var(--dash-muted)',
                  cursor: 'pointer',
                  fontFamily: 'var(--mono)',
                  fontWeight: '600',
                }}
              >
                {showGarch ? '✓' : '+'} GARCH(1,1)
              </button>
              <button
                type="button"
                onClick={() => setShowHar(v => !v)}
                style={{
                  padding: '4px 9px',
                  borderRadius: '6px',
                  border: '1px solid var(--rule)',
                  background: showHar ? '#f5f3ff' : '#ffffff',
                  color: showHar ? '#7C3AED' : 'var(--dash-muted)',
                  cursor: 'pointer',
                  fontFamily: 'var(--mono)',
                  fontWeight: '600',
                }}
              >
                {showHar ? '✓' : '+'} HAR-RV
              </button>
              <button
                type="button"
                onClick={() => setShowTrailing(v => !v)}
                style={{
                  padding: '4px 9px',
                  borderRadius: '6px',
                  border: '1px solid var(--rule)',
                  background: showTrailing ? '#f1f5f9' : '#ffffff',
                  color: showTrailing ? '#334155' : 'var(--dash-muted)',
                  cursor: 'pointer',
                  fontFamily: 'var(--mono)',
                  fontWeight: '600',
                }}
              >
                {showTrailing ? '✓' : '+'} Trailing 60D
              </button>
            </div>
          </div>

          <div
            style={{ position: 'relative', width: '100%', height: `${height}px`, background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '10px', overflow: 'hidden' }}
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const relX = (e.clientX - rect.left - padL) / usableW;
              const idx = Math.min(series.length - 1, Math.max(0, Math.round(relX * (series.length - 1))));
              setHoverIdx(idx);
            }}
            onMouseLeave={() => setHoverIdx(null)}
          >
            <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '100%', display: 'block' }} preserveAspectRatio="none">
              {[0.25, 0.5, 0.75].map((r, i) => {
                const y = height * r;
                const val = maxV - r * rangeV;
                return (
                  <g key={i}>
                    <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#f1f5f9" strokeWidth="1" strokeDasharray="4 4" />
                    <text x={width - padR + 6} y={y + 3} fill="#94a3b8" fontSize="10" fontFamily="var(--mono)">
                      {val.toFixed(1)}%
                    </text>
                  </g>
                );
              })}

              {showTrailing && (
                <polyline fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="4 3" points={toPts((p) => p.trailing_60d_vol_pct)} />
              )}
              {showHar && (
                <polyline fill="none" stroke="#7C3AED" strokeWidth="1.7" points={toPts((p) => p.har_rv_vol_pct)} />
              )}
              {showGarch && (
                <polyline fill="none" stroke="#1A7F4B" strokeWidth="2" points={toPts((p) => p.garch_vol_pct)} />
              )}
              <polyline fill="none" stroke="#3B6FD4" strokeWidth="2.3" points={toPts((p) => p.ewma_vol_pct)} />

              {hoverIdx !== null && (
                <line x1={activeX} y1={0} x2={activeX} y2={height} stroke="#3B6FD4" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.6" />
              )}
            </svg>
          </div>

          {activePoint && (
            <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px', marginTop: '10px', fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--dash-muted)' }}>
              <div>Date: <strong style={{ color: 'var(--ink)' }}>{activePoint.date}</strong></div>
              <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
                <span>EWMA: <strong style={{ color: '#3B6FD4' }}>{activePoint.ewma_vol_pct.toFixed(2)}%</strong></span>
                <span>GARCH: <strong style={{ color: '#1A7F4B' }}>{activePoint.garch_vol_pct.toFixed(2)}%</strong></span>
                <span>HAR-RV: <strong style={{ color: '#7C3AED' }}>{activePoint.har_rv_vol_pct.toFixed(2)}%</strong></span>
                <span>Trail-60D: <strong style={{ color: '#475569' }}>{activePoint.trailing_60d_vol_pct.toFixed(2)}%</strong></span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Split-Conformal Prediction Intervals & Downside Probability Gate */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
          <div>
            <div className="ledger-panel-title">Split-Conformal 21D Price & Return Intervals</div>
            <div style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '3px' }}>
              Finite-sample order-statistic calibration scaled by EWMA volatility (no arbitrary clamping). Center prior: No-Change Random Walk (₹{risk.current_price.toLocaleString('en-IN')}).
            </div>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px', fontFamily: 'var(--mono)', fontVariantNumeric: 'tabular-nums' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--rule)', background: 'var(--paper)', color: 'var(--dash-muted)' }}>
                <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Nominal</th>
                <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Empirical Cov.</th>
                <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>21D Return Band</th>
                <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Calibrated Price Band (₹)</th>
                <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Winkler Score</th>
                <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Tail Miss (L / U)</th>
              </tr>
            </thead>
            <tbody>
              {risk.conformal_intervals.map((iv) => {
                const covOk = Math.abs(iv.coverage_error) <= 0.08;
                return (
                  <tr key={iv.coverage_label} style={{ borderBottom: '1px solid var(--rule)' }}>
                    <td style={{ padding: '11px 12px', fontWeight: '700', color: 'var(--ink)' }}>{iv.coverage_label}</td>
                    <td style={{ padding: '11px 12px', color: covOk ? 'var(--gain)' : 'var(--hold)', fontWeight: '600' }}>
                      {(iv.empirical_coverage * 100).toFixed(1)}%{' '}
                      <span style={{ fontSize: '10px', color: 'var(--dash-muted)', fontWeight: '400' }}>
                        ({iv.coverage_error >= 0 ? '+' : ''}{(iv.coverage_error * 100).toFixed(1)} pp)
                      </span>
                    </td>
                    <td style={{ padding: '11px 12px', color: '#334155' }}>
                      [{(iv.return_lower * 100).toFixed(1)}%, +{(iv.return_upper * 100).toFixed(1)}%]
                    </td>
                    <td style={{ padding: '11px 12px', color: 'var(--ink)', fontWeight: '600' }}>
                      ₹{iv.price_lower.toLocaleString('en-IN')} – ₹{iv.price_upper.toLocaleString('en-IN')}
                    </td>
                    <td style={{ padding: '11px 12px', color: 'var(--dash-muted)' }}>{iv.winkler_score.toFixed(4)}</td>
                    <td style={{ padding: '11px 12px', color: 'var(--dash-muted)' }}>
                      {(iv.miss_lower_tail * 100).toFixed(1)}% / {(iv.miss_upper_tail * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Downside Probability Calibration Gate */}
        <div style={{
          marginTop: '18px',
          padding: '14px 16px',
          borderRadius: '10px',
          background: prob.is_calibrated ? 'var(--gain-bg)' : 'var(--hold-bg)',
          border: `1px solid ${prob.is_calibrated ? 'var(--gain-border)' : 'var(--hold-border)'}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '14px',
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', maxWidth: '680px' }}>
            {prob.is_calibrated ? (
              <ShieldCheck size={20} style={{ color: 'var(--gain)', flexShrink: 0, marginTop: '2px' }} />
            ) : (
              <ShieldAlert size={20} style={{ color: 'var(--hold)', flexShrink: 0, marginTop: '2px' }} />
            )}
            <div>
              <div style={{ fontSize: '13px', fontWeight: '700', color: 'var(--ink)' }}>
                Downside Risk Probability Gate (&lt; −5% in 21D):{' '}
                <span style={{ fontFamily: 'var(--mono)', color: prob.is_calibrated ? 'var(--gain)' : 'var(--hold)' }}>
                  {prob.action}
                </span>
              </div>
              <div style={{ fontSize: '12px', color: '#475569', marginTop: '3px', lineHeight: '1.5' }}>
                {prob.is_calibrated
                  ? 'Model probabilities improve upon the historical downside base rate (Brier Skill Score > 0).'
                  : 'Brier Skill Score ≤ 0 relative to the historical downside base rate. Directional probabilities are suppressed to prevent displaying overconfident estimates.'}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--mono)', fontSize: '11px' }}>
            <div>
              <div style={{ color: 'var(--dash-muted)' }}>Brier Skill Score</div>
              <strong style={{ fontSize: '15px', color: prob.brier_skill_score > 0 ? 'var(--gain)' : 'var(--loss)' }}>
                {prob.brier_skill_score.toFixed(4)}
              </strong>
            </div>
            <div>
              <div style={{ color: 'var(--dash-muted)' }}>Model / Ref Brier</div>
              <strong style={{ fontSize: '13px', color: 'var(--ink)' }}>
                {prob.brier_score_model !== null ? prob.brier_score_model.toFixed(4) : 'N/A'} / {prob.brier_score_reference !== null ? prob.brier_score_reference.toFixed(4) : 'N/A'}
              </strong>
            </div>
            <div>
              <div style={{ color: 'var(--dash-muted)' }}>Hist. Base Rate</div>
              <strong style={{ fontSize: '13px', color: 'var(--ink)' }}>
                {prob.base_rate !== null ? `${(prob.base_rate * 100).toFixed(1)}%` : 'N/A'}
              </strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
