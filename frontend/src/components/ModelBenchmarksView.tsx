import React, { useEffect, useState } from 'react';
import { api } from '../api';
import type {
  ResearchBenchmarksData,
  FeatureRegistryData,
  ModelManifestData,
} from '../types';
import { Award, Layers, GitCommit, Image as ImageIcon, X } from 'lucide-react';

export const ModelBenchmarksView: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<ResearchBenchmarksData | null>(null);
  const [features, setFeatures] = useState<FeatureRegistryData | null>(null);
  const [manifestData, setManifestData] = useState<ModelManifestData | null>(null);
  const [loading, setLoading] = useState(true);
  const [categoryFilter, setCategoryFilter] = useState<string>('ALL');
  const [expandedFamily, setExpandedFamily] = useState<string | null>('F0');
  const [lightboxUrl, setLightboxUrl] = useState<{ url: string; title: string } | null>(null);

  useEffect(() => {
    async function loadResearch() {
      try {
        setLoading(true);
        const [b, f, m] = await Promise.all([
          api.getResearchBenchmarks(),
          api.getResearchFeatures(),
          api.getResearchManifest(),
        ]);
        setBenchmarks(b);
        setFeatures(f);
        setManifestData(m);
      } catch (err) {
        console.error('Failed to load research benchmarks:', err);
      } finally {
        setLoading(false);
      }
    }
    loadResearch();
  }, []);

  if (loading || !benchmarks) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel skeleton" style={{ height: '140px' }} />
        <div className="ledger-panel skeleton" style={{ height: '380px' }} />
        <div className="ledger-panel skeleton" style={{ height: '240px' }} />
      </div>
    );
  }

  const categories = ['ALL', ...Array.from(new Set(benchmarks.model_results.map((m) => m.category)))];
  const filteredModels =
    categoryFilter === 'ALL'
      ? benchmarks.model_results
      : benchmarks.model_results.filter((m) => m.category === categoryFilter);

  const regimeEntries = Object.entries(benchmarks.regime_results || {}).filter(
    ([k]) => k !== 'All_Observations'
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Research Cohort Summary */}
      <div className="ledger-panel" style={{ borderLeft: '4px solid var(--accent)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Award size={18} style={{ color: 'var(--accent)' }} />
              <h2 style={{ fontFamily: 'var(--sans)', fontSize: '20px', fontWeight: '700', color: 'var(--ink)', margin: 0 }}>
                15-Model Empirical Evaluation & Research Audit
              </h2>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '4px' }}>
              Out-of-sample 2024–2026 diagnostic holdout across{' '}
              <strong style={{ color: 'var(--ink)' }}>{benchmarks.evaluation_cohort_obs.toLocaleString('en-IN')}</strong>{' '}
              matched NSE observations (21-trading-day horizon).
            </p>
          </div>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: '11px',
            fontWeight: '700',
            padding: '4px 10px',
            borderRadius: '6px',
            background: 'var(--hold-bg)',
            color: 'var(--hold)',
            border: '1px solid var(--hold-border)',
          }}>
            HURDLE GATE: 2.0% MIN MAPE REDUCTION
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
          <div className="ledger-cell">
            <div className="ledger-label">Promoted Price Baseline</div>
            <div style={{ fontFamily: 'var(--serif)', fontSize: '26px', color: 'var(--ink)', marginTop: '4px' }}>
              ₹{benchmarks.no_change_reference.inr_price_mae.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)', marginTop: '4px' }}>
              No-Change RW · {benchmarks.no_change_reference.true_price_mape.toFixed(4)}% MAPE
            </div>
          </div>

          <div className="ledger-cell">
            <div className="ledger-label">Best ML Price Reduction</div>
            <div style={{ fontFamily: 'var(--serif)', fontSize: '26px', color: 'var(--hold)', marginTop: '4px' }}>
              +0.113%
            </div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)', marginTop: '4px' }}>
              LightGBM Direct Log (F0) · Sub-Hurdle (&lt;2%)
            </div>
          </div>

          <div className="ledger-cell">
            <div className="ledger-label">Cross-Sectional Ranking Alpha</div>
            <div style={{ fontFamily: 'var(--serif)', fontSize: '26px', color: 'var(--gain)', marginTop: '4px' }}>
              +0.0732
            </div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)', marginTop: '4px' }}>
              Mean Daily Rank IC · ICIR 0.9836
            </div>
          </div>

          <div className="ledger-cell">
            <div className="ledger-label">High-VIX Directional Edge</div>
            <div style={{ fontFamily: 'var(--serif)', fontSize: '26px', color: 'var(--accent)', marginTop: '4px' }}>
              62.12%
            </div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)', marginTop: '4px' }}>
              India VIX ≥ 20 regime (50,013 rows)
            </div>
          </div>
        </div>
      </div>

      {/* 15-Model Comparison Table */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '14px' }}>
          <div>
            <div className="ledger-panel-title">Model & Ablation Comparison Table</div>
            <div style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '2px' }}>
              Direct empirical audit comparing statistical baselines, feature ablations, target formulations, and adaptive recency weighting
            </div>
          </div>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {categories.map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setCategoryFilter(cat)}
                style={{
                  padding: '4px 10px',
                  borderRadius: '6px',
                  border: `1px solid ${categoryFilter === cat ? 'var(--accent)' : 'var(--rule)'}`,
                  background: categoryFilter === cat ? 'var(--accent)' : '#ffffff',
                  color: categoryFilter === cat ? '#ffffff' : 'var(--dash-muted)',
                  fontSize: '11px',
                  fontFamily: 'var(--sans)',
                  fontWeight: '600',
                  cursor: 'pointer',
                }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px', fontFamily: 'var(--mono)', fontVariantNumeric: 'tabular-nums' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--rule)', background: 'var(--paper)', color: 'var(--dash-muted)' }}>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>Model</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>Category</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>Feats</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>MAPE (%)</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>MAE (₹)</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>Rel. vs RW</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>Dir Acc</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>Rank IC</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>ICIR</th>
                <th style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px' }}>Hurdle</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m) => {
                const dm = m.diag_metrics || {};
                const rel = m.rel_error_red_vs_no_change_pct ?? 0;
                const ic = dm.mean_rank_ic ?? 0;
                const isBestIc = m.model_name === 'F8_LightGBM_direct_log';
                const isRw = m.model_name.includes('No-Change');
                return (
                  <tr
                    key={m.model_name}
                    style={{
                      borderBottom: '1px solid var(--rule)',
                      background: isRw ? '#f8fafc' : isBestIc ? '#f0fdf4' : 'transparent',
                    }}
                  >
                    <td style={{ padding: '10px', fontFamily: 'var(--sans)', fontWeight: '600', color: 'var(--ink)' }}>
                      {m.model_name}
                    </td>
                    <td style={{ padding: '10px', fontFamily: 'var(--sans)', fontSize: '11px', color: 'var(--dash-muted)' }}>
                      {m.category}
                    </td>
                    <td style={{ padding: '10px', color: 'var(--dash-muted)' }}>{m.num_features}</td>
                    <td style={{ padding: '10px', fontWeight: '600', color: 'var(--ink)' }}>
                      {dm.true_price_mape?.toFixed(4)}%
                    </td>
                    <td style={{ padding: '10px', color: 'var(--ink)' }}>₹{dm.inr_price_mae?.toFixed(2)}</td>
                    <td style={{ padding: '10px', color: rel > 0 ? 'var(--gain)' : (rel < 0 ? 'var(--loss)' : 'var(--dash-muted)'), fontWeight: '600' }}>
                      {rel > 0 ? '+' : ''}{rel.toFixed(3)}%
                    </td>
                    <td style={{ padding: '10px', color: 'var(--ink)' }}>{dm.directional_accuracy_pct?.toFixed(2)}%</td>
                    <td style={{ padding: '10px', color: ic > 0.05 ? 'var(--gain)' : (ic > 0 ? 'var(--ink)' : 'var(--dash-muted)'), fontWeight: '700' }}>
                      {ic >= 0 ? '+' : ''}{ic.toFixed(4)}
                    </td>
                    <td style={{ padding: '10px', color: 'var(--dash-muted)' }}>{dm.rank_ic_ir?.toFixed(3)}</td>
                    <td style={{ padding: '10px' }}>
                      <span style={{
                        fontSize: '10px',
                        padding: '2px 7px',
                        borderRadius: '4px',
                        fontWeight: '700',
                        background: isRw ? '#e2e8f0' : 'var(--hold-bg)',
                        color: isRw ? '#334155' : 'var(--hold)',
                      }}>
                        {isRw ? 'BENCHMARK' : m.hurdle_assessment}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Regime-Conditional Performance Matrix */}
      <div className="ledger-panel">
        <div className="ledger-panel-title" style={{ marginBottom: '4px' }}>Regime-Conditional Performance Matrix</div>
        <div style={{ fontSize: '12px', color: 'var(--dash-muted)', marginBottom: '16px' }}>
          Model behavior segmented by India VIX volatility regime and NSE market breadth across the 2024–2026 cohort
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
          {regimeEntries.map(([regName, r]) => {
            const isHighVix = regName.includes('High_VIX');
            return (
              <div
                key={regName}
                className="ledger-cell"
                style={{ borderTop: `3px solid ${isHighVix ? 'var(--loss)' : 'var(--accent)'}` }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', fontWeight: '700', color: 'var(--ink)' }}>{regName.replace(/_/g, ' ')}</span>
                  <span style={{ fontSize: '10px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)' }}>
                    {r.pct_of_cohort.toFixed(1)}%
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', fontSize: '12px', fontFamily: 'var(--mono)' }}>
                  <div>
                    <div style={{ fontSize: '10px', color: 'var(--dash-muted)' }}>Dir. Acc</div>
                    <strong style={{ color: r.best_model_dir_acc >= 55 ? 'var(--gain)' : 'var(--ink)' }}>
                      {r.best_model_dir_acc.toFixed(2)}%
                    </strong>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', color: 'var(--dash-muted)' }}>Rank IC</div>
                    <strong style={{ color: r.best_model_rank_ic >= 0 ? 'var(--gain)' : 'var(--loss)' }}>
                      {r.best_model_rank_ic >= 0 ? '+' : ''}{r.best_model_rank_ic.toFixed(4)}
                    </strong>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', color: 'var(--dash-muted)' }}>MAPE</div>
                    <strong style={{ color: 'var(--ink)' }}>{r.best_model_mape.toFixed(2)}%</strong>
                  </div>
                </div>
                <div style={{ fontSize: '10px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)', marginTop: '8px' }}>
                  {r.n_obs.toLocaleString('en-IN')} obs · Rel red: {r.relative_error_reduction_pct >= 0 ? '+' : ''}{r.relative_error_reduction_pct.toFixed(2)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Feature Family Audit (F0 - F6) */}
      {features && (
        <div className="ledger-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} style={{ color: 'var(--accent)' }} />
              <div>
                <div className="ledger-panel-title">Feature Governance Audit (F0–F6)</div>
                <div style={{ fontSize: '12px', color: 'var(--dash-muted)' }}>
                  {features.populated_features_count} populated features · {features.quarantined_features_count} quarantined pre-2024 features to enforce zero lookahead bias
                </div>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {features.families.map((fam) => {
              const isQuarantined = fam.populated_count === 0;
              const isOpen = expandedFamily === fam.family;
              return (
                <div
                  key={fam.family}
                  onClick={() => setExpandedFamily(isOpen ? null : fam.family)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '10px',
                    border: `1px solid ${isQuarantined ? 'var(--hold-border)' : 'var(--rule)'}`,
                    background: isQuarantined ? '#fffdf5' : '#ffffff',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{
                        fontFamily: 'var(--mono)',
                        fontWeight: '700',
                        fontSize: '12px',
                        padding: '3px 8px',
                        borderRadius: '6px',
                        background: isQuarantined ? 'var(--hold-bg)' : 'var(--accent-subtle)',
                        color: isQuarantined ? 'var(--hold)' : 'var(--accent)',
                      }}>
                        {fam.family}
                      </span>
                      <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--ink)' }}>{fam.description}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontFamily: 'var(--mono)', fontSize: '11px' }}>
                      <span style={{ color: isQuarantined ? 'var(--hold)' : 'var(--gain)', fontWeight: '700' }}>
                        {fam.populated_count}/{fam.total_count} populated
                      </span>
                      <span style={{ color: 'var(--dash-muted)' }}>{isOpen ? '▾' : '▸'}</span>
                    </div>
                  </div>
                  {isOpen && (
                    <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--rule)', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                      {fam.features.map((fname) => (
                        <span
                          key={fname}
                          style={{
                            fontFamily: 'var(--mono)',
                            fontSize: '11px',
                            padding: '2px 8px',
                            borderRadius: '4px',
                            background: 'var(--paper)',
                            border: '1px solid var(--rule)',
                            color: '#334155',
                          }}
                        >
                          {fname}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Model Lineage & Diagnostic Plots */}
      {manifestData && (
        <div className="ledger-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <GitCommit size={18} style={{ color: 'var(--accent)' }} />
            <div>
              <div className="ledger-panel-title">Model Artifact Registry & Diagnostic Figures</div>
              <div style={{ fontSize: '12px', color: 'var(--dash-muted)' }}>
                Train cutoff: <strong>{manifestData.manifest.training_cutoff}</strong> · Calibration: <strong>{manifestData.manifest.calibration_period}</strong> · Diagnostic: <strong>{manifestData.manifest.diagnostic_period}</strong>
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '12px', marginBottom: '18px' }}>
            {manifestData.manifest.models.map((mod) => (
              <div key={mod.model_id} className="ledger-cell">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                  <strong style={{ fontFamily: 'var(--mono)', fontSize: '12px', color: 'var(--ink)' }}>{mod.model_id}</strong>
                  <span style={{
                    fontSize: '9px',
                    fontFamily: 'var(--mono)',
                    fontWeight: '700',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: mod.status.includes('PRODUCTION') ? 'var(--gain-bg)' : 'var(--hold-bg)',
                    color: mod.status.includes('PRODUCTION') ? 'var(--gain)' : 'var(--hold)',
                  }}>
                    {mod.status}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: '#334155', marginTop: '6px', fontWeight: '500' }}>{mod.algorithm}</div>
                <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--dash-muted)', marginTop: '6px' }}>
                  MAPE {mod.mape}% · MAE ₹{mod.mae_inr} · Rank IC {mod.rank_ic}
                </div>
                {mod.sha256 && (
                  <div style={{ fontSize: '10px', fontFamily: 'var(--mono)', color: '#94a3b8', marginTop: '4px' }}>
                    SHA-256: {mod.sha256.slice(0, 16)}…
                  </div>
                )}
                <div style={{ fontSize: '11px', color: 'var(--dash-muted)', marginTop: '6px', lineHeight: '1.45' }}>
                  {mod.limitations}
                </div>
              </div>
            ))}
          </div>

          {manifestData.figures.length > 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px', fontSize: '12px', fontWeight: '700', color: 'var(--ink)' }}>
                <ImageIcon size={14} style={{ color: 'var(--accent)' }} />
                <span>Verified Evaluation Plots (click to inspect)</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
                {manifestData.figures.map((fig) => (
                  <div
                    key={fig.filename}
                    onClick={() => setLightboxUrl({ url: fig.url, title: fig.title })}
                    style={{
                      border: '1px solid var(--rule)',
                      borderRadius: '10px',
                      overflow: 'hidden',
                      background: '#ffffff',
                      cursor: 'pointer',
                    }}
                  >
                    <img src={fig.url} alt={fig.title} style={{ width: '100%', height: '140px', objectFit: 'cover', display: 'block' }} />
                    <div style={{ padding: '8px 10px', fontSize: '11px', fontWeight: '600', color: 'var(--ink)' }}>
                      {fig.title}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Lightbox Modal */}
      {lightboxUrl && (
        <div
          onClick={() => setLightboxUrl(null)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.72)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: '#ffffff',
              borderRadius: '12px',
              maxWidth: '960px',
              width: '100%',
              overflow: 'hidden',
              boxShadow: '0 24px 48px rgba(0,0,0,0.28)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid var(--rule)' }}>
              <strong style={{ fontSize: '14px', color: 'var(--ink)' }}>{lightboxUrl.title}</strong>
              <button
                type="button"
                onClick={() => setLightboxUrl(null)}
                style={{ border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--dash-muted)' }}
              >
                <X size={18} />
              </button>
            </div>
            <img src={lightboxUrl.url} alt={lightboxUrl.title} style={{ width: '100%', maxHeight: '78vh', objectFit: 'contain', display: 'block', background: '#ffffff' }} />
          </div>
        </div>
      )}
    </div>
  );
};
