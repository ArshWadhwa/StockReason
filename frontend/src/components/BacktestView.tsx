import React, { useState } from 'react';
import type { BacktestData } from '../types';

interface BacktestViewProps {
  backtest: BacktestData | null;
  loading: boolean;
}

export const BacktestView: React.FC<BacktestViewProps> = ({ backtest, loading }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (loading || !backtest) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel" style={{ height: '140px' }} />
        <div className="ledger-panel" style={{ height: '350px' }} />
      </div>
    );
  }

  const strat = backtest.strategy_metrics;
  const bench = backtest.benchmark_metrics;
  const rand = backtest.random_baseline_metrics;
  const curve = backtest.equity_curve;

  const width = 880;
  const height = 270;
  const stratValues = curve.map(c => c.strategy);
  const benchValues = curve.map(c => c.benchmark);
  const randValues = curve.map(c => c.random_baseline);
  const allValues = [...stratValues, ...benchValues, ...randValues];
  const minVal = Math.min(...allValues) * 0.95;
  const maxVal = Math.max(...allValues) * 1.05;
  const valRange = maxVal - minVal || 1;

  const toSvgPoints = (arr: number[]) =>
    arr.map((val, idx) => {
      const x = (idx / (arr.length - 1)) * width;
      const y = height - ((val - minVal) / valRange) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

  const stratPoints = toSvgPoints(stratValues);
  const benchPoints = toSvgPoints(benchValues);
  const randPoints = toSvgPoints(randValues);

  const stratArea = `0,${height} ${stratPoints} ${width},${height}`;

  const activeIndex = hoverIndex !== null && hoverIndex >= 0 && hoverIndex < curve.length ? hoverIndex : curve.length - 1;
  const activeItem = curve[activeIndex];
  const activeX = (activeIndex / (curve.length - 1)) * width;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Metrics Scorecards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px' }}>
        <div className="ledger-panel" style={{ borderTop: '4px solid var(--gain)' }}>
          <div className="ledger-label" style={{ color: 'var(--gain)' }}>AI signal strategy</div>
          <div className="ledger-metric ledger-metric-brass" style={{ margin: '8px 0' }}>
            +{strat.cumulative_return}%
          </div>
          <div style={{ fontSize: '12px', color: 'var(--dash-muted)', display: 'flex', flexDirection: 'column', gap: '5px', fontVariantNumeric: 'tabular-nums' }}>
            <div>Sharpe ratio: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{strat.sharpe_ratio}</strong></div>
            <div>Max drawdown: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{strat.max_drawdown}%</strong></div>
            <div>Win rate: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{strat.win_rate}%</strong></div>
          </div>
        </div>

        <div className="ledger-panel" style={{ borderTop: '4px solid var(--accent)' }}>
          <div className="ledger-label" style={{ color: 'var(--accent)' }}>Nifty 50 benchmark</div>
          <div className="ledger-metric ledger-metric-slate" style={{ margin: '8px 0' }}>
            +{bench.cumulative_return}%
          </div>
          <div style={{ fontSize: '12px', color: 'var(--dash-muted)', display: 'flex', flexDirection: 'column', gap: '5px', fontVariantNumeric: 'tabular-nums' }}>
            <div>Sharpe ratio: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{bench.sharpe_ratio}</strong></div>
            <div>Max drawdown: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{bench.max_drawdown}%</strong></div>
            <div>Ann. return: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>+{bench.annualized_return}%</strong></div>
          </div>
        </div>

        <div className="ledger-panel" style={{ borderTop: '4px solid var(--rule)' }}>
          <div className="ledger-label" style={{ color: 'var(--dash-muted)' }}>Random trading baseline</div>
          <div className="ledger-metric" style={{ margin: '8px 0', color: 'var(--dash-muted)' }}>
            +{rand.cumulative_return}%
          </div>
          <div style={{ fontSize: '12px', color: 'var(--dash-muted)', display: 'flex', flexDirection: 'column', gap: '5px', fontVariantNumeric: 'tabular-nums' }}>
            <div>Sharpe ratio: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{rand.sharpe_ratio}</strong></div>
            <div>Max drawdown: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{rand.max_drawdown}%</strong></div>
            <div>Win rate: <strong style={{ color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{rand.win_rate}%</strong></div>
          </div>
        </div>
      </div>

      {/* Equity Curve High-Contrast Graph */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div style={{ fontFamily: 'var(--sans)', fontSize: '17px', fontWeight: '700', color: 'var(--ink)' }}>
              Cumulative out-of-sample performance
            </div>
            <div style={{ fontFamily: 'var(--sans)', fontSize: '12px', color: 'var(--dash-muted)', marginTop: '2px' }}>
              AI strategy vs benchmark and random baseline
            </div>
          </div>

          <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--mono)', fontSize: '11px', fontVariantNumeric: 'tabular-nums' }}>
            <span style={{ color: 'var(--gain)', fontWeight: '700' }}>── AI strategy (+{strat.cumulative_return}%)</span>
            <span style={{ color: 'var(--accent)', fontWeight: '600' }}>── Nifty 50 (+{bench.cumulative_return}%)</span>
            <span style={{ color: 'var(--dash-muted)' }}>- - Random (+{rand.cumulative_return}%)</span>
          </div>
        </div>

        <div
          style={{
            position: 'relative',
            width: '100%',
            height: `${height}px`,
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            overflow: 'hidden',
            boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.02)'
          }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = (e.clientX - rect.left) / rect.width;
            const idx = Math.min(curve.length - 1, Math.max(0, Math.round(relX * (curve.length - 1))));
            setHoverIndex(idx);
          }}
          onMouseLeave={() => setHoverIndex(null)}
        >
          <svg
            viewBox={`0 0 ${width} ${height}`}
            style={{ width: '100%', height: '100%', display: 'block' }}
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="stratGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2d6a2e" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#2d6a2e" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            {[0.25, 0.5, 0.75].map((ratio, i) => {
              const y = height * ratio;
              return (
                <line
                  key={i}
                  x1="0"
                  y1={y}
                  x2={width}
                  y2={y}
                  stroke="#f1f5f9"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
              );
            })}

            {/* Random Baseline Line */}
            <polyline fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="4 4" points={randPoints} />

            {/* Benchmark Line */}
            <polyline fill="none" stroke="var(--accent)" strokeWidth="2" points={benchPoints} />

            {/* Strategy Fill & Line */}
            <polygon fill="url(#stratGradient)" points={stratArea} />
            <polyline
              fill="none"
              stroke="var(--gain)"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={stratPoints}
            />

            {/* Crosshair indicator */}
            {hoverIndex !== null && (
              <>
                <line
                  x1={activeX}
                  y1={0}
                  x2={activeX}
                  y2={height}
                  stroke="var(--gain)"
                  strokeWidth="1.2"
                  strokeDasharray="3 3"
                  opacity="0.6"
                />
                <circle
                  cx={activeX}
                  cy={height - ((activeItem.strategy - minVal) / valRange) * height}
                  r="4.5"
                  fill="var(--gain)"
                  stroke="#ffffff"
                  strokeWidth="2"
                />
              </>
            )}
          </svg>
        </div>

        {/* Dynamic coordinate readout */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '14px', fontFamily: 'var(--mono)', fontSize: '11px', color: '#64748b' }}>
          <div>Date: <span style={{ color: '#0f172a', fontWeight: '600' }}>{activeItem.date}</span></div>
          <div style={{ display: 'flex', gap: '18px' }}>
            <span>Strategy: <strong style={{ color: '#16a34a' }}>+{activeItem.strategy.toFixed(1)}%</strong></span>
            <span>Benchmark: <strong style={{ color: '#2563eb' }}>+{activeItem.benchmark.toFixed(1)}%</strong></span>
            <span>Random: <strong style={{ color: '#64748b' }}>+{activeItem.random_baseline.toFixed(1)}%</strong></span>
          </div>
        </div>
      </div>

      {/* Confidence Calibration Table */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <div className="ledger-panel-title">Confidence calibration</div>
            <div style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '4px', maxWidth: '60ch' }}>
              Comparison of stated model confidence vs empirical accuracy on held-out trades.
              Rows highlighted in red are bins where the model overstates confidence by &gt;10 pp.
            </div>
          </div>
        </div>
        {backtest.confidence_calibration.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '24px 0', color: '#64748b', fontFamily: 'var(--mono)', fontSize: '13px' }}>
            Collecting and validating out-of-fold predictions to generate calibration report...
          </div>
        ) : (
          <>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px', fontFamily: 'var(--mono)', fontVariantNumeric: 'tabular-nums' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--rule)', color: 'var(--dash-muted)', background: 'var(--paper)' }}>
                    <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Confidence bin</th>
                    <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Stated prob.</th>
                    <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Empirical accuracy</th>
                    <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Gap (stated − empirical)</th>
                    <th style={{ padding: '10px 12px', fontFamily: 'var(--sans)', fontWeight: '600', fontSize: '11px' }}>Trades</th>
                  </tr>
                </thead>
                <tbody>
                  {backtest.confidence_calibration.map((bin, i) => {
                    const stated = bin.predicted_prob * 100;
                    const actual = bin.actual_accuracy * 100;
                    const gap = stated - actual;
                    const isOverconfident = gap > 10;
                    return (
                      <tr key={i} style={{
                        borderBottom: '1px solid var(--rule)',
                        background: isOverconfident ? 'var(--loss-bg)' : 'transparent',
                      }}>
                        <td style={{ padding: '11px 12px', color: 'var(--ink)', fontWeight: '600' }}>{bin.confidence_bin}</td>
                        <td style={{ padding: '11px 12px', color: 'var(--dash-muted)' }}>{stated.toFixed(0)}%</td>
                        <td style={{ padding: '11px 12px', color: 'var(--gain)', fontWeight: '700' }}>{actual.toFixed(0)}%</td>
                        <td style={{ padding: '11px 12px', color: isOverconfident ? 'var(--loss)' : 'var(--dash-muted)', fontWeight: isOverconfident ? '700' : '400' }}>
                          {isOverconfident ? `+${gap.toFixed(0)} pp ⚠` : `+${gap.toFixed(0)} pp`}
                        </td>
                        <td style={{ padding: '11px 12px', color: 'var(--dash-muted)' }}>{bin.sample_count}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: '14px', padding: '10px 14px', background: 'var(--paper)', borderRadius: '8px', fontSize: '11px', color: 'var(--dash-muted)', fontFamily: 'var(--sans)' }}>
              <strong style={{ color: 'var(--ink)' }}>Note on these numbers:</strong> Backtest and calibration figures were generated before the latest model retrain.
              Treat them as pre-fix baseline estimates until a fresh out-of-sample run confirms performance.
            </div>
          </>
        )}
      </div>
    </div>
  );
};
