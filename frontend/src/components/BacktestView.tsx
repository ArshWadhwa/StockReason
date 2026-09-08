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
        <div className="ledger-panel" style={{ borderTop: '4px solid #16a34a' }}>
          <div className="ledger-label" style={{ color: '#1e4d1f' }}>AI Signal Strategy</div>
          <div className="ledger-metric ledger-metric-brass" style={{ margin: '8px 0', fontSize: '30px' }}>
            +{strat.cumulative_return}%
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: '#64748b', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div>Sharpe Ratio: <strong style={{ color: '#0f172a' }}>{strat.sharpe_ratio}</strong></div>
            <div>Max Drawdown: <strong style={{ color: '#0f172a' }}>{strat.max_drawdown}%</strong></div>
            <div>Win Rate: <strong style={{ color: '#0f172a' }}>{strat.win_rate}%</strong></div>
          </div>
        </div>

        <div className="ledger-panel" style={{ borderTop: '4px solid #2563eb' }}>
          <div className="ledger-label" style={{ color: '#2563eb' }}>Nifty 50 Benchmark</div>
          <div className="ledger-metric" style={{ margin: '8px 0', fontSize: '30px', color: '#2563eb' }}>
            +{bench.cumulative_return}%
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: '#64748b', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div>Sharpe Ratio: <strong style={{ color: '#0f172a' }}>{bench.sharpe_ratio}</strong></div>
            <div>Max Drawdown: <strong style={{ color: '#0f172a' }}>{bench.max_drawdown}%</strong></div>
            <div>Ann. Return: <strong style={{ color: '#0f172a' }}>+{bench.annualized_return}%</strong></div>
          </div>
        </div>

        <div className="ledger-panel" style={{ borderTop: '4px solid #94a3b8' }}>
          <div className="ledger-label" style={{ color: '#64748b' }}>Random Trading Baseline</div>
          <div className="ledger-metric" style={{ margin: '8px 0', fontSize: '30px', color: '#64748b' }}>
            +{rand.cumulative_return}%
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: '#64748b', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div>Sharpe Ratio: <strong style={{ color: '#0f172a' }}>{rand.sharpe_ratio}</strong></div>
            <div>Max Drawdown: <strong style={{ color: '#0f172a' }}>{rand.max_drawdown}%</strong></div>
            <div>Win Rate: <strong style={{ color: '#0f172a' }}>{rand.win_rate}%</strong></div>
          </div>
        </div>
      </div>

      {/* Equity Curve High-Contrast Graph */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div style={{ fontFamily: 'var(--display)', fontSize: '17px', fontWeight: '700', color: 'var(--text-primary)' }}>
              Cumulative Out-of-Sample Performance Curve
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
              AI Strategy vs Benchmark & Random Baseline
            </div>
          </div>

          <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--mono)', fontSize: '11px' }}>
            <span style={{ color: '#1e4d1f', fontWeight: '700' }}>── AI Strategy (+{strat.cumulative_return}%)</span>
            <span style={{ color: '#2563eb', fontWeight: '600' }}>── Nifty 50 (+{bench.cumulative_return}%)</span>
            <span style={{ color: '#94a3b8' }}>- - Random (+{rand.cumulative_return}%)</span>
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
            <polyline fill="none" stroke="#2563eb" strokeWidth="2" points={benchPoints} />

            {/* Strategy Fill & Line */}
            <polygon fill="url(#stratGradient)" points={stratArea} />
            <polyline
              fill="none"
              stroke="#2d6a2e"
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
                  stroke="#2d6a2e"
                  strokeWidth="1.2"
                  strokeDasharray="3 3"
                  opacity="0.6"
                />
                <circle
                  cx={activeX}
                  cy={height - ((activeItem.strategy - minVal) / valRange) * height}
                  r="4.5"
                  fill="#2d6a2e"
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
        <div className="ledger-panel-title" style={{ marginBottom: '16px' }}>Confidence Calibration Report</div>
        {backtest.confidence_calibration.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '24px 0', color: '#64748b', fontFamily: 'var(--mono)', fontSize: '13px' }}>
            Collecting and validating out-of-fold predictions to generate calibration report...
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px', fontFamily: 'var(--mono)' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b', background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px' }}>CONFIDENCE BIN</th>
                  <th style={{ padding: '10px 12px' }}>STATED PROB</th>
                  <th style={{ padding: '10px 12px' }}>EMPIRICAL ACCURACY</th>
                  <th style={{ padding: '10px 12px' }}>SAMPLE COUNT</th>
                </tr>
              </thead>
              <tbody>
                {backtest.confidence_calibration.map((bin, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '12px', color: '#0f172a', fontWeight: '600' }}>{bin.confidence_bin}</td>
                    <td style={{ padding: '12px', color: '#475569' }}>{(bin.predicted_prob * 100).toFixed(0)}%</td>
                    <td style={{ padding: '12px', color: '#16a34a', fontWeight: '700' }}>{(bin.actual_accuracy * 100).toFixed(0)}%</td>
                    <td style={{ padding: '12px', color: '#64748b' }}>{bin.sample_count} trades</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
