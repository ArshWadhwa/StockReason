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
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
        <div className="ledger-panel" style={{ borderTop: '3px solid var(--success)' }}>
          <div className="ledger-label" style={{ color: 'var(--success)' }}>AI Signal Strategy</div>
          <div className="ledger-metric ledger-metric-brass" style={{ margin: '8px 0', fontSize: '28px' }}>
            +{strat.cumulative_return}%
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--muted)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div>Sharpe Ratio: <strong style={{ color: '#fff' }}>{strat.sharpe_ratio}</strong></div>
            <div>Max Drawdown: <strong style={{ color: '#fff' }}>{strat.max_drawdown}%</strong></div>
            <div>Win Rate: <strong style={{ color: '#fff' }}>{strat.win_rate}%</strong></div>
          </div>
        </div>

        <div className="ledger-panel" style={{ borderTop: '3px solid #60a5fa' }}>
          <div className="ledger-label" style={{ color: '#60a5fa' }}>Nifty 50 Benchmark</div>
          <div className="ledger-metric" style={{ margin: '8px 0', fontSize: '28px', color: '#60a5fa' }}>
            +{bench.cumulative_return}%
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--muted)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div>Sharpe Ratio: <strong style={{ color: '#fff' }}>{bench.sharpe_ratio}</strong></div>
            <div>Max Drawdown: <strong style={{ color: '#fff' }}>{bench.max_drawdown}%</strong></div>
            <div>Ann. Return: <strong style={{ color: '#fff' }}>+{bench.annualized_return}%</strong></div>
          </div>
        </div>

        <div className="ledger-panel" style={{ borderTop: '3px solid #555' }}>
          <div className="ledger-label">Random Trading Baseline</div>
          <div className="ledger-metric" style={{ margin: '8px 0', fontSize: '28px', color: '#888' }}>
            +{rand.cumulative_return}%
          </div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--muted)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div>Sharpe Ratio: {rand.sharpe_ratio}</div>
            <div>Max Drawdown: {rand.max_drawdown}%</div>
            <div>Win Rate: {rand.win_rate}%</div>
          </div>
        </div>
      </div>

      {/* Equity Curve High-Contrast Graph */}
      <div className="ledger-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div>
            <div style={{ fontSize: '15px', fontWeight: '600', color: '#f0f0f0' }}>
              Cumulative Out-of-Sample Performance Curve
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: '#777', marginTop: '2px' }}>
              AI Strategy vs Benchmark & Random Baseline
            </div>
          </div>

          <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--mono)', fontSize: '11px' }}>
            <span style={{ color: 'var(--success)', fontWeight: '600' }}>── AI Strategy (+{strat.cumulative_return}%)</span>
            <span style={{ color: '#60a5fa' }}>── Nifty 50 (+{bench.cumulative_return}%)</span>
            <span style={{ color: '#777' }}>- - Random (+{rand.cumulative_return}%)</span>
          </div>
        </div>

        <div
          style={{
            position: 'relative',
            width: '100%',
            height: `${height}px`,
            background: '#090a0d',
            border: '1px solid #1c222b',
            borderRadius: '4px',
            overflow: 'hidden'
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
                <stop offset="0%" stopColor="#4ade80" stopOpacity="0.2" />
                <stop offset="100%" stopColor="#4ade80" stopOpacity="0.0" />
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
                  stroke="#1a202c"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
              );
            })}

            {/* Random Baseline Line */}
            <polyline fill="none" stroke="#555555" strokeWidth="1.5" strokeDasharray="4 4" points={randPoints} />

            {/* Benchmark Line */}
            <polyline fill="none" stroke="#60a5fa" strokeWidth="2" points={benchPoints} />

            {/* Strategy Fill & Line */}
            <polygon fill="url(#stratGradient)" points={stratArea} />
            <polyline
              fill="none"
              stroke="#4ade80"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={stratPoints}
            />

            {/* Crosshair indicator */}
            <line
              x1={activeX}
              y1={0}
              x2={activeX}
              y2={height}
              stroke="#ffffff"
              strokeWidth="1"
              strokeDasharray="2 2"
              opacity="0.3"
            />
          </svg>
        </div>

        {/* Dynamic coordinate readout */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px', fontFamily: 'var(--mono)', fontSize: '11px', color: '#888' }}>
          <div>Date: <span style={{ color: '#fff' }}>{activeItem.date}</span></div>
          <div style={{ display: 'flex', gap: '18px' }}>
            <span>Strategy: <strong style={{ color: 'var(--success)' }}>+{activeItem.strategy.toFixed(1)}%</strong></span>
            <span>Benchmark: <strong style={{ color: '#60a5fa' }}>+{activeItem.benchmark.toFixed(1)}%</strong></span>
            <span>Random: <strong style={{ color: '#888' }}>+{activeItem.random_baseline.toFixed(1)}%</strong></span>
          </div>
        </div>
      </div>

      {/* Confidence Calibration Table */}
      <div className="ledger-panel">
        <div className="ledger-panel-title" style={{ marginBottom: '14px' }}>Confidence Calibration Report</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px', fontFamily: 'var(--mono)' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #222', color: '#777' }}>
              <th style={{ padding: '8px' }}>CONFIDENCE BIN</th>
              <th style={{ padding: '8px' }}>STATED PROB</th>
              <th style={{ padding: '8px' }}>EMPIRICAL ACCURACY</th>
              <th style={{ padding: '8px' }}>SAMPLE COUNT</th>
            </tr>
          </thead>
          <tbody>
            {backtest.confidence_calibration.map((bin, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #181818' }}>
                <td style={{ padding: '10px 8px', color: '#ccc' }}>{bin.confidence_bin}</td>
                <td style={{ padding: '10px 8px' }}>{(bin.predicted_prob * 100).toFixed(0)}%</td>
                <td style={{ padding: '10px 8px', color: 'var(--success)' }}>{(bin.actual_accuracy * 100).toFixed(0)}%</td>
                <td style={{ padding: '10px 8px', color: 'var(--muted)' }}>{bin.sample_count} trades</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
