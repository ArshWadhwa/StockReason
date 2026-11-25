import React, { useState } from 'react';
import { api } from '../api';
import type { SignalData, PredictionData, ForecastResult } from '../types';
import { ModelReliabilityPanel } from './ModelReliabilityPanel';
import { SlidersHorizontal, Calendar, AlertCircle } from 'lucide-react';

interface SignalsViewProps {
  signal: SignalData | null;
  prediction: PredictionData | null;
  loading: boolean;
  onOpenBenchmarks?: () => void;
}

// Category classification for visual grouping
const FEATURE_CATEGORIES: Record<string, { label: string; color: string }> = {
  // Momentum
  rsi_14:           { label: 'Momentum', color: '#3B6FD4' },
  rsi_7:            { label: 'Momentum', color: '#3B6FD4' },
  rsi_21:           { label: 'Momentum', color: '#3B6FD4' },
  rsi_28:           { label: 'Momentum', color: '#3B6FD4' },
  macd_line_12_26:  { label: 'Momentum', color: '#3B6FD4' },
  macd_hist_12_26:  { label: 'Momentum', color: '#3B6FD4' },
  macd_signal_12_26:{ label: 'Momentum', color: '#3B6FD4' },
  stoch_k_14:       { label: 'Momentum', color: '#3B6FD4' },
  stoch_d_14:       { label: 'Momentum', color: '#3B6FD4' },
  williams_r_14:    { label: 'Momentum', color: '#3B6FD4' },
  cci_14:           { label: 'Momentum', color: '#3B6FD4' },
  cci_20:           { label: 'Momentum', color: '#3B6FD4' },

  // Trend
  dist_sma_50:      { label: 'Trend', color: '#1A7F4B' },
  dist_sma_20:      { label: 'Trend', color: '#1A7F4B' },
  dist_sma_200:     { label: 'Trend', color: '#1A7F4B' },
  dist_ema_50:      { label: 'Trend', color: '#1A7F4B' },
  dist_ema_20:      { label: 'Trend', color: '#1A7F4B' },
  sma_50_200_ratio: { label: 'Trend', color: '#1A7F4B' },
  sma_20_50_ratio:  { label: 'Trend', color: '#1A7F4B' },
  adx_14:           { label: 'Trend', color: '#1A7F4B' },
  close_slope_20:   { label: 'Trend', color: '#1A7F4B' },
  close_slope_10:   { label: 'Trend', color: '#1A7F4B' },
  close_slope_5:    { label: 'Trend', color: '#1A7F4B' },

  // Volatility
  atr_14:           { label: 'Volatility', color: '#946C00' },
  natr_14:          { label: 'Volatility', color: '#946C00' },
  volatility_20d:   { label: 'Volatility', color: '#946C00' },
  volatility_5d:    { label: 'Volatility', color: '#946C00' },
  volatility_60d:   { label: 'Volatility', color: '#946C00' },
  bb_width_20:      { label: 'Volatility', color: '#946C00' },
  bb_pct_b_20:      { label: 'Volatility', color: '#946C00' },

  // Volume
  volume:           { label: 'Volume', color: '#6B52C8' },
  vol_ratio_20:     { label: 'Volume', color: '#6B52C8' },
  vol_ratio_5:      { label: 'Volume', color: '#6B52C8' },
  obv:              { label: 'Volume', color: '#6B52C8' },
  cmf_20:           { label: 'Volume', color: '#6B52C8' },
  pvt:              { label: 'Volume', color: '#6B52C8' },

  // Sentiment
  sentiment_mean:         { label: 'Sentiment', color: '#B0457E' },
  sentiment_pos_ratio:    { label: 'Sentiment', color: '#B0457E' },
  sentiment_neg_ratio:    { label: 'Sentiment', color: '#B0457E' },
  sentiment_momentum_3d:  { label: 'Sentiment', color: '#B0457E' },
  sentiment_momentum_7d:  { label: 'Sentiment', color: '#B0457E' },

  // Macro
  vix_close:              { label: 'Macro', color: '#C0392B' },
  vix_close_return:       { label: 'Macro', color: '#C0392B' },
  nifty_50_close:         { label: 'Macro', color: '#C0392B' },
  nifty_50_close_return:  { label: 'Macro', color: '#C0392B' },
  nifty_bank_close_return:{ label: 'Macro', color: '#C0392B' },

  // Returns
  return_1d:    { label: 'Returns', color: '#2E7D8C' },
  return_5d:    { label: 'Returns', color: '#2E7D8C' },
  return_21d:   { label: 'Returns', color: '#2E7D8C' },
  return_63d:   { label: 'Returns', color: '#2E7D8C' },
  close:        { label: 'Price', color: '#2E7D8C' },
  hl_spread_pct:{ label: 'Price', color: '#2E7D8C' },
};

// Returns null for uncategorized — chip is suppressed, not shown as "Other"
const getCategory = (feature: string) => {
  const normalized = feature.toLowerCase().replace(/[\s-]+/g, '_');
  if (FEATURE_CATEGORIES[normalized]) return FEATURE_CATEGORIES[normalized];
  if (/^(rsi|macd|williams|stoch|cci)/.test(normalized)) return { label: 'Momentum', color: '#3B6FD4' };
  if (/^(dist_|sma_|ema_|adx|close_slope)/.test(normalized)) return { label: 'Trend', color: '#1A7F4B' };
  if (/(volatility|atr|bb_)/.test(normalized)) return { label: 'Volatility', color: '#946C00' };
  if (/(log_)?return/.test(normalized)) return { label: 'Returns', color: '#2E7D8C' };
  return null;
};

/** A readable fallback for features from older model runs which do not include
 * a display name or description in their prediction payload. */
const getFeatureExplanation = (feature: string, suppliedDescription?: string) => {
  if (suppliedDescription) return suppliedDescription;

  const normalized = feature.toLowerCase().replace(/[\s-]+/g, '_');
  if (/^rsi[_]?\d*.*lag[_]?\d+/.test(normalized)) {
    return 'Relative Strength Index (RSI) from an earlier trading day. RSI measures the speed and size of recent price moves on a 0–100 scale.';
  }
  if (/^rsi[_]?\d*/.test(normalized)) {
    return 'Relative Strength Index (RSI), a 0–100 measure of the speed and size of recent price moves.';
  }
  if (normalized.includes('williams')) return 'Williams %R, a momentum measure showing where price sits within its recent high–low range.';
  if (normalized.includes('volatility')) return 'How much the price has varied over this lookback period. Higher values mean a less stable price path.';
  if (normalized.includes('dist_sma') || normalized.includes('dist_ema')) return 'How far the current price is above or below its moving-average trend line.';
  if (normalized.includes('sma') && normalized.includes('ratio')) return 'The relationship between two moving averages, used to describe the short- versus longer-term trend.';
  if (normalized.includes('log_return') || normalized.includes('return')) return 'The stock’s price change over this lookback period. Positive values indicate a gain; negative values indicate a loss.';
  if (normalized.includes('macd')) return 'MACD, a momentum indicator that compares short- and long-term price trends.';
  if (normalized.includes('volume')) return 'Recent trading activity, which can show how strongly price moves are being supported.';
  return 'A market data input used by the forecasting model.';
};

const readableFeatureName = (feature: string, suppliedName?: string) => {
  const generatedName = feature
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/_/g, ' ')
    .replace(/\bRsi\b/gi, 'RSI')
    .replace(/\bSma\b/gi, 'SMA')
    .replace(/\b\w/g, c => c.toUpperCase());
  // Keep deliberate API labels, but improve the mechanical fallback labels from older runs.
  if (suppliedName && suppliedName.toLowerCase() !== generatedName.toLowerCase()) return suppliedName;

  const normalized = feature.toLowerCase().replace(/[\s-]+/g, '_');
  const rsiLag = normalized.match(/^rsi_(\d+)_lag_?(\d+)$/);
  if (rsiLag) return `RSI (${rsiLag[1]}-day), ${rsiLag[2]} days ago`;
  const volatility = normalized.match(/^volatility_(\d+)d?$/);
  if (volatility) return `${volatility[1]}-day volatility`;
  const logReturn = normalized.match(/^log_return_(\d+)d?$/);
  if (logReturn) return `${logReturn[1]}-day log return`;
  const smaRatio = normalized.match(/^sma_(\d+)_(\d+)_ratio$/);
  if (smaRatio) return `${smaRatio[1]}-day vs. ${smaRatio[2]}-day SMA`;
  const williams = normalized.match(/^williams_r_?(\d+)?$/);
  if (williams) return `Williams %R${williams[1] ? ` (${williams[1]}-day)` : ''}`;
  return generatedName;
};

const formatCurrentValue = (feature: string, value: number | string) => {
  if (typeof value !== 'number') return value;
  const normalized = feature.toLowerCase();
  if (normalized.includes('volatility') || normalized.includes('return') || normalized.includes('dist_')) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return Math.abs(value) > 1000
    ? value.toLocaleString('en-IN', { maximumFractionDigits: 0 })
    : value.toFixed(2);
};

const DEFAULT_VISIBLE = 3;

export const SignalsView: React.FC<SignalsViewProps> = ({ signal, prediction, loading, onOpenBenchmarks }) => {
  const [expandedFeature, setExpandedFeature] = useState<number | null>(null);
  const [showAllFeatures, setShowAllFeatures] = useState(false);

  // Interactive Forecast Simulator state (POST /api/forecast)
  const [simPriceEdit, setSimPriceEdit] = useState<{ ticker: string; value: string } | null>(null);
  const [simHorizon, setSimHorizon] = useState<number>(21);
  const [simModel, setSimModel] = useState<'baseline' | 'lightgbm' | 'catboost'>('baseline');
  const [simResultState, setSimResultState] = useState<{ ticker: string; result: ForecastResult } | null>(null);
  const [simLoading, setSimLoading] = useState<boolean>(false);
  const [simError, setSimError] = useState<string | null>(null);

  const simPrice =
    simPriceEdit && signal && simPriceEdit.ticker === signal.ticker
      ? simPriceEdit.value
      : signal?.current_price
        ? String(signal.current_price)
        : '';
  const simResult =
    simResultState && signal && simResultState.ticker === signal.ticker
      ? simResultState.result
      : null;

  const handleRunForecast = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!signal) return;
    const priceNum = Number(simPrice);
    if (!Number.isFinite(priceNum) || priceNum <= 0) {
      setSimError('Enter a valid positive reference price (INR).');
      return;
    }
    try {
      setSimLoading(true);
      setSimError(null);
      const res = await api.runForecast({
        symbol: signal.ticker,
        current_price: priceNum,
        horizon_days: simHorizon,
        model_type: simModel,
        reference_date: prediction?.forecast_origin_date,
      });
      setSimResultState({ ticker: signal.ticker, result: res });
    } catch (err: any) {
      setSimError(err?.message || 'Forecast simulation failed');
    } finally {
      setSimLoading(false);
    }
  };

  if (loading || !signal || !prediction) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="ledger-panel skeleton" style={{ height: '180px' }} />
        <div className="ledger-panel skeleton" style={{ height: '300px' }} />
      </div>
    );
  }

  const horizons = prediction.horizons;
  const allFeatures = prediction.shap_explainability;

  // Relative bar scaling — bars sized relative to the top feature, not absolute 0–100%
  const maxScore = Math.max(...allFeatures.map(f => f.importance_score), 0.001);
  const visibleFeatures = showAllFeatures ? allFeatures : allFeatures.slice(0, DEFAULT_VISIBLE);

  const signalColor = signal.signal === 'BUY' ? '#1A7F4B' : (signal.signal === 'HOLD' ? '#946C00' : '#C0392B');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

      {/* ── Primary Signal Verdict — PRIMARY card treatment ── */}
      <div className="ledger-panel" style={{
        borderLeft: `4px solid ${signalColor}`,
        borderRadius: '12px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            {/* Signal badge + conviction inline */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
              <span className={`ledger-tag ${signal.signal === 'BUY' ? 'tag-buy' : (signal.signal === 'HOLD' ? 'tag-hold' : 'tag-avoid')}`}
                style={{ fontSize: '13px', padding: '5px 13px' }}>
                {signal.signal}
              </span>
              <span style={{ fontSize: '13px', color: 'var(--dash-muted)', fontFamily: 'var(--sans)' }}>
                Conviction: <strong style={{ color: 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}>{signal.signal_score} / 100</strong>
              </span>
            </div>

            {/* Company name — display font, not repeated ticker */}
            <div style={{
              fontFamily: 'var(--serif)',
              fontSize: '26px',
              fontWeight: '400',
              color: 'var(--ink)',
              marginTop: '10px',
              lineHeight: '1.15',
            }}>
              {signal.name}
            </div>
          </div>

          {/* Calibrated Probability & 80% Interval Badge */}
          <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-end' }}>
            <div>
              <div className="ledger-label">Monte Carlo Dropout confidence</div>
              <div style={{
                fontFamily: 'var(--serif)',
                fontSize: '32px',
                fontWeight: '400',
                color: signalColor,
                fontVariantNumeric: 'tabular-nums',
                lineHeight: '1',
                marginTop: '4px',
              }}>
                {Math.round(signal.confidence * 100)}%
              </div>
            </div>

            {signal.prob_up_calibrated !== undefined && signal.prob_up_calibrated !== null ? (
              <div style={{
                background: 'var(--paper)',
                padding: '6px 10px',
                borderRadius: '8px',
                border: '1px solid #e2e8f0',
                fontSize: '11px',
                textAlign: 'right',
              }}>
                <div style={{ color: 'var(--dash-muted)', fontWeight: '600' }}>Calibrated Probability (Up)</div>
                <div style={{ fontFamily: 'var(--mono)', fontWeight: '700', color: signal.prob_up_calibrated >= 0.5 ? '#1A7F4B' : '#DC2626' }}>
                  {(signal.prob_up_calibrated * 100).toFixed(1)}%
                </div>
              </div>
            ) : (
              <div style={{
                background: '#fffbeb',
                padding: '6px 10px',
                borderRadius: '8px',
                border: '1px solid #fde68a',
                fontSize: '11px',
                textAlign: 'right',
              }}>
                <div style={{ color: '#92400e', fontWeight: '600' }}>Probability Gate</div>
                <div style={{ fontFamily: 'var(--mono)', fontWeight: '700', color: '#b45309' }}>
                  SUPPRESSED (Brier Skill Score ≤ 0)
                </div>
              </div>
            )}

            {signal.interval_80 && (
              <div style={{
                background: '#f8fafc',
                padding: '6px 10px',
                borderRadius: '8px',
                border: '1px solid #e2e8f0',
                fontSize: '11px',
                textAlign: 'right',
              }}>
                <div style={{ color: 'var(--dash-muted)', fontWeight: '600' }}>
                  80% Interval{' '}
                  {signal.interval_coverage_observed !== undefined && signal.interval_coverage_observed !== null && (
                    <span style={{
                      background: '#ecfdf5',
                      color: '#065f46',
                      padding: '1px 5px',
                      borderRadius: '4px',
                      fontSize: '10px',
                      fontWeight: '700',
                    }}>
                      Observed: {(signal.interval_coverage_observed * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <div style={{ fontFamily: 'var(--mono)', color: 'var(--ink)' }}>
                  [{(signal.interval_80[0] * 100).toFixed(1)}%, {(signal.interval_80[1] * 100).toFixed(1)}%]
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Decision Reasoning Trail */}
        <div style={{ marginTop: '22px' }}>
          <div className="ledger-label" style={{ marginBottom: '10px' }}>Decision reasoning</div>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '6px', listStyleType: 'none' }}>
            {signal.reasoning.map((r, idx) => (
              <li key={idx} className="ledger-row" style={{
                fontSize: '13px',
                color: '#334155',
                display: 'flex',
                alignItems: 'baseline',
                gap: '10px',
                lineHeight: '1.5',
              }}>
                <span style={{ color: 'var(--gain)', fontWeight: '700', fontSize: '11px', flexShrink: 0 }}>✓</span>
                {r}
              </li>
            ))}
          </ul>
        </div>

        {/* Invalidation Conditions */}
        {signal.invalidation_conditions && signal.invalidation_conditions.length > 0 && (
          <div style={{ marginTop: '16px', borderTop: '1px dashed #e2e8f0', paddingTop: '14px' }}>
            <div className="ledger-label" style={{ marginBottom: '8px', color: '#B45309' }}>
              ⚡ Invalidation Conditions
            </div>
            <ul style={{ display: 'flex', flexDirection: 'column', gap: '4px', listStyleType: 'none' }}>
              {signal.invalidation_conditions.map((cond, cIdx) => (
                <li key={cIdx} style={{ fontSize: '12px', color: '#64748b', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ color: '#F59E0B', fontSize: '10px' }}>•</span>
                  {cond}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* ── Interactive v0.14.0 Forecast Simulator (POST /api/forecast) ── */}
      <div className="ledger-panel" style={{ borderLeft: '4px solid var(--accent)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <SlidersHorizontal size={18} style={{ color: 'var(--accent)' }} />
            <div>
              <div className="ledger-panel-title">Interactive Forecast & National Stock Exchange Calendar Simulator</div>
              <div style={{ fontSize: '12px', color: 'var(--dash-muted)' }}>
                Runs live <code style={{ fontFamily: 'var(--mono)' }}>POST /api/forecast</code> with nominal price reconstruction, National Stock Exchange holiday skipping, and automatic fail-safe fallback
              </div>
            </div>
          </div>
          {prediction.target_trading_date && (
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              fontFamily: 'var(--mono)',
              fontSize: '11px',
              padding: '4px 10px',
              borderRadius: '6px',
              background: 'var(--paper)',
              border: '1px solid var(--rule)',
              color: 'var(--ink)',
              fontWeight: '600',
            }}>
              <Calendar size={12} style={{ color: 'var(--accent)' }} />
              21-Day National Stock Exchange Target: {prediction.target_trading_date}
            </span>
          )}
        </div>

        <form onSubmit={handleRunForecast} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '12px', alignItems: 'end' }}>
          <div>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', color: 'var(--dash-muted)', marginBottom: '5px' }}>
              Reference Price (₹)
            </label>
            <input
              type="number"
              step="0.05"
              min="1"
              value={simPrice}
              onChange={(e) => setSimPriceEdit({ ticker: signal.ticker, value: e.target.value })}
              style={{
                width: '100%',
                padding: '9px 11px',
                borderRadius: '8px',
                border: '1px solid var(--rule)',
                fontFamily: 'var(--mono)',
                fontSize: '13px',
                color: 'var(--ink)',
                background: '#ffffff',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', color: 'var(--dash-muted)', marginBottom: '5px' }}>
              Horizon (Trading Days)
            </label>
            <select
              value={simHorizon}
              onChange={(e) => setSimHorizon(Number(e.target.value))}
              className="dash-select"
              style={{ padding: '9px 11px' }}
            >
              <option value={1}>1 Session (Next Day)</option>
              <option value={5}>5 Sessions (1 Week)</option>
              <option value={21}>21 Sessions (1 Month · Primary)</option>
              <option value={63}>63 Sessions (1 Quarter)</option>
              <option value={126}>126 Sessions (6 Months)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: '600', color: 'var(--dash-muted)', marginBottom: '5px' }}>
              Inference Model
            </label>
            <select
              value={simModel}
              onChange={(e) => setSimModel(e.target.value as 'baseline' | 'lightgbm' | 'catboost')}
              className="dash-select"
              style={{ padding: '9px 11px' }}
            >
              <option value="baseline">No-Change Random Walk (Promoted)</option>
              <option value="lightgbm">LightGBM 21-Day (Experimental)</option>
              <option value="catboost">CatBoost 21-Day (Experimental)</option>
            </select>
          </div>

          <button
            type="submit"
            disabled={simLoading}
            className="ledger-btn-accent"
            style={{ height: '39px', justifyContent: 'center', fontWeight: '700' }}
          >
            {simLoading ? 'Running…' : 'Run Forecast'}
          </button>
        </form>

        {simError && (
          <div style={{ marginTop: '12px', padding: '10px 12px', borderRadius: '8px', background: 'var(--loss-bg)', color: 'var(--loss)', fontSize: '12px', fontFamily: 'var(--mono)' }}>
            {simError}
          </div>
        )}

        {simResult && (
          <div style={{
            marginTop: '18px',
            padding: '20px 22px',
            background: 'linear-gradient(135deg, #ffffff 0%, var(--paper) 100%)',
            borderRadius: '12px',
            border: '2px solid var(--accent-border)',
            boxShadow: '0 4px 14px rgba(59, 111, 212, 0.08)',
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '18px', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <div className="ledger-label" style={{ fontSize: '11px', fontWeight: '700' }}>Simulated Predicted Price</div>
                <div style={{ fontFamily: 'var(--serif)', fontSize: '36px', lineHeight: '1.05', color: 'var(--ink)', marginTop: '4px', fontVariantNumeric: 'tabular-nums' }}>
                  ₹{simResult.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>
              <div>
                <div className="ledger-label" style={{ fontSize: '11px', fontWeight: '700' }}>Expected Return</div>
                <div style={{ fontFamily: 'var(--serif)', fontSize: '34px', lineHeight: '1.05', color: simResult.predicted_return >= 0 ? 'var(--gain)' : 'var(--loss)', marginTop: '4px', fontVariantNumeric: 'tabular-nums' }}>
                  {simResult.predicted_return >= 0 ? '+' : ''}{(simResult.predicted_return * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="ledger-label">Target National Stock Exchange Date</div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '16px', fontWeight: '700', color: 'var(--ink)', marginTop: '6px' }}>
                  {simResult.target_trading_date}
                </div>
              </div>
              <div>
                <div className="ledger-label">Model & Gate</div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '12px', fontWeight: '700', color: 'var(--accent)', marginTop: '6px' }}>
                  {simResult.model_name}
                </div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--hold)' }}>
                  {simResult.model_status}
                </div>
              </div>
            </div>

            {simResult.price_prediction_interval_80 && (
              <div style={{ fontSize: '13px', fontFamily: 'var(--mono)', color: '#334155', marginBottom: '8px' }}>
                80% Split-Conformal Price Band: <strong>₹{simResult.price_prediction_interval_80[0].toLocaleString('en-IN')} – ₹{simResult.price_prediction_interval_80[1].toLocaleString('en-IN')}</strong>
                {simResult.price_prediction_interval_90 && (
                  <span> · 90% Band: <strong>₹{simResult.price_prediction_interval_90[0].toLocaleString('en-IN')} – ₹{simResult.price_prediction_interval_90[1].toLocaleString('en-IN')}</strong></span>
                )}
              </div>
            )}

            {simResult.warnings.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '8px', paddingTop: '8px', borderTop: '1px dashed #cbd5e1' }}>
                {simResult.warnings.map((w, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: '#92400e' }}>
                    <AlertCircle size={12} style={{ flexShrink: 0 }} />
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Multi-Horizon Prediction Cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px' }}>
        {(['1D', '1W', '1M', '6M'] as const).map((hKey) => {
          const h = horizons[hKey];
          const isPos = h.expected_return >= 0;
          const isPrimary = hKey === '1M';
          const labelMap = { '1D': 'Next day', '1W': '1 week', '1M': '1 month · Primary', '6M': '6 months' };
          const badgeMap = { '1D': '1 Day', '1W': '1 Week', '1M': '1 Month', '6M': '6 Months' };
          return (
            <div
              key={hKey}
              className="ledger-cell"
              style={{
                padding: '18px 20px',
                border: isPrimary ? '2px solid var(--accent-border)' : undefined,
                background: isPrimary ? 'linear-gradient(180deg, #ffffff 0%, var(--accent-subtle) 100%)' : undefined,
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '10px',
                gap: '8px',
              }}>
                <span style={{
                  fontSize: '12px',
                  color: isPrimary ? 'var(--accent)' : 'var(--dash-muted)',
                  fontFamily: 'var(--sans)',
                  fontWeight: isPrimary ? '700' : '600',
                }}>
                  {labelMap[hKey]}
                </span>
                <span style={{
                  fontFamily: 'var(--mono)',
                  fontSize: '11px',
                  fontWeight: '700',
                  color: isPrimary ? '#ffffff' : 'var(--ink)',
                  background: isPrimary ? 'var(--accent)' : 'var(--paper)',
                  padding: '2px 7px',
                  borderRadius: '4px',
                  whiteSpace: 'nowrap',
                }}>
                  {badgeMap[hKey]}
                </span>
              </div>
              <div style={{
                fontFamily: 'var(--serif)',
                fontSize: '30px',
                fontWeight: '400',
                color: 'var(--ink)',
                fontVariantNumeric: 'tabular-nums',
                lineHeight: '1.05',
                marginBottom: '6px',
              }}>
                ₹{h.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div style={{
                fontFamily: 'var(--mono)',
                fontSize: '18px',
                fontWeight: '700',
                color: isPos ? 'var(--gain)' : 'var(--loss)',
                fontVariantNumeric: 'tabular-nums',
                lineHeight: '1.2',
                marginBottom: '8px',
              }}>
                {isPos ? `+${(h.expected_return * 100).toFixed(2)}%` : `${(h.expected_return * 100).toFixed(2)}%`}
              </div>
              <div style={{
                fontSize: '12px',
                color: '#475569',
                fontFamily: 'var(--mono)',
                fontVariantNumeric: 'tabular-nums',
              }}>
                ₹{h.lower_bound.toLocaleString('en-IN', { maximumFractionDigits: 0 })}–₹{h.upper_bound.toLocaleString('en-IN', { maximumFractionDigits: 0 })} · {Math.round(h.confidence * 100)}% confidence
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Forecast drivers — progressive disclosure ── */}
      <div className="ledger-panel">
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '20px',
          flexWrap: 'wrap',
          gap: '12px',
        }}>
          <div>
            <div style={{
              fontFamily: 'var(--sans)',
              fontSize: '17px',
              fontWeight: '700',
              color: 'var(--ink)',
            }}>
              Feature attribution (gradient × input)
            </div>
            <div style={{ fontSize: '12px', color: 'var(--dash-muted)', marginTop: '3px' }}>
              The model’s most influential market inputs and sensitivities — these are not separate buy or sell signals.
            </div>
          </div>
          <div style={{
            fontSize: '11px',
            fontFamily: 'var(--sans)',
            color: 'var(--accent)',
            background: 'var(--accent-subtle)',
            border: '1px solid var(--accent-border)',
            padding: '4px 10px',
            borderRadius: '6px',
            fontWeight: '600',
          }}>
            Explains the 1-month forecast
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
          gap: '10px',
          padding: '11px 13px',
          marginBottom: '16px',
          background: 'var(--paper)',
          borderRadius: '8px',
          color: '#475569',
          fontSize: '11px',
          lineHeight: '1.5',
        }}>
          <div><strong style={{ color: 'var(--loss)' }}>Red / lowers forecast</strong><br />This input is pulling the model’s 1-month estimate down.</div>
          <div><strong style={{ color: 'var(--gain)' }}>Green / raises forecast</strong><br />This input is pushing the model’s 1-month estimate up.</div>
          <div><strong style={{ color: 'var(--ink)' }}>Model weight</strong><br />Its share of the model’s sensitivity; bar length is relative to the top factor.</div>
        </div>

        {allFeatures.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--dash-muted)', fontSize: '13px' }}>
            No feature attribution data available for this ticker.
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {visibleFeatures.map((f, i) => {
                const cat = getCategory(f.feature);
                const displayName = readableFeatureName(f.feature, f.feature_display);
                const description = getFeatureExplanation(f.feature, f.feature_description);
                const isExpanded = expandedFeature === i;

                // Relative bar: scaled to the top-ranked feature
                const barWidth = Math.round((f.importance_score / maxScore) * 100);

                return (
                  <div
                    key={i}
                    onClick={() => setExpandedFeature(isExpanded ? null : i)}
                    style={{
                      padding: '12px 14px',
                      background: isExpanded ? 'var(--accent-subtle)' : '#ffffff',
                      border: `1px solid ${isExpanded ? 'var(--accent-border)' : 'var(--rule)'}`,
                      borderRadius: '10px',
                      cursor: 'pointer',
                      transition: 'all 0.18s ease',
                    }}
                  >
                    {/* Main row */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>

                      {/* Rank badge — positional context */}
                      <span style={{
                        fontFamily: 'var(--mono)',
                        fontSize: '10px',
                        fontWeight: '700',
                        color: 'var(--dash-muted)',
                        width: '18px',
                        textAlign: 'right',
                        flexShrink: 0,
                      }}>
                        {i + 1}
                      </span>

                      {/* Feature name, explanation, and model effect */}
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                          <span style={{
                            fontSize: '13px',
                            fontWeight: '600',
                            color: 'var(--ink)',
                            fontFamily: 'var(--sans)',
                          }}>
                            {displayName}
                          </span>
                          {/* Category chip — only shown when category is known */}
                          {cat && (
                            <span style={{
                              fontSize: '9px',
                              padding: '1px 6px',
                              borderRadius: '4px',
                              background: cat.color + '14',
                              color: cat.color,
                              fontFamily: 'var(--sans)',
                              fontWeight: '600',
                              border: `1px solid ${cat.color}28`,
                              letterSpacing: '0.2px',
                            }}>
                              {cat.label}
                            </span>
                          )}
                        </div>

                        <div style={{
                          fontSize: '11px',
                          lineHeight: '1.45',
                          color: '#64748b',
                          marginTop: '4px',
                          maxWidth: '760px',
                        }}>
                          {description}
                        </div>

                        {/* Relative importance bar */}
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                          marginTop: '7px',
                        }}>
                          <div style={{
                            flex: 1,
                            height: '5px',
                            background: 'var(--paper)',
                            borderRadius: '3px',
                            overflow: 'hidden',
                            position: 'relative',
                          }}>
                            <div style={{
                              position: 'absolute',
                              left: f.direction === 'negative' ? `${100 - barWidth}%` : '0',
                              width: `${barWidth}%`,
                              height: '100%',
                              background: f.direction === 'positive'
                                ? 'var(--gain)'
                                : 'var(--loss)',
                              borderRadius: '3px',
                              transition: 'width 0.35s ease',
                            }} />
                          </div>
                          <span style={{
                            fontSize: '12px',
                            fontWeight: '600',
                            fontFamily: 'var(--mono)',
                            fontVariantNumeric: 'tabular-nums',
                            color: f.direction === 'positive' ? 'var(--gain)' : 'var(--loss)',
                            minWidth: '52px',
                            textAlign: 'right',
                          }}>
                            {f.direction === 'positive' ? 'Raises' : 'Lowers'} forecast · {(f.importance_score * 100).toFixed(1)}% weight
                          </span>
                        </div>
                      </div>

                      {/* Current value — mono, tabular */}
                      {f.current_value !== undefined && f.current_value !== null && (
                        <div style={{ textAlign: 'right', minWidth: '72px' }}>
                          <div style={{
                            fontSize: '10px',
                            color: 'var(--dash-muted)',
                            fontFamily: 'var(--sans)',
                            marginBottom: '2px',
                          }}>
                            Latest reading
                          </div>
                          <div style={{
                            fontSize: '13px',
                            fontWeight: '700',
                            color: 'var(--ink)',
                            fontFamily: 'var(--mono)',
                            fontVariantNumeric: 'tabular-nums',
                          }}>
                            {formatCurrentValue(f.feature, f.current_value)}
                          </div>
                        </div>
                      )}

                      <span style={{
                        color: 'var(--dash-muted)',
                        fontSize: '11px',
                        transform: isExpanded ? 'rotate(90deg)' : 'none',
                        transition: 'transform 0.15s',
                        flexShrink: 0,
                      }}>
                        ▸
                      </span>
                    </div>

                    {/* Expanded detail */}
                    {isExpanded && (
                      <div style={{
                        marginTop: '12px',
                        padding: '10px 12px 10px 30px',
                        fontSize: '12px',
                        lineHeight: '1.65',
                        color: '#475569',
                        borderTop: '1px solid var(--accent-border)',
                      }}>
                        <span style={{ color: cat?.color ?? 'var(--accent)', fontWeight: '600' }}>
                          Model interpretation:{' '}
                        </span>
                        {f.direction === 'positive'
                          ? 'With the current reading, the model treats this as upward pressure on its 1-month forecast.'
                          : 'With the current reading, the model treats this as downward pressure on its 1-month forecast.'}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Progressive disclosure control */}
            {allFeatures.length > DEFAULT_VISIBLE && (
              <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'center' }}>
                <button
                  className="ledger-btn-accent"
                  onClick={() => {
                    setShowAllFeatures(v => !v);
                    if (showAllFeatures) setExpandedFeature(null);
                  }}
                >
                  {showAllFeatures
                    ? 'Show fewer factors'
                    : `Show all ${allFeatures.length} factors`}
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Model Reliability & MLOps Verification Panel ── */}
      <ModelReliabilityPanel onOpenBenchmarks={onOpenBenchmarks} />
    </div>
  );
};
