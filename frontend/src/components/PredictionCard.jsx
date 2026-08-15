import React from 'react';
import { Activity, ShieldCheck, AlertOctagon } from 'lucide-react';

export default function PredictionCard({ prediction, spatialSummary }) {
  if (!prediction) return null;

  const { predicted_class_id, predicted_class_name, confidence, probabilities } = prediction;
  const confidencePct = (confidence * 100).toFixed(1);

  return (
    <div className="glass-card fade-in" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Activity size={22} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: '700' }}>Deep Learning Severity Output</h3>
        </div>
        <div className={`severity-badge severity-${predicted_class_id}`}>
          Grade {predicted_class_id}: {predicted_class_name}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        {/* Main Grade Metric Card */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '14px',
          padding: '18px',
          border: '1px solid var(--border-color)',
          textAlign: 'center'
        }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Predicted DR Grade
          </span>
          <div style={{ fontSize: '2.5rem', fontWeight: '800', margin: '6px 0', color: 'var(--text-primary)' }}>
            Grade {predicted_class_id}
          </div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {predicted_class_name}
          </span>
        </div>

        {/* Model Confidence Metric Card */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '14px',
          padding: '18px',
          border: '1px solid var(--border-color)',
          textAlign: 'center'
        }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Classifier Confidence
          </span>
          <div style={{ fontSize: '2.5rem', fontWeight: '800', margin: '6px 0', color: 'var(--accent-cyan)' }}>
            {confidencePct}%
          </div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <ShieldCheck size={14} color="#34D399" /> High Softmax Certainty
          </span>
        </div>
      </div>

      {/* 5-Class Probability Distribution Breakdown */}
      <div>
        <h4 style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-secondary)', marginBottom: '12px' }}>
          5-Class Softmax Probability Distribution
        </h4>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {Object.entries(probabilities).map(([className, prob], idx) => {
            const pct = (prob * 100).toFixed(1);
            const isTarget = idx === predicted_class_id;

            return (
              <div key={className}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                  <span style={{ fontWeight: isTarget ? '700' : '400', color: isTarget ? '#FFFFFF' : 'var(--text-secondary)' }}>
                    Grade {idx}: {className}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: isTarget ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                    {pct}%
                  </span>
                </div>
                <div style={{
                  height: '8px',
                  width: '100%',
                  background: 'rgba(255, 255, 255, 0.08)',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${pct}%`,
                    background: isTarget
                      ? 'linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))'
                      : 'rgba(255, 255, 255, 0.2)',
                    borderRadius: '4px',
                    transition: 'width 0.5s ease-out'
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
