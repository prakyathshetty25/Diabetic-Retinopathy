import React from 'react';
import { Activity, ShieldCheck, AlertOctagon, AlertTriangle, Info, Clock, CheckCircle } from 'lucide-react';

export default function PredictionCard({ prediction, spatialSummary }) {
  if (!prediction) return null;

  const predictedClassId = prediction.predicted_class_id ?? 0;
  const predictedClassName = prediction.predicted_class_name || `Grade ${predictedClassId}`;
  const confidenceVal = prediction.confidence_score ?? prediction.confidence ?? 0;
  const confidencePct = (confidenceVal * 100).toFixed(1);

  const probabilities = prediction.all_class_probabilities || prediction.probabilities || {};
  const progressionRisk = prediction.progression_risk || 'Unspecified';
  const clinicalRecommendation = prediction.clinical_recommendation || 'Consult ophthalmologist for detailed examination.';

  // Determine progression risk color scheme
  const getRiskColor = (riskStr) => {
    const lower = riskStr.toLowerCase();
    if (lower.includes('low')) return { bg: 'rgba(16, 185, 129, 0.15)', text: '#34D399', border: 'rgba(16, 185, 129, 0.3)', icon: CheckCircle };
    if (lower.includes('mild')) return { bg: 'rgba(59, 130, 246, 0.15)', text: '#60A5FA', border: 'rgba(59, 130, 246, 0.3)', icon: Info };
    if (lower.includes('moderate')) return { bg: 'rgba(245, 158, 11, 0.15)', text: '#FBBF24', border: 'rgba(245, 158, 11, 0.3)', icon: Clock };
    if (lower.includes('high')) return { bg: 'rgba(239, 68, 68, 0.15)', text: '#F87171', border: 'rgba(239, 68, 68, 0.3)', icon: AlertTriangle };
    if (lower.includes('critical')) return { bg: 'rgba(153, 27, 27, 0.3)', text: '#EF4444', border: 'rgba(239, 68, 68, 0.5)', icon: AlertOctagon };
    return { bg: 'rgba(255, 255, 255, 0.05)', text: '#E2E8F0', border: 'rgba(255, 255, 255, 0.1)', icon: Info };
  };

  const riskStyle = getRiskColor(progressionRisk);
  const RiskIcon = riskStyle.icon;

  return (
    <div className="glass-card fade-in" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Activity size={22} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: '700' }}>Deep Learning Diagnostic Output</h3>
        </div>
        <div className={`severity-badge severity-${predictedClassId}`}>
          Grade {predictedClassId}: {predictedClassName}
        </div>
      </div>

      {/* Main Metric Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '24px' }}>
        {/* 1. DR Stage Card */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '14px',
          padding: '18px',
          border: '1px solid var(--border-color)',
          textAlign: 'center'
        }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Diabetic Retinopathy Stage
          </span>
          <div style={{ fontSize: '2.2rem', fontWeight: '800', margin: '6px 0', color: 'var(--text-primary)' }}>
            Grade {predictedClassId}
          </div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {predictedClassName}
          </span>
        </div>

        {/* 2. Disease Progression Risk Card */}
        <div style={{
          background: riskStyle.bg,
          borderRadius: '14px',
          padding: '18px',
          border: `1px solid ${riskStyle.border}`,
          textAlign: 'center'
        }}>
          <span style={{ fontSize: '0.78rem', color: riskStyle.text, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: '600' }}>
            Disease Progression Risk
          </span>
          <div style={{ fontSize: '1.4rem', fontWeight: '800', margin: '10px 0', color: riskStyle.text, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <RiskIcon size={22} />
            {progressionRisk}
          </div>
        </div>

        {/* 3. Classifier Confidence Meter Card */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '14px',
          padding: '18px',
          border: '1px solid var(--border-color)',
          textAlign: 'center'
        }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Model Confidence Meter
          </span>
          <div style={{ fontSize: '2.2rem', fontWeight: '800', margin: '6px 0', color: 'var(--accent-cyan)' }}>
            {confidencePct}%
          </div>
          
          {/* Confidence Meter Bar */}
          <div style={{
            height: '6px',
            width: '80%',
            margin: '8px auto',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '3px',
            overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              width: `${confidencePct}%`,
              background: 'linear-gradient(90deg, #06B6D4, #3B82F6)',
              borderRadius: '3px',
              transition: 'width 0.6s ease'
            }} />
          </div>

          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <ShieldCheck size={14} color="#34D399" /> Softmax Score
          </span>
        </div>
      </div>

      {/* Clinical Recommendation Box */}
      <div style={{
        background: 'rgba(6, 182, 212, 0.06)',
        border: '1px solid rgba(6, 182, 212, 0.25)',
        borderRadius: '12px',
        padding: '16px',
        marginBottom: '24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)', fontWeight: '700', fontSize: '0.9rem', marginBottom: '6px' }}>
          <Info size={18} />
          <span>Clinical Protocol & Recommendation</span>
        </div>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-primary)', margin: 0, lineHeight: '1.5' }}>
          {clinicalRecommendation}
        </p>
      </div>

      {/* 5-Class Probability Distribution Breakdown */}
      <div>
        <h4 style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-secondary)', marginBottom: '12px' }}>
          5-Class Softmax Probability Distribution
        </h4>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {Object.entries(probabilities).map(([className, prob], idx) => {
            const pct = (prob * 100).toFixed(1);
            const isTarget = idx === predictedClassId;

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

