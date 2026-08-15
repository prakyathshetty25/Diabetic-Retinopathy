import React from 'react';
import { Eye, Zap } from 'lucide-react';

export default function SampleSelector({ samples, selectedGrade, onSelectSample, isLoading }) {
  if (!samples || samples.length === 0) return null;

  return (
    <div className="glass-card" style={{ padding: '20px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Zap size={18} color="var(--accent-amber)" />
          <h3 style={{ fontSize: '1rem', fontWeight: '700' }}>Pre-Loaded Clinical Test Fundus Scans</h3>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Click to test 0-4 DR severity</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px' }}>
        {samples.map((sample) => {
          const isSelected = selectedGrade === sample.grade;
          return (
            <button
              key={sample.grade}
              onClick={() => onSelectSample(sample)}
              disabled={isLoading}
              className={`glass-card glass-card-interactive ${isSelected ? 'selected' : ''}`}
              style={{
                padding: '10px',
                textAlign: 'left',
                background: isSelected ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                borderColor: isSelected ? sample.color : 'rgba(255, 255, 255, 0.08)',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                opacity: isLoading ? 0.6 : 1
              }}
            >
              <div style={{
                width: '100%',
                height: '80px',
                borderRadius: '8px',
                overflow: 'hidden',
                marginBottom: '8px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                background: '#000'
              }}>
                <img
                  src={sample.image_base64}
                  alt={sample.title}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              </div>

              <div style={{ fontSize: '0.8rem', fontWeight: '700', color: sample.color }}>
                Grade {sample.grade}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {sample.title.split(': ')[1]}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
