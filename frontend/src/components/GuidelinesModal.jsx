import React from 'react';
import { X, BookOpen, Clock, ShieldAlert, CheckCircle2 } from 'lucide-react';

export default function GuidelinesModal({ isOpen, onClose, guidelines }) {
  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div className="glass-card fade-in" style={{
        width: '100%',
        maxWidth: '800px',
        maxHeight: '85vh',
        overflowY: 'auto',
        padding: '28px',
        border: '1px solid var(--border-accent)',
        boxShadow: '0 20px 60px rgba(0, 0, 0, 0.6)'
      }}>
        {/* Modal Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <BookOpen size={24} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.25rem', fontWeight: '800' }}>International Clinical Diabetic Retinopathy (ICDR) Scale</h2>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255, 255, 255, 0.08)',
              border: 'none',
              color: 'var(--text-primary)',
              borderRadius: '50%',
              width: '32px',
              height: '32px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Guidelines List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {guidelines?.map((item) => (
            <div key={item.id} style={{
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid var(--border-color)',
              borderRadius: '12px',
              padding: '18px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--accent-cyan)' }}>
                  {item.title}
                </h3>
                <span className={`severity-badge severity-${item.grade}`}>
                  Grade {item.grade}
                </span>
              </div>

              <p style={{ fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: '10px' }}>
                <strong>Benchmark Diagnostic Criteria:</strong> {item.criteria}
              </p>

              <div style={{ marginBottom: '10px' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-secondary)' }}>Clinical Lesion Features:</span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '4px' }}>
                  {item.lesion_indicators.map((lesion, idx) => (
                    <span key={idx} style={{ fontSize: '0.75rem', background: 'rgba(255, 255, 255, 0.05)', padding: '2px 8px', borderRadius: '4px', color: 'var(--text-secondary)' }}>
                      • {lesion}
                    </span>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem', color: 'var(--accent-amber)', fontWeight: '600' }}>
                <Clock size={14} /> Referral Timeline: {item.follow_up_recommendation}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
