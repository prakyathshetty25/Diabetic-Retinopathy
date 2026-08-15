import React from 'react';
import { Eye, Activity, FileText, ShieldAlert } from 'lucide-react';

export default function Header({ isConnected, patientId, setPatientId, onOpenGuidelines }) {
  return (
    <header className="glass-card" style={{ padding: '16px 28px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #06B6D4, #3B82F6)',
            padding: '10px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(6, 182, 212, 0.4)'
          }}>
            <Eye size={26} color="#FFFFFF" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.4rem', fontWeight: '800', letterSpacing: '-0.02em', background: 'linear-gradient(to right, #FFFFFF, #9CA3AF)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Universal Retinal Screening
              </h1>
              <span style={{ fontSize: '0.75rem', fontWeight: '600', padding: '2px 8px', borderRadius: '6px', background: 'rgba(6, 182, 212, 0.15)', color: '#06B6D4', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                v1.0 XAI-RAG
              </span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Diabetic Retinopathy Multi-Class Grading & Clinical Decision Support
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Patient ID Input */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(255, 255, 255, 0.05)', padding: '6px 12px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: '500' }}>Patient ID:</span>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#FFFFFF',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.85rem',
                fontWeight: '600',
                width: '120px',
                outline: 'none'
              }}
            />
          </div>

          {/* ICDR Guidelines Button */}
          <button className="btn-secondary" onClick={onOpenGuidelines} style={{ fontSize: '0.85rem', padding: '8px 14px' }}>
            <FileText size={16} color="var(--accent-cyan)" />
            ICDR Guidelines
          </button>

          {/* Connection Status Badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            borderRadius: '9999px',
            background: isConnected ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: isConnected ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)',
            fontSize: '0.8rem',
            fontWeight: '600',
            color: isConnected ? '#34D399' : '#F87171'
          }}>
            <Activity size={14} className={isConnected ? "pulse" : ""} />
            {isConnected ? "Engine Connected" : "Backend Offline"}
          </div>
        </div>
      </div>
    </header>
  );
}
