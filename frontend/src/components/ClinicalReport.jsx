import React from 'react';
import { FileCheck, AlertTriangle, BookOpen, Clock, Stethoscope, Printer, CheckCircle2 } from 'lucide-react';

export default function ClinicalReport({ report, patientId }) {
  if (!report) return null;

  const { summary_header, diagnostic_reasoning, spatial_lesion_analysis, clinical_recommendations, retrieved_guideline_citations } = report;

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="glass-card fade-in" style={{ padding: '28px', marginBottom: '24px', borderLeft: `4px solid ${summary_header.severity_color}` }}>
      {/* Header & Export Action */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
            <FileCheck size={24} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.3rem', fontWeight: '800' }}>RAG Clinical Decision Support Report</h2>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Patient ID: <span style={{ fontFamily: 'var(--font-mono)', color: '#FFF', fontWeight: '600' }}>{patientId}</span> • Standard ICDR Scale Assessment
          </p>
        </div>

        <button className="btn-secondary" onClick={handlePrint} style={{ fontSize: '0.85rem' }}>
          <Printer size={16} /> Print / Export PDF
        </button>
      </div>

      {/* Diagnosis Banner */}
      <div style={{
        background: 'rgba(255, 255, 255, 0.03)',
        borderRadius: '14px',
        padding: '20px',
        border: '1px solid var(--border-color)',
        marginBottom: '24px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px'
      }}>
        <div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Diagnosis & ICDR Grade</span>
          <div style={{ fontSize: '1.2rem', fontWeight: '800', color: summary_header.severity_color, marginTop: '2px' }}>
            Grade {summary_header.icdr_grade}: {summary_header.diagnosis}
          </div>
        </div>

        <div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Classifier Certainty</span>
          <div style={{ fontSize: '1.2rem', fontWeight: '700', color: '#FFF', marginTop: '2px' }}>
            {summary_header.confidence_score}
          </div>
        </div>

        <div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Clinical Urgency Level</span>
          <div style={{ fontSize: '0.95rem', fontWeight: '700', color: summary_header.severity_color, marginTop: '4px' }}>
            {summary_header.urgency_level}
          </div>
        </div>
      </div>

      {/* Diagnostic Reasoning & Criteria Match */}
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--accent-cyan)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Stethoscope size={18} /> Diagnostic Reasoning & ICDR Criteria Match
        </h3>
        
        <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-color)', marginBottom: '14px' }}>
          <div style={{ fontSize: '0.88rem', fontWeight: '600', color: '#FFF', marginBottom: '4px' }}>Matched ICDR Benchmark Criteria:</div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{diagnostic_reasoning.icdr_criteria_matched}</p>
        </div>

        <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: '0.88rem', fontWeight: '600', color: '#FFF', marginBottom: '4px' }}>Grad-CAM Visual Lesion Synthesis:</div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{diagnostic_reasoning.xai_gradcam_findings}</p>
        </div>
      </div>

      {/* Observed Lesion Indicators */}
      <div style={{ marginBottom: '24px' }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '10px' }}>
          Observed Pathological Lesion Indicators
        </h4>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
          {diagnostic_reasoning.observed_indicators?.map((indicator, idx) => (
            <span key={idx} style={{
              background: 'rgba(6, 182, 212, 0.1)',
              border: '1px solid rgba(6, 182, 212, 0.25)',
              color: '#06B6D4',
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '0.8rem',
              fontWeight: '600',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <CheckCircle2 size={14} /> {indicator}
            </span>
          ))}
        </div>
      </div>

      {/* Clinical Recommendations & Referral Plan */}
      <div style={{
        background: 'rgba(6, 182, 212, 0.05)',
        border: '1px solid rgba(6, 182, 212, 0.2)',
        borderRadius: '14px',
        padding: '20px',
        marginBottom: '24px'
      }}>
        <h3 style={{ fontSize: '1rem', fontWeight: '700', color: '#FFF', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Clock size={18} color="var(--accent-amber)" /> Actionable Clinical Recommendations & Care Plan
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
          <div>
            <div style={{ fontSize: '0.82rem', fontWeight: '700', color: 'var(--accent-amber)', marginBottom: '4px' }}>
              Referral Timeline & Specialist Exam:
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>
              {clinical_recommendations.referral_timeline}
            </p>
          </div>

          <div>
            <div style={{ fontSize: '0.82rem', fontWeight: '700', color: 'var(--accent-cyan)', marginBottom: '4px' }}>
              Patient Management Plan:
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>
              {clinical_recommendations.patient_counseling_plan}
            </p>
          </div>
        </div>
      </div>

      {/* Vector Store Guideline Citations */}
      {retrieved_guideline_citations && retrieved_guideline_citations.length > 0 && (
        <div>
          <h4 style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <BookOpen size={16} /> RAG Vector Retrieval Guideline Citations
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {retrieved_guideline_citations.map((cite, idx) => (
              <div key={idx} style={{
                background: 'rgba(255, 255, 255, 0.02)',
                padding: '12px 16px',
                borderRadius: '10px',
                border: '1px solid var(--border-color)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '12px'
              }}>
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: '700', color: '#FFF' }}>{cite.title}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>{cite.criteria}</div>
                </div>
                <span style={{
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-mono)',
                  padding: '4px 8px',
                  borderRadius: '6px',
                  background: 'rgba(16, 185, 129, 0.15)',
                  color: '#34D399',
                  whiteSpace: 'nowrap'
                }}>
                  Sim Score: {cite.relevance_score}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
