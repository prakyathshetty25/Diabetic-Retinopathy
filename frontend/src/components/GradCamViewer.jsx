import React, { useState } from 'react';
import { Sliders, Eye, Layers, MapPin, BarChart2 } from 'lucide-react';

export default function GradCamViewer({ images, spatialSummary }) {
  const [opacity, setOpacity] = useState(0.65);
  const [activeTab, setActiveTab] = useState('overlay'); // 'overlay', 'heatmap', 'side_by_side'

  if (!images) return null;

  const { preprocessed, gradcam_overlay, heatmap_only } = images;

  return (
    <div className="glass-card fade-in" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px', flexWrap: 'wrap', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Eye size={22} color="var(--accent-cyan)" />
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '700' }}>Grad-CAM Explainable AI (XAI) Visualizer</h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Spatial Lesion Localization & Feature Map Activation</p>
          </div>
        </div>

        {/* View Tabs */}
        <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.05)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          <button
            onClick={() => setActiveTab('overlay')}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              border: 'none',
              background: activeTab === 'overlay' ? 'var(--accent-cyan)' : 'transparent',
              color: activeTab === 'overlay' ? '#FFF' : 'var(--text-secondary)',
              fontSize: '0.8rem',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Heatmap Overlay
          </button>
          <button
            onClick={() => setActiveTab('side_by_side')}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              border: 'none',
              background: activeTab === 'side_by_side' ? 'var(--accent-cyan)' : 'transparent',
              color: activeTab === 'side_by_side' ? '#FFF' : 'var(--text-secondary)',
              fontSize: '0.8rem',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setActiveTab('heatmap')}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              border: 'none',
              background: activeTab === 'heatmap' ? 'var(--accent-cyan)' : 'transparent',
              color: activeTab === 'heatmap' ? '#FFF' : 'var(--text-secondary)',
              fontSize: '0.8rem',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Raw Activation
          </button>
        </div>
      </div>

      {/* Spatial Metrics Banner */}
      {spatialSummary && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '12px',
          marginBottom: '20px'
        }}>
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 14px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <MapPin size={12} color="var(--accent-cyan)" /> Primary Lesion Quadrant
            </span>
            <div style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-primary)', marginTop: '2px' }}>
              {spatialSummary.primary_lesion_quadrant}
            </div>
          </div>

          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 14px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <BarChart2 size={12} color="var(--accent-amber)" /> High Lesion Density Area
            </span>
            <div style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-primary)', marginTop: '2px' }}>
              {spatialSummary.high_lesion_density_pct}% coverage
            </div>
          </div>
        </div>
      )}

      {/* Main Image Display Area */}
      <div style={{ marginBottom: '16px' }}>
        {activeTab === 'side_by_side' ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ textAlign: 'center' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', display: 'block' }}>
                Preprocessed Fundus Scan (CLAHE)
              </span>
              <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)', background: '#000' }}>
                <img src={preprocessed} alt="Preprocessed Fundus Scan" style={{ width: '100%', height: 'auto', display: 'block' }} />
              </div>
            </div>

            <div style={{ textAlign: 'center' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', display: 'block' }}>
                Grad-CAM Lesion Heatmap Overlay
              </span>
              <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)', background: '#000' }}>
                <img src={gradcam_overlay} alt="Grad-CAM Overlay" style={{ width: '100%', height: 'auto', display: 'block' }} />
              </div>
            </div>
          </div>
        ) : (
          <div style={{ position: 'relative', width: '100%', maxWidth: '560px', margin: '0 auto', borderRadius: '14px', overflow: 'hidden', border: '1px solid var(--border-color)', background: '#000' }}>
            {activeTab === 'overlay' ? (
              <>
                <img src={preprocessed} alt="Base Fundus Scan" style={{ width: '100%', height: 'auto', display: 'block' }} />
                <img
                  src={heatmap_only}
                  alt="Grad-CAM Heatmap Layer"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    opacity: opacity,
                    mixBlendMode: 'screen',
                    pointerEvents: 'none'
                  }}
                />
              </>
            ) : (
              <img src={heatmap_only} alt="Raw Heatmap" style={{ width: '100%', height: 'auto', display: 'block' }} />
            )}
          </div>
        )}
      </div>

      {/* Opacity Slider for Layered View */}
      {activeTab === 'overlay' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', maxWidth: '400px', margin: '0 auto', background: 'rgba(255, 255, 255, 0.03)', padding: '10px 16px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          <Sliders size={16} color="var(--accent-cyan)" />
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>Heatmap Opacity:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
          />
          <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', fontWeight: '600', color: 'var(--text-primary)', minWidth: '35px' }}>
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}
    </div>
  );
}
